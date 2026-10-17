// Parquet reader for LOAD DATA / pipelines (nested columns load as JSON), PLAIN,
// dictionary (PLAIN_DICTIONARY / RLE_DICTIONARY) and DELTA_BINARY_PACKED encodings, optional
// columns, data pages v1 and v2, UNCOMPRESSED / SNAPPY / GZIP / ZSTD. The file becomes tab-separated
// text in the order of the requested columns (matched by name), which the normal LOAD DATA path loads.
#include <snappy.h>
#include <zlib.h>
#include <zstd.h>

#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "common.h"
#include "sources.h"

namespace gdb {

namespace {

[[noreturn]] void bad(const std::string& what) { throw SqlError("Parquet: " + what, 1105); }

// ---- Thrift compact protocol: just enough to read the footer and page headers.
struct TVal {
  int type = 0;  // compact type id
  int64_t i = 0;
  double d = 0;
  std::string bin;
  std::vector<TVal> list;
  std::map<int, TVal> fields;
  const TVal* f(int id) const {
    auto it = fields.find(id);
    return it == fields.end() ? nullptr : &it->second;
  }
  int64_t fi(int id, int64_t def = 0) const {
    const TVal* v = f(id);
    return v ? v->i : def;
  }
};

struct Thrift {
  const uint8_t* p;
  const uint8_t* e;
  uint8_t byte() {
    if (p >= e) bad("truncated metadata");
    return *p++;
  }
  uint64_t varint() {
    uint64_t v = 0;
    for (int s = 0;; s += 7) {
      const uint8_t b = byte();
      v |= (uint64_t)(b & 0x7f) << s;
      if (!(b & 0x80)) return v;
      if (s > 63) bad("bad varint");
    }
  }
  int64_t zigzag() {
    const uint64_t v = varint();
    return (int64_t)(v >> 1) ^ -(int64_t)(v & 1);
  }
  TVal value(int type) {
    TVal v;
    v.type = type;
    switch (type) {
      case 1: v.i = 1; break;  // BOOLEAN_TRUE (in structs)
      case 2: v.i = 0; break;  // BOOLEAN_FALSE
      case 3: v.i = (int8_t)byte(); break;
      case 4: case 5: case 6: v.i = zigzag(); break;
      case 7: {
        if (e - p < 8) bad("truncated double");
        memcpy(&v.d, p, 8);
        p += 8;
        break;
      }
      case 8: {
        const uint64_t n = varint();
        if ((uint64_t)(e - p) < n) bad("truncated binary");
        v.bin.assign((const char*)p, n);
        p += n;
        break;
      }
      case 9: case 10: {
        const uint8_t h = byte();
        uint64_t n = h >> 4;
        const int et = h & 0x0f;
        if (n == 15) n = varint();
        for (uint64_t k = 0; k < n; ++k) {
          if (et == 1 || et == 2) {  // booleans in lists: one byte each
            TVal b;
            b.type = et;
            b.i = byte() == 1;
            v.list.push_back(b);
          } else {
            v.list.push_back(value(et));
          }
        }
        break;
      }
      case 11: {
        const uint64_t n = varint();
        if (n) {
          const uint8_t kv = byte();
          for (uint64_t k = 0; k < n; ++k) {
            v.list.push_back(value(kv >> 4));
            v.list.push_back(value(kv & 0x0f));
          }
        }
        break;
      }
      case 12: v = structure(); break;
      default: bad("unknown thrift type " + std::to_string(type));
    }
    return v;
  }
  TVal structure() {
    TVal s;
    s.type = 12;
    int last = 0;
    for (;;) {
      const uint8_t h = byte();
      if (h == 0) return s;
      const int delta = h >> 4, type = h & 0x0f;
      const int id = delta ? last + delta : (int)zigzag();
      last = id;
      s.fields[id] = value(type);
    }
  }
};

// ---- page decoding helpers
std::string decompress(int codec, const uint8_t* src, size_t n, size_t outSize) {
  std::string out;
  switch (codec) {
    case 0: out.assign((const char*)src, n); break;
    case 1:
      if (!snappy::Uncompress((const char*)src, n, &out)) bad("bad snappy page");
      break;
    case 2: {
      out.resize(outSize);
      z_stream z{};
      if (inflateInit2(&z, 15 + 32) != Z_OK) bad("gzip init");
      z.next_in = (Bytef*)src;
      z.avail_in = (uInt)n;
      z.next_out = (Bytef*)out.data();
      z.avail_out = (uInt)outSize;
      const int rc = inflate(&z, Z_FINISH);
      inflateEnd(&z);
      if (rc != Z_STREAM_END) bad("bad gzip page");
      out.resize(z.total_out);
      break;
    }
    case 6: {
      out.resize(outSize);
      const size_t r = ZSTD_decompress(out.data(), outSize, src, n);
      if (ZSTD_isError(r)) bad("bad zstd page");
      out.resize(r);
      break;
    }
    default: bad("unsupported compression codec " + std::to_string(codec) + " (supported: none, snappy, gzip, zstd)");
  }
  return out;
}

// RLE / bit-packed hybrid of `count` values of `bw` bits.
void rleHybrid(const uint8_t* p, const uint8_t* e, int bw, size_t count, std::vector<uint32_t>& out) {
  out.clear();
  out.reserve(count);
  const int bytes = (bw + 7) / 8;
  while (out.size() < count && p < e) {
    uint64_t h = 0;
    for (int s = 0; p < e; s += 7) {
      const uint8_t b = *p++;
      h |= (uint64_t)(b & 0x7f) << s;
      if (!(b & 0x80)) break;
    }
    if (h & 1) {  // bit-packed: (h >> 1) groups of 8 values
      const size_t n = (size_t)(h >> 1) * 8;
      uint64_t acc = 0;
      int have = 0;
      for (size_t k = 0; k < n; ++k) {
        while (have < bw) {
          acc |= (uint64_t)(p < e ? *p : 0) << have;
          ++p;
          have += 8;
        }
        if (out.size() < count) out.push_back(bw ? (uint32_t)(acc & ((1ull << bw) - 1)) : 0);
        acc >>= bw;
        have -= bw;
      }
    } else {  // run: (h >> 1) copies of one value
      uint32_t v = 0;
      for (int k = 0; k < bytes && p < e; ++k) v |= (uint32_t)*p++ << (8 * k);
      for (uint64_t k = 0; k < (h >> 1) && out.size() < count; ++k) out.push_back(v);
    }
  }
  out.resize(count, 0);
}

// DELTA_BINARY_PACKED integers.
const uint8_t* deltaBinaryPacked(const uint8_t* p, const uint8_t* e, std::vector<int64_t>& out) {
  Thrift t{p, e};
  const uint64_t block = t.varint(), minis = t.varint(), total = t.varint();
  int64_t cur = t.zigzag();
  p = t.p;
  out.clear();
  if (total) out.push_back(cur);
  const uint64_t perMini = minis ? block / minis : block;
  while (out.size() < total) {
    Thrift h{p, e};
    const int64_t minDelta = h.zigzag();
    p = h.p;
    std::vector<int> widths(minis);
    for (uint64_t m = 0; m < minis; ++m) widths[m] = p < e ? *p++ : 0;
    for (uint64_t m = 0; m < minis && out.size() < total; ++m) {
      const int bw = widths[m];
      uint64_t acc = 0;
      int have = 0;
      for (uint64_t k = 0; k < perMini; ++k) {
        uint64_t v = 0;
        if (bw) {
          while (have < bw) {
            acc |= (uint64_t)(p < e ? *p : 0) << have;
            ++p;
            have += 8;
            if (have > 56 && have < bw) bad("delta bit width too large");
          }
          v = bw == 64 ? acc : acc & ((1ull << bw) - 1);
          acc = bw == 64 ? 0 : acc >> bw;
          have -= bw;
        }
        if (out.size() < total) {
          cur += minDelta + (int64_t)v;
          out.push_back(cur);
        }
      }
    }
  }
  return p;
}

std::string fmtDouble(double v) {
  if (std::isnan(v)) return "\\N";
  char b[64];
  auto r = std::to_chars(b, b + sizeof b, v);
  return std::string(b, r.ptr);
}
std::string fmtDecimal(__int128 v, int scale) {
  const bool neg = v < 0;
  unsigned __int128 u = neg ? (unsigned __int128)(-v) : (unsigned __int128)v;
  std::string digits;
  do {
    digits.insert(digits.begin(), (char)('0' + (int)(u % 10)));
    u /= 10;
  } while (u);
  if (scale > 0) {
    while ((int)digits.size() <= scale) digits.insert(digits.begin(), '0');
    digits.insert(digits.end() - scale, '.');
  }
  return (neg ? "-" : "") + digits;
}
std::string fmtDate(int64_t days) {
  // civil from days (Howard Hinnant)
  days += 719468;
  const int64_t era = (days >= 0 ? days : days - 146096) / 146097;
  const unsigned doe = (unsigned)(days - era * 146097);
  const unsigned yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
  const int64_t y = (int64_t)yoe + era * 400;
  const unsigned doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
  const unsigned mp = (5 * doy + 2) / 153;
  const unsigned d = doy - (153 * mp + 2) / 5 + 1;
  const unsigned m = mp < 10 ? mp + 3 : mp - 9;
  char b[32];
  snprintf(b, sizeof b, "%04lld-%02u-%02u", (long long)(y + (m <= 2)), m, d);
  return b;
}
std::string fmtTimestamp(int64_t us) {
  const int64_t day = us >= 0 ? us / 86400000000ll : -((-us + 86400000000ll - 1) / 86400000000ll);
  int64_t rem = us - day * 86400000000ll;
  const int64_t sec = rem / 1000000, frac = rem % 1000000;
  char b[48];
  if (frac)
    snprintf(b, sizeof b, "%s %02lld:%02lld:%02lld.%06lld", fmtDate(day).c_str(), (long long)(sec / 3600), (long long)(sec / 60 % 60),
             (long long)(sec % 60), (long long)frac);
  else
    snprintf(b, sizeof b, "%s %02lld:%02lld:%02lld", fmtDate(day).c_str(), (long long)(sec / 3600), (long long)(sec / 60 % 60),
             (long long)(sec % 60));
  return b;
}
// A text cell for the TSV loader (loaded with ESCAPED BY '\\'): tab / newline / CR / backslash are
// escaped; a leading quote makes the cell CSV-quoted.
void tsvAppend(std::string& out, const char* s, size_t n) {
  const bool quote = n && s[0] == '"';
  if (quote) out += '"';
  for (size_t i = 0; i < n; ++i) {
    const char c = s[i];
    if (quote && c == '"') out += "\"\"";
    else if (c == '\t') out += "\\t";
    else if (c == '\n') out += "\\n";
    else if (c == '\r') out += "\\r";
    else if (c == '\\') out += "\\\\";
    else out += c;
  }
  if (quote) out += '"';
}

struct Col {
  std::string name;
  int type = -1;      // physical: 0 BOOLEAN, 1 INT32, 2 INT64, 3 INT96, 4 FLOAT, 5 DOUBLE, 6 BYTE_ARRAY, 7 FIXED_LEN_BYTE_ARRAY
  int typeLen = 0;
  bool optional = false;
  int conv = -1;      // converted type
  int scale = 0;
  int tsUnit = 0;     // timestamps: 1 millis, 2 micros, 3 nanos
  bool isDate = false, isDecimal = false, isTime = false;
  int maxDef = 0, maxRep = 0;  // definition / repetition levels of the leaf
  bool json = false;           // values for a nested (JSON) column: strings quoted, booleans true / false
};

int bitWidth(int v) {
  int b = 0;
  while ((1 << b) <= v) ++b;
  return b;
}
std::string jsonQuote(const char* s, size_t n) {
  std::string o = "\"";
  for (size_t i = 0; i < n; ++i) {
    const unsigned char c = (unsigned char)s[i];
    if (c == '"' || c == '\\') o += '\\', o += (char)c;
    else if (c == '\n') o += "\\n";
    else if (c == '\t') o += "\\t";
    else if (c == '\r') o += "\\r";
    else if (c < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", c); o += b; }
    else o += (char)c;
  }
  return o + "\"";
}

// One column chunk: its repetition / definition levels (empty when the max level is 0) and the
// values of the entries whose definition level is the maximum (formatted).
void readLeaf(const std::string& file, const Col& col, const TVal& meta, std::vector<uint32_t>& repsOut, std::vector<uint32_t>& defsOut,
              std::vector<std::string>& valsOut) {
  const int codec = (int)meta.fi(4);
  int64_t off = meta.fi(9);
  if (const TVal* dp = meta.f(11)) off = std::min(off, dp->i ? dp->i : off);
  const int64_t total = meta.fi(7);
  if (off < 0 || off + total > (int64_t)file.size()) bad("column chunk outside the file");
  const uint8_t* p = (const uint8_t*)file.data() + off;
  const uint8_t* end = p + total;
  std::vector<std::string> dict;
  repsOut.clear();
  defsOut.clear();
  valsOut.clear();
  const int64_t totalValues = meta.fi(5);
  auto q2 = [&](std::string v) { return col.json ? "\"" + v + "\"" : v; };
  auto fmtInt = [&](int64_t v) -> std::string {
    if (col.isDate) return q2(fmtDate(v));
    if (col.tsUnit) return q2(fmtTimestamp(col.tsUnit == 1 ? v * 1000 : col.tsUnit == 3 ? v / 1000 : v));
    if (col.isDecimal) return fmtDecimal(v, col.scale);
    return std::to_string(v);
  };
  // Decodes n PLAIN values starting at q into strings.
  auto plain = [&](const uint8_t*& q, const uint8_t* qe, size_t n, std::vector<std::string>& out) {
    out.clear();
    out.reserve(n);
    for (size_t k = 0; k < n; ++k) {
      switch (col.type) {
        case 0: out.push_back((q[k / 8] >> (k % 8)) & 1 ? (col.json ? "true" : "1") : (col.json ? "false" : "0")); break;
        case 1: { int32_t v; memcpy(&v, q, 4); q += 4; out.push_back(fmtInt(v)); break; }
        case 2: { int64_t v; memcpy(&v, q, 8); q += 8; out.push_back(fmtInt(v)); break; }
        case 3: {  // INT96 timestamp: nanoseconds of the day + Julian day
          int64_t nanos;
          int32_t jd;
          memcpy(&nanos, q, 8);
          memcpy(&jd, q + 8, 4);
          q += 12;
          out.push_back(q2(fmtTimestamp(((int64_t)jd - 2440588) * 86400000000ll + nanos / 1000)));
          break;
        }
        case 4: { float v; memcpy(&v, q, 4); q += 4; out.push_back(fmtDouble(v)); break; }
        case 5: { double v; memcpy(&v, q, 8); q += 8; out.push_back(fmtDouble(v)); break; }
        case 6: {
          uint32_t len;
          memcpy(&len, q, 4);
          q += 4;
          if (q + len > qe) bad("byte array past the page");
          if (col.isDecimal) {
            __int128 v = len && (q[0] & 0x80) ? -1 : 0;
            for (uint32_t b = 0; b < len; ++b) v = (v << 8) | q[b];
            out.push_back(fmtDecimal(v, col.scale));
          } else if (col.json) {
            out.push_back(jsonQuote((const char*)q, len));
          } else {
            std::string s;
            tsvAppend(s, (const char*)q, len);
            out.push_back(std::move(s));
          }
          q += len;
          break;
        }
        case 7: {
          const int len = col.typeLen;
          if (col.isDecimal) {
            __int128 v = len && (q[0] & 0x80) ? -1 : 0;
            for (int b = 0; b < len; ++b) v = (v << 8) | q[b];
            out.push_back(fmtDecimal(v, col.scale));
          } else if (col.json) {
            out.push_back(jsonQuote((const char*)q, len));
          } else {
            std::string s;
            tsvAppend(s, (const char*)q, len);
            out.push_back(std::move(s));
          }
          q += len;
          break;
        }
        default: bad("unsupported physical type");
      }
    }
    if (col.type == 0) q += (n + 7) / 8;
  };
  std::vector<uint32_t> defs, reps, idx;
  std::vector<std::string> vals;
  std::vector<int64_t> ints;
  const int dbw = bitWidth(col.maxDef), rbw = bitWidth(col.maxRep);
  while (p < end && (int64_t)defsOut.size() < totalValues && (col.maxDef || (int64_t)valsOut.size() < totalValues)) {
    Thrift t{p, end};
    TVal ph = t.structure();
    p = t.p;
    const int ptype = (int)ph.fi(1);
    const int64_t usize = ph.fi(2), csize = ph.fi(3);
    if (p + csize > end) bad("page past the column chunk");
    const uint8_t* body = p;
    p += csize;
    if (ptype == 2) {  // dictionary page
      const TVal* dh = ph.f(7);
      const std::string raw = decompress(codec, body, csize, usize);
      const uint8_t* q = (const uint8_t*)raw.data();
      plain(q, q + raw.size(), dh ? (size_t)dh->fi(1) : 0, dict);
      continue;
    }
    if (ptype != 0 && ptype != 3) continue;  // index pages etc.
    std::string raw;
    const uint8_t *q, *qe;
    size_t n;
    int enc;
    if (ptype == 0) {
      const TVal* dh = ph.f(5);
      if (!dh) bad("data page without header");
      n = (size_t)dh->fi(1);
      enc = (int)dh->fi(2);
      raw = decompress(codec, body, csize, usize);
      q = (const uint8_t*)raw.data();
      qe = q + raw.size();
      if (col.maxRep) {  // 4-byte length + RLE repetition levels, then the same for definition levels
        uint32_t len;
        memcpy(&len, q, 4);
        rleHybrid(q + 4, q + 4 + len, rbw, n, reps);
        q += 4 + len;
      }
      if (col.maxDef) {
        uint32_t len;
        memcpy(&len, q, 4);
        rleHybrid(q + 4, q + 4 + len, dbw, n, defs);
        q += 4 + len;
      }
    } else {
      const TVal* dh = ph.f(8);
      if (!dh) bad("data page v2 without header");
      n = (size_t)dh->fi(1);
      enc = (int)dh->fi(4);
      const int64_t defLen = dh->fi(5), repLen = dh->fi(6);
      if (col.maxRep) rleHybrid(body, body + repLen, rbw, n, reps);
      if (col.maxDef) rleHybrid(body + repLen, body + repLen + defLen, dbw, n, defs);
      const int64_t lv = repLen + defLen;
      const bool compressed = dh->f(7) ? dh->fi(7) != 0 : true;
      raw = compressed ? decompress(codec, body + lv, csize - lv, usize - lv) : std::string((const char*)body + lv, csize - lv);
      q = (const uint8_t*)raw.data();
      qe = q + raw.size();
    }
    size_t present = n;
    if (col.maxDef) {
      present = 0;
      for (size_t k = 0; k < n; ++k) present += defs[k] == (uint32_t)col.maxDef;
    }
    if (enc == 0) {
      plain(q, qe, present, vals);
    } else if (enc == 2 || enc == 8) {
      const int bw = q < qe ? *q : 0;
      rleHybrid(q + 1, qe, bw, present, idx);
      vals.clear();
      vals.reserve(present);
      for (uint32_t i : idx) {
        if (i >= dict.size()) bad("dictionary index out of range");
        vals.push_back(dict[i]);
      }
    } else if (enc == 5) {
      deltaBinaryPacked(q, qe, ints);
      vals.clear();
      for (size_t k = 0; k < present && k < ints.size(); ++k) vals.push_back(fmtInt(ints[k]));
    } else {
      bad("unsupported encoding " + std::to_string(enc) + " (supported: PLAIN, dictionary, DELTA_BINARY_PACKED)");
    }
    if (col.maxRep) repsOut.insert(repsOut.end(), reps.begin(), reps.begin() + n);
    if (col.maxDef) defsOut.insert(defsOut.end(), defs.begin(), defs.begin() + n);
    for (auto& v : vals) valsOut.push_back(std::move(v));
  }
}

// A flat column chunk as cells (formatted); NULL -> "\N".
void readChunk(const std::string& file, const Col& col, const TVal& meta, int64_t rows, std::vector<std::string>& cells) {
  std::vector<uint32_t> reps, defs;
  std::vector<std::string> vals;
  readLeaf(file, col, meta, reps, defs, vals);
  cells.clear();
  cells.reserve(rows);
  size_t vi = 0;
  for (int64_t r = 0; r < rows; ++r) {
    if (col.maxDef && (size_t)r < defs.size() && defs[r] != (uint32_t)col.maxDef) cells.push_back("\\N");
    else cells.push_back(vi < vals.size() ? std::move(vals[vi++]) : std::string("\\N"));
  }
}

}  // namespace

std::vector<std::string> parquetColumns(const std::string& file) {
  if (file.size() < 12 || file.compare(file.size() - 4, 4, "PAR1") != 0) bad("not a Parquet file");
  uint32_t flen;
  memcpy(&flen, file.data() + file.size() - 8, 4);
  if (flen > file.size() - 8) bad("bad footer length");
  Thrift t{(const uint8_t*)file.data() + file.size() - 8 - flen, (const uint8_t*)file.data() + file.size() - 8};
  TVal md = t.structure();
  std::vector<std::string> names;
  const TVal* schema = md.f(2);
  if (schema)
    for (size_t i = 1; i < schema->list.size(); ++i)
      if (const TVal* nm = schema->list[i].f(4)) names.push_back(nm->bin);
  return names;
}

namespace {
// Schema tree node (flattened in the footer in depth-first order).
struct PNode {
  std::string name;
  int rep = 0;          // 0 required, 1 optional, 2 repeated
  int conv = -1;        // converted type (1 MAP, 2 MAP_KEY_VALUE, 3 LIST)
  bool list = false, map = false;
  std::vector<int> kids;
  int leaf = -1;        // index into the leaf columns
  int defLevel = 0, repLevel = 0;  // levels at which this node is present / repeats
};
Col colOf(const TVal& se) {
  Col c;
  c.name = se.f(4) ? se.f(4)->bin : "";
  c.type = (int)se.fi(1, -1);
  c.typeLen = (int)se.fi(2);
  c.conv = (int)se.fi(6, -1);
  c.scale = (int)se.fi(7);
  c.isDate = c.conv == 6;
  c.isDecimal = c.conv == 5;
  if (c.conv == 9) c.tsUnit = 1;
  if (c.conv == 10) c.tsUnit = 2;
  if (const TVal* lt = se.f(10)) {  // LogicalType union
    if (lt->f(6)) c.isDate = true;
    if (const TVal* dec = lt->f(5)) c.isDecimal = true, c.scale = (int)dec->fi(1);
    if (const TVal* ts = lt->f(8))
      if (const TVal* unit = ts->f(2)) c.tsUnit = unit->f(1) ? 1 : unit->f(2) ? 2 : unit->f(3) ? 3 : 2;
  }
  return c;
}

// Per row group: one leaf's levels and values, with the row's entry range.
struct LeafData {
  std::vector<uint32_t> reps, defs;
  std::vector<std::string> vals;
  std::vector<int64_t> valAt;     // entry -> index into vals (-1: not defined)
  std::vector<int64_t> rowStart;  // row -> first entry (rows + 1 entries)
};

// Record assembly (Dremel): every node instance is a range of entries of each leaf below it. A
// repeated node's range splits into elements where a leaf's repetition level is <= the node's own,
// so lists of lists and maps of lists nest to any depth.
struct Assembler {
  const std::vector<PNode>& nodes;
  const std::vector<Col>& leaves;
  std::vector<LeafData>& data;
  std::vector<std::vector<int>> under;  // node -> the leaves below it
  using Ranges = std::vector<std::pair<int64_t, int64_t>>;  // leaf -> [begin, end) entries
  int firstLeaf(int n) const {
    while (nodes[n].leaf < 0) n = nodes[n].kids.at(0);
    return nodes[n].leaf;
  }
  uint32_t defAt(int l, int64_t e) const { return leaves[l].maxDef && e < (int64_t)data[l].defs.size() ? data[l].defs[e] : 0; }
  void init() {
    under.assign(nodes.size(), {});
    for (int n = (int)nodes.size() - 1; n >= 0; --n) {  // children follow their parent
      if (nodes[n].leaf >= 0) under[n].push_back(nodes[n].leaf);
      for (int k : nodes[n].kids) under[n].insert(under[n].end(), under[k].begin(), under[k].end());
    }
  }
  std::string value(int l, int64_t e) const {
    const int64_t vi = e < (int64_t)data[l].valAt.size() ? data[l].valAt[e] : -1;
    return vi < 0 ? "null" : data[l].vals[vi];
  }
  std::string object(const std::vector<int>& kids, const Ranges& rg) {
    std::string o = "{";
    for (size_t k = 0; k < kids.size(); ++k) {
      const PNode& kid = nodes[kids[k]];
      o += (k ? "," : "") + jsonQuote(kid.name.data(), kid.name.size()) + ":" + json(kids[k], rg);
    }
    return o + "}";
  }
  std::string json(int n, const Ranges& rg) {
    const PNode& nd = nodes[n];
    if (nd.rep == 2) return elements(n, rg);
    const int l = firstLeaf(n);
    if (rg[l].first >= rg[l].second || defAt(l, rg[l].first) < (uint32_t)nd.defLevel) return "null";
    if (nd.leaf >= 0) return value(nd.leaf, rg[l].first);
    // a group: LIST / MAP annotated wrappers around one repeated child, else an object
    if ((nd.list || nd.map) && nd.kids.size() == 1 && nodes[nd.kids[0]].rep == 2) return elements(nd.kids[0], rg, nd.map, nd.list);
    return object(nd.kids, rg);
  }
  // The elements of repeated node r: [..] (or {k: v} for maps).
  std::string elements(int r, const Ranges& rg, bool asMap = false, bool asList = false) {
    const PNode& rn = nodes[r];
    const int l = firstLeaf(r);
    if (rg[l].first >= rg[l].second || defAt(l, rg[l].first) < (uint32_t)rn.defLevel) return asMap ? "{}" : "[]";
    // element starts per leaf
    std::vector<std::vector<int64_t>> starts(leaves.size());
    for (int L : under[r])
      for (int64_t e = rg[L].first; e < rg[L].second; ++e)
        if (e == rg[L].first || data[L].reps[e] <= (uint32_t)rn.repLevel) starts[L].push_back(e);
    const size_t n = starts[l].size();
    std::string o = asMap ? "{" : "[";
    Ranges sub = rg;
    for (size_t i = 0; i < n; ++i) {
      for (int L : under[r])
        if (i < starts[L].size()) sub[L] = {starts[L][i], i + 1 < starts[L].size() ? starts[L][i + 1] : rg[L].second};
        else sub[L] = {rg[L].second, rg[L].second};
      if (i) o += ",";
      if (asMap && rn.kids.size() == 2) {
        std::string k = json(rn.kids[0], sub);
        if (k.empty() || k[0] != '"') k = "\"" + k + "\"";
        o += k + ":" + json(rn.kids[1], sub);
      } else if (rn.leaf >= 0) {
        o += value(rn.leaf, sub[rn.leaf].first);
      } else if (asList && rn.kids.size() == 1) {  // 3-level list: repeated group list { element }
        o += json(rn.kids[0], sub);
      } else {
        o += object(rn.kids, sub);
      }
    }
    return o + (asMap ? "}" : "]");
  }
};
}  // namespace

std::string parquetToTsv(const std::string& file, const std::vector<std::string>& wanted) {
  if (file.size() < 12 || file.compare(0, 4, "PAR1") != 0 || file.compare(file.size() - 4, 4, "PAR1") != 0) bad("not a Parquet file");
  uint32_t flen;
  memcpy(&flen, file.data() + file.size() - 8, 4);
  if (flen > file.size() - 8) bad("bad footer length");
  Thrift t{(const uint8_t*)file.data() + file.size() - 8 - flen, (const uint8_t*)file.data() + file.size() - 8};
  TVal md = t.structure();
  const TVal* schema = md.f(2);
  if (!schema || schema->list.empty()) bad("no schema");
  // Rebuild the tree; leaves get their maximum definition / repetition levels.
  std::vector<PNode> nodes;
  std::vector<Col> leaves;
  size_t pos = 1;
  std::function<int(int, int, bool)> build = [&](int defL, int repL, bool underJson) -> int {
    if (pos >= schema->list.size()) bad("truncated schema");
    const TVal& se = schema->list[pos++];
    PNode nd;
    nd.name = se.f(4) ? se.f(4)->bin : "";
    nd.rep = (int)se.fi(3);
    nd.conv = (int)se.fi(6, -1);
    if (const TVal* lt = se.f(10)) nd.list = lt->f(3) != nullptr, nd.map = lt->f(2) != nullptr;
    nd.list = nd.list || nd.conv == 3;
    nd.map = nd.map || nd.conv == 1 || nd.conv == 2;
    if (nd.rep != 0) ++defL;
    if (nd.rep == 2) ++repL;
    nd.defLevel = defL;
    nd.repLevel = repL;
    const int nkids = (int)se.fi(5);
    const int idx = (int)nodes.size();
    nodes.push_back(nd);
    if (nkids == 0) {
      Col c = colOf(se);
      c.maxDef = defL;
      c.maxRep = repL;
      c.optional = defL > 0;
      c.json = underJson || nd.rep == 2;
      nodes[idx].leaf = (int)leaves.size();
      leaves.push_back(c);
    } else {
      for (int k = 0; k < nkids; ++k) {
        const int kid = build(defL, repL, true);
        nodes[idx].kids.push_back(kid);
      }
    }
    return idx;
  };
  std::vector<int> top;
  const int rootKids = (int)schema->list[0].fi(5);
  for (int k = 0; k < rootKids; ++k) top.push_back(build(0, 0, false));
  // Output column k comes from top-level field map[k] (-1: absent -> NULL).
  std::vector<int> map;
  for (auto& w : wanted) {
    int found = -1;
    for (size_t i = 0; i < top.size() && found < 0; ++i)
      if (strcasecmp(nodes[top[i]].name.c_str(), w.c_str()) == 0) found = (int)i;
    map.push_back(found);
  }
  // Leaves each wanted top-level field needs.
  std::vector<std::vector<int>> leavesOf(top.size());
  for (size_t i = 0; i < top.size(); ++i) {
    std::function<void(int)> walk = [&](int n) {
      if (nodes[n].leaf >= 0) leavesOf[i].push_back(nodes[n].leaf);
      for (int k : nodes[n].kids) walk(k);
    };
    walk(top[i]);
  }
  std::string out;
  const TVal* rgs = md.f(4);
  if (!rgs) return out;
  std::vector<LeafData> data(leaves.size());
  for (const TVal& rg : rgs->list) {
    const int64_t rows = rg.fi(3);
    const TVal* chunks = rg.f(1);
    if (!chunks) continue;
    std::vector<bool> need(leaves.size(), false);
    for (int m : map)
      if (m >= 0)
        for (int l : leavesOf[m]) need[l] = true;
    for (size_t l = 0; l < leaves.size(); ++l) {
      LeafData& d = data[l];
      d = LeafData{};
      if (!need[l]) continue;
      if (l >= chunks->list.size()) bad("row group without column chunk " + leaves[l].name);
      const TVal* meta = chunks->list[l].f(3);
      if (!meta) bad("column chunk without metadata (external file?)");
      readLeaf(file, leaves[l], *meta, d.reps, d.defs, d.vals);
      const int64_t entries = leaves[l].maxDef ? (int64_t)d.defs.size() : (int64_t)d.vals.size();
      d.valAt.assign(entries, -1);
      int64_t vi = 0;
      for (int64_t e = 0; e < entries; ++e)
        if (!leaves[l].maxDef || d.defs[e] == (uint32_t)leaves[l].maxDef) d.valAt[e] = vi++;
      d.rowStart.clear();
      for (int64_t e = 0; e < entries; ++e)
        if (!leaves[l].maxRep || d.reps[e] == 0) d.rowStart.push_back(e);
      while ((int64_t)d.rowStart.size() < rows) d.rowStart.push_back(entries);
      d.rowStart.push_back(entries);
    }
    Assembler as{nodes, leaves, data, {}};
    as.init();
    Assembler::Ranges rr(leaves.size());
    for (int64_t r = 0; r < rows; ++r) {
      for (size_t l = 0; l < leaves.size(); ++l)
        if (need[l]) rr[l] = {data[l].rowStart[r], data[l].rowStart[r + 1]};
      for (size_t k = 0; k < map.size(); ++k) {
        if (k) out += '\t';
        if (map[k] < 0) { out += "\\N"; continue; }
        const PNode& nd = nodes[top[map[k]]];
        if (nd.leaf >= 0 && nd.rep != 2) {  // a flat column
          const LeafData& d = data[nd.leaf];
          const int64_t e = d.rowStart[r];
          const int64_t vi = e < (int64_t)d.valAt.size() ? d.valAt[e] : -1;
          out += vi < 0 ? std::string("\\N") : d.vals[vi];
          continue;
        }
        const std::string j = as.json(top[map[k]], rr);
        if (j == "null") out += "\\N";
        else tsvAppend(out, j.data(), j.size());
      }
      out += '\n';
    }
  }
  return out;
}

}  // namespace gdb
