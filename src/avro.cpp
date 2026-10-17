// Avro object container files for LOAD DATA / pipelines: the records become tab-separated text (in
// the order of the requested columns, matched by field name) for the normal LOAD DATA path. Codecs:
// null, deflate, snappy, zstandard. Primitive types, unions (nullable fields), enums, fixed and the
// date / timestamp-millis / timestamp-micros / decimal logical types; nested records, arrays and
// maps become JSON text.
#include <snappy.h>
#include <zlib.h>
#include <zstd.h>

#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "common.h"
#include "sources.h"

namespace gdb {

namespace {

[[noreturn]] void bad(const std::string& what) { throw SqlError("Avro: " + what, 1105); }

// ---- a small JSON reader for the schema
struct J {
  enum K { NUL, BOOL, NUM, STR, ARR, OBJ } k = NUL;
  double n = 0;
  std::string s;
  std::vector<J> a;
  std::vector<std::pair<std::string, J>> o;
  const J* get(const std::string& key) const {
    for (auto& [k2, v] : o)
      if (k2 == key) return &v;
    return nullptr;
  }
};
struct JP {
  const std::string& t;
  size_t p = 0;
  void ws() { while (p < t.size() && isspace((unsigned char)t[p])) ++p; }
  J val() {
    ws();
    if (p >= t.size()) bad("bad schema JSON");
    J v;
    const char c = t[p];
    if (c == '{') {
      v.k = J::OBJ;
      ++p;
      ws();
      if (t[p] == '}') { ++p; return v; }
      for (;;) {
        ws();
        J key = val();
        ws();
        if (t[p] != ':') bad("bad schema JSON");
        ++p;
        v.o.push_back({key.s, val()});
        ws();
        if (t[p] == ',') { ++p; continue; }
        if (t[p] == '}') { ++p; return v; }
        bad("bad schema JSON");
      }
    }
    if (c == '[') {
      v.k = J::ARR;
      ++p;
      ws();
      if (t[p] == ']') { ++p; return v; }
      for (;;) {
        v.a.push_back(val());
        ws();
        if (t[p] == ',') { ++p; continue; }
        if (t[p] == ']') { ++p; return v; }
        bad("bad schema JSON");
      }
    }
    if (c == '"') {
      v.k = J::STR;
      ++p;
      while (p < t.size() && t[p] != '"') {
        if (t[p] == '\\' && p + 1 < t.size()) ++p;
        v.s += t[p++];
      }
      ++p;
      return v;
    }
    if (t.compare(p, 4, "null") == 0) { p += 4; return v; }
    if (t.compare(p, 4, "true") == 0) { p += 4; v.k = J::BOOL; v.n = 1; return v; }
    if (t.compare(p, 5, "false") == 0) { p += 5; v.k = J::BOOL; return v; }
    v.k = J::NUM;
    char* e = nullptr;
    v.n = std::strtod(t.c_str() + p, &e);
    p = (size_t)(e - t.c_str());
    return v;
  }
};

// ---- schema
struct Type {
  std::string kind;  // null boolean int long float double bytes string record enum array map union fixed
  std::string logical;
  int scale = 0, size = 0;
  std::vector<std::string> names;              // record field names / enum symbols
  std::vector<std::shared_ptr<Type>> members;  // record fields / union branches / array-map item
};
std::shared_ptr<Type> parseType(const J& j, std::map<std::string, std::shared_ptr<Type>>& named) {
  auto t = std::make_shared<Type>();
  if (j.k == J::STR) {
    auto it = named.find(j.s);
    if (it != named.end()) return it->second;
    t->kind = j.s;
    return t;
  }
  if (j.k == J::ARR) {
    t->kind = "union";
    for (auto& b : j.a) t->members.push_back(parseType(b, named));
    return t;
  }
  if (j.k != J::OBJ || !j.get("type")) bad("bad schema");
  const J& ty = *j.get("type");
  if (ty.k != J::STR) {  // {"type": {...}} / {"type": [..]}
    auto inner = parseType(ty, named);
    if (const J* l = j.get("logicalType")) inner->logical = l->s;
    return inner;
  }
  t->kind = ty.s;
  if (const J* l = j.get("logicalType")) t->logical = l->s;
  if (const J* sc = j.get("scale")) t->scale = (int)sc->n;
  if (const J* sz = j.get("size")) t->size = (int)sz->n;
  if (const J* nm = j.get("name")) named[nm->s] = t;
  if (t->kind == "record") {
    if (const J* fs = j.get("fields"))
      for (auto& f : fs->a) {
        t->names.push_back(f.get("name") ? f.get("name")->s : "");
        t->members.push_back(parseType(*f.get("type"), named));
      }
  } else if (t->kind == "enum") {
    if (const J* ss = j.get("symbols"))
      for (auto& s : ss->a) t->names.push_back(s.s);
  } else if (t->kind == "array") {
    t->members.push_back(parseType(*j.get("items"), named));
  } else if (t->kind == "map") {
    t->members.push_back(parseType(*j.get("values"), named));
  }
  return t;
}

// ---- binary decoding to text
struct In {
  const uint8_t* p;
  const uint8_t* e;
  int64_t lng() {
    uint64_t v = 0;
    for (int s = 0;; s += 7) {
      if (p >= e) bad("truncated data");
      const uint8_t b = *p++;
      v |= (uint64_t)(b & 0x7f) << s;
      if (!(b & 0x80)) break;
      if (s > 63) bad("bad varint");
    }
    return (int64_t)(v >> 1) ^ -(int64_t)(v & 1);
  }
  std::string bytes() {
    const int64_t n = lng();
    if (n < 0 || e - p < n) bad("truncated bytes");
    std::string s((const char*)p, (size_t)n);
    p += n;
    return s;
  }
};

std::string fmtDouble(double v) {
  char b[64];
  auto r = std::to_chars(b, b + sizeof b, v);
  return std::string(b, r.ptr);
}
std::string fmtDecimal(const std::string& be, int scale) {
  __int128 v = !be.empty() && (be[0] & 0x80) ? -1 : 0;
  for (unsigned char c : be) v = (v << 8) | c;
  const bool neg = v < 0;
  unsigned __int128 u = neg ? (unsigned __int128)(-v) : (unsigned __int128)v;
  std::string d;
  do {
    d.insert(d.begin(), (char)('0' + (int)(u % 10)));
    u /= 10;
  } while (u);
  if (scale > 0) {
    while ((int)d.size() <= scale) d.insert(d.begin(), '0');
    d.insert(d.end() - scale, '.');
  }
  return (neg ? "-" : "") + d;
}
std::string fmtDate(int64_t days) {
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
std::string fmtTs(int64_t us) {
  const int64_t day = us >= 0 ? us / 86400000000ll : -((-us + 86400000000ll - 1) / 86400000000ll);
  const int64_t rem = us - day * 86400000000ll, sec = rem / 1000000, frac = rem % 1000000;
  char b[48];
  if (frac)
    snprintf(b, sizeof b, "%s %02lld:%02lld:%02lld.%06lld", fmtDate(day).c_str(), (long long)(sec / 3600), (long long)(sec / 60 % 60),
             (long long)(sec % 60), (long long)frac);
  else
    snprintf(b, sizeof b, "%s %02lld:%02lld:%02lld", fmtDate(day).c_str(), (long long)(sec / 3600), (long long)(sec / 60 % 60),
             (long long)(sec % 60));
  return b;
}
std::string jsonStr(const std::string& s) {
  std::string o = "\"";
  for (char c : s) {
    if (c == '"' || c == '\\') o += '\\';
    if (c == '\n') { o += "\\n"; continue; }
    o += c;
  }
  return o + "\"";
}

// Decodes one value; `null` set for a null. json: nested context (strings quoted).
std::string decode(const Type& t, In& in, bool& null, bool json) {
  null = false;
  const std::string& k = t.kind;
  if (k == "null") { null = true; return json ? "null" : ""; }
  if (k == "boolean") {
    if (in.p >= in.e) bad("truncated");
    return *in.p++ ? (json ? "true" : "1") : (json ? "false" : "0");
  }
  if (k == "int" || k == "long") {
    const int64_t v = in.lng();
    if (t.logical == "date") return json ? "\"" + fmtDate(v) + "\"" : fmtDate(v);
    if (t.logical == "timestamp-millis") return json ? "\"" + fmtTs(v * 1000) + "\"" : fmtTs(v * 1000);
    if (t.logical == "timestamp-micros") return json ? "\"" + fmtTs(v) + "\"" : fmtTs(v);
    return std::to_string(v);
  }
  if (k == "float") {
    float f;
    if (in.e - in.p < 4) bad("truncated");
    memcpy(&f, in.p, 4);
    in.p += 4;
    return fmtDouble(f);
  }
  if (k == "double") {
    double d;
    if (in.e - in.p < 8) bad("truncated");
    memcpy(&d, in.p, 8);
    in.p += 8;
    return fmtDouble(d);
  }
  if (k == "bytes" || k == "string") {
    std::string s = in.bytes();
    if (t.logical == "decimal") return fmtDecimal(s, t.scale);
    return json ? jsonStr(s) : s;
  }
  if (k == "fixed") {
    if (in.e - in.p < t.size) bad("truncated");
    std::string s((const char*)in.p, t.size);
    in.p += t.size;
    if (t.logical == "decimal") return fmtDecimal(s, t.scale);
    return json ? jsonStr(s) : s;
  }
  if (k == "enum") {
    const int64_t i = in.lng();
    const std::string s = i >= 0 && i < (int64_t)t.names.size() ? t.names[i] : std::to_string(i);
    return json ? jsonStr(s) : s;
  }
  if (k == "union") {
    const int64_t i = in.lng();
    if (i < 0 || i >= (int64_t)t.members.size()) bad("bad union branch");
    return decode(*t.members[i], in, null, json);
  }
  bool n2;
  if (k == "record") {
    std::string o = "{";
    for (size_t f = 0; f < t.members.size(); ++f) {
      const std::string v = decode(*t.members[f], in, n2, true);
      o += (f ? "," : "") + jsonStr(t.names[f]) + ":" + (n2 ? std::string("null") : v);
    }
    return o + "}";
  }
  if (k == "array" || k == "map") {
    std::string o = k == "array" ? "[" : "{";
    bool first = true;
    for (;;) {
      int64_t n = in.lng();
      if (n == 0) break;
      if (n < 0) n = -n, in.lng();  // negative count: a byte size follows
      for (int64_t i = 0; i < n; ++i) {
        if (!first) o += ",";
        first = false;
        if (k == "map") o += jsonStr(in.bytes()) + ":";
        const std::string v = decode(*t.members[0], in, n2, true);
        o += n2 ? "null" : v;
      }
    }
    return o + (k == "array" ? "]" : "}");
  }
  bad("unsupported type " + k);
}

void tsvAppend(std::string& out, const std::string& s) {
  const bool quote = !s.empty() && s[0] == '"';
  if (quote) out += '"';
  for (char c : s) {
    if (quote && c == '"') out += "\"\"";
    else if (c == '\t') out += "\\t";
    else if (c == '\n') out += "\\n";
    else if (c == '\r') out += "\\r";
    else if (c == '\\') out += "\\\\";
    else out += c;
  }
  if (quote) out += '"';
}

}  // namespace

std::string avroToTsv(const std::string& file, const std::vector<std::string>& wanted) {
  if (file.size() < 4 || file.compare(0, 4, "Obj\x01") != 0) bad("not an Avro container file");
  In in{(const uint8_t*)file.data() + 4, (const uint8_t*)file.data() + file.size()};
  std::string schemaJson, codec = "null";
  for (;;) {  // file metadata map
    int64_t n = in.lng();
    if (n == 0) break;
    if (n < 0) n = -n, in.lng();
    for (int64_t i = 0; i < n; ++i) {
      const std::string key = in.bytes(), val = in.bytes();
      if (key == "avro.schema") schemaJson = val;
      if (key == "avro.codec") codec = val;
    }
  }
  if (in.e - in.p < 16) bad("truncated header");
  const std::string sync((const char*)in.p, 16);
  in.p += 16;
  JP jp{schemaJson};
  std::map<std::string, std::shared_ptr<Type>> named;
  auto root = parseType(jp.val(), named);
  if (root->kind != "record") bad("the top-level schema must be a record");
  std::vector<int> map;  // output column -> field index (-1: absent)
  for (auto& w : wanted) {
    int found = -1;
    for (size_t f = 0; f < root->names.size() && found < 0; ++f)
      if (strcasecmp(root->names[f].c_str(), w.c_str()) == 0) found = (int)f;
    map.push_back(found);
  }
  std::string out;
  std::vector<std::string> vals(root->members.size());
  std::vector<bool> nul(root->members.size());
  while (in.p < in.e) {
    const int64_t count = in.lng(), size = in.lng();
    if (size < 0 || in.e - in.p < size + 16) bad("truncated block");
    std::string data;
    const uint8_t* blk = in.p;
    if (codec == "null") {
      data.assign((const char*)blk, size);
    } else if (codec == "deflate") {
      z_stream z{};
      if (inflateInit2(&z, -15) != Z_OK) bad("deflate init");
      z.next_in = (Bytef*)blk;
      z.avail_in = (uInt)size;
      char buf[1 << 16];
      int rc;
      do {
        z.next_out = (Bytef*)buf;
        z.avail_out = sizeof buf;
        rc = inflate(&z, Z_NO_FLUSH);
        if (rc != Z_OK && rc != Z_STREAM_END) { inflateEnd(&z); bad("bad deflate block"); }
        data.append(buf, sizeof buf - z.avail_out);
      } while (rc != Z_STREAM_END && z.avail_in > 0);
      inflateEnd(&z);
    } else if (codec == "snappy") {  // + 4-byte CRC32 of the uncompressed data
      if (size < 4 || !snappy::Uncompress((const char*)blk, size - 4, &data)) bad("bad snappy block");
    } else if (codec == "zstandard") {
      const unsigned long long n = ZSTD_getFrameContentSize(blk, size);
      if (n == ZSTD_CONTENTSIZE_ERROR || n == ZSTD_CONTENTSIZE_UNKNOWN) bad("bad zstd block");
      data.resize(n);
      if (ZSTD_isError(ZSTD_decompress(data.data(), n, blk, size))) bad("bad zstd block");
    } else {
      bad("unsupported codec " + codec);
    }
    in.p += size;
    if (std::string((const char*)in.p, 16) != sync) bad("sync marker mismatch");
    in.p += 16;
    In rec{(const uint8_t*)data.data(), (const uint8_t*)data.data() + data.size()};
    for (int64_t r = 0; r < count; ++r) {
      for (size_t f = 0; f < root->members.size(); ++f) {
        bool n;
        vals[f] = decode(*root->members[f], rec, n, false);
        nul[f] = n;
      }
      for (size_t k = 0; k < map.size(); ++k) {
        if (k) out += '\t';
        if (map[k] < 0 || nul[map[k]]) out += "\\N";
        else tsvAppend(out, vals[map[k]]);
      }
      out += '\n';
    }
  }
  return out;
}

}  // namespace gdb
