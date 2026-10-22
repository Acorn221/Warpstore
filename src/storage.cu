#include <thread>
#include "storage.h"

#include <algorithm>
#include <charconv>
#include <cstdio>
#include <cstring>
#include "gpu.h"
#include <climits>
#include <deque>
#include <future>
#include <unistd.h>
#include <zstd.h>

namespace gdb {

std::string lower(std::string s) {
  for (auto& c : s) c = (char)tolower((unsigned char)c);
  return s;
}

thread_local uintptr_t tTxnToken = 0;
thread_local std::vector<std::pair<Table*, TxnPending*>> tTxnEntries;

std::vector<UndoRec>*& undoLog() {
  thread_local std::vector<UndoRec>* log = nullptr;
  return log;
}

std::vector<Value> Table::rowValues(Loc l) const {
  std::vector<Value> r;
  for (size_t c = 0; c < cols.size(); ++c) {
    const auto& col = colData(l, (int)c);
    const ColumnMeta& m = cols[c];
    switch (m.pt) {
      case PType::VEC: {
        const float* f = (const float*)col.data() + (int64_t)l.row * m.vecDim;
        r.push_back(f[0] != f[0] ? Value::Null() : Value::S(formatVector(f, m.vecDim)));
        break;
      }
      case PType::I64: {
        int64_t v = ((const int64_t*)col.data())[l.row];
        r.push_back(v == kNullI64 ? Value::Null() : m.lt == LType::DATETIME ? Value::Ts(v)
                    : m.lt == LType::DECIMAL ? Value::Dec((double)v / pow10i(m.scale), m.scale) : Value::I(v));
        break;
      }
      case PType::I32: {
        int32_t v = ((const int32_t*)col.data())[l.row];
        if (v == kNullI32) r.push_back(Value::Null());
        else if (m.lt == LType::VARCHAR) r.push_back(Value::S(std::string(dicts[c]->strs[v])));
        else if (m.lt == LType::DATE) r.push_back(Value::Dt(v));
        else r.push_back(Value::I(v));
        break;
      }
      default: r.push_back(Value::D(((const double*)col.data())[l.row]));
    }
  }
  return r;
}

std::string& currentDb() {
  thread_local std::string db = "warpstore";
  return db;
}

std::string Catalog::key(const std::string& name) {
  return name.find('.') != std::string::npos ? lower(name) : lower(currentDb()) + "." + lower(name);
}

thread_local const std::map<std::string, std::string>* tTableOverride = nullptr;

std::shared_ptr<Table> Catalog::find(const std::string& name) {
  std::lock_guard<std::mutex> g(mu);
  std::string k = key(name);
  if (tTableOverride) {
    auto o = tTableOverride->find(k);
    if (o != tTableOverride->end()) k = o->second;
  }
  auto it = tables.find(k);
  return it == tables.end() ? nullptr : it->second;
}

std::shared_ptr<Table> Catalog::get(const std::string& name) {
  auto t = find(name);
  if (!t) throw SqlError("Table '" + (name.find('.') != std::string::npos ? name : currentDb() + "." + name) + "' doesn't exist", 1146);
  return t;
}

// (Mirrors and bitmaps come and go with snapshots: from the stream-ordered pool, as cudaMalloc /
// cudaFree synchronize the whole device.)
DevMirror::~DevMirror() {
  for (void* p : cols) tmpFree(p);
}

DelBits::~DelBits() { tmpFree(dev); }

const uint32_t* DelBits::device(int64_t words) {
  std::lock_guard<std::mutex> g(mu);
  words = std::max<int64_t>(words, 1);
  if (!dev || devCap < words) {  // room for what the host bitmap can grow to without reallocating
    const int64_t cap = std::max<int64_t>({words, (int64_t)host.capacity(), 1});
    uint32_t* nd = (uint32_t*)tmpAlloc((size_t)cap * 4);
    if (dev) {
      uint32_t* old = dev;
      rcuDefer([old] { tmpFree(old); });  // a query may still be reading it
    }
    dev = nd;
    devCap = cap;
    devWords = 0;
    devVer = ~0ull;
  }
  const bool fresh = devVer == ver;  // no bit changed since the last upload: only new words are missing
  const int64_t have = std::min<int64_t>(words, (int64_t)host.size());
  const int64_t from = fresh ? devWords : 0;
  if (have > from) h2d(dev + from, host.data() + from, (size_t)(have - from) * 4);
  if (words > have) {  // (rows the host bitmap has no word for: not deleted)
    std::vector<uint32_t> z((size_t)(words - have), 0);
    h2d(dev + have, z.data(), z.size() * 4);
  }
  devWords = std::max(fresh ? devWords : 0, have);
  devVer = ver;
  return dev;
}

DelBits& delForWrite(DelP& d) {
  if (d.use_count() > 1) {  // a snapshot holds this version: change a private copy
    auto nd = std::make_shared<DelBits>();
    nd->host.reserve(d->host.capacity());
    nd->host.assign(d->host.begin(), d->host.end());
    nd->ndel = d->ndel;
    d = nd;
  }
  return *d;
}

void WriteBuffer::reserveForAppend(int64_t rows) {
  if (data.use_count() > 1) {  // shared with a snapshot: never reallocate under it
    bool grow = false;
    for (auto& c : data->cols) {
      const size_t per = n ? c.size() / (size_t)n : 0;
      if (per && c.size() + per * (size_t)rows > c.capacity()) grow = true;
    }
    if (grow) {
      auto nd = std::make_shared<BufData>();
      nd->cols.resize(data->cols.size());
      for (size_t i = 0; i < nd->cols.size(); ++i) {
        const auto& src = data->cols[i];
        const size_t per = n ? src.size() / (size_t)n : 0;
        nd->cols[i].reserve(std::max(src.size() * 2, src.size() + per * (size_t)rows + 4096));
        nd->cols[i].assign(src.begin(), src.end());
      }
      data = nd;  // (no VRAM mirror yet: the next scan uploads it)
      ++gen;
    }
  }
  if (del.use_count() > 1 && del->host.size() + (size_t)(rows + 31) / 32 + 1 > del->host.capacity()) {
    auto nd = std::make_shared<DelBits>();
    nd->host.reserve(std::max(del->host.size() * 2, del->host.size() + (size_t)(rows + 31) / 32 + 1024));
    nd->host.assign(del->host.begin(), del->host.end());
    nd->ndel = del->ndel;
    del = nd;
  }
}

// ---- RCU: a ticket per snapshot query; deferred frees wait for the queries older than them.
namespace {
std::mutex rcuMu;
uint64_t rcuNext = 1;
std::multiset<uint64_t> rcuActive;
std::vector<std::pair<uint64_t, std::function<void()>>> rcuPending;  // (ticket at deferral, free)
void rcuReclaim(std::unique_lock<std::mutex>& lk) {
  const uint64_t oldest = rcuActive.empty() ? UINT64_MAX : *rcuActive.begin();
  std::vector<std::function<void()>> run;
  for (size_t i = 0; i < rcuPending.size();)
    if (rcuPending[i].first <= oldest) {
      run.push_back(std::move(rcuPending[i].second));
      rcuPending[i] = std::move(rcuPending.back());
      rcuPending.pop_back();
    } else {
      ++i;
    }
  lk.unlock();
  for (auto& f : run) f();
  lk.lock();
}
}  // namespace

uint64_t rcuEnter() {
  std::lock_guard<std::mutex> g(rcuMu);
  const uint64_t t = rcuNext++;
  rcuActive.insert(t);
  return t;
}

void rcuExit(uint64_t ticket) {
  std::unique_lock<std::mutex> lk(rcuMu);
  rcuActive.erase(rcuActive.find(ticket));
  if (!rcuPending.empty()) rcuReclaim(lk);
}

void rcuDefer(std::function<void()> fn) {
  std::unique_lock<std::mutex> lk(rcuMu);
  // queries that entered before now hold tickets < rcuNext: run once all of those have exited
  rcuPending.push_back({rcuNext++, std::move(fn)});
  rcuReclaim(lk);
}

bool parseVector(const std::string& s, std::vector<float>& out) {
  out.clear();
  size_t i = s.find('[');
  if (i == std::string::npos) return false;
  ++i;
  while (i < s.size()) {
    while (i < s.size() && (isspace((unsigned char)s[i]) || s[i] == ',')) ++i;
    if (i < s.size() && s[i] == ']') return true;
    float v = 0;
    auto [p, ec] = std::from_chars(s.data() + i, s.data() + s.size(), v);
    if (ec != std::errc() || p == s.data() + i) {  // from_chars rejects a leading '+': fall back
      char* end = nullptr;
      v = strtof(s.c_str() + i, &end);
      if (end == s.c_str() + i) return false;
      p = end;
    }
    out.push_back(v);
    i = p - s.data();
  }
  return false;
}

std::string formatVector(const float* v, int dim) {
  if (dim > 0 && v[0] != v[0]) return "";
  std::string r = "[";
  r.reserve((size_t)dim * 12 + 2);
  char b[32];
  for (int i = 0; i < dim; ++i) {  // shortest text that reads back as the same float
    if (i) r += ',';
    auto [p, ec] = std::to_chars(b, b + sizeof b, v[i]);
    r.append(b, p);
  }
  return r + "]";
}

uint64_t shardHashInt(int64_t v) {
  uint64_t x = (uint64_t)v;
  x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ull;
  x ^= x >> 27; x *= 0x94d049bb133111ebull;
  return x ^ (x >> 31);
}
uint64_t shardHashDbl(double v) {
  if (v == 0) v = 0;  // -0.0 == 0.0
  int64_t bits;
  memcpy(&bits, &v, 8);
  return shardHashInt(bits);
}
uint64_t shardHashStr(std::string_view s) {
  uint64_t h = 0xcbf29ce484222325ull;
  for (unsigned char ch : s) h = (h ^ ch) * 0x100000001b3ull;
  return shardHashInt((int64_t)h);
}

thread_local bool tStrictTypes = false;

void deriveLimits(ColumnMeta& c) {
  std::string t;
  for (char ch : c.typeText) t += (char)toupper((unsigned char)ch);
  const bool uns = t.find(" UNSIGNED") != std::string::npos;
  std::string base = t.substr(0, t.find_first_of("( "));
  std::vector<std::string> args;  // inside the parentheses
  if (size_t o = t.find('('); o != std::string::npos) {
    const size_t e = t.rfind(')');
    std::string in = c.typeText.substr(o + 1, e == std::string::npos ? std::string::npos : e - o - 1), cur;
    bool q = false;
    for (size_t i = 0; i < in.size(); ++i) {
      const char ch = in[i];
      if (ch == '\'' && (!q || i + 1 >= in.size() || in[i + 1] != '\'')) { q = !q; continue; }
      if (ch == '\'' && q) { cur += ch; ++i; continue; }  // ('' inside quotes)
      if (ch == ',' && !q) { args.push_back(cur); cur.clear(); continue; }
      if (q || !isspace((unsigned char)ch)) cur += ch;
    }
    args.push_back(cur);
  }
  auto num = [&](size_t i, int64_t d) { return i < args.size() && !args[i].empty() && isdigit((unsigned char)args[i][0]) ? std::atoll(args[i].c_str()) : d; };
  auto range = [&](int64_t lo, int64_t hi) { c.ranged = true, c.imin = lo, c.imax = hi; };
  if (base == "TINYINT" || base == "BOOL" || base == "BOOLEAN") uns ? range(0, 255) : range(-128, 127);
  else if (base == "SMALLINT") uns ? range(0, 65535) : range(-32768, 32767);
  else if (base == "MEDIUMINT") uns ? range(0, 16777215) : range(-8388608, 8388607);
  else if (base == "INT" || base == "INTEGER") uns ? range(0, 4294967295ll) : range(INT32_MIN + 1ll, INT32_MAX);  // (INT32_MIN: NULL)
  else if (base == "BIGINT") uns ? range(0, INT64_MAX) : range(INT64_MIN + 1, INT64_MAX);
  else if (base == "YEAR") range(0, 2155);
  else if (base == "BIT") range(0, (int64_t)((1ull << std::min<int64_t>(num(0, 1), 63)) - 1));
  else if (c.lt == LType::DECIMAL) c.precision = (int)num(0, 10);
  else if (base == "VARCHAR" || base == "CHAR" || base == "NVARCHAR" || base == "NCHAR") c.maxChars = num(0, base == "CHAR" || base == "NCHAR" ? 1 : -1);
  else if (base == "BINARY") c.maxBytes = num(0, 1);
  else if (base == "VARBINARY") c.maxBytes = num(0, -1);
  else if (base == "TINYTEXT" || base == "TINYBLOB") c.maxBytes = 255;
  else if (base == "TEXT" || base == "BLOB") c.maxBytes = 65535;
  else if (base == "MEDIUMTEXT" || base == "MEDIUMBLOB") c.maxBytes = 16777215;
  else if (base == "ENUM") c.enumVals = args;
}

Table::Table(std::string n, std::vector<ColumnMeta> c, int pk_, int sk, std::vector<int> pkc, std::vector<int> shard,
             int np, bool ref)
    : name(std::move(n)), cols(std::move(c)), pk(pk_), sortKey(sk), pkCols(std::move(pkc)), shardKey(std::move(shard)),
      nparts(ref ? 1 : std::max(np, 1)), reference(ref) {
  dicts.resize(cols.size());
  hasNull.assign(cols.size(), 0);
  curVec.resize(cols.size());
  for (size_t i = 0; i < cols.size(); ++i) {
    deriveLimits(cols[i]);
    if (cols[i].lt == LType::VARCHAR) dicts[i] = std::make_unique<Dict>();
    if (cols[i].autoInc) autoIncCol = (int)i;
  }
  for (int p = 0; p < nparts; ++p) {
    bufs.push_back(std::make_unique<WriteBuffer>());
    bufs.back()->cols.resize(cols.size());
  }
}

int Table::partOf(const std::vector<int64_t>& ints, const std::vector<double>& dbls) const {
  if (nparts == 1) return 0;
  if (shardKey.empty()) return (int)(nextPart++ % nparts);
  uint64_t h = 0;
  for (int c : shardKey) {
    uint64_t x = cols[c].lt == LType::VARCHAR ? (ints[c] == kNullI32 ? 0 : shardHashStr(dicts[c]->strs[ints[c]]))
               : cols[c].pt == PType::F64     ? shardHashDbl(dbls[c])
                                              : shardHashInt(ints[c]);
    h = shardCombine(h, x);
  }
  return (int)(h % (uint64_t)nparts);
}

int Table::partOfValues(const std::vector<Value>& vals) const {
  if (nparts == 1) return 0;
  if (shardKey.empty() || vals.size() != shardKey.size()) return -1;
  uint64_t h = 0;
  for (size_t i = 0; i < shardKey.size(); ++i) {
    const ColumnMeta& cm = cols[shardKey[i]];
    const Value& v = vals[i];
    uint64_t x;
    if (cm.lt == LType::VARCHAR) {
      if (v.kind != Value::STR) return -1;
      x = shardHashStr(v.s);
    } else if (cm.lt == LType::DATE) {
      int32_t d;
      if (v.kind == Value::STR) {
        if (!parseDate(v.s, d)) return -1;
      } else {
        d = (int32_t)v.i;
      }
      x = shardHashInt(d);
    } else if (cm.lt == LType::DECIMAL) {
      return -1;  // no shard pruning on DECIMAL keys
    } else if (cm.pt == PType::F64) {
      if (v.kind == Value::STR) return -1;
      x = shardHashDbl(v.num());
    } else {
      if (v.kind == Value::STR) return -1;
      double d = v.num();
      if (d != std::floor(d)) return -1;  // no row can match a fractional value
      x = shardHashInt(v.kind == Value::DBL ? (int64_t)d : v.i);
    }
    h = shardCombine(h, x);
  }
  return (int)(h % (uint64_t)nparts);
}

int Table::colIndex(const std::string& n) const {
  for (size_t i = 0; i < cols.size(); ++i)
    if (strcasecmp(cols[i].name.c_str(), n.c_str()) == 0) return (int)i;
  return -1;
}

int64_t Table::bufferRows() const {
  int64_t r = 0;
  for (auto& b : bufs) r += b->n - b->ndel();
  return r;
}
int64_t Table::bufferTotal() const {
  int64_t r = 0;
  for (auto& b : bufs) r += b->n;
  return r;
}

void Table::fillHostMirror(Segment& sg, int c) const {
  static std::mutex mu;
  std::lock_guard<std::mutex> g(mu);
  SegData& d = sg.cd(c);
  segEnsureHost(d, c, (uint64_t)sg.n * colBytes(c));  // (GPU-built segments, GPU-only UPDATE versions)
}

int64_t Table::liveRows() const {
  int64_t r = bufferRows();
  for (auto& s : segs) r += s->n - s->ndel();
  return r;
}

int64_t Table::vramBytes() const {
  int64_t b = 0;
  for (auto& s : segs) {
    b += s->devBytes();
    b += (s->n + 31) / 32 * 4;
  }
  for (auto& bp : bufs) {
    std::lock_guard<std::mutex> g(bp->data->devMu);
    if (bp->data->mirror)
      for (size_t c = 0; c < cols.size(); ++c) b += bp->data->mirror->cap * colBytes((int)c);
  }
  return b;
}

static inline void pushRaw(std::vector<uint8_t>& v, const void* src, int sz) {
  size_t o = v.size();
  v.resize(o + sz);
  memcpy(v.data() + o, src, sz);
}

static double readNum(const std::vector<uint8_t>& col, PType t, int64_t r) {  // NULL -> NaN
  switch (t) {
    case PType::VEC: return nullv();
    case PType::I32: { int32_t v = ((const int32_t*)col.data())[r]; return v == kNullI32 ? nullv() : v; }
    case PType::I64: { int64_t v = ((const int64_t*)col.data())[r]; return v == kNullI64 ? nullv() : (double)v; }
    default: return ((const double*)col.data())[r];
  }
}
static int64_t readInt(const std::vector<uint8_t>& col, PType t, int64_t r) {
  switch (t) {
    case PType::I32: return ((const int32_t*)col.data())[r];
    case PType::I64: return ((const int64_t*)col.data())[r];
    default: return (int64_t)((const double*)col.data())[r];
  }
}

double Table::cellNum(Loc l, int c) const {
  double v = readNum(colData(l, c), cols[c].pt, l.row);
  return cols[c].lt == LType::DECIMAL ? v / pow10i(cols[c].scale) : v;
}

int64_t Table::pkValue(Loc l) const { return readInt(colData(l, pk), cols[pk].pt, l.row); }

bool Table::rawKey(Loc l, int c, int64_t& k) const {
  const auto& col = colData(l, c);
  switch (cols[c].pt) {
    case PType::I32: k = ((const int32_t*)col.data())[l.row]; return k != kNullI32;
    case PType::I64: k = ((const int64_t*)col.data())[l.row]; return k != kNullI64;
    default: {
      double d = ((const double*)col.data())[l.row];
      if (d != d) return false;
      if (d == 0) d = 0;
      memcpy(&k, &d, 8);
      return true;
    }
  }
}

bool Table::ixKey(const IndexMeta& ix, Loc l, int64_t& k) const {
  if (!ix.vec) return rawKey(l, ix.cols[0], k);
  if (ix.cent.empty()) {
    k = 0;
    return true;
  }
  const float* v = (const float*)colData(l, ix.cols[0]).data() + (size_t)l.row * ix.dim;
  float best = 3.4e38f;
  for (int c = 0; c < ix.nlist; ++c) {
    const float* ce = ix.cent.data() + (size_t)c * ix.dim;
    float d = 0;
    for (int i = 0; i < ix.dim; ++i) d += (v[i] - ce[i]) * (v[i] - ce[i]);
    if (d < best) best = d, k = c;
  }
  return true;
}

// k-means (Lloyd, L2) over a sample of the live vectors; nlist defaults to sqrt(rows).
void Table::trainVectorIndex(IndexMeta& ix) {
  const int c0 = ix.cols[0];
  ix.dim = cols[c0].vecDim;
  std::vector<Loc> locs;
  for (size_t si = 0; si < segs.size(); ++si)
    for (int64_t r = 0; r < segs[si]->n; ++r)
      if (!segs[si]->del->test(r)) locs.push_back(Loc{(int32_t)si, (int32_t)r});
  for (int p = 0; p < nparts; ++p)
    for (int64_t r = 0; r < bufs[p]->n; ++r)
      if (!bufs[p]->del->test(r)) locs.push_back(bufLoc(p, r));
  const int64_t n = (int64_t)locs.size();
  ix.trainedRows = n;
  ix.cent.clear();
  if (n < 256) {  // too few rows to be worth clustering: one list (exact search)
    ix.nlist = 0;
    return;
  }
  int want = 0;
  {
    size_t p = ix.options.find("\"nlist\"");
    if (p != std::string::npos) want = std::atoi(ix.options.c_str() + ix.options.find(':', p) + 1);
  }
  const int nl = std::max(1, std::min<int>(want > 0 ? want : (int)std::sqrt((double)n), (int)std::min<int64_t>(n / 8, 65536)));
  const int d = ix.dim;
  const int64_t sampleN = std::min<int64_t>(n, std::max<int64_t>((int64_t)nl * 40, 20000));
  std::vector<float> sample((size_t)sampleN * d);
  uint64_t rng = 0x9E3779B97F4A7C15ull;
  for (int64_t i = 0; i < sampleN; ++i) {
    rng ^= rng << 13, rng ^= rng >> 7, rng ^= rng << 17;
    Loc l = sampleN == n ? locs[i] : locs[rng % n];
    memcpy(&sample[(size_t)i * d], (const float*)colData(l, c0).data() + (size_t)l.row * d, 4 * d);
  }
  std::vector<float> cent((size_t)nl * d);
  for (int c = 0; c < nl; ++c) memcpy(&cent[(size_t)c * d], &sample[(size_t)((int64_t)c * sampleN / nl) * d], 4 * d);
  std::vector<int> asg(sampleN);
  const int nth = std::max(1u, std::min(16u, std::thread::hardware_concurrency()));
  for (int it = 0; it < 12; ++it) {
    std::vector<std::thread> th;
    for (int t = 0; t < nth; ++t)
      th.emplace_back([&, t] {
        for (int64_t i = t; i < sampleN; i += nth) {
          const float* v = &sample[(size_t)i * d];
          float best = 3.4e38f;
          int bi = 0;
          for (int c = 0; c < nl; ++c) {
            const float* ce = &cent[(size_t)c * d];
            float s = 0;
            for (int j = 0; j < d; ++j) s += (v[j] - ce[j]) * (v[j] - ce[j]);
            if (s < best) best = s, bi = c;
          }
          asg[i] = bi;
        }
      });
    for (auto& x : th) x.join();
    std::vector<double> sum((size_t)nl * d, 0);
    std::vector<int64_t> cnt(nl, 0);
    for (int64_t i = 0; i < sampleN; ++i) {
      ++cnt[asg[i]];
      for (int j = 0; j < d; ++j) sum[(size_t)asg[i] * d + j] += sample[(size_t)i * d + j];
    }
    for (int c = 0; c < nl; ++c) {
      if (!cnt[c]) {  // empty cluster: reseed from a random sample
        rng ^= rng << 13, rng ^= rng >> 7, rng ^= rng << 17;
        memcpy(&cent[(size_t)c * d], &sample[(size_t)(rng % sampleN) * d], 4 * d);
        continue;
      }
      for (int j = 0; j < d; ++j) cent[(size_t)c * d + j] = (float)(sum[(size_t)c * d + j] / cnt[c]);
    }
  }
  ix.cent = std::move(cent);
  ix.nlist = nl;
}

void Table::indexAdd(Loc l) {
  for (auto& ix : indexes) {
    if (ix.vec && ix.algo == 1) continue;  // CAGRA graphs are per segment, built on demand
    int64_t k;
    if (ixKey(ix, l, k)) ix.map[k].push_back(l);
  }
}

void Table::indexRemove(Loc l) {
  for (auto& ix : indexes) {
    if (ix.vec && ix.algo == 1) continue;
    int64_t k;
    if (!ixKey(ix, l, k)) continue;
    auto it = ix.map.find(k);
    if (it == ix.map.end()) continue;
    auto& v = it->second;
    v.erase(std::remove(v.begin(), v.end(), l), v.end());
    if (v.empty()) ix.map.erase(it);
  }
}

void Table::buildIndex(IndexMeta& ix) {
  ix.map.clear();
  if (ix.vec && ix.algo == 1) return;  // CAGRA: graphs are built per segment when first searched
  if (ix.vec) trainVectorIndex(ix);
  // Assigning rows to IVF lists is the expensive part: precompute keys in parallel.
  std::vector<Loc> all;
  if (ix.vec && ix.nlist > 0) {
    for (size_t si = 0; si < segs.size(); ++si)
      for (int64_t r = 0; r < segs[si]->n; ++r)
        if (!segs[si]->del->test(r)) all.push_back(Loc{(int32_t)si, (int32_t)r});
    for (int p = 0; p < nparts; ++p)
      for (int64_t r = 0; r < bufs[p]->n; ++r)
        if (!bufs[p]->del->test(r)) all.push_back(bufLoc(p, r));
    std::vector<int64_t> keys(all.size());
    const int nth = std::max(1u, std::min(16u, std::thread::hardware_concurrency()));
    std::vector<std::thread> th;
    for (int t = 0; t < nth; ++t)
      th.emplace_back([&, t] {
        for (size_t i = t; i < all.size(); i += nth) ixKey(ix, all[i], keys[i]);
      });
    for (auto& x : th) x.join();
    for (size_t i = 0; i < all.size(); ++i) ix.map[keys[i]].push_back(all[i]);
    return;
  }
  auto add = [&](Loc l) {
    int64_t k;
    if (ixKey(ix, l, k)) ix.map[k].push_back(l);
  };
  for (size_t si = 0; si < segs.size(); ++si)
    for (int64_t r = 0; r < segs[si]->n; ++r)
      if (!segs[si]->del->test(r)) add(Loc{(int32_t)si, (int32_t)r});
  for (int p = 0; p < nparts; ++p)
    for (int64_t r = 0; r < bufs[p]->n; ++r)
      if (!bufs[p]->del->test(r)) add(bufLoc(p, r));
}

void Table::appendRaw(const std::vector<int64_t>& ints, const std::vector<double>& dbls) {
  // UNIQUE secondary keys: checked before anything is modified.
  lastDup = Loc{INT32_MIN, 0};
  for (auto& ix : indexes) {
    if (!ix.unique) continue;
    const int c0 = ix.cols[0];
    int64_t k;
    if (cols[c0].pt == PType::F64) {
      double d = dbls[c0];
      if (d != d) continue;
      if (d == 0) d = 0;
      memcpy(&k, &d, 8);
    } else {
      k = ints[c0];
      if (k == (cols[c0].pt == PType::I32 ? (int64_t)kNullI32 : kNullI64)) continue;
    }
    auto it = ix.map.find(k);
    if (it == ix.map.end()) continue;
    for (const Loc& l : it->second) {
      bool same = true;
      for (int c : ix.cols) {
        int64_t a;
        bool nn = rawKey(l, c, a);
        int64_t b = cols[c].pt == PType::F64 ? 0 : ints[c];
        if (cols[c].pt == PType::F64) {
          double d = dbls[c];
          if (d == 0) d = 0;
          memcpy(&b, &d, 8);
        }
        same = same && nn && a == b;
      }
      if (same) {
        lastDup = l;
        throw SqlError("Duplicate entry for key '" + ix.name + "'", 1062);
      }
    }
  }
  const int part = partOf(ints, dbls);
  WriteBuffer& buf = *bufs[part];
  buf.reserveForAppend(1);  // (copies storage a snapshot shares before it would reallocate)
  if (pk >= 0) {
    if (ints[pk] == kNullI32 && cols[pk].pt == PType::I32) throw SqlError("Column '" + cols[pk].name + "' cannot be null", 1048);
    if (ints[pk] == kNullI64 && cols[pk].pt == PType::I64) throw SqlError("Column '" + cols[pk].name + "' cannot be null", 1048);
    int64_t key = cols[pk].pt == PType::F64 ? (int64_t)dbls[pk] : ints[pk];
    auto [it, inserted] = pkIndex.try_emplace(key, bufLoc(part, buf.n));
    if (!inserted) throw SqlError("Duplicate entry '" + std::to_string(key) + "' for key 'PRIMARY'", 1062);
  }
  for (size_t c = 0; c < cols.size(); ++c) {
    switch (cols[c].pt) {
      case PType::I32: { int32_t v = (int32_t)ints[c]; pushRaw(buf.cols[c], &v, 4); break; }
      case PType::I64: { int64_t v = ints[c]; pushRaw(buf.cols[c], &v, 8); break; }
      case PType::F64: { double v = dbls[c]; pushRaw(buf.cols[c], &v, 8); break; }
      case PType::VEC: {
        std::vector<float>& v = curVec[c];
        v.resize(cols[c].vecDim, v.empty() ? nullv() : 0.0f);
        pushRaw(buf.cols[c], v.data(), 4 * cols[c].vecDim);
        v.clear();
        break;
      }
    }
  }
  if ((buf.n & 31) == 0) buf.del->host.push_back(0);
  const Loc nl = bufLoc(part, buf.n);
  ++buf.n;
  ++buf.version;
  if (txnKey() && db != "__tmp") {  // invisible to others until COMMIT
    TxnPending& tp = pendingOfMine();
    tp.hidden.push_back(nl);
    tp.hiddenIx.insert(TxnPending::key(nl));
  }
  if (!indexes.empty()) indexAdd(nl);
  if (buf.n >= kSegRows && !deferFlush) flushPart(part);
}

// A value as column c stores it (NULL checks, conversions, string -> code); VECTOR cells go to curVec.
void Table::checkFits(int c, int64_t raw) const {
  const ColumnMeta& cm = cols[c];
  if (raw == (cm.pt == PType::I64 ? kNullI64 : (int64_t)kNullI32)) return;
  bool bad = cm.ranged && (raw < cm.imin || raw > cm.imax);
  if (cm.lt == LType::DECIMAL && cm.precision > 0 && cm.precision < 19) {
    const int64_t lim = (int64_t)pow10i(cm.precision);
    bad = raw >= lim || raw <= -lim;
  }
  if (cm.pt == PType::I32 && (raw > INT32_MAX || raw <= INT32_MIN)) bad = true;  // (storage)
  if (bad) throw SqlError("Out of range value for column '" + cm.name + "'", 1264);
}

void Table::checkFitsColumn(int c, const std::vector<int64_t>& iv, size_t from, size_t n) const {
  const ColumnMeta& cm = cols[c];
  if (cm.lt == LType::VARCHAR) {
    if (cm.maxChars < 0 && cm.maxBytes < 0 && cm.enumVals.empty()) return;
    const Dict& d = *dicts[c];
    std::vector<uint8_t> seen(d.strs.size(), 0);
    for (size_t k = from; k < from + n; ++k) {
      const int64_t code = iv[k];
      if (code < 0 || (size_t)code >= seen.size() || seen[code]) continue;
      seen[code] = 1;
      checkFitsStr(c, d.strs[(size_t)code]);
    }
    return;
  }
  if (cm.pt == PType::F64 || cm.lt == LType::DATE || cm.lt == LType::DATETIME) return;
  if (!cm.ranged && !(cm.lt == LType::DECIMAL && cm.precision > 0) && cm.pt != PType::I32) return;
  for (size_t k = from; k < from + n; ++k) checkFits(c, iv[k]);
}

void Table::checkFitsStr(int c, std::string_view str) const {
  const ColumnMeta& cm = cols[c];
  if (cm.maxBytes >= 0 && (int64_t)str.size() > cm.maxBytes) throw SqlError("Data too long for column '" + cm.name + "'", 1406);
  if (cm.maxChars >= 0 && (int64_t)str.size() > cm.maxChars) {  // (bytes <= limit: fits whatever the characters)
    int64_t chars = 0;
    for (unsigned char ch : str) chars += (ch & 0xC0) != 0x80;
    if (chars > cm.maxChars) throw SqlError("Data too long for column '" + cm.name + "'", 1406);
  }
  if (!cm.enumVals.empty()) {
    for (auto& e : cm.enumVals)
      if (e.size() == str.size() && strncasecmp(e.data(), str.data(), str.size()) == 0) return;
    throw SqlError("Data truncated for column '" + cm.name + "'", 1265);
  }
}

void Table::rawOf(int c, const Value& v, int64_t& i, double& d) {
  if (v.isNull()) {
    if (cols[c].notNull || (int)c == pk) throw SqlError("Column '" + cols[c].name + "' cannot be null", 1048);
    hasNull[c] = 1;
    i = cols[c].pt == PType::I64 ? kNullI64 : kNullI32;
    d = nullv();
    return;
  }
  switch (cols[c].lt) {
    case LType::INT:
    case LType::BIGINT:
      if (v.kind == Value::STR) {  // (strict: the whole string a number; otherwise its leading number, as MySQL)
        const char* b = v.s.c_str();
        char* e = nullptr;
        errno = 0;
        const double dv = strtod(b, &e);
        while (e && isspace((unsigned char)*e)) ++e;
        const bool whole = e != b && e && *e == 0;
        if (tStrictTypes && (!whole || errno == ERANGE || dv != dv))
          throw SqlError("Incorrect integer value: '" + v.s + "' for column '" + cols[c].name + "'", 1366);
        if (tStrictTypes && (dv >= 9.2233720368547758e18 || dv < -9.2233720368547758e18))
          throw SqlError("Out of range value for column '" + cols[c].name + "'", 1264);
        i = whole && v.s.find_first_of(".eE") == std::string::npos ? std::strtoll(b, nullptr, 10) : (int64_t)std::llround(dv);
      } else if (v.kind == Value::DBL) {
        if (tStrictTypes && !(v.d < 9.2233720368547758e18 && v.d >= -9.2233720368547758e18))
          throw SqlError("Out of range value for column '" + cols[c].name + "'", 1264);
        i = (int64_t)std::llround(v.d);
      } else {
        i = v.i;
      }
      if (tStrictTypes) checkFits(c, i);
      break;
    case LType::DOUBLE: d = v.kind == Value::STR ? std::stod(v.s) : v.num(); break;
    case LType::DECIMAL: {
      int64_t raw;
      if (v.kind == Value::STR) {
        if (!parseDecimal(v.s.data(), v.s.data() + v.s.size(), cols[c].scale, raw))
          throw SqlError("Incorrect decimal value: '" + v.s + "' for column '" + cols[c].name + "'", 1366);
      } else if (v.kind == Value::DBL) {
        raw = std::llround(v.d * pow10i(cols[c].scale));
      } else {
        raw = v.i * (int64_t)pow10i(cols[c].scale);
      }
      i = raw;
      if (tStrictTypes) checkFits(c, i);
      break;
    }
    case LType::DATE: {
      if (v.kind == Value::STR) {
        int32_t day;
        if (!parseDate(v.s, day)) throw SqlError("Incorrect date value: '" + v.s + "'", 1292);
        i = day;
      } else if (v.kind == Value::DATETIME) {
        i = floorDiv(v.i, kUsPerDay);
      } else {
        i = v.i;
      }
      break;
    }
    case LType::DATETIME: {
      if (v.kind == Value::STR) {
        int64_t us;
        if (!parseDateTime(v.s, us)) throw SqlError("Incorrect datetime value: '" + v.s + "'", 1292);
        i = us;
      } else if (v.kind == Value::DATE) {
        i = v.i * kUsPerDay;
      } else {
        i = v.i;
      }
      break;
    }
    case LType::VARCHAR: {
      const std::string& str = v.kind == Value::STR ? v.s : v.toString();
      if (tStrictTypes) checkFitsStr(c, str);
      const std::string* put = &str;
      for (auto& e : cols[c].enumVals)  // (an ENUM keeps the member's spelling)
        if (e.size() == str.size() && strncasecmp(e.data(), str.data(), str.size()) == 0) put = &e;
      i = dicts[c]->getOrAdd(*put);
      break;
    }
    case LType::VECTOR: {
      if (!parseVector(v.toString(), curVec[c]) || (int)curVec[c].size() != cols[c].vecDim)
        throw SqlError("Invalid VECTOR(" + std::to_string(cols[c].vecDim) + ") value for column '" + cols[c].name + "'", 1105);
      i = 0;
      break;
    }
  }
}

void Table::appendColumns(const std::vector<std::vector<int64_t>>& iv, const std::vector<std::vector<double>>& dv, int64_t n,
                          const int32_t* partOfRow, bool nullsKnown) {
  const size_t nc = cols.size();
  for (size_t c = 0; c < nc; ++c)
    if (cols[c].pt == PType::VEC) throw SqlError("internal: appendColumns with a VECTOR column", 1105);
  if (!nullsKnown)  // (the column scans on all cores; a caller that set hasNull while converting skips them)
    ioParallel(nc, [&](size_t c) {
      if (hasNull[c]) return;
      if (cols[c].pt == PType::F64) {
        for (double x : dv[c])
          if (x != x) { hasNull[c] = 1; return; }
      } else {
        const int64_t nul = cols[c].pt == PType::I64 ? kNullI64 : kNullI32;
        for (int64_t x : iv[c])
          if (x == nul) { hasNull[c] = 1; return; }
      }
    }, true);
  std::vector<int64_t> ints(nc);
  std::vector<double> dbls(nc);
  auto cell = [&](size_t c, int64_t r) {
    if (cols[c].pt == PType::F64) dbls[c] = dv[c][r];
    else ints[c] = iv[c][r];
  };
  if (pk >= 0 || !indexes.empty() || (txnKey() && db != "__tmp")) {  // keys, indexes, transactions: row by row
    for (int64_t r = 0; r < n; ++r) {
      for (size_t c = 0; c < nc; ++c) cell(c, r);
      appendRaw(ints, dbls);
    }
    return;
  }
  std::vector<std::vector<uint32_t>> rowsOf(nparts);
  if (nparts == 1 || shardKey.empty()) {  // no shard key: a contiguous slice per partition
    for (int i = 0; i < nparts; ++i) {
      const int p = (int)((nextPart + i) % nparts);
      for (int64_t g = n * i / nparts; g < n * (i + 1) / nparts; ++g) rowsOf[p].push_back((uint32_t)g);
    }
    ++nextPart;
  } else if (partOfRow) {
    for (int64_t r = 0; r < n; ++r) rowsOf[partOfRow[r]].push_back((uint32_t)r);
  } else {
    for (int64_t r = 0; r < n; ++r) {
      for (int c : shardKey) cell(c, r);
      rowsOf[partOf(ints, dbls)].push_back((uint32_t)r);
    }
  }
  for (int p = 0; p < nparts; ++p) {
    const std::vector<uint32_t>& list = rowsOf[p];
    for (size_t done = 0; done < list.size();) {
      WriteBuffer& buf = *bufs[p];
      const size_t room = buf.n < kSegRows ? (size_t)(kSegRows - buf.n) : list.size() - done;
      const size_t k = std::min(list.size() - done, room);
      buf.reserveForAppend((int64_t)k);
      const uint32_t* idx = list.data() + done;
      const bool run = k > 0 && idx[k - 1] - idx[0] == k - 1;  // consecutive rows (no shard key): no index
      ioParallel(nc, [&](size_t c) {
        std::vector<uint8_t>& dst = buf.cols[c];
        const size_t w = (size_t)psize(cols[c].pt), o = dst.size();
        if (k >= 4096 && dst.capacity() < o + k * w)  // bulk: room for a whole segment at once, no copy per growth (pages touched as filled)
          dst.reserve(std::max<size_t>(o + k * w, (size_t)kSegRows * w));
        dst.resize(o + k * w);
        uint8_t* d = dst.data() + o;
        if (run) {
          const size_t f = idx[0];
          switch (cols[c].pt) {
            case PType::I32: {
              const int64_t* sp = iv[c].data() + f;
              int32_t* dp = (int32_t*)d;
              for (size_t j = 0; j < k; ++j) dp[j] = (int32_t)sp[j];
              break;
            }
            case PType::I64: memcpy(d, iv[c].data() + f, k * 8); break;
            default: memcpy(d, dv[c].data() + f, k * 8); break;
          }
          return;
        }
        switch (cols[c].pt) {
          case PType::I32: for (size_t j = 0; j < k; ++j) ((int32_t*)d)[j] = (int32_t)iv[c][idx[j]]; break;
          case PType::I64: for (size_t j = 0; j < k; ++j) ((int64_t*)d)[j] = iv[c][idx[j]]; break;
          default: for (size_t j = 0; j < k; ++j) ((double*)d)[j] = dv[c][idx[j]]; break;
        }
      }, true);
      buf.del->host.resize((size_t)(buf.n + (int64_t)k + 31) / 32, 0);
      buf.n += (int64_t)k;
      ++buf.version;
      done += k;
      if (buf.n >= kSegRows && !deferFlush) flushPart(p);
    }
  }
}

void Table::appendCopy(const Loc* ls, size_t n, const int32_t* part, const std::vector<uint8_t>& given,
                       const std::vector<std::vector<int64_t>>& iv, const std::vector<std::vector<double>>& dv) {
  const size_t nc = cols.size();
  ioParallel(nc, [&](size_t c) {  // (copied columns: their NULLs are in hasNull already)
    if (!given[c] || hasNull[c]) return;
    if (cols[c].pt == PType::F64) {
      for (double x : dv[c])
        if (x != x) { hasNull[c] = 1; return; }
    } else {
      const int64_t nul = cols[c].pt == PType::I64 ? kNullI64 : kNullI32;
      for (int64_t x : iv[c])
        if (x == nul) { hasNull[c] = 1; return; }
    }
  }, true);
  std::vector<std::vector<uint32_t>> rowsOf(nparts);
  for (size_t i = 0; i < n; ++i) rowsOf[part[i]].push_back((uint32_t)i);
  // Segment columns resolved here, on this thread: a cold column (disk tier) is loaded through the I/O
  // pool, which a pool task must not wait for (every pool thread waiting: a deadlock).
  std::map<int32_t, std::vector<const uint8_t*>> segCols;
  for (size_t i = 0; i < n; ++i)
    if (ls[i].seg >= 0 && !segCols.count(ls[i].seg)) {
      auto& v = segCols[ls[i].seg];
      v.assign(nc, nullptr);
      for (size_t c = 0; c < nc; ++c)
        if (!given[c]) v[c] = colData(ls[i], (int)c).data();
    }
  for (int p = 0; p < nparts; ++p) {
    const std::vector<uint32_t>& list = rowsOf[p];
    for (size_t done = 0; done < list.size();) {
      WriteBuffer& buf = *bufs[p];
      const size_t room = buf.n < kSegRows ? (size_t)(kSegRows - buf.n) : list.size() - done;
      const size_t k = std::min(list.size() - done, room);
      buf.reserveForAppend((int64_t)k);
      const uint32_t* idx = list.data() + done;
      ioParallel(nc, [&](size_t c) {
        std::vector<uint8_t>& dst = buf.cols[c];
        const size_t w = (size_t)psize(cols[c].pt), o = dst.size();
        if (k >= 4096 && dst.capacity() < o + k * w) dst.reserve(std::max<size_t>(o + k * w, (size_t)kSegRows * w));
        dst.resize(o + k * w);  // (first: a source row may be in this very buffer)
        uint8_t* d = dst.data() + o;
        if (given[c]) {
          switch (cols[c].pt) {
            case PType::I32: for (size_t j = 0; j < k; ++j) ((int32_t*)d)[j] = (int32_t)iv[c][idx[j]]; break;
            case PType::I64: for (size_t j = 0; j < k; ++j) ((int64_t*)d)[j] = iv[c][idx[j]]; break;
            default: for (size_t j = 0; j < k; ++j) ((double*)d)[j] = dv[c][idx[j]]; break;
          }
          return;
        }
        auto src = [&](const Loc& l) -> const uint8_t* {  // (write buffers: after the resize above)
          return l.seg >= 0 ? segCols.at(l.seg)[c] : bufs[-1 - l.seg]->cols[c].data();
        };
        auto copy = [&](auto* out) {  // (typed by width: a load and a store per cell)
          using T = std::remove_pointer_t<decltype(out)>;
          const T* base = nullptr;
          int32_t baseSeg = INT32_MIN;
          for (size_t j = 0; j < k; ++j) {
            const Loc l = ls[idx[j]];
            if (l.seg != baseSeg) base = (const T*)src(l), baseSeg = l.seg;
            out[j] = base[l.row];
          }
        };
        if (w == 4) copy((int32_t*)d);
        else if (w == 8) copy((int64_t*)d);
        else {  // (other widths: VECTOR columns don't come here; kept general)
          const uint8_t* base = nullptr;
          int32_t baseSeg = INT32_MIN;
          for (size_t j = 0; j < k; ++j) {
            const Loc l = ls[idx[j]];
            if (l.seg != baseSeg) base = src(l), baseSeg = l.seg;
            memcpy(d + j * w, base + (size_t)l.row * w, w);
          }
        }
      }, true);
      buf.del->host.resize((size_t)(buf.n + (int64_t)k + 31) / 32, 0);
      buf.n += (int64_t)k;
      ++buf.version;
      done += k;
      if (buf.n >= kSegRows && !deferFlush) flushPart(p);
    }
  }
}

void Table::appendRow(const std::vector<Value>& rowIn) {
  std::vector<int64_t> ints(cols.size());
  std::vector<double> dbls(cols.size());
  std::vector<Value> row = rowIn;
  if (autoIncCol >= 0) {
    Value& v = row[autoIncCol];
    if (v.isNull() || (v.kind == Value::INT && v.i == 0)) v = Value::I(autoNext++);
    else autoNext = std::max<int64_t>(autoNext, (int64_t)v.num() + 1);
    lastInsertId = (int64_t)row[autoIncCol].num();
  }
  struct UndoGuard {  // record the final row (after AUTO_INCREMENT) once it is stored
    Table* t;
    const std::vector<Value>& row;
    bool ok = false;
    ~UndoGuard() {  // (temporary tables of a query are not part of the transaction)
      if (ok && undoLog() && t->db != "__tmp") undoLog()->push_back({t, true, row});
    }
  } undo{this, row};
  for (size_t c = 0; c < cols.size(); ++c) rawOf((int)c, row[c], ints[c], dbls[c]);
  appendRaw(ints, dbls);
  undo.ok = true;
}

TxnPending& Table::pendingOfMine() {
  auto [it, fresh] = txnPending.try_emplace(txnKey());
  if (fresh) tTxnEntries.push_back({this, &it->second});
  return it->second;
}

void Table::gcPending() {
  if (txnPending.empty()) return;
  const uint64_t oldest = oldestSnapshotSeq();
  for (auto it = txnPending.begin(); it != txnPending.end();) {
    const uint64_t c = it->second.commitSeq;
    it = c && (oldest == UINT64_MAX || c <= oldest) ? txnPending.erase(it) : std::next(it);
  }
}

void Table::txnNoteDelete(Loc l) {
  auto& tp = pendingOfMine();
  if (tp.hiddenIx.erase(TxnPending::key(l))) {  // a row this transaction inserted: just gone
    auto h = std::find(tp.hidden.begin(), tp.hidden.end(), l);
    if (h != tp.hidden.end()) tp.hidden.erase(h);
  } else {
    tp.ghosts.push_back(l);  // a committed row: others still see it until COMMIT
  }
}

void Table::undoAppendsSince(const std::vector<int64_t>& mark) {
  for (int p = 0; p < nparts && p < (int)mark.size(); ++p)
    for (int64_t r = mark[p]; r < bufs[p]->n; ++r) {
      if (bufs[p]->del->test(r)) continue;
      const Loc l = bufLoc(p, r);
      if (pk >= 0)
        if (auto it = pkIndex.find(pkValue(l)); it != pkIndex.end() && it->second == l) pkIndex.erase(it);
      if (!indexes.empty()) indexRemove(l);
      WriteBuffer& buf = *bufs[p];
      delSet(delForWrite(buf.del), r);
      ++buf.version;
    }
}

void Table::deleteAt(Loc l) {
  if (undoLog()) {
    undoLog()->push_back({this, false, rowValues(l)});
    txnNoteDelete(l);
  }
  if (pk >= 0) pkIndex.erase(pkValue(l));
  if (!indexes.empty()) indexRemove(l);
  if (l.seg < 0) {
    WriteBuffer& buf = *bufs[bufPart(l)];
    delSet(delForWrite(buf.del), l.row);
    ++buf.version;
  } else {
    delSet(delForWrite(segs[l.seg]->del), l.row);
  }
}

void delSet(DelBits& d, int64_t r) {
  uint32_t& w = d.host[r >> 5];
  const uint32_t bit = 1u << (r & 31);
  if (w & bit) return;
  w |= bit;
  ++d.ndel;
  std::lock_guard<std::mutex> g(d.mu);  // keep a current device copy current
  if (d.dev && d.devVer == d.ver && (r >> 5) < d.devWords) h2d(&d.dev[r >> 5], &w, 4);
}

void Table::flush(bool force) {
  (void)force;
  for (int p = 0; p < nparts; ++p) flushPart(p);
}

// GPU compression of one segment column (lossless; decoded on the fly by the kernels, see ColView):
// integer-like columns (ints, dates, dictionary codes, DECIMAL) as uint8 / uint16 / uint32 offsets
// from the minimum, DOUBLE columns with at most 255 distinct values as a uint8 dictionary. Returns
// the encoding (0: keep raw) and fills out with header + data.
int g_compress = -1;

// Doubles that are all exact decimals with few digits (prices, rates: 406.1, 0.25): stored as
// k = x * 10^s, frame-of-reference packed to 16 / 32 bits; decoding (base + code) / 10^s gives the
// same double back (checked here for every value). 0: not encodable this way.
static uint8_t encodeDecimalDoubles(const double* x, int64_t n, std::vector<uint8_t>& out) {
  for (int s = 0; s <= 4; ++s) {
    const double p = pow10i(s);
    long long mn = LLONG_MAX, mx = LLONG_MIN;
    bool ok = true;
    for (int64_t i = 0; i < n && ok; ++i) {
      if (x[i] != x[i]) continue;  // NULL
      const double y = x[i] * p;
      if (!(std::fabs(y) < 4e15)) { ok = false; break; }
      const long long k = std::llround(y);
      if ((double)k / p != x[i]) { ok = false; break; }
      mn = std::min(mn, k), mx = std::max(mx, k);
    }
    if (!ok) continue;
    if (mn > mx) mn = mx = 0;
    const unsigned long long range = (unsigned long long)(mx - mn);
    const uint8_t e = range < 0xFFFFull ? 5 : range < 0xFFFFFFFFull ? 6 : 0;
    if (!e) return 0;
    const int w = e == 5 ? 2 : 4;
    out.assign(kEncHeader + (size_t)n * w + 16, 0);
    memcpy(out.data(), &mn, 8);
    memcpy(out.data() + 8, &p, 8);
    uint8_t* d = out.data() + kEncHeader;
    for (int64_t i = 0; i < n; ++i) {
      const unsigned long long code = x[i] != x[i] ? (w == 2 ? 0xFFFFull : 0xFFFFFFFFull) : (unsigned long long)(std::llround(x[i] * p) - mn);
      if (w == 2) ((uint16_t*)d)[i] = (uint16_t)code;
      else ((uint32_t*)d)[i] = (uint32_t)code;
    }
    return e;
  }
  return 0;
}

bool Table::encodesNewColumns() const { return g_compress == 1 || (g_compress < 0 && gpuVramFill() >= 0.6); }

bool Table::gpuOnlyVersion(int32_t si, int c) const {
  const SegData& O = segs[si]->cd(c);
  return cols[c].pt != PType::VEC && !segCold(O, c) && (size_t)c < O.dev.size() && O.dev[c] && !O.mapped[c] && !O.enc[c] &&
         vramHint(c) != 2 && !encodesNewColumns() && segs[si]->full();
}

uint8_t Table::encodeColumn(int c, const std::vector<uint8_t>& h, int64_t n, std::vector<uint8_t>& out) const {
  // --compress: always; by default only once VRAM is filling up (compression costs ~10% scan speed
  // but multiplies how much fits on the GPU before segments spill to the slower host tier).
  if (n < 1024) return 0;
  if (g_compress == 0 || (g_compress < 0 && gpuVramFill() < 0.6)) return 0;
  const PType pt = cols[c].pt;
  if (pt == PType::I32 || pt == PType::I64) {
    const bool i32 = pt == PType::I32;
    const long long nul = i32 ? (long long)kNullI32 : (long long)kNullI64;
    long long mn = LLONG_MAX, mx = LLONG_MIN;
    for (int64_t i = 0; i < n; ++i) {
      const long long v = i32 ? (long long)((const int32_t*)h.data())[i] : (long long)((const int64_t*)h.data())[i];
      if (v == nul) continue;
      mn = std::min(mn, v);
      mx = std::max(mx, v);
    }
    if (mn > mx) mn = mx = 0;  // all NULL
    const unsigned long long range = (unsigned long long)mx - (unsigned long long)mn;
    const uint8_t e = range < 0xFFull ? 1 : range < 0xFFFFull ? 2 : (!i32 && range < 0xFFFFFFFFull) ? 3 : 0;
    if (!e) return 0;
    const int w = encWidth(e);
    out.assign(kEncHeader + (size_t)n * w + 16, 0);
    memcpy(out.data(), &mn, 8);
    uint8_t* d = out.data() + kEncHeader;
    for (int64_t i = 0; i < n; ++i) {
      const long long v = i32 ? (long long)((const int32_t*)h.data())[i] : (long long)((const int64_t*)h.data())[i];
      const unsigned long long code = v == nul ? (w == 1 ? 0xFFull : w == 2 ? 0xFFFFull : 0xFFFFFFFFull) : (unsigned long long)(v - mn);
      if (w == 1) d[i] = (uint8_t)code;
      else if (w == 2) ((uint16_t*)d)[i] = (uint16_t)code;
      else ((uint32_t*)d)[i] = (uint32_t)code;
    }
    return e;
  }
  if (pt == PType::F64) {
    std::unordered_map<uint64_t, uint8_t> dict;
    std::vector<double> vals;
    const double* x = (const double*)h.data();
    for (int64_t i = 0; i < n; ++i) {
      uint64_t bits;
      memcpy(&bits, &x[i], 8);
      if (dict.count(bits)) continue;
      if (dict.size() == 255) return encodeDecimalDoubles(x, n, out);
      dict.emplace(bits, (uint8_t)dict.size());
      vals.push_back(x[i]);
    }
    out.assign(kEncHeader + 256 * 8 + (size_t)n + 16, 0);
    memcpy(out.data() + kEncHeader, vals.data(), vals.size() * 8);
    uint8_t* d = out.data() + kEncHeader + 256 * 8;
    for (int64_t i = 0; i < n; ++i) {
      uint64_t bits;
      memcpy(&bits, &x[i], 8);
      d[i] = dict[bits];
    }
    return 4;
  }
  return 0;
}

uint64_t nextSegmentId() {
  static std::atomic<uint64_t> ids{0};
  return ++ids;
}

// Segment elimination metadata of a column (DECIMAL in real units); true if it has a NULL.
bool Table::colStats(int c, const std::vector<uint8_t>& h, int64_t n, double& mnOut, double& mxOut) const {
  double mn = INFINITY, mx = -INFINITY;
  bool nul = false;
  const double unscale = cols[c].lt == LType::DECIMAL ? pow10i(cols[c].scale) : 1.0;
  for (int64_t i = 0; i < n; ++i) {
    double v = readNum(h, cols[c].pt, i) / unscale;
    if (v != v) {
      nul = true;
      continue;
    }
    mn = std::min(mn, v);
    mx = std::max(mx, v);
  }
  mnOut = mn;
  mxOut = mx;
  return nul;
}

void Table::replaceColumns(const std::vector<int32_t>& sis, const std::vector<int>& setCols, std::vector<ColPatch>& patches) {
  const size_t nc = cols.size(), S = sis.size(), K = setCols.size();
  std::vector<std::shared_ptr<SegData>> nd(S);
  for (auto& D : nd) {
    D = std::make_shared<SegData>();
    D->host.resize(nc);
    D->dev.assign(nc, nullptr);
    D->mapped.assign(nc, 0);
    D->uses.assign(nc, 0);
    D->heat.assign(nc, 0.f);
    D->mn.assign(nc, INFINITY);
    D->mx.assign(nc, -INFINITY);
    D->enc.assign(nc, 0);
    D->encHost.resize(nc);
    D->lazy.assign(nc, 0);
  }
  std::vector<uint8_t> nul(S * K, 0);
  std::vector<int64_t> placed(S * K, 0);
  ioParallel(S * K, [&](size_t i) {  // stats, encoding and the device copy: a task per segment and column
    const size_t s = i / K, k = i % K;
    const int c = setCols[k];
    SegData& D = *nd[s];
    const Segment& sg = *segs[sis[s]];
    ColPatch& p = patches[s];
    const SegData& O = sg.cd(c);
    if (k < p.dev.size() && p.dev[k]) {  // built on the GPU already (UPDATE's fused path)
      const double unscale = cols[c].lt == LType::DECIMAL ? pow10i(cols[c].scale) : 1.0;
      D.dev[c] = p.dev[k];
      D.lazy[c] = (uint64_t)sg.n * colBytes(c);
      D.mn[c] = std::min(p.dmn[k] / unscale, O.mn[c]);
      D.mx[c] = std::max(p.dmx[k] / unscale, O.mx[c]);
      nul[i] = p.dnul[k];
      placed[i] = (int64_t)D.lazy[c];
      return;
    }
    if (p.cols[k].empty() && sg.n > 0) {  // GPU-only (gpuOnlyVersion): the VRAM copy patched, the host mirror later
      double mn, mx;
      nul[i] = colStats(c, p.vals[k], (int64_t)p.rows.size(), mn, mx);
      const size_t bytes = (size_t)sg.n * colBytes(c);
      D.dev[c] = gpuPatchColumn(O.dev[c], bytes, colBytes(c), p.rows.data(), p.vals[k].data(), (int64_t)p.rows.size());
      if (D.dev[c]) {
        D.lazy[c] = bytes;
        D.mn[c] = std::min(mn, O.mn[c]);
        D.mx[c] = std::max(mx, O.mx[c]);
        placed[i] = (int64_t)bytes;
        return;
      }
      // (no VRAM for it: the host column after all, the old one with the new values written in)
      segEnsureHost(const_cast<SegData&>(O), c);
      p.cols[k] = O.host[c];
      const int w = colBytes(c);
      for (size_t j = 0; j < p.rows.size(); ++j) memcpy(p.cols[k].data() + (size_t)p.rows[j] * w, p.vals[k].data() + j * w, w);
    }
    D.host[c] = std::move(p.cols[k]);
    D.enc[c] = encodeColumn(c, D.host[c], sg.n, D.encHost[c]);
    // The old version raw in VRAM: copied there with just the changed rows uploaded, and the segment
    // elimination range widened by the new values (exact again when the segment is rebuilt).
    if (!D.enc[c] && !O.enc[c] && !segCold(O, c) && O.dev[c] && !O.mapped[c] && vramHint(c) != 2) {
      double mn, mx;
      nul[i] = colStats(c, p.vals[k], (int64_t)p.rows.size(), mn, mx);
      D.dev[c] = gpuPatchColumn(O.dev[c], D.host[c].size(), colBytes(c), p.rows.data(), p.vals[k].data(), (int64_t)p.rows.size());
      if (D.dev[c]) {
        D.mn[c] = std::min(mn, O.mn[c]);
        D.mx[c] = std::max(mx, O.mx[c]);
        placed[i] = (int64_t)D.host[c].size();
        return;
      }
    }
    nul[i] = colStats(c, D.host[c], sg.n, D.mn[c], D.mx[c]);
    bool m = false;
    std::vector<uint8_t>& pl = D.enc[c] ? D.encHost[c] : D.host[c];
    D.dev[c] = placeColumn(pl.data(), pl.size(), m, vramHint(c) == 2);
    D.mapped[c] = m;
    placed[i] = (int64_t)pl.size();
  }, true);
  for (size_t i = 0; i < S * K; ++i)
    if (nul[i]) hasNull[setCols[i % K]] = 1;
  std::vector<std::pair<std::shared_ptr<SegData>, int>> olds;  // the versions replaced
  for (size_t s = 0; s < S; ++s) {
    SegData& D = *nd[s];
    for (size_t k = 0; k < K; ++k) D.devBytes += placed[s * K + k];
    segAccount(D);
    const std::shared_ptr<Segment>& old = segs[sis[s]];
    auto ns = std::make_shared<Segment>(*old);  // (shares the data, the other versions and the deleted-row bitmap)
    ns->id = nextSegmentId();
    ns->ov.resize(nc);
    for (int c : setCols) {
      olds.push_back({old->cdp(c), c});
      ns->ov[c] = nd[s];
    }
    for (auto& ix : indexes)
      if (ix.cagra) {
        std::lock_guard<std::mutex> g(ix.cagra->mu);
        ix.cagra->graphs.erase(old->id);
      }
    segs[sis[s]] = std::move(ns);
  }
  // Snapshots of the table taken earlier still read the old versions: freed after them.
  rcuDefer([olds] {
    for (auto& [d, c] : olds) segDropColumn(*d, c);
  });
}

std::shared_ptr<Segment> Table::makeSegment(int part, size_t n, const std::function<const uint8_t*(int)>& src, const int32_t* idx) const {
  auto seg = std::make_shared<Segment>();
  seg->id = nextSegmentId();
  seg->part = part;
  seg->n = (int64_t)n;
  seg->d->host.resize(cols.size());
  seg->d->dev.resize(cols.size(), nullptr);
  seg->d->mapped.assign(cols.size(), 0);
  seg->d->uses.assign(cols.size(), 0);
  seg->d->heat.assign(cols.size(), 0.f);
  seg->d->mn.resize(cols.size());
  seg->d->mx.resize(cols.size());
  seg->d->enc.assign(cols.size(), 0);
  seg->d->encHost.resize(cols.size());
  ioParallel(cols.size(), [&](size_t c) {  // gather, stats and encoding: one column per task
    int sz = colBytes((int)c);
    auto& h = seg->d->host[c];
    h.resize(n * sz);
    const uint8_t* from = src((int)c);
    if (sz == 8) {  // (fixed sizes: the copies inline)
      for (size_t i = 0; i < n; ++i) memcpy(&h[i * 8], from + (size_t)idx[i] * 8, 8);
    } else if (sz == 4) {
      for (size_t i = 0; i < n; ++i) memcpy(&h[i * 4], from + (size_t)idx[i] * 4, 4);
    } else {
      for (size_t i = 0; i < n; ++i) memcpy(&h[i * sz], from + (size_t)idx[i] * sz, sz);
    }
    colStats((int)c, h, (int64_t)n, seg->d->mn[c], seg->d->mx[c]);
    seg->d->enc[c] = encodeColumn((int)c, h, (int64_t)n, seg->d->encHost[c]);
  }, true);
  for (size_t c = 0; c < cols.size(); ++c) {
    bool m = false;
    std::vector<uint8_t>& placed = seg->d->enc[c] ? seg->d->encHost[c] : seg->d->host[c];
    seg->d->dev[c] = placeColumn(placed.data(), placed.size(), m, vramHint((int)c) == 2);
    seg->d->mapped[c] = m;
    seg->d->devBytes += (int64_t)placed.size();
  }
  seg->del->host.assign((n + 31) / 32, 0);
  segAccount(*seg->d);
  return seg;
}

// ---- Compaction. Deletes and updates leave dead rows in segments, and small segments (flushed
// by snapshots and OPTIMIZE) multiply. A partition's segments that are a quarter dead, or small, are
// merged into new sorted segments: picked under the shared lock (with a copy of their deleted-row
// bits), built without any lock, swapped in under the exclusive lock (rows deleted meanwhile are
// deleted in the new segments too; keys and indexes are pointed at the new places).
std::vector<CompactJob> Table::compactPlan() const {
  std::vector<CompactJob> jobs;
  const int64_t small = kSegRows / 4;
  for (int p = 0; p < nparts; ++p) {
    std::vector<std::shared_ptr<Segment>> cand;
    bool dead = false;
    int64_t live = 0;
    for (auto& sp : segs) {
      const Segment& sg = *sp;
      if (sg.part != p || sg.n == 0 || !sg.full()) continue;
      const int64_t nd = sg.ndel();
      const bool d = nd * 4 >= sg.n;
      if (!d && sg.n - nd >= small) continue;
      if (live + (sg.n - nd) > 4 * kSegRows) break;
      dead = dead || d;
      live += sg.n - nd;
      cand.push_back(sp);
    }
    if (cand.empty() || (!dead && cand.size() < 2)) continue;
    CompactJob j;
    j.part = p;
    j.olds = cand;
    for (auto& sp : cand) j.bits.push_back(sp->del->host);
    jobs.push_back(std::move(j));
  }
  return jobs;
}

void Table::compactBuild(CompactJob& j) const {
  const size_t nc = cols.size(), no = j.olds.size();
  std::vector<std::pair<int32_t, int32_t>> origin;  // gathered row -> (old segment, row)
  for (size_t k = 0; k < no; ++k) {
    const auto& bits = j.bits[k];
    for (int64_t r = 0; r < j.olds[k]->n; ++r)
      if (!((size_t)(r >> 5) < bits.size() && ((bits[r >> 5] >> (r & 31)) & 1u))) origin.push_back({(int32_t)k, (int32_t)r});
  }
  const size_t L = origin.size();
  std::vector<std::vector<uint8_t>> tmp(nc);  // the live rows, gathered
  ioParallel(nc, [&](size_t c) {
    const int sz = colBytes((int)c);
    tmp[c].resize(L * sz);
    std::vector<uint8_t> scratch;
    size_t g = 0;
    for (size_t k = 0; k < no; ++k) {
      const uint8_t* raw = segRawColumn(j.olds[k]->cd(c), (int)c, scratch);  // (cold: read from its file)
      for (; g < L && origin[g].first == (int32_t)k; ++g) memcpy(&tmp[c][g * sz], raw + (size_t)origin[g].second * sz, sz);
    }
  });
  std::vector<int32_t> order(L);
  for (size_t i = 0; i < L; ++i) order[i] = (int32_t)i;
  if (sortKey >= 0 && L > 1) order = gpuSortPermHost(tmp[sortKey].data(), cols[sortKey].pt, (int64_t)L);
  j.where.assign(no, {});
  for (size_t k = 0; k < no; ++k) j.where[k].assign((size_t)j.olds[k]->n, Loc{-1, -1});
  for (size_t start = 0; start < L; start += kSegRows) {
    const size_t n = std::min<size_t>(kSegRows, L - start);
    j.news.push_back(makeSegment(j.part, n, [&](int c) { return tmp[c].data(); }, order.data() + start));
    for (size_t i = 0; i < n; ++i) {
      const auto& o = origin[order[start + i]];
      j.where[o.first][o.second] = Loc{(int32_t)j.news.size() - 1, (int32_t)i};
    }
  }
}

bool Table::compactInstall(CompactJob& j) {
  gcPending();
  if (!txnPending.empty()) return false;  // open transactions refer to rows by location
  std::vector<int> oldAt(segs.size(), -1);  // segs index -> old segment number
  for (size_t k = 0; k < j.olds.size(); ++k) {
    size_t i = 0;
    while (i < segs.size() && segs[i] != j.olds[k]) ++i;
    if (i == segs.size()) return false;
    oldAt[i] = (int)k;
  }
  for (size_t k = 0; k < j.olds.size(); ++k) {  // rows deleted since the plan
    const auto& now = j.olds[k]->del->host;
    for (size_t w = 0; w < now.size(); ++w) {
      uint32_t diff = now[w] & ~(w < j.bits[k].size() ? j.bits[k][w] : 0u);
      for (; diff; diff &= diff - 1) {
        const int64_t r = (int64_t)w * 32 + __builtin_ctz(diff);
        if (r >= j.olds[k]->n) break;
        const Loc nl = j.where[k][r];
        if (nl.seg < 0) continue;
        DelBits& d = *j.news[nl.seg]->del;
        d.host[nl.row >> 5] |= 1u << (nl.row & 31);
        ++d.ndel;
      }
    }
  }
  std::vector<std::shared_ptr<Segment>> ns;
  std::vector<int32_t> remap(segs.size(), -1);
  for (size_t i = 0; i < segs.size(); ++i)
    if (oldAt[i] < 0) remap[i] = (int32_t)ns.size(), ns.push_back(segs[i]);
  const int32_t base = (int32_t)ns.size();
  for (auto& sp : j.news) ns.push_back(sp);
  auto move = [&](Loc& l) {
    if (l.seg < 0) return true;
    if (remap[l.seg] >= 0) {
      l.seg = remap[l.seg];
      return true;
    }
    const Loc nl = j.where[oldAt[l.seg]][l.row];
    if (nl.seg < 0) return false;
    l = Loc{base + nl.seg, nl.row};
    return true;
  };
  for (auto it = pkIndex.begin(); it != pkIndex.end();) it = move(it->second) ? std::next(it) : pkIndex.erase(it);
  for (auto& ix : indexes) {
    for (auto& [k, v] : ix.map) {
      size_t o = 0;
      for (size_t i = 0; i < v.size(); ++i)
        if (move(v[i])) v[o++] = v[i];
      v.resize(o);
    }
    if (ix.cagra) {  // graphs of the merged segments go (the new ones are built when searched)
      std::lock_guard<std::mutex> g(ix.cagra->mu);
      for (auto& sp : j.olds) ix.cagra->graphs.erase(sp->id);
    }
  }
  segs = std::move(ns);
  ++flushes;
  return true;
}

void Table::flushPart(int part) {
  const std::shared_ptr<WriteBuffer> hold = bufs[part];  // (replaced at the end; snapshots keep it)
  WriteBuffer& buf = *hold;
  if (buf.n == 0) return;
  gcPending();
  if (!txnPending.empty()) return;  // open transactions refer to buffer rows by location: flush later
  // Live rows in buffer order, then sorted by the SORT KEY (on the GPU).
  std::vector<int32_t> live;
  live.reserve(buf.n - buf.ndel());
  for (int64_t r = 0; r < buf.n; ++r)
    if (!buf.del->test(r)) live.push_back((int32_t)r);
  if (sortKey >= 0 && live.size() > 1) {
    const auto& kc = buf.cols[sortKey];
    int sz = psize(cols[sortKey].pt);
    std::vector<uint8_t> keys(live.size() * sz);
    for (size_t i = 0; i < live.size(); ++i) memcpy(&keys[i * sz], &kc[(size_t)live[i] * sz], sz);
    std::vector<int32_t> perm = gpuSortPermHost(keys.data(), cols[sortKey].pt, (int64_t)live.size());
    std::vector<int32_t> sorted(live.size());
    for (size_t i = 0; i < live.size(); ++i) sorted[i] = live[perm[i]];
    live.swap(sorted);
  }
  for (size_t start = 0; start < live.size(); start += kSegRows) {
    size_t n = std::min<size_t>(kSegRows, live.size() - start);
    std::shared_ptr<Segment> seg = makeSegment(part, n, [&](int c) { return buf.cols[c].data(); }, live.data() + start);
    int32_t segId = (int32_t)segs.size();
    if (pk >= 0)
      for (size_t i = 0; i < n; ++i) pkIndex[readInt(seg->d->host[pk], cols[pk].pt, (int64_t)i)] = Loc{segId, (int32_t)i};
    segs.push_back(std::move(seg));
    for (auto& ix : indexes)  // the rows moved from the write buffer into the new segment
      for (size_t i = 0; i < n; ++i) {
        const Loc nl{segId, (int32_t)i}, old = bufLoc(part, live[start + i]);
        int64_t k;
        if (!ixKey(ix, nl, k)) continue;
        auto it = ix.map.find(k);
        if (it == ix.map.end()) continue;
        for (auto& l : it->second)
          if (l == old) { l = nl; break; }
      }
  }
  // A fresh buffer replaces the flushed one (snapshots still reading the old one keep it).
  auto fresh = std::make_shared<WriteBuffer>();
  fresh->cols.resize(cols.size());
  fresh->gen = buf.gen + 1;
  bufs[part] = fresh;
  ++flushes;
  for (auto& ix : indexes)  // vector indexes train once there is data, and retrain as it grows 4x
    if (ix.vec) {
      int64_t n = liveRows();
      if (n >= 256 && (ix.cent.empty() || n > 4 * ix.trainedRows)) buildIndex(ix);
    }
}

void Table::syncBufferToDevice(int part) {
  WriteBuffer& buf = *bufs[part];
  if (buf.devRef && buf.devVersion == buf.version) return;
  BufData& bd = *buf.data;
  std::shared_ptr<DevMirror> m;
  {
    std::lock_guard<std::mutex> g(bd.devMu);
    m = bd.mirror;
    if (!m || m->gen != buf.gen || m->cap < buf.n) {  // (re)build: a new mirror, so snapshots keep theirs
      auto nm = std::make_shared<DevMirror>();
      nm->gen = buf.gen;
      nm->cap = std::max<int64_t>(buf.n, std::min<int64_t>(kSegRows, std::max<int64_t>(4096, (m ? m->cap : 0) * 2)));
      nm->cols.resize(cols.size());
      for (size_t c = 0; c < cols.size(); ++c) nm->cols[c] = tmpAlloc((size_t)nm->cap * colBytes((int)c));
      if (m && m->gen == buf.gen && m->rows > 0) {
        nm->rows = std::min(m->rows, buf.n);
        for (size_t c = 0; c < cols.size(); ++c) d2d(nm->cols[c], m->cols[c], (size_t)nm->rows * colBytes((int)c));
      }
      bd.mirror = m = nm;
    }
    if (m->rows < buf.n) {  // only appended rows are new
      for (size_t c = 0; c < cols.size(); ++c) {
        const int64_t w = colBytes((int)c);
        h2d((char*)m->cols[c] + m->rows * w, bd.cols[c].data() + m->rows * w, (size_t)(buf.n - m->rows) * w);
      }
      m->rows = buf.n;
    }
  }
  buf.dev = m->cols;
  buf.devRef = m;
  buf.delDev = buf.del->device((buf.n + 31) / 32);
  buf.devVersion = buf.version;
}

std::shared_ptr<Table> Table::shadow() const {
  auto s = std::make_shared<Table>(name, cols, pk, sortKey, pkCols, shardKey, nparts, reference);
  s->base = this;
  s->baseFlushes = flushes;
  s->db = db;
  s->txnPending = txnPending;
  s->fks = fks;
  s->fulltextCols = fulltextCols;
  s->dicts = dicts;
  s->vram = vram;
  s->hasNull = hasNull;
  s->autoIncCol = autoIncCol;
  s->autoNext = autoNext;
  s->flushes = flushes;
  s->indexes.reserve(indexes.size());
  for (auto& ix : indexes) {  // metadata (and the shared CAGRA graphs), not the lookup maps
    IndexMeta m;
    m.name = ix.name;
    m.cols = ix.cols;
    m.unique = ix.unique;
    m.vec = ix.vec;
    m.metric = ix.metric;
    m.nlist = ix.nlist;
    m.nprobe = ix.nprobe;
    m.dim = ix.dim;
    m.algo = ix.algo;
    m.graphDegree = ix.graphDegree;
    m.itopk = ix.itopk;
    m.cagra = ix.cagra;
    m.trainedRows = ix.trainedRows;
    m.cent = ix.cent;
    m.options = ix.options;
    s->indexes.push_back(std::move(m));
  }
  s->segs.clear();
  s->segs.reserve(segs.size());
  for (auto& sg : segs) s->segs.push_back(std::make_shared<Segment>(*sg));
  s->bufs.clear();
  for (auto& b : bufs) s->bufs.push_back(std::make_shared<WriteBuffer>(*b));
  return s;
}

bool Table::shadowHas(Loc l) const {
  if (!base) return true;
  if (l.seg >= 0) {
    if ((size_t)l.seg >= segs.size() || (size_t)l.seg >= base->segs.size() || segs[l.seg]->d != base->segs[l.seg]->d) return false;
    return l.row < segs[l.seg]->n && !segs[l.seg]->del->test(l.row);
  }
  const int p = bufPart(l);
  if (p >= (int)bufs.size() || bufs[p]->data != base->bufs[p]->data) return false;
  return l.row < bufs[p]->n && !bufs[p]->del->test(l.row);
}

// ---------------------------------------------------------------- LOAD DATA

// "-12.345" with scale 2 -> -1235 (exact digits, half-up rounding of the dropped digits)
bool parseDecimal(const char* b, const char* e, int scale, int64_t& out) {
  while (b < e && isspace((unsigned char)*b)) ++b;
  const char* b0 = b;
  bool neg = false;
  if (b < e && (*b == '-' || *b == '+')) neg = *b++ == '-';
  int64_t v = 0;
  int frac = -1;
  bool any = false, roundUp = false;
  for (; b < e; ++b) {
    if (*b == '.') { if (frac >= 0) return false; frac = 0; continue; }
    if (*b == 'e' || *b == 'E') {  // scientific notation: go through double
      out = std::llround(std::strtod(std::string(b0, e).c_str(), nullptr) * pow10i(scale));
      return true;
    }
    if (!isdigit((unsigned char)*b)) break;
    any = true;
    if (frac >= 0 && frac >= scale) {
      if (frac == scale && *b >= '5') roundUp = true;
      ++frac;
      continue;
    }
    v = v * 10 + (*b - '0');
    if (frac >= 0) ++frac;
  }
  if (!any) return false;
  for (int f = frac < 0 ? 0 : std::min(frac, scale); f < scale; ++f) v *= 10;
  if (roundUp) ++v;
  out = neg ? -v : v;
  return true;
}

static inline int64_t parseI(const char* b, const char* e) {
  bool neg = false;
  if (b < e && (*b == '-' || *b == '+')) neg = *b++ == '-';
  int64_t v = 0;
  for (; b < e && *b >= '0' && *b <= '9'; ++b) v = v * 10 + (*b - '0');
  return neg ? -v : v;
}

namespace {
// The bytes of a file to load: plain, or zstd frames (the WAL's compressed copies of loaded files).
struct LoadSource {
  FILE* f = nullptr;
  ZSTD_DCtx* z = nullptr;
  std::vector<char> in;
  ZSTD_inBuffer ib{nullptr, 0, 0};
  bool inEof = false;
  size_t lastRc = 0;
  explicit LoadSource(const std::string& path) {
    f = fopen(path.c_str(), "rb");
    if (!f) return;
    unsigned char m[4];
    if (fread(m, 1, 4, f) == 4 && m[0] == 0x28 && m[1] == 0xB5 && m[2] == 0x2F && m[3] == 0xFD) {
      z = ZSTD_createDCtx();
      in.resize(4 << 20);
    }
    rewind(f);
  }
  ~LoadSource() {
    if (f) fclose(f);
    if (z) ZSTD_freeDCtx(z);
  }
  size_t read(char* dst, size_t n) {
    if (!z) return fread(dst, 1, n, f);
    ZSTD_outBuffer ob{dst, n, 0};
    while (ob.pos < ob.size) {
      if (ib.pos == ib.size && !inEof) {
        ib.size = fread(in.data(), 1, in.size(), f);
        ib.src = in.data();
        ib.pos = 0;
        inEof = ib.size == 0;
      }
      const size_t before = ob.pos, inBefore = ib.pos;
      const size_t rc = ZSTD_decompressStream(z, &ob, &ib);
      if (ZSTD_isError(rc)) throw SqlError(std::string("corrupt compressed load file: ") + ZSTD_getErrorName(rc), 1105);
      if (ob.pos != before || ib.pos != inBefore) lastRc = rc;  // (0: the last frame is complete)
      else if (inEof) {
        if (lastRc != 0) throw SqlError("truncated compressed load file", 1105);
        break;
      }
    }
    return ob.pos;
  }
};

// One worker's share of a chunk: whole lines [b, e), parsed into columns. VARCHAR fields stay text
// (sv) until the dictionary pass codes them (in row order, one thread per column).
struct LoadPiece {
  const char *b = nullptr, *e = nullptr, *zb = nullptr;  // zb: the compressed copy covers [zb, e)
  int64_t rows = 0, lines = 0;
  std::vector<std::vector<int64_t>> iv;
  std::vector<std::vector<double>> dv;
  std::vector<std::vector<std::string_view>> sv;
  std::vector<std::vector<float>> vv;
  std::deque<std::string> own;  // unquoted / unescaped text the views point at
  std::vector<uint8_t> nulls;
  bool failed = false;
  std::string msg;  // "\x01" stands for the line number
  int code = 0;
  std::string z;
};

inline bool digits(const char* p, int n) {
  for (int i = 0; i < n; ++i)
    if ((unsigned)(p[i] - '0') > 9) return false;
  return true;
}
inline int num(const char* p, int n) {
  int v = 0;
  for (int i = 0; i < n; ++i) v = v * 10 + (p[i] - '0');
  return v;
}
}  // namespace

// Lines are parsed in parallel pieces of 64 MB chunks (the next chunk is read meanwhile); VARCHAR
// values are coded per column in row order, so dictionaries come out exactly as a serial load makes
// them. Plain tables (no primary key, index or open transaction) append whole columns at a time.
int64_t Table::loadFile(const std::string& path, char delim, int64_t skip, std::vector<int> srcOf,
                        const std::function<void()>& yield, char escape, const std::string& copyTo, int64_t* loaded) {
  const size_t nc = cols.size();
  if (srcOf.empty())
    for (size_t c = 0; c < nc; ++c) srcOf.push_back((int)c);
  int needFields = 0;
  for (int x : srcOf) needFields = std::max(needFields, x + 1);
  for (size_t c = 0; c < nc; ++c)
    if (srcOf[c] < 0 && (int)c != autoIncCol && (cols[c].notNull || (int)c == pk))
      throw SqlError("Column '" + cols[c].name + "' cannot be null", 1048);
  LoadSource src(path);
  if (!src.f) throw SqlError("File '" + path + "' not found", 29);
  FILE* out = nullptr;
  const std::string tmp = copyTo + ".tmp";
  if (!copyTo.empty() && !(out = fopen(tmp.c_str(), "wb"))) throw SqlError("cannot write " + copyTo, 1105);
  struct Cleanup {
    FILE*& f;
    const std::string& t;
    ~Cleanup() {
      if (f) fclose(f), unlink(t.c_str());
    }
  } cleanup{out, tmp};

  const int nth = std::clamp<int>((int)std::thread::hardware_concurrency(), 1, 16);
  struct Chunk {  // whole lines of the file, parsed in pieces
    std::vector<char> data;
    size_t end = 0;       // bytes [0, end) are whole lines
    int64_t skipped = 0;  // header lines skipped in it
    int np = 0;
    std::vector<LoadPiece> pieces;
  };
  const size_t CH = 64 << 20;
  Chunk chunks[2];
  for (auto& ch : chunks) {
    ch.data.resize(CH + 1);
    ch.pieces.resize(nth * 4);
    for (auto& pc : ch.pieces) pc.iv.resize(nc), pc.dv.resize(nc), pc.sv.resize(nc), pc.vv.resize(nc);
  }
  for (auto& b : bufs)
    if (b->data.use_count() == 1)  // (never under a snapshot)
      for (auto& c : b->cols) c.reserve(std::min<int64_t>(kSegRows, (kSegRows * 4) / nparts) * 8);
  const bool bulk = pk < 0 && indexes.empty() && !(txnKey() && db != "__tmp");
  const bool autoFromFile = autoIncCol >= 0 && srcOf[autoIncCol] >= 0;
  int64_t rows = 0, lineNo = 0, skipLeft = skip, yieldAt = 0x40000;
  std::vector<int64_t> ints(nc);
  std::vector<double> dbls(nc);
  std::vector<std::vector<uint32_t>> rowsOf(nparts);
  std::vector<int64_t> off;

  auto parse = [&](LoadPiece& pc) {
    pc.rows = pc.lines = 0;
    pc.failed = false;
    pc.own.clear();
    pc.nulls.assign(nc, 0);
    for (size_t c = 0; c < nc; ++c) pc.iv[c].clear(), pc.dv[c].clear(), pc.sv[c].clear(), pc.vv[c].clear();
    std::vector<std::pair<const char*, const char*>> fields(std::max<size_t>(nc, needFields) + 1);
    std::vector<float> vec;
    auto fail = [&](std::string m, int code) {
      pc.failed = true;
      pc.msg = std::move(m);
      pc.code = code;
    };
    try {
      for (const char* p = pc.b; p < pc.e;) {
        const char* le = (const char*)memchr(p, '\n', pc.e - p);
        const char* lineEnd = le > p && le[-1] == '\r' ? le - 1 : le;
        ++pc.lines;
        if (lineEnd == p) {
          p = le + 1;
          continue;
        }
        size_t nf = 0;
        const char* fs = p;
        if (!memchr(p, '"', lineEnd - p)) {  // no quotes: split with memchr
          for (;;) {
            const char* q = (const char*)memchr(fs, delim, lineEnd - fs);
            if (!q) q = lineEnd;
            if (nf < fields.size()) fields[nf] = {fs, q};
            ++nf;
            if (q == lineEnd) break;
            fs = q + 1;
          }
        } else {
          bool inQ = false;  // delimiters inside "quoted" fields don't split (CSV)
          for (const char* q = p;; ++q) {
            if (q != lineEnd && *q == '"' && (q == fs || inQ)) {
              inQ = !inQ;
              continue;
            }
            if (q == lineEnd || (*q == delim && !inQ)) {
              if (nf < fields.size()) fields[nf] = {fs, q};
              ++nf;
              fs = q + 1;
              if (q == lineEnd) break;
            }
          }
        }
        if ((int)nf < needFields) return fail("Row \x01 doesn't contain data for all columns", 1261);
        for (size_t c = 0; c < nc; ++c) {
          const ColumnMeta& cm = cols[c];
          auto null = [&] {
            if (cm.lt == LType::VARCHAR) pc.sv[c].push_back({});
            else if (cm.pt == PType::F64) pc.dv[c].push_back(nullv());
            else if (cm.pt == PType::VEC) pc.vv[c].insert(pc.vv[c].end(), cm.vecDim, nullv());
            else pc.iv[c].push_back(cm.pt == PType::I64 ? kNullI64 : kNullI32);
            pc.nulls[c] = 1;
          };
          if (srcOf[c] < 0) {  // not in the file: AUTO_INCREMENT (numbered when applied) or NULL
            if ((int)c == autoIncCol) pc.iv[c].push_back(0);
            else null();
            continue;
          }
          auto [b, e] = fields[srcOf[c]];
          if (e - b >= 2 && *b == '"' && e[-1] == '"') {
            ++b, --e;
            if (std::string_view(b, e - b).find("\"\"") != std::string_view::npos) {  // "" inside quotes = "
              std::string& u = pc.own.emplace_back();
              for (const char* x = b; x < e; ++x) {
                u += *x;
                if (*x == '"' && x + 1 < e && x[1] == '"') ++x;
              }
              b = u.data();
              e = u.data() + u.size();
            }
          }
          if (e - b == 2 && b[0] == '\\' && b[1] == 'N') {  // \N = NULL (MySQL LOAD DATA convention)
            if (cm.notNull || (int)c == pk) return fail("Column '" + cm.name + "' cannot be null (row \x01)", 1048);
            null();
            continue;
          }
          if (escape && std::memchr(b, escape, e - b)) {  // ESCAPED BY: \t \n \r \0 \Z, and the escape itself
            std::string& u = pc.own.emplace_back();
            for (const char* x = b; x < e; ++x) {
              if (*x != escape || x + 1 >= e) {
                u += *x;
                continue;
              }
              const char n = *++x;
              u += n == 't' ? '\t' : n == 'n' ? '\n' : n == 'r' ? '\r' : n == '0' ? '\0' : n == 'Z' ? '\x1a' : n;
            }
            b = u.data();
            e = u.data() + u.size();
          }
          switch (cm.lt) {
            case LType::INT:
            case LType::BIGINT: pc.iv[c].push_back(parseI(b, e)); break;
            case LType::DECIMAL: {
              int64_t v;
              if (!parseDecimal(b, e, cm.scale, v)) return fail("Incorrect decimal value: '" + std::string(b, e) + "' on row \x01", 1366);
              pc.iv[c].push_back(v);
              break;
            }
            case LType::DOUBLE: {
              double v = 0;
              std::from_chars(b, e, v);
              pc.dv[c].push_back(v);
              break;
            }
            case LType::DATE: {
              int32_t d = 0;
              if (e - b == 10 && b[4] == '-' && b[7] == '-')
                d = (int32_t)daysFromCivil(parseI(b, b + 4), (unsigned)parseI(b + 5, b + 7), (unsigned)parseI(b + 8, b + 10));
              else if (!parseDate(std::string(b, e), d)) return fail("Incorrect date value: '" + std::string(b, e) + "'", 1292);
              pc.iv[c].push_back(d);
              break;
            }
            case LType::VARCHAR: pc.sv[c].push_back(std::string_view(b, e - b)); break;
            case LType::VECTOR:
              if (!parseVector(std::string(b, e), vec) || (int)vec.size() != cm.vecDim)
                return fail("Invalid VECTOR value on row \x01", 1105);
              pc.vv[c].insert(pc.vv[c].end(), vec.begin(), vec.end());
              break;
            case LType::DATETIME: {
              int64_t us = 0;
              if (e - b == 19 && b[4] == '-' && b[7] == '-' && b[10] == ' ' && b[13] == ':' && b[16] == ':' && digits(b, 4) &&
                  digits(b + 5, 2) && digits(b + 8, 2) && digits(b + 11, 2) && digits(b + 14, 2) && digits(b + 17, 2) &&
                  num(b + 5, 2) >= 1 && num(b + 5, 2) <= 12 && num(b + 8, 2) >= 1 && num(b + 8, 2) <= 31)
                us = daysFromCivil(num(b, 4), (unsigned)num(b + 5, 2), (unsigned)num(b + 8, 2)) * kUsPerDay +
                     ((int64_t)num(b + 11, 2) * 3600 + num(b + 14, 2) * 60 + num(b + 17, 2)) * 1000000ll;
              else if (!parseDateTime(std::string(b, e), us)) return fail("Incorrect datetime value: '" + std::string(b, e) + "'", 1292);
              pc.iv[c].push_back(us);
              break;
            }
          }
        }
        ++pc.rows;
        p = le + 1;
      }
    } catch (std::exception& ex) {
      fail(ex.what(), 1105);
    }
  };

  std::string carry;  // the partial last line of the previous chunk
  bool eof = false;
  auto readChunk = [&](Chunk& ch) {
    if (eof && carry.empty()) return false;
    memcpy(ch.data.data(), carry.data(), carry.size());
    size_t have = carry.size();
    if (!eof) {
      have += src.read(ch.data.data() + have, CH - have);
      eof = have < CH;
    }
    if (have == 0) return false;
    if (eof && ch.data[have - 1] != '\n') ch.data[have++] = '\n';
    size_t end = have;
    while (end > 0 && ch.data[end - 1] != '\n') --end;
    if (end == 0) throw SqlError("line too long in " + path, 1105);
    carry.assign(ch.data.data() + end, have - end);
    ch.end = end;
    return true;
  };
  auto readAndParse = [&](Chunk& ch) {
    if (!readChunk(ch)) return false;
    const char* p = ch.data.data();
    const char* stop = p + ch.end;
    ch.skipped = 0;
    for (; skipLeft > 0 && p < stop; --skipLeft, ++ch.skipped) p = (const char*)memchr(p, '\n', stop - p) + 1;
    // Pieces: about equal, cut after a newline (at least one: it carries the copy of skipped lines).
    const size_t per = std::max<size_t>((size_t)(stop - p) / (nth * 4) + 1, 1 << 16);  // (short tasks: urgent work gets a thread soon)
    ch.np = 0;
    for (const char* b = p;;) {
      const char* e = std::min(stop, b + per);
      if (e < stop) e = (const char*)memchr(e - 1, '\n', stop - (e - 1)) + 1;
      LoadPiece& pc = ch.pieces[ch.np++];
      pc.b = b;
      pc.e = e;
      pc.zb = ch.np == 1 ? ch.data.data() : b;
      if ((b = e) >= stop) break;
    }
    ioParallel(ch.np, [&](size_t i) {
      LoadPiece& pc = ch.pieces[i];
      parse(pc);
      if (out) {
        thread_local ZSTD_CCtx* cc = [] {  // (frames with checksums: a damaged copy fails at replay, not loads other rows)
          ZSTD_CCtx* c = ZSTD_createCCtx();
          ZSTD_CCtx_setParameter(c, ZSTD_c_compressionLevel, 1);
          ZSTD_CCtx_setParameter(c, ZSTD_c_checksumFlag, 1);
          return c;
        }();
        pc.z.resize(ZSTD_compressBound(pc.e - pc.zb));
        const size_t k = ZSTD_compress2(cc, pc.z.data(), pc.z.size(), pc.zb, pc.e - pc.zb);
        pc.z.resize(ZSTD_isError(k) ? 0 : k);
        if (ZSTD_isError(k)) pc.failed = true, pc.msg = "cannot compress the load copy", pc.code = 1105;
      }
    });
    return true;
  };

  // Codes the strings, then appends the rows of pieces [0, use) (the last one may have failed: its
  // rows before the bad line are kept, as a serial load would).
  int written = 0;  // pieces of the current chunk in the copy
  auto finish = [&](Chunk& ch) {
    written = 0;
    int bad = -1;
    for (int i = 0; i < ch.np && bad < 0; ++i)
      if (ch.pieces[i].failed) bad = i;
    const int use = bad < 0 ? ch.np : bad + 1;
    std::vector<size_t> strCols;
    for (size_t c = 0; c < nc; ++c)
      if (cols[c].lt == LType::VARCHAR) strCols.push_back(c);
    // Existing strings are looked up in parallel (per piece and column); new ones are then added
    // per column in row order.
    ioParallel(strCols.size() * use, [&](size_t i) {
      const size_t c = strCols[i / use];
      LoadPiece& pc = ch.pieces[i % use];
      pc.iv[c].resize(pc.rows);
      dicts[c]->findMany(pc.sv[c].data(), (size_t)pc.rows, pc.iv[c].data(), kNullI32);
    }, true);
    ioParallel(strCols.size(), [&](size_t i) {
      const size_t c = strCols[i];
      for (int k = 0; k < use; ++k) {
        LoadPiece& pc = ch.pieces[k];
        dicts[c]->addMissing(pc.sv[c].data(), (size_t)pc.rows, pc.iv[c].data());
      }
    }, true);
    off.assign(1, 0);
    for (int k = 0; k < use; ++k) {
      LoadPiece& pc = ch.pieces[k];
      if (autoIncCol >= 0)
        for (int64_t r = 0; r < pc.rows; ++r) {
          int64_t& v = pc.iv[autoIncCol][r];
          if (!autoFromFile) v = autoNext++;
          else autoNext = std::max<int64_t>(autoNext, v + 1);
        }
      for (size_t c = 0; c < nc; ++c)
        if (pc.nulls[c]) hasNull[c] = 1;
      off.push_back(off.back() + pc.rows);
    }
    auto cell = [&](LoadPiece& pc, size_t c, int64_t r) {  // -> ints / dbls / curVec
      const ColumnMeta& cm = cols[c];
      if (cm.pt == PType::F64) dbls[c] = pc.dv[c][r];
      else if (cm.pt == PType::VEC) curVec[c].assign(pc.vv[c].begin() + r * cm.vecDim, pc.vv[c].begin() + (r + 1) * cm.vecDim);
      else ints[c] = pc.iv[c][r];
    };
    if (!bulk) {
      for (int k = 0; k < use; ++k)
        for (int64_t r = 0; r < ch.pieces[k].rows; ++r) {
          for (size_t c = 0; c < nc; ++c) cell(ch.pieces[k], c, r);
          appendRaw(ints, dbls);
          ++rows;
        }
    } else {
      for (auto& v : rowsOf) v.clear();
      if (nparts == 1 || shardKey.empty()) {  // no shard key: a contiguous slice per partition
        const int64_t R = off.back();
        for (int i = 0; i < nparts; ++i) {
          const int p = (int)((nextPart + i) % nparts);
          for (int64_t g = R * i / nparts; g < R * (i + 1) / nparts; ++g) rowsOf[p].push_back((uint32_t)g);
        }
        ++nextPart;
      } else {
        for (int k = 0; k < use; ++k)
          for (int64_t r = 0; r < ch.pieces[k].rows; ++r) {
            for (int c : shardKey) cell(ch.pieces[k], c, r);
            rowsOf[partOf(ints, dbls)].push_back((uint32_t)(off[k] + r));
          }
      }
      for (int p = 0; p < nparts; ++p) {
        const std::vector<uint32_t>& list = rowsOf[p];
        for (size_t done = 0; done < list.size();) {
          WriteBuffer& buf = *bufs[p];
          const size_t room = buf.n < kSegRows ? (size_t)(kSegRows - buf.n) : list.size() - done;
          const size_t k = std::min(list.size() - done, room);
          buf.reserveForAppend((int64_t)k);
          const uint32_t* idx = list.data() + done;
          ioParallel(nc, [&](size_t c) {  // one column per task; rows are in order, so pieces are walked
            std::vector<uint8_t>& dst = buf.cols[c];
            const ColumnMeta& cm = cols[c];
            const size_t o = dst.size(), w = cm.pt == PType::VEC ? 4 * (size_t)cm.vecDim : (size_t)psize(cm.pt);
            dst.resize(o + k * w);
            uint8_t* d = dst.data() + o;
            int pi = 0;
            for (size_t j = 0; j < k; ++j) {
              const int64_t g = idx[j];
              while (g >= off[pi + 1]) ++pi;
              const LoadPiece& pc = ch.pieces[pi];
              const int64_t r = g - off[pi];
              switch (cm.pt) {
                case PType::I32: ((int32_t*)d)[j] = (int32_t)pc.iv[c][r]; break;
                case PType::I64: ((int64_t*)d)[j] = pc.iv[c][r]; break;
                case PType::F64: ((double*)d)[j] = pc.dv[c][r]; break;
                case PType::VEC: memcpy(d + j * w, pc.vv[c].data() + (size_t)r * cm.vecDim, w); break;
              }
            }
          }, true);
          buf.del->host.resize((size_t)(buf.n + (int64_t)k + 31) / 32, 0);
          buf.n += (int64_t)k;
          ++buf.version;
          rows += (int64_t)k;
          done += k;
          if (buf.n >= kSegRows) flushPart(p);
        }
      }
    }
    lineNo += ch.skipped;
    for (int k = 0; k < use; ++k) {
      LoadPiece& pc = ch.pieces[k];
      if (k == bad) {
        std::string m = pc.msg;
        const size_t at = m.find('\x01');
        if (at != std::string::npos) m.replace(at, 1, std::to_string(lineNo + pc.lines));
        throw SqlError(m, pc.code);
      }
      lineNo += pc.lines;
      if (out && !pc.z.empty() && fwrite(pc.z.data(), 1, pc.z.size(), out) != pc.z.size()) throw SqlError("cannot write " + copyTo, 1105);
      written = k + 1;
    }
    if (rows >= yieldAt && yield) {  // let readers in between chunks of a long load
      yield();
      yieldAt = rows + 0x40000;
    }
  };

  // Pipelined: the next chunk is read and parsed while this one is coded and appended.
  auto closeCopy = [&] {
    const bool ok = fflush(out) == 0 && fdatasync(fileno(out)) == 0;
    fclose(out);
    out = nullptr;
    if (!ok || rename(tmp.c_str(), copyTo.c_str()) != 0) {
      unlink(tmp.c_str());
      throw SqlError("cannot write " + copyTo, 1105);
    }
  };
  Chunk* cur = &chunks[0];
  Chunk* nxt = &chunks[1];
  bool started = false;
  try {
    for (bool more = readAndParse(*cur); more; std::swap(cur, nxt)) {
      std::future<bool> ahead = std::async(std::launch::async, [&, nxt] { return readAndParse(*nxt); });
      started = true;
      finish(*cur);
      more = ahead.get();
    }
  } catch (...) {
    // Rows loaded before the error stay; the copy then holds every line up to the failing chunk's
    // end, so the logged statement replays to the same rows and the same error.
    if (loaded) *loaded = rows;
    if (out && rows > 0) {
      if (started)
        for (int k = written; k < cur->np; ++k)
          if (!cur->pieces[k].z.empty()) fwrite(cur->pieces[k].z.data(), 1, cur->pieces[k].z.size(), out);
      try {
        closeCopy();
      } catch (...) {
      }
    }
    throw;
  }
  if (out) closeCopy();
  if (loaded) *loaded = rows;
  return rows;
}

}  // namespace gdb
