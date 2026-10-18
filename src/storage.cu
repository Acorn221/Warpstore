#include <thread>
#include "storage.h"

#include <algorithm>
#include <charconv>
#include <cstdio>
#include <cstring>
#include "gpu.h"
#include <climits>

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
        else if (m.lt == LType::VARCHAR) r.push_back(Value::S(dicts[c]->strs[v]));
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

SegData::~SegData() {
  for (size_t c = 0; c < dev.size(); ++c) {
    const bool e = c < enc.size() && enc[c];
    const std::vector<uint8_t>* src = e ? &encHost[c] : c < host.size() ? &host[c] : nullptr;
    releaseColumn(dev[c], src ? (void*)src->data() : nullptr, src ? src->size() : 0, c < mapped.size() && mapped[c]);
  }
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

Table::Table(std::string n, std::vector<ColumnMeta> c, int pk_, int sk, std::vector<int> pkc, std::vector<int> shard,
             int np, bool ref)
    : name(std::move(n)), cols(std::move(c)), pk(pk_), sortKey(sk), pkCols(std::move(pkc)), shardKey(std::move(shard)),
      nparts(ref ? 1 : std::max(np, 1)), reference(ref) {
  dicts.resize(cols.size());
  hasNull.assign(cols.size(), 0);
  curVec.resize(cols.size());
  for (size_t i = 0; i < cols.size(); ++i) {
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
  if (!sg.host[c].empty()) return;
  std::vector<uint8_t> h((size_t)sg.n * colBytes(c));
  d2h(h.data(), sg.dev[c], h.size());
  sg.host[c] = std::move(h);
}

int64_t Table::liveRows() const {
  int64_t r = bufferRows();
  for (auto& s : segs) r += s->n - s->ndel();
  return r;
}

int64_t Table::vramBytes() const {
  int64_t b = 0;
  for (auto& s : segs) {
    b += s->devBytes;
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
  if (buf.n >= kSegRows) flushPart(part);
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
  for (size_t c = 0; c < cols.size(); ++c) {
    const Value& v = row[c];
    if (v.isNull()) {
      if (cols[c].notNull || (int)c == pk) throw SqlError("Column '" + cols[c].name + "' cannot be null", 1048);
      hasNull[c] = 1;
      ints[c] = cols[c].pt == PType::I64 ? kNullI64 : kNullI32;
      dbls[c] = nullv();
      continue;
    }
    switch (cols[c].lt) {
      case LType::INT:
      case LType::BIGINT:
        ints[c] = v.kind == Value::STR ? std::stoll(v.s) : v.kind == Value::DBL ? (int64_t)std::llround(v.d) : v.i;
        break;
      case LType::DOUBLE: dbls[c] = v.kind == Value::STR ? std::stod(v.s) : v.num(); break;
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
        ints[c] = raw;
        break;
      }
      case LType::DATE: {
        if (v.kind == Value::STR) {
          int32_t d;
          if (!parseDate(v.s, d)) throw SqlError("Incorrect date value: '" + v.s + "'", 1292);
          ints[c] = d;
        } else if (v.kind == Value::DATETIME) {
          ints[c] = floorDiv(v.i, kUsPerDay);
        } else {
          ints[c] = v.i;
        }
        break;
      }
      case LType::DATETIME: {
        if (v.kind == Value::STR) {
          int64_t us;
          if (!parseDateTime(v.s, us)) throw SqlError("Incorrect datetime value: '" + v.s + "'", 1292);
          ints[c] = us;
        } else if (v.kind == Value::DATE) {
          ints[c] = v.i * kUsPerDay;
        } else {
          ints[c] = v.i;
        }
        break;
      }
      case LType::VARCHAR: ints[c] = dicts[c]->getOrAdd(v.kind == Value::STR ? v.s : v.toString()); break;
      case LType::VECTOR: {
        if (!parseVector(v.toString(), curVec[c]) || (int)curVec[c].size() != cols[c].vecDim)
          throw SqlError("Invalid VECTOR(" + std::to_string(cols[c].vecDim) + ") value for column '" + cols[c].name + "'", 1105);
        ints[c] = 0;
        break;
      }
    }
  }
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
    static std::atomic<uint64_t> segIds{0};
    auto seg = std::make_unique<Segment>();
    seg->id = ++segIds;
    seg->part = part;
    seg->n = (int64_t)n;
    seg->host.resize(cols.size());
    seg->dev.resize(cols.size(), nullptr);
    seg->mapped.assign(cols.size(), 0);
    seg->uses.assign(cols.size(), 0);
    seg->heat.assign(cols.size(), 0.f);
    seg->mn.resize(cols.size());
    seg->mx.resize(cols.size());
    seg->enc.assign(cols.size(), 0);
    seg->encHost.resize(cols.size());
    for (size_t c = 0; c < cols.size(); ++c) {
      int sz = colBytes((int)c);
      auto& h = seg->host[c];
      h.resize(n * sz);
      const uint8_t* src = buf.cols[c].data();
      for (size_t i = 0; i < n; ++i) memcpy(&h[i * sz], src + (size_t)live[start + i] * sz, sz);
      double mn = INFINITY, mx = -INFINITY;
      const double unscale = cols[c].lt == LType::DECIMAL ? pow10i(cols[c].scale) : 1.0;
      for (size_t i = 0; i < n; ++i) {
        double v = readNum(h, cols[c].pt, (int64_t)i) / unscale;  // DECIMAL stats in real units
        if (v != v) continue;  // NULL
        mn = std::min(mn, v);
        mx = std::max(mx, v);
      }
      seg->mn[c] = mn;
      seg->mx[c] = mx;
      bool m = false;
      uint8_t e = encodeColumn((int)c, h, (int64_t)n, seg->encHost[c]);
      seg->enc[c] = e;
      std::vector<uint8_t>& placed = e ? seg->encHost[c] : h;
      seg->dev[c] = placeColumn(placed.data(), placed.size(), m, vramHint((int)c) == 2);
      seg->mapped[c] = m;
      seg->devBytes += (int64_t)placed.size();
    }
    seg->del->host.assign((n + 31) / 32, 0);
    int32_t segId = (int32_t)segs.size();
    if (pk >= 0)
      for (size_t i = 0; i < n; ++i) pkIndex[readInt(seg->host[pk], cols[pk].pt, (int64_t)i)] = Loc{segId, (int32_t)i};
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

int64_t Table::loadFile(const std::string& path, char delim, int64_t skip, std::vector<int> srcOf,
                        const std::function<void()>& yield, char escape) {
  std::string unesc;
  if (srcOf.empty())
    for (size_t c = 0; c < cols.size(); ++c) srcOf.push_back((int)c);
  int needFields = 0;
  for (int x : srcOf) needFields = std::max(needFields, x + 1);
  std::string unq;
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) throw SqlError("File '" + path + "' not found", 29);
  const size_t CH = 64 << 20;
  std::vector<char> data(CH + 1);
  size_t have = 0;
  int64_t rows = 0, lineNo = 0;
  std::vector<int64_t> ints(cols.size());
  std::vector<double> dbls(cols.size());
  std::vector<std::pair<const char*, const char*>> fields(std::max<size_t>(cols.size(), needFields) + 1);
  for (auto& b : bufs)
    if (b->data.use_count() == 1)  // (never under a snapshot)
      for (auto& c : b->cols) c.reserve(std::min<int64_t>(kSegRows, (kSegRows * 4) / nparts) * 8);
  bool eof = false;
  while (!eof) {
    size_t got = fread(data.data() + have, 1, CH - have, f);
    if (got == 0) eof = true;
    have += got;
    if (have == 0) break;
    size_t end = have;
    if (!eof) {
      while (end > 0 && data[end - 1] != '\n') --end;
      if (end == 0) { fclose(f); throw SqlError("line too long in " + path, 1105); }
    } else if (data[have - 1] != '\n') {
      data[have++] = '\n';
      end = have;
    }
    const char* p = data.data();
    const char* stop = data.data() + end;
    while (p < stop) {
      const char* le = (const char*)memchr(p, '\n', stop - p);
      const char* lineEnd = le && le > p && le[-1] == '\r' ? le - 1 : le;
      if (lineNo++ < skip || lineEnd == p) { p = le + 1; continue; }
      size_t nf = 0;
      const char* fs = p;
      bool inQ = false;  // delimiters inside "quoted" fields don't split (CSV)
      for (const char* q = p;; ++q) {
        if (q != lineEnd && *q == '"' && (q == fs || inQ)) { inQ = !inQ; continue; }
        if (q == lineEnd || (*q == delim && !inQ)) {
          if (nf < fields.size()) fields[nf] = {fs, q};
          ++nf;
          fs = q + 1;
          if (q == lineEnd) break;
        }
      }
      if ((int)nf < needFields) {
        fclose(f);
        throw SqlError("Row " + std::to_string(lineNo) + " doesn't contain data for all columns", 1261);
      }
      for (size_t c = 0; c < cols.size(); ++c) {
        if (srcOf[c] < 0) {  // column not in the file: AUTO_INCREMENT or NULL
          if ((int)c == autoIncCol) {
            ints[c] = autoNext;
            continue;
          }
          if (cols[c].notNull || (int)c == pk) {
            fclose(f);
            throw SqlError("Column '" + cols[c].name + "' cannot be null", 1048);
          }
          hasNull[c] = 1;
          ints[c] = cols[c].pt == PType::I64 ? kNullI64 : kNullI32;
          dbls[c] = nullv();
          continue;
        }
        auto [b, e] = fields[srcOf[c]];
        if (e - b >= 2 && *b == '"' && e[-1] == '"') {
          ++b, --e;
          if (std::string_view(b, e - b).find("\"\"") != std::string_view::npos) {  // "" inside quotes = "
            unq.clear();
            for (const char* x = b; x < e; ++x) {
              unq += *x;
              if (*x == '"' && x + 1 < e && x[1] == '"') ++x;
            }
            b = unq.data();
            e = unq.data() + unq.size();
          }
        }
        if (e - b == 2 && b[0] == '\\' && b[1] == 'N') {  // \N = NULL (MySQL LOAD DATA convention)
          if (cols[c].notNull || (int)c == pk) {
            fclose(f);
            throw SqlError("Column '" + cols[c].name + "' cannot be null (row " + std::to_string(lineNo) + ")", 1048);
          }
          hasNull[c] = 1;
          ints[c] = cols[c].pt == PType::I64 ? kNullI64 : kNullI32;
          dbls[c] = nullv();
          continue;
        }
        if (escape && std::memchr(b, escape, e - b)) {  // ESCAPED BY: \t \n \r \0 \Z, and the escape itself
          unesc.clear();
          for (const char* x = b; x < e; ++x) {
            if (*x != escape || x + 1 >= e) { unesc += *x; continue; }
            const char n = *++x;
            unesc += n == 't' ? '\t' : n == 'n' ? '\n' : n == 'r' ? '\r' : n == '0' ? '\0' : n == 'Z' ? '\x1a' : n;
          }
          b = unesc.data();
          e = unesc.data() + unesc.size();
        }
        switch (cols[c].lt) {
          case LType::INT:
          case LType::BIGINT: ints[c] = parseI(b, e); break;
          case LType::DECIMAL:
            if (!parseDecimal(b, e, cols[c].scale, ints[c])) {
              fclose(f);
              throw SqlError("Incorrect decimal value: '" + std::string(b, e) + "' on row " + std::to_string(lineNo), 1366);
            }
            break;
          case LType::DOUBLE: {
            double v = 0;
            std::from_chars(b, e, v);
            dbls[c] = v;
            break;
          }
          case LType::DATE: {
            int32_t d = 0;
            if (e - b == 10 && b[4] == '-' && b[7] == '-')
              d = (int32_t)daysFromCivil(parseI(b, b + 4), (unsigned)parseI(b + 5, b + 7), (unsigned)parseI(b + 8, b + 10));
            else if (!parseDate(std::string(b, e), d)) {
              fclose(f);
              throw SqlError("Incorrect date value: '" + std::string(b, e) + "'", 1292);
            }
            ints[c] = d;
            break;
          }
          case LType::VARCHAR: ints[c] = dicts[c]->getOrAdd(std::string_view(b, e - b)); break;
          case LType::VECTOR:
            if (!parseVector(std::string(b, e), curVec[c]) || (int)curVec[c].size() != cols[c].vecDim) {
              fclose(f);
              throw SqlError("Invalid VECTOR value on row " + std::to_string(lineNo), 1105);
            }
            break;
          case LType::DATETIME: {
            int64_t us = 0;
            if (!parseDateTime(std::string(b, e), us)) {
              fclose(f);
              throw SqlError("Incorrect datetime value: '" + std::string(b, e) + "'", 1292);
            }
            ints[c] = us;
            break;
          }
        }
      }
      if (autoIncCol >= 0) autoNext = std::max<int64_t>(autoNext, ints[autoIncCol] + 1);
      appendRaw(ints, dbls);
      if ((++rows & 0x3ffff) == 0 && yield) yield();  // let readers in between chunks of a long load
      p = le + 1;
    }
    memmove(data.data(), data.data() + end, have - end);
    have -= end;
  }
  fclose(f);
  return rows;
}

}  // namespace gdb
