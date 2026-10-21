// Disk tier for segment column data (see SegFile in storage.h).
#include <dirent.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <atomic>
#include <cstring>
#include <condition_variable>
#include <deque>
#include <functional>
#include <map>
#include <thread>
#include <tuple>
#include <unordered_set>
#include "devtypes.h"
#include "gpu.h"
#include "storage.h"

namespace gdb {

namespace {
#define CK(x)                                                                                 \
  do {                                                                                        \
    cudaError_t e_ = (x);                                                                     \
    if (e_ != cudaSuccess) {                                                                  \
      if (e_ == cudaErrorMemoryAllocation) { cudaGetLastError(); throw SqlError("GPU out of memory", 1041); } \
      throw SqlError(std::string("CUDA error: ") + cudaGetErrorString(e_) + " at " #x, 1105); \
    }                                                                                         \
  } while (0)

std::string g_segDir;  // empty: no data dir, no files
int64_t g_memLimit = INT64_MAX;
std::atomic<int64_t> g_hostBytes{0}, g_pending{0};
std::atomic<uint64_t> g_nextId{1};
std::mutex g_regMu;
std::unordered_set<uint64_t> g_live, g_installed;  // files with an open SegFile / in the installed snapshot

constexpr char kMagic[8] = {'W', 'S', 'E', 'G', '0', '0', '0', '1'};
constexpr uint64_t kAlign = 4096;

std::string pathOf(uint64_t id) { return g_segDir + "/" + std::to_string(id) + ".wseg"; }

bool preadAll(int fd, void* buf, size_t len, uint64_t off) {
  char* p = (char*)buf;
  while (len > 0) {
    const ssize_t k = ::pread(fd, p, len, (off_t)off);
    if (k <= 0) return false;
    p += k, len -= (size_t)k, off += (uint64_t)k;
  }
  return true;
}
bool pwriteAll(int fd, const void* buf, size_t len, uint64_t off) {
  const char* p = (const char*)buf;
  while (len > 0) {
    const ssize_t k = ::pwrite(fd, p, len, (off_t)off);
    if (k <= 0) return false;
    p += k, len -= (size_t)k, off += (uint64_t)k;
  }
  return true;
}

void readCol(const SegFile& f, int c, void* dst) {
  const auto& col = f.cols[c];
  if (!preadAll(f.fd, dst, col.len, col.off)) throw SqlError("cannot read segment file " + pathOf(f.id), 1105);
}

// Raw column bytes from the stored ones (decodes a compressed column).
void decodeInto(const SegFile::Col& col, const uint8_t* stored, int64_t n, uint8_t* raw) {
  if (!col.enc) {
    memcpy(raw, stored, col.len);
    return;
  }
  ColView v{};
  v.p = stored;
  v.enc = col.enc;
  v.off = 0;
  if (col.enc <= 3) {
    v.t = col.width == 4 ? 0 : 1;
    for (int64_t r = 0; r < n; ++r) {
      const long long x = encInt(v, r);
      if (col.width == 4) ((int32_t*)raw)[r] = (int32_t)x;
      else ((int64_t*)raw)[r] = (int64_t)x;
    }
  } else {
    v.t = 2;
    for (int64_t r = 0; r < n; ++r) ((double*)raw)[r] = encDbl(v, r);
  }
}

int64_t colHostBytes(const SegData& d, int c) {
  return (int64_t)d.host[c].size() + (d.enc[c] ? (int64_t)d.encHost[c].size() : 0);
}

// Under d.loadMu, column cold: brings back an eviction whose frees are still deferred.
bool revive(SegData& d, int c) {
  if (!d.evicted[c].first) return false;
  d.dev[c] = d.evicted[c].first;
  d.mapped[c] = d.evicted[c].second;
  d.evicted[c] = {nullptr, 0};
  g_pending -= colHostBytes(d, c);
  ++d.gen[c];
  __atomic_store_n(&d.cold[c], (uint8_t)0, __ATOMIC_RELEASE);
  return true;
}

// Under d.loadMu, column cold and not pending: reads it into the cache (host mirror + device copy).
void loadLocked(SegData& d, int c, int64_t n) {
  const auto& col = d.file->cols[c];
  std::vector<uint8_t> stored(col.len);
  readCol(*d.file, c, stored.data());
  std::vector<uint8_t> raw;
  if (col.enc) {
    raw.resize((size_t)n * col.width);
    decodeInto(col, stored.data(), n, raw.data());
    d.encHost[c] = std::move(stored);
    d.host[c] = std::move(raw);
  } else {
    d.host[c] = std::move(stored);
  }
  std::vector<uint8_t>& placed = col.enc ? d.encHost[c] : d.host[c];
  bool m = false;
  d.dev[c] = placeColumn(placed.data(), placed.size(), m, false);
  d.mapped[c] = m;
  const int64_t b = colHostBytes(d, c);
  d.hostBytes += b;
  g_hostBytes += b;
  ++d.gen[c];
  __atomic_store_n(&d.cold[c], (uint8_t)0, __ATOMIC_RELEASE);
}

// This thread's per-batch column copies: device memory (kind 0) or mapped pinned host memory (1).
struct Temp {
  void* p;
  int kind;  // 0 device memory, 2 pinned buffer (back to the pool)
  size_t cap;
};
thread_local std::vector<Temp> tTemps;
thread_local int64_t tTempBytes = 0;
}  // namespace

void tierInit(const std::string& dataDir, int64_t memLimit) {
  g_memLimit = memLimit < 0 ? INT64_MAX : memLimit;
  if (dataDir.empty()) return;
  g_segDir = dataDir + "/segments";
  mkdir(g_segDir.c_str(), 0755);
  uint64_t mx = 0;
  if (DIR* dir = opendir(g_segDir.c_str())) {
    while (dirent* e = readdir(dir)) mx = std::max<uint64_t>(mx, strtoull(e->d_name, nullptr, 10));
    closedir(dir);
  }
  g_nextId = mx + 1;
}
bool tierEnabled() { return !g_segDir.empty(); }
int64_t tierMemLimit() { return g_segDir.empty() ? INT64_MAX : g_memLimit; }
int64_t tierHostBytes() { return g_hostBytes; }
int64_t tierPendingBytes() { return g_pending; }

SegFile::~SegFile() {
  if (fd >= 0) ::close(fd);
  std::lock_guard<std::mutex> g(g_regMu);
  g_live.erase(id);
  if (!g_installed.count(id)) unlink(pathOf(id).c_str());
}

SegData::~SegData() {
  for (size_t c = 0; c < dev.size(); ++c) {
    const bool e = c < enc.size() && enc[c];
    const std::vector<uint8_t>* src = e ? &encHost[c] : c < host.size() ? &host[c] : nullptr;
    releaseColumn(dev[c], src ? (void*)src->data() : nullptr, src ? src->size() : 0, c < mapped.size() && mapped[c]);
  }
  g_hostBytes -= hostBytes;
}

void segAccount(SegData& d) {
  const size_t nc = d.dev.size();
  d.cold.assign(nc, 0);
  d.gen.assign(nc, 0);
  d.evicted.assign(nc, {nullptr, 0});
  int64_t b = 0;
  for (size_t c = 0; c < nc; ++c) b += colHostBytes(d, (int)c);
  d.hostBytes = b;
  g_hostBytes += b;
}

bool segCold(const SegData& d, int c) {
  return (size_t)c < d.cold.size() && __atomic_load_n(&d.cold[c], __ATOMIC_ACQUIRE) != 0;
}

std::shared_ptr<SegFile> segFileEnsure(SegData& d, int64_t n) {
  if (g_segDir.empty()) return nullptr;
  std::lock_guard<std::mutex> g(d.loadMu);
  if (d.file) return d.file;
  const size_t nc = d.dev.size();
  auto f = std::make_shared<SegFile>();
  f->id = g_nextId++;
  f->rows = n;
  {
    std::lock_guard<std::mutex> rg(g_regMu);
    g_live.insert(f->id);
  }
  f->fd = ::open(pathOf(f->id).c_str(), O_RDWR | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
  if (f->fd < 0) throw SqlError("cannot write segment file " + pathOf(f->id), 1105);
  f->cols.resize(nc);
  uint64_t off = (8 + 16 + nc * 24 + kAlign - 1) / kAlign * kAlign;
  for (size_t c = 0; c < nc; ++c) {  // (no column is cold: a segment without a file has them all)
    const bool e = d.enc[c] != 0;
    const std::vector<uint8_t>& b = e ? d.encHost[c] : d.host[c];
    f->cols[c] = {d.enc[c], (uint32_t)(n ? d.host[c].size() / (size_t)n : 0), off, b.size()};
    if (!pwriteAll(f->fd, b.data(), b.size(), off)) throw SqlError("cannot write segment file " + pathOf(f->id) + " (disk full?)", 1105);
    off = (off + b.size() + kAlign - 1) / kAlign * kAlign;
  }
  std::vector<uint8_t> hdr(8 + 16 + nc * 24, 0);
  memcpy(hdr.data(), kMagic, 8);
  const uint64_t ncols = nc, rows = (uint64_t)n;
  memcpy(hdr.data() + 8, &ncols, 8);
  memcpy(hdr.data() + 16, &rows, 8);
  for (size_t c = 0; c < nc; ++c) {
    uint8_t* p = hdr.data() + 24 + c * 24;
    p[0] = f->cols[c].enc;
    memcpy(p + 4, &f->cols[c].width, 4);
    memcpy(p + 8, &f->cols[c].off, 8);
    memcpy(p + 16, &f->cols[c].len, 8);
  }
  if (!pwriteAll(f->fd, hdr.data(), hdr.size(), 0)) throw SqlError("cannot write segment file " + pathOf(f->id), 1105);
  d.file = f;
    return f;
}

std::shared_ptr<SegFile> segFileOpen(uint64_t id, size_t ncols, int64_t n) {
  auto f = std::make_shared<SegFile>();
  f->id = id;
  f->rows = n;
  {
    std::lock_guard<std::mutex> rg(g_regMu);
    g_live.insert(id);
  }
  f->synced = true;  // (it is in the snapshot being loaded, which is the installed one)
  {
    std::lock_guard<std::mutex> rg(g_regMu);
    g_installed.insert(id);
  }
  f->fd = ::open(pathOf(id).c_str(), O_RDONLY | O_CLOEXEC);
  if (f->fd < 0) throw SqlError("missing segment file " + pathOf(id), 1105);
  std::vector<uint8_t> hdr(8 + 16 + ncols * 24);
  uint64_t nc = 0, rows = 0;
  if (!preadAll(f->fd, hdr.data(), hdr.size(), 0) || memcmp(hdr.data(), kMagic, 8) != 0) throw SqlError("bad segment file " + pathOf(id), 1105);
  memcpy(&nc, hdr.data() + 8, 8);
  memcpy(&rows, hdr.data() + 16, 8);
  if (nc != ncols || rows != (uint64_t)n) throw SqlError("segment file " + pathOf(id) + " does not match the snapshot", 1105);
  struct stat st;
  fstat(f->fd, &st);
  f->cols.resize(ncols);
  for (size_t c = 0; c < ncols; ++c) {
    const uint8_t* p = hdr.data() + 24 + c * 24;
    f->cols[c].enc = p[0];
    memcpy(&f->cols[c].width, p + 4, 4);
    memcpy(&f->cols[c].off, p + 8, 8);
    memcpy(&f->cols[c].len, p + 16, 8);
    if (f->cols[c].off + f->cols[c].len > (uint64_t)st.st_size) throw SqlError("truncated segment file " + pathOf(id), 1105);
  }
  return f;
}

void segFileSync(SegFile& f) {
  if (f.synced) return;
  if (fdatasync(f.fd) != 0) throw SqlError("cannot sync segment file " + pathOf(f.id), 1105);
  f.synced = true;
}

void segDirSync() {
  if (g_segDir.empty()) return;
  int fd = ::open(g_segDir.c_str(), O_RDONLY);
  if (fd >= 0) fsync(fd), ::close(fd);
}

void segFilesInstalled(const std::vector<uint64_t>& ids) {
  std::lock_guard<std::mutex> g(g_regMu);
  std::unordered_set<uint64_t> now(ids.begin(), ids.end());
  for (uint64_t id : g_installed)
    if (!now.count(id) && !g_live.count(id)) unlink(pathOf(id).c_str());
  g_installed.swap(now);
}

void segFilesCleanup() {
  if (g_segDir.empty()) return;
  std::lock_guard<std::mutex> g(g_regMu);
  if (DIR* dir = opendir(g_segDir.c_str())) {
    while (dirent* e = readdir(dir)) {
      if (e->d_name[0] == '.') continue;
      const uint64_t id = strtoull(e->d_name, nullptr, 10);
      if (!g_live.count(id) && !g_installed.count(id)) unlink((g_segDir + "/" + e->d_name).c_str());
    }
    closedir(dir);
  }
}

void segLoad(SegData& d, int c) {
  std::lock_guard<std::mutex> g(d.loadMu);
  if (!d.cold[c] || revive(d, c)) return;
  loadLocked(d, c, d.file->rows);
}

namespace {
// ---- Reads from the segment files for scans: a small I/O thread pool, pinned host buffers (pooled)
// and read-ahead of the column a scan is reading for the batches it visits next.
std::atomic<int64_t> g_readBytes{0};

struct IoPool {
  std::mutex mu;
  std::condition_variable cv;
  std::deque<std::function<void()>> q;
  std::vector<std::thread> th;
  IoPool() {
    const int n = std::clamp<int>((int)std::thread::hardware_concurrency(), 4, 16);
    for (int i = 0; i < n; ++i)
      th.emplace_back([this] {
        for (;;) {
          std::function<void()> f;
          {
            std::unique_lock<std::mutex> g(mu);
            cv.wait(g, [&] { return !q.empty(); });
            f = std::move(q.front());
            q.pop_front();
          }
          f();
        }
      });
    for (auto& t : th) t.detach();
  }
  void submit(std::function<void()> f) {
    {
      std::lock_guard<std::mutex> g(mu);
      q.push_back(std::move(f));
    }
    cv.notify_one();
  }
};
IoPool& ioPool() {
  static IoPool* p = new IoPool();
  return *p;
}

// Pinned (and mapped) host buffers by power-of-two size class.
std::mutex g_pinMu;
std::map<size_t, std::vector<void*>> g_pinFree;
int64_t g_pinFreeBytes = 0;
size_t pinClass(size_t n) {
  size_t c = 1 << 20;
  while (c < n) c <<= 1;
  return c;
}
void* pinGet(size_t cap) {
  {
    std::lock_guard<std::mutex> g(g_pinMu);
    auto& v = g_pinFree[cap];
    if (!v.empty()) {
      void* p = v.back();
      v.pop_back();
      g_pinFreeBytes -= (int64_t)cap;
      return p;
    }
  }
  void* p = nullptr;
  CK(cudaHostAlloc(&p, cap, cudaHostAllocMapped | cudaHostAllocPortable));
  return p;
}
void pinPut(void* p, size_t cap);
// Buffers whose copy to the GPU is queued: back to the pool once their event has completed.
struct Pending {
  void* p;
  size_t cap;
  cudaEvent_t ev;
};
thread_local std::vector<Pending> tPending;
void pinReap(bool all) {
  for (size_t i = 0; i < tPending.size();) {
    auto& x = tPending[i];
    if (all ? cudaEventSynchronize(x.ev) == cudaSuccess || true : cudaEventQuery(x.ev) == cudaSuccess) {
      cudaEventDestroy(x.ev);
      pinPut(x.p, x.cap);
      x = tPending.back();
      tPending.pop_back();
    } else {
      cudaGetLastError();
      ++i;
    }
  }
}

void pinPut(void* p, size_t cap) {
  std::lock_guard<std::mutex> g(g_pinMu);
  if (g_pinFreeBytes + (int64_t)cap > (1ll << 30)) {  // keep at most 1 GB around
    cudaFreeHost(p);
    return;
  }
  g_pinFree[cap].push_back(p);
  g_pinFreeBytes += (int64_t)cap;
}

// What to read for rows [lo, lo + n) of a column: one piece (raw), or the encoding's header (and
// dictionary) plus the codes of the rows (encoded; read with ColView::off = 0).
struct Piece {
  uint64_t off, len, dst;
};
std::vector<Piece> piecesOf(const SegFile::Col& col, int64_t rows, int64_t lo, int64_t n, size_t& total) {
  if (lo == 0 && n >= rows) {
    total = col.len;
    return {{col.off, col.len, 0}};
  }
  if (!col.enc) {
    total = (size_t)n * col.width;
    return {{col.off + (uint64_t)lo * col.width, total, 0}};
  }
  const uint64_t pre = kEncHeader + (col.enc == 4 ? 256 * 8 : 0), w = (uint64_t)encWidth(col.enc);
  total = pre + n * w + 16;
  return {{col.off, pre, 0}, {col.off + pre + (uint64_t)lo * w, (uint64_t)n * w, pre}};
}

void readPieces(const SegFile& f, const std::vector<Piece>& ps, uint8_t* dst) {
  for (auto& p : ps)
    if (!preadAll(f.fd, dst + p.dst, p.len, p.off)) throw SqlError("cannot read segment file " + pathOf(f.id), 1105);
  for (auto& p : ps) g_readBytes += (int64_t)p.len;
}

// Reads pieces with the I/O threads in 8 MB chunks (several requests in flight on the SSD).
void readParallel(const SegFile& f, const std::vector<Piece>& ps, uint8_t* dst) {
  constexpr uint64_t kChunk = 8 << 20;
  std::vector<Piece> chunks;
  for (auto& p : ps)
    for (uint64_t o = 0; o < p.len; o += kChunk) chunks.push_back({p.off + o, std::min(kChunk, p.len - o), p.dst + o});
  if (chunks.size() <= 1) return readPieces(f, chunks, dst);
  std::mutex m;
  std::condition_variable cv;
  size_t left = chunks.size();
  bool failed = false;
  for (auto& ch : chunks)
    ioPool().submit([&, ch] {
      bool ok = preadAll(f.fd, dst + ch.dst, ch.len, ch.off);
      std::lock_guard<std::mutex> g(m);
      failed |= !ok;
      if (--left == 0) cv.notify_all();
    });
  std::unique_lock<std::mutex> g(m);
  cv.wait(g, [&] { return left == 0; });
  if (failed) throw SqlError("cannot read segment file " + pathOf(f.id), 1105);
  for (auto& ch : chunks) g_readBytes += (int64_t)ch.len;
}

// A read-ahead in flight: the column's bytes for one batch, into a pinned buffer.
struct Fetch {
  std::mutex mu;
  std::condition_variable cv;
  bool done = false, failed = false;
  void* buf = nullptr;
  size_t cap = 0, len = 0;
};
struct FetchKey {
  const SegData* d;
  int c;
  int64_t lo;
  bool operator<(const FetchKey& o) const { return std::tie(d, c, lo) < std::tie(o.d, o.c, o.lo); }
};
thread_local std::map<FetchKey, std::shared_ptr<Fetch>> tFetches;
thread_local int64_t tFetchBytes = 0;
thread_local std::vector<ScanAheadItem> tAhead;
thread_local size_t tAheadPos = 0;
int64_t aheadBudget() {
  static const int64_t b = [] {
    size_t freeB = 0, total = 0;
    cudaMemGetInfo(&freeB, &total);
    return std::clamp<int64_t>((int64_t)total / 16, 64ll << 20, 1ll << 30);
  }();
  return b;
}

void waitFetch(Fetch& f) {
  std::unique_lock<std::mutex> g(f.mu);
  f.cv.wait(g, [&] { return f.done; });
}

// Starts reading column c of the next batches of the scan (the ones that are on disk).
void readAhead(int c) {
  int started = 0;
  for (size_t j = tAheadPos + 1; j < tAhead.size() && started < 8 && tFetchBytes < aheadBudget(); ++j) {
    const ScanAheadItem& it = tAhead[j];
    if (!segCold(*it.d, c) || tFetches.count({it.d, c, it.lo})) continue;
    auto file = it.d->file;
    const auto& col = file->cols[c];
    if (it.lo == 0 && it.n >= file->rows && g_hostBytes + (int64_t)col.len + (col.enc ? file->rows * (int64_t)col.width : 0) <= g_memLimit)
      continue;  // (it goes into the cache when the scan gets to it)
    size_t total = 0;
    auto ps = piecesOf(file->cols[c], file->rows, it.lo, it.n, total);
    auto f = std::make_shared<Fetch>();
    f->len = total;
    f->cap = pinClass(std::max<size_t>(total, 1));
    tFetches[{it.d, c, it.lo}] = f;
    tFetchBytes += (int64_t)f->cap;
    ++started;
    ioPool().submit([f, file, ps] {
      bool ok = true;
      try {
        f->buf = pinGet(f->cap);
        readPieces(*file, ps, (uint8_t*)f->buf);
      } catch (...) {
        ok = false;
      }
      std::lock_guard<std::mutex> g(f->mu);
      f->failed = !ok;
      f->done = true;
      f->cv.notify_all();
    });
  }
}
}  // namespace

void scanAheadExchange(std::vector<ScanAheadItem>& items, size_t& pos) {
  std::swap(tAhead, items);
  std::swap(tAheadPos, pos);
}
void scanAheadAt(size_t pos) { tAheadPos = pos; }
int64_t tierReadBytes() { return g_readBytes; }

void* segScanColumn(SegData& d, int c, int64_t lo, int64_t n, bool& rebased) {
  rebased = false;
  if (!segCold(d, c)) return d.dev[c];
  const bool whole = lo == 0 && n >= d.file->rows;
  auto it = tFetches.find({&d, c, lo});
  {
    std::lock_guard<std::mutex> g(d.loadMu);
    if (!d.cold[c] || revive(d, c)) return d.dev[c];
    const auto& col = d.file->cols[c];
    const int64_t need = (int64_t)col.len + (col.enc ? (int64_t)d.file->rows * col.width : 0);
    if (whole && it == tFetches.end() && g_hostBytes + need <= g_memLimit) {  // room in the cache
      loadLocked(d, c, d.file->rows);
      return d.dev[c];
    }
  }
  // Over the budget: the rows of this batch, for this scan only (read ahead if possible).
  std::shared_ptr<Fetch> f;
  if (it != tFetches.end()) {
    f = it->second;
    tFetches.erase(it);
    waitFetch(*f);
    tFetchBytes -= (int64_t)f->cap;
    if (f->failed) {
      if (f->buf) pinPut(f->buf, f->cap);
      throw SqlError("cannot read segment file " + pathOf(d.file->id), 1105);
    }
  } else {
    size_t total = 0;
    auto ps = piecesOf(d.file->cols[c], d.file->rows, lo, n, total);
    f = std::make_shared<Fetch>();
    f->len = total;
    f->cap = pinClass(std::max<size_t>(total, 1));
    f->buf = pinGet(f->cap);
    try {
      readParallel(*d.file, ps, (uint8_t*)f->buf);
    } catch (...) {
      pinPut(f->buf, f->cap);
      throw;
    }
  }
  pinReap(false);
  readAhead(c);
  rebased = !whole;
  tTempBytes += (int64_t)f->len;
  void* p = nullptr;
  if (cudaMallocAsync(&p, std::max<size_t>(f->len, 1), 0) == cudaSuccess) {
    CK(cudaMemcpyAsync(p, f->buf, f->len, cudaMemcpyHostToDevice, 0));
    tTemps.push_back({p, 0, 0});
    cudaEvent_t ev;
    CK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
    CK(cudaEventRecord(ev, 0));
    tPending.push_back({f->buf, f->cap, ev});  // (reused once the copy has run)
    return p;
  }
  cudaGetLastError();  // VRAM full: the kernels read the pinned buffer over PCIe
  tTemps.push_back({f->buf, 2, f->cap});
  CK(cudaHostGetDevicePointer(&p, f->buf, 0));
  return p;
}

void releaseScanTemps() {
  if (tTemps.empty()) return;
  bool synced = false;
  for (auto& t : tTemps) {
    if (t.kind == 0) cudaFreeAsync(t.p, 0);
    else {
      if (!synced) cudaStreamSynchronize(0), synced = true;  // (copies and kernels reading it are done)
      pinPut(t.p, t.cap);
    }
  }
  tTemps.clear();
  tTempBytes = 0;
}

void releaseStatementTemps() {
  releaseScanTemps();
  pinReap(true);
  for (auto& [k, f] : tFetches) {  // read-aheads the scans did not get to
    waitFetch(*f);
    if (f->buf) pinPut(f->buf, f->cap);
  }
  tFetches.clear();
  tFetchBytes = 0;
}

bool scanWindowFull() {
  static const int64_t window = [] {
    if (const char* v = envSetting("SCAN_WINDOW_MB")) return (int64_t)atoll(v) << 20;
    size_t freeB = 0, total = 0;
    cudaMemGetInfo(&freeB, &total);
    return std::clamp<int64_t>((int64_t)total / 8, 64ll << 20, 2ll << 30);
  }();
  return tTempBytes >= window;
}

const uint8_t* segRawColumn(const SegData& d, int c, std::vector<uint8_t>& tmp) {
  if (!segCold(d, c)) return d.host[c].data();
  const auto& col = d.file->cols[c];
  std::vector<uint8_t> stored(col.len);
  readCol(*d.file, c, stored.data());
  if (!col.enc) {
    tmp = std::move(stored);
    return tmp.data();
  }
  tmp.resize((size_t)d.file->rows * col.width);
  decodeInto(col, stored.data(), d.file->rows, tmp.data());
  return tmp.data();
}

long long segRawInt(SegData& d, int c, int64_t r, bool i64) {
  if (!segCold(d, c)) {
    const uint8_t* h = d.host[c].data();
    return i64 ? (long long)((const int64_t*)h)[r] : (long long)((const int32_t*)h)[r];
  }
  const auto& col = d.file->cols[c];
  if (!col.enc) {
    if (col.width == 4) {
      int32_t v;
      if (!preadAll(d.file->fd, &v, 4, col.off + (uint64_t)r * 4)) throw SqlError("cannot read segment file", 1105);
      return v;
    }
    int64_t v;
    if (!preadAll(d.file->fd, &v, 8, col.off + (uint64_t)r * 8)) throw SqlError("cannot read segment file", 1105);
    return v;
  }
  const int w = encWidth(col.enc);
  uint8_t buf[kEncHeader + 8] = {};
  if (!preadAll(d.file->fd, buf, kEncHeader, col.off) || !preadAll(d.file->fd, buf + kEncHeader, w, col.off + kEncHeader + (uint64_t)r * w))
    throw SqlError("cannot read segment file", 1105);
  ColView v{};
  v.p = buf;
  v.enc = col.enc;
  v.t = col.width == 4 ? 0 : 1;
  return encInt(v, 0);
}

void segRawCell(SegData& d, int c, int64_t r, uint8_t* out) {
  const auto& col = d.file->cols[c];
  if (!col.enc) {
    if (!preadAll(d.file->fd, out, col.width, col.off + (uint64_t)r * col.width)) throw SqlError("cannot read segment file", 1105);
    return;
  }
  const uint64_t pre = kEncHeader + (col.enc == 4 ? 256 * 8 : 0), w = (uint64_t)encWidth(col.enc);
  std::vector<uint8_t> buf(pre + 8, 0);
  if (!preadAll(d.file->fd, buf.data(), pre, col.off) || !preadAll(d.file->fd, buf.data() + pre, w, col.off + pre + (uint64_t)r * w))
    throw SqlError("cannot read segment file", 1105);
  decodeInto(SegFile::Col{col.enc, col.width, 0, 0}, buf.data(), 1, out);
}

int64_t segEvictBegin(SegData& d, int c, uint32_t& gen) {
  std::lock_guard<std::mutex> g(d.loadMu);
  if ((size_t)c >= d.cold.size() || d.cold[c] || !d.file) return 0;
  d.evicted[c] = {d.dev[c], d.mapped[c]};
  d.dev[c] = nullptr;
  gen = ++d.gen[c];
  const int64_t b = colHostBytes(d, c);
  g_pending += b;
  __atomic_store_n(&d.cold[c], (uint8_t)1, __ATOMIC_RELEASE);
  return b;
}

void segEvictFinish(SegData& d, int c, uint32_t gen) {
  std::lock_guard<std::mutex> g(d.loadMu);
  if (d.gen[c] != gen || !d.cold[c]) return;  // loaded again (revived) since
  auto [p, m] = d.evicted[c];
  std::vector<uint8_t>& placed = d.enc[c] ? d.encHost[c] : d.host[c];
  if (m) hostUnmap(placed.data());
  else vramRelease(p, placed.size());
  d.evicted[c] = {nullptr, 0};
  d.mapped[c] = 0;
  const int64_t b = colHostBytes(d, c);
  std::vector<uint8_t>().swap(d.host[c]);
  std::vector<uint8_t>().swap(d.encHost[c]);
  d.hostBytes -= b;
  g_hostBytes -= b;
  g_pending -= b;
}

}  // namespace gdb
