// Disk tier for segment column data (see SegFile in storage.h).
#include <dirent.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <zstd.h>
#include <algorithm>
#include <array>
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
#include "gpuio.h"
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

constexpr char kMagic1[8] = {'W', 'S', 'E', 'G', '0', '0', '0', '1'};  // stored as is (reads only)
constexpr char kMagic2[8] = {'W', 'S', 'E', 'G', '0', '0', '0', '2'};  // + zstd blocks, index at the end
constexpr char kMagic3[8] = {'W', 'S', 'E', 'G', '0', '0', '0', '3'};  // + CRC32Cs (blocks, prefixes, the index)

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

std::atomic<int64_t> g_readBytes{0};

thread_local bool tInPool = false;
std::atomic<int> g_poolBusy{0};  // pool workers running a task
}  // namespace
bool ioPoolCrowded() { return tInPool && g_poolBusy.load(std::memory_order_relaxed) > 2; }
namespace {
// A small I/O thread pool: reads (several requests in flight on the SSD), decompression, compression.
struct IoPool {
  std::mutex mu;
  std::condition_variable cv;
  std::deque<std::function<void()>> q;
  std::vector<std::thread> th;
  IoPool() {
    const int n = std::clamp<int>((int)std::thread::hardware_concurrency(), 4, 16);
    for (int i = 0; i < n; ++i)
      th.emplace_back([this] {
        tInPool = true;
        for (;;) {
          std::function<void()> f;
          {
            std::unique_lock<std::mutex> g(mu);
            cv.wait(g, [&] { return !q.empty(); });
            f = std::move(q.front());
            q.pop_front();
          }
          ++g_poolBusy;
          f();
          --g_poolBusy;
        }
      });
    for (auto& t : th) t.detach();
  }
  void submit(std::function<void()> f, bool urgent = false) {
    {
      std::lock_guard<std::mutex> g(mu);
      if (urgent) q.push_front(std::move(f));
      else q.push_back(std::move(f));
    }
    cv.notify_one();
  }
};
IoPool& ioPool() {
  static IoPool* p = new IoPool();
  return *p;
}


}  // namespace

// Runs fn(i) for i in [0, n) on the I/O threads and waits (fn must not throw).
void ioParallel(size_t n, const std::function<void(size_t)>& fn, bool urgent) {
  if (n <= 1 || tInPool) {  // (inline on a pool thread: waiting there could deadlock the pool)
    for (size_t i = 0; i < n; ++i) fn(i);
    return;
  }
  std::mutex m;
  std::condition_variable cv;
  size_t left = n;
  const bool strict = tStrictTypes;  // (the statement's strict SQL mode, on the pool threads too)
  for (size_t i = 0; i < n; ++i)
    ioPool().submit([&, i, strict] {
      tStrictTypes = strict;
      fn(i);
      tStrictTypes = false;
      std::lock_guard<std::mutex> g(m);
      if (--left == 0) cv.notify_all();
    }, urgent);
  std::unique_lock<std::mutex> g(m);
  cv.wait(g, [&] { return left == 0; });
}

namespace {

constexpr uint32_t kBlockRows = 16384;
uint64_t prefixOf(uint8_t enc) { return enc ? kEncHeader + (enc == 4 ? 256 * 8 : 0) : 0; }
uint64_t elemOf(const SegFile::Col& col) { return col.enc ? (uint64_t)encWidth(col.enc) : col.width; }

// Size of what readRows produces for rows [lo, lo + n).
size_t storedSize(const SegFile& f, int c, int64_t lo, int64_t n) {
  const auto& col = f.cols[c];
  if (lo == 0 && n >= f.rows) return col.len;
  return col.enc ? prefixOf(col.enc) + (size_t)n * elemOf(col) + 16 : (size_t)n * col.width;
}

thread_local ZSTD_DCtx* tDctx = nullptr;
// This thread's last decompressed blocks (single-row reads: binary searches, late materialization).
struct BlockCache {
  uint64_t file = 0;
  int c = -1;
  int64_t b = -1;
  std::vector<uint8_t> data;
};
thread_local BlockCache tBlocks[8];
thread_local unsigned tBlockNext = 0;

// Block b of a compressed column, uncompressed, into out (rows * elem bytes).
[[noreturn]] void corrupt(const SegFile& f, const std::string& where) {
  throw SqlError("corrupt segment file " + pathOf(f.id) + " (" + where + ": checksum mismatch)", 1105);
}
// Format 3: block b's stored bytes against their CRC32C.
void checkBlock(const SegFile& f, const SegFile::Col& col, int64_t b, const void* p, size_t len) {
  if (!col.crcs.empty() && crc32c(p, len) != col.crcs[b]) corrupt(f, "column " + std::to_string(&col - f.cols.data()) + ", block " + std::to_string(b));
}

void readBlock(const SegFile& f, const SegFile::Col& col, int64_t b, uint8_t* out, size_t rawLen) {
  const uint64_t a = col.blocks[b], e = col.blocks[b + 1], clen = e - a;
  if (clen == rawLen) {
    if (!preadAll(f.fd, out, rawLen, a)) throw SqlError("cannot read segment file " + pathOf(f.id), 1105);
    checkBlock(f, col, b, out, rawLen);
  } else {
    std::vector<uint8_t> z(clen);
    if (!preadAll(f.fd, z.data(), clen, a)) throw SqlError("cannot read segment file " + pathOf(f.id), 1105);
    checkBlock(f, col, b, z.data(), clen);
    if (!tDctx) tDctx = ZSTD_createDCtx();
    const size_t r = ZSTD_decompressDCtx(tDctx, out, rawLen, z.data(), clen);
    if (ZSTD_isError(r) || r != rawLen) throw SqlError("corrupt segment file " + pathOf(f.id), 1105);
  }
  g_readBytes += (int64_t)clen;
}

// The stored bytes of rows [lo, lo + n) of column c: the whole column as written when the range is
// all of it, otherwise the rows from lo (encoded columns: header / dictionary, then the rows' codes,
// read with ColView::off = 0). parallel: blocks on the I/O threads (not from an I/O thread).
void readRows(const SegFile& f, int c, int64_t lo, int64_t n, uint8_t* dst, bool parallel) {
  const auto& col = f.cols[c];
  const bool whole = lo == 0 && n >= f.rows;
  const uint64_t pre = prefixOf(col.enc), w = elemOf(col);
  const size_t total = storedSize(f, c, lo, n);
  auto fail = [&] { throw SqlError("cannot read segment file " + pathOf(f.id), 1105); };
  if (!col.blockRows) {  // stored as is
    std::vector<std::array<uint64_t, 3>> ps;  // file offset, length, destination
    if (whole) ps.push_back({col.off, col.len, 0});
    else if (!col.enc) ps.push_back({col.off + (uint64_t)lo * w, (uint64_t)n * w, 0});
    else {
      ps.push_back({col.off, pre, 0});
      ps.push_back({col.off + pre + (uint64_t)lo * w, (uint64_t)n * w, pre});
      memset(dst + pre + n * w, 0, 16);
    }
    std::vector<std::array<uint64_t, 3>> chunks;  // 8 MB pieces
    for (auto& p : ps)
      for (uint64_t o = 0; o < p[1]; o += 8 << 20) chunks.push_back({p[0] + o, std::min<uint64_t>(8 << 20, p[1] - o), p[2] + o});
    std::atomic<bool> bad{false};
    auto one = [&](size_t i) {
      if (!preadAll(f.fd, dst + chunks[i][2], chunks[i][1], chunks[i][0])) bad = true;
      g_readBytes += (int64_t)chunks[i][1];
    };
    if (parallel) ioParallel(chunks.size(), one);
    else
      for (size_t i = 0; i < chunks.size(); ++i) one(i);
    if (bad) fail();
    if (whole && !f.cols[c].crcs.empty() && crc32c(dst, col.len) != col.crc) corrupt(f, "column " + std::to_string(c));
    return;
  }
  if (pre && !preadAll(f.fd, dst, pre, col.off)) fail();
  if (pre && !col.crcs.empty() && crc32c(dst, pre) != col.crc) corrupt(f, "column " + std::to_string(c) + " header");
  uint8_t* base = dst + pre;
  const int64_t hi = whole ? f.rows : lo + n;
  if (total > pre + (uint64_t)(hi - lo) * w) memset(base + (hi - lo) * w, 0, total - pre - (hi - lo) * w);  // (slack after the codes)
  if (hi <= lo) return;
  const int64_t B = col.blockRows, b0 = lo / B, b1 = (hi - 1) / B;
  std::string err;
  std::mutex errMu;
  auto one = [&](size_t i) {
    const int64_t b = b0 + (int64_t)i, r0 = b * B, r1 = std::min<int64_t>(r0 + B, f.rows);
    const int64_t s0 = std::max(r0, lo), s1 = std::min(r1, hi);  // rows of this block wanted
    const size_t rawLen = (size_t)(r1 - r0) * w;
    try {
      if (s0 == r0 && s1 == r1) {
        readBlock(f, col, b, base + (r0 - lo) * w, rawLen);
        return;
      }
      if (hi - lo <= 64) {  // a few rows: through this thread's block cache
        for (auto& e : tBlocks)
          if (e.file == f.id && e.c == c && e.b == b) {
            memcpy(base + (s0 - lo) * w, e.data.data() + (s0 - r0) * w, (s1 - s0) * w);
            return;
          }
        BlockCache& e = tBlocks[tBlockNext++ % 8];
        e.file = 0;
        e.data.resize(rawLen);
        readBlock(f, col, b, e.data.data(), rawLen);
        e.file = f.id, e.c = c, e.b = b;
        memcpy(base + (s0 - lo) * w, e.data.data() + (s0 - r0) * w, (s1 - s0) * w);
        return;
      }
      std::vector<uint8_t> tmp(rawLen);
      readBlock(f, col, b, tmp.data(), rawLen);
      memcpy(base + (s0 - lo) * w, tmp.data() + (s0 - r0) * w, (s1 - s0) * w);
    } catch (std::exception& ex) {
      std::lock_guard<std::mutex> g(errMu);
      err = ex.what();
    }
  };
  const size_t nb = (size_t)(b1 - b0 + 1);
  if (parallel) ioParallel(nb, one);
  else
    for (size_t i = 0; i < nb; ++i) one(i);
  if (!err.empty()) throw SqlError(err, 1105);
}

void readCol(const SegFile& f, int c, void* dst) { readRows(f, c, 0, f.rows, (uint8_t*)dst, true); }

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
static int64_t g_memCap = -1;  // the process's memory cap (cgroup), when known
void tierSetMemCap(int64_t cap) { g_memCap = cap; }
// --memory-limit, lowered under a memory cap so that everything else the process holds (dictionaries,
// write buffers, query memory: resident anonymous memory that is not segment data) still fits in 85%.
int64_t tierMemLimit() {
  if (g_segDir.empty()) return INT64_MAX;
  int64_t lim = g_memLimit;
  if (g_memCap > 0) {
    FILE* f = fopen("/proc/self/statm", "r");
    long long size = 0, rss = 0, shared = 0;
    if (f) {
      if (fscanf(f, "%lld %lld %lld", &size, &rss, &shared) != 3) rss = shared = 0;
      fclose(f);
    }
    const int64_t anon = (int64_t)(rss - shared) * sysconf(_SC_PAGESIZE);
    const int64_t other = std::max<int64_t>(0, anon - g_hostBytes);
    lim = std::min<int64_t>(lim, std::max<int64_t>(g_memCap / 10, g_memCap / 20 * 17 - other));
  }
  return lim;
}
int64_t tierHostBytes() { return g_hostBytes; }
int64_t tierPendingBytes() { return g_pending; }

SegFile::~SegFile() {
  gdsClose(gds);
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

// Format 2: header (magic, ncols, rows, index offset, index size), the columns' data from 4096 on,
// then the index: per column enc, width, blockRows, prefix, len, off, block count and offsets.
std::shared_ptr<SegFile> segFileEnsure(SegData& d, int64_t n, int rawCol) {
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
  // Compress every block of every column on the I/O threads (no column is cold: a segment without
  // a file has them all).
  struct Job {
    size_t c;
    int64_t b;
  };
  std::vector<Job> jobs;
  std::vector<std::vector<std::vector<uint8_t>>> z(nc);  // [column][block]: compressed, or empty = as is
  std::vector<std::vector<uint32_t>> crcs(nc);           // [column][block]: CRC32C of the stored bytes
  for (size_t c = 0; c < nc; ++c) {
    auto& col = f->cols[c];
    const std::vector<uint8_t>& bytes = d.enc[c] ? d.encHost[c] : d.host[c];
    col.enc = d.enc[c];
    col.width = (uint32_t)(n ? d.host[c].size() / (size_t)n : 0);
    col.len = bytes.size();
    const uint64_t w = elemOf(col);
    if ((int)c == rawCol || n < 4096 || w == 0) continue;
    col.blockRows = kBlockRows;
    col.prefix = prefixOf(col.enc);
    const int64_t nb = (n + kBlockRows - 1) / kBlockRows;
    z[c].resize(nb);
    crcs[c].resize(nb);
    for (int64_t b = 0; b < nb; ++b) jobs.push_back({c, b});
  }
  ioParallel(jobs.size(), [&](size_t i) {
    static thread_local ZSTD_CCtx* cctx = ZSTD_createCCtx();
    const Job& j = jobs[i];
    const auto& col = f->cols[j.c];
    const std::vector<uint8_t>& bytes = d.enc[j.c] ? d.encHost[j.c] : d.host[j.c];
    const uint64_t w = elemOf(col);
    const int64_t r0 = j.b * kBlockRows, r1 = std::min<int64_t>(r0 + kBlockRows, n);
    const uint8_t* src = bytes.data() + col.prefix + r0 * w;
    const size_t len = (size_t)(r1 - r0) * w;
    std::vector<uint8_t> out(ZSTD_compressBound(len));
    const size_t k = ZSTD_compressCCtx(cctx, out.data(), out.size(), src, len, 1);
    if (!ZSTD_isError(k) && k < len) {
      out.resize(k);
      crcs[j.c][j.b] = crc32c(out.data(), k);
      z[j.c][j.b] = std::move(out);
    } else {
      crcs[j.c][j.b] = crc32c(src, len);
    }
  });
  uint64_t off = 4096;
  auto put = [&](const void* p, size_t len) {
    if (len && !pwriteAll(f->fd, p, len, off)) throw SqlError("cannot write segment file " + pathOf(f->id) + " (disk full?)", 1105);
    off += len;
  };
  for (size_t c = 0; c < nc; ++c) {
    auto& col = f->cols[c];
    const std::vector<uint8_t>& bytes = d.enc[c] ? d.encHost[c] : d.host[c];
    col.off = off;
    if (!col.blockRows) {
      col.crc = crc32c(bytes.data(), bytes.size());
      col.crcs.assign(1, 0);  // (non-empty: checked; the crc is the whole column's)
      put(bytes.data(), bytes.size());
      continue;
    }
    col.crc = crc32c(bytes.data(), col.prefix);
    col.crcs = std::move(crcs[c]);
    put(bytes.data(), col.prefix);
    const uint64_t w = elemOf(col);
    for (size_t b = 0; b < z[c].size(); ++b) {
      col.blocks.push_back(off);
      if (!z[c][b].empty()) put(z[c][b].data(), z[c][b].size());
      else {
        const int64_t r0 = (int64_t)b * kBlockRows, r1 = std::min<int64_t>(r0 + kBlockRows, n);
        put(bytes.data() + col.prefix + r0 * w, (size_t)(r1 - r0) * w);
      }
    }
    col.blocks.push_back(off);
    std::vector<std::vector<uint8_t>>().swap(z[c]);
  }
  std::vector<uint8_t> ix;
  auto u64 = [&](uint64_t v) { ix.insert(ix.end(), (uint8_t*)&v, (uint8_t*)&v + 8); };
  for (auto& col : f->cols) {
    u64(col.enc);
    u64(col.width);
    u64(col.blockRows);
    u64(col.prefix);
    u64(col.len);
    u64(col.off);
    u64(col.blocks.size());
    for (uint64_t b : col.blocks) u64(b);
    u64(col.crc);
    for (size_t b = 0; b + 1 < col.blocks.size(); ++b) u64(col.crcs[b]);
  }
  u64(crc32c(ix.data(), ix.size()));  // (the index's own CRC, last)
  const uint64_t ixOff = off;
  put(ix.data(), ix.size());
  uint8_t hdr[40];
  const uint64_t hv[4] = {(uint64_t)nc, (uint64_t)n, ixOff, (uint64_t)ix.size()};
  memcpy(hdr, kMagic3, 8);
  memcpy(hdr + 8, hv, 32);
  if (!pwriteAll(f->fd, hdr, sizeof hdr, 0)) throw SqlError("cannot write segment file " + pathOf(f->id), 1105);
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
    g_installed.insert(id);  // (it is in the snapshot being loaded, which is the installed one)
  }
  f->synced = true;
  f->fd = ::open(pathOf(id).c_str(), O_RDONLY | O_CLOEXEC);
  if (f->fd < 0) throw SqlError("missing segment file " + pathOf(id), 1105);
  auto bad = [&](const char* what) { throw SqlError(std::string(what) + " segment file " + pathOf(id), 1105); };
  struct stat st;
  fstat(f->fd, &st);
  const uint64_t size = (uint64_t)st.st_size;
  char magic[8];
  uint64_t hv[4] = {0, 0, 0, 0};
  if (!preadAll(f->fd, magic, 8, 0) || !preadAll(f->fd, hv, 16, 8)) bad("bad");
  if (hv[0] != ncols || hv[1] != (uint64_t)n) bad("mismatched (does not match the snapshot)");
  f->cols.resize(ncols);
  if (memcmp(magic, kMagic1, 8) == 0) {
    std::vector<uint8_t> hdr(ncols * 24);
    if (!preadAll(f->fd, hdr.data(), hdr.size(), 24)) bad("bad");
    for (size_t c = 0; c < ncols; ++c) {
      const uint8_t* p = hdr.data() + c * 24;
      f->cols[c].enc = p[0];
      memcpy(&f->cols[c].width, p + 4, 4);
      memcpy(&f->cols[c].off, p + 8, 8);
      memcpy(&f->cols[c].len, p + 16, 8);
      if (f->cols[c].off + f->cols[c].len > size) bad("truncated");
    }
    return f;
  }
  const bool v3 = memcmp(magic, kMagic3, 8) == 0;
  if ((!v3 && memcmp(magic, kMagic2, 8) != 0) || !preadAll(f->fd, hv + 2, 16, 24) || hv[2] + hv[3] > size) bad("bad");
  std::vector<uint64_t> ix(hv[3] / 8);
  if (!preadAll(f->fd, ix.data(), ix.size() * 8, hv[2])) bad("bad");
  if (v3 && (ix.empty() || crc32c(ix.data(), (ix.size() - 1) * 8) != (uint32_t)ix.back())) bad("corrupt (index checksum)");
  size_t k = 0;
  auto next = [&]() -> uint64_t {
    if (k >= ix.size()) bad("bad");
    return ix[k++];
  };
  for (auto& col : f->cols) {
    col.enc = (uint8_t)next();
    col.width = (uint32_t)next();
    col.blockRows = (uint32_t)next();
    col.prefix = next();
    col.len = next();
    col.off = next();
    const uint64_t nb = next();
    col.blocks.resize(nb);
    for (auto& b : col.blocks) b = next();
    if (v3) {
      col.crc = (uint32_t)next();
      col.crcs.resize(nb ? nb - 1 : 0);
      for (auto& x : col.crcs) x = (uint32_t)next();
      if (!col.blockRows) col.crcs.assign(1, 0);  // (non-empty: the whole column's crc is checked)
    }
    if (col.blockRows ? (col.blocks.empty() || col.blocks.back() > size) : col.off + col.len > size) bad("truncated");
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
// ---- Reads from the segment files for scans: pinned host buffers (pooled) and read-ahead of the
// column a scan is reading for the batches it visits next.

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
// VRAM buffers for GPUDirect Storage reads (cudaMalloc'd, by size class, reused).
std::mutex g_devMu;
std::map<size_t, std::vector<void*>> g_devFree;
int64_t g_devFreeBytes = 0;
void* devGet(size_t cap) {
  {
    std::lock_guard<std::mutex> g(g_devMu);
    auto& v = g_devFree[cap];
    if (!v.empty()) {
      void* p = v.back();
      v.pop_back();
      g_devFreeBytes -= (int64_t)cap;
      return p;
    }
  }
  void* p = nullptr;
  CK(cudaMalloc(&p, cap));
  return p;
}
void devPut(void* p, size_t cap) {
  std::lock_guard<std::mutex> g(g_devMu);
  if (g_devFreeBytes + (int64_t)cap > (256ll << 20)) {
    cudaFree(p);
    return;
  }
  g_devFree[cap].push_back(p);
  g_devFreeBytes += (int64_t)cap;
}
// Buffers that queued GPU work still reads: back to their pool once their event has completed.
struct Pending {
  void* p;
  size_t cap;
  cudaEvent_t ev;
  bool dev = false;  // devPut, else pinPut
};
thread_local std::vector<Pending> tPending;
void pinReap(bool all) {
  for (size_t i = 0; i < tPending.size();) {
    auto& x = tPending[i];
    if (all ? cudaEventSynchronize(x.ev) == cudaSuccess || true : cudaEventQuery(x.ev) == cudaSuccess) {
      cudaEventDestroy(x.ev);
      if (x.dev) devPut(x.p, x.cap);
      else pinPut(x.p, x.cap);
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

// A read-ahead in flight: the column's bytes for one batch, into a pinned buffer.
struct Fetch {
  std::mutex mu;
  std::condition_variable cv;
  bool done = false, failed = false;
  std::string error;  // (failed: why)
  void* buf = nullptr;  // pinned: the stored bytes (CPU path) or the compressed blocks (GPU path)
  size_t cap = 0, len = 0;
  // GPU path: blocks b0..b1, compressed in buf (or dbuf: read into VRAM with GPUDirect Storage), and
  // the encoding's prefix.
  bool gpu = false;
  int64_t b0 = 0, b1 = -1;
  uint64_t span = 0;
  std::vector<uint8_t> prefix;
  void* dbuf = nullptr;
  size_t dcap = 0;
};
void fetchFree(Fetch& f) {
  if (f.buf) pinPut(f.buf, f.cap);
  if (f.dbuf) devPut(f.dbuf, f.dcap);
  f.buf = f.dbuf = nullptr;
}
// Decompress this column's blocks on the GPU? (Small reads, like a narrow SORT KEY range, are faster
// on the CPU.)
bool gpuPath(const SegFile::Col& col, int64_t n) { return col.blockRows && nvcompAvailable() && (uint64_t)n * elemOf(col) >= (1u << 20); }
void* gdsHandle(SegFile& f) {
  std::lock_guard<std::mutex> g(f.gdsMu);
  if (!f.gdsTried) {
    f.gdsTried = true;
    f.gds = gdsOpen(pathOf(f.id));
  }
  return f.gds;
}
// GPU path: the compressed blocks with rows [lo, lo + n) and the prefix (on an I/O thread or not).
void readCompressed(SegFile& file, int c, int64_t lo, int64_t n, Fetch& f) {
  const auto& col = file.cols[c];
  const int64_t B = col.blockRows;
  f.gpu = true;
  f.b0 = lo / B;
  f.b1 = (std::min(lo + n, file.rows) - 1) / B;
  f.span = col.blocks[f.b1 + 1] - col.blocks[f.b0];
  f.prefix.resize(col.prefix);
  if (col.prefix && !preadAll(file.fd, f.prefix.data(), col.prefix, col.off)) throw SqlError("cannot read segment file " + pathOf(file.id), 1105);
  if (col.prefix && !col.crcs.empty() && crc32c(f.prefix.data(), col.prefix) != col.crc) corrupt(file, "column " + std::to_string(c) + " header");
  if (void* h = gdsHandle(file)) {
    f.dcap = pinClass(std::max<size_t>(f.span, 1));
    f.dbuf = devGet(f.dcap);
    if (gdsRead(h, f.dbuf, f.span, col.blocks[f.b0])) {
      g_readBytes += (int64_t)f.span;
      return;
    }
    devPut(f.dbuf, f.dcap);  // (failed: through host memory)
    f.dbuf = nullptr;
  }
  f.cap = pinClass(std::max<size_t>(f.span, 1));
  f.buf = pinGet(f.cap);
  if (!preadAll(file.fd, f.buf, f.span, col.blocks[f.b0])) throw SqlError("cannot read segment file " + pathOf(file.id), 1105);
  if (!col.crcs.empty())  // (format 3; GPUDirect reads above skip host memory and are not checked)
    for (int64_t b = f.b0; b <= f.b1; ++b)
      checkBlock(file, col, b, (const uint8_t*)f.buf + (col.blocks[b] - col.blocks[f.b0]), col.blocks[b + 1] - col.blocks[b]);
  g_readBytes += (int64_t)f.span;
}
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
    SegData* d = &it.seg->cd(c);
    if (!segCold(*d, c) || tFetches.count({d, c, it.lo})) continue;
    auto file = d->file;
    const auto& col = file->cols[c];
    if (it.lo == 0 && it.n >= file->rows && g_hostBytes + (int64_t)col.len + (col.enc ? file->rows * (int64_t)col.width : 0) <= g_memLimit)
      continue;  // (it goes into the cache when the scan gets to it)
    const size_t total = storedSize(*file, c, it.lo, it.n);
    const int64_t lo = it.lo, n = it.n;
    auto f = std::make_shared<Fetch>();
    f->len = total;
    const bool gpu = gpuPath(col, n);
    const size_t cap = pinClass(std::max<size_t>(gpu ? col.blocks[(std::min(lo + n, file->rows) - 1) / col.blockRows + 1] - col.blocks[lo / col.blockRows] : total, 1));
    tFetches[{d, c, it.lo}] = f;
    tFetchBytes += (int64_t)cap;
    ++started;
    ioPool().submit([f, file, c, lo, n, gpu, cap] {
      bool ok = true;
      std::string err;
      try {
        if (gpu) readCompressed(*file, c, lo, n, *f);
        else {
          f->cap = cap;
          f->buf = pinGet(f->cap);
          readRows(*file, c, lo, n, (uint8_t*)f->buf, false);
        }
      } catch (std::exception& e) {
        ok = false;
        err = e.what();
      } catch (...) {
        ok = false;
      }
      std::lock_guard<std::mutex> g(f->mu);
      f->failed = !ok;
      f->error = err;
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

namespace {
// Queued GPU decompressions: run as one nvCOMP call (it costs ~0.4 ms however much it does) before
// kernels read the data (scanFlush), with the buffers they read released after.
struct Queued {
  std::vector<const void*> in;
  std::vector<size_t> inLen, outLen;
  std::vector<void*> out;
  std::vector<void*> freeAfter;  // VRAM (cudaFreeAsync once the decompression has run)
  std::vector<Pending> pending;  // pool buffers (back once it has run; ev set at the flush)
  size_t bytes = 0;
} tQueue;

void flushQueue() {
  if (tQueue.in.empty() && tQueue.freeAfter.empty() && tQueue.pending.empty()) return;
  gpuZstdDecompress(tQueue.in, tQueue.inLen, tQueue.out, tQueue.outLen);
  const cudaStream_t s = cudaStreamPerThread;
  for (void* p : tQueue.freeAfter) cudaFreeAsync(p, s);
  for (Pending p : tQueue.pending) {
    CK(cudaEventCreateWithFlags(&p.ev, cudaEventDisableTiming));
    CK(cudaEventRecord(p.ev, s));
    tPending.push_back(p);
  }
  tQueue = Queued{};
}

// A Fetch of the GPU path -> the batch's rows decompressed in VRAM (a scan temp). Returns the pointer
// to the decompressed blocks (prefix first for encoded columns); first = the first row in it.
void* decompressOnGpu(SegFile& file, int c, Fetch& f, int64_t& first) {
  const auto& col = file.cols[c];
  const int64_t B = col.blockRows, r0 = f.b0 * B, r1 = std::min<int64_t>((f.b1 + 1) * B, file.rows);
  const uint64_t pre = col.prefix, w = elemOf(col);
  const size_t outLen = pre + (size_t)(r1 - r0) * w + 16;
  first = r0;
  const cudaStream_t s = cudaStreamPerThread;
  uint8_t* out = nullptr;
  if (cudaMallocAsync((void**)&out, outLen, s) != cudaSuccess) {
    cudaGetLastError();
    return nullptr;
  }
  tTemps.push_back({out, 0, 0});
  tTempBytes += (int64_t)outLen;
  if (pre) {  // (through pinned memory: a copy from pageable memory would wait for the stream)
    void* hp = pinGet(pinClass(pre));
    memcpy(hp, f.prefix.data(), pre);
    CK(cudaMemcpyAsync(out, hp, pre, cudaMemcpyHostToDevice, s));
    tQueue.pending.push_back({hp, pinClass(pre), nullptr, false});
  }
  CK(cudaMemsetAsync(out + pre + (r1 - r0) * w, 0, 16, s));
  uint8_t* z = (uint8_t*)f.dbuf;
  bool zOwned = false;
  if (!z) {  // compressed blocks in pinned memory: to VRAM first
    if (cudaMallocAsync((void**)&z, std::max<uint64_t>(f.span, 1), s) != cudaSuccess) {
      cudaGetLastError();
      return nullptr;
    }
    zOwned = true;
    CK(cudaMemcpyAsync(z, f.buf, f.span, cudaMemcpyHostToDevice, s));
  }
  std::vector<const void*> in;
  std::vector<size_t> inLen, outLens;
  std::vector<void*> outs;
  const uint64_t base = col.blocks[f.b0];
  for (int64_t b = f.b0; b <= f.b1; ++b) {
    const uint64_t clen = col.blocks[b + 1] - col.blocks[b];
    const size_t rawLen = (size_t)(std::min<int64_t>((b + 1) * B, file.rows) - b * B) * w;
    uint8_t* dst = out + pre + (b - f.b0) * B * w;
    if (clen == rawLen) CK(cudaMemcpyAsync(dst, z + (col.blocks[b] - base), rawLen, cudaMemcpyDeviceToDevice, s));  // (stored as is)
    else {
      in.push_back(z + (col.blocks[b] - base));
      inLen.push_back(clen);
      outs.push_back(dst);
      outLens.push_back(rawLen);
    }
  }
  // Queued (scanFlush runs it); the compressed input goes once it has run.
  tQueue.in.insert(tQueue.in.end(), in.begin(), in.end());
  tQueue.inLen.insert(tQueue.inLen.end(), inLen.begin(), inLen.end());
  tQueue.out.insert(tQueue.out.end(), outs.begin(), outs.end());
  tQueue.outLen.insert(tQueue.outLen.end(), outLens.begin(), outLens.end());
  if (zOwned) tQueue.freeAfter.push_back(z);
  if (f.dbuf) tQueue.pending.push_back({f.dbuf, f.dcap, nullptr, true});
  else tQueue.pending.push_back({f.buf, f.cap, nullptr, false});
  f.buf = f.dbuf = nullptr;
  tQueue.bytes += outLen;
  if (tQueue.bytes >= (256u << 20)) flushQueue();
  return out;
}
}  // namespace

void* segScanColumn(SegData& d, int c, int64_t lo, int64_t n, int64_t& encOff) {
  encOff = 0;
  SegFile& file = *d.file;  // (called for columns that were cold: they have a file)
  const auto& col = file.cols[c];
  auto present = [&](void* p) -> void* {  // a column in memory: row lo of it
    if (col.enc) {
      encOff = lo;
      return p;
    }
    return (char*)p + lo * (int64_t)col.width;
  };
  if (!segCold(d, c)) return present(d.dev[c]);
  const bool whole = lo == 0 && n >= file.rows;
  auto it = tFetches.find({&d, c, lo});
  {
    std::lock_guard<std::mutex> g(d.loadMu);
    if (!d.cold[c] || revive(d, c)) return present(d.dev[c]);
    const int64_t need = (int64_t)col.len + (col.enc ? (int64_t)file.rows * col.width : 0);
    if (whole && it == tFetches.end() && g_hostBytes + need <= g_memLimit) {  // room in the cache
      loadLocked(d, c, file.rows);
      return present(d.dev[c]);
    }
  }
  // Over the budget: the rows of this batch, for this scan only (read ahead if possible).
  std::shared_ptr<Fetch> f;
  if (it != tFetches.end()) {
    f = it->second;
    tFetches.erase(it);
    waitFetch(*f);
    tFetchBytes -= (int64_t)std::max(f->cap, f->dcap);
    if (f->failed) {
      fetchFree(*f);
      throw SqlError(f->error.empty() ? "cannot read segment file " + pathOf(file.id) : f->error, 1105);
    }
  } else {
    f = std::make_shared<Fetch>();
    f->len = storedSize(file, c, lo, n);
    try {
      if (gpuPath(col, n)) readCompressed(file, c, lo, n, *f);
      else {
        f->cap = pinClass(std::max<size_t>(f->len, 1));
        f->buf = pinGet(f->cap);
        readRows(file, c, lo, n, (uint8_t*)f->buf, true);
      }
    } catch (...) {
      fetchFree(*f);
      throw;
    }
  }
  pinReap(false);
  readAhead(c);
  if (f->gpu) {
    int64_t first = 0;
    if (uint8_t* p = (uint8_t*)decompressOnGpu(file, c, *f, first)) {
      if (col.enc) {
        encOff = lo - first;
        return p;
      }
      return p + (lo - first) * (int64_t)col.width;
    }
    // (out of VRAM: decompressed on the CPU into mapped pinned memory below)
    fetchFree(*f);
    f->gpu = false;
    f->cap = pinClass(std::max<size_t>(f->len, 1));
    f->buf = pinGet(f->cap);
    readRows(file, c, lo, n, (uint8_t*)f->buf, true);
  }
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

void scanFlush() { flushQueue(); }

void releaseScanTemps() {
  flushQueue();
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
    fetchFree(*f);
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

void segRawCell(SegData& d, int c, int64_t r, uint8_t* out) {
  const SegFile& f = *d.file;
  const auto& col = f.cols[c];
  std::vector<uint8_t> buf(storedSize(f, c, r, 1));
  readRows(f, c, r, 1, buf.data(), false);
  if (!col.enc) memcpy(out, buf.data(), col.width);
  else decodeInto(SegFile::Col{col.enc, col.width, 0, 0}, buf.data(), 1, out);
}

void segRawCells(std::vector<CellRead>& reqs) {
  // Grouped by (segment, column, block of rows): each block is read and decompressed once.
  std::sort(reqs.begin(), reqs.end(), [](const CellRead& a, const CellRead& b) {
    return std::tie(a.d, a.c, a.row) < std::tie(b.d, b.c, b.row);
  });
  std::vector<std::pair<size_t, size_t>> groups;
  for (size_t i = 0; i < reqs.size();) {
    const auto& col = reqs[i].d->file->cols[reqs[i].c];
    const int64_t B = col.blockRows ? col.blockRows : 1;
    size_t j = i + 1;
    while (j < reqs.size() && reqs[j].d == reqs[i].d && reqs[j].c == reqs[i].c && reqs[j].row / B == reqs[i].row / B) ++j;
    groups.push_back({i, j});
    i = j;
  }
  std::string err;
  std::mutex errMu;
  ioParallel(groups.size(), [&](size_t g) {
    try {
      auto [i, j] = groups[g];
      const SegFile& f = *reqs[i].d->file;
      const int c = reqs[i].c;
      const auto& col = f.cols[c];
      const int64_t lo = reqs[i].row, n = reqs[j - 1].row - lo + 1;
      std::vector<uint8_t> buf(storedSize(f, c, lo, n));
      readRows(f, c, lo, n, buf.data(), false);
      const uint64_t pre = prefixOf(col.enc), w = elemOf(col);
      std::vector<uint8_t> one(pre + w + 16, 0);
      for (size_t k = i; k < j; ++k) {
        const int64_t r = reqs[k].row - lo;
        if (!col.enc) {
          memcpy(reqs[k].out, buf.data() + r * w, col.width);
          continue;
        }
        memcpy(one.data(), buf.data(), pre);
        memcpy(one.data() + pre, buf.data() + pre + r * w, w);
        decodeInto(SegFile::Col{col.enc, col.width, 0, 0}, one.data(), 1, reqs[k].out);
      }
    } catch (std::exception& ex) {
      std::lock_guard<std::mutex> lk(errMu);
      err = ex.what();
    }
  });
  if (!err.empty()) throw SqlError(err, 1105);
}

long long segRawInt(SegData& d, int c, int64_t r, bool i64) {
  if (!segCold(d, c)) {
    const uint8_t* h = d.host[c].data();
    return i64 ? (long long)((const int64_t*)h)[r] : (long long)((const int32_t*)h)[r];
  }
  uint8_t v[8];
  segRawCell(d, c, r, v);
  int64_t x;
  int32_t y;
  if (i64) return memcpy(&x, v, 8), (long long)x;
  return memcpy(&y, v, 4), (long long)y;
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

void segDropColumn(SegData& d, int c) {
  std::lock_guard<std::mutex> g(d.loadMu);
  if ((size_t)c >= d.cold.size() || d.cold[c]) return;  // only in the file (a pending eviction frees itself)
  std::vector<uint8_t>& placed = d.enc[c] ? d.encHost[c] : d.host[c];
  if (d.dev[c]) releaseColumn(d.dev[c], placed.data(), placed.size(), d.mapped[c]);
  d.devBytes = std::max<int64_t>(0, d.devBytes - (int64_t)placed.size());
  const int64_t b = colHostBytes(d, c);
  d.dev[c] = nullptr;
  d.mapped[c] = 0;
  std::vector<uint8_t>().swap(d.host[c]);
  std::vector<uint8_t>().swap(d.encHost[c]);
  d.enc[c] = 0;
  d.hostBytes -= b;
  g_hostBytes -= b;
}

}  // namespace gdb
