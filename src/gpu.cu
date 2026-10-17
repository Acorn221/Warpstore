#include "gpu.h"
#include "jit.h"
#include "kernels.cuh"

#include <cub/cub.cuh>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/gather.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <atomic>
#include <cstdio>
#include <string>

// Ahead-of-time kernels: the shared bodies with the bytecode interpreter (fallback when a query
// isn't runtime-compiled).
GDB_DEFINE_KERNELS(gdb::InterpEv, static)

namespace gdb {

#define CK(x)                                                                                 \
  do {                                                                                        \
    cudaError_t e_ = (x);                                                                     \
    if (e_ != cudaSuccess) {                                                                  \
      if (e_ == cudaErrorMemoryAllocation) { cudaGetLastError(); throw SqlError("GPU out of memory", 1041); } \
      throw SqlError(std::string("CUDA error: ") + cudaGetErrorString(e_) + " at " #x, 1105); \
    }                                                                                         \
  } while (0)

static int g_sms = 68, g_arch = 86;

int gpuSmCount() { return g_sms; }
int gpuArch() { return g_arch; }

void gpuInit() {
  CK(cudaSetDevice(0));
  cudaDeviceProp prop;
  CK(cudaGetDeviceProperties(&prop, 0));
  g_sms = prop.multiProcessorCount;
  g_arch = prop.major * 10 + prop.minor;
  // Per-thread stack: the default 1 KB overflows in aggregation kernels that call FP64 division
  // (DECIMAL scaling) on top of their accumulator arrays. Costs about stack x resident threads of VRAM.
  CK(cudaDeviceSetLimit(cudaLimitStackSize, 2048));
  cudaMemPool_t pool;
  CK(cudaDeviceGetDefaultMemPool(&pool, 0));
  uint64_t thresh = UINT64_MAX;
  CK(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &thresh));
}

std::string gpuDescribe() {
  cudaDeviceProp prop;
  CK(cudaGetDeviceProperties(&prop, 0));
  size_t fr, tot;
  CK(cudaMemGetInfo(&fr, &tot));
  char buf[256];
  snprintf(buf, sizeof buf, "%s (sm_%d%d, %d SMs, %.1f GiB free / %.1f GiB)", prop.name, prop.major, prop.minor,
           prop.multiProcessorCount, fr / 1073741824.0, tot / 1073741824.0);
  return buf;
}

void* devAlloc(size_t bytes) {
  void* p = nullptr;
  if (bytes) CK(cudaMalloc(&p, bytes));
  return p;
}
void devFree(void* p) { if (p) cudaFree(p); }

static std::atomic<int64_t> g_vramUsed{0};
static int64_t g_vramLimit = INT64_MAX;
void gpuSetVramLimit(int64_t bytes) { g_vramLimit = bytes; }
int64_t gpuVramUsed() { return g_vramUsed; }
double gpuVramFill() {
  if (g_vramLimit != INT64_MAX) return (double)g_vramUsed / (double)g_vramLimit;
  size_t freeB = 0, total = 1;
  if (cudaMemGetInfo(&freeB, &total) != cudaSuccess) return 0;
  return 1.0 - (double)freeB / (double)total;
}

void* placeColumn(void* host, size_t bytes, bool& mapped, bool hostOnly) {
  mapped = false;
  if (!bytes) return nullptr;
  if (!hostOnly && g_vramUsed + (int64_t)bytes <= g_vramLimit) {
    void* p = nullptr;
    if (cudaMalloc(&p, bytes) == cudaSuccess) {
      g_vramUsed += (int64_t)bytes;
      CK(cudaMemcpy(p, host, bytes, cudaMemcpyHostToDevice));
      return p;
    }
    cudaGetLastError();  // out of VRAM: fall through to the host tier
  }
  CK(cudaHostRegister(host, bytes, cudaHostRegisterMapped | cudaHostRegisterPortable));
  void* d = nullptr;
  CK(cudaHostGetDevicePointer(&d, host, 0));
  mapped = true;
  return d;
}

void* vramAcquire(const void* host, size_t bytes) {
  if (!bytes || g_vramUsed + (int64_t)bytes > g_vramLimit) return nullptr;
  void* p = nullptr;
  if (cudaMalloc(&p, bytes) != cudaSuccess) { cudaGetLastError(); return nullptr; }
  g_vramUsed += (int64_t)bytes;
  CK(cudaMemcpy(p, host, bytes, cudaMemcpyHostToDevice));
  return p;
}
void vramRelease(void* dev, size_t bytes) {
  if (!dev) return;
  cudaFree(dev);
  g_vramUsed -= (int64_t)bytes;
}
void* hostMap(void* host, size_t bytes) {
  if (cudaHostRegister(host, bytes, cudaHostRegisterMapped | cudaHostRegisterPortable) != cudaSuccess) { cudaGetLastError(); return nullptr; }
  void* d = nullptr;
  if (cudaHostGetDevicePointer(&d, host, 0) != cudaSuccess) { cudaGetLastError(); cudaHostUnregister(host); return nullptr; }
  return d;
}
void hostUnmap(void* host) { cudaHostUnregister(host); }
int64_t gpuVramLimit() { return g_vramLimit; }
int64_t gpuVramFree() {
  size_t freeB = 0, total = 0;
  if (cudaMemGetInfo(&freeB, &total) != cudaSuccess) return 0;
  return (int64_t)freeB;
}

void releaseColumn(void* dev, void* host, size_t bytes, bool mapped) {
  if (!dev) return;
  if (mapped) {
    cudaHostUnregister(host);
  } else {
    cudaFree(dev);
    g_vramUsed -= (int64_t)bytes;
  }
}
__global__ void gatherEncK(ColView v, int t, const int32_t* idx, int64_t m, void* dst) {
  for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < m; i += (int64_t)gridDim.x * blockDim.x) {
    const int64_t r = idx[i];
    if (t == (int)PType::F64) ((double*)dst)[i] = encDbl(v, r);
    else if (t == (int)PType::I32) ((int32_t*)dst)[i] = (int32_t)encInt(v, r);
    else ((int64_t*)dst)[i] = encInt(v, r);
  }
}
void gpuGatherEnc(const ColView& v, PType t, const int32_t* idx, int64_t m, void* dst) {
  if (m <= 0) return;
  ColView w = v;
  w.t = t == PType::I32 ? 3 : t == PType::I64 ? 4 : 2;  // keep NULL sentinels as stored
  gatherEncK<<<(unsigned)std::min<int64_t>((m + 255) / 256, 4096), 256>>>(w, (int)t, idx, m, dst);
  CK(cudaGetLastError());
}

void* tmpAlloc(size_t bytes) {
  void* p = nullptr;
  if (bytes) CK(cudaMallocAsync(&p, bytes, 0));
  return p;
}
void tmpFree(void* p) { if (p) cudaFreeAsync(p, 0); }
void h2d(void* dst, const void* src, size_t b) { if (b) CK(cudaMemcpy(dst, src, b, cudaMemcpyHostToDevice)); }
void d2h(void* dst, const void* src, size_t b) { if (b) CK(cudaMemcpy(dst, src, b, cudaMemcpyDeviceToHost)); }
void d2d(void* dst, const void* src, size_t b) { if (b) CK(cudaMemcpy(dst, src, b, cudaMemcpyDeviceToDevice)); }
void gpuSync() { CK(cudaStreamSynchronize(0)); }

static int gridFor(int64_t n, int block = 256, int perSm = 8) {
  int64_t g = (n + block - 1) / block;
  int64_t cap = (int64_t)g_sms * perSm;
  if (g > cap) g = cap;
  return (int)(g < 1 ? 1 : g);
}

__device__ __forceinline__ long long keyAt(const ColView& c, int64_t r) {
  switch (c.t) {
    case 0: case 3: return ((const int32_t*)c.p)[r];
    case 1: case 4: return ((const int64_t*)c.p)[r];
    default: return (long long)((const double*)c.p)[r];
  }
}
// NULL join keys never match anything (SQL equality with NULL is unknown).
__device__ __forceinline__ bool keyNull(const ColView& c, int64_t r) {
  switch (c.t) {
    case 0: case 3: return ((const int32_t*)c.p)[r] == kNullI32;
    case 1: case 4: return ((const int64_t*)c.p)[r] == kNullI64;
    default: { double v = ((const double*)c.p)[r]; return v != v; }
  }
}

// ------------------------------------------------------------------ aggregation

__global__ void initTable(GTable t, int nkeys, int naggs, AggPlan plan) {
  for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i <= (int64_t)t.mask; i += (int64_t)gridDim.x * blockDim.x) {
    t.keys[i] = EMPTY;
    for (int a = 0; a < naggs; ++a) t.acc[i * naggs + a] = plan.aggBits[a] ? bitsIdentity(plan.ops[a]) : aggIdentity(plan.ops[a]);
  }
}

// For b = 1..nb-1: how many selected (ascending) indices are below boff[b] (binary search).
__global__ void batchBounds(const int32_t* idx, const int64_t* total, const int64_t* boff, int nb, int64_t* out) {
  for (int b = 1 + threadIdx.x; b < nb; b += blockDim.x) {
    int64_t lo = 0, hi = *total;
    while (lo < hi) {
      const int64_t mid = (lo + hi) >> 1;
      if (idx[mid] < boff[b]) lo = mid + 1;
      else hi = mid;
    }
    out[b - 1] = lo;
  }
}

// Pinned staging memory for small host <-> device transfers (per thread, grows as needed). An event
// marks the last async copy out of it, so it is only waited for when the buffer is reused.
thread_local cudaEvent_t tStageEv = nullptr;
static uint8_t* pinnedStage(size_t bytes) {
  thread_local uint8_t* buf = nullptr;
  thread_local size_t cap = 0;
  if (tStageEv) CK(cudaEventSynchronize(tStageEv));
  else CK(cudaEventCreateWithFlags(&tStageEv, cudaEventDisableTiming));
  if (bytes > cap) {
    if (buf) cudaFreeHost(buf);
    cap = std::max<size_t>(bytes, 64 << 10);
    CK(cudaHostAlloc((void**)&buf, cap, cudaHostAllocDefault));
  }
  return buf;
}
static double hostIdentity(uint8_t op) { return op == AGG_MIN ? INFINITY : op == AGG_MAX ? -INFINITY : 0.0; }

__global__ void initNoKeys(double* out, AggPlan plan) {
  if (threadIdx.x < plan.naggs) out[threadIdx.x] = aggIdentity(plan.ops[threadIdx.x]);
}

// Dense arrays -> the hash-table layout (occupied slot = group with rows), so compaction and the
// post-aggregation path are shared.
__global__ void denseFinalize(DenseSpec ds, AggPlan plan, int nkeys, int naggs, unsigned long long* keys, double* kv, const double* acc) {
  for (int64_t g = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; g < ds.G; g += (int64_t)gridDim.x * blockDim.x) {
    if (acc[g * naggs] > 0) {
      keys[g] = (unsigned long long)g;
      for (int k = 0; k < nkeys; ++k) {
        const long long v = ds.kmin[k] + (g / ds.stride[k]) % ds.range[k];
        kv[g * nkeys + k] = plan.keyInt[k] ? asBits(v) : (double)v;
      }
    } else {
      keys[g] = EMPTY;
    }
  }
}

GpuAggregator::GpuAggregator(const AggPlan& plan, int64_t capacity, const DenseSpec* dense) : plan_(plan) {
  if (dense && dense->G > 0 && plan.nkeys) dense_ = *dense;
  int64_t c = 1;
  while (c < capacity) c <<= 1;
  cap_ = plan.nkeys ? (dense_.G ? dense_.G : c) : 1;
  if (plan.nkeys == 0) return;  // accumulators are allocated and initialized with the batches in run()
  CK(cudaMallocAsync(&flags_, sizeof(int) * 2, 0));
  CK(cudaMemsetAsync(flags_, 0, sizeof(int) * 2, 0));
  CK(cudaMallocAsync(&count_, sizeof(unsigned long long), 0));
  CK(cudaMemsetAsync(count_, 0, sizeof(unsigned long long), 0));
  CK(cudaMallocAsync(&gkeys_, sizeof(unsigned long long) * cap_, 0));
  CK(cudaMallocAsync(&gkv_, sizeof(double) * cap_ * plan.nkeys, 0));
  CK(cudaMallocAsync(&gacc_, sizeof(double) * cap_ * std::max(plan.naggs, 1), 0));
  GTable t{gkeys_, gkv_, gacc_, (unsigned long long)(cap_ - 1), flags_, count_};
  initTable<<<gridFor(cap_), 256>>>(t, plan.nkeys, plan.naggs, plan_);
  CK(cudaGetLastError());
}

GpuAggregator::~GpuAggregator() {
  if (block_) {
    cudaFreeAsync(block_, 0);
    gacc_ = nullptr;
  }
  for (void* p : {(void*)gkeys_, (void*)gkv_, (void*)gacc_, (void*)flags_, (void*)count_})
    if (p) cudaFreeAsync(p, 0);
}

void GpuAggregator::addBatch(const ExprEnv& env, int ncols, int64_t n, const uint32_t* del) {
  if (n <= 0) return;
  ncols_ = ncols;
  bn_.push_back(n);
  bdel_.push_back(del);
  for (int i = 0; i < ncols; ++i) bcols_.push_back(env.cols[i]);
}

int64_t GpuAggregator::rowsIn() const {
  int64_t r = 0;
  for (int64_t n : bn_) r += n;
  return r;
}

std::vector<uint8_t> GpuAggregator::sumBitsSlots() const {
  std::vector<uint8_t> f(plan_.naggs, 0);
  if (plan_.nkeys)
    for (int a = 0; a < plan_.naggs; ++a) f[a] = plan_.aggBits[a];
  else
    for (int a = 0; a < plan_.naggs; ++a)
      f[a] = (plan_.ops[a] == AGG_SUM || plan_.ops[a] == AGG_MIN || plan_.ops[a] == AGG_MAX) && plan_.aggInt[a];
  return f;
}

void GpuAggregator::run(const ExprEnv& env, const JitProgram* jit) {
  if (!jit && (plan_.naggs > kInterpAggs || plan_.nkeys > kInterpKeys))
    throw SqlError("this aggregation needs a compiled plan (more than 16 aggregates or 4 GROUP BY keys); codegen failed or is off", 1235);
  const size_t nb = bn_.size();
  if (nb == 0 && plan_.nkeys) return;
  // One descriptor for all batches: [n x nb][del x nb][cols x nb*ncols], uploaded asynchronously from
  // pinned memory. Without GROUP BY the accumulators (initialized here) share the allocation.
  const size_t accBytes = plan_.nkeys == 0 ? sizeof(double) * kMaxAggs * 2 : 0;  // (+ high words of exact sums)
  const size_t colBytes = 16 * nb + bcols_.size() * sizeof(ColView);
  const size_t envOff = (colBytes + 15) & ~(size_t)15;  // interpreted: the ExprEnv follows (see AggPlan::genv)
  const size_t blobBytes = jit ? colBytes : envOff + sizeof(ExprEnv);
  uint8_t* stage = pinnedStage(accBytes + blobBytes);
  if (accBytes)
    for (int a = 0; a < kMaxAggs; ++a) {
      double v = a < plan_.naggs ? hostIdentity(plan_.ops[a]) : 0.0;
      if (a < plan_.naggs && plan_.aggInt[a] && (plan_.ops[a] == AGG_MIN || plan_.ops[a] == AGG_MAX)) {  // exact int64 min / max
        const long long id = plan_.ops[a] == AGG_MIN ? LLONG_MAX : LLONG_MIN;
        memcpy(&v, &id, 8);
      }
      memcpy(stage + 8 * a, &v, 8);
      memset(stage + 8 * (kMaxAggs + a), 0, 8);
    }
  uint8_t* blob = stage + accBytes;
  memcpy(blob, bn_.data(), 8 * nb);
  memcpy(blob + 8 * nb, bdel_.data(), 8 * nb);
  if (!bcols_.empty()) memcpy(blob + 16 * nb, bcols_.data(), bcols_.size() * sizeof(ColView));
  if (!jit) memcpy(blob + envOff, &env, sizeof(ExprEnv));
  DevVec<uint8_t> d;
  uint8_t* dp;
  if (accBytes) {
    CK(cudaMallocAsync((void**)&block_, accBytes + blobBytes, 0));
    gacc_ = (double*)block_;
    dp = block_ + accBytes;
    CK(cudaMemcpyAsync(block_, stage, accBytes + blobBytes, cudaMemcpyHostToDevice, 0));
  } else {
    d = DevVec<uint8_t>((int64_t)blobBytes);
    dp = d.p;
    CK(cudaMemcpyAsync(d.p, stage, blobBytes, cudaMemcpyHostToDevice, 0));
  }
  CK(cudaEventRecord(tStageEv, 0));  // the staging buffer is free again once this copy ran
  if (nb == 0) return;
  BatchSet bs;
  bs.nb = (int)nb;
  bs.ncols = ncols_;
  bs.n = (const int64_t*)dp;
  bs.del = (const uint32_t* const*)(dp + 8 * nb);
  bs.cols = (const ColView*)(dp + 16 * nb);
  plan_.genv = jit ? nullptr : (const ExprEnv*)(dp + envOff);
  const unsigned block = 256;
  if (plan_.nkeys == 0) {
    unsigned grid = (unsigned)g_sms * 8;
    void* args[] = {(void*)&env, &plan_, &bs, &gacc_};
    if (jit && jit->fn[JF_NOKEYS]) jitLaunch(jit, JF_NOKEYS, grid, block, 0, args);
    else gdb_aggNoKeys<<<grid, block>>>(env, plan_, bs, gacc_);
  } else if (dense_.G) {
    unsigned blk = block;
    size_t shmem = (size_t)dense_.copies * dense_.copyBytes;
    if (dense_.priv) {  // G x naggs slots of 8 bytes per thread
      const size_t S = (size_t)dense_.G * plan_.naggs;
      blk = S <= 24 ? 256 : 128;
      shmem = S * 8 * blk;
    }
    unsigned perSm = shmem ? (unsigned)std::max<size_t>(1, std::min<size_t>(8, (99 * 1024) / (shmem + 1024))) : 8;
    unsigned grid = (unsigned)g_sms * perSm;
    void* args[] = {(void*)&env, &plan_, &dense_, &bs, &gacc_};
    if (jit && jit->fn[JF_DENSE]) jitLaunch(jit, JF_DENSE, grid, blk, shmem, args);
    else {
      if (shmem > 32 * 1024) cudaFuncSetAttribute((const void*)gdb_aggDense, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem);
      gdb_aggDense<<<grid, blk, shmem>>>(env, plan_, dense_, bs, gacc_);
    }
  } else {
    const size_t perSlot = 8 + 8 * plan_.nkeys + 8 * plan_.naggs;
    const int warps = block / 32;
    int W = 128;
    while (W > 8 && warps * W * perSlot > 40 * 1024) W >>= 1;
    const size_t shmem = warps * W * perSlot;
    GTable t{gkeys_, gkv_, gacc_, (unsigned long long)(cap_ - 1), flags_, count_};
    unsigned grid = (unsigned)g_sms * 4;
    void* args[] = {(void*)&env, &plan_, &bs, &t, &W};
    if (jit && jit->fn[JF_GROUPED]) jitLaunch(jit, JF_GROUPED, grid, block, shmem, args);
    else {
      if (shmem > 32 * 1024) cudaFuncSetAttribute((const void*)gdb_aggGrouped, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem);
      gdb_aggGrouped<<<grid, block, shmem>>>(env, plan_, bs, t, W);
    }
  }
  CK(cudaGetLastError());
}

void GpuAggregator::finalizeDense() {
  if (!dense_.G || finalized_) return;
  finalized_ = true;
  denseFinalize<<<gridFor(dense_.G), 256>>>(dense_, plan_, plan_.nkeys, plan_.naggs, gkeys_, gkv_, gacc_);
  CK(cudaGetLastError());
}

bool GpuAggregator::overflowed() {
  if (plan_.nkeys == 0) return false;  // fetch() synchronizes
  if (dense_.G) {
    gpuSync();
    return false;
  }
  int f[2];
  unsigned long long c;
  d2h(f, flags_, sizeof f);
  d2h(&c, count_, sizeof c);
  return f[0] != 0;
}

__global__ void occupiedFlags(const unsigned long long* keys, int64_t n, uint8_t* flags) {
  for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < n; i += (int64_t)gridDim.x * blockDim.x)
    flags[i] = keys[i] != EMPTY;
}

__global__ void rowGather(const double* src, int stride, const int32_t* idx, int64_t m, double* dst) {
  for (int64_t j = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; j < m * stride; j += (int64_t)gridDim.x * blockDim.x)
    dst[j] = src[(int64_t)idx[j / stride] * stride + j % stride];
}

int64_t GpuAggregator::fetch(std::vector<double>& keys, std::vector<double>& accs) {
  if (plan_.nkeys == 0) {
    accs.resize(plan_.naggs);
    keys.clear();
    hi_.assign(plan_.naggs, 0);
    if (!gacc_) {  // no batches: identities
      for (int a = 0; a < plan_.naggs; ++a) accs[a] = hostIdentity(plan_.ops[a]);
      return 1;
    }
    uint8_t* stage = pinnedStage(sizeof(double) * kMaxAggs * 2);
    CK(cudaMemcpyAsync(stage, gacc_, sizeof(double) * kMaxAggs * 2, cudaMemcpyDeviceToHost, 0));
    CK(cudaStreamSynchronize(0));
    memcpy(accs.data(), stage, sizeof(double) * plan_.naggs);
    memcpy(hi_.data(), stage + 8 * kMaxAggs, 8 * plan_.naggs);
    return 1;
  }
  finalizeDense();
  // Compact occupied slots on the GPU so only real groups cross PCIe.
  DevVec<uint8_t> flags(cap_);
  occupiedFlags<<<gridFor(cap_), 256>>>(gkeys_, cap_, flags.p);
  DevVec<int32_t> idx(cap_);
  DevVec<int64_t> cnt(1);
  cub::CountingInputIterator<int32_t> it(0);
  size_t tb = 0;
  CK(cub::DeviceSelect::Flagged(nullptr, tb, it, flags.p, idx.p, cnt.p, cap_, 0));
  DevVec<uint8_t> tmp((int64_t)tb);
  CK(cub::DeviceSelect::Flagged(tmp.p, tb, it, flags.p, idx.p, cnt.p, cap_, 0));
  int64_t g;
  d2h(&g, cnt.p, sizeof g);
  DevVec<double> kv(std::max<int64_t>(g * plan_.nkeys, 1)), ac(std::max<int64_t>(g * plan_.naggs, 1));
  if (g) {
    rowGather<<<gridFor(g * plan_.nkeys), 256>>>(gkv_, plan_.nkeys, idx.p, g, kv.p);
    rowGather<<<gridFor(g * plan_.naggs), 256>>>(gacc_, plan_.naggs, idx.p, g, ac.p);
    CK(cudaGetLastError());
  }
  keys = kv.download(g * plan_.nkeys);
  accs = ac.download(g * plan_.naggs);
  return g;
}

__global__ void colGather(const double* src, int stride, const int32_t* idx, int64_t m, double* dst) {
  for (int64_t j = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; j < m * stride; j += (int64_t)gridDim.x * blockDim.x) {
    const int64_t c = j / m, i = j % m;
    dst[j] = src[(int64_t)idx[i] * stride + c];
  }
}

int64_t GpuAggregator::compact(DevVec<double>& keys, DevVec<double>& accs) {
  finalizeDense();
  DevVec<uint8_t> flags(cap_);
  occupiedFlags<<<gridFor(cap_), 256>>>(gkeys_, cap_, flags.p);
  DevVec<int32_t> idx(cap_);
  DevVec<int64_t> cnt(1);
  cub::CountingInputIterator<int32_t> it(0);
  size_t tb = 0;
  CK(cub::DeviceSelect::Flagged(nullptr, tb, it, flags.p, idx.p, cnt.p, cap_, 0));
  DevVec<uint8_t> tmp((int64_t)tb);
  CK(cub::DeviceSelect::Flagged(tmp.p, tb, it, flags.p, idx.p, cnt.p, cap_, 0));
  int64_t g;
  d2h(&g, cnt.p, sizeof g);
  keys = DevVec<double>(std::max<int64_t>(g * plan_.nkeys, 1));
  accs = DevVec<double>(std::max<int64_t>(g * plan_.naggs, 1));
  if (g) {
    colGather<<<gridFor(g * plan_.nkeys), 256>>>(gkv_, plan_.nkeys, idx.p, g, keys.p);
    colGather<<<gridFor(g * plan_.naggs), 256>>>(gacc_, plan_.naggs, idx.p, g, accs.p);
    CK(cudaGetLastError());
  }
  return g;
}

int64_t GpuAggregator::groupCount() {
  unsigned long long c;
  d2h(&c, count_, sizeof c);
  return (int64_t)c;
}

// ------------------------------------------------------------------ selection / projection

// The ExprEnv in device memory, for interpreted kernels (AggPlan::genv).
static DevVec<uint8_t> envOnDevice(const ExprEnv& env) {
  DevVec<uint8_t> d((int64_t)sizeof(ExprEnv));
  h2d(d.p, &env, sizeof(ExprEnv));
  return d;
}

int64_t gpuSelect(const ExprEnv& env, bool hasWhere, Prog where, int64_t n, const uint32_t* del, int32_t* outIdx,
                  const JitProgram* jit) {
  if (n <= 0) return 0;
  DevVec<uint8_t> flags(n);
  AggPlan fp;
  fp.hasWhere = hasWhere;
  fp.where = where;
  unsigned grid = (unsigned)gridFor(n);
  if (jit && jit->fn[JF_FLAG]) {
    uint8_t* fl = flags.p;
    void* args[] = {(void*)&env, &fp, &n, &del, &fl};
    jitLaunch(jit, JF_FLAG, grid, 256, 0, args);
  } else {
    DevVec<uint8_t> genv = envOnDevice(env);
    fp.genv = (const ExprEnv*)genv.p;
    gdb_flag<<<grid, 256>>>(env, fp, n, del, flags.p);
    CK(cudaStreamSynchronize(0));  // (genv is freed at scope end)
  }
  CK(cudaGetLastError());
  DevVec<int64_t> cnt(1);
  cub::CountingInputIterator<int32_t> it(0);
  size_t tb = 0;
  CK(cub::DeviceSelect::Flagged(nullptr, tb, it, flags.p, outIdx, cnt.p, n, 0));
  DevVec<uint8_t> tmp((int64_t)tb);
  CK(cub::DeviceSelect::Flagged(tmp.p, tb, it, flags.p, outIdx, cnt.p, n, 0));
  int64_t c;
  d2h(&c, cnt.p, sizeof c);
  return c;
}

int64_t gpuSelectMulti(const ExprEnv& env, bool hasWhere, Prog where, const std::vector<int64_t>& bn,
                       const std::vector<const uint32_t*>& bdel, const std::vector<ColView>& bcols, int ncols,
                       DevVec<int32_t>& outIdx, std::vector<int64_t>& perBatch, const JitProgram* jit) {
  const size_t nb = bn.size();
  perBatch.assign(nb, 0);
  std::vector<int64_t> boff(nb + 1, 0);
  for (size_t b = 0; b < nb; ++b) boff[b + 1] = boff[b] + bn[b];
  const int64_t total = boff[nb];
  if (total <= 0) return 0;
  // Descriptor: [n x nb][del x nb][cols x nb*ncols][boff x nb+1], one pinned upload.
  const size_t colBytes = bcols.size() * sizeof(ColView);
  const size_t bytes = 16 * nb + colBytes + 8 * (nb + 1);
  uint8_t* stage = pinnedStage(bytes);
  memcpy(stage, bn.data(), 8 * nb);
  memcpy(stage + 8 * nb, bdel.data(), 8 * nb);
  if (colBytes) memcpy(stage + 16 * nb, bcols.data(), colBytes);
  memcpy(stage + 16 * nb + colBytes, boff.data(), 8 * (nb + 1));
  DevVec<uint8_t> d((int64_t)bytes);
  CK(cudaMemcpyAsync(d.p, stage, bytes, cudaMemcpyHostToDevice, 0));
  CK(cudaEventRecord(tStageEv, 0));
  BatchSet bs;
  bs.nb = (int)nb;
  bs.ncols = ncols;
  bs.n = (const int64_t*)d.p;
  bs.del = (const uint32_t* const*)(d.p + 8 * nb);
  bs.cols = (const ColView*)(d.p + 16 * nb);
  const int64_t* dboff = (const int64_t*)(d.p + 16 * nb + colBytes);
  DevVec<uint8_t> flags(total);
  AggPlan fp;
  fp.hasWhere = hasWhere;
  fp.where = where;
  const unsigned grid = (unsigned)gridFor(total);
  uint8_t* fl = flags.p;
  if (jit && jit->fn[JF_FLAGM]) {
    void* args[] = {(void*)&env, &fp, &bs, (void*)&dboff, &fl};
    jitLaunch(jit, JF_FLAGM, grid, 256, 0, args);
  } else {
    DevVec<uint8_t> genv = envOnDevice(env);
    fp.genv = (const ExprEnv*)genv.p;
    gdb_flagMulti<<<grid, 256>>>(env, fp, bs, dboff, fl);
    CK(cudaStreamSynchronize(0));
  }
  CK(cudaGetLastError());
  outIdx = DevVec<int32_t>(total);
  // [total selected][count of selected indices below boff[b] for b = 1..nb]
  DevVec<int64_t> cnt((int64_t)nb + 1);
  cub::CountingInputIterator<int32_t> it(0);
  size_t tb = 0;
  CK(cub::DeviceSelect::Flagged(nullptr, tb, it, flags.p, outIdx.p, cnt.p, total, 0));
  DevVec<uint8_t> tmp((int64_t)tb);
  CK(cub::DeviceSelect::Flagged(tmp.p, tb, it, flags.p, outIdx.p, cnt.p, total, 0));
  if (nb > 1) batchBounds<<<1, 32>>>(outIdx.p, cnt.p, dboff, (int)nb, cnt.p + 1);
  std::vector<int64_t> c = cnt.download();
  const int64_t m = c[0];
  int64_t prev = 0;
  for (size_t b = 0; b < nb; ++b) {
    const int64_t upto = b + 1 < nb ? c[b + 1] : m;
    perBatch[b] = upto - prev;
    prev = upto;
  }
  return m;
}

__global__ void gatherMultiK(const ColView* src, const int64_t* boff, int nb, int t, int esize, const int32_t* idx,
                             int64_t m, uint8_t* dst) {
  for (int64_t j = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; j < m; j += (int64_t)gridDim.x * blockDim.x) {
    const int64_t g = idx[j];
    int lo = 0, hi = nb - 1;  // last batch with boff <= g
    while (lo < hi) {
      const int mid = (lo + hi + 1) >> 1;
      if (boff[mid] <= g) lo = mid;
      else hi = mid - 1;
    }
    const ColView& v = src[lo];
    const int64_t r = g - boff[lo];
    if (v.enc) {
      if (t == (int)PType::F64) ((double*)dst)[j] = encDbl(v, r);
      else if (t == (int)PType::I32) ((int32_t*)dst)[j] = (int32_t)encInt(v, r);
      else ((int64_t*)dst)[j] = encInt(v, r);
    } else if (esize == 4) {
      ((int32_t*)dst)[j] = ((const int32_t*)v.p)[r];
    } else if (esize == 8) {
      ((int64_t*)dst)[j] = ((const int64_t*)v.p)[r];
    } else {
      const uint8_t* s = (const uint8_t*)v.p + r * esize;
      for (int k = 0; k < esize; ++k) dst[j * esize + k] = s[k];
    }
  }
}

void gpuGatherMulti(const std::vector<ColView>& src, const std::vector<int64_t>& boff, PType t, int esize,
                    const int32_t* idx, int64_t m, void* dst) {
  if (m <= 0) return;
  const size_t nb = src.size();
  std::vector<ColView> vs = src;
  for (auto& v : vs) v.t = t == PType::I32 ? 3 : t == PType::I64 ? 4 : 2;  // encoded: keep NULL sentinels as stored
  const size_t bytes = nb * sizeof(ColView) + 8 * nb;
  uint8_t* stage = pinnedStage(bytes);
  memcpy(stage, vs.data(), nb * sizeof(ColView));
  memcpy(stage + nb * sizeof(ColView), boff.data(), 8 * nb);
  DevVec<uint8_t> d((int64_t)bytes);
  CK(cudaMemcpyAsync(d.p, stage, bytes, cudaMemcpyHostToDevice, 0));
  CK(cudaEventRecord(tStageEv, 0));
  gatherMultiK<<<(unsigned)std::min<int64_t>((m + 255) / 256, 8192), 256>>>(
      (const ColView*)d.p, (const int64_t*)(d.p + nb * sizeof(ColView)), (int)nb, (int)t, esize, idx, m, (uint8_t*)dst);
  CK(cudaGetLastError());
}

__global__ void remapCodesK(const int32_t* in, int64_t n, const int32_t* map, int64_t mapN, int32_t* out) {
  for (int64_t j = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; j < n; j += (int64_t)gridDim.x * blockDim.x) {
    const int32_t c = in[j];
    out[j] = c == kNullI32 ? kNullI32 : (c >= 0 && c < mapN) ? map[c] : -2;
  }
}
void gpuRemapCodes(const int32_t* in, int64_t n, const int32_t* map, int64_t mapN, int32_t* out) {
  if (n <= 0) return;
  remapCodesK<<<gridFor(n), 256>>>(in, n, map, mapN, out);
  CK(cudaGetLastError());
}

__global__ void bitsToDoubleK(const double* src, int64_t n, double* dst) {
  for (int64_t j = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; j < n; j += (int64_t)gridDim.x * blockDim.x)
    dst[j] = (double)bitsOf(src[j]);
}
void gpuBitsToDouble(const double* src, int64_t n, double* dst) {
  if (n <= 0) return;
  bitsToDoubleK<<<gridFor(n), 256>>>(src, n, dst);
  CK(cudaGetLastError());
}

// Exact int64 evaluation of integer columns, integer constants, + - * and negation (NULLs propagate,
// overflow flags env.ovf); false for anything else (then the double interpreter).
__device__ bool evalIntExact(const ExprEnv& env, Prog p, int64_t r, long long& out) {
  long long st[16];
  bool nul[16];
  int sp = 0;
  for (int i = 0; i < p.len; ++i) {
    const Instr in = env.code[p.start + i];
    switch (in.op) {
      case OP_COL: {
        const ColView& c = env.cols[in.arg];
        if (c.t == 2 || c.t == 5 || c.t == 6 || sp >= 16) return false;
        const long long v = colInt(c, r);
        st[sp] = v;
        nul[sp++] = (c.t == 3 || c.t == 0) ? v == (long long)kNullI32 && c.t == 3 : c.t == 4 && v == kNullI64;
        break;
      }
      case OP_CONST:
        if (!in.isInt || sp >= 16) return false;
        st[sp] = bitsOf(env.consts[in.arg]);
        nul[sp++] = false;
        break;
      case OP_ADD: case OP_SUB: case OP_MUL: {
        if (sp < 2) return false;
        --sp;
        const bool n = nul[sp - 1] || nul[sp];
        if (!n)
          st[sp - 1] = in.op == OP_ADD ? iaddc(st[sp - 1], st[sp], env.ovf) : in.op == OP_SUB ? isubc(st[sp - 1], st[sp], env.ovf)
                                                                              : imulc(st[sp - 1], st[sp], env.ovf);
        nul[sp - 1] = n;
        break;
      }
      case OP_NEG:
        if (sp < 1) return false;
        if (!nul[sp - 1]) st[sp - 1] = isubc(0, st[sp - 1], env.ovf);
        break;
      default: return false;
    }
  }
  if (sp != 1) return false;
  out = nul[0] ? kNullI64 : st[0];
  return true;
}

__global__ void evalAtIntKernel(const ExprEnv env, Prog prog, const int32_t* idx, int64_t m, int64_t* out) {
  for (int64_t j = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; j < m; j += (int64_t)gridDim.x * blockDim.x) {
    const int64_t r = idx ? idx[j] : j;
    const Instr in = env.code[prog.start];
    long long iv;
    if (prog.len == 1 && in.op == OP_COL && env.cols[in.arg].t != 2) {
      const long long v = colInt(env.cols[in.arg], r);
      out[j] = v == (long long)kNullI32 && (env.cols[in.arg].t == 0 || env.cols[in.arg].t == 3) ? kNullI64 : v;
    } else if (evalIntExact(env, prog, r, iv)) {
      out[j] = iv;
    } else {
      const double d = evalProg(env, prog, r);
      out[j] = d != d ? kNullI64 : llround(d);
    }
  }
}
void gpuEvalAtInt(const ExprEnv& env, Prog prog, const int32_t* idx, int64_t m, int64_t* out) {
  if (m <= 0) return;
  evalAtIntKernel<<<gridFor(m), 256>>>(env, prog, idx, m, out);
  CK(cudaGetLastError());
}

__global__ void distinctBitsKernel(const ExprEnv env, Prog prog, const int32_t* idx, int64_t m, int64_t base, int64_t span,
                                   unsigned long long* bits) {
  for (int64_t j = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; j < m; j += (int64_t)gridDim.x * blockDim.x) {
    const int64_t r = idx ? idx[j] : j;
    long long v;
    if (!evalIntExact(env, prog, r, v) || v == kNullI64) continue;
    const int64_t k = v - base;
    if (k < 0 || k >= span) continue;
    const unsigned long long bit = 1ull << (k & 63);
    if (!(bits[k >> 6] & bit)) atomicOr(&bits[k >> 6], bit);
  }
}
std::vector<int64_t> gpuDistinctInts(const ExprEnv& env, Prog prog, const int32_t* idx, int64_t m, int64_t base, int64_t span) {
  std::vector<int64_t> out;
  if (m <= 0 || span <= 0) return out;
  const int64_t words = (span + 63) / 64;
  DevVec<unsigned long long> bits(words);
  CK(cudaMemsetAsync(bits.p, 0, words * 8, 0));
  distinctBitsKernel<<<gridFor(m), 256>>>(env, prog, idx, m, base, span, bits.p);
  CK(cudaGetLastError());
  std::vector<unsigned long long> h = bits.download(words);
  for (int64_t w = 0; w < words; ++w)
    for (unsigned long long b = h[w]; b; b &= b - 1) out.push_back(base + w * 64 + __builtin_ctzll(b));
  return out;
}

__global__ void tempColKernel(const double* keys, int nk, const double* accs, int na, int64_t g, TempColSpec s, void* out,
                              long long* mnmx) {
  // (compacted results are column-major: slot j of group r at [j * g + r])
  for (int64_t r = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; r < g; r += (int64_t)gridDim.x * blockDim.x) {
    auto A = [&](int j) { return accs[(int64_t)j * g + r]; };
    bool nul = s.src == 1 && s.cnt >= 0 && A(s.cnt) == 0;
    bool isInt = false;
    long long iv = 0;
    double dv = 0;
    if (s.src == 0) {
      const double k = keys[(int64_t)s.idx * g + r];
      if (s.keyInt) {
        iv = bitsOf(k);
        isInt = true;
        nul = iv == kNullI64 || iv == (long long)kNullI32;
      } else {
        dv = k;
        nul = k != k;
      }
    } else if (!nul) {
      switch (s.kind) {
        case 1: iv = bitsOf(A(s.idx)); isInt = true; break;
        case 2: iv = bitsOf(A(s.hi)) * 4294967296ll + bitsOf(A(s.idx)); isInt = true; break;
        case 3: iv = (long long)A(s.idx); isInt = true; break;
        case 4: {
          const double sum = s.sumBits ? (s.hi >= 0 ? (double)bitsOf(A(s.hi)) * 4294967296.0 + (double)bitsOf(A(s.idx)) : (double)bitsOf(A(s.idx)))
                                       : A(s.idx);
          dv = sum / s.div / A(s.cnt);
          break;
        }
        default: dv = A(s.idx) / s.div; nul = dv != dv;
      }
    }
    if (s.mul != 1 && !nul) {
      dv = (isInt ? (double)iv : dv) * s.mul;
      isInt = false;
    }
    if (s.out == 2) {
      ((double*)out)[r] = nul ? nullv() : isInt ? (double)iv : dv;
      continue;
    }
    long long v = nul ? 0 : isInt ? iv : llround(dv);
    if (!nul) {
      atomicMin(&mnmx[0], v);
      atomicMax(&mnmx[1], v);
    }
    if (s.out == 1) ((long long*)out)[r] = nul ? kNullI64 : v;
    else ((int32_t*)out)[r] = nul ? kNullI32 : (int32_t)v;
  }
}
void* gpuTempColumn(const double* keys, int nk, const double* accs, int na, int64_t g, const TempColSpec& s, double& mn, double& mx) {
  const size_t w = s.out == 0 ? 4 : 8;
  void* out = nullptr;
  CK(cudaMalloc(&out, std::max<size_t>(1, (size_t)g * w)));
  DevVec<long long> mm(2);
  const long long init[2] = {LLONG_MAX, LLONG_MIN};
  h2d(mm.p, init, sizeof init);
  if (g > 0) tempColKernel<<<gridFor(g), 256>>>(keys, nk, accs, na, g, s, out, mm.p);
  CK(cudaGetLastError());
  long long h[2];
  d2h(h, mm.p, sizeof h);
  mn = s.out == 2 || h[0] > h[1] ? -INFINITY : (double)h[0];
  mx = s.out == 2 || h[0] > h[1] ? INFINITY : (double)h[1];
  return out;
}

__global__ void evalAtKernel(const ExprEnv env, Prog prog, const int32_t* idx, int64_t m, double* out) {
  for (int64_t j = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; j < m; j += (int64_t)gridDim.x * blockDim.x)
    out[j] = evalProg(env, prog, idx ? idx[j] : j);
}

void gpuEvalAt(const ExprEnv& env, Prog prog, const int32_t* idx, int64_t m, double* out) {
  if (m <= 0) return;
  evalAtKernel<<<gridFor(m), 256>>>(env, prog, idx, m, out);
  CK(cudaGetLastError());
}

template <class T>
__global__ void gatherKernel(const T* src, const int32_t* idx, int64_t m, T* dst) {
  for (int64_t j = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; j < m; j += (int64_t)gridDim.x * blockDim.x)
    dst[j] = src[idx[j]];
}
// Index -1 (outer join, no match) produces NULL.
template <class T>
__global__ void gatherNullKernel(const T* src, const int32_t* idx, int64_t m, T* dst, T nullValue) {
  for (int64_t j = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; j < m; j += (int64_t)gridDim.x * blockDim.x)
    dst[j] = idx[j] < 0 ? nullValue : src[idx[j]];
}

__global__ void gatherWordsKernel(const uint32_t* src, int words, const int32_t* idx, int64_t m, uint32_t* dst, bool nulls) {
  const int64_t n = m * words;
  for (int64_t j = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; j < n; j += (int64_t)gridDim.x * blockDim.x) {
    const int64_t r = j / words, w = j % words;
    const int32_t i = idx[r];
    dst[j] = i < 0 ? (nulls ? 0x7fc00000u : 0u) : src[(int64_t)i * words + w];  // float NaN marks a NULL vector
  }
}

void gpuGatherBytes(const void* src, int esize, const int32_t* idx, int64_t m, void* dst, bool nulls) {
  if (m <= 0) return;
  gatherWordsKernel<<<gridFor(m * (esize / 4)), 256>>>((const uint32_t*)src, esize / 4, idx, m, (uint32_t*)dst, nulls);
  CK(cudaGetLastError());
}

void gpuGatherNull(const void* src, PType t, const int32_t* idx, int64_t m, void* dst) {
  if (m <= 0) return;
  if (t == PType::I32) gatherNullKernel<<<gridFor(m), 256>>>((const int32_t*)src, idx, m, (int32_t*)dst, kNullI32);
  else if (t == PType::I64) gatherNullKernel<<<gridFor(m), 256>>>((const int64_t*)src, idx, m, (int64_t*)dst, kNullI64);
  else gatherNullKernel<<<gridFor(m), 256>>>((const double*)src, idx, m, (double*)dst, nullv());
  CK(cudaGetLastError());
}

void gpuGather(const void* src, PType t, const int32_t* idx, int64_t m, void* dst) {
  if (m <= 0) return;
  if (t == PType::I32) gatherKernel<<<gridFor(m), 256>>>((const int32_t*)src, idx, m, (int32_t*)dst);
  else gatherKernel<<<gridFor(m), 256>>>((const int64_t*)src, idx, m, (int64_t*)dst);  // I64/F64: 8-byte copy
  CK(cudaGetLastError());
}

__global__ void setBitsKernel(uint32_t* bits, const int32_t* idx, int64_t m) {
  for (int64_t j = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; j < m; j += (int64_t)gridDim.x * blockDim.x)
    atomicOr(&bits[idx[j] >> 5], 1u << (idx[j] & 31));
}

__global__ void keyBitmapKernel(ColView key, int64_t n, int64_t lo, int64_t nbits, uint32_t* bits) {
  for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < n; i += (int64_t)gridDim.x * blockDim.x) {
    if (keyNull(key, i)) continue;
    const int64_t v = keyAt(key, i) - lo;
    if (v >= 0 && v < nbits) atomicOr(&bits[v >> 5], 1u << (v & 31));
  }
}

void gpuKeyBitmap(ColView key, int64_t n, int64_t lo, int64_t nbits, uint32_t* bits) {
  CK(cudaMemsetAsync(bits, 0, ((nbits + 31) / 32) * 4, 0));
  if (n > 0) keyBitmapKernel<<<gridFor(n), 256>>>(key, n, lo, nbits, bits);
  CK(cudaGetLastError());
}

void gpuSetBits(uint32_t* bits, const int32_t* idx, int64_t m) {
  if (m <= 0) return;
  setBitsKernel<<<gridFor(m), 256>>>(bits, idx, m);
  CK(cudaGetLastError());
}

// ------------------------------------------------------------------ hash join

constexpr long long JEMPTY = (long long)0x8000000000000000ll;

__global__ void joinBuild(ColView key, int64_t base, int64_t nb, unsigned long long* tk, int32_t* tr, unsigned long long mask) {
  for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < nb; i += (int64_t)gridDim.x * blockDim.x) {
    if (keyNull(key, base + i)) continue;
    long long k = keyAt(key, base + i);
    unsigned long long s = mix64((unsigned long long)k) & mask;
    for (;;) {
      unsigned long long prev = atomicCAS(&tk[s], (unsigned long long)JEMPTY, (unsigned long long)k);
      if (prev == (unsigned long long)JEMPTY) { tr[s] = (int32_t)(base + i); break; }
      s = (s + 1) & mask;
    }
  }
}

template <bool WRITE>
__global__ void joinProbe(ColView key, int64_t base, int64_t np, const unsigned long long* tk, const int32_t* tr,
                          unsigned long long mask, int64_t* counts, const int64_t* offs, int32_t* ob, int32_t* op) {
  for (int64_t jj = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; jj < np; jj += (int64_t)gridDim.x * blockDim.x) {
    const int64_t j = base + jj;
    if (keyNull(key, j)) {
      if (!WRITE) counts[j] = 0;
      continue;
    }
    long long k = keyAt(key, j);
    unsigned long long s = mix64((unsigned long long)k) & mask;
    int64_t c = 0, o = WRITE ? offs[j] : 0;
    while (tk[s] != (unsigned long long)JEMPTY) {
      if ((long long)tk[s] == k) {
        if (WRITE) { ob[o + c] = tr[s]; op[o + c] = (int32_t)j; }
        ++c;
      }
      s = (s + 1) & mask;
    }
    if (!WRITE) counts[j] = c;
  }
}

int64_t gpuHashJoinParts(ColView buildKey, const std::vector<int64_t>& bOff, ColView probeKey,
                         const std::vector<int64_t>& pOff, DevVec<int32_t>& outBuild, DevVec<int32_t>& outProbe,
                         std::vector<int64_t>* outOff) {
  const int pb = (int)bOff.size() - 1, pp = (int)pOff.size() - 1;
  if (pb != 1 && pb != pp) throw SqlError("internal: partition count mismatch in join", 1105);
  const int64_t nb = bOff[pb], np = pOff[pp];
  if (outOff) outOff->assign(pp + 1, 0);
  if (nb == 0 || np == 0) {
    outBuild = DevVec<int32_t>();
    outProbe = DevVec<int32_t>();
    return 0;
  }
  // One open-addressing table per build partition, all in one allocation.
  std::vector<int64_t> reg(pb + 1, 0), cap(pb);
  for (int q = 0; q < pb; ++q) {
    int64_t c = 2;
    while (c < (bOff[q + 1] - bOff[q]) * 2) c <<= 1;
    cap[q] = c;
    reg[q + 1] = reg[q] + c;
  }
  DevVec<unsigned long long> tk(reg[pb]);
  DevVec<int32_t> tr(reg[pb]);
  thrust::fill(thrust::device, tk.p, tk.p + reg[pb], (unsigned long long)JEMPTY);
  for (int q = 0; q < pb; ++q) {
    const int64_t n = bOff[q + 1] - bOff[q];
    if (n) joinBuild<<<gridFor(n), 256>>>(buildKey, bOff[q], n, tk.p + reg[q], tr.p + reg[q], (unsigned long long)(cap[q] - 1));
  }
  CK(cudaGetLastError());
  DevVec<int64_t> counts(np), offs(np);
  for (int p = 0; p < pp; ++p) {
    const int q = pb == 1 ? 0 : p;
    const int64_t n = pOff[p + 1] - pOff[p];
    if (n)
      joinProbe<false><<<gridFor(n), 256>>>(probeKey, pOff[p], n, tk.p + reg[q], tr.p + reg[q],
                                            (unsigned long long)(cap[q] - 1), counts.p, nullptr, nullptr, nullptr);
  }
  CK(cudaGetLastError());
  size_t tb = 0;
  CK(cub::DeviceScan::ExclusiveSum(nullptr, tb, counts.p, offs.p, np, 0));
  DevVec<uint8_t> tmp((int64_t)tb);
  CK(cub::DeviceScan::ExclusiveSum(tmp.p, tb, counts.p, offs.p, np, 0));
  int64_t lastOff, lastCnt;
  d2h(&lastOff, offs.p + np - 1, 8);
  d2h(&lastCnt, counts.p + np - 1, 8);
  int64_t m = lastOff + lastCnt;
  if (m > INT32_MAX) throw SqlError("join result too large for this PoC (>2^31 rows)", 1105);
  // Output is in probe order, so it stays grouped by the probe side's partitions.
  if (outOff)
    for (int p = 0; p <= pp; ++p) {
      if (pOff[p] < np) d2h(&(*outOff)[p], offs.p + pOff[p], 8);
      else (*outOff)[p] = m;
    }
  outBuild = DevVec<int32_t>(m);
  outProbe = DevVec<int32_t>(m);
  if (m)
    for (int p = 0; p < pp; ++p) {
      const int q = pb == 1 ? 0 : p;
      const int64_t n = pOff[p + 1] - pOff[p];
      if (n)
        joinProbe<true><<<gridFor(n), 256>>>(probeKey, pOff[p], n, tk.p + reg[q], tr.p + reg[q],
                                             (unsigned long long)(cap[q] - 1), nullptr, offs.p, outBuild.p, outProbe.p);
    }
  CK(cudaGetLastError());
  return m;
}

// Outer join probe: probe rows without a match produce one pair with build index -1; with `matched`
// every build row that matched is flagged (FULL OUTER JOIN appends the rest afterwards).
template <bool WRITE>
__global__ void joinProbeOuter(ColView key, int64_t np, const unsigned long long* tk, const int32_t* tr,
                               unsigned long long mask, int64_t* counts, const int64_t* offs, int32_t* ob, int32_t* op,
                               uint8_t* matched) {
  for (int64_t j = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; j < np; j += (int64_t)gridDim.x * blockDim.x) {
    int64_t c = 0, o = WRITE ? offs[j] : 0;
    if (!keyNull(key, j)) {
      long long k = keyAt(key, j);
      unsigned long long s = mix64((unsigned long long)k) & mask;
      while (tk[s] != (unsigned long long)JEMPTY) {
        if ((long long)tk[s] == k) {
          if (WRITE) {
            ob[o + c] = tr[s];
            op[o + c] = (int32_t)j;
            if (matched) matched[tr[s]] = 1;
          }
          ++c;
        }
        s = (s + 1) & mask;
      }
    }
    if (c == 0) {
      if (WRITE) { ob[o] = -1; op[o] = (int32_t)j; }
      c = 1;
    }
    if (!WRITE) counts[j] = c;
  }
}

__global__ void unmatchedFlags(const uint8_t* matched, int64_t n, uint8_t* flags) {
  for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < n; i += (int64_t)gridDim.x * blockDim.x)
    flags[i] = !matched[i];
}
__global__ void fillInt(int32_t* p, int64_t n, int32_t v) {
  for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < n; i += (int64_t)gridDim.x * blockDim.x) p[i] = v;
}

int64_t gpuHashJoinOuter(ColView buildKey, int64_t nb, ColView probeKey, int64_t np, DevVec<int32_t>& outBuild,
                         DevVec<int32_t>& outProbe, bool full) {
  int64_t cap = 2;
  while (cap < nb * 2) cap <<= 1;
  DevVec<unsigned long long> tk(cap);
  DevVec<int32_t> tr(cap);
  thrust::fill(thrust::device, tk.p, tk.p + cap, (unsigned long long)JEMPTY);
  if (nb) joinBuild<<<gridFor(nb), 256>>>(buildKey, 0, nb, tk.p, tr.p, (unsigned long long)(cap - 1));
  DevVec<uint8_t> matched(full ? std::max<int64_t>(nb, 1) : 0);
  if (full && nb) CK(cudaMemsetAsync(matched.p, 0, nb, 0));
  int64_t m = 0;
  DevVec<int64_t> counts(std::max<int64_t>(np, 1)), offs(std::max<int64_t>(np, 1));
  if (np) {
    joinProbeOuter<false><<<gridFor(np), 256>>>(probeKey, np, tk.p, tr.p, (unsigned long long)(cap - 1), counts.p, nullptr,
                                                 nullptr, nullptr, nullptr);
    size_t tb = 0;
    CK(cub::DeviceScan::ExclusiveSum(nullptr, tb, counts.p, offs.p, np, 0));
    DevVec<uint8_t> tmp((int64_t)tb);
    CK(cub::DeviceScan::ExclusiveSum(tmp.p, tb, counts.p, offs.p, np, 0));
    int64_t lastOff, lastCnt;
    d2h(&lastOff, offs.p + np - 1, 8);
    d2h(&lastCnt, counts.p + np - 1, 8);
    m = lastOff + lastCnt;
  }
  // FULL: build rows that never matched, paired with probe index -1.
  int64_t extra = 0;
  DevVec<int32_t> un;
  if (full && nb) {
    // matched flags are only known after the write pass; run it into temporaries first.
  }
  if (m > INT32_MAX) throw SqlError("join result too large for this PoC (>2^31 rows)", 1105);
  DevVec<int32_t> b0(std::max<int64_t>(m, 1)), p0(std::max<int64_t>(m, 1));
  if (np)
    joinProbeOuter<true><<<gridFor(np), 256>>>(probeKey, np, tk.p, tr.p, (unsigned long long)(cap - 1), nullptr, offs.p, b0.p,
                                                p0.p, full ? matched.p : nullptr);
  if (full && nb) {
    DevVec<uint8_t> fl(nb);
    unmatchedFlags<<<gridFor(nb), 256>>>(matched.p, nb, fl.p);
    un = DevVec<int32_t>(nb);
    DevVec<int64_t> cnt(1);
    cub::CountingInputIterator<int32_t> it(0);
    size_t tb = 0;
    CK(cub::DeviceSelect::Flagged(nullptr, tb, it, fl.p, un.p, cnt.p, nb, 0));
    DevVec<uint8_t> tmp((int64_t)tb);
    CK(cub::DeviceSelect::Flagged(tmp.p, tb, it, fl.p, un.p, cnt.p, nb, 0));
    d2h(&extra, cnt.p, 8);
  }
  outBuild = DevVec<int32_t>(std::max<int64_t>(m + extra, 1));
  outProbe = DevVec<int32_t>(std::max<int64_t>(m + extra, 1));
  if (m) {
    d2d(outBuild.p, b0.p, m * 4);
    d2d(outProbe.p, p0.p, m * 4);
  }
  if (extra) {
    d2d(outBuild.p + m, un.p, extra * 4);
    fillInt<<<gridFor(extra), 256>>>(outProbe.p + m, extra, -1);
  }
  CK(cudaGetLastError());
  return m + extra;
}

__global__ void crossKernel(int64_t na, int64_t nb, int32_t* oa, int32_t* ob) {
  const int64_t n = na * nb;
  for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < n; i += (int64_t)gridDim.x * blockDim.x) {
    oa[i] = (int32_t)(i / nb);
    ob[i] = (int32_t)(i % nb);
  }
}

int64_t gpuCrossJoin(int64_t na, int64_t nb, DevVec<int32_t>& outA, DevVec<int32_t>& outB) {
  const int64_t n = na * nb;
  if (n > 200'000'000) throw SqlError("cross join too large (" + std::to_string(n) + " rows); add a join condition", 1105);
  outA = DevVec<int32_t>(std::max<int64_t>(n, 1));
  outB = DevVec<int32_t>(std::max<int64_t>(n, 1));
  if (n) crossKernel<<<gridFor(n), 256>>>(na, nb, outA.p, outB.p);
  CK(cudaGetLastError());
  return n;
}

int64_t gpuHashJoin(ColView buildKey, int64_t nb, ColView probeKey, int64_t np, DevVec<int32_t>& outBuild,
                    DevVec<int32_t>& outProbe) {
  return gpuHashJoinParts(buildKey, {0, nb}, probeKey, {0, np}, outBuild, outProbe, nullptr);
}

// ------------------------------------------------------------------ sorting

__global__ void nullsLow(double* v, int64_t m) {
  for (int64_t j = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; j < m; j += (int64_t)gridDim.x * blockDim.x)
    if (v[j] != v[j]) v[j] = -INFINITY;
}

__global__ void seqK(int32_t* p, int64_t m) {
  for (int64_t j = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; j < m; j += (int64_t)gridDim.x * blockDim.x) p[j] = (int32_t)j;
}
// out[j] = key[perm[j]] with NULL (NaN) -> -inf (NULLs sort first, as in MySQL) and -0 -> 0.
__global__ void sortKeyK(const int32_t* perm, const double* key, int64_t m, double* out) {
  for (int64_t j = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; j < m; j += (int64_t)gridDim.x * blockDim.x) {
    double v = key[perm[j]];
    out[j] = v != v ? -INFINITY : v == 0 ? 0.0 : v;
  }
}

// Stable multi-key sort: LSD radix passes (CUB) from the last key to the first, with temporary
// storage from the stream-ordered pool (no synchronous cudaMalloc as with thrust's allocator).
DevVec<int32_t> gpuSortPerm(const std::vector<std::pair<const double*, bool>>& keys, int64_t m) {
  DevVec<int32_t> perm(m);
  if (m == 0) return perm;
  seqK<<<gridFor(m), 256>>>(perm.p, m);
  DevVec<int32_t> perm2(m);
  DevVec<double> k1(m), k2(m);
  DevVec<uint8_t> temp;
  for (int i = (int)keys.size() - 1; i >= 0; --i) {
    sortKeyK<<<gridFor(m), 256>>>(perm.p, keys[i].first, m, k1.p);
    size_t tb = 0;
    if (keys[i].second) CK(cub::DeviceRadixSort::SortPairsDescending(nullptr, tb, k1.p, k2.p, perm.p, perm2.p, (int)m, 0, 64, 0));
    else CK(cub::DeviceRadixSort::SortPairs(nullptr, tb, k1.p, k2.p, perm.p, perm2.p, (int)m, 0, 64, 0));
    if ((int64_t)tb > temp.n) temp = DevVec<uint8_t>((int64_t)tb);
    if (keys[i].second) CK(cub::DeviceRadixSort::SortPairsDescending(temp.p, tb, k1.p, k2.p, perm.p, perm2.p, (int)m, 0, 64, 0));
    else CK(cub::DeviceRadixSort::SortPairs(temp.p, tb, k1.p, k2.p, perm.p, perm2.p, (int)m, 0, 64, 0));
    std::swap(perm, perm2);
  }
  return perm;
}

__global__ void mapRankKernel(double* keys, const int32_t* rank, int64_t m) {
  for (int64_t j = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; j < m; j += (int64_t)gridDim.x * blockDim.x)
    keys[j] = keys[j] != keys[j] ? -1.0 : rank[(int64_t)keys[j]];  // NULL first (MySQL)
}

void gpuMapRank(double* keys, const int32_t* rank, int64_t m) {
  if (m <= 0) return;
  mapRankKernel<<<gridFor(m), 256>>>(keys, rank, m);
  CK(cudaGetLastError());
}

template <class T>
__global__ void toDouble(const T* src, int64_t n, double* dst) {
  for (int64_t j = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; j < n; j += (int64_t)gridDim.x * blockDim.x)
    dst[j] = (double)src[j];
}

std::vector<int32_t> gpuSortPermHost(const void* hostCol, PType t, int64_t n) {
  if (n == 0) return {};
  DevVec<uint8_t> raw(n * psize(t));
  h2d(raw.p, hostCol, n * psize(t));
  DevVec<double> k(n);
  if (t == PType::I32) toDouble<<<gridFor(n), 256>>>((const int32_t*)raw.p, n, k.p);
  else if (t == PType::I64) toDouble<<<gridFor(n), 256>>>((const int64_t*)raw.p, n, k.p);
  else d2d(k.p, raw.p, n * 8);
  CK(cudaGetLastError());
  DevVec<int32_t> perm = gpuSortPerm({{k.p, false}}, n);
  return perm.download();
}

}  // namespace gdb
