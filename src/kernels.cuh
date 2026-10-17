// Kernel bodies shared by the ahead-of-time build (gpu.cu, expressions interpreted from bytecode)
// and the runtime-compiled build (jit.cu, expressions generated as typed CUDA C++ per query).
// They are templates over an evaluator `Ev` that supplies:
//   static constexpr int kKeys, kAggs;   // loop bounds (>= nkeys(), naggs())
//   Ev(const ExprEnv&, const AggPlan&);  void bind(const ColView* cols);   // per batch
//   int nkeys(), naggs(), op(a); bool aggIsInt(a), hasWhere();
//   bool where(r); double key(k, r); long long ikey(k, r); double agg(a, r); long long iagg(a, r);
// In the generated evaluator the shape queries are compile-time constants, so the unrolled loops
// below collapse to exactly the work the query needs.
// Must stay compilable by NVRTC: no host headers.
#pragma once
#ifndef __CUDACC_RTC__
#include "devtypes.h"
#endif

namespace gdb {

constexpr unsigned long long EMPTY = ~0ull;
constexpr long long kLLMax = 0x7fffffffffffffffll;
constexpr long long kLLMin = -kLLMax - 1;

extern __shared__ __align__(16) unsigned char gdbSmem[];

__device__ __forceinline__ unsigned long long mix64(unsigned long long x) {
  x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ull;
  x ^= x >> 27; x *= 0x94d049bb133111ebull;
  x ^= x >> 31;
  return x;
}

__device__ __forceinline__ bool isDeleted(const uint32_t* del, int64_t r) {
  return del && ((del[r >> 5] >> (r & 31)) & 1u);
}

__device__ __forceinline__ double aggIdentity(uint8_t op) {
  return op == AGG_MIN ? __longlong_as_double(0x7ff0000000000000ll)
       : op == AGG_MAX ? __longlong_as_double(0xfff0000000000000ll) : 0.0;
}
// Aggregates skip NULLs (NaN): fmin/fmax already ignore NaN, sums skip it explicitly.
__device__ __forceinline__ double combine(uint8_t op, double a, double b) {
  // (a is NULL too in the warp's segmented scan, where it is a lane's own value.)
  return op == AGG_MIN ? fmin(a, b) : op == AGG_MAX ? fmax(a, b) : b != b ? a : a != a ? b : a + b;
}
__device__ __forceinline__ void atomicCombine(uint8_t op, double* addr, double v) {
  if (v != v) return;
  if (op == AGG_SUM || op == AGG_COUNT) {
    atomicAdd(addr, v);
    return;
  }
  unsigned long long* a = (unsigned long long*)addr;
  unsigned long long old = *a, assumed;
  do {
    assumed = old;
    double cur = __longlong_as_double((long long)assumed);
    double nv = op == AGG_MIN ? fmin(cur, v) : fmax(cur, v);
    if (nv == cur) return;
    old = atomicCAS(a, assumed, (unsigned long long)__double_as_longlong(nv));
  } while (old != assumed);
}

// 64-bit integer aggregates with GROUP BY (AggPlan::aggBits): the slot holds exact int64 bits.
// A BIGINT SUM is split into the sums of its low (unsigned) and high 32 bits (aggPart 1 / 2), each
// exact in int64 below 2^31 rows per group; the host recombines them in 128 bits.
// A NULL of a nullable integer column (AggPlan::aggNull) becomes the aggregate's neutral value.
__device__ __forceinline__ long long ipart(const AggPlan& p, int a, long long v) {
  if (p.aggNull[a] && v == (p.aggNull[a] == 3 ? (long long)kNullI32 : (long long)kNullI64))
    return p.ops[a] == AGG_MIN ? kLLMax : p.ops[a] == AGG_MAX ? kLLMin : 0;
  return !p.aggBits[a] ? v : p.aggPart[a] == 1 ? (long long)(v & 0xffffffffll) : p.aggPart[a] == 2 ? (v >> 32) : v;
}
__device__ __forceinline__ double bitsIdentity(uint8_t op) { return asBits(op == AGG_MIN ? kLLMax : op == AGG_MAX ? kLLMin : 0); }
__device__ __forceinline__ double combineB(bool bits, uint8_t op, double a, double b) {
  if (!bits) return combine(op, a, b);
  const long long x = bitsOf(a), y = bitsOf(b);
  return asBits(op == AGG_MIN ? (y < x ? y : x) : op == AGG_MAX ? (y > x ? y : x) : x + y);
}
__device__ __forceinline__ void atomicCombineB(bool bits, uint8_t op, double* addr, double v) {
  if (!bits) return atomicCombine(op, addr, v);
  const long long y = bitsOf(v);
  if (op == AGG_MIN) atomicMin((long long*)addr, y);
  else if (op == AGG_MAX) atomicMax((long long*)addr, y);
  else atomicAdd((unsigned long long*)addr, (unsigned long long)y);
}

// Helpers used by generated expression code.
__device__ __forceinline__ long long gdbInset(const uint32_t* bits, int size, long long c) {
  return (c >= 0 && c < size) ? (long long)((bits[c >> 5] >> (c & 31)) & 1u) : 0ll;
}
__device__ __forceinline__ long long gdbYear(long long days) {
  int64_t y;
  unsigned m, d;
  civilFromDays(days, y, m, d);
  return y;
}

// ------------------------------------------------------------------ interpreter evaluator

// One out-of-line interpreter call: inlined into every unrolled key / aggregate evaluation, its
// value stack made the dense-aggregation kernel need ~6 KB of stack per thread.
static __device__ __noinline__ double evalProgNI(const ExprEnv& env, const ColView* cols, Prog pg, int64_t row) {
  return evalProgC(env, cols, pg, row);
}

struct InterpEv {
  // Loop bounds for the kernel bodies; the generated evaluator uses the query's exact counts, so
  // NVRTC doesn't unroll (and then discard) code for 16 aggregates and 4 keys.
  static constexpr int kKeys = kInterpKeys, kAggs = kInterpAggs;
  static constexpr int kDensePriv = 0;  // decided at run time from DenseSpec::priv
  const ExprEnv& env;
  const AggPlan& plan;
  const ColView* cols = nullptr;
  // Always the device copy (AggPlan::genv): referencing the kernel parameter would copy it to the stack.
  __device__ InterpEv(const ExprEnv&, const AggPlan& p) : env(*p.genv), plan(p) {}
  __device__ void bind(const ColView* c) { cols = c; }
  __device__ const ColView* envCols(const ExprEnv&) const { return env.cols; }  // (the device copy)
  __device__ int nkeys() const { return plan.nkeys; }
  __device__ int naggs() const { return plan.naggs; }
  __device__ int op(int a) const { return plan.ops[a]; }
  __device__ bool aggIsInt(int a) const { return plan.aggInt[a] != 0; }
  __device__ bool hasWhere() const { return plan.hasWhere; }
  __device__ bool where(int64_t r) const { return !plan.hasWhere || truthy(evalProgNI(env, cols, plan.where, r)); }
  __device__ double key(int k, int64_t r) const { return evalProgNI(env, cols, plan.keys[k], r); }
  __device__ long long ikey(int k, int64_t r) const {
    const Prog& p = plan.keys[k];  // a plain integer column: read it exactly (BIGINT above 2^53)
    if (p.len == 1 && env.code[p.start].op == OP_COL && cols[env.code[p.start].arg].t != 2) return colInt(cols[env.code[p.start].arg], r);
    return (long long)key(k, r);
  }
  __device__ bool keyIsInt(int k) const { return plan.keyInt[k] != 0; }
  __device__ double agg(int a, int64_t r) const {
    return plan.ops[a] == AGG_COUNT ? 1.0 : evalProgNI(env, cols, plan.aggs[a], r);
  }
  __device__ long long iagg(int a, int64_t r) const {
    const Prog& p = plan.aggs[a];  // a plain integer column: read it exactly (BIGINT above 2^53)
    if (plan.ops[a] != AGG_COUNT && p.len == 1 && env.code[p.start].op == OP_COL && cols[env.code[p.start].arg].t != 2)
      return colInt(cols[env.code[p.start].arg], r);
    return (long long)agg(a, r);
  }
  __device__ uint8_t dkind(int a, const DenseSpec& ds) const { return ds.kind[a]; }
};

// ------------------------------------------------------------------ batches

// Block-wide: stage batch b's column views in shared memory (all threads must call).
__device__ __forceinline__ void loadBatch(const BatchSet& bs, int b, ColView* scols, int64_t* sn, const uint32_t** sdel) {
  __syncthreads();
  for (int i = threadIdx.x; i < bs.ncols; i += blockDim.x) scols[i] = bs.cols[(int64_t)b * bs.ncols + i];
  if (threadIdx.x == 0) {
    *sn = bs.n[b];
    *sdel = bs.del[b];
  }
  __syncthreads();
}

// ------------------------------------------------------------------ hash table (grouped aggregation)

struct GTable {
  unsigned long long* keys;
  double* kv;
  double* acc;
  unsigned long long mask;
  int* flags;
  unsigned long long* count;
};

__device__ inline bool globalInsert(const GTable& t, int nkeys, int naggs, const uint8_t* ops, const uint8_t* bits, unsigned long long h,
                                    const double* kv, const double* vals) {
  if (*(volatile int*)t.flags) return false;  // table overflowed: the query will be retried bigger
  unsigned long long s = h & t.mask;
  for (int probe = 0; probe < 256; ++probe) {
    unsigned long long prev = atomicCAS(&t.keys[s], EMPTY, h);
    if (prev == EMPTY) {
      for (int k = 0; k < nkeys; ++k) t.kv[s * nkeys + k] = kv[k];
      if ((atomicAdd(t.count, 1ull) + 1) * 10 > (t.mask + 1) * 7) t.flags[0] = 1;
    }
    if (prev == EMPTY || prev == h) {
      for (int a = 0; a < naggs; ++a) atomicCombineB(bits[a], ops[a], &t.acc[s * naggs + a], vals[a]);
      return true;
    }
    s = (s + 1) & t.mask;
  }
  t.flags[0] = 1;
  return false;
}

// ------------------------------------------------------------------ kernels

// Aggregates without GROUP BY: per-thread accumulators across all batches, then a block reduction.
// COUNT and integer SUM/MIN/MAX accumulate in int64 (exact, and no FP64 work per row).
template <class Ev>
__device__ void aggNoKeysBody(const ExprEnv& env, const AggPlan& plan, const BatchSet& bs, double* out) {
  __shared__ ColView scols[kMaxCols];
  __shared__ int64_t sn;
  __shared__ const uint32_t* sdel;
  Ev ev(env, plan);
  double acc[Ev::kAggs];
  long long iacc[Ev::kAggs];
  __int128 wacc[Ev::kAggs];  // integer SUMs: 128-bit (no overflow), exact
#pragma unroll
  for (int a = 0; a < Ev::kAggs; ++a) {
    acc[a] = a < ev.naggs() ? aggIdentity(ev.op(a)) : 0.0;
    iacc[a] = a < ev.naggs() && ev.op(a) == AGG_MIN ? kLLMax : a < ev.naggs() && ev.op(a) == AGG_MAX ? kLLMin : 0;
    wacc[a] = 0;
  }
  long long cnt = 0;
  for (int b = 0; b < bs.nb; ++b) {
    loadBatch(bs, b, scols, &sn, &sdel);
    ev.bind(scols);
    const int64_t n = sn;
    const uint32_t* del = sdel;
    for (int64_t r = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; r < n; r += (int64_t)gridDim.x * blockDim.x) {
      if (isDeleted(del, r)) continue;
      if (ev.hasWhere() && !ev.where(r)) continue;
      ++cnt;
#pragma unroll
      for (int a = 0; a < Ev::kAggs; ++a) {
        if (a >= ev.naggs() || ev.op(a) == AGG_COUNT) continue;
        if (ev.aggIsInt(a)) {
          const long long v = ipart(plan, a, ev.iagg(a, r));
          if (ev.op(a) == AGG_SUM) wacc[a] += v;
          else iacc[a] = ev.op(a) == AGG_MIN ? (v < iacc[a] ? v : iacc[a]) : (v > iacc[a] ? v : iacc[a]);
        } else {
          acc[a] = combine(ev.op(a), acc[a], ev.agg(a, r));
        }
      }
    }
  }
#pragma unroll
  for (int a = 0; a < Ev::kAggs; ++a) {
    if (a >= ev.naggs()) continue;
    if (ev.op(a) == AGG_COUNT) acc[a] = (double)cnt;
    else if (ev.aggIsInt(a))
      acc[a] = ev.op(a) == AGG_SUM ? (double)iacc[a] : (iacc[a] == kLLMax || iacc[a] == kLLMin) ? aggIdentity(ev.op(a)) : (double)iacc[a];
  }
  __shared__ double red[32];
  __shared__ long long ired[32];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
#pragma unroll
  for (int a = 0; a < Ev::kAggs; ++a) {
    if (a >= ev.naggs()) break;
    if (ev.aggIsInt(a) && ev.op(a) == AGG_SUM) {  // exact 128-bit sum: low half in out[a], high in out[kMaxAggs + a]
      __int128 v = wacc[a];
      for (int o = 16; o > 0; o >>= 1) {
        const unsigned long long lo = __shfl_down_sync(0xffffffffu, (unsigned long long)v, o);
        const long long hi = __shfl_down_sync(0xffffffffu, (long long)(v >> 64), o);
        v += (__int128)(((unsigned __int128)(unsigned long long)hi << 64) | lo);
      }
      __shared__ unsigned long long wlo[32];
      __shared__ long long whi[32];
      if (lane == 0) wlo[warp] = (unsigned long long)v, whi[warp] = (long long)(v >> 64);
      __syncthreads();
      if (warp == 0) {
        v = lane < nw ? (__int128)(((unsigned __int128)(unsigned long long)whi[lane] << 64) | wlo[lane]) : 0;
        for (int o = 16; o > 0; o >>= 1) {
          const unsigned long long lo = __shfl_down_sync(0xffffffffu, (unsigned long long)v, o);
          const long long hi = __shfl_down_sync(0xffffffffu, (long long)(v >> 64), o);
          v += (__int128)(((unsigned __int128)(unsigned long long)hi << 64) | lo);
        }
        if (lane == 0) {
          const unsigned long long lo = (unsigned long long)v;
          unsigned long long hi = (unsigned long long)(v >> 64);
          const unsigned long long old = atomicAdd((unsigned long long*)&out[a], lo);
          if (old + lo < old) ++hi;  // carry out of the low word
          if (hi) atomicAdd((unsigned long long*)&out[kMaxAggs + a], hi);
        }
      }
      __syncthreads();
      continue;
    }
    if (ev.aggIsInt(a) && (ev.op(a) == AGG_MIN || ev.op(a) == AGG_MAX)) {  // exact: int64 min / max as bits
      const bool mn = ev.op(a) == AGG_MIN;
      long long v = iacc[a];
      for (int o = 16; o > 0; o >>= 1) {
        const long long w = __shfl_down_sync(0xffffffffu, v, o);
        v = mn ? (w < v ? w : v) : (w > v ? w : v);
      }
      if (lane == 0) ired[warp] = v;
      __syncthreads();
      if (warp == 0) {
        v = lane < nw ? ired[lane] : (mn ? kLLMax : kLLMin);
        for (int o = 16; o > 0; o >>= 1) {
          const long long w = __shfl_down_sync(0xffffffffu, v, o);
          v = mn ? (w < v ? w : v) : (w > v ? w : v);
        }
        if (lane == 0) {
          if (mn) atomicMin((long long*)&out[a], v);
          else atomicMax((long long*)&out[a], v);
        }
      }
      __syncthreads();
      continue;
    }
    double v = acc[a];
    for (int o = 16; o > 0; o >>= 1) v = combine(ev.op(a), v, __shfl_down_sync(0xffffffffu, v, o));
    if (lane == 0) red[warp] = v;
    __syncthreads();
    if (warp == 0) {
      v = lane < nw ? red[lane] : aggIdentity(ev.op(a));
      for (int o = 16; o > 0; o >>= 1) v = combine(ev.op(a), v, __shfl_down_sync(0xffffffffu, v, o));
      if (lane == 0) atomicCombine(ev.op(a), &out[a], v);
    }
    __syncthreads();
  }
}

// Grouped aggregation by hashing: warp-level pre-aggregation -> a small hash table private to each
// warp in shared memory (persisting across all batches) -> global hash table.
template <class Ev>
__device__ void aggGroupedBody(const ExprEnv& env, const AggPlan& plan, const BatchSet& bs, GTable gt, int W) {
  __shared__ ColView scols[kMaxCols];
  __shared__ int64_t sn;
  __shared__ const uint32_t* sdel;
  Ev ev(env, plan);
  const int NK = ev.nkeys(), NA = ev.naggs();
  uint8_t ops[Ev::kAggs];
#pragma unroll
  for (int a = 0; a < Ev::kAggs; ++a) ops[a] = a < NA ? (uint8_t)ev.op(a) : 0;
  const int nw = blockDim.x >> 5, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  unsigned long long* sk = (unsigned long long*)gdbSmem;
  double* skv = (double*)(sk + (size_t)nw * W);
  double* sacc = skv + (size_t)nw * W * NK;
  for (int i = threadIdx.x; i < nw * W; i += blockDim.x) {
    sk[i] = EMPTY;
    for (int a = 0; a < NA; ++a) sacc[(size_t)i * NA + a] = plan.aggBits[a] ? bitsIdentity(ops[a]) : aggIdentity(ops[a]);
  }
  const int base = warp * W;
  for (int b = 0; b < bs.nb; ++b) {
    loadBatch(bs, b, scols, &sn, &sdel);
    ev.bind(scols);
    const int64_t n = sn;
    const uint32_t* del = sdel;
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t b0 = blockIdx.x * (int64_t)blockDim.x; b0 < n; b0 += stride) {
      const int64_t r = b0 + threadIdx.x;
      bool valid = r < n && !isDeleted(del, r);
      if (valid && ev.hasWhere()) valid = ev.where(r);
      if (!__any_sync(0xffffffffu, valid)) continue;
      double kv[Ev::kKeys];
      double av[Ev::kAggs];
      unsigned long long h = EMPTY;
      if (valid) {
        h = 0x9e3779b97f4a7c15ull;
#pragma unroll
        for (int k = 0; k < Ev::kKeys; ++k)
          if (k < NK) {
            kv[k] = ev.keyIsInt(k) ? asBits(ev.ikey(k, r)) : ev.key(k, r) + 0.0;
            h = mix64(h ^ (unsigned long long)__double_as_longlong(kv[k])) + k;
          }
        if (h == EMPTY) h ^= 1;
#pragma unroll
        for (int a = 0; a < Ev::kAggs; ++a)
          if (a < NA) av[a] = plan.aggBits[a] ? asBits(ipart(plan, a, ev.iagg(a, r))) : ev.agg(a, r);
      }
      // Warp-level pre-aggregation without FP64-heavy all-pairs reductions (consumer GPUs run FP64
      // at 1/64 rate): bitonic-sort the 32 group hashes across lanes (integer shuffles only), then a
      // 5-step segmented scan per aggregate. The last lane of each run of equal keys is the leader.
      unsigned long long sk2 = h;
      int src = lane;
#pragma unroll
      for (int k = 2; k <= 32; k <<= 1)
#pragma unroll
        for (int j = k >> 1; j > 0; j >>= 1) {
          unsigned long long ok = __shfl_xor_sync(0xffffffffu, sk2, j);
          int os = __shfl_xor_sync(0xffffffffu, src, j);
          bool asc = (lane & k) == 0, lower = (lane & j) == 0;
          bool take = lower == asc ? ok < sk2 : ok > sk2;
          if (take) { sk2 = ok; src = os; }
        }
      unsigned sameMask = 0;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        unsigned long long up = __shfl_up_sync(0xffffffffu, sk2, o);
        if (lane >= o && up == sk2) sameMask |= o;
      }
      const unsigned long long nextKey = __shfl_down_sync(0xffffffffu, sk2, 1);
      const bool leader = sk2 != EMPTY && (lane == 31 || nextKey != sk2);
      double lv[Ev::kAggs];
#pragma unroll
      for (int a = 0; a < Ev::kAggs; ++a) {
        if (a >= NA) break;
        double v = __shfl_sync(0xffffffffu, valid ? av[a] : 0.0, src);
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
          double other = __shfl_up_sync(0xffffffffu, v, o);
          if (sameMask & o) v = combineB(plan.aggBits[a], ops[a], v, other);
        }
        lv[a] = v;
      }
      double lk[Ev::kKeys];
#pragma unroll
      for (int k = 0; k < Ev::kKeys; ++k)
        if (k < NK) lk[k] = __shfl_sync(0xffffffffu, valid ? kv[k] : 0.0, src);
      if (leader) {
        const unsigned long long hk = sk2;
        unsigned s = (unsigned)(hk & (W - 1));
        bool done = false;
        for (int probe = 0; probe < 8; ++probe) {
          unsigned long long prev = atomicCAS(&sk[base + s], EMPTY, hk);
          if (prev == EMPTY)
            for (int k = 0; k < NK; ++k) skv[(size_t)(base + s) * NK + k] = lk[k];
          if (prev == EMPTY || prev == hk) {
            double* acc = &sacc[(size_t)(base + s) * NA];
            for (int a = 0; a < NA; ++a) acc[a] = combineB(plan.aggBits[a], ops[a], acc[a], lv[a]);
            done = true;
            break;
          }
          s = (s + 1) & (W - 1);
        }
        if (!done) globalInsert(gt, NK, NA, ops, plan.aggBits, hk, lk, lv);
      }
      __syncwarp();
    }
  }
  __syncthreads();
  for (int i = base + lane; i < base + W; i += 32)
    if (sk[i] != EMPTY) globalInsert(gt, NK, NA, ops, plan.aggBits, sk[i], &skv[(size_t)i * NK], &sacc[(size_t)i * NA]);
}

// Dense aggregation over very few groups (G x naggs small): each thread accumulates into its own
// slots in shared memory, laid out [slot][thread] so a warp's accesses are conflict-free. No atomics,
// shuffles or divergence per row: one plain add per aggregate (FP64 is 1/64 rate on consumer GPUs).
// The block then reduces each slot with one warp and flushes it to global memory once.
template <class Ev>
__device__ void aggDensePrivBody(const ExprEnv& env, const AggPlan& plan, const DenseSpec& ds, const BatchSet& bs, double* gacc) {
  __shared__ ColView scols[kMaxCols];
  __shared__ int64_t sn;
  __shared__ const uint32_t* sdel;
  Ev ev(env, plan);
  const int NK = ev.nkeys(), NA = ev.naggs();
  const int T = blockDim.x, tid = threadIdx.x;
  const int S = (int)ds.G * NA;
  double* sd = (double*)gdbSmem;
  long long* sl = (long long*)gdbSmem;
  for (int s = 0; s < S; ++s) {
    const int a = s % NA;
    const size_t i = (size_t)s * T + tid;
    switch (ds.kind[a]) {
      case DK_SUM_D: sd[i] = 0.0; break;
      case DK_MIN_D: sd[i] = aggIdentity(AGG_MIN); break;
      case DK_MAX_D: sd[i] = aggIdentity(AGG_MAX); break;
      case DK_MIN_I: sl[i] = kLLMax; break;
      case DK_MAX_I: sl[i] = kLLMin; break;
      default: sl[i] = 0; break;
    }
  }
  for (int b = 0; b < bs.nb; ++b) {
    loadBatch(bs, b, scols, &sn, &sdel);
    ev.bind(scols);
    const int64_t n = sn;
    const uint32_t* del = sdel;
    for (int64_t r = blockIdx.x * (int64_t)blockDim.x + tid; r < n; r += (int64_t)gridDim.x * blockDim.x) {
      if (isDeleted(del, r)) continue;
      if (ev.hasWhere() && !ev.where(r)) continue;
      long long g = 0;
#pragma unroll
      for (int k = 0; k < Ev::kKeys; ++k)
        if (k < NK) g += (ev.ikey(k, r) - ds.kmin[k]) * ds.stride[k];
      const size_t base = (size_t)g * NA * T + tid;
#pragma unroll
      for (int a = 0; a < Ev::kAggs; ++a) {
        if (a >= NA) break;
        const size_t i = base + (size_t)a * T;
        switch (ev.dkind(a, ds)) {
          case DK_COUNT: sl[i] += 1; break;
          case DK_SUM_I: sl[i] += ipart(plan, a, ev.iagg(a, r)); break;
          case DK_SUM_D: {
            const double v = ev.agg(a, r);
            if (v == v) sd[i] += v;
            break;
          }
          case DK_MIN_I: { const long long v = ipart(plan, a, ev.iagg(a, r)); if (v < sl[i]) sl[i] = v; break; }
          case DK_MAX_I: { const long long v = ipart(plan, a, ev.iagg(a, r)); if (v > sl[i]) sl[i] = v; break; }
          case DK_MIN_D: sd[i] = fmin(sd[i], ev.agg(a, r)); break;
          case DK_MAX_D: sd[i] = fmax(sd[i], ev.agg(a, r)); break;
        }
      }
    }
  }
  __syncthreads();
  const int lane = tid & 31, warp = tid >> 5, nw = T >> 5;
  for (int s = warp; s < S; s += nw) {
    const int g = s / NA, a = s % NA;
    // Rows of this group in this block (slot 0 is the hidden COUNT(*)).
    long long cnt = 0;
    for (int t = lane; t < T; t += 32) cnt += sl[(size_t)g * NA * T + t];
    for (int o = 16; o > 0; o >>= 1) cnt += __shfl_down_sync(0xffffffffu, cnt, o);
    cnt = __shfl_sync(0xffffffffu, cnt, 0);
    if (cnt == 0) continue;
    const uint8_t k = ds.kind[a];
    double v;
    if (k == DK_COUNT || k == DK_SUM_I || k == DK_MIN_I || k == DK_MAX_I) {
      long long x = k == DK_MIN_I ? kLLMax : k == DK_MAX_I ? kLLMin : 0;
      for (int t = lane; t < T; t += 32) {
        const long long y = sl[(size_t)s * T + t];
        x = k == DK_MIN_I ? (y < x ? y : x) : k == DK_MAX_I ? (y > x ? y : x) : x + y;
      }
      for (int o = 16; o > 0; o >>= 1) {
        const long long y = __shfl_down_sync(0xffffffffu, x, o);
        x = k == DK_MIN_I ? (y < x ? y : x) : k == DK_MAX_I ? (y > x ? y : x) : x + y;
      }
      v = plan.aggBits[a] ? asBits(x) : (double)x;
    } else {
      const uint8_t op = k == DK_MIN_D ? AGG_MIN : k == DK_MAX_D ? AGG_MAX : AGG_SUM;
      v = aggIdentity(op);
      for (int t = lane; t < T; t += 32) v = combine(op, v, sd[(size_t)s * T + t]);
      for (int o = 16; o > 0; o >>= 1) v = combine(op, v, __shfl_down_sync(0xffffffffu, v, o));
    }
    if (lane == 0) atomicCombineB(plan.aggBits[a], a == 0 ? AGG_SUM : ev.op(a), &gacc[(size_t)g * NA + a], v);
  }
}

// Dense grouped aggregation (see DenseSpec). Shared-memory copies are selected by warp, so the 32
// lanes of a warp contend only with each other, and hot groups (skewed keys) spread over copies.
template <class Ev>
__device__ void aggDenseBody(const ExprEnv& env, const AggPlan& plan, const DenseSpec& ds, const BatchSet& bs, double* gacc) {
  // Generated code knows the mode: the other variant is dead code (keeps NVRTC compile time down).
  if (Ev::kDensePriv == 1 || (Ev::kDensePriv == 0 && ds.priv)) {
    aggDensePrivBody<Ev>(env, plan, ds, bs, gacc);
    return;
  }
  if (Ev::kDensePriv == 1) return;
  __shared__ ColView scols[kMaxCols];
  __shared__ int64_t sn;
  __shared__ const uint32_t* sdel;
  Ev ev(env, plan);
  const int NK = ev.nkeys(), NA = ev.naggs();
  const int C = ds.copies;
  const int64_t G = ds.G;
  if (C > 0) {
    for (int c = 0; c < C; ++c) {
      unsigned char* cb = gdbSmem + (size_t)c * ds.copyBytes;
      for (int a = 0; a < NA; ++a) {
        unsigned char* arr = cb + ds.slotOff[a];
        for (int64_t g = threadIdx.x; g < G; g += blockDim.x) {
          switch (ds.kind[a]) {
            case DK_COUNT: ((unsigned*)arr)[g] = 0u; break;
            case DK_SUM_I: ((long long*)arr)[g] = 0; break;
            case DK_SUM_D: ((double*)arr)[g] = 0.0; break;
            case DK_MIN_I: ((long long*)arr)[g] = kLLMax; break;
            case DK_MAX_I: ((long long*)arr)[g] = kLLMin; break;
            case DK_MIN_D: ((double*)arr)[g] = aggIdentity(AGG_MIN); break;
            case DK_MAX_D: ((double*)arr)[g] = aggIdentity(AGG_MAX); break;
          }
        }
      }
    }
  }
  const int warp = threadIdx.x >> 5;
  unsigned char* mine = C > 0 ? gdbSmem + (size_t)(warp % C) * ds.copyBytes : nullptr;
  for (int b = 0; b < bs.nb; ++b) {
    loadBatch(bs, b, scols, &sn, &sdel);
    ev.bind(scols);
    const int64_t n = sn;
    const uint32_t* del = sdel;
    for (int64_t r = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; r < n; r += (int64_t)gridDim.x * blockDim.x) {
      if (isDeleted(del, r)) continue;
      if (ev.hasWhere() && !ev.where(r)) continue;
      long long g = 0;
#pragma unroll
      for (int k = 0; k < Ev::kKeys; ++k)
        if (k < NK) g += (ev.ikey(k, r) - ds.kmin[k]) * ds.stride[k];
      if (C > 0) {
#pragma unroll
        for (int a = 0; a < Ev::kAggs; ++a) {
          if (a >= NA) break;
          unsigned char* arr = mine + ds.slotOff[a];
          switch (ev.dkind(a, ds)) {
            case DK_COUNT: atomicAdd(&((unsigned*)arr)[g], 1u); break;
            case DK_SUM_I: atomicAdd(&((unsigned long long*)arr)[g], (unsigned long long)ipart(plan, a, ev.iagg(a, r))); break;
            case DK_SUM_D: {
              const double v = ev.agg(a, r);
              if (v == v) atomicAdd(&((double*)arr)[g], v);
              break;
            }
            case DK_MIN_I: atomicMin(&((long long*)arr)[g], ipart(plan, a, ev.iagg(a, r))); break;
            case DK_MAX_I: atomicMax(&((long long*)arr)[g], ipart(plan, a, ev.iagg(a, r))); break;
            case DK_MIN_D: atomicCombine(AGG_MIN, &((double*)arr)[g], ev.agg(a, r)); break;
            case DK_MAX_D: atomicCombine(AGG_MAX, &((double*)arr)[g], ev.agg(a, r)); break;
          }
        }
      } else {
        double* row = gacc + g * NA;
#pragma unroll
        for (int a = 0; a < Ev::kAggs; ++a) {
          if (a >= NA) break;
          switch (ev.dkind(a, ds)) {
            case DK_COUNT: atomicAdd(&row[a], 1.0); break;
            case DK_SUM_I:
              if (plan.aggBits[a]) atomicCombineB(true, AGG_SUM, &row[a], asBits(ipart(plan, a, ev.iagg(a, r))));
              else atomicAdd(&row[a], (double)ipart(plan, a, ev.iagg(a, r)));
              break;
            case DK_SUM_D: atomicCombine(AGG_SUM, &row[a], ev.agg(a, r)); break;
            case DK_MIN_I: case DK_MIN_D:
              if (plan.aggBits[a]) atomicMin((long long*)&row[a], ipart(plan, a, ev.iagg(a, r)));
              else atomicCombine(AGG_MIN, &row[a], ev.agg(a, r));
              break;
            case DK_MAX_I: case DK_MAX_D:
              if (plan.aggBits[a]) atomicMax((long long*)&row[a], ipart(plan, a, ev.iagg(a, r)));
              else atomicCombine(AGG_MAX, &row[a], ev.agg(a, r));
              break;
          }
        }
      }
    }
  }
  if (C == 0) return;
  __syncthreads();
  // Merge the copies and flush each non-empty group once per block.
  for (int64_t g = threadIdx.x; g < G; g += blockDim.x) {
    unsigned long long cnt = 0;
    for (int c = 0; c < C; ++c) cnt += ((unsigned*)(gdbSmem + (size_t)c * ds.copyBytes + ds.slotOff[0]))[g];
    if (cnt == 0) continue;
    double* row = gacc + g * NA;
    atomicAdd(&row[0], (double)cnt);
    for (int a = 1; a < NA; ++a) {
      const uint8_t k = ds.kind[a];
      double v;
      if (k == DK_SUM_I || k == DK_MIN_I || k == DK_MAX_I) {
        long long x = k == DK_SUM_I ? 0 : k == DK_MIN_I ? kLLMax : kLLMin;
        for (int c = 0; c < C; ++c) {
          long long y = ((long long*)(gdbSmem + (size_t)c * ds.copyBytes + ds.slotOff[a]))[g];
          x = k == DK_SUM_I ? x + y : k == DK_MIN_I ? (y < x ? y : x) : (y > x ? y : x);
        }
        v = plan.aggBits[a] ? asBits(x) : (double)x;
      } else {
        v = aggIdentity(k == DK_MIN_D ? AGG_MIN : k == DK_MAX_D ? AGG_MAX : AGG_SUM);
        for (int c = 0; c < C; ++c) {
          double y = ((double*)(gdbSmem + (size_t)c * ds.copyBytes + ds.slotOff[a]))[g];
          v = k == DK_MIN_D ? fmin(v, y) : k == DK_MAX_D ? fmax(v, y) : v + y;
        }
      }
      atomicCombineB(plan.aggBits[a], ev.op(a), &row[a], v);
    }
  }
}

// Row filter for one batch (columns come from env.cols).
template <class Ev>
__device__ void flagBody(const ExprEnv& env, const AggPlan& plan, int64_t n, const uint32_t* del, uint8_t* flags) {
  Ev ev(env, plan);
  ev.bind(ev.envCols(env));
  const int64_t stride = (int64_t)gridDim.x * blockDim.x;
  for (int64_t r0 = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; r0 < n; r0 += 4 * stride) {
    bool f[4];
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      const int64_t r = r0 + u * stride;
      f[u] = r < n && !isDeleted(del, r) && (!ev.hasWhere() || ev.where(r));
    }
#pragma unroll
    for (int u = 0; u < 4; ++u)
      if (r0 + u * stride < n) flags[r0 + u * stride] = f[u];
  }
}

// Row filter over many batches in one launch: batch b's flags go to flags[boff[b] + r].
template <class Ev>
__device__ void flagMultiBody(const ExprEnv& env, const AggPlan& plan, const BatchSet& bs, const int64_t* boff, uint8_t* flags) {
  __shared__ ColView scols[kMaxCols];
  __shared__ int64_t sn;
  __shared__ const uint32_t* sdel;
  Ev ev(env, plan);
  for (int b = 0; b < bs.nb; ++b) {
    loadBatch(bs, b, scols, &sn, &sdel);
    ev.bind(scols);
    const int64_t n = sn;
    const uint32_t* del = sdel;
    uint8_t* out = flags + boff[b];
    for (int64_t r = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; r < n; r += (int64_t)gridDim.x * blockDim.x)
      out[r] = !isDeleted(del, r) && (!ev.hasWhere() || ev.where(r));
  }
}

}  // namespace gdb

// Kernel entry points. gpu.cu instantiates all of them with the interpreter; jit.cu instantiates
// only the one a query needs, with generated code (extern "C" so it can be looked up by name).
#define GDB_KERNEL_NOKEYS(EV, LINKAGE)                                                                            \
  LINKAGE __global__ void gdb_aggNoKeys(const gdb::ExprEnv env, const gdb::AggPlan plan, const gdb::BatchSet bs,   \
                                        double* out) {                                                             \
    gdb::aggNoKeysBody<EV>(env, plan, bs, out);                                                                    \
  }
#define GDB_KERNEL_GROUPED(EV, LINKAGE)                                                                           \
  LINKAGE __global__ void gdb_aggGrouped(const gdb::ExprEnv env, const gdb::AggPlan plan, const gdb::BatchSet bs,  \
                                         gdb::GTable gt, int W) {                                                  \
    gdb::aggGroupedBody<EV>(env, plan, bs, gt, W);                                                                 \
  }
#define GDB_KERNEL_DENSE(EV, LINKAGE)                                                                             \
  LINKAGE __global__ void gdb_aggDense(const gdb::ExprEnv env, const gdb::AggPlan plan, const gdb::DenseSpec ds,   \
                                       const gdb::BatchSet bs, double* gacc) {                                     \
    gdb::aggDenseBody<EV>(env, plan, ds, bs, gacc);                                                                \
  }
#define GDB_KERNEL_FLAG(EV, LINKAGE)                                                                              \
  LINKAGE __global__ void gdb_flag(const gdb::ExprEnv env, const gdb::AggPlan plan, int64_t n, const uint32_t* del, \
                                   uint8_t* flags) {                                                               \
    gdb::flagBody<EV>(env, plan, n, del, flags);                                                                   \
  }
#define GDB_KERNEL_FLAGM(EV, LINKAGE)                                                                             \
  LINKAGE __global__ void gdb_flagMulti(const gdb::ExprEnv env, const gdb::AggPlan plan, const gdb::BatchSet bs,   \
                                        const int64_t* boff, uint8_t* flags) {                                     \
    gdb::flagMultiBody<EV>(env, plan, bs, boff, flags);                                                            \
  }
#define GDB_DEFINE_KERNELS(EV, LINKAGE) \
  GDB_KERNEL_NOKEYS(EV, LINKAGE)        \
  GDB_KERNEL_GROUPED(EV, LINKAGE)       \
  GDB_KERNEL_DENSE(EV, LINKAGE)         \
  GDB_KERNEL_FLAG(EV, LINKAGE)          \
  GDB_KERNEL_FLAGM(EV, LINKAGE)
