// Host-callable GPU primitives. All CUDA runtime usage is confined to .cu translation units,
// which are compiled with --default-stream per-thread so each connection thread gets its own stream.
#pragma once
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>
#include "expr.h"

namespace gdb {

void gpuInit();
int gpuSmCount();
int gpuArch();  // e.g. 86
std::string gpuDescribe();
void* devAlloc(size_t bytes);         // persistent (cudaMalloc)
void devFree(void* p);
// Tiered column storage: VRAM while it fits under the limit (--vram-limit), otherwise the host
// mirror itself is pinned and mapped so kernels read it over PCIe (tables larger than VRAM).
void gpuSetVramLimit(int64_t bytes);
int64_t gpuVramUsed();
double gpuVramFill();  // fraction of the VRAM budget (or of the device) in use
void* placeColumn(void* host, size_t bytes, bool& mapped, bool hostOnly = false);  // hostOnly: VRAM = NEVER
void releaseColumn(void* dev, void* host, size_t bytes, bool mapped);
// VRAM cache moves (Engine::vramBalance): a VRAM copy of host bytes (nullptr if it doesn't fit),
// freeing one, mapping host bytes for zero-copy reads over PCIe, unmapping them.
void* vramAcquire(const void* host, size_t bytes);
void vramRelease(void* dev, size_t bytes);
void* hostMap(void* host, size_t bytes);
void hostUnmap(void* host);
int64_t gpuVramLimit();  // INT64_MAX: no --vram-limit
int64_t gpuVramFree();   // free device memory
void* tmpAlloc(size_t bytes);         // stream-ordered pool allocation
void tmpFree(void* p);
void h2d(void* dst, const void* src, size_t bytes);
void d2h(void* dst, const void* src, size_t bytes);
void d2d(void* dst, const void* src, size_t bytes);
void gpuSync();

// RAII temp device array.
template <class T>
struct DevVec {
  T* p = nullptr;
  int64_t n = 0;
  DevVec() = default;
  explicit DevVec(int64_t count) : n(count) { p = count ? (T*)tmpAlloc(sizeof(T) * count) : nullptr; }
  DevVec(const DevVec&) = delete;
  DevVec& operator=(const DevVec&) = delete;
  DevVec(DevVec&& o) noexcept : p(o.p), n(o.n) { o.p = nullptr; o.n = 0; }
  DevVec& operator=(DevVec&& o) noexcept {
    if (this != &o) { reset(); p = o.p; n = o.n; o.p = nullptr; o.n = 0; }
    return *this;
  }
  ~DevVec() { reset(); }
  void reset() { if (p) tmpFree(p); p = nullptr; n = 0; }
  std::vector<T> download(int64_t count = -1) const {
    std::vector<T> v(count < 0 ? n : count);
    if (!v.empty()) d2h(v.data(), p, sizeof(T) * v.size());
    return v;
  }
};

struct JitProgram;  // jit.h

// GPU aggregation state for one query. Batches (segments, write buffers or a joined relation) are
// collected with addBatch() and aggregated by a single kernel launch in run(), either with a
// runtime-compiled kernel (jit != nullptr) or the bytecode interpreter. Grouped aggregation uses a
// hash table, or dense arrays when `dense` is given (small known key ranges).
class GpuAggregator {
 public:
  GpuAggregator(const AggPlan& plan, int64_t capacity, const DenseSpec* dense = nullptr);
  ~GpuAggregator();
  void addBatch(const ExprEnv& env, int ncols, int64_t n, const uint32_t* del);
  void run(const ExprEnv& env, const JitProgram* jit);
  // Aggregates the batches added so far into the running result and drops them (a scan window: the
  // batches' column data can go); run() then does the rest.
  void runWindow(const ExprEnv& env, const JitProgram* jit);
  int batches() const { return batchesDone_ + (int)bn_.size(); }
  int windows() const { return windows_; }
  int64_t rowsIn() const;
  bool dense() const { return dense_.G > 0; }
  bool overflowed();
  // Returns number of groups; keys is groups*nkeys, accs is groups*naggs.
  int64_t fetch(std::vector<double>& keys, std::vector<double>& accs);
  // Without GROUP BY, integer SUM slots hold the exact int64 sum as bits (1 per such slot).
  std::vector<uint8_t> sumBitsSlots() const;
  const std::vector<int64_t>& sumHigh() const { return hi_; }  // ... and their high 64 bits
  // Same, but leaves the groups in VRAM, column-major: keys[k * g + i], accs[a * g + i].
  int64_t compact(DevVec<double>& keys, DevVec<double>& accs);
  int64_t capacity() const { return cap_; }
  int64_t groupCount();

 private:
  void finalizeDense();
  AggPlan plan_;
  DenseSpec dense_;
  bool finalized_ = false;
  int64_t cap_;
  unsigned long long* gkeys_ = nullptr;
  double* gkv_ = nullptr;
  double* gacc_ = nullptr;
  std::vector<int64_t> hi_;
  int* flags_ = nullptr;  // [0]=overflow
  unsigned long long* count_ = nullptr;
  uint8_t* block_ = nullptr;  // no GROUP BY: one allocation for the accumulators + batch descriptor
  int ncols_ = 0;
  std::vector<int64_t> bn_;
  std::vector<const uint32_t*> bdel_;
  std::vector<ColView> bcols_;
  int64_t rowsDone_ = 0;
  int batchesDone_ = 0, windows_ = 0;
};

// Row selection: indices (int32) of rows passing `where` and not deleted. Returns count.
int64_t gpuSelect(const ExprEnv& env, bool hasWhere, Prog where, int64_t n, const uint32_t* del, int32_t* outIdx,
                  const JitProgram* jit = nullptr);
// Row selection over many batches at once (one launch, one synchronization). Batch b holds bn[b]
// rows; its columns are bcols[b * ncols .. ]. Returns the selected rows as global indices into the
// concatenation of the batches (ascending) and fills perBatch with each batch's count.
int64_t gpuSelectMulti(const ExprEnv& env, bool hasWhere, Prog where, const std::vector<int64_t>& bn,
                       const std::vector<const uint32_t*>& bdel, const std::vector<ColView>& bcols, int ncols,
                       DevVec<int32_t>& outIdx, std::vector<int64_t>& perBatch, const JitProgram* jit = nullptr);
// dst[j] = row idx[j] of the concatenated batches (src[b] = batch b's column, boff = batch start offsets).
void gpuGatherMulti(const std::vector<ColView>& src, const std::vector<int64_t>& boff, PType t, int esize,
                    const int32_t* idx, int64_t m, void* dst);
// out[i] = map[in[i]] for dictionary codes (NULL stays NULL, codes outside the map -> -2).
void gpuRemapCodes(const int32_t* in, int64_t n, const int32_t* map, int64_t mapN, int32_t* out);
// dst[j] = (double) of the int64 stored bit-for-bit in src[j]
void gpuBitsToDouble(const double* src, int64_t n, double* dst);
// Exact int64 of an integer column program (single OP_COL); other programs are rounded.
void gpuEvalAtInt(const ExprEnv& env, Prog prog, const int32_t* idx, int64_t m, int64_t* out);
// Distinct values of an integer expression at idx (or rows 0..m-1) within [base, base + span): a bitmap
// built on the GPU, returned as the sorted distinct values (NULLs skipped).
std::vector<int64_t> gpuDistinctInts(const ExprEnv& env, Prog prog, const int32_t* idx, int64_t m, int64_t base, int64_t span);
// A column of a GPU-built temp table from compacted GROUP BY results (keys g x nk, accs g x na, on
// the device): see TempColSpec. Returns a cudaMalloc'd column (devFree); mn / mx of an int column.
struct TempColSpec {
  int8_t src = 0;   // 0: GROUP BY key idx, 1: aggregate slot idx
  int idx = 0, hi = -1, cnt = -1;  // hi: high-half slot of a split BIGINT SUM; cnt: non-NULL count slot
  uint8_t kind = 0; // 0 value (double / key), 1 int64 bits, 2 split sum, 3 count, 4 average
  uint8_t keyInt = 0, sumBits = 0;  // key stored as int64 bits; (average) the sum slot holds bits
  uint8_t out = 2;  // 0 int32, 1 int64, 2 double
  double div = 1;   // DECIMAL raw sums: 10^scale
  double mul = 1;   // constant factor (an item like 0.5 * SUM(x)); not 1: a double result
};
void* gpuTempColumn(const double* keys, int nk, const double* accs, int na, int64_t g, const TempColSpec& s, double& mn, double& mx);
// out[j] = eval(prog, idx ? idx[j] : j)
void gpuEvalAt(const ExprEnv& env, Prog prog, const int32_t* idx, int64_t m, double* out);
// dst[j] = src[idx[j]] for a column of the given physical type
void gpuGather(const void* src, PType t, const int32_t* idx, int64_t m, void* dst);
// Gather from a compressed segment column (ColView::enc != 0) into raw values of type t.
void gpuGatherEnc(const ColView& v, PType t, const int32_t* idx, int64_t m, void* dst);
// Exact key bitmap for join filters: bit (key - lo) for every key in [lo, lo + nbits).
void gpuKeyBitmap(ColView key, int64_t n, int64_t lo, int64_t nbits, uint32_t* bits);
// Set deleted bits for listed rows.
void gpuSetBits(uint32_t* bits, const int32_t* idx, int64_t m);
// Inner equi hash join; returns matching (build,probe) row index pairs.
int64_t gpuHashJoin(ColView buildKey, int64_t nb, ColView probeKey, int64_t np, DevVec<int32_t>& outBuild,
                    DevVec<int32_t>& outProbe);
// Partition-wise inner equi hash join. bOff / pOff are row offsets of each partition in the build /
// probe inputs (sizes P+1). The build side has either the same P partitions (co-located join: each
// probe partition only meets its own build partition) or a single one (shared by all). Output pairs
// are grouped by probe partition; outOff (optional) receives their offsets.
int64_t gpuHashJoinParts(ColView buildKey, const std::vector<int64_t>& bOff, ColView probeKey,
                         const std::vector<int64_t>& pOff, DevVec<int32_t>& outBuild, DevVec<int32_t>& outProbe,
                         std::vector<int64_t>* outOff);
// Outer hash join: every probe row appears (build index -1 when unmatched); `full` also appends build
// rows that matched nothing (probe index -1).
int64_t gpuHashJoinOuter(ColView buildKey, int64_t nb, ColView probeKey, int64_t np, DevVec<int32_t>& outBuild,
                         DevVec<int32_t>& outProbe, bool full);
int64_t gpuCrossJoin(int64_t na, int64_t nb, DevVec<int32_t>& outA, DevVec<int32_t>& outB);
// Like gpuGather, but index -1 writes NULL.
void gpuGatherNull(const void* src, PType t, const int32_t* idx, int64_t m, void* dst);
// Gather of fixed-size elements (multiple of 4 bytes, e.g. VECTOR rows); index -1 gives a NULL vector.
void gpuGatherBytes(const void* src, int esize, const int32_t* idx, int64_t m, void* dst, bool nulls);
// Stable multi-key sort permutation (keys[0] is the most significant).
DevVec<int32_t> gpuSortPerm(const std::vector<std::pair<const double*, bool>>& keys, int64_t m);
// keys[j] = rank[(int)keys[j]]  (dictionary code -> lexicographic rank, for ORDER BY on strings)
void gpuMapRank(double* keys, const int32_t* rank, int64_t m);
// Convenience: sort permutation of a host column (used when flushing sorted segments).
std::vector<int32_t> gpuSortPermHost(const void* hostCol, PType t, int64_t n);

}  // namespace gdb
