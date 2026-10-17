// GPU graph vector index (NVIDIA cuVS CAGRA), loaded at runtime when cuVS is installed: one graph per
// segment over the segment's VRAM vector column (no copy), searched together in one call.
#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace gdb {

struct CagraGraph;  // opaque: a built index plus the dataset view it searches

// Loads cuVS once ($GPUDB_CUVS_DIR, else ~/.local/share/gpudb-cuvs). False if unavailable.
bool cagraAvailable();
std::string cagraStatus();  // why it is unavailable, or the loaded location
// Graph over n x dim float32 rows at devData (device memory; dim % 4 == 0). dot: inner product,
// else L2. Null on failure (err says why).
std::shared_ptr<CagraGraph> cagraBuild(const float* devData, int64_t n, int dim, bool dot, int graphDegree, std::string& err);
struct CagraHit {
  int graph;   // index into the graphs passed to cagraSearch
  int64_t row; // row within that graph's dataset
};
// k nearest rows to the host query over all graphs. keepBits[i]: device bitset over graph i's rows
// (bit set = searchable), or null. itopk: search breadth (recall vs speed).
bool cagraSearch(const std::vector<CagraGraph*>& graphs, const std::vector<const uint32_t*>& keepBits, const float* query, int dim,
                 int k, int itopk, std::vector<CagraHit>& out, std::string& err);
// Compiles cuVS's kernels (its first search JIT-compiles for seconds): call once in the background.
void cagraWarm(int dim, bool dot);

}  // namespace gdb
