// cuVS CAGRA through its C API, resolved with dlopen so warpstore builds and runs without cuVS. The
// headers are only needed for the struct layouts (GDB_HAVE_CUVS); without them this is a stub.
#include "cagra.h"
#include "common.h"

#include <dlfcn.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <type_traits>

#ifdef GDB_HAVE_CUVS
#include <cuvs/core/c_api.h>
#include <cuvs/core/dataset.h>
#include <cuvs/neighbors/cagra.h>
#endif

namespace gdb {

#ifndef GDB_HAVE_CUVS

bool cagraAvailable() { return false; }
std::string cagraStatus() { return "built without cuVS headers"; }
std::shared_ptr<CagraGraph> cagraBuild(const float*, int64_t, int, bool, int, std::string& err) {
  err = "cuVS not available";
  return nullptr;
}
bool cagraSearch(const std::vector<CagraGraph*>&, const std::vector<const uint32_t*>&, const float*, int, int, int,
                 std::vector<CagraHit>&, std::string& err) {
  err = "cuVS not available";
  return false;
}
void cagraWarm(int, bool) {}

#else

namespace {

struct Api {
  cuvsError_t (*resourcesCreate)(cuvsResources_t*);
  const char* (*lastError)();
  cuvsError_t (*streamSync)(cuvsResources_t);
  cuvsError_t (*datasetPaddedView)(cuvsResources_t, DLManagedTensor*, cuvsDataset_t*);
  cuvsError_t (*datasetDestroy)(cuvsDataset_t);
  cuvsError_t (*indexParamsCreate)(cuvsCagraIndexParams_t*);
  cuvsError_t (*indexParamsDestroy)(cuvsCagraIndexParams_t);
  cuvsError_t (*indexCreate)(cuvsCagraIndex_t*);
  cuvsError_t (*indexDestroy)(cuvsCagraIndex_t);
  cuvsError_t (*build)(cuvsResources_t, cuvsCagraIndexParams_t, cuvsDataset_t, cuvsCagraIndex_t);
  cuvsError_t (*searchParamsCreate)(cuvsCagraSearchParams_t*);
  cuvsError_t (*searchParamsDestroy)(cuvsCagraSearchParams_t);
  cuvsError_t (*search)(cuvsResources_t, cuvsCagraSearchParams_t, cuvsCagraIndex_t, DLManagedTensor*, DLManagedTensor*, DLManagedTensor*, cuvsFilter);
  cuvsError_t (*searchMulti)(cuvsResources_t, cuvsCagraSearchParams_t, uint32_t, cuvsCagraIndex_t*, DLManagedTensor*, DLManagedTensor*,
                             DLManagedTensor*, DLManagedTensor*, cuvsFilter*);
};
Api api;
bool loaded = false;
std::string status = "not loaded";

bool load() {
  static std::once_flag once;
  std::call_once(once, [] {
    std::string dir;
    if (const char* e = envSetting("CUVS_DIR")) dir = e;
    else if (const char* h = getenv("HOME")) dir = std::string(h) + "/.local/share/warpstore-cuvs";
    // Dependencies first (RTLD_GLOBAL, so later libraries bind to these by soname).
    const char* libs[] = {"rapids_logger/lib64/librapids_logger.so", "librmm/lib64/librmm.so", "nvidia/nvjitlink/lib/libnvJitLink.so.12",
                          "nvidia/nccl/lib/libnccl.so.2", "nvidia/cublas/lib/libcublasLt.so.12", "nvidia/cublas/lib/libcublas.so.12",
                          "nvidia/curand/lib/libcurand.so.10", "nvidia/cusparse/lib/libcusparse.so.12",
                          "nvidia/cusolver/lib/libcusolver.so.11", "libkvikio/lib64/libkvikio.so", "libraft/lib64/libraft.so",
                          "libcuvs/lib64/libcuvs.so", "libcuvs/lib64/libcuvs_c.so"};
    void* h = nullptr;
    for (const char* l : libs) {
      const std::string p = dir + "/" + l;
      h = dlopen(p.c_str(), RTLD_NOW | RTLD_GLOBAL);
      if (!h) {
        status = std::string("cannot load ") + p + ": " + dlerror();
        return;
      }
    }
    bool ok = true;
    auto sym = [&](auto& fp, const char* name) {
      fp = reinterpret_cast<std::remove_reference_t<decltype(fp)>>(dlsym(h, name));
      ok = ok && fp;
    };
    sym(api.resourcesCreate, "cuvsResourcesCreate");
    sym(api.lastError, "cuvsGetLastErrorText");
    sym(api.streamSync, "cuvsStreamSync");
    sym(api.datasetPaddedView, "cuvsDatasetMakePaddedView");
    sym(api.datasetDestroy, "cuvsDatasetDestroy");
    sym(api.indexParamsCreate, "cuvsCagraIndexParamsCreate");
    sym(api.indexParamsDestroy, "cuvsCagraIndexParamsDestroy");
    sym(api.indexCreate, "cuvsCagraIndexCreate");
    sym(api.indexDestroy, "cuvsCagraIndexDestroy");
    sym(api.build, "cuvsCagraBuild");
    sym(api.searchParamsCreate, "cuvsCagraSearchParamsCreate");
    sym(api.searchParamsDestroy, "cuvsCagraSearchParamsDestroy");
    sym(api.search, "cuvsCagraSearch");
    sym(api.searchMulti, "cuvsCagraSearchMultiPartition");
    if (!ok) {
      status = "cuVS found in " + dir + " but its C API is missing functions (version mismatch?)";
      return;
    }
    loaded = true;
    status = "cuVS CAGRA from " + dir;
  });
  return loaded;
}

// One cuVS handle per thread (a handle owns a stream and is not thread-safe).
cuvsResources_t res() {
  thread_local cuvsResources_t r = 0;
  thread_local bool made = false;
  if (!made) {
    made = api.resourcesCreate(&r) == CUVS_SUCCESS;
    if (!made) return 0;
  }
  return r;
}

DLManagedTensor tensor2(void* p, DLDataType t, int64_t* shape, int ndim) {
  DLManagedTensor m{};
  m.dl_tensor.data = p;
  m.dl_tensor.device = DLDevice{kDLCUDA, 0};
  m.dl_tensor.ndim = ndim;
  m.dl_tensor.dtype = t;
  m.dl_tensor.shape = shape;
  return m;
}

}  // namespace

struct CagraGraph {
  int64_t n = 0;
  int dim = 0;
  int64_t shape[2] = {0, 0};
  DLManagedTensor data{};
  cuvsDataset_t view = nullptr;
  cuvsCagraIndex_t index = nullptr;
  ~CagraGraph() {
    if (index) api.indexDestroy(index);
    if (view) api.datasetDestroy(view);
  }
};

bool cagraAvailable() { return load(); }
std::string cagraStatus() {
  load();
  return status;
}

std::shared_ptr<CagraGraph> cagraBuild(const float* devData, int64_t n, int dim, bool dot, int graphDegree, std::string& err) {
  if (!load()) {
    err = status;
    return nullptr;
  }
  if (dim % 4 != 0) {
    err = "CAGRA needs a dimension divisible by 4";
    return nullptr;
  }
  auto g = std::make_shared<CagraGraph>();
  g->n = n;
  g->dim = dim;
  g->shape[0] = n;
  g->shape[1] = dim;
  g->data = tensor2((void*)devData, DLDataType{kDLFloat, 32, 1}, g->shape, 2);
  cuvsResources_t r = res();
  cuvsCagraIndexParams_t ip = nullptr;
  bool ok = r && api.datasetPaddedView(r, &g->data, &g->view) == CUVS_SUCCESS && api.indexParamsCreate(&ip) == CUVS_SUCCESS &&
            api.indexCreate(&g->index) == CUVS_SUCCESS;
  if (ok) {
    ip->metric = dot ? InnerProduct : L2Expanded;
    ip->graph_degree = (size_t)std::max(8, graphDegree);
    ip->intermediate_graph_degree = ip->graph_degree * 2;
    ok = api.build(r, ip, g->view, g->index) == CUVS_SUCCESS && api.streamSync(r) == CUVS_SUCCESS;
  }
  if (ip) api.indexParamsDestroy(ip);
  if (!ok) {
    err = api.lastError ? api.lastError() : "CAGRA build failed";
    return nullptr;
  }
  return g;
}

bool cagraSearch(const std::vector<CagraGraph*>& graphs, const std::vector<const uint32_t*>& keepBits, const float* query, int dim,
                 int k, int itopk, std::vector<CagraHit>& out, std::string& err) {
  out.clear();
  if (graphs.empty()) return true;
  if (!load()) {
    err = status;
    return false;
  }
  cuvsResources_t r = res();
  if (!r) {
    err = "cannot create a cuVS handle";
    return false;
  }
  float* dq = nullptr;
  uint32_t *dn = nullptr, *dp = nullptr;
  float* dd = nullptr;
  cudaMalloc(&dq, (size_t)dim * 4);
  cudaMalloc(&dn, (size_t)k * 4);
  cudaMalloc(&dp, (size_t)k * 4);
  cudaMalloc(&dd, (size_t)k * 4);
  cudaMemcpy(dq, query, (size_t)dim * 4, cudaMemcpyHostToDevice);
  int64_t qs[2] = {1, dim}, os[2] = {1, k};
  DLManagedTensor tq = tensor2(dq, DLDataType{kDLFloat, 32, 1}, qs, 2);
  DLManagedTensor tn = tensor2(dn, DLDataType{kDLUInt, 32, 1}, os, 2);
  DLManagedTensor tp = tensor2(dp, DLDataType{kDLUInt, 32, 1}, os, 2);
  DLManagedTensor td = tensor2(dd, DLDataType{kDLFloat, 32, 1}, os, 2);
  // Keep-bitsets as 1-D uint32 tensors (they must outlive the call).
  std::vector<int64_t> bshape(graphs.size());
  std::vector<DLManagedTensor> bts(graphs.size());
  std::vector<cuvsFilter> filters(graphs.size());
  for (size_t i = 0; i < graphs.size(); ++i) {
    const uint32_t* b = i < keepBits.size() ? keepBits[i] : nullptr;
    if (!b) {
      filters[i] = cuvsFilter{0, NO_FILTER};
      continue;
    }
    bshape[i] = (graphs[i]->n + 31) / 32;
    bts[i] = tensor2((void*)b, DLDataType{kDLUInt, 32, 1}, &bshape[i], 1);
    filters[i] = cuvsFilter{(uintptr_t)&bts[i], BITSET};
  }
  cuvsCagraSearchParams_t sp = nullptr;
  bool ok = api.searchParamsCreate(&sp) == CUVS_SUCCESS;
  if (ok) {
    sp->itopk_size = (size_t)std::max(itopk, k);
    if (graphs.size() == 1) {
      ok = api.search(r, sp, graphs[0]->index, &tq, &tn, &td, filters[0]) == CUVS_SUCCESS;
      if (ok) cudaMemset(dp, 0, (size_t)k * 4);
    } else {
      std::vector<cuvsCagraIndex_t> idx;
      for (auto* g : graphs) idx.push_back(g->index);
      ok = api.searchMulti(r, sp, (uint32_t)idx.size(), idx.data(), &tq, &tp, &tn, &td, filters.data()) == CUVS_SUCCESS;
    }
    ok = ok && api.streamSync(r) == CUVS_SUCCESS;
  }
  if (sp) api.searchParamsDestroy(sp);
  if (ok) {
    std::vector<uint32_t> hn(k), hp(k);
    cudaMemcpy(hn.data(), dn, (size_t)k * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(hp.data(), dp, (size_t)k * 4, cudaMemcpyDeviceToHost);
    for (int i = 0; i < k; ++i)
      if (hp[i] < graphs.size() && hn[i] < (uint64_t)graphs[hp[i]]->n) out.push_back({(int)hp[i], (int64_t)hn[i]});
  } else {
    err = api.lastError ? api.lastError() : "CAGRA search failed";
  }
  cudaFree(dq);
  cudaFree(dn);
  cudaFree(dp);
  cudaFree(dd);
  return ok;
}

void cagraWarm(int dim, bool dot) {
  if (!load() || dim % 4 != 0) return;
  const int64_t n = 4096;
  std::vector<float> h((size_t)n * dim);
  for (size_t i = 0; i < h.size(); ++i) h[i] = (float)((i * 2654435761u) % 1000) / 1000.f - 0.5f;
  float* d = nullptr;
  if (cudaMalloc(&d, h.size() * 4) != cudaSuccess) return;
  cudaMemcpy(d, h.data(), h.size() * 4, cudaMemcpyHostToDevice);
  std::string err;
  auto g = cagraBuild(d, n, dim, dot, 32, err);
  if (g) {
    std::vector<CagraHit> hits;
    cagraSearch({g.get()}, {nullptr}, h.data(), dim, 10, 64, hits, err);
    auto g2 = cagraBuild(d, n, dim, dot, 32, err);  // the multi-partition kernels too
    if (g2) cagraSearch({g.get(), g2.get()}, {nullptr, nullptr}, h.data(), dim, 10, 64, hits, err);
  }
  g.reset();
  cudaFree(d);
}

#endif

}  // namespace gdb
