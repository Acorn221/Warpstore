// nvCOMP and cuFile through dlopen (see gpuio.h).
#include "gpuio.h"
#include <cufile.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <mutex>
#include "common.h"

namespace gdb {

namespace {
// ---- nvCOMP (declared here: only these two functions of its C API are used)
enum NvcompStatus : int { kNvcompSuccess = 0 };
struct NvcompZstdDecompressOpts {
  int backend;  // nvcompDecompressBackend_t: 0 = let nvCOMP choose
  char reserved[60];
};
struct Nvcomp {
  int (*tempSize)(size_t, size_t, NvcompZstdDecompressOpts, size_t*, size_t) = nullptr;
  int (*decompress)(const void* const*, const size_t*, const size_t*, size_t*, size_t, void*, size_t, void* const*,
                    NvcompZstdDecompressOpts, int*, cudaStream_t) = nullptr;
  bool ok = false;
  std::string status = "not loaded";
};
Nvcomp& nvcomp() {
  static Nvcomp n;
  static std::once_flag once;
  std::call_once(once, [] {
    const char* off = envSetting("GPU_DECOMPRESS");
    if (off && std::string(off) == "0") {
      n.status = "off (WARPSTORE_GPU_DECOMPRESS=0)";
      return;
    }
    std::vector<std::string> paths;
    if (const char* e = envSetting("NVCOMP_DIR")) paths.push_back(std::string(e) + "/nvidia/libnvcomp/lib64/libnvcomp.so.5"), paths.push_back(std::string(e) + "/libnvcomp.so.5");
    if (const char* h = getenv("HOME")) paths.push_back(std::string(h) + "/.local/share/warpstore-nvcomp/nvidia/libnvcomp/lib64/libnvcomp.so.5");
    paths.push_back("libnvcomp.so.5");
    void* h = nullptr;
    std::string where;
    for (auto& p : paths)
      if ((h = dlopen(p.c_str(), RTLD_NOW | RTLD_LOCAL))) {
        where = p;
        break;
      }
    if (!h) {
      n.status = "nvCOMP not found (pip install --target ~/.local/share/warpstore-nvcomp nvidia-nvcomp-cu12): CPU decompression";
      return;
    }
    n.tempSize = reinterpret_cast<decltype(n.tempSize)>(dlsym(h, "nvcompBatchedZstdDecompressGetTempSizeAsync"));
    n.decompress = reinterpret_cast<decltype(n.decompress)>(dlsym(h, "nvcompBatchedZstdDecompressAsync"));
    if (!n.tempSize || !n.decompress) {
      n.status = where + " lacks the nvCOMP 5 zstd API: CPU decompression";
      return;
    }
    n.ok = true;
    n.status = "nvCOMP zstd from " + where;
  });
  return n;
}

// ---- cuFile
struct CuFile {
  CUfileError_t (*driverOpen)() = nullptr;
  CUfileError_t (*handleRegister)(CUfileHandle_t*, CUfileDescr_t*) = nullptr;
  void (*handleDeregister)(CUfileHandle_t) = nullptr;
  ssize_t (*read)(CUfileHandle_t, void*, size_t, off_t, off_t) = nullptr;
  bool ok = false;
  std::string status = "off";
};
CuFile& cufile() {
  static CuFile c;
  static std::once_flag once;
  std::call_once(once, [] {
    const char* m = envSetting("GPUDIRECT");
    const std::string mode = m ? m : "auto";
    if (mode == "off") return;
    const bool nvfs = access("/proc/driver/nvidia-fs", F_OK) == 0;
    if (mode != "on" && !nvfs) {
      c.status = "off (the nvidia-fs driver is not loaded; WARPSTORE_GPUDIRECT=on uses cuFile's compatibility mode)";
      return;
    }
    void* h = dlopen("libcufile.so.0", RTLD_NOW | RTLD_LOCAL);
    if (!h) h = dlopen("libcufile.so", RTLD_NOW | RTLD_LOCAL);
    if (!h) {
      c.status = "off (libcufile not found)";
      return;
    }
    c.driverOpen = reinterpret_cast<decltype(c.driverOpen)>(dlsym(h, "cuFileDriverOpen"));
    c.handleRegister = reinterpret_cast<decltype(c.handleRegister)>(dlsym(h, "cuFileHandleRegister"));
    c.handleDeregister = reinterpret_cast<decltype(c.handleDeregister)>(dlsym(h, "cuFileHandleDeregister"));
    c.read = reinterpret_cast<decltype(c.read)>(dlsym(h, "cuFileRead"));
    if (!c.driverOpen || !c.handleRegister || !c.handleDeregister || !c.read) {
      c.status = "off (libcufile lacks the functions used)";
      return;
    }
    const CUfileError_t e = c.driverOpen();
    if (e.err != CU_FILE_SUCCESS) {
      c.status = "off (cuFileDriverOpen failed: " + std::to_string((int)e.err) + ")";
      return;
    }
    c.ok = true;
    c.status = nvfs ? "GPUDirect Storage (nvidia-fs)" : "cuFile compatibility mode (no nvidia-fs: reads go through host memory)";
  });
  return c;
}

struct GdsHandle {
  int fd;
  CUfileHandle_t h;
};
}  // namespace

bool nvcompAvailable() { return nvcomp().ok; }
std::string nvcompStatus() { return nvcomp().status; }

void gpuZstdDecompress(const std::vector<const void*>& in, const std::vector<size_t>& inLen, const std::vector<void*>& out,
                       const std::vector<size_t>& outLen) {
  const size_t n = in.size();
  if (!n) return;
  Nvcomp& nv = nvcomp();
  size_t maxOut = 0, total = 0;
  for (size_t x : outLen) maxOut = std::max(maxOut, x), total += x;
  NvcompZstdDecompressOpts opts{};
  size_t temp = 0;
  if (nv.tempSize(n, maxOut, opts, &temp, total) != kNvcompSuccess) throw SqlError("nvCOMP: cannot size the decompression", 1105);
  // Device arrays: input pointers, input sizes, output sizes, actual sizes, output pointers.
  std::vector<uint64_t> desc(5 * n);
  for (size_t i = 0; i < n; ++i) {
    desc[i] = (uint64_t)(uintptr_t)in[i];
    desc[n + i] = inLen[i];
    desc[2 * n + i] = outLen[i];
    desc[3 * n + i] = 0;
    desc[4 * n + i] = (uint64_t)(uintptr_t)out[i];
  }
  const cudaStream_t s = cudaStreamPerThread;
  uint8_t* d = nullptr;
  const size_t descBytes = desc.size() * 8, tempOff = (descBytes + 255) & ~(size_t)255;
  if (cudaMallocAsync((void**)&d, tempOff + std::max<size_t>(temp, 1), s) != cudaSuccess) {
    cudaGetLastError();
    throw SqlError("GPU out of memory", 1041);
  }
  // Through a pinned staging buffer (a copy from pageable memory would wait for the stream).
  static thread_local void* stage = nullptr;
  static thread_local size_t stageCap = 0;
  static thread_local cudaEvent_t stageEv = nullptr;
  if (stageEv) cudaEventSynchronize(stageEv);  // (the previous call's copy has long run)
  else cudaEventCreateWithFlags(&stageEv, cudaEventDisableTiming);
  if (stageCap < descBytes) {
    if (stage) cudaFreeHost(stage);
    stageCap = std::max(descBytes, (size_t)1 << 16);
    if (cudaMallocHost(&stage, stageCap) != cudaSuccess) {
      cudaGetLastError();
      stage = nullptr, stageCap = 0;
      throw SqlError("cannot allocate pinned memory", 1041);
    }
  }
  memcpy(stage, desc.data(), descBytes);
  cudaMemcpyAsync(d, stage, descBytes, cudaMemcpyHostToDevice, s);
  cudaEventRecord(stageEv, s);
  const uint64_t* dd = (const uint64_t*)d;
  const int st = nv.decompress((const void* const*)dd, (const size_t*)(dd + n), (const size_t*)(dd + 2 * n), (size_t*)(dd + 3 * n), n,
                               temp ? d + tempOff : nullptr, temp, (void* const*)(dd + 4 * n), opts, nullptr, s);
  cudaFreeAsync(d, s);
  if (st != kNvcompSuccess) throw SqlError("nvCOMP decompression failed (" + std::to_string(st) + ")", 1105);
}

bool gdsEnabled() { return cufile().ok; }
std::string gdsStatus() { return cufile().status; }

void* gdsOpen(const std::string& path) {
  CuFile& c = cufile();
  if (!c.ok) return nullptr;
  const int fd = ::open(path.c_str(), O_RDONLY | O_DIRECT | O_CLOEXEC);
  if (fd < 0) return nullptr;
  CUfileDescr_t d;
  memset(&d, 0, sizeof d);
  d.handle.fd = fd;
  d.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
  CUfileHandle_t h;
  if (c.handleRegister(&h, &d).err != CU_FILE_SUCCESS) {
    ::close(fd);
    return nullptr;
  }
  return new GdsHandle{fd, h};
}

void gdsClose(void* handle) {
  if (!handle) return;
  auto* g = (GdsHandle*)handle;
  cufile().handleDeregister(g->h);
  ::close(g->fd);
  delete g;
}

bool gdsRead(void* handle, void* dev, size_t len, uint64_t off) {
  auto* g = (GdsHandle*)handle;
  size_t done = 0;
  while (done < len) {
    const ssize_t k = cufile().read(g->h, dev, len - done, (off_t)(off + done), (off_t)done);
    if (k <= 0) return false;
    done += (size_t)k;
  }
  return true;
}

}  // namespace gdb
