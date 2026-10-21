// GPU decompression (nvCOMP) and GPUDirect Storage (cuFile), both loaded at run time: warpstore builds
// and runs without them, and uses them when they are there.
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace gdb {

// nvCOMP's batched zstd (libnvcomp.so.5 from WARPSTORE_NVCOMP_DIR, ~/.local/share/warpstore-nvcomp
// (pip install --target ~/.local/share/warpstore-nvcomp nvidia-nvcomp-cu12) or the library path).
// WARPSTORE_GPU_DECOMPRESS=0 turns it off.
bool nvcompAvailable();
std::string nvcompStatus();
// Decompresses zstd frames on the GPU, on this thread's stream: frame i at in[i] (inLen[i] bytes, in
// VRAM) into out[i] (outLen[i] bytes, VRAM). Asynchronous: stream-ordered like a kernel launch.
void gpuZstdDecompress(const std::vector<const void*>& in, const std::vector<size_t>& inLen,
                       const std::vector<void*>& out, const std::vector<size_t>& outLen);

// GPUDirect Storage: reads from files straight into VRAM (libcufile from the CUDA toolkit). Used when
// the nvidia-fs driver is loaded (WARPSTORE_GPUDIRECT=auto, the default), always (on: cuFile's
// compatibility mode without nvidia-fs) or never (off).
bool gdsEnabled();
std::string gdsStatus();
void* gdsOpen(const std::string& path);  // a registered file handle, null if it can't be opened
void gdsClose(void* handle);
bool gdsRead(void* handle, void* dev, size_t len, uint64_t off);  // all len bytes into VRAM

}  // namespace gdb
