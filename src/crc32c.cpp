// CRC32C (Castagnoli): SSE4.2's crc32 instruction where the CPU has it, else a table.
#include <nmmintrin.h>
#include <cstdint>
#include <cstring>
#include "common.h"

namespace gdb {

namespace {
__attribute__((target("sse4.2"))) uint32_t crcHw(const uint8_t* p, size_t n, uint32_t c) {
  uint64_t c64 = c;
  for (; n >= 8; p += 8, n -= 8) {
    uint64_t v;
    memcpy(&v, p, 8);
    c64 = _mm_crc32_u64(c64, v);
  }
  c = (uint32_t)c64;
  for (; n; --n) c = _mm_crc32_u8(c, *p++);
  return c;
}
uint32_t crcSw(const uint8_t* p, size_t n, uint32_t c) {
  static uint32_t table[256];
  static const bool init = [] {
    for (uint32_t i = 0; i < 256; ++i) {
      uint32_t x = i;
      for (int k = 0; k < 8; ++k) x = x & 1 ? (x >> 1) ^ 0x82F63B78u : x >> 1;
      table[i] = x;
    }
    return true;
  }();
  (void)init;
  for (; n; --n) c = table[(c ^ *p++) & 0xff] ^ (c >> 8);
  return c;
}
}  // namespace

uint32_t crc32c(const void* data, size_t n, uint32_t crc) {
  static const bool hw = __builtin_cpu_supports("sse4.2");
  const uint32_t c = ~crc;
  return ~(hw ? crcHw((const uint8_t*)data, n, c) : crcSw((const uint8_t*)data, n, c));
}

}  // namespace gdb
