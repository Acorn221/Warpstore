cmake_minimum_required(VERSION 3.24)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()
project(gpudb LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

set(JIT_SOURCES_H ${CMAKE_BINARY_DIR}/gen/jit_sources.h)
add_custom_command(
  OUTPUT ${JIT_SOURCES_H}
  COMMAND ${CMAKE_COMMAND} -DOUT=${JIT_SOURCES_H} -DDEVTYPES=${CMAKE_SOURCE_DIR}/src/devtypes.h
          -DKERNELS=${CMAKE_SOURCE_DIR}/src/kernels.cuh -P ${CMAKE_SOURCE_DIR}/cmake/embed.cmake
  DEPENDS src/devtypes.h src/kernels.cuh cmake/embed.cmake
  COMMENT "Embedding device sources for runtime compilation")

add_executable(gpudb
  src/server.cpp
  src/sql.cpp
  src/engine.cu
  src/storage.cu
  src/gpu.cu
  src/jit.cu
  src/proc.cpp
  src/sources.cpp
  src/parquet.cpp
  src/avro.cpp
  src/orc.cpp
  src/repl.cpp
  src/regex.cpp
  src/cluster.cpp
  ${JIT_SOURCES_H})
target_include_directories(gpudb PRIVATE ${CMAKE_BINARY_DIR}/gen)
target_compile_options(gpudb PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--default-stream=per-thread --expt-relaxed-constexpr -Xcompiler=-Wall>
  $<$<COMPILE_LANGUAGE:CXX>:-Wall>)
target_compile_definitions(gpudb PRIVATE CUDA_API_PER_THREAD_DEFAULT_STREAM=1)
find_package(OpenSSL REQUIRED)
target_link_libraries(gpudb PRIVATE CUDA::cudart CUDA::nvrtc CUDA::cuda_driver Threads::Threads OpenSSL::SSL OpenSSL::Crypto curl zstd lz4 snappy z re2)
# PCRE2 (JIT) for REGEXP_REPLACE captures, ~5x faster than RE2's submatch engine. Only the runtime
# library is needed (the few C functions are declared in regex.cpp); without it RE2 does everything.
find_library(PCRE2_LIB NAMES pcre2-8 libpcre2-8.so.0)
if(PCRE2_LIB)
  target_link_libraries(gpudb PRIVATE ${PCRE2_LIB})
  target_compile_definitions(gpudb PRIVATE GDB_PCRE2=1)
endif()
