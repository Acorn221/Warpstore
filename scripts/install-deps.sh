#!/bin/sh
# Build dependencies for warpstore on Debian / Ubuntu (22.04+). The CUDA toolkit (12.x: nvcc, NVRTC,
# cuBLAS) is not installed here: get it from NVIDIA (https://developer.nvidia.com/cuda-downloads) or
# use the Docker build (see the README). Run as root or with sudo.
set -e
apt-get update
apt-get install -y --no-install-recommends \
  build-essential cmake git pkg-config \
  libssl-dev libcurl4-openssl-dev libzstd-dev liblz4-dev libsnappy-dev zlib1g-dev libre2-dev libpcre2-dev
echo "done; now: cmake -S . -B build && cmake --build build -j"
