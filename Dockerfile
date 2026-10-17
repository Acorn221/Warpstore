# gpu-db in a container. Build:   docker build -t gpudb .
# Run (needs the NVIDIA Container Toolkit on the host):
#   docker run --gpus all -p 3307:3307 -v gpudb-data:/data gpudb
#   mysql -h 127.0.0.1 -P 3307 -u root
# Options go after the image name, e.g. ... gpudb --root-password secret --bind 0.0.0.0
ARG CUDA_VERSION=12.6.3
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04 AS build
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake git pkg-config \
      libssl-dev libcurl4-openssl-dev libzstd-dev liblz4-dev libsnappy-dev zlib1g-dev libre2-dev libpcre2-dev \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY CMakeLists.txt ./
COPY cmake cmake
COPY src src
COPY third_party third_party
# GPU architectures: machine code for Turing .. Ada, plus sm_89 PTX that the driver compiles for newer
# GPUs (Hopper, Blackwell) at load time. (CUDA 12.6's compiler crashes on gpu.cu for sm_90 directly.)
# Query plans are JIT-compiled for the GPU at hand either way.
ARG CUDA_ARCHS="75-real;80-real;86-real;89"
# Parallel compile jobs (each nvcc job needs a few GB of RAM; lower it on small machines: --build-arg JOBS=2)
ARG JOBS=4
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" \
    && cmake --build build -j"${JOBS}"

FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu24.04
RUN apt-get update && apt-get install -y --no-install-recommends \
      libssl3t64 libcurl4t64 libzstd1 liblz4-1 libsnappy1v5 zlib1g libre2-10 libpcre2-8-0 ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /src/build/gpudb /usr/local/bin/gpudb
COPY scripts/fetch-embed-model.sh /usr/local/bin/gpudb-fetch-embed-model
VOLUME /data
EXPOSE 3307 9000
# Listens on all interfaces inside the container: set a root password (or pass
# --insecure-no-root-password for local experiments; the server refuses otherwise).
ENTRYPOINT ["gpudb", "--data-dir", "/data", "--port", "3307", "--http-port", "9000", "--bind", "0.0.0.0"]
