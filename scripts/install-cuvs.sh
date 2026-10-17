#!/bin/sh
# Installs NVIDIA cuVS (CAGRA GPU graph vector index) for gpu-db into ~/.local/share/gpudb-cuvs
# (or $GPUDB_CUVS_DIR). gpu-db finds its headers at build time and loads the libraries at run time;
# without it, VECTOR INDEXes use the built-in IVF index. About 3 GB.
set -e
DIR=${GPUDB_CUVS_DIR:-$HOME/.local/share/gpudb-cuvs}
python3 -m pip install --target "$DIR" libcuvs-cu12
echo "cuVS in $DIR (re-run cmake so the build picks up the headers)"
