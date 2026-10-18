#!/bin/sh
# Installs NVIDIA cuVS (CAGRA GPU graph vector index) for warpstore into ~/.local/share/warpstore-cuvs
# (or $WARPSTORE_CUVS_DIR). warpstore finds its headers at build time and loads the libraries at run time;
# without it, VECTOR INDEXes use the built-in IVF index. About 3 GB.
set -e
DIR=${WARPSTORE_CUVS_DIR:-$HOME/.local/share/warpstore-cuvs}
python3 -m pip install --target "$DIR" libcuvs-cu12
echo "cuVS in $DIR (re-run cmake so the build picks up the headers)"
