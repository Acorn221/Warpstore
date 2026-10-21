#!/bin/sh
# Installs NVIDIA nvCOMP for warpstore into ~/.local/share/warpstore-nvcomp (or $WARPSTORE_NVCOMP_DIR).
# warpstore loads it at run time to decompress segment files on the GPU (tables larger than
# --memory-limit); without it they are decompressed on the CPU. About 70 MB.
set -e
DIR=${WARPSTORE_NVCOMP_DIR:-$HOME/.local/share/warpstore-nvcomp}
python3 -m pip install --target "$DIR" nvidia-nvcomp-cu12==5.3.0.16
echo "nvCOMP in $DIR (picked up when warpstore starts)"
