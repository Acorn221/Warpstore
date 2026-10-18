#!/bin/sh
# Downloads the default EMBED() model (sentence-transformers/all-MiniLM-L6-v2, 384 dims, ~90 MB) to
# ~/.cache/warpstore/models/all-MiniLM-L6-v2 (or $WARPSTORE_EMBED_MODEL). Any BERT-style model with
# config.json, vocab.txt and an fp32 model.safetensors works the same way.
set -e
MODEL=${1:-sentence-transformers/all-MiniLM-L6-v2}
DIR=${WARPSTORE_EMBED_MODEL:-$HOME/.cache/warpstore/models/$(basename "$MODEL")}
mkdir -p "$DIR"
for f in config.json vocab.txt model.safetensors; do
  curl -fSL --retry 3 -o "$DIR/$f" "https://huggingface.co/$MODEL/resolve/main/$f"
done
echo "model in $DIR"
