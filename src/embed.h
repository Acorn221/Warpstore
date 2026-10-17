// Text embeddings inside the database: a BERT-style sentence encoder (all-MiniLM-L6-v2 by default)
// run on the GPU with cuBLAS. EMBED('text') in SQL returns its VECTOR.
#pragma once
#include <string>
#include <vector>

namespace gdb {

// Loads the model once ($GPUDB_EMBED_MODEL, else ~/.cache/gpudb/models/all-MiniLM-L6-v2: config.json,
// vocab.txt, model.safetensors). False if it can't; embedStatus() says why.
bool embedAvailable();
std::string embedStatus();
int embedDim();
std::vector<int> embedTokens(const std::string& text);  // WordPiece ids (EMBED_TOKENS(), debugging)
// texts -> texts.size() x embedDim() floats, mean-pooled and L2-normalized. Thread-safe.
bool embedBatch(const std::vector<std::string>& texts, std::vector<float>& out, std::string& err);

}  // namespace gdb
