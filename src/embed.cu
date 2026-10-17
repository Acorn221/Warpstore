// BERT sentence encoder on the GPU: WordPiece tokenizer, safetensors weights (fp32), encoder layers
// with cuBLAS GEMMs (attention as strided-batched GEMMs per sequence) and small fused kernels,
// mean pooling + L2 normalization (sentence-transformers' recipe for all-MiniLM-L6-v2).
#include "embed.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <stdexcept>
#include <mutex>
#include <numeric>
#include <sstream>
#include <unordered_map>

namespace gdb {

namespace {

// ---------------------------------------------------------------- tokenizer (BERT uncased)

uint32_t nextCp(const std::string& s, size_t& i) {  // UTF-8 decode (invalid bytes as Latin-1)
  const unsigned char c = (unsigned char)s[i];
  if (c < 0x80) return s[i++];
  const int n = c >= 0xF0 ? 3 : c >= 0xE0 ? 2 : c >= 0xC0 ? 1 : 0;
  if (!n || i + n >= s.size()) return s[i++] & 0xFF;
  uint32_t cp = c & (0x3F >> n);
  for (int k = 1; k <= n; ++k) {
    if (((unsigned char)s[i + k] & 0xC0) != 0x80) return s[i++] & 0xFF;
    cp = (cp << 6) | ((unsigned char)s[i + k] & 0x3F);
  }
  i += n + 1;
  return cp;
}
void putCp(std::string& o, uint32_t cp) {
  if (cp < 0x80) o += (char)cp;
  else if (cp < 0x800) o += (char)(0xC0 | (cp >> 6)), o += (char)(0x80 | (cp & 0x3F));
  else if (cp < 0x10000) o += (char)(0xE0 | (cp >> 12)), o += (char)(0x80 | ((cp >> 6) & 0x3F)), o += (char)(0x80 | (cp & 0x3F));
  else o += (char)(0xF0 | (cp >> 18)), o += (char)(0x80 | ((cp >> 12) & 0x3F)), o += (char)(0x80 | ((cp >> 6) & 0x3F)), o += (char)(0x80 | (cp & 0x3F));
}
// Accent stripping for Latin letters (NFD + dropping combining marks, for the common range).
uint32_t stripAccent(uint32_t cp) {
  static const char* map00C0 = "AAAAAAACEEEEIIIIDNOOOOOxOUUUUYTsaaaaaaaceeeeiiiidnooooo/ouuuuyty";
  if (cp >= 0xC0 && cp <= 0xFF) {
    char m = map00C0[cp - 0xC0];
    if (cp == 0xD7 || cp == 0xF7 || cp == 0xDF || cp == 0xDE || cp == 0xFE || cp == 0xD0 || cp == 0xF0 || cp == 0xC6 || cp == 0xE6) return cp;
    return (uint32_t)(unsigned char)m;
  }
  if (cp >= 0x100 && cp <= 0x17F) {  // Latin Extended-A: pairs of upper/lower variants of a base letter
    static const char* base = "AaAaAaCcCcCcCcDdDdEeEeEeEeEeGgGgGgGgHhHhIiIiIiIiIiJjJjKkkLlLlLlLlLlNnNnNnnNnOoOoOoOoRrRrRrSsSsSsSsTtTtTtUuUuUuUuUuUuWwYyYZzZzZzs";
    char m = base[cp - 0x100];
    if (cp == 0x132 || cp == 0x133 || cp == 0x152 || cp == 0x153 || cp == 0x138 || cp == 0x149 || cp == 0x14A || cp == 0x14B) return cp;
    return (uint32_t)(unsigned char)m;
  }
  return cp;
}
bool isCombining(uint32_t cp) { return (cp >= 0x300 && cp <= 0x36F) || (cp >= 0x1AB0 && cp <= 0x1AFF) || (cp >= 0x20D0 && cp <= 0x20FF); }
bool isPunct(uint32_t cp) {
  if ((cp >= 33 && cp <= 47) || (cp >= 58 && cp <= 64) || (cp >= 91 && cp <= 96) || (cp >= 123 && cp <= 126)) return true;
  return (cp >= 0x2000 && cp <= 0x206F) || (cp >= 0x3000 && cp <= 0x303F) || cp == 0xA1 || cp == 0xA7 || cp == 0xAB || cp == 0xB6 ||
         cp == 0xB7 || cp == 0xBB || cp == 0xBF;
}
bool isCjk(uint32_t cp) {
  return (cp >= 0x4E00 && cp <= 0x9FFF) || (cp >= 0x3400 && cp <= 0x4DBF) || (cp >= 0x20000 && cp <= 0x2A6DF) || (cp >= 0xF900 && cp <= 0xFAFF) ||
         (cp >= 0x2F800 && cp <= 0x2FA1F);
}
uint32_t lowerCp(uint32_t cp) {
  if (cp < 0x80) return (uint32_t)tolower((int)cp);
  if ((cp >= 0xC0 && cp <= 0xDE && cp != 0xD7)) return cp + 32;
  if (cp >= 0x100 && cp <= 0x17F && cp % 2 == 0 && cp != 0x130 && cp != 0x138) return cp + 1;
  if (cp >= 0x391 && cp <= 0x3A9) return cp + 32;  // Greek
  if (cp >= 0x410 && cp <= 0x42F) return cp + 32;  // Cyrillic
  if (cp >= 0x400 && cp <= 0x40F) return cp + 80;
  return cp;
}

struct Tokenizer {
  std::unordered_map<std::string, int> vocab;
  int cls = 101, sep = 102, unk = 100, pad = 0;
  bool load(const std::string& path) {
    std::ifstream f(path);
    if (!f) return false;
    std::string line;
    int id = 0;
    while (std::getline(f, line)) {
      if (!line.empty() && line.back() == '\r') line.pop_back();
      vocab.emplace(line, id++);
    }
    auto get = [&](const char* t, int def) {
      auto it = vocab.find(t);
      return it == vocab.end() ? def : it->second;
    };
    cls = get("[CLS]", 101), sep = get("[SEP]", 102), unk = get("[UNK]", 100), pad = get("[PAD]", 0);
    return !vocab.empty();
  }
  // BasicTokenizer (clean, lowercase, strip accents, split whitespace / punctuation / CJK) + WordPiece.
  std::vector<int> encode(const std::string& text, size_t maxLen) const {
    std::vector<std::string> words;
    std::string cur;
    auto flush = [&] {
      if (!cur.empty()) words.push_back(cur), cur.clear();
    };
    for (size_t i = 0; i < text.size();) {
      uint32_t cp = nextCp(text, i);
      if (cp == 0 || cp == 0xFFFD || (cp < 32 && cp != '\t' && cp != '\n' && cp != '\r')) continue;
      if (cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' || cp == 0xA0 || cp == 0x3000 || (cp >= 0x2000 && cp <= 0x200A)) {
        flush();
        continue;
      }
      cp = stripAccent(lowerCp(cp));
      if (isCombining(cp)) continue;
      if (isPunct(cp) || isCjk(cp)) {
        flush();
        std::string p;
        putCp(p, cp);
        words.push_back(p);
        continue;
      }
      putCp(cur, cp);
    }
    flush();
    std::vector<int> ids{cls};
    for (auto& w : words) {
      if (ids.size() >= maxLen - 1) break;
      if (w.size() > 200) {
        ids.push_back(unk);
        continue;
      }
      std::vector<int> pieces;
      size_t start = 0;
      bool bad = false;
      while (start < w.size()) {
        size_t end = w.size();
        int found = -1;
        while (start < end) {
          std::string sub = (start > 0 ? "##" : "") + w.substr(start, end - start);
          auto it = vocab.find(sub);
          if (it != vocab.end()) {
            found = it->second;
            break;
          }
          --end;
          while (end > start && ((unsigned char)w[end] & 0xC0) == 0x80) --end;  // stay on character boundaries
        }
        if (found < 0) {
          bad = true;
          break;
        }
        pieces.push_back(found);
        start = end;
      }
      if (bad) ids.push_back(unk);
      else for (int p : pieces) ids.push_back(p);
    }
    if (ids.size() > maxLen - 1) ids.resize(maxLen - 1);
    ids.push_back(sep);
    return ids;
  }
};

// ---------------------------------------------------------------- kernels

__global__ void embedK(const int* ids, int T, int L, int H, const float* word, const float* pos, const float* type, float* out) {
  int t = blockIdx.x;
  if (t >= T) return;
  const float* w = word + (size_t)ids[t] * H;
  const float* p = pos + (size_t)(t % L) * H;
  for (int h = threadIdx.x; h < H; h += blockDim.x) out[(size_t)t * H + h] = w[h] + p[h] + type[h];
}
// x = LayerNorm(x + r) (r optional), one block per row.
__global__ void layerNormK(float* x, const float* r, const float* gamma, const float* beta, int H, float eps) {
  float* row = x + (size_t)blockIdx.x * H;
  const float* rr = r ? r + (size_t)blockIdx.x * H : nullptr;
  __shared__ float red[2][32];
  float s = 0, s2 = 0;
  for (int h = threadIdx.x; h < H; h += blockDim.x) {
    float v = row[h] + (rr ? rr[h] : 0.f);
    row[h] = v;
    s += v;
    s2 += v * v;
  }
  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffff, s, o), s2 += __shfl_xor_sync(0xffffffff, s2, o);
  if ((threadIdx.x & 31) == 0) red[0][threadIdx.x >> 5] = s, red[1][threadIdx.x >> 5] = s2;
  __syncthreads();
  if (threadIdx.x < 32) {
    int nw = (blockDim.x + 31) >> 5;
    s = threadIdx.x < nw ? red[0][threadIdx.x] : 0.f;
    s2 = threadIdx.x < nw ? red[1][threadIdx.x] : 0.f;
    for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffff, s, o), s2 += __shfl_xor_sync(0xffffffff, s2, o);
    if (threadIdx.x == 0) red[0][0] = s, red[1][0] = s2;
  }
  __syncthreads();
  const float mean = red[0][0] / H, var = fmaxf(red[1][0] / H - mean * mean, 0.f), inv = rsqrtf(var + eps);
  for (int h = threadIdx.x; h < H; h += blockDim.x) row[h] = (row[h] - mean) * inv * gamma[h] + beta[h];
}
__global__ void biasK(float* x, const float* b, int64_t n, int H, bool gelu) {
  for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < n; i += (int64_t)gridDim.x * blockDim.x) {
    float v = x[i] + b[i % H];
    x[i] = gelu ? 0.5f * v * (1.f + erff(v * 0.70710678118654752f)) : v;
  }
}
// Row-wise softmax of scores [B, heads, L, L] (scaled), masking keys at positions >= len[b].
__global__ void softmaxK(float* S, const int* len, int heads, int L, float scale) {
  const int row = blockIdx.x;  // (b * heads + h) * L + i
  const int b = row / (heads * L);
  float* s = S + (size_t)row * L;
  const int n = len[b];
  __shared__ float red[32];
  float mx = -1e30f;
  for (int j = threadIdx.x; j < n; j += blockDim.x) mx = fmaxf(mx, s[j] * scale);
  for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o));
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = mx;
  __syncthreads();
  if (threadIdx.x < 32) {
    mx = threadIdx.x < (blockDim.x + 31) / 32 ? red[threadIdx.x] : -1e30f;
    for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o));
    if (threadIdx.x == 0) red[0] = mx;
  }
  __syncthreads();
  mx = red[0];
  __syncthreads();
  float sum = 0;
  for (int j = threadIdx.x; j < L; j += blockDim.x) {
    float e = j < n ? __expf(s[j] * scale - mx) : 0.f;
    s[j] = e;
    sum += e;
  }
  for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, o);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = sum;
  __syncthreads();
  if (threadIdx.x < 32) {
    sum = threadIdx.x < (blockDim.x + 31) / 32 ? red[threadIdx.x] : 0.f;
    for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, o);
    if (threadIdx.x == 0) red[0] = sum;
  }
  __syncthreads();
  const float inv = 1.f / red[0];
  for (int j = threadIdx.x; j < L; j += blockDim.x) s[j] *= inv;
}
// [B*L, nh*dh] (row-major tokens) <-> [B, nh, L, dh] (per-head blocks), for batched attention GEMMs.
__global__ void toHeadsK(const float* x, float* y, int B, int L, int nh, int dh) {
  const int64_t n = (int64_t)B * L * nh * dh;
  for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < n; i += (int64_t)gridDim.x * blockDim.x) {
    const int d = i % dh, h = (i / dh) % nh, l = (i / ((int64_t)dh * nh)) % L, b = i / ((int64_t)dh * nh * L);
    y[(((int64_t)b * nh + h) * L + l) * dh + d] = x[i];
  }
}
__global__ void fromHeadsK(const float* y, float* x, int B, int L, int nh, int dh) {
  const int64_t n = (int64_t)B * L * nh * dh;
  for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < n; i += (int64_t)gridDim.x * blockDim.x) {
    const int d = i % dh, h = (i / dh) % nh, l = (i / ((int64_t)dh * nh)) % L, b = i / ((int64_t)dh * nh * L);
    x[i] = y[(((int64_t)b * nh + h) * L + l) * dh + d];
  }
}

// Mean over each sequence's real tokens, then L2-normalize. One block per sequence.
__global__ void poolK(const float* x, const int* len, int L, int H, float* out) {
  const int b = blockIdx.x, n = len[b];
  __shared__ float red[32];
  float ss = 0;
  for (int h = threadIdx.x; h < H; h += blockDim.x) {
    float acc = 0;
    for (int t = 0; t < n; ++t) acc += x[((size_t)b * L + t) * H + h];
    acc /= n;
    out[(size_t)b * H + h] = acc;
    ss += acc * acc;
  }
  for (int o = 16; o; o >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, o);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
  __syncthreads();
  if (threadIdx.x == 0) {
    float t = 0;
    for (int w = 0; w < (int)(blockDim.x + 31) / 32; ++w) t += red[w];
    red[0] = rsqrtf(fmaxf(t, 1e-24f));
  }
  __syncthreads();
  for (int h = threadIdx.x; h < H; h += blockDim.x) out[(size_t)b * H + h] *= red[0];
}

// ---------------------------------------------------------------- model

struct Layer {
  float *wq, *bq, *wk, *bk, *wv, *bv, *wo, *bo, *ln1g, *ln1b, *w1, *b1, *w2, *b2, *ln2g, *ln2b;
};
struct Model {
  int H = 384, layers = 6, heads = 12, inter = 1536, maxPos = 512, maxLen = 256;
  float eps = 1e-12f;
  float *word = nullptr, *pos = nullptr, *type = nullptr, *lng = nullptr, *lnb = nullptr;
  std::vector<Layer> L;
  Tokenizer tok;
  cublasHandle_t blas = nullptr;
  cudaStream_t stream = nullptr;
};
Model* gModel = nullptr;
std::string gStatus = "not loaded";
std::mutex gMu;  // one forward pass at a time (shared scratch buffers and stream)

double jsonNum(const std::string& js, const char* key, double def) {
  size_t p = js.find(std::string("\"") + key + "\"");
  if (p == std::string::npos) return def;
  p = js.find(':', p);
  return p == std::string::npos ? def : std::atof(js.c_str() + p + 1);
}

bool loadModel() {
  static std::once_flag once;
  std::call_once(once, [] {
    std::string dir;
    if (const char* e = getenv("GPUDB_EMBED_MODEL")) dir = e;
    else if (const char* h = getenv("HOME")) dir = std::string(h) + "/.cache/gpudb/models/all-MiniLM-L6-v2";
    auto m = std::make_unique<Model>();
    std::ifstream cf(dir + "/config.json");
    if (!cf) {
      gStatus = "no embedding model in " + dir + " (config.json, vocab.txt, model.safetensors; see scripts/fetch-embed-model.sh)";
      return;
    }
    std::stringstream ss;
    ss << cf.rdbuf();
    const std::string cfg = ss.str();
    m->H = (int)jsonNum(cfg, "hidden_size", 384);
    m->layers = (int)jsonNum(cfg, "num_hidden_layers", 6);
    m->heads = (int)jsonNum(cfg, "num_attention_heads", 12);
    m->inter = (int)jsonNum(cfg, "intermediate_size", 1536);
    m->maxPos = (int)jsonNum(cfg, "max_position_embeddings", 512);
    m->eps = (float)jsonNum(cfg, "layer_norm_eps", 1e-12);
    m->maxLen = std::min(m->maxPos, (int)jsonNum(cfg, "gpudb_max_tokens", 256));
    if (!m->tok.load(dir + "/vocab.txt")) {
      gStatus = "cannot read " + dir + "/vocab.txt";
      return;
    }
    // safetensors: u64 header length, JSON header (name -> dtype, shape, data_offsets), raw data.
    std::ifstream wf(dir + "/model.safetensors", std::ios::binary);
    uint64_t hl = 0;
    if (!wf || !wf.read((char*)&hl, 8) || hl > (1u << 26)) {
      gStatus = "cannot read " + dir + "/model.safetensors";
      return;
    }
    std::string hdr(hl, '\0');
    wf.read(hdr.data(), (std::streamsize)hl);
    std::vector<char> data((std::istreambuf_iterator<char>(wf)), std::istreambuf_iterator<char>());
    auto tensor = [&](const std::string& name, size_t expect) -> float* {
      std::string key = name;
      size_t p = hdr.find("\"" + key + "\"");
      if (p == std::string::npos) {  // some exports prefix the names with "bert."
        key = "bert." + name;
        p = hdr.find("\"" + key + "\"");
      }
      if (p == std::string::npos) throw std::runtime_error("missing tensor " + name);
      size_t dt = hdr.find("\"dtype\"", p), q = hdr.find("\"data_offsets\"", p);
      if (hdr.compare(hdr.find('"', hdr.find(':', dt) + 1) + 1, 3, "F32") != 0) throw std::runtime_error(name + " is not F32");
      size_t a = hdr.find('[', q);
      const size_t begin = std::strtoull(hdr.c_str() + a + 1, nullptr, 10);
      const size_t end = std::strtoull(hdr.c_str() + hdr.find(',', a) + 1, nullptr, 10);
      if (end - begin != expect * 4 || end > data.size()) throw std::runtime_error("unexpected size of " + name);
      float* d = nullptr;
      cudaMalloc(&d, end - begin);
      cudaMemcpy(d, data.data() + begin, end - begin, cudaMemcpyHostToDevice);
      return d;
    };
    try {
      const int H = m->H, I = m->inter;
      m->word = tensor("embeddings.word_embeddings.weight", (size_t)m->tok.vocab.size() * H);
      m->pos = tensor("embeddings.position_embeddings.weight", (size_t)m->maxPos * H);
      m->type = tensor("embeddings.token_type_embeddings.weight", (size_t)jsonNum(cfg, "type_vocab_size", 2) * H);
      m->lng = tensor("embeddings.LayerNorm.weight", H);
      m->lnb = tensor("embeddings.LayerNorm.bias", H);
      for (int l = 0; l < m->layers; ++l) {
        const std::string p = "encoder.layer." + std::to_string(l) + ".";
        Layer y;
        y.wq = tensor(p + "attention.self.query.weight", (size_t)H * H), y.bq = tensor(p + "attention.self.query.bias", H);
        y.wk = tensor(p + "attention.self.key.weight", (size_t)H * H), y.bk = tensor(p + "attention.self.key.bias", H);
        y.wv = tensor(p + "attention.self.value.weight", (size_t)H * H), y.bv = tensor(p + "attention.self.value.bias", H);
        y.wo = tensor(p + "attention.output.dense.weight", (size_t)H * H), y.bo = tensor(p + "attention.output.dense.bias", H);
        y.ln1g = tensor(p + "attention.output.LayerNorm.weight", H), y.ln1b = tensor(p + "attention.output.LayerNorm.bias", H);
        y.w1 = tensor(p + "intermediate.dense.weight", (size_t)I * H), y.b1 = tensor(p + "intermediate.dense.bias", I);
        y.w2 = tensor(p + "output.dense.weight", (size_t)H * I), y.b2 = tensor(p + "output.dense.bias", H);
        y.ln2g = tensor(p + "output.LayerNorm.weight", H), y.ln2b = tensor(p + "output.LayerNorm.bias", H);
        m->L.push_back(y);
      }
    } catch (std::exception& e) {
      gStatus = std::string("bad model in ") + dir + ": " + e.what();
      return;
    }
    if (cublasCreate(&m->blas) != CUBLAS_STATUS_SUCCESS) {
      gStatus = "cuBLAS init failed";
      return;
    }
    cudaStreamCreateWithFlags(&m->stream, cudaStreamNonBlocking);
    cublasSetStream(m->blas, m->stream);
    // TF32 tensor cores for the GEMMs (Ampere+): several times faster, cosine to fp32 output ~1 - 1e-6.
    if (!getenv("GPUDB_EMBED_FP32")) cublasSetMathMode(m->blas, CUBLAS_TF32_TENSOR_OP_MATH);
    gStatus = "embedding model " + dir + " (" + std::to_string(m->H) + " dims, " + std::to_string(m->layers) + " layers)";
    gModel = m.release();
  });
  return gModel != nullptr;
}

// Y[M,N] = X[M,K] * W[N,K]^T (row-major), cuBLAS column-major trick.
void gemmXWt(cublasHandle_t h, const float* X, const float* W, float* Y, int M, int N, int K) {
  const float one = 1, zero = 0;
  cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &one, W, K, X, K, &zero, Y, N);
}

// One padded batch: ids [B*L] (row-major), lens[B] -> out [B, H].
void forward(Model& m, const std::vector<int>& ids, const std::vector<int>& lens, int B, int L, float* hostOut) {
  const int H = m.H, I = m.inter, nh = m.heads, dh = H / nh, T = B * L;
  cudaStream_t st = m.stream;
  int *dIds, *dLen;
  float *x, *q, *k, *v, *ctx, *tmp, *ff, *S, *out;
  cudaMallocAsync(&dIds, (size_t)T * 4, st);
  cudaMallocAsync(&dLen, (size_t)B * 4, st);
  cudaMallocAsync(&x, (size_t)T * H * 4, st);
  cudaMallocAsync(&q, (size_t)T * H * 4, st);
  cudaMallocAsync(&k, (size_t)T * H * 4, st);
  cudaMallocAsync(&v, (size_t)T * H * 4, st);
  cudaMallocAsync(&ctx, (size_t)T * H * 4, st);
  cudaMallocAsync(&tmp, (size_t)T * H * 4, st);
  cudaMallocAsync(&ff, (size_t)T * I * 4, st);
  cudaMallocAsync(&S, (size_t)B * nh * L * L * 4, st);
  cudaMallocAsync(&out, (size_t)B * H * 4, st);
  cudaMemcpyAsync(dIds, ids.data(), (size_t)T * 4, cudaMemcpyHostToDevice, st);
  cudaMemcpyAsync(dLen, lens.data(), (size_t)B * 4, cudaMemcpyHostToDevice, st);
  embedK<<<T, 128, 0, st>>>(dIds, T, L, H, m.word, m.pos, m.type, x);
  layerNormK<<<T, 128, 0, st>>>(x, nullptr, m.lng, m.lnb, H, m.eps);
  const int grid = (int)std::min<int64_t>(((int64_t)T * I + 255) / 256, 65535);
  const float one = 1, zero = 0, scale = 1.f / std::sqrt((float)dh);
  for (auto& y : m.L) {
    gemmXWt(m.blas, x, y.wq, q, T, H, H);
    gemmXWt(m.blas, x, y.wk, k, T, H, H);
    gemmXWt(m.blas, x, y.wv, v, T, H, H);
    biasK<<<grid, 256, 0, st>>>(q, y.bq, (int64_t)T * H, H, false);
    biasK<<<grid, 256, 0, st>>>(k, y.bk, (int64_t)T * H, H, false);
    biasK<<<grid, 256, 0, st>>>(v, y.bv, (int64_t)T * H, H, false);
    // Attention for all sequences and heads at once: per-head layout [B, nh, L, dh], then
    // scores = Q K^T and context = P V as strided-batched GEMMs over the B * nh blocks.
    const int tg = (int)std::min<int64_t>(((int64_t)T * H + 255) / 256, 65535);
    toHeadsK<<<tg, 256, 0, st>>>(q, ctx, B, L, nh, dh);  // ctx: scratch for Q heads
    toHeadsK<<<tg, 256, 0, st>>>(k, tmp, B, L, nh, dh);  // tmp: scratch for K heads
    cublasSgemmStridedBatched(m.blas, CUBLAS_OP_T, CUBLAS_OP_N, L, L, dh, &one, tmp, dh, (long long)L * dh, ctx, dh, (long long)L * dh, &zero,
                              S, L, (long long)L * L, B * nh);
    softmaxK<<<B * nh * L, 128, 0, st>>>(S, dLen, nh, L, scale);
    toHeadsK<<<tg, 256, 0, st>>>(v, tmp, B, L, nh, dh);  // V heads
    cublasSgemmStridedBatched(m.blas, CUBLAS_OP_N, CUBLAS_OP_N, dh, L, L, &one, tmp, dh, (long long)L * dh, S, L, (long long)L * L, &zero,
                              q, dh, (long long)L * dh, B * nh);  // q: context in head layout
    fromHeadsK<<<tg, 256, 0, st>>>(q, ctx, B, L, nh, dh);
    gemmXWt(m.blas, ctx, y.wo, tmp, T, H, H);
    biasK<<<grid, 256, 0, st>>>(tmp, y.bo, (int64_t)T * H, H, false);
    layerNormK<<<T, 128, 0, st>>>(x, tmp, y.ln1g, y.ln1b, H, m.eps);
    gemmXWt(m.blas, x, y.w1, ff, T, I, H);
    biasK<<<grid, 256, 0, st>>>(ff, y.b1, (int64_t)T * I, I, true);
    gemmXWt(m.blas, ff, y.w2, tmp, T, H, I);
    biasK<<<grid, 256, 0, st>>>(tmp, y.b2, (int64_t)T * H, H, false);
    layerNormK<<<T, 128, 0, st>>>(x, tmp, y.ln2g, y.ln2b, H, m.eps);
  }
  poolK<<<B, 128, 0, st>>>(x, dLen, L, H, out);
  cudaMemcpyAsync(hostOut, out, (size_t)B * H * 4, cudaMemcpyDeviceToHost, st);
  for (void* p : {(void*)dIds, (void*)dLen, (void*)x, (void*)q, (void*)k, (void*)v, (void*)ctx, (void*)tmp, (void*)ff, (void*)S, (void*)out})
    cudaFreeAsync(p, st);
  cudaStreamSynchronize(st);
}

}  // namespace

bool embedAvailable() { return loadModel(); }
std::string embedStatus() {
  loadModel();
  return gStatus;
}
int embedDim() { return loadModel() ? gModel->H : 0; }
std::vector<int> embedTokens(const std::string& text) { return loadModel() ? gModel->tok.encode(text, (size_t)gModel->maxLen) : std::vector<int>{}; }

bool embedBatch(const std::vector<std::string>& texts, std::vector<float>& out, std::string& err) {
  if (!loadModel()) {
    err = gStatus;
    return false;
  }
  Model& m = *gModel;
  const int H = m.H;
  out.assign(texts.size() * (size_t)H, 0.f);
  std::vector<std::vector<int>> toks(texts.size());
  for (size_t i = 0; i < texts.size(); ++i) toks[i] = m.tok.encode(texts[i], (size_t)m.maxLen);
  // Sorted by length, in batches of about 16K tokens: little padding, bounded scratch memory.
  std::vector<size_t> order(texts.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return toks[a].size() < toks[b].size(); });
  std::lock_guard<std::mutex> g(gMu);
  std::vector<float> res;
  for (size_t at = 0; at < order.size();) {
    size_t L = toks[order[at]].size(), end = at;
    while (end < order.size() && (end - at + 1) * std::max(L, toks[order[end]].size()) <= 16384 && end - at < 512) L = std::max(L, toks[order[end++]].size());
    if (end == at) end = at + 1;
    const int B = (int)(end - at);
    std::vector<int> ids((size_t)B * L, m.tok.pad), lens(B);
    for (int b = 0; b < B; ++b) {
      const auto& t = toks[order[at + b]];
      std::copy(t.begin(), t.end(), ids.begin() + (size_t)b * L);
      lens[b] = (int)t.size();
    }
    res.resize((size_t)B * H);
    forward(m, ids, lens, B, (int)L, res.data());
    for (int b = 0; b < B; ++b) std::copy(res.begin() + (size_t)b * H, res.begin() + (size_t)(b + 1) * H, out.begin() + order[at + b] * H);
    at = end;
  }
  if (cudaGetLastError() != cudaSuccess) {
    err = "embedding forward pass failed";
    return false;
  }
  return true;
}

}  // namespace gdb
