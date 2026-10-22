// Unified table storage (LSM-style):
//   * an in-memory write buffer (the "rowstore" top of the LSM) on the CPU takes OLTP writes,
//   * full buffers are flushed (sorted by SORT KEY on the GPU) into immutable column segments
//     that live in VRAM, with a host mirror acting as the durable/cold tier,
//   * per-segment min/max metadata for segment elimination, deleted-row bitmasks for DML,
//   * an optional PRIMARY KEY hash index for point lookups and uniqueness,
//   * hash partitioning by SHARD KEY (default: the PRIMARY KEY; keyless tables are spread
//     round-robin; REFERENCE tables are a single unsharded partition). Each partition has its
//     own write buffer and its own segments, so shard-key lookups touch one partition and joins
//     on the shard keys of two tables run partition by partition (co-located joins).
#pragma once
#include <memory>
#include <cstring>
#include <map>
#include <thread>
#include <functional>
#include <atomic>
#include <mutex>
#include <set>
#include <shared_mutex>
#include <condition_variable>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "common.h"
#include "expr.h"

namespace gdb {

constexpr int64_t kSegRows = 1 << 20;

struct ColumnMeta {
  std::string name;
  LType lt;
  PType pt;
  bool notNull = false;
  bool autoInc = false;
  bool hasDefault = false;
  Value def;               // DEFAULT value (NULL when hasDefault is false)
  std::string typeText;    // as written in CREATE TABLE (SHOW CREATE TABLE / information_schema)
  int vecDim = 0;          // VECTOR(n)
  int scale = 0;           // DECIMAL(p, s): stored as int64 scaled by 10^s
  int vram = 0;            // VRAM hint: 0 AUTO, 1 PIN, 2 NEVER (0: the table's)
  std::string defText;     // DEFAULT (now()) as written; empty for plain defaults
};

// "[1, 2.5, -3]" -> floats (false if malformed)
bool parseVector(const std::string& s, std::vector<float>& out);
bool parseDecimal(const char* b, const char* e, int scale, int64_t& out);
std::string formatVector(const float* v, int dim);

// wyhash (final version 4, public domain): a fast 64-bit string hash for in-memory tables (nothing
// stored on disk depends on it).
inline uint64_t wyMix(uint64_t a, uint64_t b) {
  const __uint128_t r = (__uint128_t)a * b;
  return (uint64_t)r ^ (uint64_t)(r >> 64);
}
inline uint64_t wyHash(const void* key, size_t len) {
  auto r8 = [](const uint8_t* p) { uint64_t v; memcpy(&v, p, 8); return v; };
  auto r4 = [](const uint8_t* p) { uint32_t v; memcpy(&v, p, 4); return (uint64_t)v; };
  const uint8_t* p = (const uint8_t*)key;
  const uint64_t s0 = 0xa0761d6478bd642full, s1 = 0xe7037ed1a0b428dbull, s2 = 0x8ebc6af09c88c6e3ull, s3 = 0x589965cc75374cc3ull;
  uint64_t seed = wyMix(s0, s1), a, b;
  if (len <= 16) {
    if (len >= 4) {
      a = (r4(p) << 32) | r4(p + ((len >> 3) << 2));
      b = (r4(p + len - 4) << 32) | r4(p + len - 4 - ((len >> 3) << 2));
    } else if (len > 0) {
      a = ((uint64_t)p[0] << 16) | ((uint64_t)p[len >> 1] << 8) | p[len - 1];
      b = 0;
    } else {
      a = b = 0;
    }
  } else {
    size_t i = len;
    if (i > 48) {
      uint64_t see1 = seed, see2 = seed;
      do {
        seed = wyMix(r8(p) ^ s1, r8(p + 8) ^ seed);
        see1 = wyMix(r8(p + 16) ^ s2, r8(p + 24) ^ see1);
        see2 = wyMix(r8(p + 32) ^ s3, r8(p + 40) ^ see2);
        p += 48;
        i -= 48;
      } while (i > 48);
      seed ^= see1 ^ see2;
    }
    while (i > 16) {
      seed = wyMix(r8(p) ^ s1, r8(p + 8) ^ seed);
      i -= 16;
      p += 16;
    }
    a = r8(p + i - 16);
    b = r8(p + i - 8);
  }
  a ^= s1;
  b ^= seed;
  const __uint128_t r = (__uint128_t)a * b;
  return wyMix((uint64_t)r ^ s0 ^ len, (uint64_t)(r >> 64) ^ s1);
}

struct SvHash {
  using is_transparent = void;
  size_t operator()(std::string_view s) const { return std::hash<std::string_view>{}(s); }
};

// Append-only string array that readers may index while a writer appends (MVCC: queries read a
// table's dictionaries without its lock). The bytes live in arena chunks and the views in fixed
// blocks; neither ever moves (the block directory is replaced, old ones kept until destruction, when
// it fills); size() is published last. ~16 bytes + the string per entry (no allocation per string).
class StrVec {
  static constexpr int kShift = 12;
  static constexpr size_t kBlock = size_t(1) << kShift, kMask = kBlock - 1;
  static constexpr size_t kChunk = size_t(1) << 20;
  // An entry: 48-bit pointer | 16-bit length (0xFFFF: the length is the 8 bytes before the string).
  static constexpr uint64_t kLong = 0xFFFF;
  static uint64_t pack(const char* p, size_t len) { return (uint64_t)(uintptr_t)p | (std::min<uint64_t>(len, kLong) << 48); }
  static std::string_view unpack(uint64_t e) {
    const char* p = (const char*)(uintptr_t)(e & ((uint64_t(1) << 48) - 1));
    const uint64_t len = e >> 48;
    if (len < kLong) return std::string_view(p, len);
    uint64_t l;
    memcpy(&l, p - 8, 8);
    return std::string_view(p, l);
  }
  static size_t room(size_t len) { return len + (len >= kLong ? 8 : 0); }  // arena bytes for a string
  std::atomic<uint64_t**> dir_{nullptr};
  size_t dirCap_ = 0, nblocks_ = 0;
  std::atomic<size_t> n_{0};
  std::vector<std::unique_ptr<uint64_t*[]>> dirs_;  // every directory ever used (the last is current)
  std::vector<std::unique_ptr<uint64_t[]>> blocks_;
  std::vector<std::unique_ptr<char[]>> chunks_;
  char* cur_ = nullptr;
  size_t left_ = 0;
  uint64_t bytes_ = 0;
  uint64_t& slot(size_t i) {
    if (i >= nblocks_ * kBlock) {
      if (nblocks_ == dirCap_) {
        const size_t cap = std::max<size_t>(16, dirCap_ * 2);
        std::unique_ptr<uint64_t*[]> d(new uint64_t*[cap]());
        for (size_t b = 0; b < nblocks_; ++b) d[b] = blocks_[b].get();
        dir_.store(d.get(), std::memory_order_release);
        dirs_.push_back(std::move(d));
        dirCap_ = cap;
      }
      blocks_.emplace_back(new uint64_t[kBlock]());
      dirs_.back()[nblocks_++] = blocks_.back().get();
    }
    return dir_.load(std::memory_order_relaxed)[i >> kShift][i & kMask];
  }
  const char* store(std::string_view s) {
    if (s.empty()) return "";
    const size_t need = room(s.size());
    if (need > left_) {
      const size_t sz = std::max(kChunk, need);
      chunks_.emplace_back(new char[sz]);
      cur_ = chunks_.back().get();
      left_ = sz;
    }
    char* p = cur_;
    if (need != s.size()) {  // (long: its length first)
      const uint64_t l = s.size();
      memcpy(p, &l, 8);
      p += 8;
    }
    memcpy(p, s.data(), s.size());
    cur_ += need, left_ -= need, bytes_ += s.size();
    return p;
  }

 public:
  StrVec() = default;
  StrVec(const StrVec& o) { *this = o; }
  StrVec& operator=(const StrVec& o) {
    if (this == &o) return *this;
    clear();
    const size_t n = o.size();
    for (size_t i = 0; i < n; ++i) push_back(o[i]);
    return *this;
  }
  size_t size() const { return n_.load(std::memory_order_acquire); }
  bool empty() const { return size() == 0; }
  uint64_t bytes() const { return bytes_; }  // string bytes stored
  std::string_view operator[](size_t i) const { return unpack(dir_.load(std::memory_order_acquire)[i >> kShift][i & kMask]); }
  std::string_view back() const { return (*this)[size() - 1]; }
  void push_back(std::string_view s) {
    const size_t i = n_.load(std::memory_order_relaxed);
    const char* p = store(s);
    slot(i) = pack(p, s.size());
    n_.store(i + 1, std::memory_order_release);
  }
  void emplace_back(std::string_view s) { push_back(s); }
  // Appends k strings get(0..k-1) at once: one arena chunk, the copies on several threads, size()
  // published at the end (readers see all of them or none).
  template <class Get>
  void pushMany(size_t k, const Get& get) {
    if (k < 65536) {
      for (size_t j = 0; j < k; ++j) push_back(get(j));
      return;
    }
    const size_t base = n_.load(std::memory_order_relaxed);
    std::vector<uint64_t> off(k + 1, 0);
    for (size_t j = 0; j < k; ++j) off[j + 1] = off[j] + room(get(j).size());
    const size_t total = off[k];
    char* p = nullptr;
    if (total) {
      chunks_.emplace_back(new char[total]);
      p = chunks_.back().get();
      for (size_t j = 0; j < k; ++j) bytes_ += get(j).size();
    }
    while (nblocks_ * kBlock < base + k) slot(nblocks_ * kBlock);  // (blocks and directory for the whole range, in order)
    uint64_t** dir = dir_.load(std::memory_order_relaxed);
    const size_t nth = std::min<size_t>(std::max(1u, std::thread::hardware_concurrency()), 16);
    std::vector<std::thread> th;
    for (size_t t = 0; t < nth; ++t)
      th.emplace_back([&, t] {
        for (size_t j = k * t / nth; j < k * (t + 1) / nth; ++j) {
          const std::string_view v = get(j);
          char* d = p + off[j];
          if (room(v.size()) != v.size()) {
            const uint64_t l = v.size();
            memcpy(d, &l, 8);
            d += 8;
          }
          if (!v.empty()) memcpy(d, v.data(), v.size());
          const size_t i = base + j;
          dir[i >> kShift][i & kMask] = pack(v.empty() ? "" : d, v.size());
        }
      });
    for (auto& x : th) x.join();
    n_.store(base + k, std::memory_order_release);
  }
  void reserve(size_t) {}
  void clear() {  // (only on dictionaries nobody else reads)
    dir_.store(nullptr);
    dirs_.clear();
    blocks_.clear();
    chunks_.clear();
    cur_ = nullptr;
    left_ = 0;
    bytes_ = 0;
    dirCap_ = nblocks_ = 0;
    n_.store(0);
  }
  struct const_iterator {
    const StrVec* v;
    size_t i;
    using iterator_category = std::random_access_iterator_tag;
    using value_type = std::string_view;
    using difference_type = std::ptrdiff_t;
    using pointer = const std::string_view*;
    using reference = std::string_view;
    reference operator*() const { return (*v)[i]; }
    const_iterator& operator++() { ++i; return *this; }
    const_iterator operator+(difference_type d) const { return {v, i + d}; }
    difference_type operator-(const const_iterator& o) const { return (difference_type)i - (difference_type)o.i; }
    bool operator!=(const const_iterator& o) const { return i != o.i; }
    bool operator==(const const_iterator& o) const { return i == o.i; }
  };
  const_iterator begin() const { return {this, 0}; }
  const_iterator end() const { return {this, size()}; }
};

bool ioPoolCrowded();  // this thread is an I/O pool worker and others are busy too (work in parallel already)
struct Dict {
  StrVec strs;
  // Dictionary file (snapshots in a data dir; engine.cu): strs [0, fileStrs) are in it, ending at
  // byte fileBytes. (Not copied with the dictionary: a copy gets its own file.)
  std::string fileName;
  size_t fileStrs = 0;
  uint64_t fileBytes = 0;
  mutable size_t orderFileStrs[2] = {0, 0};  // strings covered by the saved binary / case-insensitive order
  mutable std::string orderFile[2];   // a saved order to read when first needed (loaded dictionaries)
  // string -> code: an open-addressing table of (32-bit hash << 32 | code + 1) slots; the strings
  // themselves are compared in strs. Strings added with appendUnique() are indexed lazily, on the
  // first lookup (derived dictionaries are often only ever read by code).
  const uint64_t uid = nextUid();  // unique for the process (never reused, unlike the address)
  Dict() = default;
  Dict(const Dict& o) : strs(o.strs) {
    o.index();
    std::shared_lock<std::shared_mutex> lk(o.mapMu);
    slots_ = o.slots_;
    indexed_ = o.indexed_;
  }
  Dict& operator=(const Dict& o) {
    if (this != &o) {
      o.index();
      strs = o.strs;
      std::shared_lock<std::shared_mutex> lk(o.mapMu);
      slots_ = o.slots_;
      indexed_ = o.indexed_;
      lazy = false;
      std::lock_guard<std::mutex> g(cacheMu);
      derived.clear();
    }
    return *this;
  }
  // Data derived from the strings (sort order, ranks, text index ...), cached with the dictionary so it
  // dies with it. Entries record the size they were built for: dictionaries only grow.
  template <class T>
  std::shared_ptr<T> getDerived(const std::string& k) const {
    std::lock_guard<std::mutex> g(cacheMu);
    auto it = derived.find(k);
    return it == derived.end() ? nullptr : std::static_pointer_cast<T>(it->second);
  }
  void setDerived(const std::string& k, std::shared_ptr<void> v) const {
    std::lock_guard<std::mutex> g(cacheMu);
    derived[k] = std::move(v);
  }
  void dropDerived(const std::string& prefix) const {  // e.g. per-partner entries, to bound the cache
    std::lock_guard<std::mutex> g(cacheMu);
    for (auto it = derived.lower_bound(prefix); it != derived.end() && it->first.compare(0, prefix.size(), prefix) == 0;) it = derived.erase(it);
  }
  size_t derivedCount() const {
    std::lock_guard<std::mutex> g(cacheMu);
    return derived.size();
  }
  int32_t getOrAdd(std::string_view s) {
    index();
    std::unique_lock<std::shared_mutex> lk(mapMu);
    return add(s, hash32(s));
  }
  // getOrAdd for n strings (codes in order of first appearance); a view with no data (NULL) gets
  // nullCode. Big batches: hashes, lookups of known strings and copies of new ones on all cores, then
  // only the new strings are added under the exclusive lock.
  void getOrAddMany(const std::string_view* s, size_t n, int64_t* out, int64_t nullCode) {
    index();
    if (n < 8192) {
      std::unique_lock<std::shared_mutex> lk(mapMu);
      for (size_t i = 0; i < n; ++i) out[i] = s[i].data() ? add(s[i], hash32(s[i])) : nullCode;
      return;
    }
    // On an I/O pool worker with others busy (e.g. one column of a load, the others coded at the same
    // time) one thread: the stages are memory-bound, 12 threads did 1.8x the speed for 7x the CPU.
    const bool serial = ioPoolCrowded();
    auto par = [serial](size_t m, const std::function<void(size_t, size_t)>& f) {
      const size_t nth = serial ? 1 : std::min<size_t>(std::max(1u, std::thread::hardware_concurrency()), m / 16384 + 1);
      std::vector<std::thread> th;
      for (size_t t = 1; t < nth; ++t) th.emplace_back([&, t] { f(m * t / nth, m * (t + 1) / nth); });
      f(0, m / nth);
      for (auto& x : th) x.join();
    };
    std::vector<uint32_t> hs(n);
    {
      std::shared_lock<std::shared_mutex> lk(mapMu);
      par(n, [&](size_t b, size_t e) {
        for (size_t i = b; i < e; ++i) {
          if (!s[i].data()) { out[i] = nullCode; continue; }
          hs[i] = hash32(s[i]);
          out[i] = lookup(s[i], hs[i]);
        }
      });
    }
    std::vector<size_t> miss;
    for (size_t i = 0; i < n; ++i)
      if (out[i] == -1) miss.push_back(i);
    if (miss.empty()) return;
    std::unique_lock<std::shared_mutex> lk(mapMu);
    const size_t m = miss.size();
    if (m < 16384) {
      grow(strs.size() + m);
      const size_t mask = slots_.size() - 1;
      for (size_t j = 0; j < m; ++j) {
        if (j + 8 < m) __builtin_prefetch(&slots_[hs[miss[j + 8]] & mask]);
        const size_t i = miss[j];
        int32_t f = lookup(s[i], hs[i]);  // (an earlier repeat in this batch, or another writer)
        if (f < 0) {
          f = (int32_t)strs.size();
          strs.push_back(s[i]);
          place(hs[i], (uint32_t)f);
        }
        out[i] = f;
      }
      indexed_ = strs.size();
      return;
    }
    // Many new strings: (1) those another writer added meanwhile, (2) the first occurrence of each
    // new one, deduplicated per hash shard on all cores, (3) codes in order of first appearance (as
    // the serial path assigns them), (4) the table entries placed on all cores.
    std::vector<int32_t> found(m);
    par(m, [&](size_t b, size_t e) {
      for (size_t j = b; j < e; ++j) found[j] = lookup(s[miss[j]], hs[miss[j]]);
    });
    constexpr int kShards = 64;
    std::vector<std::vector<uint32_t>> shard(kShards);
    for (size_t j = 0; j < m; ++j)
      if (found[j] < 0) shard[hs[miss[j]] >> 26].push_back((uint32_t)j);
    std::vector<uint32_t> first(m, UINT32_MAX);
    std::atomic<int> nextShard{0};
    par((size_t)kShards * 16384, [&](size_t, size_t) {  // (one worker per core; shards taken in turn)
      std::vector<uint32_t> tab;
      for (int sh; (sh = nextShard++) < kShards;) {
        const auto& list = shard[sh];
        size_t cap = 16;
        while (cap < list.size() * 2) cap <<= 1;
        tab.assign(cap, 0);
        for (uint32_t j : list) {
          const uint32_t h = hs[miss[j]];
          size_t at = (h * 0x9E3779B1u) & (cap - 1);
          for (;; at = (at + 1) & (cap - 1)) {
            if (!tab[at]) { tab[at] = j + 1; first[j] = j; break; }
            const uint32_t k = tab[at] - 1;
            if (hs[miss[k]] == h && s[miss[k]] == s[miss[j]]) { first[j] = k; break; }
          }
        }
      }
    });
    size_t fresh = 0;
    for (size_t j = 0; j < m; ++j) fresh += found[j] < 0 && first[j] == j;
    grow(strs.size() + fresh);
    std::vector<int32_t> codeOf(m, -1);
    std::vector<uint32_t> added;
    added.reserve(fresh);
    const size_t base = strs.size();
    for (size_t j = 0; j < m; ++j)
      if (found[j] < 0 && first[j] == j) {
        codeOf[j] = (int32_t)(base + added.size());
        added.push_back((uint32_t)j);
      }
    strs.pushMany(added.size(), [&](size_t k) { return s[miss[added[k]]]; });
    par(added.size(), [&](size_t b, size_t e) {
      for (size_t k = b; k < e; ++k) placeAtomic(hs[miss[added[k]]], (uint32_t)codeOf[added[k]]);
    });
    for (size_t j = 0; j < m; ++j) out[miss[j]] = found[j] >= 0 ? found[j] : codeOf[first[j]];
    indexed_ = strs.size();
  }
  // Codes of n strings that are in the dictionary (-1 for the others; nullCode for NULL views).
  void findMany(const std::string_view* s, size_t n, int64_t* out, int64_t nullCode) const {
    index();
    std::shared_lock<std::shared_mutex> lk(mapMu);
    for (size_t i = 0; i < n; ++i) out[i] = s[i].data() ? lookup(s[i], hash32(s[i])) : nullCode;
  }
  // After findMany: adds the strings coded -1, in order.
  void addMissing(const std::string_view* s, size_t n, int64_t* out) {
    std::unique_lock<std::shared_mutex> lk(mapMu);
    for (size_t i = 0; i < n; ++i)
      if (out[i] == -1) out[i] = add(s[i], hash32(s[i]));
  }
  // Appends a string known not to be in the dictionary yet; returns its code.
  int32_t appendUnique(std::string_view s) {
    strs.push_back(s);
    lazy = true;
    return (int32_t)strs.size() - 1;
  }
  int32_t find(std::string_view s) const {
    index();
    std::shared_lock<std::shared_mutex> lk(mapMu);
    return lookup(s, hash32(s));
  }
  void index() const {
    if (!lazy.load(std::memory_order_acquire)) return;
    std::lock_guard<std::mutex> g(lazyMu);
    if (!lazy.load(std::memory_order_relaxed)) return;
    std::unique_lock<std::shared_mutex> lk(mapMu);
    const size_t n = strs.size(), from = indexed_;
    grow(n);
    std::vector<uint32_t> hs(n - from);  // hashes first, on several threads (long strings: most of the work)
    const size_t nth = std::min<size_t>(std::max(1u, std::thread::hardware_concurrency()), hs.size() / 65536 + 1);
    std::vector<std::thread> th;
    for (size_t t = 0; t < nth; ++t)
      th.emplace_back([&, t] {
        for (size_t i = from + (n - from) * t / nth; i < from + (n - from) * (t + 1) / nth; ++i) hs[i - from] = hash32(strs[i]);
      });
    for (auto& x : th) x.join();
    for (size_t i = from; i < n; ++i)  // (a repeated string keeps its first code)
      if (lookup(strs[i], hs[i - from]) < 0) place(hs[i - from], (uint32_t)i);
    indexed_ = n;
    lazy.store(false, std::memory_order_release);
  }

 private:
  static uint64_t nextUid() {
    static std::atomic<uint64_t> n{0};
    return ++n;
  }
  static uint32_t hash32(std::string_view s) {
    const uint64_t h = wyHash(s.data(), s.size());
    return (uint32_t)(h ^ (h >> 32));
  }
  int32_t lookup(std::string_view s, uint32_t h) const {  // (mapMu held)
    if (slots_.empty()) return -1;
    const size_t mask = slots_.size() - 1;
    for (size_t i = h & mask;; i = (i + 1) & mask) {
      const uint64_t v = slots_[i];
      if (!v) return -1;
      if ((uint32_t)(v >> 32) == h) {
        const uint32_t c = (uint32_t)v - 1;
        if (strs[c] == s) return (int32_t)c;
      }
    }
  }
  void place(uint32_t h, uint32_t code) const {  // (mapMu held exclusively; room made by grow)
    const size_t mask = slots_.size() - 1;
    size_t i = h & mask;
    while (slots_[i]) i = (i + 1) & mask;
    slots_[i] = (uint64_t)h << 32 | (code + 1);
  }
  void placeAtomic(uint32_t h, uint32_t code) const {  // place() from several threads (distinct strings)
    const size_t mask = slots_.size() - 1;
    const uint64_t v = (uint64_t)h << 32 | (code + 1);
    for (size_t i = h & mask;; i = (i + 1) & mask) {
      uint64_t expect = 0;
      if (__atomic_compare_exchange_n(&slots_[i], &expect, v, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) return;
    }
  }
  void grow(size_t n) const {  // room for n strings: grows past 70% full, to at most half full
    if (n * 10 <= slots_.size() * 7) return;
    size_t cap = std::max<size_t>(64, slots_.size());
    while (cap < n * 2) cap *= 2;
    std::vector<uint64_t> old(cap, 0);
    old.swap(slots_);
    if (old.size() < (1u << 20)) {
      for (uint64_t v : old)
        if (v) place((uint32_t)(v >> 32), (uint32_t)v - 1);
      return;
    }
    const size_t nth = std::min<size_t>(std::max(1u, std::thread::hardware_concurrency()), 16);  // big tables: rehash on all cores
    std::vector<std::thread> th;
    for (size_t t = 0; t < nth; ++t)
      th.emplace_back([&, t] {
        for (size_t i = old.size() * t / nth; i < old.size() * (t + 1) / nth; ++i)
          if (old[i]) placeAtomic((uint32_t)(old[i] >> 32), (uint32_t)old[i] - 1);
      });
    for (auto& x : th) x.join();
  }
  int32_t add(std::string_view s, uint32_t h) {  // (mapMu held exclusively; everything indexed)
    const int32_t f = lookup(s, h);
    if (f >= 0) return f;
    const uint32_t c = (uint32_t)strs.size();
    grow(c + 1);
    strs.emplace_back(s);
    place(h, c);
    indexed_ = c + 1;
    return (int32_t)c;
  }
  mutable std::vector<uint64_t> slots_;
  mutable size_t indexed_ = 0;  // strs [0, indexed_) are in slots_
  mutable std::atomic<bool> lazy{false};
  mutable std::mutex lazyMu;
  mutable std::shared_mutex mapMu;  // slots_: readers look strings up while a writer adds them
  mutable std::mutex cacheMu;
  mutable std::map<std::string, std::shared_ptr<void>> derived;
};

// ---- MVCC. Queries read a per-statement snapshot of each table (Table::shadow) without holding its
// lock, so writers never wait for readers and readers never wait for writers. What a snapshot
// shares with the live table is never changed in place where the snapshot can see it:
//   * deleted-row bitmaps are copy-on-write (DelBits: a writer clones one that a snapshot holds),
//   * segment column data is immutable (SegData; the VRAM balancer frees memory it moves only after
//     every query that started earlier has finished, see rcuDefer),
//   * write buffers only grow at the end; storage that has to be reallocated while a snapshot holds
//     it is copied first (BufData), VRAM mirrors are reference counted (DevMirror), and a flush
//     installs a fresh buffer instead of clearing the old one,
//   * dictionaries only grow, in blocks that never move (StrVec).

// Deleted-row bitmap (bit set = deleted) with a lazily uploaded device copy.
struct DelBits {
  std::vector<uint32_t> host;
  int64_t ndel = 0;
  std::mutex mu;  // guards the device copy
  uint32_t* dev = nullptr;
  int64_t devCap = 0;     // words allocated on the device
  int64_t devWords = 0;   // words [0, devWords) of dev are current
  uint64_t ver = 0, devVer = ~0ull;  // ver: bumped when an existing bit changes
  DelBits() = default;
  DelBits(const DelBits& o) : host(o.host), ndel(o.ndel) {}
  ~DelBits();
  bool test(int64_t r) const { return (size_t)(r >> 5) < host.size() && ((host[r >> 5] >> (r & 31)) & 1u); }
  // Device bitmap valid for at least `words` words (uploads what changed).
  const uint32_t* device(int64_t words);
};
using DelP = std::shared_ptr<DelBits>;
// The bitmap to modify: `d` itself, or a private copy when a snapshot shares it.
DelBits& delForWrite(DelP& d);
void delSet(DelBits& d, int64_t r);  // marks row r deleted (and patches a current device copy)

// ---- Disk tier: tables larger than host memory. A flushed segment gets a file in <data dir>/segments
// (written lazily: before its columns are evicted, or by the SNAPSHOT whose manifest refers to it).
// Under --memory-limit the cache balancer (Engine::vramBalance) evicts the coldest column data from
// host memory; an evicted ("cold") column is read back from the file when it is used: into the cache
// while it has room, otherwise for one scan batch only, so scans larger than the budget stream through.
struct SegFile {
  uint64_t id = 0;
  int fd = -1;
  int64_t rows = 0;
  bool synced = false;  // fdatasync'd (a snapshot refers only to synced files)
  struct Col {
    uint8_t enc;     // ColView::enc of the bytes (0: raw)
    uint32_t width;  // bytes per row of the raw column
    uint64_t off, len;  // where the stored bytes start; len: their size uncompressed
    // Compressed (format 2): the encoding's header / dictionary (prefix bytes, as is) at off, then the
    // rows in blocks of blockRows, each zstd-compressed (or as is when that is not smaller): block b
    // is at [blocks[b], blocks[b + 1]). blockRows 0: stored as is at off.
    uint32_t blockRows = 0;
    uint64_t prefix = 0;
    std::vector<uint64_t> blocks;
    // Format 3: CRC32C of each block's stored bytes, and of the prefix (blocked) or of the whole
    // column (stored as is: checked when it is read whole). Empty: format 1 / 2, not checked.
    std::vector<uint32_t> crcs;
    uint32_t crc = 0;
  };
  std::vector<Col> cols;
  std::mutex gdsMu;
  void* gds = nullptr;  // GPUDirect Storage handle (opened on first use)
  bool gdsTried = false;
  ~SegFile();  // closes the file; deletes it unless the installed snapshot refers to it
};
struct SegData;
void tierInit(const std::string& dataDir, int64_t memLimit);  // memLimit < 0: no limit
bool tierEnabled();                                           // segment files can be written (a data dir)
int64_t tierMemLimit();
void tierSetMemCap(int64_t cap);                             // a cgroup memory cap: tierMemLimit leaves room for the rest                                       // INT64_MAX: none
int64_t tierHostBytes();                                      // segment column bytes in host memory
int64_t tierPendingBytes();                                   // evicted, freed once older queries finish
// The segment's file (written and opened if it has none yet; null without a data dir).
// rawCol: a column stored uncompressed (the sort key: binary searches read single rows of it).
std::shared_ptr<SegFile> segFileEnsure(SegData& d, int64_t n, int rawCol = -1);
std::shared_ptr<SegFile> segFileOpen(uint64_t id, size_t ncols, int64_t n);  // throws if missing / bad
void segFilesInstalled(const std::vector<uint64_t>& ids);  // a snapshot referring to exactly these is installed
void segFilesCleanup();                                    // startup: deletes files no segment uses
bool segCold(const SegData& d, int c);
void segLoad(SegData& d, int c);            // makes a cold column present (host mirror + device copy)
// Device data of column c for a scan batch of rows [lo, lo + n): raw columns, a pointer to row lo;
// encoded ones, a pointer to the encoding's header, with encOff the ColView::off of row lo. A cold
// column is read into the cache (whole batches, while there is room) or for the scan only: then just
// the blocks with the batch's rows, decompressed on the GPU when nvCOMP is there (gpuio.h).
void* segScanColumn(SegData& d, int c, int64_t lo, int64_t n, int64_t& encOff);
// Runs the queued GPU decompressions of columns read for the scan: before launching kernels that read
// columns bound since the last flush (scanTable flushes when it ends).
void scanFlush();
void releaseScanTemps();       // frees this thread's scan-only column copies (after the kernels using them)
void releaseStatementTemps();  // ... and read-aheads not used (end of statement)
// The segment batches the running scan visits, in order, for read-ahead (exchange: save / restore).
struct Segment;
struct ScanAheadItem {
  const Segment* seg;  // (column c's data: seg->cd(c))
  int64_t lo, n;
};
void scanAheadExchange(std::vector<ScanAheadItem>& items, size_t& pos);
void scanAheadAt(size_t pos);
int64_t tierReadBytes();  // bytes read from segment files for scans
// The column copies read for the current scan window reached the window size: an operator that
// collects batches for one launch runs on what it has now, then releaseScanTemps().
bool scanWindowFull();
const uint8_t* segRawColumn(const SegData& d, int c, std::vector<uint8_t>& tmp);  // raw bytes (cold: into tmp)
long long segRawInt(SegData& d, int c, int64_t r, bool i64);
void segRawCell(SegData& d, int c, int64_t r, uint8_t* out);  // one raw cell of a cold column, from the file
struct CellRead {
  SegData* d;  // a segment with a file; column c cold
  int c;
  int64_t row;
  uint8_t* out;  // the raw cell
};
void segRawCells(std::vector<CellRead>& reqs);  // many cells, each block read once, in parallel  // one raw integer (cold: read from the file)
// Eviction: begin under the table's exclusive lock, finish (frees) once older queries are done.
// Returns the bytes evicted (0: the column was not evictable).
int64_t segEvictBegin(SegData& d, int c, uint32_t& gen);
void segEvictFinish(SegData& d, int c, uint32_t gen);
// A column version no live segment uses any more (a column-level UPDATE replaced it): frees its
// memory. Called once queries that started earlier are done (rcuDefer).
void segDropColumn(SegData& d, int c);
void segAccount(SegData& d);  // counts a new segment's host bytes (after it is filled)
void segFileSync(SegFile& f);  // fdatasync (once)
void segDirSync();             // fsync of the segments directory (new files' entries)
uint64_t nextSegmentId();      // Segment::id

// Immutable column data of a segment (shared by the live table's Segment and snapshots of it).
struct SegData {
  std::vector<std::vector<uint8_t>> host;  // host mirror, per column
  std::vector<void*> dev;                  // VRAM copy (or mapped host mirror), per column
  std::vector<uint8_t> mapped;             // per column: 1 = host tier (read over PCIe)
  // VRAM cache (Engine::vramBalance): scans count batch reads per column; the balancer decays them
  // into heat and keeps the hottest columns in VRAM, the rest on the host tier.
  std::vector<uint32_t> uses;
  std::vector<float> heat;
  // GPU compression per column (ColView::enc; 0 = raw). Encoded columns keep the encoded bytes in
  // encHost (what dev points to); the host mirror stays raw for the CPU paths.
  std::vector<uint8_t> enc;
  std::vector<std::vector<uint8_t>> encHost;
  int64_t devBytes = 0;  // VRAM (or pinned host) bytes of the column copies
  std::vector<double> mn, mx;  // segment elimination metadata
  // Disk tier (see SegFile). cold[c] (read atomically): the column's bytes are only in the file. A
  // load or an eviction changes a column's host / device data under loadMu; evicted[c] holds the
  // device pointer of an eviction whose frees are still deferred (a load before then revives it).
  std::shared_ptr<SegFile> file;
  std::mutex loadMu;
  std::vector<uint8_t> cold;
  std::vector<uint32_t> gen;
  std::vector<std::pair<void*, uint8_t>> evicted;
  int64_t hostBytes = 0;  // host + encHost bytes counted in tierHostBytes
  ~SegData();
};

struct Segment {
  uint64_t id = 0;  // unique for the process (keys per-segment caches such as vector graphs)
  int part = 0;  // partition this segment belongs to
  int64_t n = 0;
  std::shared_ptr<SegData> d = std::make_shared<SegData>();
  // Column-level UPDATE: per column, the SegData holding its current version when that is not d
  // (empty: every column is in d). A SegData made by an UPDATE holds only the columns it changed
  // (the others have no bytes: width 0 in its file). Like d, never changed once installed.
  std::vector<std::shared_ptr<SegData>> ov;
  SegData& cd(size_t c) const { return c < ov.size() && ov[c] ? *ov[c] : *d; }
  const std::shared_ptr<SegData>& cdp(size_t c) const { return c < ov.size() && ov[c] ? ov[c] : d; }
  DelP del = std::make_shared<DelBits>();
  Segment() = default;
  Segment(const Segment& o) : id(o.id), part(o.part), n(o.n), d(o.d), ov(o.ov), del(o.del) {}  // a snapshot: shares data and bitmap
  Segment& operator=(const Segment&) = delete;
  int64_t ndel() const { return del->ndel; }
  const std::vector<uint32_t>& delHost() const { return del->host; }
  const uint32_t* delDev() { return del->device((n + 31) / 32); }
  // d and the column-level UPDATE versions (each once)
  template <class F> void eachData(F&& f) const {
    f(*d);
    for (size_t c = 0; c < ov.size(); ++c) {
      bool seen = !ov[c];
      for (size_t k = 0; k < c && !seen; ++k) seen = ov[k] == ov[c];
      if (!seen) f(*ov[c]);
    }
  }
  int64_t devBytes() const {
    int64_t b = 0;
    eachData([&](const SegData& x) { b += x.devBytes; });
    return b;
  }
  bool full() const { return d->uses.size() == d->dev.size(); }
  // Every column's device pointer / encoding: d's own vectors, or (columns with an UPDATE's
  // version) assembled in tmp.
  const std::vector<void*>& devs(std::vector<void*>& tmp) const {
    if (ov.empty()) return d->dev;
    tmp.resize(d->dev.size());
    for (size_t c = 0; c < tmp.size(); ++c) tmp[c] = cd(c).dev[c];
    return tmp;
  }
  const std::vector<uint8_t>& encs(std::vector<uint8_t>& tmp) const {
    if (ov.empty()) return d->enc;
    tmp.resize(d->enc.size());
    for (size_t c = 0; c < tmp.size(); ++c) tmp[c] = cd(c).enc[c];
    return tmp;
  }
  bool hasFile() const {  // some column's data is in a segment file (disk tier)
    bool f = false;
    eachData([&](const SegData& x) { f = f || x.file; });
    return f;
  }  // has the cache metadata (not a GPU-built temp table)
};

// VRAM copy of a write buffer's columns: rows [0, rows) uploaded, room for cap rows. Replaced (not
// reallocated) when it has to grow, so snapshots keep using the one they got.
struct DevMirror {
  std::vector<void*> cols;
  int64_t cap = 0, rows = 0;
  uint64_t gen = 0;
  ~DevMirror();
};

// Write buffer column storage, shared by the live buffer and its snapshots.
struct BufData {
  std::vector<std::vector<uint8_t>> cols;
  std::mutex devMu;  // guards mirror
  std::shared_ptr<DevMirror> mirror;
  // Per-column value range of rows [0, rangeN) (colRange), extended as rows are appended. Shared by
  // the snapshots of the buffer (a snapshot with fewer rows gets a range that may be wider: safe).
  std::mutex rangeMu;
  uint64_t rangeGen = ~0ull;
  int64_t rangeN = 0;
  std::vector<double> rmn, rmx;
};

struct WriteBuffer {
  std::shared_ptr<BufData> data = std::make_shared<BufData>();
  struct ColsView {  // cols[c]: the current storage's column (data can be replaced, see reserveForAppend)
    WriteBuffer* w;
    std::vector<uint8_t>& operator[](size_t i) const { return w->data->cols[i]; }
    size_t size() const { return w->data->cols.size(); }
    std::vector<std::vector<uint8_t>>::iterator begin() const { return w->data->cols.begin(); }
    std::vector<std::vector<uint8_t>>::iterator end() const { return w->data->cols.end(); }
    void resize(size_t n) const { w->data->cols.resize(n); }
  };
  ColsView cols{this};
  DelP del = std::make_shared<DelBits>();
  int64_t n = 0;
  uint64_t version = 1;
  // Columns change only by appends, except when replaced (temp tables) or the buffer is replaced
  // (flush): gen bumps, and the VRAM copy is re-uploaded whole.
  uint64_t gen = 0;
  // What GPU scans read (Table::syncBufferToDevice): device columns and deleted-row bitmap for rows
  // [0, n), and the mirror they belong to (kept alive).
  std::vector<void*> dev;
  const uint32_t* delDev = nullptr;
  std::shared_ptr<DevMirror> devRef;
  uint64_t devVersion = 0;
  WriteBuffer() = default;
  WriteBuffer(const WriteBuffer& o)  // a snapshot: shares the storage, sees rows [0, o.n)
      : data(o.data), del(o.del), n(o.n), version(o.version), gen(o.gen) {}
  WriteBuffer& operator=(const WriteBuffer&) = delete;
  int64_t ndel() const { return del->ndel; }
  // Before appending bytes to column storage that a snapshot shares and that would have to grow:
  // a private copy (with room to grow) takes its place.
  void reserveForAppend(int64_t rows);
};

// Deferred frees for memory a running query may still read (MVCC): rcuEnter/rcuExit bracket a
// snapshot query; rcuDefer runs `fn` once every query that entered before the call has exited.
uint64_t rcuEnter();
void rcuExit(uint64_t ticket);
void rcuDefer(std::function<void()> fn);
struct RcuGuard {  // (a reader outside a query's snapshot scope: compaction, snapshot writing, the tier balancer)
  uint64_t ticket = rcuEnter();
  RcuGuard() = default;
  RcuGuard(const RcuGuard&) = delete;
  ~RcuGuard() { rcuExit(ticket); }
};
// Runs fn(i) for i in [0, n) on the I/O thread pool and waits (fn must not throw). Urgent work goes
// ahead of what is queued.
void ioParallel(size_t n, const std::function<void(size_t)>& fn, bool urgent = false);


// Partition hashing. Integer-like values (INT, BIGINT, DATE) hash by value and strings by content,
// so equal keys land in the same partition in every table with the same partition count.
extern int g_compress;  // compress new GPU segments: 1 always (--compress), 0 never (--no-compress), -1 once VRAM is 60% full
uint64_t shardHashInt(int64_t v);
uint64_t shardHashDbl(double v);
uint64_t shardHashStr(std::string_view s);
inline uint64_t shardCombine(uint64_t h, uint64_t x) { return h * 0x100000001b3ull ^ x; }

struct Loc {
  int32_t seg;  // >= 0: segment index; < 0: write buffer of partition (-1 - seg)
  int32_t row;
  bool operator==(const Loc& o) const { return seg == o.seg && row == o.row; }
};

// int64 -> V hash map with open addressing (linear probing, tombstones): the primary-key index.
// ~17 bytes per slot instead of a node per entry (std::unordered_map: 15M keys took ~1 GB and seconds
// to rebuild). Iterators stay valid across erase(); inserts may rehash.
template <class V>
class FlatMap {
  std::vector<int64_t> keys_;
  std::vector<V> vals_;
  std::vector<uint8_t> state_;  // 0 empty, 1 full, 2 erased
  size_t n_ = 0, used_ = 0;     // full; full + erased
  static size_t mix(int64_t k) {
    uint64_t x = (uint64_t)k * 0x9E3779B97F4A7C15ull;
    return (size_t)(x ^ (x >> 29));
  }
  void rehash(size_t cap) {
    FlatMap o;
    o.keys_.assign(cap, 0);
    o.vals_.assign(cap, V{});
    o.state_.assign(cap, 0);
    for (size_t i = 0; i < state_.size(); ++i)
      if (state_[i] == 1) o.insertNew(keys_[i], vals_[i]);
    *this = std::move(o);
  }
  size_t insertNew(int64_t k, const V& v) {  // (k absent, room made)
    const size_t mask = state_.size() - 1;
    size_t i = mix(k) & mask;
    while (state_[i] == 1) i = (i + 1) & mask;
    if (state_[i] == 0) ++used_;
    keys_[i] = k, vals_[i] = v, state_[i] = 1, ++n_;
    return i;
  }
  size_t slotOf(int64_t k) const {  // index, or SIZE_MAX
    if (state_.empty()) return SIZE_MAX;
    const size_t mask = state_.size() - 1;
    for (size_t i = mix(k) & mask;; i = (i + 1) & mask) {
      if (state_[i] == 0) return SIZE_MAX;
      if (state_[i] == 1 && keys_[i] == k) return i;
    }
  }

 public:
  struct Item {
    int64_t first;
    V& second;
  };
  struct iterator {
    using iterator_category = std::forward_iterator_tag;
    using value_type = Item;
    using difference_type = std::ptrdiff_t;
    using pointer = void;
    using reference = Item;
    FlatMap* m;
    size_t i;
    void skip() {
      while (i < m->state_.size() && m->state_[i] != 1) ++i;
    }
    Item operator*() const { return {m->keys_[i], m->vals_[i]}; }
    struct Arrow {
      Item it;
      Item* operator->() { return &it; }
    };
    Arrow operator->() const { return {**this}; }
    iterator& operator++() {
      ++i;
      skip();
      return *this;
    }
    bool operator==(const iterator& o) const { return i == o.i; }
    bool operator!=(const iterator& o) const { return i != o.i; }
  };
  size_t size() const { return n_; }
  bool empty() const { return n_ == 0; }
  void clear() { *this = FlatMap(); }
  void reserve(size_t n) {
    size_t cap = 16;
    while (cap < n * 2) cap <<= 1;
    if (cap > state_.size()) rehash(cap);
  }
  iterator begin() {
    iterator it{this, 0};
    it.skip();
    return it;
  }
  iterator end() { return {this, state_.size()}; }
  iterator find(int64_t k) {
    const size_t i = slotOf(k);
    return i == SIZE_MAX ? end() : iterator{this, i};
  }
  size_t count(int64_t k) const { return slotOf(k) != SIZE_MAX; }
  std::pair<iterator, bool> try_emplace(int64_t k, const V& v) {
    const size_t f = slotOf(k);
    if (f != SIZE_MAX) return {iterator{this, f}, false};
    if ((used_ + 1) * 10 > state_.size() * 7) rehash(std::max<size_t>(16, n_ * 4 > state_.size() ? state_.size() * 2 : state_.size()));
    return {iterator{this, insertNew(k, v)}, true};
  }
  V& operator[](int64_t k) { return (*try_emplace(k, V{}).first).second; }
  size_t erase(int64_t k) {
    const size_t i = slotOf(k);
    if (i == SIZE_MAX) return 0;
    state_[i] = 2, --n_;
    return 1;
  }
  iterator erase(iterator it) {
    state_[it.i] = 2, --n_;
    ++it;
    return it;
  }
};
inline Loc bufLoc(int part, int64_t row) { return Loc{-1 - part, (int32_t)row}; }
inline int bufPart(Loc l) { return -1 - l.seg; }

struct Segment;
// One partition's compaction (Table::compactPlan / compactBuild / compactInstall).
struct CompactJob {
  int part = -1;
  std::vector<std::shared_ptr<Segment>> olds;  // the segments merged
  std::vector<std::vector<uint32_t>> bits;     // their deleted-row bits when planned
  std::vector<std::shared_ptr<Segment>> news;  // the new segments
  std::vector<std::vector<Loc>> where;         // per old segment and row: its place in news (seg -1: dead)
};

// CAGRA vector index state: one GPU graph per segment (by segment id), built when first searched.
struct CagraGraph;
struct CagraState {
  std::mutex mu;
  std::map<uint64_t, std::shared_ptr<CagraGraph>> graphs;
  std::set<uint64_t> failed;
};

// Secondary index (KEY / UNIQUE KEY / CREATE INDEX): a CPU hash index on the first column's raw
// value -> row locations, maintained on every write. Equality lookups on it skip the GPU scan.
struct IndexMeta {
  std::string name;
  std::vector<int> cols;
  bool unique = false;
  std::unordered_map<int64_t, std::vector<Loc>> map;
  // VECTOR INDEX (IVF): map key = nearest centroid (inverted lists); queries probe the nprobe best lists.
  bool vec = false;
  int metric = 0;  // 0 DOT_PRODUCT (higher is closer), 1 EUCLIDEAN_DISTANCE
  int nlist = 0, nprobe = 0, dim = 0;
  int algo = 0;  // 0 IVF (CPU lists), 1 CAGRA (GPU graph per segment, cuVS)
  int graphDegree = 32, itopk = 0;
  std::shared_ptr<CagraState> cagra;  // built graphs by segment id (shared by copies of the index)
  int64_t trainedRows = 0;
  std::vector<float> cent;  // nlist x dim
  std::string options;
};

// Changes of an open transaction to one table: rows it inserted (hidden from other sessions until
// COMMIT) and committed rows it deleted (still shown to other sessions until COMMIT).
struct TxnPending {
  std::vector<Loc> hidden, ghosts;
  std::unordered_set<uint64_t> hiddenIx;  // hidden, for lookups (big transactions)
  static uint64_t key(Loc l) { return (uint64_t)(uint32_t)l.seg << 32 | (uint32_t)l.row; }
  TxnPending() = default;
  TxnPending(const TxnPending& o) : hidden(o.hidden), ghosts(o.ghosts), hiddenIx(o.hiddenIx), commitSeq(o.commitSeq.load()), logged(o.logged.load()) {}
  TxnPending& operator=(const TxnPending& o) {
    hidden = o.hidden;
    ghosts = o.ghosts;
    hiddenIx = o.hiddenIx;
    commitSeq = o.commitSeq.load();
    logged = o.logged.load();
    return *this;
  }
  // 0: open; else the commit's global sequence number. Set for all of a transaction's tables at
  // once (under the commit gate, without table locks), so a commit becomes visible atomically.
  std::atomic<uint64_t> commitSeq{0};
  // The commit's WAL record is written (set under the WAL lock, before commitSeq): a SNAPSHOT, which
  // holds that lock, counts the change set as committed, since the rotated-out WAL is gone after it.
  std::atomic<bool> logged{false};
};

// Table lock: readers / writer like std::shared_mutex, but a waiting writer blocks new readers, so
// a stream of analytic queries cannot starve INSERT / UPDATE (glibc's rwlock prefers readers).
// A thread already reading some table takes further read locks without waiting (nested subqueries).
class FairSharedMutex {
  std::mutex m_;
  std::condition_variable cv_;
  int readers_ = 0, waitingW_ = 0;
  bool w_ = false;
  static inline thread_local std::vector<std::pair<const void*, int>> held_;
  int& held() {
    for (auto& [p, n] : held_) if (p == this) return n;
    held_.push_back({this, 0});
    return held_.back().second;
  }
  void release() {
    for (size_t i = 0; i < held_.size(); ++i)
      if (held_[i].first == this && --held_[i].second == 0) { held_.erase(held_.begin() + i); return; }
  }
 public:
  static inline thread_local bool barge = false;  // set around lock_shared for microsecond reads (PK point lookups)
  void lock() {
    std::unique_lock<std::mutex> g(m_);
    ++waitingW_;
    cv_.wait(g, [&] { return !w_ && readers_ == 0; });
    --waitingW_;
    w_ = true;
  }
  bool try_lock() {
    std::lock_guard<std::mutex> g(m_);
    if (w_ || readers_) return false;
    return w_ = true;
  }
  void unlock() {
    { std::lock_guard<std::mutex> g(m_); w_ = false; }
    cv_.notify_all();
  }
  void lock_shared() {
    // Inside another table read (or a microsecond read): don't queue behind waiting writers (no
    // lock-order deadlock: writers hold one table lock), but never run alongside an active writer.
    bool nested = !held_.empty() || barge;
    std::unique_lock<std::mutex> g(m_);
    cv_.wait(g, [&] { return !w_ && (nested || waitingW_ == 0); });
    ++readers_;
    ++held();
  }
  bool try_lock_shared() {
    bool nested = !held_.empty();  // inside another table read: don't queue behind waiting writers
    std::lock_guard<std::mutex> g(m_);
    if (w_ || (!nested && waitingW_)) return false;
    ++readers_;
    ++held();
    return true;
  }
  void unlock_shared() {
    release();
    bool last;
    { std::lock_guard<std::mutex> g(m_); last = --readers_ == 0; }
    if (last) cv_.notify_all();
  }
};

struct Table {
  std::string name;
  std::map<const void*, TxnPending> txnPending;  // per open transaction (key: its undo log); guarded by mu
  std::string db = "warpstore";
  std::vector<IndexMeta> indexes;  // secondary keys (KEY / UNIQUE KEY)
  struct FkMeta {  // FOREIGN KEY metadata (not enforced)
    std::string name, refTable, onDelete, onUpdate;
    std::vector<std::string> cols, refCols;
  };
  std::vector<FkMeta> fks;
  std::vector<int> fulltextCols;  // FULLTEXT columns: inverted index built in the background
  std::vector<ColumnMeta> cols;
  std::vector<std::shared_ptr<Dict>> dicts;  // per column, VARCHAR only
  int pk = -1, sortKey = -1;           // pk: single-column PRIMARY KEY with a hash index
  std::vector<int> pkCols;             // all PRIMARY KEY columns
  std::vector<int> shardKey;           // empty => keyless (round-robin)
  int nparts = 1;
  bool reference = false;
  int vram = 0;  // VRAM hint for columns without their own: 0 AUTO (the cache decides), 1 PIN, 2 NEVER
  int vramHint(int c) const { return cols[c].vram ? cols[c].vram : vram; }
  std::vector<std::shared_ptr<Segment>> segs;
  std::vector<std::shared_ptr<WriteBuffer>> bufs;  // one per partition
  // MVCC snapshot (shadow()): a read-only copy of the table's state for one statement. It shares
  // the live table's data (see DelBits / SegData / BufData) but has its own segment / buffer lists,
  // row counts and bitmaps, so later writes to the live table don't show. Index maps are not
  // copied: lookups go to `base` (under its lock) and are checked against the snapshot.
  const Table* base = nullptr;
  int64_t baseFlushes = 0;  // base->flushes when the snapshot was taken (rows move between batches at a flush)
  std::shared_ptr<Table> shadow() const;  // caller holds a lock on mu
  bool isShadow() const { return base != nullptr; }
  // Is location l (from base's indexes) a live row of this snapshot?
  bool shadowHas(Loc l) const;
  FlatMap<Loc> pkIndex;
  FairSharedMutex mu;
  int64_t flushes = 0;
  mutable int64_t nextPart = 0;
  std::vector<uint8_t> hasNull;  // per column: a NULL was ever stored (codegen keeps the int fast path otherwise)
  int autoIncCol = -1;
  int64_t autoNext = 1, lastInsertId = 0;  // keyless round-robin cursor (writes hold the exclusive lock)

  Table(std::string name, std::vector<ColumnMeta> cols, int pk, int sortKey, std::vector<int> pkCols,
        std::vector<int> shardKey, int nparts, bool reference);
  int colIndex(const std::string& name) const;  // -1 if missing (case-insensitive)
  int64_t liveRows() const;
  int64_t bufferRows() const;
  int64_t bufferTotal() const;  // including deleted rows
  bool partitioned() const { return nparts > 1 && !shardKey.empty(); }
  // Partition of a row given its shard-key values (physical form, one per shard-key column).
  int partOf(const std::vector<int64_t>& ints, const std::vector<double>& dbls) const;
  // Partition for literal shard-key values (one Value per shard-key column), or -1.
  int partOfValues(const std::vector<Value>& vals) const;
  int colBytes(int c) const { return cols[c].pt == PType::VEC ? 4 * cols[c].vecDim : psize(cols[c].pt); }
  std::vector<std::vector<float>> curVec;  // scratch: vector cells of the row being appended
  const std::vector<uint8_t>& colData(Loc l, int c) const {
    if (l.seg < 0) return bufs[bufPart(l)]->cols[c];
    Segment& sg = *segs[l.seg];
    SegData& d = sg.cd(c);
    if (segCold(d, c)) segLoad(d, c);
    if (d.host[c].empty() && sg.n > 0 && d.dev[c] && !d.enc[c]) fillHostMirror(sg, c);  // GPU-built temp tables
    return d.host[c];
  }
  void fillHostMirror(Segment& sg, int c) const;

  // Writes (caller holds exclusive lock).
  void appendRow(const std::vector<Value>& row);  // converts types, enforces PK
  Loc lastDup{INT32_MIN, 0};  // the existing row of the last UNIQUE-key violation (writers hold the lock)
  void appendRaw(const std::vector<int64_t>& ints, const std::vector<double>& dbls);  // physical values
  void rawOf(int c, const Value& v, int64_t& i, double& d);  // a value as column c stores it
  // Appends n rows given as physical columns (iv[c] for I32 / I64 columns, dv[c] for F64; no VECTOR
  // columns): whole columns per partition for plain tables, row by row otherwise.
  // partOfRow: each row's partition when known (an UPDATE that keeps the shard key): no re-hashing.
  void appendColumns(const std::vector<std::vector<int64_t>>& iv, const std::vector<std::vector<double>>& dv, int64_t n,
                     const int32_t* partOfRow = nullptr, bool nullsKnown = false);  // nullsKnown: hasNull already set
  // UPDATE's new rows: rows ls[0..n) copied as they are, except the columns with given[c] (values in
  // iv / dv), each into partition part[i]. Tables without keys, indexes or open transactions only.
  void appendCopy(const Loc* ls, size_t n, const int32_t* part, const std::vector<uint8_t>& given,
                  const std::vector<std::vector<int64_t>>& iv, const std::vector<std::vector<double>>& dv);
  // Set while a statement holds row locations across appends: full write buffers are flushed after.
  bool deferFlush = false;
  void deleteAt(Loc l);
  // A failed statement's rows: deletes the write-buffer rows appended since mark (per partition:
  // buffer row counts then; no flush meanwhile), their keys and index entries included.
  void undoAppendsSince(const std::vector<int64_t>& mark);
  void txnNoteDelete(Loc l);  // an open transaction deleted row l (read committed bookkeeping)
  TxnPending& pendingOfMine();  // this transaction's change set here (created and registered on first use)
  void gcPending();             // drops committed change sets no snapshot needs (caller holds the exclusive lock)
  void flush(bool force);  // all partitions' buffers -> sorted GPU segment(s)
  void flushPart(int part);
  // A segment of partition `part` from n rows: row i is row idx[i] of the raw column bytes src(c)
  // (stats, encoding and placement as in a flush; not added to segs).
  std::shared_ptr<Segment> makeSegment(int part, size_t n, const std::function<const uint8_t*(int)>& src, const int32_t* idx) const;
  // Compaction (see storage.cu): plan under the shared lock, build without a lock, install under the
  // exclusive lock (false: the table changed shape meanwhile; try again later).
  std::vector<struct CompactJob> compactPlan() const;
  void compactBuild(struct CompactJob& j) const;
  // Column-level UPDATE: in segment sis[s], column setCols[k] becomes patches[s].cols[k] (the whole
  // raw column, moved from), which is the old one with rows[i] set to vals[k][i]. Each segment is
  // replaced by a new version (Segment::ov) sharing the rest; rows keep their places, so keys and
  // indexes on other columns are unchanged.
  struct ColPatch {
    std::vector<std::vector<uint8_t>> cols;  // per SET column: the new column
    std::vector<int32_t> rows;               // the rows changed
    std::vector<std::vector<uint8_t>> vals;  // per SET column: their new raw values, in rows' order
  };
  void replaceColumns(const std::vector<int32_t>& sis, const std::vector<int>& setCols, std::vector<ColPatch>& patches);
  bool compactInstall(struct CompactJob& j);
  uint8_t encodeColumn(int c, const std::vector<uint8_t>& h, int64_t n, std::vector<uint8_t>& out) const;
  bool colStats(int c, const std::vector<uint8_t>& h, int64_t n, double& mn, double& mx) const;
  // srcOf[c]: which field of each line feeds column c (-1: NULL / AUTO_INCREMENT); empty = in order.
  // yield: called every 256K rows with no row half-written (the caller may drop and retake its lock).
  // copyTo: also writes the file there, zstd-compressed (for the WAL). A load that fails keeps the
  // rows before the bad line (*loaded says how many) and, if there are any, a copy that replays to them.
  int64_t loadFile(const std::string& path, char delim, int64_t skip, std::vector<int> srcOf = {},
                   const std::function<void()>& yield = {}, char escape = 0, const std::string& copyTo = "",
                   int64_t* loaded = nullptr);
  int64_t pkValue(Loc l) const;
  std::vector<Value> rowValues(Loc l) const;  // a stored row as typed values (NULL-aware)
  // Secondary indexes: raw 64-bit key of a cell (false for NULL), maintenance, and (re)building.
  bool rawKey(Loc l, int c, int64_t& k) const;
  bool ixKey(const IndexMeta& ix, Loc l, int64_t& k) const;  // rawKey, or the IVF list of a vector
  void trainVectorIndex(IndexMeta& ix);                      // k-means centroids from the stored rows
  void indexAdd(Loc l);
  void indexRemove(Loc l);
  void buildIndex(IndexMeta& ix);
  double cellNum(Loc l, int c) const;

  // Reads (caller holds shared lock).
  void syncBufferToDevice(int part);  // makes bufs[part]->dev current
  int64_t vramBytes() const;
};

// Transaction undo log of the calling connection (null outside BEGIN .. COMMIT/ROLLBACK).
struct UndoRec {
  Table* t;
  bool inserted;  // true: row was inserted (undo = delete it); false: row was deleted (undo = re-insert)
  std::vector<Value> row;
};
std::vector<UndoRec>*& undoLog();
// The open transaction's key in Table::txnPending (null outside a transaction): unique per transaction
// (an odd number, never an address). It stays set while COMMIT / ROLLBACK finish: a ROLLBACK's
// re-inserted rows join the transaction's hidden rows, so the rollback shows all at once.
extern thread_local uintptr_t tTxnToken;
inline const void* txnKey() { return (const void*)tTxnToken; }
struct Table;
// The change sets the open transaction created (COMMIT stamps them; their map nodes are stable).
extern thread_local std::vector<std::pair<Table*, TxnPending*>> tTxnEntries;
// Oldest REPEATABLE READ snapshot (UINT64_MAX: none): committed change sets older than it can go.
uint64_t oldestSnapshotSeq();

// Current database of the calling connection (thread); "warpstore" by default.
std::string& currentDb();

struct Catalog {
  std::mutex mu;
  std::map<std::string, std::shared_ptr<Table>> tables;  // key: "db.table", lower case
  std::set<std::string> dbs{"warpstore", "mysql", "sys", "performance_schema"};
  std::set<std::string> syncDbs;  // CREATE DATABASE .. WITH SYNC DURABILITY: commits wait for the disk  // the system schemas tools expect (empty)
  std::map<std::string, std::string> views;              // key -> SELECT text
  // "t" -> "<current db>.t", "db.t" -> "db.t" (lower case)
  static std::string key(const std::string& name);
  std::shared_ptr<Table> get(const std::string& name);   // throws if missing
  std::shared_ptr<Table> find(const std::string& name);  // nullptr if missing
};
// Per-thread table substitution ("db.t" -> "db2.t2", catalog keys): a cluster aggregator points a
// distributed table's name at a temp table holding the rows fetched from the leaves.
extern thread_local const std::map<std::string, std::string>* tTableOverride;

std::string lower(std::string s);

}  // namespace gdb
