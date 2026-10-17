// Unified table storage, modelled on SingleStore's design:
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
#include <map>
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

struct SvHash {
  using is_transparent = void;
  size_t operator()(std::string_view s) const { return std::hash<std::string_view>{}(s); }
};

// Append-only string array that readers may index while a writer appends (MVCC: queries read a
// table's dictionaries without its lock). Strings live in fixed blocks that never move; the block
// directory is replaced (old ones kept until destruction) when it fills; size() is published last.
class StrVec {
  static constexpr int kShift = 12;
  static constexpr size_t kBlock = size_t(1) << kShift, kMask = kBlock - 1;
  std::atomic<std::string**> dir_{nullptr};
  size_t dirCap_ = 0, nblocks_ = 0;
  std::atomic<size_t> n_{0};
  std::vector<std::unique_ptr<std::string*[]>> dirs_;    // every directory ever used (the last is current)
  std::vector<std::unique_ptr<std::string[]>> blocks_;
  std::string& slot(size_t i) {
    if (i >= nblocks_ * kBlock) {
      if (nblocks_ == dirCap_) {
        const size_t cap = std::max<size_t>(16, dirCap_ * 2);
        std::unique_ptr<std::string*[]> d(new std::string*[cap]());
        for (size_t b = 0; b < nblocks_; ++b) d[b] = blocks_[b].get();
        dir_.store(d.get(), std::memory_order_release);
        dirs_.push_back(std::move(d));
        dirCap_ = cap;
      }
      blocks_.emplace_back(new std::string[kBlock]);
      dirs_.back()[nblocks_++] = blocks_.back().get();
    }
    return dir_.load(std::memory_order_relaxed)[i >> kShift][i & kMask];
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
  const std::string& operator[](size_t i) const { return dir_.load(std::memory_order_acquire)[i >> kShift][i & kMask]; }
  std::string& operator[](size_t i) { return dir_.load(std::memory_order_acquire)[i >> kShift][i & kMask]; }
  const std::string& back() const { return (*this)[size() - 1]; }
  void push_back(std::string s) {
    const size_t i = n_.load(std::memory_order_relaxed);
    slot(i) = std::move(s);
    n_.store(i + 1, std::memory_order_release);
  }
  template <class... A>
  std::string& emplace_back(A&&... a) {
    push_back(std::string(std::forward<A>(a)...));
    return (*this)[size() - 1];
  }
  void reserve(size_t) {}
  void clear() {  // (only on dictionaries nobody else reads)
    dir_.store(nullptr);
    dirs_.clear();
    blocks_.clear();
    dirCap_ = nblocks_ = 0;
    n_.store(0);
  }
  struct const_iterator {
    const StrVec* v;
    size_t i;
    using iterator_category = std::random_access_iterator_tag;
    using value_type = std::string;
    using difference_type = std::ptrdiff_t;
    using pointer = const std::string*;
    using reference = const std::string&;
    reference operator*() const { return (*v)[i]; }
    pointer operator->() const { return &(*v)[i]; }
    const_iterator& operator++() { ++i; return *this; }
    const_iterator operator+(difference_type d) const { return {v, i + d}; }
    difference_type operator-(const const_iterator& o) const { return (difference_type)i - (difference_type)o.i; }
    bool operator!=(const const_iterator& o) const { return i != o.i; }
    bool operator==(const const_iterator& o) const { return i == o.i; }
  };
  const_iterator begin() const { return {this, 0}; }
  const_iterator end() const { return {this, size()}; }
};

struct Dict {
  StrVec strs;
  // string -> code. Strings added with appendUnique() are indexed lazily, on the first lookup
  // (derived dictionaries are often only ever read by code).
  mutable std::unordered_map<std::string, int32_t, SvHash, std::equal_to<>> map;
  const uint64_t uid = nextUid();  // unique for the process (never reused, unlike the address)
  Dict() = default;
  Dict(const Dict& o) : strs(o.strs) {
    o.index();
    std::shared_lock<std::shared_mutex> lk(o.mapMu);
    map = o.map;
  }
  Dict& operator=(const Dict& o) {
    if (this != &o) {
      o.index();
      strs = o.strs;
      std::shared_lock<std::shared_mutex> lk(o.mapMu);
      map = o.map;
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
    auto it = map.find(s);
    if (it != map.end()) return it->second;
    int32_t c = (int32_t)strs.size();
    strs.emplace_back(s);
    map.emplace(strs.back(), c);
    return c;
  }
  // Appends a string known not to be in the dictionary yet; returns its code.
  int32_t appendUnique(std::string&& s) {
    strs.push_back(std::move(s));
    lazy = true;
    return (int32_t)strs.size() - 1;
  }
  int32_t find(std::string_view s) const {
    index();
    std::shared_lock<std::shared_mutex> lk(mapMu);
    auto it = map.find(s);
    return it == map.end() ? -1 : it->second;
  }
  void index() const {
    if (!lazy.load(std::memory_order_acquire)) return;
    std::lock_guard<std::mutex> g(lazyMu);
    if (!lazy.load(std::memory_order_relaxed)) return;
    std::unique_lock<std::shared_mutex> lk(mapMu);
    map.reserve(strs.size());
    for (size_t i = map.size(); i < strs.size(); ++i) map.emplace(strs[i], (int32_t)i);
    lazy.store(false, std::memory_order_release);
  }

 private:
  static uint64_t nextUid() {
    static std::atomic<uint64_t> n{0};
    return ++n;
  }
  mutable std::atomic<bool> lazy{false};
  mutable std::mutex lazyMu;
  mutable std::shared_mutex mapMu;  // map: readers look strings up while a writer adds them
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
  ~SegData();
};

struct Segment {
  uint64_t id = 0;  // unique for the process (keys per-segment caches such as vector graphs)
  int part = 0;  // partition this segment belongs to
  int64_t n = 0;
  std::shared_ptr<SegData> d = std::make_shared<SegData>();
  // (the column data, by its old names)
  std::vector<std::vector<uint8_t>>& host = d->host;
  std::vector<void*>& dev = d->dev;
  std::vector<uint8_t>& mapped = d->mapped;
  std::vector<uint32_t>& uses = d->uses;
  std::vector<float>& heat = d->heat;
  std::vector<uint8_t>& enc = d->enc;
  std::vector<std::vector<uint8_t>>& encHost = d->encHost;
  int64_t& devBytes = d->devBytes;
  std::vector<double>& mn = d->mn;
  std::vector<double>& mx = d->mx;
  DelP del = std::make_shared<DelBits>();
  Segment() = default;
  Segment(const Segment& o) : id(o.id), part(o.part), n(o.n), d(o.d), del(o.del) {}  // a snapshot: shares data and bitmap
  Segment& operator=(const Segment&) = delete;
  int64_t ndel() const { return del->ndel; }
  const std::vector<uint32_t>& delHost() const { return del->host; }
  const uint32_t* delDev() { return del->device((n + 31) / 32); }
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
inline Loc bufLoc(int part, int64_t row) { return Loc{-1 - part, (int32_t)row}; }
inline int bufPart(Loc l) { return -1 - l.seg; }

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
  uint64_t commitSeq = 0;  // 0: open; else committed, kept for older snapshots (REPEATABLE READ)
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
  std::string db = "gpudb";
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
  std::unordered_map<int64_t, Loc> pkIndex;
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
    if (sg.host[c].empty() && sg.n > 0 && sg.dev[c] && !sg.enc[c]) fillHostMirror(sg, c);  // GPU-built temp tables
    return sg.host[c];
  }
  void fillHostMirror(Segment& sg, int c) const;

  // Writes (caller holds exclusive lock).
  void appendRow(const std::vector<Value>& row);  // converts types, enforces PK
  Loc lastDup{INT32_MIN, 0};  // the existing row of the last UNIQUE-key violation (writers hold the lock)
  void appendRaw(const std::vector<int64_t>& ints, const std::vector<double>& dbls);  // physical values
  void deleteAt(Loc l);
  void txnNoteDelete(Loc l);  // an open transaction deleted row l (read committed bookkeeping)
  void flush(bool force);  // all partitions' buffers -> sorted GPU segment(s)
  void flushPart(int part);
  uint8_t encodeColumn(int c, const std::vector<uint8_t>& h, int64_t n, std::vector<uint8_t>& out) const;
  // srcOf[c]: which field of each line feeds column c (-1: NULL / AUTO_INCREMENT); empty = in order.
  // yield: called every 256K rows with no row half-written (the caller may drop and retake its lock).
  int64_t loadFile(const std::string& path, char delim, int64_t skip, std::vector<int> srcOf = {},
                   const std::function<void()>& yield = {}, char escape = 0);
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
// The open transaction whose change sets (Table::txnPending) writes are recorded under: undoLog(),
// or tTxnKey while a ROLLBACK replays the undo log (its re-inserted rows stay hidden until the
// change sets are dropped, so the rollback shows all at once).
extern thread_local const void* tTxnKey;
inline const void* txnKey() { return tTxnKey ? tTxnKey : (const void*)undoLog(); }

// Current database of the calling connection (thread); "gpudb" by default.
std::string& currentDb();

struct Catalog {
  std::mutex mu;
  std::map<std::string, std::shared_ptr<Table>> tables;  // key: "db.table", lower case
  std::set<std::string> dbs{"gpudb", "mysql", "sys", "performance_schema"};
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
