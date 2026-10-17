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

struct Dict {
  std::vector<std::string> strs;
  // string -> code. Strings added with appendUnique() are indexed lazily, on the first lookup
  // (derived dictionaries are often only ever read by code).
  mutable std::unordered_map<std::string, int32_t, SvHash, std::equal_to<>> map;
  Dict() = default;
  Dict(const Dict& o) : strs(o.strs) {
    o.index();
    map = o.map;
  }
  Dict& operator=(const Dict& o) {
    if (this != &o) {
      o.index();
      strs = o.strs;
      map = o.map;
      lazy = false;
    }
    return *this;
  }
  int32_t getOrAdd(std::string_view s) {
    index();
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
    auto it = map.find(s);
    return it == map.end() ? -1 : it->second;
  }
  void index() const {
    if (!lazy.load(std::memory_order_acquire)) return;
    std::lock_guard<std::mutex> g(lazyMu);
    if (!lazy.load(std::memory_order_relaxed)) return;
    map.reserve(strs.size());
    for (size_t i = map.size(); i < strs.size(); ++i) map.emplace(strs[i], (int32_t)i);
    lazy.store(false, std::memory_order_release);
  }

 private:
  mutable std::atomic<bool> lazy{false};
  mutable std::mutex lazyMu;
};

struct Segment {
  int part = 0;  // partition this segment belongs to
  int64_t n = 0;
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
  std::vector<double> mn, mx;              // segment elimination metadata
  std::vector<uint32_t> delHost;
  uint32_t* delDev = nullptr;
  int64_t ndel = 0;
  ~Segment();
};

struct WriteBuffer {
  std::vector<std::vector<uint8_t>> cols;
  std::vector<uint32_t> del;
  int64_t n = 0, ndel = 0;
  uint64_t version = 1;
  // Lazily refreshed VRAM copy used by GPU scans.
  std::mutex devMu;
  uint64_t devVersion = 0;
  // Columns change only by appends, except when replaced or cleared (gen bumps): then the VRAM copy
  // is re-uploaded whole, otherwise only rows [devN, n).
  uint64_t gen = 0, devGen = ~0ull;
  int64_t devN = 0;
  // Cached per-column value range of rows [0, rangeN) (colRange), extended as rows are appended.
  uint64_t rangeGen = ~0ull;
  int64_t rangeN = 0;
  std::vector<double> rmn, rmx;
  std::vector<void*> dev;
  uint32_t* delDev = nullptr;
  int64_t devCap = 0;
  ~WriteBuffer();
};


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
    bool nested = !held_.empty() || barge;  // inside another table read: never wait behind writers (no lock-order deadlock)
    std::unique_lock<std::mutex> g(m_);
    if (!nested) cv_.wait(g, [&] { return !w_ && waitingW_ == 0; });
    ++readers_;
    ++held();
  }
  bool try_lock_shared() {
    bool nested = !held_.empty();  // inside another table read: never wait behind writers (no lock-order deadlock)
    std::lock_guard<std::mutex> g(m_);
    if (!nested && (w_ || waitingW_)) return false;
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
  std::vector<ColumnMeta> cols;
  std::vector<std::unique_ptr<Dict>> dicts;  // per column, VARCHAR only
  int pk = -1, sortKey = -1;           // pk: single-column PRIMARY KEY with a hash index
  std::vector<int> pkCols;             // all PRIMARY KEY columns
  std::vector<int> shardKey;           // empty => keyless (round-robin)
  int nparts = 1;
  bool reference = false;
  int vram = 0;  // VRAM hint for columns without their own: 0 AUTO (the cache decides), 1 PIN, 2 NEVER
  int vramHint(int c) const { return cols[c].vram ? cols[c].vram : vram; }
  std::vector<std::unique_ptr<Segment>> segs;
  std::vector<std::unique_ptr<WriteBuffer>> bufs;  // one per partition
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
