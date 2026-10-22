#pragma once
// The engine's shared internals (helpers, binding, the bytecode compiler, ..): included by its sources
// (engine.cpp, engine_query.cpp, engine_snap.cpp). Everything at namespace scope is inline: one
// instance of each variable and function in the program.
#include <cuda_runtime.h>
#include "engine.h"

#include <future>
#include <sys/mman.h>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <functional>
#include <map>
#include <set>
#include <glob.h>
#include <openssl/md5.h>
#include <openssl/sha.h>
#include <random>
#include <zlib.h>
#include <semaphore>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <sys/stat.h>
#include <regex>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <unistd.h>
#include <dirent.h>
#include <fcntl.h>
#include <zstd.h>
#include "gpu.h"
#include "jit.h"
#include "cagra.h"
#include "embed.h"
#include "monitor.h"
#include "mysqlcdc.h"
#include "sources.h"

namespace gdb {
const std::string& tUserName();  // the session user (defined with the user code below)
extern thread_local bool tApplying;  // applying replicated records (repl.cpp)
// Read committed (see commitTxn): another session's open transaction has changes in this table.
inline bool foreignPending(const Table& t) {
  for (auto& [k, v] : t.txnPending)
    if (k != txnKey() && !v.commitSeq && (!v.hidden.empty() || !v.ghosts.empty())) return true;
  return false;
}
// Snapshot isolation (REPEATABLE READ): a snapshot transaction sees the commits up to tSnapSeq; the
// changes of later commits stay in Table::txnPending (retired, with their commitSeq) while it runs.
inline thread_local uint64_t tSnapSeq = 0;
inline std::atomic<uint64_t> gCommitSeq{1};
inline std::mutex gSnapMu;
inline std::multiset<uint64_t> gSnaps;  // active snapshots
inline bool snapshotsActive() {
  std::lock_guard<std::mutex> g(gSnapMu);
  return !gSnaps.empty();
}
inline thread_local std::vector<std::shared_ptr<DevVec<uint32_t>>> tTxnViews;  // visibility bitmaps of the current statement
bool regexLike(const std::string& s, const std::string& pat, bool ci);  // regex.cpp (RE2)
std::string regexReplace(const std::string& s, const std::string& pat, const std::string& rep);
bool tzOffset(const std::string& zone, int64_t secs, bool local, int64_t& offsetSecs);  // regex.cpp
bool regexSubstr(const std::string& s, const std::string& pat, bool ci, std::string& out, int64_t& pos);  // regex.cpp
Value geoFunction(const std::string& f, const std::vector<Value>& a);                   // regex.cpp

ExprP cloneExpr(const ExprP& e);  // query layer

inline const std::string kNullCell("\0NULL", 5);
ExprP cloneExpr(const ExprP& e);  // defined with the query layer

// ---- Collation. Default utf8mb4_general_ci (case-insensitive, like SingleStore); per session
// SET collation_connection = utf8mb4_bin for binary. Strings are dictionary encoded, so case
// folding happens once per dictionary entry, not per row.
inline std::atomic<bool> gCiDefault{true};
inline thread_local int tCi = -1;
inline bool ciColl() { return tCi >= 0 ? tCi != 0 : gCiDefault.load(); }
inline std::string foldCase(std::string_view x) {
  std::string r(x);
  for (auto& ch : r) ch = (char)tolower((unsigned char)ch);
  return r;
}
// Runs f(begin, end) over [0, n) on all cores in chunks (multiples of 32, so bitmap words are never
// shared). Workers inherit the session's collation.
inline void parallelFor(size_t n, const std::function<void(size_t, size_t)>& f, size_t minChunk = 32768) {
  const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
  const size_t nth = std::min<size_t>(hw, (n + minChunk - 1) / minChunk);
  if (nth <= 1) {
    f(0, n);
    return;
  }
  size_t chunk = (n + nth - 1) / nth;
  if (minChunk >= 32) chunk = (chunk + 31) & ~(size_t)31;  // bitmap words never shared
  const int ci = tCi;
  std::vector<std::thread> th;
  std::exception_ptr err;
  std::mutex errMu;
  for (size_t b = 0; b < n; b += chunk)
    th.emplace_back([&, b] {
      tCi = ci;
      try {
        f(b, std::min(n, b + chunk));
      } catch (...) {
        std::lock_guard<std::mutex> g(errMu);
        err = std::current_exception();
      }
    });
  for (auto& t : th) t.join();
  if (err) std::rethrow_exception(err);
}

// A LIKE matcher specialized on the pattern: '%x%' / 'x%' / '%x' / 'x' use substring search
// (memmem) instead of the general wildcard matcher.
inline std::function<bool(std::string_view)> likeMatcher(const std::string& pat) {
  bool special = pat.find('_') != std::string::npos || pat.find('\\') != std::string::npos;
  const size_t inner = pat.size() >= 2 ? pat.find('%', 1) : std::string::npos;
  if (!special) {
    const bool lead = !pat.empty() && pat[0] == '%', trail = pat.size() > (size_t)lead && pat.back() == '%';
    const std::string lit = pat.substr(lead, pat.size() - lead - (trail ? 1 : 0));
    if (lit.find('%') == std::string::npos && (inner == std::string::npos || inner == pat.size() - 1)) {
      if (lead && trail) return [lit](std::string_view s) { return memmem(s.data(), s.size(), lit.data(), lit.size()) != nullptr; };
      if (trail) return [lit](std::string_view s) { return s.size() >= lit.size() && memcmp(s.data(), lit.data(), lit.size()) == 0; };
      if (lead)
        return [lit](std::string_view s) { return s.size() >= lit.size() && memcmp(s.data() + s.size() - lit.size(), lit.data(), lit.size()) == 0; };
      return [lit](std::string_view s) { return s == lit; };
    }
  }
  if (!special) {  // only '%' wildcards: the pieces in order (memmem), the first / last anchored unless '%'
    std::vector<std::string> seg;
    size_t b = 0;
    for (size_t i = 0; i <= pat.size(); ++i)
      if (i == pat.size() || pat[i] == '%') {
        seg.push_back(pat.substr(b, i - b));
        b = i + 1;
      }
    // seg.front() / seg.back(): anchored at the start / end (empty when the pattern starts / ends with '%')
    return [seg](std::string_view s) {
      const std::string& head = seg.front();
      const std::string& tail = seg.back();
      if (s.size() < head.size() + tail.size()) return false;
      if (memcmp(s.data(), head.data(), head.size()) != 0) return false;
      if (memcmp(s.data() + s.size() - tail.size(), tail.data(), tail.size()) != 0) return false;
      size_t at = head.size();
      const size_t end = s.size() - tail.size();
      for (size_t k = 1; k + 1 < seg.size(); ++k) {
        const std::string& m = seg[k];
        if (m.empty()) continue;
        if (end - at < m.size()) return false;
        const void* f = memmem(s.data() + at, end - at, m.data(), m.size());
        if (!f) return false;
        at = (const char*)f - s.data() + m.size();
      }
      return true;
    };
  }
  return [pat](std::string_view s) { return likeMatch(s, pat); };
}

int collCmp(std::string_view a, std::string_view b);
void derivShareMap(const std::shared_ptr<std::vector<double>>& m);
// Sorted order of a dictionary (collation-aware): (rank -> code, code -> rank). Dictionaries only
// grow, so (dictionary, size, collation) identifies a version: cached, built with a parallel sort.
std::shared_ptr<std::vector<int32_t>> loadOrderFile(const std::string& path, size_t n);
struct SortedOrder {  // a dictionary's strings in collation order (Dict::getDerived "sorted", "sorted/ci")
  size_t n;
  std::shared_ptr<const std::vector<int32_t>> order;
  std::shared_ptr<std::vector<double>> rank;
};
// The cached order of the first n strings under binary (ci = false) or case-insensitive collation.
inline std::shared_ptr<const std::vector<int32_t>> sortedOrderCached(const Dict& d, bool ci, size_t n) {
  auto c = d.getDerived<SortedOrder>(ci ? "sorted/ci" : "sorted");
  return c && c->n == n ? c->order : nullptr;
}
// An order read back from a dictionary file's side file (snapshot load).
inline void sortedOrderInstall(const Dict& d, bool ci, std::shared_ptr<const std::vector<int32_t>> order) {
  auto rank = std::make_shared<std::vector<double>>(order->size());
  for (size_t r = 0; r < order->size(); ++r) (*rank)[(*order)[r]] = (double)r;
  d.setDerived(ci ? "sorted/ci" : "sorted", std::make_shared<SortedOrder>(SortedOrder{order->size(), order, rank}));
  derivShareMap(rank);
}

inline std::pair<std::shared_ptr<const std::vector<int32_t>>, std::shared_ptr<std::vector<double>>> sortedDict(const Dict& d) {
  using E = SortedOrder;
  const bool ci = ciColl();
  const std::string ckey = ci ? "sorted/ci" : "sorted";
  const size_t n = d.strs.size();
  if (auto c = d.getDerived<E>(ckey))
    if (c->n == n) return {c->order, c->rank};
  // One sort per dictionary at a time: a query that needs the order while warmDicts (or another
  // query) is sorting waits for that sort instead of repeating it.
  static std::mutex inflightMu;
  static std::map<std::pair<uint64_t, std::string>, std::shared_ptr<std::mutex>> inflight;
  std::shared_ptr<std::mutex> m;
  {
    std::lock_guard<std::mutex> g(inflightMu);
    auto& slot = inflight[{d.uid, ckey}];
    if (!slot) slot = std::make_shared<std::mutex>();
    m = slot;
  }
  std::lock_guard<std::mutex> sortLock(*m);
  struct Forget {  // (the entry goes once nobody holds it)
    std::shared_ptr<std::mutex>& m;
    uint64_t uid;
    const std::string& k;
    ~Forget() {
      std::lock_guard<std::mutex> g(inflightMu);
      auto it = inflight.find({uid, k});
      if (it != inflight.end() && it->second.use_count() <= 2) inflight.erase(it);
    }
  } forget{m, d.uid, ckey};
  std::shared_ptr<const std::vector<int32_t>> prev;  // order of an older (smaller) version
  if (auto c = d.getDerived<E>(ckey)) {
    if (c->n == n) return {c->order, c->rank};
    if (c->n < n && n - c->n <= n / 8) prev = c->order;
  }
  if (!d.orderFile[ci].empty()) {  // saved by the snapshot this dictionary was loaded from
    const std::string path = d.orderFile[ci];
    d.orderFile[ci].clear();
    if (auto o = loadOrderFile(path, d.fileStrs)) {
      d.orderFileStrs[ci] = d.fileStrs;
      sortedOrderInstall(d, ci, o);
      if (auto c = d.getDerived<E>(ckey)) {
        if (c->n == n) return {c->order, c->rank};
        if (c->n < n && n - c->n <= n / 8) prev = c->order;  // (strings added since: merged in below)
      }
    }
  }
  const auto tStart = std::chrono::steady_clock::now();
  if (prev) {  // the dictionary grew a little: binary-search the new strings into the old order
    const size_t n0 = prev->size();
    auto less = [&](int32_t a, int32_t b) {
      const int c = collCmp(d.strs[a], d.strs[b]);
      return c ? c < 0 : a < b;
    };
    std::vector<int32_t> fresh;
    for (size_t i = n0; i < n; ++i) fresh.push_back((int32_t)i);
    std::sort(fresh.begin(), fresh.end(), less);
    std::vector<size_t> at(fresh.size());
    parallelFor(fresh.size(), [&](size_t b, size_t e) {
      for (size_t k = b; k < e; ++k) at[k] = std::lower_bound(prev->begin(), prev->end(), fresh[k], less) - prev->begin();
    }, 64);
    auto order = std::make_shared<std::vector<int32_t>>();
    order->reserve(n);
    size_t from = 0;
    for (size_t k = 0; k < fresh.size(); ++k) {
      order->insert(order->end(), prev->begin() + from, prev->begin() + at[k]);
      order->push_back(fresh[k]);
      from = at[k];
    }
    order->insert(order->end(), prev->begin() + from, prev->end());
    auto rank = std::make_shared<std::vector<double>>(n);
    for (size_t r = 0; r < n; ++r) (*rank)[(*order)[r]] = (double)r;
    if (envSetting("TRACE"))
      fprintf(stderr, "[trace] sorted dictionary of %zu strings incrementally (+%zu) in %.1f ms\n", n, n - n0,
              std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tStart).count());
    d.setDerived(ckey, std::make_shared<E>(E{n, order, rank}));
    derivShareMap(rank);
    return {order, rank};
  }
  auto order = std::make_shared<std::vector<int32_t>>(n);
  for (size_t i = 0; i < n; ++i) (*order)[i] = (int32_t)i;
  // MSD sort on 8-byte big-endian chunks (lower-cased under _ci, which is strcasecmp's order): sort
  // by the first chunk, then re-sort each run of equal chunks by the next one. Plain comparison
  // sorts crawl on URL-like data, where most strings share their first 8+ bytes.
  const bool fold = ci;
  std::vector<uint64_t> key(n);
  auto chunk = [&](int32_t i, size_t off) {
    const std::string_view s = d.strs[i];
    uint64_t p = 0;
    for (size_t k = off; k < off + 8; ++k) {
      unsigned char ch = k < s.size() ? (unsigned char)s[k] : 0;
      if (fold) ch = (unsigned char)tolower(ch);
      p = (p << 8) | ch;
    }
    return p;
  };
  auto byKey = [&](int32_t a, int32_t b) { return key[a] != key[b] ? key[a] < key[b] : a < b; };
  struct Run {
    size_t lo, hi, off;
  };
  // Runs of equal keys (that still have bytes left to compare) within [lo, hi).
  auto runsOf = [&](size_t lo, size_t hi, size_t off, std::vector<Run>& out) {
    for (size_t i = lo; i < hi;) {
      size_t j = i + 1;
      size_t maxLen = d.strs[(*order)[i]].size();
      while (j < hi && key[(*order)[j]] == key[(*order)[i]]) maxLen = std::max(maxLen, d.strs[(*order)[j++]].size());
      if (j - i > 1 && maxLen > off + 8) out.push_back({i, j, off + 8});
      i = j;
    }
  };
  // Sort [lo, hi) by key: on all cores (chunks, then pairwise merges) when the range is large.
  auto sortRange = [&](size_t lo, size_t hi) {
    const size_t nth = std::max<size_t>(1, std::min<size_t>(std::thread::hardware_concurrency(), (hi - lo) / 65536));
    std::vector<size_t> bounds;
    for (size_t t = 0; t <= nth; ++t) bounds.push_back(lo + (hi - lo) * t / nth);
    parallelFor(nth, [&](size_t b, size_t e) {
      for (size_t t = b; t < e; ++t) std::sort(order->begin() + bounds[t], order->begin() + bounds[t + 1], byKey);
    }, 1);
    for (size_t w = 1; w < nth; w *= 2)
      for (size_t t = 0; t + w < nth; t += 2 * w)
        std::inplace_merge(order->begin() + bounds[t], order->begin() + bounds[t + w], order->begin() + bounds[std::min(nth, t + 2 * w)], byKey);
  };
  // Small runs: one thread each, recursively. Large runs (shared URL prefixes): on all cores.
  std::function<void(size_t, size_t, size_t)> refineSmall = [&](size_t lo, size_t hi, size_t off) {
    for (size_t i = lo; i < hi; ++i) key[(*order)[i]] = chunk((*order)[i], off);
    std::sort(order->begin() + lo, order->begin() + hi, byKey);
    std::vector<Run> sub;
    runsOf(lo, hi, off, sub);
    for (auto& r : sub) refineSmall(r.lo, r.hi, r.off);
  };
  std::function<void(std::vector<Run>&)> refineAll = [&](std::vector<Run>& runs) {
    std::vector<Run> small;
    for (auto& r : runs) {
      if (r.hi - r.lo < 131072) {
        small.push_back(r);
        continue;
      }
      parallelFor(r.hi - r.lo, [&](size_t b, size_t e) {
        for (size_t i = r.lo + b; i < r.lo + e; ++i) key[(*order)[i]] = chunk((*order)[i], r.off);
      });
      sortRange(r.lo, r.hi);
      std::vector<Run> sub;
      runsOf(r.lo, r.hi, r.off, sub);
      refineAll(sub);
    }
    std::sort(small.begin(), small.end(), [](const Run& a, const Run& b) { return a.hi - a.lo > b.hi - b.lo; });
    std::atomic<size_t> next{0};
    parallelFor(std::max<size_t>(1, std::min<size_t>(std::thread::hardware_concurrency(), small.size())), [&](size_t, size_t) {
      for (size_t i; (i = next++) < small.size();) refineSmall(small[i].lo, small[i].hi, small[i].off);
    }, 1);
  };
  parallelFor(n, [&](size_t b, size_t e) {
    for (size_t i = b; i < e; ++i) key[i] = chunk((int32_t)i, 0);
  });
  const auto tK = std::chrono::steady_clock::now();
  sortRange(0, n);
  const auto tS = std::chrono::steady_clock::now();
  std::vector<Run> runs;
  runsOf(0, n, 0, runs);
  size_t inRuns = 0, big = 0;
  for (auto& r : runs) inRuns += r.hi - r.lo, big = std::max(big, r.hi - r.lo);
  refineAll(runs);
  if (envSetting("TRACE")) {
    auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    fprintf(stderr, "[trace]   keys %.1f ms, first sort %.1f ms, %zu runs (%zu strings, largest %zu), refine %.1f ms\n", ms(tStart, tK),
            ms(tK, tS), runs.size(), inRuns, big, ms(tS, std::chrono::steady_clock::now()));
  }
  auto rank = std::make_shared<std::vector<double>>(n);
  for (size_t r = 0; r < n; ++r) (*rank)[(*order)[r]] = (double)r;
  if (envSetting("TRACE"))
    fprintf(stderr, "[trace] sorted dictionary of %zu strings in %.1f ms\n", n,
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tStart).count());
  d.setDerived(ckey, std::make_shared<E>(E{n, order, rank}));
  derivShareMap(rank);
  return {order, rank};
}

// After a bulk load: sort the table's large dictionaries in the background (one pass per table at
// a time, under the read lock), so the first ORDER BY / MIN / MAX / comparison on a string column
// does not pay for the sort.
inline void (*gFtsWarm)(const Dict& d) = nullptr;  // builds a FULLTEXT column's inverted index (set by the full-text code)
// Background dictionary warm-up yields the CPU to queries: the thread, and the threads it starts,
// run at nice 10.
inline void lowPriority() { setpriority(PRIO_PROCESS, (id_t)syscall(SYS_gettid), 10); }

inline void warmDicts(const std::shared_ptr<Table>& t) {
  static std::mutex mu;
  static std::set<const Table*> busy;
  {
    std::lock_guard<std::mutex> g(mu);
    if (!busy.insert(t.get()).second) return;
  }
  std::thread([t] {
    lowPriority();
    try {
      // (the table lock only to take the dictionaries: they are safe to read and index without it, and
      // holding it through seconds of sorting kept writers waiting)
      std::vector<std::shared_ptr<Dict>> ds;
      {
        std::shared_lock<FairSharedMutex> lk(t->mu);
        ds = t->dicts;
      }
      for (auto& d : ds)  // string -> code indexes first (dictionaries loaded from a snapshot are indexed lazily)
        if (d) d->index();
      for (int ci : {0, 1}) {  // binary collation first (no per-table collation is stored)
        tCi = ci;
        for (auto& d : ds)
          if (d && d->strs.size() >= 50000) sortedDict(*d);
      }
      std::shared_lock<FairSharedMutex> lk(t->mu);
      for (int c : t->fulltextCols)
        if (c >= 0 && c < (int)t->dicts.size() && t->dicts[c] && gFtsWarm) gFtsWarm(*t->dicts[c]);
    } catch (...) {
    }
    std::lock_guard<std::mutex> g(mu);
    busy.erase(t.get());
  }).detach();
}

// ---- Cache of dictionary-derived data: bitmaps of string predicates (LIKE, comparisons), code ->
// value maps of encoded expressions, derived dictionaries. Dictionaries only grow, so an entry is
// valid while its dictionary has the same size. Device copies are made once and shared (holders
// keep them alive, so eviction never frees memory a running query uses).
struct DictDerived {
  const Dict* d = nullptr;
  size_t n = 0;
  std::shared_ptr<std::vector<uint32_t>> bits;
  std::shared_ptr<std::vector<double>> map;
  std::shared_ptr<Dict> out;
  std::shared_ptr<DevVec<uint32_t>> bitsDev;
  std::shared_ptr<DevVec<double>> mapDev;
  size_t bytes = 0;
};
inline std::mutex gDerivMu;
inline std::unordered_map<std::string, DictDerived> gDeriv;
// Host map -> (the map, kept alive so its address is never reused; its device copy, made lazily).
inline std::unordered_map<const void*, std::pair<std::shared_ptr<std::vector<double>>, std::shared_ptr<DevVec<double>>>> gMapDev;
inline size_t gDerivBytes = 0;
inline std::string derivKey(const Dict& d, const std::string& what) {
  return std::to_string(d.uid) + "|" + (ciColl() ? "i" : "b") + "|" + what;
}
inline bool derivGet(const Dict& d, const std::string& key, DictDerived& out) {
  std::lock_guard<std::mutex> g(gDerivMu);
  auto it = gDeriv.find(key);
  if (it == gDeriv.end() || it->second.d != &d || it->second.n != d.strs.size()) return false;
  out = it->second;
  return true;
}
// An entry for an older (smaller) version of the dictionary: it can be extended with the new entries.
inline bool derivGetStale(const Dict& d, const std::string& key, DictDerived& out) {
  std::lock_guard<std::mutex> g(gDerivMu);
  auto it = gDeriv.find(key);
  if (it == gDeriv.end() || it->second.d != &d || it->second.n >= d.strs.size() || !(it->second.map || it->second.bits)) return false;
  out = it->second;
  return true;
}
inline void derivPut(const std::string& key, DictDerived e) {
  e.bytes = (e.bits ? e.bits->size() * 4 : 0) + (e.map ? e.map->size() * 8 : 0);
  std::lock_guard<std::mutex> g(gDerivMu);
  if (gDerivBytes + e.bytes > ((size_t)1 << 30)) {  // simple bound: start over
    gDeriv.clear();
    gMapDev.clear();
    gDerivBytes = 0;
  }
  if (auto old = gDeriv.find(key); old != gDeriv.end() && old->second.map) gMapDev.erase(old->second.map.get());
  if (e.map) gMapDev[e.map.get()] = {e.map, nullptr};  // eligible for a shared device copy
  gDerivBytes += e.bytes;
  gDeriv[key] = std::move(e);
}
// Device copy of a cached map (made once), or null when the map is not a cached one.
inline std::shared_ptr<DevVec<double>> derivMapDev(const std::vector<double>* m) {
  std::lock_guard<std::mutex> g(gDerivMu);
  auto it = gMapDev.find(m);
  if (it == gMapDev.end()) return nullptr;
  auto& dev = it->second.second;
  if (!dev) {
    dev = std::make_shared<DevVec<double>>(std::max<int64_t>((int64_t)m->size(), 1));
    if (!m->empty()) h2d(dev->p, m->data(), m->size() * 8);
  }
  return dev;
}
// A long-lived map (e.g. sorted-dictionary ranks) whose device copy should be shared too.
inline void derivShareMap(const std::shared_ptr<std::vector<double>>& m) {
  std::lock_guard<std::mutex> g(gDerivMu);
  if (!gMapDev.count(m.get())) {
    gMapDev[m.get()] = {m, nullptr};
    gDerivBytes += m->size() * 8;
  }
}

inline int collCmp(std::string_view a, std::string_view b) {
  if (ciColl()) {  // (strcasecmp's order, without needing terminated strings)
    const size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) {
      const int x = tolower((unsigned char)a[i]), y = tolower((unsigned char)b[i]);
      if (x != y) return x < y ? -1 : 1;
    }
    return a.size() < b.size() ? -1 : a.size() > b.size() ? 1 : 0;
  }
  int c = a.compare(b);
  return c < 0 ? -1 : c > 0 ? 1 : 0;
}
// code -> canonical code (first entry with the same case-folded text), or null when the
// dictionary has no case variants (then nothing needs mapping). Cached per dictionary size.
inline std::shared_ptr<std::vector<double>> ciCanon(const Dict& d) {
  using Slot = std::pair<size_t, std::shared_ptr<std::vector<double>>>;
  if (auto c = d.getDerived<Slot>("cicanon"); c && c->first == d.strs.size()) return c->second;
  Slot slot;
  std::unordered_map<std::string, int32_t> first;
  auto m = std::make_shared<std::vector<double>>(d.strs.size());
  bool any = false;
  for (size_t i = 0; i < d.strs.size(); ++i) {
    auto [it, fresh] = first.emplace(foldCase(d.strs[i]), (int32_t)i);
    (*m)[i] = it->second;
    any |= !fresh;
  }
  slot = {d.strs.size(), any ? m : nullptr};
  d.setDerived("cicanon", std::make_shared<Slot>(slot));
  return slot.second;
}

// Two dictionaries' codes -> ranks in the collation order of their union (equal strings under the
// collation get equal ranks), so columns of different tables compare on the GPU. Cached per pair
// and sizes.
inline std::pair<std::shared_ptr<std::vector<double>>, std::shared_ptr<std::vector<double>>> crossRank(const Dict& a, const Dict& b) {
  using Maps = std::pair<std::shared_ptr<std::vector<double>>, std::shared_ptr<std::vector<double>>>;
  // cached on a, keyed by b's identity and both sizes
  const std::string key = "cross/" + std::to_string(b.uid) + "/" + std::to_string(a.strs.size()) + "/" + std::to_string(b.strs.size()) + (ciColl() ? "/ci" : "");
  if (auto c = a.getDerived<Maps>(key)) return *c;
  std::vector<std::pair<std::string_view, int64_t>> all;  // (string, code; b's codes offset by |a|)
  all.reserve(a.strs.size() + b.strs.size());
  for (size_t i = 0; i < a.strs.size(); ++i) all.push_back({a.strs[i], (int64_t)i});
  for (size_t i = 0; i < b.strs.size(); ++i) all.push_back({b.strs[i], (int64_t)(a.strs.size() + i)});
  std::sort(all.begin(), all.end(), [](auto& x, auto& y) { return collCmp(x.first, y.first) < 0; });
  auto ma = std::make_shared<std::vector<double>>(a.strs.size()), mb = std::make_shared<std::vector<double>>(b.strs.size());
  double rk = 0;
  for (size_t j = 0; j < all.size(); ++j) {
    if (j && collCmp(all[j - 1].first, all[j].first) != 0) ++rk;
    const int64_t c = all[j].second;
    if (c < (int64_t)a.strs.size()) (*ma)[c] = rk;
    else (*mb)[c - a.strs.size()] = rk;
  }
  if (a.derivedCount() > 64) a.dropDerived("cross/");
  a.setDerived(key, std::make_shared<Maps>(ma, mb));
  return {ma, mb};
}

// (namespace {: removed, engine_core.h is shared by the engine's sources)

// ================================================================== binding

struct BindCtx {
  std::vector<std::shared_ptr<Table>> tabs;
  std::vector<std::string> aliases;
  std::vector<std::vector<std::string>> usingCols;  // per FROM entry: JOIN .. USING columns
  bool rowWise = false;  // UPDATE SET: evaluated for the matched rows only (no per-dictionary encoding)
  bool inAgg = false;
  // Binding the select list of an unordered, LIMIT-ed plain projection: string functions are
  // evaluated per output row on the host instead of over the whole (large) dictionary.
  bool lazyStr = false;
  // Encoded string expressions: derived dictionaries (for VARCHAR-valued "#ENC" nodes) and the
  // code -> value maps their GPU form applies with OP_DICTMAP.
  std::vector<std::unique_ptr<Dict>> ownDicts;
  std::vector<std::shared_ptr<Dict>> sharedDicts;  // derived dictionaries from the dictionary cache
  std::unordered_map<const Expr*, Dict*> exprDict;
  // MIN/MAX over strings aggregate ranks; rank -> code of the source dictionary (exprDict[e]).
  std::unordered_map<const Expr*, std::shared_ptr<const std::vector<int32_t>>> unrank;
  std::set<int> outerTabs;  // tables on the NULL-extended side of an outer join
  Dict* dict(const Expr& e) const {
    if (e.kind == Expr::COL) return e.bt >= 0 ? tabs[e.bt]->dicts[e.bc].get() : nullptr;
    auto it = exprDict.find(&e);
    return it == exprDict.end() ? nullptr : it->second;
  }
};
bool isJsonExpr(const ExprP& x, const BindCtx* c);  // defined with the host functions
std::string jsonOfValue(const Value& v, bool raw);

inline bool isAggName(const std::string& n) { return n == "SUM" || n == "COUNT" || n == "AVG" || n == "MIN" || n == "MAX"; }
// Aggregates computed by the host aggregation path of the query layer.
inline bool isHostAggName(const std::string& n) {
  return n == "GROUP_CONCAT" || n == "JSON_ARRAYAGG" || n == "JSON_AGG" || n == "JSON_OBJECTAGG" || n == "STRING_AGG" ||
         n == "STDDEV" || n == "STD" || n == "STDDEV_POP" || n == "STDDEV_SAMP" || n == "VARIANCE" || n == "VAR_POP" ||
         n == "VAR_SAMP" || n == "ANY_VALUE" || n == "MEDIAN" || n == "BIT_AND" || n == "BIT_OR" || n == "BIT_XOR" ||
         n == "FIRST" || n == "LAST" || n == "PERCENTILE_CONT" || n == "PERCENTILE_DISC" || n == "APPROX_PERCENTILE";
}
inline bool isAgg(const ExprP& e) { return e->kind == Expr::FUNC && isAggName(e->op); }
inline bool isCmp(const std::string& o) { return o == "=" || o == "<>" || o == "<" || o == "<=" || o == ">" || o == ">="; }

inline bool hasAgg(const ExprP& e) {
  if (!e) return false;
  if (isAgg(e)) return true;
  for (auto& a : e->args)
    if (hasAgg(a)) return true;
  return false;
}

// Canonical form after binding: columns are identified by (table, column) index.
inline std::string canon(const ExprP& e) {
  switch (e->kind) {
    case Expr::COL: return "#" + std::to_string(e->bt) + "." + std::to_string(e->bc);
    case Expr::LIT: return (e->lit.kind == Value::STR ? "'" + e->lit.s + "'" : e->lit.toString()) + ":" + std::to_string((int)e->lit.kind);
    default: break;
  }
  // Encoded string expressions (#ENC:<n>, a fresh number per occurrence): the original expression.
  if (e->kind == Expr::FUNC && e->op.rfind("#ENC:", 0) == 0 && e->args.size() == 2) return "ENC(" + canon(e->args[1]) + ")";
  std::string r = std::to_string((int)e->kind) + e->op + (e->neg ? "!" : "") + (e->distinct ? "D" : "") + "(";
  for (auto& a : e->args) r += canon(a) + ",";
  return r + ")";
}

void bind(ExprP& e, BindCtx& c);

inline void bindChildren(ExprP& e, BindCtx& c) {
  for (auto& a : e->args) bind(a, c);
}

inline int64_t addInterval(int64_t days, int64_t n, const std::string& unit) {
  if (unit == "DAY") return days + n;
  int64_t y;
  unsigned m, d;
  civilFromDays(days, y, m, d);
  int64_t months = (int64_t)y * 12 + (m - 1) + (unit == "MONTH" ? n : n * 12);
  y = months >= 0 ? months / 12 : (months - 11) / 12;
  m = (unsigned)(months - y * 12) + 1;
  static const unsigned mdays[] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
  unsigned lim = mdays[m - 1] + (m == 2 && ((y % 4 == 0 && y % 100 != 0) || y % 400 == 0));
  return daysFromCivil(y, m, std::min(d, lim));
}

inline thread_local int64_t tLastInsertId = 0;  // per connection, like MySQL
// "#INBITS": an IN (SELECT ..) value set as a device bitmap (bit k: value base + k), host copy on demand.
struct DevBits {
  uint32_t* p = nullptr;
  int64_t nbits = 0, base = 0;
  std::once_flag once;
  std::vector<uint32_t> host;
  ~DevBits() { if (p) devFree(p); }
  bool test(int64_t v) {
    std::call_once(once, [&] {
      host.resize((size_t)(nbits + 31) / 32);
      if (!host.empty()) d2h(host.data(), p, host.size() * 4);
    });
    const int64_t k = v - base;
    return k >= 0 && k < nbits && (host[(size_t)k >> 5] >> (k & 31) & 1u);
  }
};
inline thread_local int64_t tFoundRows = 0;
inline thread_local bool tSnapTryOnly = false;  // automatic snapshots: skip the round rather than wait for writers
inline thread_local bool tWriteGateHeld = false;  // this thread's statement holds Engine::writeGate_ (shared)
inline thread_local uint64_t tAckLsn = 0;          // semi-sync: the replica acknowledgement to wait for once the gate is let go     // FOUND_ROWS(): rows of the last SELECT (without its LIMIT if SQL_CALC_FOUND_ROWS)
inline thread_local std::vector<std::string> tTempTables;  // CREATE TEMPORARY TABLE: dropped when the connection ends
inline std::string dbOf(const std::string& name) {
  size_t d = name.find('.');
  return d == std::string::npos ? currentDb() : name.substr(0, d);
}
inline std::string plainName(const std::string& name) {
  size_t d = name.find('.');
  return d == std::string::npos ? name : name.substr(d + 1);
}
inline thread_local bool tWantVals = false;     // doSelect returns typed rows (for nested queries)
inline thread_local bool tWinOuterCols = false;  // windows(): its final projection may come back as columns

inline int64_t nowUs() {
  return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

// The statement's clock and random numbers: NOW() / CURRENT_TIMESTAMP defaults read tStmtTimeUs, and
// RAND() / UUID() draw from tStmtRng, seeded with tStmtSeed. Both are set when a statement starts
// and written with it to the WAL ("/*ws <us> <seed>*/ " in front), so a replay (crash recovery,
// replicas, point-in-time restores) computes the same values, as MySQL's binlog does.
inline thread_local int64_t tStmtTimeUs = 0;
inline thread_local uint64_t tStmtSeed = 0;
inline thread_local std::mt19937_64 tStmtRng;
inline int64_t stmtNowUs() { return tStmtTimeUs ? tStmtTimeUs : nowUs(); }
inline void beginStatementClock() {
  static std::atomic<uint64_t> salt{std::random_device{}() ^ ((uint64_t)std::random_device{}() << 32)};
  tStmtTimeUs = nowUs();
  tStmtSeed = salt.fetch_add(0x9E3779B97F4A7C15ull) ^ (uint64_t)tStmtTimeUs;
  tStmtRng.seed(tStmtSeed);
}
inline std::string clockPrefix() {
  char b[64];
  snprintf(b, sizeof b, "/*ws %lld %llu*/ ", (long long)tStmtTimeUs, (unsigned long long)tStmtSeed);
  return b;
}
// A logged statement's clock prefix (replay): taken off, and the statement's clock set from it.
inline std::string takeClockPrefix(const std::string& sql) {
  if (sql.compare(0, 5, "/*ws ") != 0) return sql;
  const size_t end = sql.find("*/");
  if (end == std::string::npos) return sql;
  long long t = 0;
  unsigned long long seed = 0;
  if (sscanf(sql.c_str() + 5, "%lld %llu", &t, &seed) == 2) {
    tStmtTimeUs = t;
    tStmtSeed = seed;
    tStmtRng.seed(seed);
  }
  size_t from = end + 2;
  while (from < sql.size() && sql[from] == ' ') ++from;
  return sql.substr(from);
}

inline ExprP boundNode(Expr::Kind k, const std::string& op, std::vector<ExprP> args, LType ty) {
  auto e = std::make_shared<Expr>();
  e->kind = k;
  e->op = op;
  e->args = std::move(args);
  e->ty = ty;
  e->bound = true;
  return e;
}
inline ExprP mkCall(uint16_t id, std::vector<ExprP> args, LType ty) {
  ExprP e = boundNode(Expr::FUNC, (args.size() == 1 ? "#CALL1:" : "#CALL2:") + std::to_string(id), std::move(args), ty);
  e->fnId = id;
  return e;
}
inline ExprP boundLit(Value v) {
  ExprP e = mkLit(std::move(v));
  e->ty = e->lit.isNull() ? LType::DOUBLE
        : e->lit.kind == Value::INT ? LType::BIGINT : e->lit.kind == Value::DBL ? LType::DOUBLE
        : e->lit.kind == Value::DATE ? LType::DATE : e->lit.kind == Value::DATETIME ? LType::DATETIME : LType::VARCHAR;
  e->bound = true;
  return e;
}
inline bool isNullLit(const ExprP& e) { return e->kind == Expr::LIT && e->lit.isNull(); }
inline bool allLits(const ExprP& e) {
  if (e->kind == Expr::COL || e->kind == Expr::STAR) return false;
  if (e->kind == Expr::FUNC && isAggName(e->op)) return false;
  for (auto& a : e->args)
    if (!allLits(a)) return false;
  return true;
}
Value evalConst(const ExprP& e);  // constant folding (evalPost without context)

// A string literal used where a date / datetime is expected.
inline ExprP coerceTemporal(const ExprP& x, LType want) {
  if (x->kind != Expr::LIT || x->lit.kind != Value::STR) return x;
  if (want == LType::DATE) {
    int32_t d;
    if (parseDate(x->lit.s, d)) return boundLit(Value::Dt(d));
  }
  int64_t us;
  if (parseDateTime(x->lit.s, us)) return want == LType::DATE ? boundLit(Value::Dt(floorDiv(us, kUsPerDay))) : boundLit(Value::Ts(us));
  throw SqlError("Incorrect " + std::string(want == LType::DATE ? "DATE" : "DATETIME") + " value: '" + x->lit.s + "'", 1292);
}
inline ExprP asDate(const ExprP& x) {
  ExprP y = coerceTemporal(x, LType::DATE);
  return y->ty == LType::DATETIME ? mkCall(FN_DT_DATE, {y}, LType::DATE) : y;
}
inline ExprP asDateTime(const ExprP& x) {
  ExprP y = coerceTemporal(x, LType::DATETIME);
  return y->ty == LType::DATE ? mkCall(FN_DATE_TO_DT, {y}, LType::DATETIME) : y;
}

// date/datetime +/- INTERVAL n unit
inline ExprP intervalArith(ExprP dt, int64_t n, const std::string& unit) {
  static const std::map<std::string, int64_t> us = {{"MICROSECOND", 1}, {"SECOND", 1000000ll}, {"MINUTE", 60000000ll},
                                                     {"HOUR", 3600000000ll}, {"DAY", kUsPerDay}, {"WEEK", 7 * kUsPerDay}};
  const int64_t months = unit == "MONTH" ? n : unit == "QUARTER" ? 3 * n : unit == "YEAR" ? 12 * n : 0;
  dt = coerceTemporal(dt, LType::DATETIME);
  if (dt->kind == Expr::LIT && dt->lit.kind == Value::DATETIME && dt->lit.i % kUsPerDay == 0 && !us.count(unit))
    dt = boundLit(Value::Dt(dt->lit.i / kUsPerDay));  // '1995-01-01' + INTERVAL 1 MONTH stays a DATE
  if (dt->kind == Expr::LIT && dt->lit.kind == Value::DATETIME && dt->lit.i % kUsPerDay == 0 && (unit == "DAY" || unit == "WEEK"))
    dt = boundLit(Value::Dt(dt->lit.i / kUsPerDay));
  ExprP r;
  const bool sub = unit != "DAY" && unit != "WEEK" && us.count(unit);
  if (dt->ty == LType::DATETIME || sub) {
    dt = asDateTime(dt);
    r = months ? mkCall(FN_DT_ADD_MONTHS, {dt, boundLit(Value::I(months))}, LType::DATETIME)
               : boundNode(Expr::BIN, "+", {dt, boundLit(Value::I(n * us.at(unit)))}, LType::DATETIME);
  } else {
    r = months ? mkCall(FN_ADD_MONTHS, {dt, boundLit(Value::I(months))}, LType::DATE)
               : boundNode(Expr::BIN, "+", {dt, boundLit(Value::I(unit == "WEEK" ? 7 * n : n))}, LType::DATE);
  }
  if (allLits(r)) return boundLit(evalConst(r));
  return r;
}

void bind(ExprP& e, BindCtx& c);
void encodeStrings(ExprP& e, BindCtx& c);  // defined after evalPost
ExprP strToNum(const ExprP& x, BindCtx& c, bool integer = false);  // string -> number (defined with bind)
void unifyStrings(ExprP& e, BindCtx& c);

// Scalar (non-aggregate) functions: lowered to VM-level calls, CASE, or other expressions.
inline void bindFunc(ExprP& e, BindCtx& c) {
  const std::string f = e->op;
  // Constants evaluated at bind time.
  if (f == "NOW" || f == "CURRENT_TIMESTAMP" || f == "SYSDATE" || f == "LOCALTIME" || f == "LOCALTIMESTAMP" ||
      f == "UTC_TIMESTAMP") {
    e = boundLit(Value::Ts(stmtNowUs() / 1000000 * 1000000));
    return;
  }
  if (f == "CURDATE" || f == "CURRENT_DATE" || f == "UTC_DATE") { e = boundLit(Value::Dt(floorDiv(stmtNowUs(), kUsPerDay))); return; }
  if (f == "PI") { e = boundLit(Value::D(3.141592653589793)); return; }
  if (f == "LAST_INSERT_ID") { e = boundLit(Value::I(tLastInsertId)); return; }
  if (f == "FOUND_ROWS") { e = boundLit(Value::I(tFoundRows)); return; }
  if (f == "UNIX_TIMESTAMP" && e->args.empty()) { e = boundLit(Value::I(stmtNowUs() / 1000000)); return; }
  for (auto& a : e->args) bind(a, c);
  auto& A = e->args;
  const size_t n = A.size();
  auto argc = [&](size_t lo, size_t hi) {
    if (n < lo || n > hi) throw SqlError("Incorrect parameter count in the call to native function '" + f + "'", 1582);
  };
  auto numTy = [](const std::vector<ExprP>& xs) {
    LType t = LType::BIGINT;
    for (auto& x : xs) {
      if (x->ty == LType::DOUBLE) t = LType::DOUBLE;
      else if (x->ty == LType::DECIMAL && t != LType::DOUBLE) t = LType::DECIMAL;
      else if ((x->ty == LType::DATE || x->ty == LType::DATETIME) && t != LType::DOUBLE && t != LType::DECIMAL) t = x->ty;
    }
    return t;
  };
  auto maxScale = [](const std::vector<ExprP>& xs) {  // DECIMAL result scale: the widest argument's
    int sc = 0;
    for (auto& x : xs) if (x->ty == LType::DECIMAL) sc = std::max(sc, (int)x->scale);
    return sc;
  };
  auto finish = [&](ExprP r) {
    e = r;
    if (allLits(e) && !isNullLit(e) && e->kind != Expr::LIT) e = boundLit(evalConst(e));
  };
  if (f == "ISNULL" || f == "ISNOTNULL" || f == "#INSORTED" || f == "#INBITS") { argc(1, 1); e->ty = LType::INT; e->bound = true; return; }
  if (f == "ISTRUE" || f == "ISFALSE") {
    argc(1, 1);
    ExprP one = boundLit(Value::I(1)), zero = boundLit(Value::I(0));
    if (f == "ISTRUE") finish(boundNode(Expr::CASE, "", {A[0], one, zero}, LType::INT));
    else finish(boundNode(Expr::CASE, "", {boundNode(Expr::FUNC, "ISNULL", {A[0]}, LType::INT), zero, A[0], zero, one}, LType::INT));
    return;
  }
  if (f == "COALESCE" || f == "IFNULL" || f == "NVL") {
    argc(2, 64);
    LType t = A[0]->ty;
    for (auto& x : A) if (!isNullLit(x) && x->ty != t) t = numTy(A);
    ExprP r = A[n - 1];
    for (size_t i = n - 1; i-- > 0;) {
      r = boundNode(Expr::FUNC, "COALESCE2", {A[i], r}, t);
      if (t == LType::DECIMAL) r->scale = maxScale(A);
    }
    finish(r);
    if (e->ty == LType::VARCHAR && e->kind != Expr::LIT) unifyStrings(e, c);
    return;
  }
  if (f == "NULLIF") {
    argc(2, 2);
    e->ty = A[0]->ty;
    e->bound = true;
    if (e->ty == LType::VARCHAR) unifyStrings(e, c);
    return;
  }
  if (f == "IF" || f == "IIF") {
    argc(3, 3);
    ExprP r = boundNode(Expr::CASE, "", {A[0], A[1], A[2]}, LType::BIGINT);
    r->bound = false;  // let CASE binding compute the result type
    bind(r, c);
    e = r;
    return;
  }
  static const std::map<std::string, uint16_t> math1 = {
      {"ABS", FN_ABS}, {"CEIL", FN_CEIL}, {"CEILING", FN_CEIL}, {"FLOOR", FN_FLOOR}, {"SQRT", FN_SQRT}, {"EXP", FN_EXP},
      {"LN", FN_LN}, {"LOG10", FN_LOG10}, {"LOG2", FN_LOG2}, {"SIN", FN_SIN}, {"COS", FN_COS}, {"TAN", FN_TAN},
      {"ASIN", FN_ASIN}, {"ACOS", FN_ACOS}, {"SIGN", FN_SIGN}, {"DEGREES", FN_DEGREES}, {"RADIANS", FN_RADIANS}};
  if (math1.count(f) || (f == "ROUND" && n == 1) || (f == "LOG" && n == 1) || (f == "ATAN" && n == 1)) {
    argc(1, 1);
    uint16_t id = f == "ROUND" ? FN_ROUND : f == "LOG" ? FN_LN : f == "ATAN" ? FN_ATAN : math1.at(f);
    bool keepInt = (id == FN_ABS || id == FN_SIGN) && A[0]->ty != LType::DOUBLE;
    LType ty = keepInt || id == FN_CEIL || id == FN_FLOOR || (id == FN_ROUND && A[0]->ty != LType::DOUBLE) ? LType::BIGINT : LType::DOUBLE;
    if (id == FN_ROUND && A[0]->ty == LType::DOUBLE) ty = LType::DOUBLE;
    ExprP r = mkCall(id, {A[0]}, ty);
    if (id == FN_ABS && A[0]->ty == LType::DECIMAL) r->ty = LType::DECIMAL, r->scale = A[0]->scale;
    finish(r);
    return;
  }
  static const std::map<std::string, uint16_t> math2 = {{"ROUND", FN_ROUND2}, {"TRUNCATE", FN_TRUNCATE2}, {"POW", FN_POW},
                                                         {"POWER", FN_POW}, {"ATAN2", FN_ATAN2}, {"ATAN", FN_ATAN2},
                                                         {"LOG", FN_LOG_B}};
  if (math2.count(f)) {
    argc(2, 2);
    finish(mkCall(math2.at(f), {A[0], A[1]}, LType::DOUBLE));
    return;
  }
  if (f == "MOD") { argc(2, 2); ExprP r = boundNode(Expr::BIN, "%", {A[0], A[1]}, numTy(A)); r->scale = maxScale(A); finish(r); return; }
  if (f == "GREATEST" || f == "LEAST") {
    argc(2, 64);
    bool str = false;
    for (auto& x : A) str = str || x->ty == LType::VARCHAR;
    if (str) {  // strings: compared as strings, on the host
      e->ty = LType::VARCHAR;
      e->bound = true;
      if (allLits(e)) e = boundLit(evalConst(e));
      else encodeStrings(e, c);
      return;
    }
    LType t = numTy(A);
    ExprP r = A[0];
    for (size_t i = 1; i < n; ++i) {
      r = mkCall(f == "LEAST" ? FN_LEAST : FN_GREATEST, {r, A[i]}, t);
      if (t == LType::DECIMAL) r->scale = maxScale(A);
    }
    finish(r);
    return;
  }
  // Date parts. DATETIME arguments go through DATE() first where the function is day-based.
  static const std::map<std::string, uint16_t> dpart = {
      {"MONTH", FN_MONTH}, {"DAY", FN_DAY}, {"DAYOFMONTH", FN_DAY}, {"DAYOFWEEK", FN_DAYOFWEEK},
      {"WEEKDAY", FN_WEEKDAY}, {"DAYOFYEAR", FN_DAYOFYEAR}, {"QUARTER", FN_QUARTER}, {"WEEK", FN_WEEK},
      {"YEARWEEK", FN_YEARWEEK}, {"LAST_DAY", FN_LAST_DAY}, {"TO_DAYS", FN_TO_DAYS}};
  if (f == "YEAR") {
    argc(1, 1);
    A[0] = asDate(A[0]);
    e->ty = LType::INT;
    e->bound = true;
    if (allLits(e)) e = boundLit(evalConst(e));
    return;
  }
  if (dpart.count(f)) {
    argc(1, 2);
    uint16_t id = dpart.at(f);
    finish(mkCall(id, {asDate(A[0])}, id == FN_LAST_DAY ? LType::DATE : LType::INT));
    return;
  }
  if (f == "HOUR" || f == "MINUTE" || f == "SECOND" || f == "MICROSECOND") {
    argc(1, 1);
    uint16_t id = f == "HOUR" ? FN_DT_HOUR : f == "MINUTE" ? FN_DT_MINUTE : f == "SECOND" ? FN_DT_SECOND : FN_DT_MICRO;
    finish(mkCall(id, {asDateTime(A[0])}, LType::INT));
    return;
  }
  if (f == "DATE_TRUNC") {  // DATE_TRUNC('unit', x): SingleStore / PostgreSQL
    argc(2, 2);
    if (A[0]->kind != Expr::LIT || A[0]->lit.kind != Value::STR) throw SqlError("DATE_TRUNC needs a constant unit", 1235);
    std::string u = upper(A[0]->lit.s);
    const bool dt = A[1]->ty == LType::DATETIME;
    static const std::map<std::string, int64_t> us = {{"MICROSECOND", 1}, {"MILLISECOND", 1000}, {"SECOND", 1000000ll},
                                                       {"MINUTE", 60000000ll}, {"HOUR", 3600000000ll}};
    if (us.count(u)) {
      ExprP x = asDateTime(A[1]);
      ExprP q = mkCall(FN_IDIV, {x, boundLit(Value::I(us.at(u)))}, LType::BIGINT);
      finish(boundNode(Expr::BIN, "*", {q, boundLit(Value::I(us.at(u)))}, LType::DATETIME));
      return;
    }
    ExprP d = asDate(A[1]);
    ExprP r;
    if (u == "DAY") r = d;
    else if (u == "WEEK") {  // Monday-based weeks (day 0 = 1970-01-01, a Thursday)
      ExprP shifted = boundNode(Expr::BIN, "+", {d, boundLit(Value::I(3))}, LType::BIGINT);
      ExprP wk = mkCall(FN_IDIV, {shifted, boundLit(Value::I(7))}, LType::BIGINT);
      r = boundNode(Expr::BIN, "-", {boundNode(Expr::BIN, "*", {wk, boundLit(Value::I(7))}, LType::BIGINT), boundLit(Value::I(3))}, LType::DATE);
    } else if (u == "MONTH" || u == "YEAR") {
      r = mkCall(FN_DATE_TRUNC_MONTH, {d, boundLit(Value::I(u == "YEAR"))}, LType::DATE);
    } else {
      throw SqlError("DATE_TRUNC unit '" + A[0]->lit.s + "' is not supported", 1235);
    }
    r->ty = LType::DATE;
    finish(dt ? asDateTime(r) : r);
    return;
  }
  if (f == "DATE") { argc(1, 1); finish(asDate(A[0])); return; }
  if (f == "TIMESTAMP" || f == "DATETIME") { argc(1, 1); finish(asDateTime(A[0])); return; }
  if (f == "FROM_DAYS") { argc(1, 1); finish(mkCall(FN_FROM_DAYS, {A[0]}, LType::DATE)); return; }
  if (f == "UNIX_TIMESTAMP") { argc(1, 1); finish(mkCall(FN_DT_UNIX, {asDateTime(A[0])}, LType::BIGINT)); return; }
  if (f == "FROM_UNIXTIME") { argc(1, 1); finish(mkCall(FN_UNIX_DT, {A[0]}, LType::DATETIME)); return; }
  if (f == "DATEDIFF") {
    argc(2, 2);
    finish(boundNode(Expr::BIN, "-", {asDate(A[0]), asDate(A[1])}, LType::BIGINT));
    return;
  }
  if (f == "DATE_ADD" || f == "ADDDATE" || f == "DATE_SUB" || f == "SUBDATE") {
    argc(2, 2);
    bool sub = f == "DATE_SUB" || f == "SUBDATE";
    if (A[1]->kind == Expr::INTERVAL) e = intervalArith(A[0], sub ? -A[1]->lit.i : A[1]->lit.i, A[1]->op);
    else e = intervalArith(A[0], 0, "DAY"), e = boundNode(Expr::BIN, sub ? "-" : "+", {e, A[1]}, e->ty);  // ADDDATE(d, n days)
    return;
  }
  if (f == "TIMESTAMPDIFF") {
    argc(2, 2);
    const std::string u = e->castType;
    static const std::map<std::string, int64_t> us = {{"MICROSECOND", 1}, {"SECOND", 1000000ll}, {"MINUTE", 60000000ll},
                                                       {"HOUR", 3600000000ll}, {"DAY", kUsPerDay}, {"WEEK", 7 * kUsPerDay}};
    if (us.count(u)) {
      ExprP d = boundNode(Expr::BIN, "-", {asDateTime(A[1]), asDateTime(A[0])}, LType::BIGINT);
      finish(mkCall(FN_IDIV, {d, boundLit(Value::I(us.at(u)))}, LType::BIGINT));
    } else {  // MONTH / QUARTER / YEAR: calendar months between the dates
      auto months = [&](const ExprP& x) {
        ExprP d = asDate(x);
        ExprP y = boundNode(Expr::FUNC, "YEAR", {d}, LType::INT);
        return boundNode(Expr::BIN, "+", {boundNode(Expr::BIN, "*", {y, boundLit(Value::I(12))}, LType::BIGINT),
                                          mkCall(FN_MONTH, {d}, LType::INT)}, LType::BIGINT);
      };
      ExprP m = boundNode(Expr::BIN, "-", {months(A[1]), months(A[0])}, LType::BIGINT);
      int64_t div = u == "YEAR" ? 12 : u == "QUARTER" ? 3 : 1;
      finish(div == 1 ? m : mkCall(FN_IDIV, {m, boundLit(Value::I(div))}, LType::BIGINT));
    }
    return;
  }
  if (f == "DIV") {  // a DIV b: integer division (truncating)
    argc(2, 2);
    finish(mkCall(FN_IDIV, {A[0], A[1]}, LType::BIGINT));
    return;
  }
  if (f == "TIME_BUCKET") {  // TIME_BUCKET('5m', ts [, origin]): start of ts's bucket
    argc(2, 3);
    if (A[0]->kind != Expr::LIT || A[0]->lit.kind != Value::STR) throw SqlError("TIME_BUCKET needs a constant width such as '5m'", 1235);
    const std::string w = A[0]->lit.s;
    char* endp = nullptr;
    const double num = std::strtod(w.c_str(), &endp);
    std::string unit = lower(std::string(endp));
    while (!unit.empty() && unit[0] == ' ') unit.erase(0, 1);
    const int64_t us = unit == "us" ? 1 : unit == "ms" ? 1000 : unit == "s" ? 1000000 : unit == "m" ? 60000000ll
                     : unit == "h" ? 3600000000ll : unit == "d" ? 86400000000ll : unit == "w" ? 604800000000ll : 0;
    if (!us || num <= 0) throw SqlError("TIME_BUCKET width must look like '30s', '5m', '1h' or '1d'", 1235);
    const int64_t width = (int64_t)std::llround(num * (double)us);
    int64_t origin = 0;
    if (n == 3) {
      if (A[2]->kind != Expr::LIT) throw SqlError("TIME_BUCKET needs a constant origin", 1235);
      if (A[2]->lit.kind == Value::STR) {
        if (!parseDateTime(A[2]->lit.s, origin)) throw SqlError("bad TIME_BUCKET origin", 1292);
      } else {
        origin = A[2]->lit.kind == Value::DATE ? A[2]->lit.i * kUsPerDay : A[2]->lit.i;
      }
    }
    ExprP ts = asDateTime(A[1]);
    ExprP diff = boundNode(Expr::BIN, "-", {ts, boundLit(Value::I(origin))}, LType::BIGINT);
    ExprP q = mkCall(FN_IDIV, {diff, boundLit(Value::I(width))}, LType::BIGINT);
    ExprP start = boundNode(Expr::BIN, "+", {boundNode(Expr::BIN, "*", {q, boundLit(Value::I(width))}, LType::BIGINT), boundLit(Value::I(origin))},
                            LType::DATETIME);
    finish(start);
    return;
  }
  if (f == "TIMESTAMPADD") {
    argc(2, 2);
    if (A[0]->kind != Expr::LIT) throw SqlError("TIMESTAMPADD needs a constant amount", 1235);
    e = intervalArith(A[1], (int64_t)A[0]->lit.num(), e->castType);
    return;
  }
  if (f == "CAST") {
    const std::string t = e->castType;
    if (t == "SIGNED" || t == "UNSIGNED" || t == "INTEGER" || t == "INT" || t == "BIGINT") {
      if (A[0]->ty == LType::VARCHAR) { finish(strToNum(A[0], c, true)); return; }
      finish(A[0]->ty == LType::DOUBLE || A[0]->ty == LType::DECIMAL ? mkCall(FN_ROUND, {A[0]}, LType::BIGINT) : A[0]);
      if (e->ty != LType::DOUBLE && e->kind != Expr::LIT) e->ty = LType::BIGINT;
      return;
    }
    if (t == "DECIMAL" || t == "DOUBLE" || t == "FLOAT" || t == "REAL" || t == "NUMERIC") {
      if (A[0]->ty == LType::VARCHAR) { finish(strToNum(A[0], c)); return; }
      ExprP r = boundNode(Expr::BIN, "*", {A[0], boundLit(Value::D(1.0))}, LType::DOUBLE);
      finish(r);
      return;
    }
    if (t == "DATE") { finish(asDate(A[0])); return; }
    if (t == "DATETIME" || t == "TIMESTAMP") { finish(asDateTime(A[0])); return; }
    if (t == "CHAR" || t == "VARCHAR" || t == "TEXT" || t == "BINARY" || t == "NCHAR" || t == "JSON") {
      if (A[0]->ty == LType::VARCHAR) { e = A[0]; return; }
      e->op = "CASTCHAR";
      e->ty = LType::VARCHAR;
      e->bound = true;
      if (allLits(e)) e = boundLit(evalConst(e));
      else encodeStrings(e, c);
      return;
    }
    throw SqlError("CAST to " + t + " is not supported here", 1235);
  }
  // Vector similarity (SingleStore): one VECTOR column against a constant vector.
  if (f == "JSON_ARRAY_PACK" || f == "JSON_ARRAY_UNPACK" || f == "VECTOR" || f == "TO_VECTOR") {
    argc(1, 1);
    e = A[0];
    return;
  }
  if (f == "DOT_PRODUCT" || f == "EUCLIDEAN_DISTANCE" || f == "VECTOR_DOT_PRODUCT") {
    argc(2, 2);
    ExprP col = A[0]->ty == LType::VECTOR ? A[0] : A[1], k = A[0]->ty == LType::VECTOR ? A[1] : A[0];
    auto parseConst = [&](const ExprP& x, std::vector<float>& v) {
      return x->kind == Expr::LIT && x->lit.kind == Value::STR && parseVector(x->lit.s, v);
    };
    std::vector<float> qa, qb;
    if (parseConst(A[0], qa) && parseConst(A[1], qb)) {  // both constant
      if (qa.size() != qb.size()) throw SqlError("vector dimensions differ", 1105);
      double acc = 0;
      for (size_t i = 0; i < qa.size(); ++i) acc += f == "EUCLIDEAN_DISTANCE" ? (qa[i] - qb[i]) * (qa[i] - qb[i]) : qa[i] * qb[i];
      e = boundLit(Value::D(f == "EUCLIDEAN_DISTANCE" ? std::sqrt(acc) : acc));
      return;
    }
    if (col->kind != Expr::COL || col->ty != LType::VECTOR || !parseConst(k, qa))
      throw SqlError(f + "() needs a VECTOR column and a constant vector such as '[0.1, 0.2]'", 1235);
    const int dim = c.tabs[col->bt]->cols[col->bc].vecDim;
    if ((int)qa.size() != dim)
      throw SqlError("vector has " + std::to_string(qa.size()) + " elements but the column is VECTOR(" + std::to_string(dim) + ")", 1105);
    ExprP r = boundNode(Expr::FUNC, f == "EUCLIDEAN_DISTANCE" ? "#VEUC" : "#VDOT", {col}, LType::DOUBLE);
    r->aux = std::make_shared<std::vector<double>>(qa.begin(), qa.end());
    e = r;
    return;
  }
  if (f == "VECTOR_NUM_ELEMENTS") {
    argc(1, 1);
    if (A[0]->kind != Expr::COL || A[0]->ty != LType::VECTOR) throw SqlError("VECTOR_NUM_ELEMENTS needs a VECTOR column", 1235);
    e = boundLit(Value::I(c.tabs[A[0]->bt]->cols[A[0]->bc].vecDim));
    return;
  }
  // String / JSON functions: host evaluated, encoded over the dictionary when possible.
  static const std::map<std::string, LType> strFns = {
      {"CONCAT", LType::VARCHAR}, {"CONCAT_WS", LType::VARCHAR}, {"UPPER", LType::VARCHAR}, {"UCASE", LType::VARCHAR},
      {"LOWER", LType::VARCHAR}, {"LCASE", LType::VARCHAR}, {"SUBSTRING", LType::VARCHAR}, {"SUBSTR", LType::VARCHAR},
      {"MID", LType::VARCHAR}, {"LEFT", LType::VARCHAR}, {"RIGHT", LType::VARCHAR}, {"TRIM", LType::VARCHAR},
      {"LTRIM", LType::VARCHAR}, {"RTRIM", LType::VARCHAR}, {"REPLACE", LType::VARCHAR}, {"REVERSE", LType::VARCHAR},
      {"LPAD", LType::VARCHAR}, {"RPAD", LType::VARCHAR}, {"REPEAT", LType::VARCHAR}, {"SPACE", LType::VARCHAR},
      {"SUBSTRING_INDEX", LType::VARCHAR}, {"FORMAT", LType::VARCHAR}, {"DATE_FORMAT", LType::VARCHAR},
      {"TIME_FORMAT", LType::VARCHAR}, {"DAYNAME", LType::VARCHAR}, {"MONTHNAME", LType::VARCHAR}, {"HEX", LType::VARCHAR},
      {"JSON_EXTRACT_STRING", LType::VARCHAR}, {"JSON_EXTRACT_JSON", LType::VARCHAR}, {"JSON_EXTRACT", LType::VARCHAR},
      {"EMBED", LType::VARCHAR}, {"EMBED_TOKENS", LType::VARCHAR}, {"JSON_UNQUOTE", LType::VARCHAR}, {"JSON_ARRAY", LType::VARCHAR}, {"JSON_OBJECT", LType::VARCHAR}, {"JSON_QUOTE", LType::VARCHAR},
      {"UUID", LType::VARCHAR}, {"ELT", LType::VARCHAR}, {"REGEXP_REPLACE", LType::VARCHAR},
      {"LENGTH", LType::BIGINT}, {"OCTET_LENGTH", LType::BIGINT}, {"CHAR_LENGTH", LType::BIGINT},
      {"CHARACTER_LENGTH", LType::BIGINT}, {"LOCATE", LType::BIGINT}, {"POSITION", LType::BIGINT}, {"INSTR", LType::BIGINT},
      {"ASCII", LType::BIGINT}, {"STRCMP", LType::BIGINT}, {"FIELD", LType::BIGINT}, {"JSON_EXTRACT_BIGINT", LType::BIGINT},
      {"JSON_LENGTH", LType::BIGINT}, {"REGEXP_LIKE", LType::INT}, {"RLIKE", LType::INT}, {"REGEXP", LType::INT},
      {"JSON_EXTRACT_DOUBLE", LType::DOUBLE}, {"MATCH_AGAINST", LType::DOUBLE},
      {"REGEXP_SUBSTR", LType::VARCHAR}, {"REGEXP_INSTR", LType::BIGINT},
      {"SHA2", LType::VARCHAR}, {"MD5", LType::VARCHAR}, {"SHA1", LType::VARCHAR}, {"SHA", LType::VARCHAR}, {"TO_BASE64", LType::VARCHAR}, {"FROM_BASE64", LType::VARCHAR}, {"CRC32", LType::BIGINT}, {"BIT_COUNT", LType::BIGINT}, {"RAND", LType::DOUBLE},
      {"CONVERT_TZ", LType::DATETIME}, {"GEOGRAPHY_POINT", LType::VARCHAR}, {"GEOGRAPHY_DISTANCE", LType::DOUBLE},
      {"GEOGRAPHY_WITHIN_DISTANCE", LType::INT}, {"GEOGRAPHY_CONTAINS", LType::INT}, {"GEOGRAPHY_INTERSECTS", LType::INT},
      {"GEOGRAPHY_LONGITUDE", LType::DOUBLE}, {"GEOGRAPHY_LATITUDE", LType::DOUBLE},
      {"FIND_IN_SET", LType::BIGINT}, {"UNHEX", LType::VARCHAR}, {"INSERT", LType::VARCHAR},
      {"BITAND", LType::BIGINT}, {"BITOR", LType::BIGINT}, {"BITXOR", LType::BIGINT}, {"SHL", LType::BIGINT}, {"SHR", LType::BIGINT},
      {"BITNOT", LType::BIGINT}, {"TIMEDIFF", LType::VARCHAR}, {"SEC_TO_TIME", LType::VARCHAR}, {"TIME", LType::VARCHAR},
      {"TIME_TO_SEC", LType::BIGINT}, {"MAKETIME", LType::VARCHAR}, {"MAKEDATE", LType::DATE}, {"CHAR", LType::VARCHAR},
      {"BIT_LENGTH", LType::BIGINT}, {"INET_ATON", LType::BIGINT}, {"INET_NTOA", LType::VARCHAR}, {"IS_IPV4", LType::INT},
      {"JSON_SET", LType::VARCHAR}, {"JSON_INSERT", LType::VARCHAR}, {"JSON_REPLACE", LType::VARCHAR}, {"JSON_REMOVE", LType::VARCHAR},
      {"ORD", LType::BIGINT}, {"BIN", LType::VARCHAR}, {"OCT", LType::VARCHAR}, {"CONV", LType::VARCHAR}, {"JSON_TYPE", LType::VARCHAR}, {"JSON_CONTAINS_PATH", LType::INT}, {"JSON_KEYS", LType::VARCHAR}, {"JSON_VALID", LType::INT}};
  if (f == "#UDF") {  // stored function with a procedural body over columns (proc.cpp inlineUdf)
    const std::string t = A[1]->lit.s.substr(0, A[1]->lit.s.find('#'));
    e->ty = t == "BIGINT" ? LType::BIGINT : t == "DOUBLE" || t == "DECIMAL" ? LType::DOUBLE : t == "DATE" ? LType::DATE
          : t == "DATETIME" ? LType::DATETIME : LType::VARCHAR;
    e->bound = true;
    encodeStrings(e, c);
    return;
  }
  if (f == "STR_TO_DATE") {  // DATETIME when the format has time fields, else DATE
    argc(2, 2);
    bool timeFields = true;
    if (A[1]->kind == Expr::LIT && A[1]->lit.kind == Value::STR) {
      timeFields = false;
      const std::string& fm = A[1]->lit.s;
      for (size_t i = 0; i + 1 < fm.size(); ++i)
        if (fm[i] == '%' && strchr("HkhIlisSfpTr", fm[i + 1])) timeFields = true;
    }
    e->ty = timeFields ? LType::DATETIME : LType::DATE;
    e->bound = true;
    if (allLits(e)) e = boundLit(evalConst(e));
    else encodeStrings(e, c);
    return;
  }
  if (strFns.count(f)) {
    e->ty = strFns.at(f);
    e->bound = true;
    if (f == "JSON_ARRAY" || f == "JSON_OBJECT" || f == "JSON_SET" || f == "JSON_INSERT" || f == "JSON_REPLACE") {  // which arguments already are JSON (embedded, not quoted)
      e->name.clear();
      for (auto& x : A) e->name += isJsonExpr(x, &c) ? '1' : '0';
    }
    const bool volatileFn = f == "UUID" || f == "RAND";
    if (!volatileFn && allLits(e)) {
      const bool js = isJsonExpr(e, &c);
      e = boundLit(evalConst(e));
      if (js) e->castType = "JSON";
      return;
    }
    if (!volatileFn) encodeStrings(e, c);
    return;
  }
  throw SqlError("FUNCTION warpstore." + f + " does not exist", 1305);
}

// A string expression as a number (MySQL's implicit / CAST conversion): evaluated once per
// dictionary entry; literals are converted right away.
inline ExprP strToNum(const ExprP& x, BindCtx& c, bool integer) {
  if (x->kind == Expr::LIT && x->lit.kind == Value::STR)
    return boundLit(integer ? Value::I(std::strtoll(x->lit.s.c_str(), nullptr, 10)) : Value::D(std::strtod(x->lit.s.c_str(), nullptr)));
  ExprP n = boundNode(Expr::FUNC, integer ? "CASTINT" : "CASTNUM", {x}, integer ? LType::BIGINT : LType::DOUBLE);
  encodeStrings(n, c);
  return n;
}

inline void bind(ExprP& e, BindCtx& c) {
  if (e->bound) return;
  switch (e->kind) {
    case Expr::COL: {
      int found = 0;
      for (size_t t = 0; t < c.tabs.size(); ++t) {
        if (!e->table.empty() && strcasecmp(e->table.c_str(), c.aliases[t].c_str()) != 0 &&
            strcasecmp(e->table.c_str(), c.tabs[t]->name.c_str()) != 0)
          continue;
        int ci = c.tabs[t]->colIndex(e->name);
        if (ci < 0) continue;
        if (found && e->table.empty() && t < c.usingCols.size() &&
            std::any_of(c.usingCols[t].begin(), c.usingCols[t].end(), [&](const std::string& u) { return strcasecmp(u.c_str(), e->name.c_str()) == 0; }))
          continue;  // USING column: the earlier table's copy (equal by the join condition)
        if (found++) throw SqlError("Column '" + e->name + "' in field list is ambiguous", 1052);
        e->bt = (int)t;
        e->bc = ci;
        e->ty = c.tabs[t]->cols[ci].lt;
        e->scale = e->ty == LType::DECIMAL ? c.tabs[t]->cols[ci].scale : -1;
      }
      if (!found) throw SqlError("Unknown column '" + e->str() + "' in 'field list'", 1054);
      break;
    }
    case Expr::LIT:
      if (e->lit.kind == Value::DBL && e->lit.scale >= 0) {  // exact decimal literal
        e->ty = LType::DECIMAL;
        e->scale = e->lit.scale;
        break;
      }
      e->ty = e->lit.kind == Value::INT ? LType::BIGINT : e->lit.kind == Value::DBL ? LType::DOUBLE
            : e->lit.kind == Value::DATE ? LType::DATE : e->lit.kind == Value::DATETIME ? LType::DATETIME : LType::VARCHAR;
      break;
    case Expr::INTERVAL: e->ty = LType::BIGINT; break;
    case Expr::STAR: e->ty = LType::BIGINT; break;
    case Expr::BIN: {
      bindChildren(e, c);
      ExprP& a = e->args[0];
      ExprP& b = e->args[1];
      if ((e->op == "+" || e->op == "-") && (a->kind == Expr::INTERVAL || b->kind == Expr::INTERVAL)) {
        ExprP iv = a->kind == Expr::INTERVAL ? a : b;
        ExprP dt = a->kind == Expr::INTERVAL ? b : a;
        e = intervalArith(dt, e->op == "-" ? -iv->lit.i : iv->lit.i, iv->op);
        return;
      }
      if (isCmp(e->op)) {
        // Coerce string literals compared with DATE / DATETIME values, and DATE vs DATETIME.
        for (int i = 0; i < 2; ++i) {
          ExprP& x = e->args[i];
          ExprP& y = e->args[1 - i];
          if ((y->ty == LType::DATE || y->ty == LType::DATETIME) && x->kind == Expr::LIT && x->lit.kind == Value::STR)
            x = coerceTemporal(x, y->ty);
        }
        if ((a->ty == LType::DATE && b->ty == LType::DATETIME) || (a->ty == LType::DATETIME && b->ty == LType::DATE)) {
          a = asDateTime(a);
          b = asDateTime(b);
        }
        // string vs number compares as numbers (MySQL)
        auto numeric = [](LType t) { return t == LType::INT || t == LType::BIGINT || t == LType::DOUBLE || t == LType::DECIMAL; };
        for (int i = 0; i < 2; ++i) {
          ExprP& x = e->args[i];
          const ExprP& y = e->args[1 - i];
          if (x->ty == LType::VARCHAR && !isNullLit(x) && numeric(y->ty) && !(y->kind == Expr::LIT && y->lit.kind == Value::STR)) x = strToNum(x, c);
        }
        // DECIMAL arithmetic runs in doubles: (b - m) + (m + 11) can come out as 14.999999999 where
        // MySQL's exact arithmetic gives 15. Compared, it is rounded to its scale first.
        for (int i = 0; i < 2; ++i) {
          ExprP& x = e->args[i];
          if (x->ty == LType::DECIMAL && x->kind == Expr::BIN && (x->op == "+" || x->op == "-" || x->op == "*") && x->scale >= 0 && x->scale <= 9) {
            const double p = pow10i(x->scale);
            ExprP r = mkCall(FN_ROUND, {boundNode(Expr::BIN, "*", {x, boundLit(Value::D(p))}, LType::DOUBLE)}, LType::DOUBLE);
            const int sc = x->scale;
            x = boundNode(Expr::BIN, "/", {r, boundLit(Value::D(p))}, LType::DECIMAL);
            x->scale = sc;
          }
        }
        e->ty = LType::INT;
      } else if (e->op == "AND" || e->op == "OR") {
        e->ty = LType::INT;
      } else {
        for (int i = 0; i < 2; ++i)  // MySQL converts numeric-looking string literals in arithmetic
          if (e->args[i]->kind == Expr::LIT && e->args[i]->lit.kind == Value::STR)
            e->args[i] = boundLit(Value::D(std::atof(e->args[i]->lit.s.c_str())));
        for (int i = 0; i < 2; ++i)  // string columns / expressions in arithmetic: as numbers (MySQL)
          if (e->args[i]->ty == LType::VARCHAR && !isNullLit(e->args[i])) e->args[i] = strToNum(e->args[i], c);
        auto numTy = [](LType t) { return t == LType::INT || t == LType::BIGINT || t == LType::DECIMAL; };
        if (a->ty == LType::DOUBLE || b->ty == LType::DOUBLE) e->ty = LType::DOUBLE;
        else if (e->op == "/") {  // exact numbers divide to DECIMAL with 4 more digits (div_precision_increment)
          e->ty = numTy(a->ty) && numTy(b->ty) ? LType::DECIMAL : LType::DOUBLE;
          if (e->ty == LType::DECIMAL) e->scale = std::min(30, std::max(0, (int)a->scale) + 4);
        } else if (a->ty == LType::DECIMAL || b->ty == LType::DECIMAL) {
          e->ty = LType::DECIMAL;
          const int sa = std::max(0, (int)a->scale), sb = std::max(0, (int)b->scale);
          e->scale = e->op == "*" ? std::min(9, sa + sb) : std::max(sa, sb);
        }
        else if ((a->ty == LType::DATETIME) != (b->ty == LType::DATETIME) && (e->op == "+" || e->op == "-")) e->ty = LType::DATETIME;
        else if ((a->ty == LType::DATE) != (b->ty == LType::DATE) && (e->op == "+" || e->op == "-")) e->ty = LType::DATE;
        else e->ty = LType::BIGINT;
      }
      break;
    }
    case Expr::UNARY:
      bindChildren(e, c);
      e->ty = e->op == "NOT" ? LType::INT : e->args[0]->ty;
      if (e->ty == LType::DECIMAL) e->scale = e->args[0]->scale;
      break;
    case Expr::BETWEEN: {
      ExprP r = mkBin("AND", mkBin(">=", e->args[0], e->args[1]), mkBin("<=", e->args[0], e->args[2]));
      if (e->neg) {
        auto n = std::make_shared<Expr>();
        n->kind = Expr::UNARY;
        n->op = "NOT";
        n->args = {r};
        r = n;
      }
      e = r;
      bind(e, c);
      return;
    }
    case Expr::INLIST: {
      bindChildren(e, c);
      if (e->args[0]->ty == LType::VARCHAR) {
        for (size_t i = 1; i < e->args.size(); ++i)
          if (e->args[i]->kind != Expr::LIT || e->args[i]->lit.kind != Value::STR)
            throw SqlError("IN list on strings must contain string literals", 1235);
        e->ty = LType::INT;
        break;
      }
      // Long literal lists: a sorted set probed by binary search on the GPU (an OR chain of
      // thousands of equalities would not fit the bytecode).
      bool allNum = e->args.size() > 9;
      for (size_t i = 1; i < e->args.size() && allNum; ++i) {
        const ExprP& a = e->args[i];
        allNum = a->kind == Expr::LIT && !a->lit.isNull() && (a->lit.kind == Value::INT || a->lit.kind == Value::DBL || a->lit.kind == Value::DATE) &&
                 std::fabs(a->lit.num()) < 9007199254740992.0;
      }
      if (allNum) {
        auto vals = std::make_shared<std::vector<double>>();
        for (size_t i = 1; i < e->args.size(); ++i) vals->push_back(e->args[i]->lit.num());
        std::sort(vals->begin(), vals->end());
        vals->erase(std::unique(vals->begin(), vals->end()), vals->end());
        auto f = std::make_shared<Expr>();
        f->kind = Expr::FUNC;
        f->op = "#INSORTED";
        f->args = {e->args[0]};
        f->aux = vals;
        ExprP r = f;
        if (e->neg) {
          auto n = std::make_shared<Expr>();
          n->kind = Expr::UNARY;
          n->op = "NOT";
          n->args = {f};
          r = n;
        }
        e = r;
        bind(e, c);
        return;
      }
      ExprP r;
      for (size_t i = 1; i < e->args.size(); ++i) {
        ExprP eq = mkBin("=", e->args[0], e->args[i]);
        r = r ? mkBin("OR", r, eq) : eq;
      }
      if (e->neg) {
        auto n = std::make_shared<Expr>();
        n->kind = Expr::UNARY;
        n->op = "NOT";
        n->args = {r};
        r = n;
      }
      e = r;
      bind(e, c);
      return;
    }
    case Expr::LIKE:
      bindChildren(e, c);
      if (isNullLit(e->args[1]) || isNullLit(e->args[0])) {  // x LIKE NULL is NULL
        e = boundLit(Value::Null());
        e->ty = LType::INT;
        return;
      }
      if (e->args[1]->kind != Expr::LIT || e->args[1]->lit.kind != Value::STR)
        throw SqlError("LIKE pattern must be a string literal", 1235);
      e->ty = LType::INT;
      break;
    case Expr::CASE: {
      bindChildren(e, c);
      e->ty = LType::BIGINT;
      int sc = 0;
      auto result = [&](const ExprP& x) {
        if (isNullLit(x)) return;
        LType t = x->ty;
        if (t == LType::DOUBLE) e->ty = LType::DOUBLE;
        else if (t == LType::DECIMAL) {
          sc = std::max(sc, (int)x->scale);
          if (e->ty == LType::BIGINT || e->ty == LType::INT) e->ty = LType::DECIMAL;
        } else if (t == LType::VARCHAR || t == LType::DATE || t == LType::DATETIME) e->ty = t;
      };
      for (size_t i = 1; i < e->args.size(); i += (i + 1 < e->args.size() ? 2 : 1)) result(e->args[i]);
      if (e->args.size() % 2) result(e->args.back());
      if (e->ty == LType::DECIMAL) e->scale = sc;
      if (e->ty == LType::VARCHAR) {
        e->bound = true;
        unifyStrings(e, c);
        return;
      }
      break;
    }
    case Expr::FUNC: {
      if (isAggName(e->op)) {
        if (c.inAgg) throw SqlError("Invalid use of group function", 1111);
        if (e->distinct && e->op != "COUNT") throw SqlError("internal: DISTINCT aggregate reached the binder", 1105);
        if (e->args.size() != 1) throw SqlError("Incorrect parameter count in the call to " + e->op, 1582);
        c.inAgg = true;
        if (e->args[0]->kind != Expr::STAR) bind(e->args[0], c);
        c.inAgg = false;
        LType at = e->args[0]->ty;
        if (e->op == "COUNT") e->ty = LType::BIGINT;
        else if (e->op == "AVG")  // of exact numbers: DECIMAL with 4 more digits (MySQL)
          e->ty = at == LType::INT || at == LType::BIGINT || at == LType::DECIMAL ? LType::DECIMAL : LType::DOUBLE;
        else if (e->op == "SUM") e->ty = at == LType::DOUBLE ? LType::DOUBLE : at == LType::DECIMAL ? LType::DECIMAL : LType::BIGINT;
        else e->ty = at;
        if (e->ty == LType::DECIMAL) e->scale = e->op == "AVG" ? std::min(30, std::max(0, (int)e->args[0]->scale) + 4) : e->args[0]->scale;
        if (e->op != "COUNT" && e->args[0]->kind == Expr::STAR) throw SqlError("Invalid use of '*'", 1064);
        if ((e->op == "MIN" || e->op == "MAX") && at == LType::VARCHAR) {
          // Aggregate each string's rank in sorted order; the rank decodes through a sorted dictionary.
          Dict* d = c.dict(*e->args[0]);
          if (!d) throw SqlError(e->op + "() on this string expression is not supported", 1235);
          auto sd = sortedDict(*d);  // cached per dictionary version
          ExprP m = boundNode(Expr::FUNC, "#MAP", {e->args[0]}, LType::BIGINT);
          m->aux = sd.second;
          e->args[0] = m;
          e->ty = LType::VARCHAR;
          c.exprDict[e.get()] = d;
          c.unrank[e.get()] = sd.first;
          break;
        }
        if ((e->op == "SUM" || e->op == "AVG") && at == LType::VARCHAR)
          throw SqlError(e->op + "() on strings is not supported", 1235);
        break;
      }
      bindFunc(e, c);
      return;
    }
  }
  e->bound = true;
}

// Can the expression evaluate to NULL? (Conservative: columns that ever held a NULL, NULL literals,
// division, and functions with domain errors.)
inline bool nullable(const ExprP& e, const BindCtx& c) {
  switch (e->kind) {
    case Expr::COL: return e->bt >= 0 && (c.tabs[e->bt]->hasNull[e->bc] || c.outerTabs.count(e->bt));
    case Expr::LIT: return e->lit.isNull();
    case Expr::STAR: return false;
    default: break;
  }
  if (e->kind == Expr::FUNC) {
    if (e->op == "ISNULL" || e->op == "ISNOTNULL") return false;
    if (e->op == "COALESCE2") return nullable(e->args[0], c) && nullable(e->args[1], c);
    if (e->op == "NULLIF" || e->fnId >= 0 || e->op.rfind("#ENC", 0) == 0) return true;
  }
  if (e->kind == Expr::BIN && (e->op == "/" || e->op == "%")) return true;
  if (e->kind == Expr::CASE && e->args.size() % 2 == 0) return true;  // no ELSE
  for (auto& a : e->args)
    if (nullable(a, c)) return true;
  return false;
}

inline void tablesOf(const ExprP& e, std::set<int>& out) {
  if (e->kind == Expr::COL) out.insert(e->bt);
  for (auto& a : e->args) tablesOf(a, out);
}
inline void colsOf(const ExprP& e, std::set<std::pair<int, int>>& out) {
  if (!e) return;
  if (e->kind == Expr::COL) out.insert({e->bt, e->bc});
  for (auto& a : e->args) colsOf(a, out);
}
inline void flattenAnd(const ExprP& e, std::vector<ExprP>& out) {
  if (!e) return;
  if (e->kind == Expr::BIN && e->op == "AND") {
    flattenAnd(e->args[0], out);
    flattenAnd(e->args[1], out);
  } else {
    out.push_back(e);
  }
}
inline ExprP andAll(const std::vector<ExprP>& v) {
  ExprP r;
  for (auto& x : v) {
    if (!r) { r = x; continue; }
    r = mkBin("AND", r, x);
    r->ty = LType::INT;
    r->bound = true;
  }
  return r;
}

// ================================================================== bytecode compiler

// Per-thread device flag for overflowing compiled int64 arithmetic (ExprEnv::ovf).
inline int* ovfFlagDev() {
  thread_local int* p = nullptr;
  if (!p) {
    cudaMalloc(&p, sizeof(int));
    cudaMemset(p, 0, sizeof(int));
  }
  return p;
}

struct Compiler {
  BindCtx* ctx;
  ExprEnv env;
  int ncode = 0, nconst = 0, depth = 0;
  std::map<std::pair<int, int>, int> slotOf;
  std::vector<std::pair<int, int>> slotCols;
  std::vector<std::vector<uint32_t>> setsHost;
  std::vector<DevVec<uint32_t>> setsDev;
  // Optional override: emits code for `e` and returns true (used to compile post-aggregation
  // expressions over grouped results, where group keys and aggregates are the columns).
  std::function<bool(const ExprP&)> hook;

  explicit Compiler(BindCtx* c) : ctx(c) { memset(&env, 0, sizeof env); }

  void emit(uint8_t op, int arg, int stackDelta, bool isInt = false) {
    if (ncode >= kMaxCode) throw SqlError("expression too complex", 1105);
    env.code[ncode++] = Instr{op, (uint8_t)isInt, (uint16_t)arg};
    depth += stackDelta;
    if (depth > kMaxStack) throw SqlError("expression too deeply nested", 1105);
  }
  // Integer constants (OP_CONST with isInt) keep their exact 64-bit pattern in the double slot;
  // the interpreter and generated code decode it (BIGINT literals above 2^53 stay exact).
  int constantInt(int64_t v) {
    double d;
    memcpy(&d, &v, 8);
    for (int i = 0; i < nconst; ++i)
      if (memcmp(&env.consts[i], &d, 8) == 0) return i;
    if (nconst >= kMaxConsts) throw SqlError("too many constants in query", 1105);
    env.consts[nconst] = d;
    return nconst++;
  }
  int constant(double v) {
    for (int i = 0; i < nconst; ++i)
      if (env.consts[i] == v) return i;
    if (nconst >= kMaxConsts) throw SqlError("too many constants in query", 1105);
    env.consts[nconst] = v;
    return nconst++;
  }
  int slot(int bt, int bc) {
    auto it = slotOf.find({bt, bc});
    if (it != slotOf.end()) return it->second;
    int s = (int)slotCols.size();
    if (s >= kMaxCols) throw SqlError("too many columns referenced", 1105);
    slotOf[{bt, bc}] = s;
    slotCols.push_back({bt, bc});
    // Typed codegen needs the type up front; integer columns that have ever held a NULL use the
    // NULL-aware variants (3/4) so columns without NULLs keep the plain int64 fast path.
    const Table& tb = *ctx->tabs[bt];
    const uint8_t pt = (uint8_t)tb.cols[bc].pt;
    const bool nul = tb.hasNull[bc] || ctx->outerTabs.count(bt);
    const bool dec = tb.cols[bc].lt == LType::DECIMAL;
    env.cols[s] = ColView{nullptr, (uint8_t)(pt == (uint8_t)PType::VEC ? 5 : dec ? 6 : nul && pt != (uint8_t)PType::F64 ? pt + 3 : pt),
                          (uint8_t)(dec ? tb.cols[bc].scale : 0)};
    // Codegen hint: generate the decoding branch only for columns with compressed segments (each
    // such branch makes NVRTC version the loop; many of them blow up compile time).
    for (auto& sg : tb.segs)
      if ((int)sg->enc.size() > bc && sg->enc[bc]) { env.cols[s].enc = sg->enc[bc]; break; }
    return s;
  }
  std::vector<const uint32_t*> extSets;  // bitmaps already in VRAM (join filters), by set index
  std::vector<const std::vector<double>*> mapIds;  // value table per OP_DICTMAP / OP_INSORTED slot
  std::vector<DevVec<double>> mapsDev;
  std::vector<std::shared_ptr<void>> keepAlive;  // cached device copies this query uses
  std::vector<std::shared_ptr<std::vector<double>>> keepMaps;  // maps referenced only by this compiler
  int addMap(const std::vector<double>* m) {
    for (size_t i = 0; i < mapIds.size(); ++i)
      if (mapIds[i] == m) return (int)i;
    if ((int)mapIds.size() >= kMaxSets) throw SqlError("too many encoded expressions / IN subqueries in one query", 1105);
    mapIds.push_back(m);
    env.mapSize[mapIds.size() - 1] = (int32_t)m->size();
    return (int)mapIds.size() - 1;
  }
  int addSet(std::vector<uint32_t> bits, int nbits) {
    if ((int)setsHost.size() >= kMaxSets) throw SqlError("too many string predicates", 1105);
    env.setSize[setsHost.size()] = nbits;
    setsHost.push_back(std::move(bits));
    extSets.push_back(nullptr);
    return (int)setsHost.size() - 1;
  }
  int addDeviceSet(const uint32_t* devBits, int nbits) {
    int i = addSet({}, nbits);
    extSets[i] = devBits;
    return i;
  }
  // Bitmap of the dictionary entries satisfying pred (evaluated once per entry, on all cores).
  int dictSet(const Dict& d, const std::function<bool(std::string_view)>& pred, const std::string& cacheKey = "") {
    if (!cacheKey.empty()) {  // cached bitmap (and its device copy) for this dictionary version
      const std::string key = derivKey(d, cacheKey);
      DictDerived e;
      if (!derivGet(d, key, e)) {
        DictDerived stale;
        size_t from = derivGetStale(d, key, stale) && stale.bits ? stale.n & ~(size_t)31 : 0;  // whole words
        e = DictDerived{};
        e.d = &d;
        e.n = d.strs.size();
        e.bits = std::make_shared<std::vector<uint32_t>>((d.strs.size() + 31) / 32 + 1, 0);
        auto& bits = *e.bits;
        if (from) std::copy(stale.bits->begin(), stale.bits->begin() + from / 32, bits.begin());
        parallelFor(d.strs.size() - from, [&, from](size_t b, size_t en) {
          b += from, en += from;
          for (size_t i = b; i < en; ++i)
            if (pred(d.strs[i])) bits[i >> 5] |= 1u << (i & 31);
        });
        e.bitsDev = std::make_shared<DevVec<uint32_t>>((int64_t)bits.size());
        h2d(e.bitsDev->p, bits.data(), bits.size() * 4);
        derivPut(key, e);
      }
      keepAlive.push_back(e.bitsDev);
      int i = addSet(*e.bits, (int)d.strs.size());
      extSets[i] = e.bitsDev->p;
      return i;
    }
    std::vector<uint32_t> bits((d.strs.size() + 31) / 32 + 1, 0);
    parallelFor(d.strs.size(), [&](size_t b, size_t e) {
      for (size_t i = b; i < e; ++i)
        if (pred(d.strs[i])) bits[i >> 5] |= 1u << (i & 31);
    });
    return addSet(std::move(bits), (int)d.strs.size());
  }

  Prog compile(const ExprP& e) {
    Prog p;
    p.start = (int16_t)ncode;
    depth = 0;
    gen(e);
    p.len = (int16_t)(ncode - p.start);
    return p;
  }

  void strCompare(const std::string& op, const ExprP& col, const std::string& lit) {
    Dict& d = *ctx->dict(*col);
    if ((op == "=" || op == "<>") && ciColl()) {  // every case variant of the literal
      const std::string f = foldCase(lit);
      int s = dictSet(d, [&](std::string_view v) { return foldCase(v) == f; });
      gen(col);
      emit(OP_INSET, s, 0);
      if (op == "<>") emit(OP_NOT, 0, 0);
      return;
    }
    if (op == "=" || op == "<>") {
      gen(col);
      emit(OP_CONST, constantInt(d.find(lit)), 1, true);
      emit(op == "=" ? OP_EQ : OP_NE, 0, -1);
      return;
    }
    int s = dictSet(d, [&](std::string_view v) {
      int c = collCmp(v, lit);
      return op == "<" ? c < 0 : op == "<=" ? c <= 0 : op == ">" ? c > 0 : c >= 0;
    });
    gen(col);
    emit(OP_INSET, s, 0);
  }

  void gen(const ExprP& e) {
    if (hook && hook(e)) return;
    switch (e->kind) {
      case Expr::COL:
        if (e->ty == LType::VECTOR) throw SqlError("VECTOR values are only usable through vector functions on the GPU", 1235);
        emit(OP_COL, slot(e->bt, e->bc), 1);
        return;
      case Expr::LIT:
        if (e->lit.kind == Value::STR) throw SqlError("string literal '" + e->lit.s + "' used in a numeric context", 1235);
        if (e->lit.kind == Value::INT || e->lit.kind == Value::DATE) emit(OP_CONST, constantInt(e->lit.i), 1, true);
        else emit(OP_CONST, constant(e->lit.num()), 1, false);
        return;
      case Expr::BIN: {
        const ExprP& a = e->args[0];
        const ExprP& b = e->args[1];
        if (isCmp(e->op) && (isNullLit(a) || isNullLit(b))) {  // x = NULL is NULL
          emit(OP_CONST, constant(nullv()), 1);
          return;
        }
        if (isCmp(e->op) && (a->ty == LType::VARCHAR || b->ty == LType::VARCHAR)) {
          static const std::map<std::string, std::string> flip = {{"=", "="}, {"<>", "<>"}, {"<", ">"}, {"<=", ">="}, {">", "<"}, {">=", "<="}};
          if (ctx->dict(*a) && b->kind == Expr::LIT && b->lit.kind == Value::STR) return strCompare(e->op, a, b->lit.s);
          if (ctx->dict(*b) && a->kind == Expr::LIT && a->lit.kind == Value::STR) return strCompare(flip.at(e->op), b, a->lit.s);
          if (ctx->dict(*a) && ctx->dict(*a) == ctx->dict(*b) && (e->op == "=" || e->op == "<>")) {
            auto canon = ciColl() ? ciCanon(*ctx->dict(*a)) : nullptr;
            gen(a);
            if (canon) emit(OP_DICTMAP, addMap(canon.get()), 0), keepMaps.push_back(canon);
            gen(b);
            if (canon) emit(OP_DICTMAP, addMap(canon.get()), 0);
            emit(e->op == "=" ? OP_EQ : OP_NE, 0, -1);
            return;
          }
          if (ctx->dict(*a) && ctx->dict(*b)) {  // columns with different dictionaries: both to ranks in their union
            auto [ma, mb] = crossRank(*ctx->dict(*a), *ctx->dict(*b));
            gen(a);
            emit(OP_DICTMAP, addMap(ma.get()), 0);
            gen(b);
            emit(OP_DICTMAP, addMap(mb.get()), 0);
            keepMaps.push_back(ma);
            keepMaps.push_back(mb);
            static const std::map<std::string, Op> cmp = {{"=", OP_EQ}, {"<>", OP_NE}, {"<", OP_LT}, {"<=", OP_LE}, {">", OP_GT}, {">=", OP_GE}};
            emit(cmp.at(e->op), 0, -1);
            return;
          }
          throw SqlError("unsupported string comparison: " + e->str(), 1235);
        }
        gen(a);
        gen(b);
        static const std::map<std::string, Op> ops = {
            {"+", OP_ADD}, {"-", OP_SUB}, {"*", OP_MUL}, {"/", OP_DIV}, {"%", OP_MOD}, {"=", OP_EQ}, {"<>", OP_NE},
            {"<", OP_LT}, {"<=", OP_LE}, {">", OP_GT}, {">=", OP_GE}, {"AND", OP_AND}, {"OR", OP_OR}};
        emit(ops.at(e->op), 0, -1);
        return;
      }
      case Expr::UNARY:
        gen(e->args[0]);
        emit(e->op == "NOT" ? OP_NOT : OP_NEG, 0, 0);
        return;
      case Expr::INLIST: {
        if (!ctx->dict(*e->args[0])) throw SqlError("IN on this string expression can't run on the GPU", 1235);
        Dict& d = *ctx->dict(*e->args[0]);
        std::set<std::string> vals;
        const bool ci = ciColl();
        for (size_t i = 1; i < e->args.size(); ++i) vals.insert(ci ? foldCase(e->args[i]->lit.s) : e->args[i]->lit.s);
        int s = dictSet(d, [&](std::string_view v) { return vals.count(ci ? foldCase(v) : std::string(v)) > 0; });
        gen(e->args[0]);
        emit(OP_INSET, s, 0);
        if (e->neg) emit(OP_NOT, 0, 0);
        return;
      }
      case Expr::LIKE: {
        if (e->args[0]->ty != LType::VARCHAR || !ctx->dict(*e->args[0]))
          throw SqlError("LIKE on this expression can't run on the GPU", 1235);
        const std::string pat = e->args[1]->lit.s;
        bool letters = false;
        for (char ch : pat) letters = letters || isalpha((unsigned char)ch);
        const bool ci = ciColl() && letters;  // (no letters in the pattern: case can't matter)
        const std::string fp = ci ? foldCase(pat) : pat;
        const auto m = likeMatcher(fp);
        int s = dictSet(*ctx->dict(*e->args[0]), [&](std::string_view v) {
          if (!ci) return m(v);
          thread_local std::string buf;  // (folded without an allocation per string)
          buf.assign(v);
          for (auto& ch : buf) ch = (char)tolower((unsigned char)ch);
          return m(buf);
        }, "LIKE|" + pat);
        gen(e->args[0]);
        emit(OP_INSET, s, 0);
        if (e->neg) emit(OP_NOT, 0, 0);
        return;
      }
      case Expr::CASE: genCase(e, 0); return;
      case Expr::FUNC:
        if (e->op == "YEAR") {
          gen(e->args[0]);
          emit(OP_YEAR, 0, 0);
          return;
        }
        if (e->op == "ISNULL" || e->op == "ISNOTNULL") {
          gen(e->args[0]);
          emit(OP_ISNULL, 0, 0);
          if (e->op == "ISNOTNULL") emit(OP_NOT, 0, 0);
          return;
        }
        if (e->op == "COALESCE2" || e->op == "NULLIF") {
          gen(e->args[0]);
          gen(e->args[1]);
          emit(e->op == "NULLIF" ? OP_NULLIF : OP_COALESCE, 0, -1);
          return;
        }
        if (e->op.rfind("#ENC", 0) == 0) {  // encoded expression: compile its GPU form
          gen(e->args[0]);
          return;
        }
        if (e->op == "#VDOT" || e->op == "#VEUC") {
          const ExprP& c = e->args[0];
          const int sl = slot(c->bt, c->bc), m = addMap(e->aux.get());
          emit(e->op == "#VDOT" ? OP_VDOT : OP_VEUC, sl | (m << 8), 1);
          return;
        }
        if (e->op == "#RAW") {  // stored integer of a DECIMAL column (exact sums)
          emit(OP_COLRAW, slot(e->args[0]->bt, e->args[0]->bc), 1, !nullable(e->args[0], *ctx));
          return;
        }
        if (e->op == "#INBITS") {  // arg: x - base; bit test in the device bitmap
          auto* b = (DevBits*)e->devAux.get();
          gen(e->args[0]);
          emit(OP_INSET, addDeviceSet(b->p, (int)b->nbits), 0);
          return;
        }
        if (e->op == "#MAP" || e->op == "#INSORTED") {
          gen(e->args[0]);
          emit(e->op == "#MAP" ? OP_DICTMAP : OP_INSORTED, addMap(e->aux.get()), 0);
          return;
        }
        if (e->fnId >= 0) {
          for (auto& a : e->args) gen(a);
          emit(e->args.size() == 1 ? OP_CALL1 : OP_CALL2, e->fnId, 1 - (int)e->args.size());
          return;
        }
        if (isAggName(e->op)) throw SqlError("aggregate in an invalid position: " + e->str(), 1111);
        throw SqlError("cannot run " + e->op + "() on the GPU", 1235);
      default: throw SqlError("cannot compile expression: " + e->str(), 1235);
    }
  }
  void genCase(const ExprP& e, size_t i) {
    if (i + 1 >= e->args.size()) {
      if (i < e->args.size()) gen(e->args[i]);
      else emit(OP_CONST, constantInt(0), 1, true);
      return;
    }
    if (e->args[i + 1]->ty == LType::VARCHAR) throw SqlError("CASE returning strings is not supported", 1235);
    gen(e->args[i]);
    gen(e->args[i + 1]);
    genCase(e, i + 2);
    emit(OP_SELECT, 0, -2);
  }

  void uploadSets() {
    env.ovf = ovfFlagDev();
    mapsDev.clear();
    for (size_t i = 0; i < mapIds.size(); ++i) {
      const auto& m = *mapIds[i];
      if (auto dv = derivMapDev(&m)) {  // cached map: shared device copy, no upload
        keepAlive.push_back(dv);
        env.maps[i] = dv->p;
        continue;
      }
      mapsDev.emplace_back(std::max<int64_t>((int64_t)m.size(), 1));
      if (!m.empty()) h2d(mapsDev.back().p, m.data(), m.size() * 8);
      env.maps[i] = mapsDev.back().p;
    }
    setsDev.clear();
    for (size_t i = 0; i < setsHost.size(); ++i) {
      if (extSets[i]) {
        env.sets[i] = extSets[i];
        continue;
      }
      setsDev.emplace_back((int64_t)setsHost[i].size());
      h2d(setsDev.back().p, setsHost[i].data(), setsHost[i].size() * 4);
      env.sets[i] = setsDev.back().p;
    }
  }
  ExprEnv hostEnv() const {
    ExprEnv h = env;
    for (size_t i = 0; i < setsHost.size(); ++i) h.sets[i] = setsHost[i].data();
    for (size_t i = 0; i < mapIds.size(); ++i) h.maps[i] = mapIds[i]->data();
    return h;
  }
};

// ================================================================== helpers

inline Value typedVal(double v, LType t, const Dict* d, int scale = -1) {
  if (std::isnan(v)) return Value::Null();
  switch (t) {
    case LType::DECIMAL: return Value::Dec(v, scale < 0 ? 2 : scale);
    case LType::INT:
    case LType::BIGINT: return Value::I((int64_t)std::llround(v));
    case LType::DOUBLE: return Value::D(v);
    case LType::DATE: return Value::Dt((int64_t)std::llround(v));
    case LType::DATETIME: return Value::Ts((int64_t)std::llround(v));
    case LType::VARCHAR: {
      int64_t c = (int64_t)v;
      return Value::S(d && c >= 0 && c < (int64_t)d->strs.size() ? std::string(d->strs[c]) : std::string());
    }
  }
  return Value::D(v);
}

inline std::string cell(const Value& v) { return v.isNull() ? kNullCell : v.toString(); }

inline int cmpVal(const Value& a, const Value& b) {
  if (a.isNull() || b.isNull()) return a.isNull() == b.isNull() ? 0 : a.isNull() ? -1 : 1;
  if (a.kind == Value::STR && b.kind == Value::STR) return collCmp(a.s, b.s);
  if (a.kind != Value::DBL && b.kind != Value::DBL && a.kind != Value::STR && b.kind != Value::STR)
    return a.i < b.i ? -1 : a.i > b.i ? 1 : 0;
  double x = a.num(), y = b.num();
  return x < y ? -1 : x > y ? 1 : 0;
}

struct AggRef {
  std::string fn;
  int slot = -1;     // SUM/MIN/MAX slot; for COUNT the slot holding the count (0 = COUNT(*))
  int cntSlot = 0;   // slot counting the argument's non-NULL rows (0 = COUNT(*) when it can't be NULL)
  LType argTy = LType::BIGINT;
  const Dict* dict = nullptr;  // MIN/MAX over strings: the value is a rank of this dictionary's strings
  std::shared_ptr<const std::vector<int32_t>> unrank;  // ... decoded through rank -> code
  int decScale = -1;           // SUM over a DECIMAL column: the slot holds the exact raw (scaled) sum
  int scale = -1;              // display scale of a DECIMAL result
  bool distinct = false;       // COUNT(DISTINCT x) (two-phase GPU path)
  int hiSlot = -1;             // grouped BIGINT SUM: slot summing the high 32 bits (slot sums the low 32)
};

// Host-side evaluation of post-aggregation expressions (and constant expressions).
struct PostCtx {
  const BindCtx* bc = nullptr;
  std::vector<std::string> keyCanon;
  std::vector<LType> keyTy;
  std::vector<const Dict*> keyDict;
  std::vector<int> keyScale;
  const double* keys = nullptr;
  std::vector<uint8_t> keyInt;  // key k holds its exact int64 bits (AggPlan::keyInt)
  std::map<std::string, AggRef>* aggs = nullptr;
  const double* accs = nullptr;
  const uint8_t* accBits = nullptr;  // slot holds an exact sum's low 64 bits (GpuAggregator::sumBitsSlots)
  const int64_t* accHi = nullptr;    // ... and these its high 64 bits
  const std::vector<Value>* row = nullptr;  // host row evaluation: values of bound columns (see colSlot)
  std::function<int(int, int)> colSlot;     // (table, column) -> index into *row
  mutable std::unordered_map<const Expr*, std::string> canonCache;
  const std::string& canonOf(const ExprP& e) const {
    auto it = canonCache.find(e.get());
    if (it == canonCache.end()) it = canonCache.emplace(e.get(), canon(e)).first;
    return it->second;
  }
};

Value evalHostFunc(const ExprP& e, const struct PostCtx& c);

inline double numOf(const Value& v) {
  if (v.isNull()) return nullv();
  switch (v.kind) {
    case Value::DBL: return v.d;
    case Value::STR: return std::atof(v.s.c_str());
    default: return (double)v.i;
  }
}
inline bool isTrue(const Value& v) { return !v.isNull() && (v.kind == Value::STR ? std::atof(v.s.c_str()) != 0 : numOf(v) != 0); }
inline Value boolVal(double v) { return v != v ? Value::Null() : Value::I(v != 0); }

inline Value evalPost(const ExprP& e, const PostCtx& c) {
  if (!c.keyCanon.empty() && c.keys) {
    const std::string& k = c.canonOf(e);
    for (size_t i = 0; i < c.keyCanon.size(); ++i)
      if (c.keyCanon[i] == k) {
        if (i < c.keyInt.size() && c.keyInt[i]) {  // exact integer key
          const long long v = bitsOf(c.keys[i]);
          switch (c.keyTy[i]) {
            case LType::INT: case LType::BIGINT: return Value::I(v);
            case LType::DATE: return Value::Dt(v);
            case LType::DATETIME: return Value::Ts(v);
            default: return typedVal((double)v, c.keyTy[i], c.keyDict[i], i < c.keyScale.size() ? c.keyScale[i] : -1);
          }
        }
        return typedVal(c.keys[i], c.keyTy[i], c.keyDict[i], i < c.keyScale.size() ? c.keyScale[i] : -1);
      }
  }
  switch (e->kind) {
    case Expr::LIT: return e->lit;
    case Expr::COL:
      if (c.row && c.colSlot) {
        int i = c.colSlot(e->bt, e->bc);
        if (i >= 0) return (*c.row)[i];
      }
      throw SqlError("Expression '" + e->str() + "' is not in GROUP BY clause", 1055);
    case Expr::FUNC: {
      if (isAggName(e->op)) {
        if (!c.aggs || !c.accs) throw SqlError("Invalid use of group function", 1111);
        const AggRef& r = c.aggs->at(c.canonOf(e));
        if (r.fn == "COUNT") return Value::I((int64_t)c.accs[r.slot]);
        double cnt = c.accs[r.cntSlot];
        if (cnt == 0) return Value::Null();
        if (c.accBits && c.accBits[r.slot] && (r.fn == "SUM" || r.fn == "AVG")) {  // exact integer sum
          const __int128 w = r.hiSlot >= 0 && c.accBits[r.hiSlot]
                                 ? (__int128)bitsOf(c.accs[r.hiSlot]) * ((__int128)1 << 32) + bitsOf(c.accs[r.slot])
                                 : (__int128)(((unsigned __int128)(uint64_t)(c.accHi ? c.accHi[r.slot] : 0) << 64) | (uint64_t)bitsOf(c.accs[r.slot]));
          const double wd = (double)w;
          if (r.fn == "AVG") {
            const double av = wd / (r.decScale >= 0 ? pow10i(r.decScale) : 1.0) / cnt;
            return r.scale >= 0 ? Value::Dec(av, r.scale) : Value::D(av);
          }
          if (r.argTy == LType::DECIMAL || r.decScale >= 0) return Value::Dec(wd / pow10i(std::max(0, r.decScale)), std::max(0, r.scale));
          if (w >= (__int128)INT64_MIN && w <= (__int128)INT64_MAX) return Value::I((int64_t)w);
          return Value::D(wd);  // beyond 64 bits
        }
        double v = c.accs[r.slot];
        if (r.decScale >= 0) v /= pow10i(r.decScale);
        if (r.fn == "AVG") return r.scale >= 0 && !r.dict ? Value::Dec(v / cnt, r.scale) : Value::D(v / cnt);
        if (r.fn == "SUM")
          return r.argTy == LType::DOUBLE ? Value::D(v) : r.argTy == LType::DECIMAL ? Value::Dec(v, std::max(0, r.scale))
                                                         : Value::I((int64_t)std::llround(v));
        if (c.accBits && c.accBits[r.slot]) {  // exact int64 MIN / MAX
          const int64_t iv = bitsOf(c.accs[r.slot]);
          if (r.unrank) return typedVal(iv >= 0 && iv < (int64_t)r.unrank->size() ? (*r.unrank)[iv] : -1, LType::VARCHAR, r.dict);
          if (r.argTy == LType::DATETIME) return Value::Ts(iv);
          if (r.argTy == LType::DATE) return Value::Dt(iv);
          if (r.argTy == LType::INT || r.argTy == LType::BIGINT) return Value::I(iv);
          return typedVal((double)iv, r.argTy, r.dict, r.scale);
        }
        if (r.unrank) {  // string MIN/MAX: rank -> code
          const int64_t rk = std::llround(v);
          return typedVal(rk >= 0 && rk < (int64_t)r.unrank->size() ? (*r.unrank)[rk] : -1, LType::VARCHAR, r.dict);
        }
        return typedVal(v, r.argTy, r.dict, r.scale);
      }
      if (e->op == "ISNULL" || e->op == "ISNOTNULL") return Value::I(evalPost(e->args[0], c).isNull() == (e->op == "ISNULL"));
      if (e->op == "COALESCE2") {
        Value a = evalPost(e->args[0], c);
        return a.isNull() ? evalPost(e->args[1], c) : a;
      }
      if (e->op == "NULLIF") {
        Value a = evalPost(e->args[0], c), b = evalPost(e->args[1], c);
        return !a.isNull() && !b.isNull() && cmpVal(a, b) == 0 ? Value::Null() : a;
      }
      if (e->op == "YEAR") {
        Value a = evalPost(e->args[0], c);
        if (a.isNull()) return a;
        int64_t y; unsigned m, d;
        civilFromDays(a.kind == Value::DATETIME ? floorDiv(a.i, kUsPerDay) : a.i, y, m, d);
        return Value::I(y);
      }
      if (e->fnId >= 0) {
        double x = numOf(evalPost(e->args[0], c));
        double r = e->args.size() == 1 ? gdbCall1(e->fnId, x) : gdbCall2(e->fnId, x, numOf(evalPost(e->args[1], c)));
        return typedVal(r, e->ty, nullptr, e->scale);
      }
      if (e->op.rfind("#ENC", 0) == 0) return evalPost(e->args[1], c);
      if (e->op == "#MAP") throw SqlError("internal: encoded map evaluated on the host", 1105);
      if (e->op == "#VDOT" || e->op == "#VEUC") {
        Value x = evalPost(e->args[0], c);
        std::vector<float> v;
        if (x.isNull() || !parseVector(x.toString(), v) || v.size() != e->aux->size()) return Value::Null();
        double acc = 0;
        for (size_t i = 0; i < v.size(); ++i) {
          double q = (*e->aux)[i];
          acc += e->op == "#VDOT" ? v[i] * q : (v[i] - q) * (v[i] - q);
        }
        return Value::D(e->op == "#VDOT" ? acc : std::sqrt(acc));
      }
      if (e->op == "#INBITS") {  // (args[0] is x - base)
        Value x = evalPost(e->args[0], c);
        if (x.isNull()) return x;
        auto* b = (DevBits*)e->devAux.get();
        return Value::I(b->test((int64_t)std::llround(numOf(x)) + b->base));
      }
      if (e->op == "#INSORTED") {
        Value x = evalPost(e->args[0], c);
        if (x.isNull()) return x;
        const auto& v = *e->aux;
        return Value::I(std::binary_search(v.begin(), v.end(), numOf(x)));
      }
      return evalHostFunc(e, c);
    }
    case Expr::UNARY: {
      Value a = evalPost(e->args[0], c);
      if (e->op == "NOT") return a.isNull() ? a : Value::I(!isTrue(a));
      if (a.isNull()) return a;
      return a.kind == Value::DBL ? Value::D(-a.d) : Value::I(-a.i);
    }
    case Expr::BIN: {
      const std::string& o = e->op;
      Value a = evalPost(e->args[0], c);
      if (o == "AND" && !a.isNull() && !isTrue(a)) return Value::I(0);
      if (o == "OR" && isTrue(a)) return Value::I(1);
      Value b = evalPost(e->args[1], c);
      if (o == "AND") return boolVal(and3(a.isNull() ? nullv() : isTrue(a), b.isNull() ? nullv() : isTrue(b)));
      if (o == "OR") return boolVal(or3(a.isNull() ? nullv() : isTrue(a), b.isNull() ? nullv() : isTrue(b)));
      if (a.isNull() || b.isNull()) return Value::Null();
      if (isCmp(o)) {
        int r = cmpVal(a, b);
        return Value::I(o == "=" ? r == 0 : o == "<>" ? r != 0 : o == "<" ? r < 0 : o == "<=" ? r <= 0 : o == ">" ? r > 0 : r >= 0);
      }
      const bool ints = a.kind != Value::DBL && a.kind != Value::STR && b.kind != Value::DBL && b.kind != Value::STR;
      if (ints && o != "/") {
        int64_t x = a.i, y = b.i;
        if (o == "%" && y == 0) return Value::Null();
        int64_t r;
        const bool ovf = o == "+" ? __builtin_add_overflow(x, y, &r) : o == "-" ? __builtin_sub_overflow(x, y, &r)
                       : o == "*" ? __builtin_mul_overflow(x, y, &r) : (r = (y == -1 ? 0 : x % y), false);
        if (ovf && e->ty != LType::DATE && e->ty != LType::DATETIME)
          throw SqlError("BIGINT value is out of range in '" + e->str() + "'", 1690);
        return typedVal((double)r, e->ty, nullptr).kind == Value::DBL ? Value::D((double)r)
             : e->ty == LType::DATE ? Value::Dt(r) : e->ty == LType::DATETIME ? Value::Ts(r) : Value::I(r);
      }
      double x = numOf(a), y = numOf(b);
      double r = o == "+" ? x + y : o == "-" ? x - y : o == "*" ? x * y : o == "/" ? divz(x, y) : (y == 0 ? nullv() : std::fmod(x, y));
      return typedVal(r, e->ty == LType::VARCHAR ? LType::DOUBLE : e->ty, nullptr, e->scale);
    }
    case Expr::CASE: {
      for (size_t i = 0; i + 1 < e->args.size(); i += 2)
        if (isTrue(evalPost(e->args[i], c))) return evalPost(e->args[i + 1], c);
      return e->args.size() % 2 ? evalPost(e->args.back(), c) : Value::Null();
    }
    case Expr::INLIST: {
      Value a = evalPost(e->args[0], c);
      if (a.isNull()) return a;
      bool hit = false;
      for (size_t i = 1; i < e->args.size() && !hit; ++i) hit = cmpVal(a, evalPost(e->args[i], c)) == 0;
      return Value::I(hit != e->neg);
    }
    case Expr::LIKE: {
      Value a = evalPost(e->args[0], c);
      if (a.isNull()) return a;
      return Value::I((ciColl() ? likeMatch(foldCase(a.toString()), foldCase(e->args[1]->lit.s))
                                : likeMatch(a.toString(), e->args[1]->lit.s)) != e->neg);
    }
    default: throw SqlError("cannot evaluate " + e->str(), 1235);
  }
}

inline Value evalConst(const ExprP& e) {
  PostCtx pc;
  return evalPost(e, pc);
}

// ---- Minimal JSON navigation for JSON_EXTRACT_* (SingleStore) and JSON_EXTRACT (MySQL).
inline size_t jsonWs(const std::string& s, size_t i) {
  while (i < s.size() && isspace((unsigned char)s[i])) ++i;
  return i;
}
inline size_t jsonSkip(const std::string& s, size_t i) {  // index just past the value starting at i
  i = jsonWs(s, i);
  if (i >= s.size()) return i;
  char ch = s[i];
  if (ch == '"') {
    for (++i; i < s.size() && s[i] != '"'; ++i)
      if (s[i] == '\\') ++i;
    return i + 1;
  }
  if (ch == '{' || ch == '[') {
    char close = ch == '{' ? '}' : ']';
    ++i;
    for (;;) {
      i = jsonWs(s, i);
      if (i >= s.size()) return i;
      if (s[i] == close) return i + 1;
      if (s[i] == ',') { ++i; continue; }
      if (ch == '{') {
        i = jsonSkip(s, i);  // key
        i = jsonWs(s, i);
        if (i < s.size() && s[i] == ':') ++i;
      }
      i = jsonSkip(s, i);
    }
  }
  while (i < s.size() && s[i] != ',' && s[i] != '}' && s[i] != ']' && !isspace((unsigned char)s[i])) ++i;
  return i;
}
inline std::string jsonUnquote(const std::string& v) {
  if (v.size() < 2 || v[0] != '"') return v;
  std::string r;
  for (size_t i = 1; i + 1 < v.size(); ++i) {
    if (v[i] == '\\' && i + 2 < v.size()) {
      char n = v[++i];
      r += n == 'n' ? '\n' : n == 't' ? '\t' : n == 'r' ? '\r' : n;
    } else {
      r += v[i];
    }
  }
  return r;
}
// Returns the raw JSON text at the path (object keys or array indexes), or false.
inline bool jsonFind(const std::string& js, const std::vector<std::string>& path, std::string& out) {
  size_t b = jsonWs(js, 0), e = jsonSkip(js, b);
  for (const std::string& key : path) {
    if (b >= js.size()) return false;
    if (js[b] == '{') {
      size_t i = b + 1;
      bool found = false;
      for (;;) {
        i = jsonWs(js, i);
        if (i >= e || js[i] == '}') break;
        if (js[i] == ',') { ++i; continue; }
        size_t ke = jsonSkip(js, i);
        std::string k = jsonUnquote(js.substr(i, ke - i));
        i = jsonWs(js, ke);
        if (i < js.size() && js[i] == ':') ++i;
        size_t vb = jsonWs(js, i), ve = jsonSkip(js, vb);
        if (k == key) { b = vb; e = ve; found = true; break; }
        i = ve;
      }
      if (!found) return false;
    } else if (js[b] == '[') {
      long idx = std::atol(key.c_str());
      size_t i = b + 1;
      long n = 0;
      bool found = false;
      for (;;) {
        i = jsonWs(js, i);
        if (i >= e || js[i] == ']') break;
        if (js[i] == ',') { ++i; continue; }
        size_t ve = jsonSkip(js, i);
        if (n++ == idx) { b = i; e = ve; found = true; break; }
        i = ve;
      }
      if (!found) return false;
    } else {
      return false;
    }
  }
  out = js.substr(b, e - b);
  return true;
}
inline std::vector<std::string> jsonPathOf(const std::string& p) {  // '$.a.b[2]' -> {a, b, 2}
  std::vector<std::string> r;
  size_t i = p.rfind('$', 0) == 0 ? 1 : 0;
  while (i < p.size()) {
    if (p[i] == '.') {
      size_t j = ++i;
      if (j < p.size() && p[j] == '"') {
        size_t k = p.find('"', j + 1);
        r.push_back(p.substr(j + 1, k - j - 1));
        i = k + 1;
        continue;
      }
      while (i < p.size() && p[i] != '.' && p[i] != '[') ++i;
      r.push_back(p.substr(j, i - j));
    } else if (p[i] == '[') {
      size_t k = p.find(']', i);
      r.push_back(p.substr(i + 1, k - i - 1));
      i = k + 1;
    } else {
      ++i;
    }
  }
  return r;
}

std::string jsonQuoteStr(const std::string& x);
// JSON_SET / JSON_INSERT / JSON_REPLACE / JSON_REMOVE on one path (mode 0..3): the edited JSON text.
inline std::string jsonEdit(const std::string& v, const std::vector<std::string>& path, size_t k, const std::string& val, int mode) {
  const size_t b = jsonWs(v, 0);
  if (k >= path.size() || b >= v.size() || (v[b] != '{' && v[b] != '[')) return v;
  const bool obj = v[b] == '{', last = k + 1 == path.size();
  struct Member { size_t from, vb, ve; std::string key; };  // from: start of the member (key or value)
  std::vector<Member> ms;
  size_t i = b + 1, endPos = v.size();
  for (;;) {
    i = jsonWs(v, i);
    if (i >= v.size()) break;
    if (v[i] == ']' || v[i] == '}') { endPos = i; break; }
    if (v[i] == ',') { ++i; continue; }
    Member m{i, i, i, ""};
    if (obj) {
      const size_t ke = jsonSkip(v, i);
      m.key = jsonUnquote(v.substr(i, ke - i));
      i = jsonWs(v, ke);
      if (i < v.size() && v[i] == ':') ++i;
      i = jsonWs(v, i);
    }
    m.vb = i;
    m.ve = i = jsonSkip(v, i);
    ms.push_back(m);
  }
  int at = -1;
  if (obj) {
    for (size_t j = 0; j < ms.size() && at < 0; ++j)
      if (ms[j].key == path[k]) at = (int)j;
  } else {
    const std::string& ix = path[k];
    if (ix == "last") at = ms.empty() ? -1 : (int)ms.size() - 1;
    else if (!ix.empty() && std::all_of(ix.begin(), ix.end(), ::isdigit) && std::stoull(ix) < ms.size()) at = (int)std::stoull(ix);
    else if (ix.empty() || !std::all_of(ix.begin(), ix.end(), ::isdigit)) return v;
  }
  if (at >= 0) {
    const Member& m = ms[at];
    if (!last) return v.substr(0, m.vb) + jsonEdit(v.substr(m.vb, m.ve - m.vb), path, k + 1, val, mode) + v.substr(m.ve);
    if (mode == 1) return v;
    if (mode == 3) {  // remove the member and one adjacent comma
      size_t from = m.from, to = m.ve;
      if (at + 1 < (int)ms.size()) to = ms[at + 1].from;
      else if (at > 0) from = ms[at - 1].ve;
      return v.substr(0, from) + v.substr(to);
    }
    return v.substr(0, m.vb) + val + v.substr(m.ve);
  }
  if (!last || mode == 2 || mode == 3) return v;
  const std::string add = (ms.empty() ? "" : ", ") + (obj ? jsonQuoteStr(path[k]) + ": " : std::string()) + val;
  const size_t ins = ms.empty() ? endPos : ms.back().ve;
  return v.substr(0, ins) + add + v.substr(ins);
}

inline std::string dateFormat(const Value& v, const std::string& fmt) {
  int64_t us = v.kind == Value::DATETIME ? v.i : v.kind == Value::DATE ? v.i * kUsPerDay : 0;
  if (v.kind == Value::STR && !parseDateTime(v.s, us)) return "";
  int64_t days = floorDiv(us, kUsPerDay), rem = us - days * kUsPerDay;
  int64_t y;
  unsigned m, d;
  civilFromDays(days, y, m, d);
  int H = (int)(rem / 3600000000ll), M = (int)(rem / 60000000ll % 60), S = (int)(rem / 1000000ll % 60);
  static const char* mon[] = {"January", "February", "March", "April", "May", "June", "July", "August", "September",
                              "October", "November", "December"};
  static const char* wd[] = {"Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"};
  int dow = (int)((days + 4) - floorDiv(days + 4, 7) * 7);
  std::string r;
  char b[32];
  for (size_t i = 0; i < fmt.size(); ++i) {
    if (fmt[i] != '%' || i + 1 == fmt.size()) { r += fmt[i]; continue; }
    switch (fmt[++i]) {
      case 'Y': snprintf(b, sizeof b, "%04lld", (long long)y); break;
      case 'y': snprintf(b, sizeof b, "%02lld", (long long)(y % 100)); break;
      case 'm': snprintf(b, sizeof b, "%02u", m); break;
      case 'c': snprintf(b, sizeof b, "%u", m); break;
      case 'd': snprintf(b, sizeof b, "%02u", d); break;
      case 'e': snprintf(b, sizeof b, "%u", d); break;
      case 'H': snprintf(b, sizeof b, "%02d", H); break;
      case 'k': snprintf(b, sizeof b, "%d", H); break;
      case 'h': case 'I': snprintf(b, sizeof b, "%02d", H % 12 ? H % 12 : 12); break;
      case 'l': snprintf(b, sizeof b, "%d", H % 12 ? H % 12 : 12); break;
      case 'i': snprintf(b, sizeof b, "%02d", M); break;
      case 's': case 'S': snprintf(b, sizeof b, "%02d", S); break;
      case 'f': snprintf(b, sizeof b, "%06lld", (long long)(rem % 1000000)); break;
      case 'p': snprintf(b, sizeof b, "%s", H < 12 ? "AM" : "PM"); break;
      case 'M': snprintf(b, sizeof b, "%s", mon[m - 1]); break;
      case 'b': snprintf(b, sizeof b, "%.3s", mon[m - 1]); break;
      case 'W': snprintf(b, sizeof b, "%s", wd[dow]); break;
      case 'a': snprintf(b, sizeof b, "%.3s", wd[dow]); break;
      case 'w': snprintf(b, sizeof b, "%d", dow); break;
      case 'j': snprintf(b, sizeof b, "%03lld", (long long)(days - daysFromCivil(y, 1, 1) + 1)); break;
      case 'T': snprintf(b, sizeof b, "%02d:%02d:%02d", H, M, S); break;
      case 'r': snprintf(b, sizeof b, "%02d:%02d:%02d %s", H % 12 ? H % 12 : 12, M, S, H < 12 ? "AM" : "PM"); break;
      case 'D': snprintf(b, sizeof b, "%u%s", d, (d % 10 == 1 && d != 11) ? "st" : (d % 10 == 2 && d != 12) ? "nd" : (d % 10 == 3 && d != 13) ? "rd" : "th"); break;
      default: b[0] = fmt[i]; b[1] = 0;
    }
    r += b;
  }
  return r;
}

inline size_t utf8Len(const std::string& s) {
  size_t n = 0;
  for (unsigned char ch : s) n += (ch & 0xc0) != 0x80;
  return n;
}

// Character (not byte) positions in UTF-8 text, as MySQL's string functions count them.
inline bool asciiOnly(const std::string& s) {
  for (unsigned char ch : s)
    if (ch & 0x80) return false;
  return true;
}
// Byte offset of character k (clamped to the end).
inline size_t u8Off(const std::string& s, int64_t k) {
  if (k <= 0) return 0;
  size_t i = 0;
  for (int64_t c = 0; i < s.size(); ++i)
    if (((unsigned char)s[i] & 0xc0) != 0x80 && c++ == k) return i;
  return s.size();
}
// Characters [from, from + n) (from 0-based).
inline std::string u8Sub(const std::string& s, int64_t from, int64_t n) {
  if (asciiOnly(s)) return from >= (int64_t)s.size() || n <= 0 ? std::string() : s.substr(from, n);
  const size_t b = u8Off(s, from), e = n >= (int64_t)s.size() ? s.size() : u8Off(s, from + n);
  return b >= e ? std::string() : s.substr(b, e - b);
}
// Code points <-> UTF-8, for case mapping.
inline void u8Decode(const std::string& s, std::vector<uint32_t>& out) {
  out.clear();
  for (size_t i = 0; i < s.size();) {
    const unsigned char c = s[i];
    const int len = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 1;
    uint32_t cp = len == 1 ? c : len == 2 ? c & 0x1f : len == 3 ? c & 0x0f : c & 0x07;
    for (int k = 1; k < len && i + k < s.size(); ++k) cp = (cp << 6) | ((unsigned char)s[i + k] & 0x3f);
    out.push_back(len == 1 && c >= 0x80 ? 0xFFFD : cp);
    i += len;
  }
}
inline void u8Append(std::string& o, uint32_t cp) {
  if (cp < 0x80) o += (char)cp;
  else if (cp < 0x800) o += (char)(0xc0 | (cp >> 6)), o += (char)(0x80 | (cp & 0x3f));
  else if (cp < 0x10000) o += (char)(0xe0 | (cp >> 12)), o += (char)(0x80 | ((cp >> 6) & 0x3f)), o += (char)(0x80 | (cp & 0x3f));
  else o += (char)(0xf0 | (cp >> 18)), o += (char)(0x80 | ((cp >> 12) & 0x3f)), o += (char)(0x80 | ((cp >> 6) & 0x3f)), o += (char)(0x80 | (cp & 0x3f));
}
// Upper / lower case: ASCII, Latin-1, Latin Extended-A, Greek and Cyrillic.
inline uint32_t cpUpper(uint32_t c) {
  if (c < 0x80) return c >= 'a' && c <= 'z' ? c - 32 : c;
  if ((c >= 0xE0 && c <= 0xFE && c != 0xF7)) return c - 0x20;
  if (c == 0xFF) return 0x178;
  if (c >= 0x100 && c <= 0x17F && c != 0x130 && c != 0x131 && c != 0x138 && c != 0x149 && c != 0x17F) {
    const bool pairEven = (c >= 0x100 && c <= 0x137) || (c >= 0x14A && c <= 0x177);
    if (pairEven) return (c & 1) ? c - 1 : c;
    if ((c >= 0x139 && c <= 0x148) || (c >= 0x179 && c <= 0x17E)) return (c & 1) ? c : c - 1;
  }
  if (c >= 0x3B1 && c <= 0x3C9 && c != 0x3C2) return c - 0x20;
  if (c == 0x3C2) return 0x3A3;
  if (c >= 0x430 && c <= 0x44F) return c - 0x20;
  if (c >= 0x450 && c <= 0x45F) return c - 0x50;
  return c;
}
inline uint32_t cpLower(uint32_t c) {
  if (c < 0x80) return c >= 'A' && c <= 'Z' ? c + 32 : c;
  if (c >= 0xC0 && c <= 0xDE && c != 0xD7) return c + 0x20;
  if (c == 0x178) return 0xFF;
  if (c >= 0x100 && c <= 0x17F && c != 0x130 && c != 0x131 && c != 0x138 && c != 0x149 && c != 0x17F) {
    const bool pairEven = (c >= 0x100 && c <= 0x137) || (c >= 0x14A && c <= 0x177);
    if (pairEven) return (c & 1) ? c : c + 1;
    if ((c >= 0x139 && c <= 0x148) || (c >= 0x179 && c <= 0x17E)) return (c & 1) ? c + 1 : c;
  }
  if (c >= 0x391 && c <= 0x3A9 && c != 0x3A2) return c + 0x20;
  if (c >= 0x410 && c <= 0x42F) return c + 0x20;
  if (c >= 0x400 && c <= 0x40F) return c + 0x50;
  return c;
}

// String (and other host-only) scalar functions: evaluated on the CPU, per dictionary entry when
// the expression has a single string column (encoded execution), otherwise per result row.
// EMBED: texts -> VECTOR literals through the GPU model, with a process-wide cache (embedding the
// same text again costs a lookup). Misses are embedded in one batch.
inline void embedCached(const std::vector<std::string>& texts, std::vector<std::string>& out) {
  static std::mutex mu;
  static std::unordered_map<std::string, std::string> cache;
  out.assign(texts.size(), std::string());
  std::vector<size_t> miss;
  {
    std::lock_guard<std::mutex> g(mu);
    for (size_t i = 0; i < texts.size(); ++i) {
      auto it = cache.find(texts[i]);
      if (it != cache.end()) out[i] = it->second;
      else miss.push_back(i);
    }
  }
  if (miss.empty()) return;
  std::vector<std::string> todo;
  for (size_t i : miss) todo.push_back(texts[i]);
  std::vector<float> vecs;
  std::string err;
  const auto t0 = std::chrono::steady_clock::now();
  if (!embedBatch(todo, vecs, err)) throw SqlError("EMBED: " + err, 1105);
  if (envSetting("TRACE"))
    fprintf(stderr, "[trace] EMBED: %zu texts through the model in %.1f ms\n", todo.size(),
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
  const int d = embedDim();
  std::lock_guard<std::mutex> g(mu);
  if (cache.size() > 200000) cache.clear();
  for (size_t j = 0; j < miss.size(); ++j) {
    out[miss[j]] = formatVector(vecs.data() + j * d, d);
    cache[todo[j]] = out[miss[j]];
  }
}

// JSON text of a value: numbers bare, strings / dates quoted (MySQL's JSON rendering); raw: the value
// already is JSON text (a JSON column, JSON_ARRAY(..), ..) and is embedded as is.
inline std::string jsonQuoteStr(const std::string& x) {
  std::string o = "\"";
  for (unsigned char ch : x) {
    if (ch == '"' || ch == '\\') { o += '\\'; o += (char)ch; }
    else if (ch == '\n') o += "\\n";
    else if (ch == '\r') o += "\\r";
    else if (ch == '\t') o += "\\t";
    else if (ch < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", ch); o += b; }
    else o += (char)ch;
  }
  return o + "\"";
}
inline std::string jsonOfValue(const Value& v, bool raw) {
  if (v.isNull()) return "null";
  if (raw && v.kind == Value::STR) return v.s.empty() ? "null" : v.s;
  switch (v.kind) {
    case Value::INT: return std::to_string(v.i);
    case Value::DBL: {
      if (!std::isfinite(v.d)) return "null";
      std::string t = v.toString();
      return t;
    }
    case Value::DATE: case Value::DATETIME: return jsonQuoteStr(v.toString() + (v.kind == Value::DATETIME && v.toString().find('.') == std::string::npos ? ".000000" : ""));
    default: return jsonQuoteStr(v.toString());
  }
}

// Is x JSON-typed: a JSON column, a JSON-building function or a JSON aggregate? (ctx: to look up columns)
inline bool isJsonExpr(const ExprP& x, const BindCtx* c) {
  if (!x) return false;
  if (x->kind == Expr::FUNC) {
    static const std::set<std::string> fns = {"JSON_ARRAY", "JSON_OBJECT", "JSON_ARRAYAGG", "JSON_OBJECTAGG", "JSON_AGG",
                                              "JSON_EXTRACT", "JSON_EXTRACT_JSON", "JSON_SET", "JSON_MERGE_PATCH",
                                              "JSON_INSERT", "JSON_REPLACE", "JSON_REMOVE", "JSON_KEYS"};
    if (fns.count(upper(x->op))) return true;
    if (x->op.rfind("#ENC:", 0) == 0 && x->args.size() == 2) return isJsonExpr(x->args[1], c);
    if ((upper(x->op) == "COALESCE" || upper(x->op) == "IFNULL") && !x->args.empty()) return isJsonExpr(x->args[0], c);
    if (upper(x->op) == "CAST" && x->castType == "JSON") return true;
    return false;
  }
  if (x->kind == Expr::LIT) return x->castType == "JSON";  // (a folded JSON-building call)
  if (x->kind == Expr::COL && c && x->bt >= 0 && x->bt < (int)c->tabs.size() && x->bc >= 0 && x->bc < (int)c->tabs[x->bt]->cols.size())
    return lower(c->tabs[x->bt]->cols[x->bc].typeText).rfind("json", 0) == 0;
  return false;
}

// Full-text tokens: lower-cased runs of letters / digits (bytes >= 0x80 count as letters).
inline std::vector<std::string> ftsWords(std::string_view x) {
  std::vector<std::string> w;
  std::string cur;
  for (char ch : x) {
    if (isalnum((unsigned char)ch) || (unsigned char)ch >= 0x80) cur += (char)tolower((unsigned char)ch);
    else if (!cur.empty()) { w.push_back(cur); cur.clear(); }
  }
  if (!cur.empty()) w.push_back(cur);
  return w;
}
// Query terms of AGAINST('..'): (modifier '+' / '-' / 0, word, prefix match).
inline std::vector<std::tuple<char, std::string, bool>> ftsTerms(const std::string& qs, bool boolean) {
  std::vector<std::tuple<char, std::string, bool>> r;
  size_t i = 0;
  while (i < qs.size()) {
    while (i < qs.size() && isspace((unsigned char)qs[i])) ++i;
    char mod = 0;
    if (boolean && i < qs.size() && (qs[i] == '+' || qs[i] == '-')) mod = qs[i++];
    size_t j = i;
    while (j < qs.size() && !isspace((unsigned char)qs[j])) ++j;
    std::string term = qs.substr(i, j - i);
    i = j;
    bool prefix = !term.empty() && term.back() == '*';
    if (prefix) term.pop_back();
    auto tw = ftsWords(term);
    if (!tw.empty()) r.emplace_back(mod, tw[0], prefix);
  }
  return r;
}
// Inverted index of a VARCHAR column's dictionary (each distinct value is a document): term ->
// (code, term count) postings and per-code lengths, extended as the dictionary grows. MATCH ..
// AGAINST then scores only the postings of the query's terms instead of re-reading every document.
struct FtsIndex {
  std::mutex mu;
  size_t n = 0;  // dictionary entries indexed
  std::map<std::string, std::vector<std::pair<int32_t, uint32_t>>> post;
  std::vector<uint32_t> len;
  double total = 0;
  std::string first;  // the dictionary's first entry: detects a different dictionary at a reused address
};
inline FtsIndex& ftsIndexFor(const Dict& d) {
  static std::mutex mu;
  std::lock_guard<std::mutex> g(mu);
  auto p = d.getDerived<FtsIndex>("fts");
  if (!p) d.setDerived("fts", p = std::make_shared<FtsIndex>());
  return *p;  // lives as long as the dictionary
}
// Brings ix up to date with d (caller holds ix.mu; the table is locked, so d does not change).
inline void ftsIndexUpdate(FtsIndex& ix, const Dict& d) {
  if (ix.n > d.strs.size() || (ix.n > 0 && ix.first != d.strs[0])) {  // not the dictionary it was built for
    ix.n = 0;
    ix.post.clear();
    ix.len.clear();
    ix.total = 0;
  }
  if (ix.n == 0 && !d.strs.empty()) ix.first = d.strs[0];
  const size_t from = ix.n, n = d.strs.size();
  if (from >= n) return;
  const size_t nth = std::max<size_t>(1, std::min<size_t>(std::thread::hardware_concurrency(), (n - from) / 4096));
  std::vector<std::vector<std::pair<std::string, std::pair<int32_t, uint32_t>>>> parts(nth);
  std::vector<uint32_t> lens(n - from);
  parallelFor(nth, [&](size_t b, size_t e) {
    for (size_t t = b; t < e; ++t) {
      const size_t lo = from + (n - from) * t / nth, hi = from + (n - from) * (t + 1) / nth;
      std::unordered_map<std::string, uint32_t> tf;
      for (size_t i = lo; i < hi; ++i) {
        tf.clear();
        auto w = ftsWords(d.strs[i]);
        lens[i - from] = (uint32_t)w.size();
        for (auto& x : w) ++tf[x];
        for (auto& [term, cnt] : tf) parts[t].push_back({term, {(int32_t)i, cnt}});
      }
    }
  }, 1);
  for (auto& part : parts)  // in code order, so posting lists stay sorted
    for (auto& [term, pc] : part) ix.post[term].push_back(pc);
  for (uint32_t l : lens) ix.total += l;
  ix.len.insert(ix.len.end(), lens.begin(), lens.end());
  ix.n = n;
}
inline void ftsWarm(const Dict& d) {
  FtsIndex& ix = ftsIndexFor(d);
  std::lock_guard<std::mutex> g(ix.mu);
  ftsIndexUpdate(ix, d);
}
inline const bool ftsWarmRegistered = (gFtsWarm = ftsWarm, true);
// MATCH (col) AGAINST ('..' [IN BOOLEAN MODE]) for every code of col's dictionary, from the index
// (the same BM25 as the per-document evaluation).
inline bool ftsMapFromIndex(const ExprP& e, const Dict& d, std::vector<double>& map) {
  if (!(e->kind == Expr::FUNC && e->op == "MATCH_AGAINST" && e->args.size() == 2 && e->args[1]->kind == Expr::LIT)) return false;
  const bool boolean = e->castType == "BOOLEAN";
  const auto terms = ftsTerms(e->args[1]->lit.toString(), boolean);
  FtsIndex& ix = ftsIndexFor(d);
  std::lock_guard<std::mutex> g(ix.mu);
  ftsIndexUpdate(ix, d);
  const size_t N = ix.n;
  map.assign(d.strs.size(), 0.0);
  if (!N) return true;
  const double avgdl = std::max(1.0, ix.total / (double)N), k1 = 1.2, b = 0.75;
  std::vector<uint8_t> veto(boolean ? N : 0, 0);  // boolean mode: excluded by -term or missing a +term
  auto docsOf = [&](const std::string& term, bool prefix, std::unordered_map<int32_t, uint32_t>& tf) {
    if (!prefix) {
      auto it = ix.post.find(term);
      if (it != ix.post.end())
        for (auto& [code, cnt] : it->second) tf[code] += cnt;
      return;
    }
    for (auto it = ix.post.lower_bound(term); it != ix.post.end() && it->first.rfind(term, 0) == 0; ++it)
      for (auto& [code, cnt] : it->second) tf[code] += cnt;
  };
  for (auto& [mod, term, prefix] : terms) {
    std::unordered_map<int32_t, uint32_t> tf;
    docsOf(term, prefix, tf);
    if (mod == '-') {
      for (auto& [code, cnt] : tf) veto[code] = 1;
      continue;
    }
    if (mod == '+') {
      std::vector<uint8_t> has(N, 0);
      for (auto& [code, cnt] : tf) has[code] = 1;
      for (size_t i = 0; i < N; ++i)
        if (!has[i]) veto[i] = 1;
    }
    const double df = (double)tf.size();
    const double idf = std::log(1 + ((double)N - df + 0.5) / (df + 0.5));
    for (auto& [code, cnt] : tf)
      map[code] += idf * cnt / (cnt + k1 * (1 - b + b * ix.len[code] / avgdl));
  }
  for (size_t i = 0; i < veto.size(); ++i)
    if (veto[i]) map[i] = 0;
  return true;
}

// BM25 corpus statistics for the MATCH .. AGAINST nodes of e over one column's dictionary: N,
// average document length, df per query term (stored in the node's aux).
inline void ftsStats(const ExprP& e, const Dict& d) {
  for (auto& x : e->args) ftsStats(x, d);
  if (!(e->kind == Expr::FUNC && e->op == "MATCH_AGAINST" && e->args.size() == 2 && e->args.back()->kind == Expr::LIT)) return;
  const auto terms = ftsTerms(e->args.back()->lit.toString(), e->castType == "BOOLEAN");
  const size_t N = d.strs.size(), T = terms.size();
  // Cached per dictionary size and query (dictionaries only grow).
  static std::mutex cmu;
  static std::unordered_map<std::string, std::shared_ptr<std::vector<double>>> cache;
  const std::string key = std::to_string(d.uid) + "|" + std::to_string(N) + "|" + e->castType + "|" + e->args.back()->lit.toString();
  {
    std::lock_guard<std::mutex> g(cmu);
    auto it = cache.find(key);
    if (it != cache.end()) { e->aux = it->second; return; }
  }
  std::vector<double> sum(T + 1, 0);
  std::mutex mu;
  parallelFor(N, [&](size_t b, size_t en) {
    std::vector<double> p(T + 1, 0);
    for (size_t i = b; i < en; ++i) {
      auto w = ftsWords(d.strs[i]);
      p[T] += w.size();
      std::sort(w.begin(), w.end());
      for (size_t t = 0; t < T; ++t) {
        auto& [mod, term, prefix] = terms[t];
        auto it = std::lower_bound(w.begin(), w.end(), term);
        if (it != w.end() && (prefix ? it->rfind(term, 0) == 0 : *it == term)) p[t] += 1;
      }
    }
    std::lock_guard<std::mutex> g(mu);
    for (size_t t = 0; t <= T; ++t) sum[t] += p[t];
  });
  auto st = std::make_shared<std::vector<double>>(2 + T, 0.0);
  (*st)[0] = (double)N;
  (*st)[1] = sum[T] / std::max<size_t>(1, N);
  for (size_t t = 0; t < T; ++t) (*st)[2 + t] = sum[t];
  e->aux = st;
  std::lock_guard<std::mutex> g(cmu);
  if (cache.size() > 4096) cache.clear();
  cache[key] = st;
}

inline Value evalHostFunc(const ExprP& e, const PostCtx& c) {
  const std::string& f = e->op;
  std::vector<Value> a;
  for (auto& x : e->args) a.push_back(evalPost(x, c));
  auto str = [&](size_t i) { return a[i].toString(); };
  auto num = [&](size_t i) { return numOf(a[i]); };
  if (f == "#UDF") {  // run the function body once per distinct argument tuple (memoized per call site)
    std::vector<std::string> lits;
    std::string key = str(1);
    for (size_t i = 2; i < a.size(); ++i) {
      lits.push_back(udfLiteral(a[i]));
      key += '\x1f' + lits.back();
    }
    thread_local std::unordered_map<std::string, Value> memo;
    if (auto it = memo.find(key); it != memo.end()) return it->second;
    if (memo.size() > (1u << 20)) memo.clear();
    Value v = callUdf(str(0), lits);
    if (!v.isNull() && v.kind == Value::STR && (e->ty == LType::DATE || e->ty == LType::DATETIME)) {
      ExprP l = boundLit(v);
      v = coerceTemporal(l, e->ty)->lit;
    }
    return memo[key] = v;
  }
  auto anyNull = [&]() {
    for (auto& v : a)
      if (v.isNull()) return true;
    return false;
  };
  if (f == "CONCAT_WS") {
    if (a.empty() || a[0].isNull()) return Value::Null();
    std::string r;
    bool first = true;
    for (size_t i = 1; i < a.size(); ++i) {
      if (a[i].isNull()) continue;
      if (!first) r += str(0);
      r += str(i);
      first = false;
    }
    return Value::S(r);
  }
  if (f == "ELT") {
    if (a.empty() || a[0].isNull()) return Value::Null();
    int64_t k = (int64_t)num(0);
    return k >= 1 && k < (int64_t)a.size() ? a[k] : Value::Null();
  }
  if (f == "FIELD") {
    if (a.empty() || a[0].isNull()) return Value::I(0);
    for (size_t i = 1; i < a.size(); ++i)
      if (!a[i].isNull() && cmpVal(a[0], a[i]) == 0) return Value::I((int64_t)i);
    return Value::I(0);
  }
  if (f == "UUID") {  // version 4 (random), from the statement's generator
    const uint64_t hi = tStmtRng(), lo = tStmtRng();
    char b[40];
    snprintf(b, sizeof b, "%08llx-%04llx-4%03llx-%04llx-%012llx", (unsigned long long)(hi >> 32), (unsigned long long)((hi >> 16) & 0xffff),
             (unsigned long long)(hi & 0xfff), (unsigned long long)(((lo >> 48) & 0x3fff) | 0x8000), (unsigned long long)(lo & 0xffffffffffffull));
    return Value::S(b);
  }
  if (f == "JSON_ARRAY" || f == "JSON_OBJECT") {  // e->name: per argument '1' = JSON-typed (embedded raw)
    const bool obj = f == "JSON_OBJECT";
    std::string o = obj ? "{" : "[";
    for (size_t i = 0; i < a.size(); ++i) {
      if (i) o += obj && (i & 1) ? ": " : ", ";
      if (obj && !(i & 1)) {
        if (a[i].isNull()) throw SqlError("JSON documents may not contain NULL member names", 3158);
        o += jsonQuoteStr(a[i].toString());
      } else {
        o += jsonOfValue(a[i], i < e->name.size() && e->name[i] == '1');
      }
    }
    return Value::S(o + (obj ? "}" : "]"));
  }
  if (f == "JSON_SET" || f == "JSON_INSERT" || f == "JSON_REPLACE" || f == "JSON_REMOVE") {
    if (a[0].isNull()) return Value::Null();
    const int mode = f == "JSON_SET" ? 0 : f == "JSON_INSERT" ? 1 : f == "JSON_REPLACE" ? 2 : 3;
    std::string js = a[0].toString();
    for (size_t i = 1; i < a.size(); i += mode == 3 ? 1 : 2) {
      if (a[i].isNull()) return Value::Null();
      const std::string path = a[i].toString();
      if (path.rfind("$", 0) != 0) throw SqlError("Invalid JSON path expression", 3143);
      const std::string val = mode == 3 || i + 1 >= a.size() ? "" : jsonOfValue(a[i + 1], i + 1 < e->name.size() && e->name[i + 1] == '1');
      js = jsonEdit(js, jsonPathOf(path), 0, val, mode);
    }
    return Value::S(js);
  }
  if (f == "BIN" || f == "OCT" || f == "CONV") {  // CONV(n, from_base, to_base); negative to_base: signed
    for (auto& x : a)
      if (x.isNull()) return Value::Null();
    int from = 10, to = f == "BIN" ? 2 : 8;
    if (f == "CONV") from = (int)num(1), to = (int)num(2);
    const bool signedOut = to < 0;
    from = std::abs(from), to = std::abs(to);
    if (from < 2 || from > 36 || to < 2 || to > 36) return Value::Null();
    std::string t = a[0].kind == Value::DBL && f != "CONV" ? std::to_string((int64_t)num(0)) : a[0].toString();
    size_t i = 0;
    while (i < t.size() && isspace((unsigned char)t[i])) ++i;
    bool neg = i < t.size() && t[i] == '-';
    i += neg;
    uint64_t v = 0;
    for (; i < t.size(); ++i) {
      const char ch = (char)tolower(t[i]);
      const int dg = isdigit((unsigned char)ch) ? ch - '0' : ch >= 'a' && ch <= 'z' ? ch - 'a' + 10 : 99;
      if (dg >= from) break;
      v = v * from + dg;
    }
    if (neg && !signedOut) v = (uint64_t)(-(int64_t)v);
    std::string r;
    do r.insert(r.begin(), "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"[v % to]), v /= to; while (v);
    return Value::S((neg && signedOut ? "-" : "") + r);
  }
  if (f == "ORD") {  // the first character's bytes, as a number
    if (a[0].isNull()) return Value::Null();
    const std::string t = a[0].toString();
    if (t.empty()) return Value::I(0);
    const unsigned char c0 = t[0];
    const size_t n = c0 < 0x80 ? 1 : c0 < 0xE0 ? 2 : c0 < 0xF0 ? 3 : 4;
    int64_t r = 0;
    for (size_t i = 0; i < n && i < t.size(); ++i) r = r * 256 + (unsigned char)t[i];
    return Value::I(r);
  }
  if (f == "JSON_QUOTE") return a[0].isNull() ? Value::Null() : Value::S(jsonQuoteStr(a[0].toString()));
  if (f == "EMBED_TOKENS") {
    if (a.empty() || a[0].isNull()) return Value::Null();
    std::string o;
    for (int t : embedTokens(a[0].toString())) o += (o.empty() ? "" : ",") + std::to_string(t);
    return Value::S(o);
  }
  if (f == "EMBED") {  // EMBED(text): the sentence embedding as a VECTOR literal (GPU model, cached)
    if (a.empty() || a[0].isNull()) return Value::Null();
    std::vector<std::string> one{a[0].toString()};
    std::vector<std::string> vec;
    embedCached(one, vec);
    return Value::S(vec[0]);
  }
  if (anyNull()) return Value::Null();
  if (f == "CONCAT") {
    std::string r;
    for (size_t i = 0; i < a.size(); ++i) r += str(i);
    return Value::S(r);
  }
  if (f == "UPPER" || f == "UCASE" || f == "LOWER" || f == "LCASE") {
    const bool up = f == "UPPER" || f == "UCASE";
    std::string r = str(0);
    if (asciiOnly(r)) {
      for (auto& ch : r) ch = (char)(up ? toupper((unsigned char)ch) : tolower((unsigned char)ch));
      return Value::S(r);
    }
    std::vector<uint32_t> cps;
    u8Decode(r, cps);
    std::string o;
    o.reserve(r.size());
    for (uint32_t cp : cps) u8Append(o, up ? cpUpper(cp) : cpLower(cp));
    return Value::S(o);
  }
  if (f == "LENGTH" || f == "OCTET_LENGTH") return Value::I((int64_t)str(0).size());
  if (f == "CHAR_LENGTH" || f == "CHARACTER_LENGTH") return Value::I((int64_t)utf8Len(str(0)));
  if (f == "SUBSTRING" || f == "SUBSTR" || f == "MID") {  // (characters, not bytes)
    std::string s0 = str(0);
    const int64_t n = asciiOnly(s0) ? (int64_t)s0.size() : (int64_t)utf8Len(s0);
    int64_t pos = (int64_t)num(1), len = a.size() > 2 ? (int64_t)num(2) : n;
    int64_t start = pos > 0 ? pos - 1 : pos < 0 ? n + pos : n;
    if (start < 0 || start >= n || len <= 0) return Value::S("");
    return Value::S(u8Sub(s0, start, len));
  }
  if (f == "LEFT") { int64_t k = (int64_t)num(1); return Value::S(k <= 0 ? "" : u8Sub(str(0), 0, k)); }
  if (f == "RIGHT") {
    std::string s0 = str(0);
    const int64_t n = asciiOnly(s0) ? (int64_t)s0.size() : (int64_t)utf8Len(s0);
    int64_t k = std::min<int64_t>((int64_t)num(1), n);
    return Value::S(k <= 0 ? "" : u8Sub(s0, n - k, k));
  }
  if (f == "TRIM" || f == "LTRIM" || f == "RTRIM") {
    std::string s0 = str(0);
    size_t b = 0, en = s0.size();
    if (f != "RTRIM") while (b < en && s0[b] == ' ') ++b;
    if (f != "LTRIM") while (en > b && s0[en - 1] == ' ') --en;
    return Value::S(s0.substr(b, en - b));
  }
  if (f == "REPLACE") {
    std::string s0 = str(0), from = str(1), to = str(2);
    if (from.empty()) return Value::S(s0);
    std::string r;
    for (size_t i = 0;;) {
      size_t j = s0.find(from, i);
      if (j == std::string::npos) { r += s0.substr(i); break; }
      r += s0.substr(i, j - i) + to;
      i = j + from.size();
    }
    return Value::S(r);
  }
  if (f == "REVERSE") {
    std::string r = str(0);
    if (asciiOnly(r)) {
      std::reverse(r.begin(), r.end());
      return Value::S(r);
    }
    std::vector<uint32_t> cps;
    u8Decode(r, cps);
    std::string o;
    for (auto it = cps.rbegin(); it != cps.rend(); ++it) u8Append(o, *it);
    return Value::S(o);
  }
  if (f == "LPAD" || f == "RPAD") {  // (lengths in characters)
    std::string s0 = str(0), pad = a.size() > 2 ? str(2) : " ";
    int64_t n = (int64_t)num(1);
    const int64_t len0 = (int64_t)utf8Len(s0);
    if (n <= len0) return Value::S(u8Sub(s0, 0, std::max<int64_t>(n, 0)));
    if (pad.empty() || n > (1 << 24)) return Value::Null();
    std::string fill;
    int64_t have = 0;
    const int64_t padLen = (int64_t)utf8Len(pad);
    while (have + padLen <= n - len0) fill += pad, have += padLen;
    fill += u8Sub(pad, 0, n - len0 - have);
    return Value::S(f == "LPAD" ? fill + s0 : s0 + fill);
  }
  if (f == "REPEAT") {
    std::string r, s0 = str(0);
    for (int64_t i = 0; i < (int64_t)num(1) && r.size() < (1u << 24); ++i) r += s0;
    return Value::S(r);
  }
  if (f == "SPACE") return Value::S(std::string((size_t)std::max<double>(0, std::min<double>(num(0), 1 << 20)), ' '));
  if (f == "LOCATE" || f == "POSITION") {  // (character positions)
    std::string sub = str(0), s0 = str(1);
    const int64_t from = a.size() > 2 ? (int64_t)std::max<double>(1, num(2)) - 1 : 0;
    size_t j = s0.find(sub, u8Off(s0, from));
    return Value::I(j == std::string::npos ? 0 : (int64_t)utf8Len(s0.substr(0, j)) + 1);
  }
  if (f == "INSTR") {
    const std::string s0 = str(0);
    size_t j = s0.find(str(1));
    return Value::I(j == std::string::npos ? 0 : (int64_t)utf8Len(s0.substr(0, j)) + 1);
  }
  if (f == "ASCII") { std::string s0 = str(0); return Value::I(s0.empty() ? 0 : (unsigned char)s0[0]); }
  if (f == "STRCMP") { int r = str(0).compare(str(1)); return Value::I(r < 0 ? -1 : r > 0 ? 1 : 0); }
  if (f == "SUBSTRING_INDEX") {
    std::string s0 = str(0), d = str(1);
    int64_t k = (int64_t)num(2);
    if (d.empty() || k == 0) return Value::S("");
    if (k > 0) {
      size_t pos = 0;
      for (int64_t i = 0; i < k; ++i) {
        pos = s0.find(d, i ? pos + d.size() : 0);
        if (pos == std::string::npos) return Value::S(s0);
      }
      return Value::S(s0.substr(0, pos));
    }
    size_t pos = s0.size();
    for (int64_t i = 0; i < -k; ++i) {
      if (pos == 0) return Value::S(s0);
      pos = s0.rfind(d, pos - 1);
      if (pos == std::string::npos) return Value::S(s0);
    }
    return Value::S(s0.substr(pos + d.size()));
  }
  if (f == "FORMAT") {
    char b[64];
    snprintf(b, sizeof b, "%.*f", (int)std::max<double>(0, a.size() > 1 ? num(1) : 0), num(0));
    std::string t = b, r;
    size_t dot = t.find('.');
    std::string ip = t.substr(0, dot), fp = dot == std::string::npos ? "" : t.substr(dot);
    bool neg = !ip.empty() && ip[0] == '-';
    if (neg) ip = ip.substr(1);
    for (size_t i = 0; i < ip.size(); ++i) {
      if (i && (ip.size() - i) % 3 == 0) r += ',';
      r += ip[i];
    }
    return Value::S((neg ? "-" : "") + r + fp);
  }
  if (f == "DATE_FORMAT" || f == "TIME_FORMAT") return Value::S(dateFormat(a[0], str(1)));
  if (f == "DAYNAME") return Value::S(dateFormat(a[0], "%W"));
  if (f == "MONTHNAME") return Value::S(dateFormat(a[0], "%M"));
  if (f == "CASTCHAR") return Value::S(str(0));
  if (f == "CASTNUM" || f == "CASTINT") {  // string -> number, MySQL style: the leading numeric prefix, else 0
    if (a[0].isNull()) return Value::Null();
    const std::string s0 = a[0].kind == Value::STR ? a[0].s : a[0].toString();
    if (f == "CASTINT") return Value::I(std::strtoll(s0.c_str(), nullptr, 10));
    return Value::D(std::strtod(s0.c_str(), nullptr));
  }
  if (f == "GREATEST" || f == "LEAST") {  // (string arguments)
    for (auto& x : a)
      if (x.isNull()) return Value::Null();
    Value best = a[0];
    for (size_t i = 1; i < a.size(); ++i) {
      const int cmp = collCmp(a[i].toString(), best.toString());
      if (f == "GREATEST" ? cmp > 0 : cmp < 0) best = a[i];
    }
    return Value::S(best.toString());
  }
  if (f == "BITAND" || f == "BITOR" || f == "BITXOR" || f == "SHL" || f == "SHR" || f == "BITNOT") {
    for (auto& x : a)
      if (x.isNull()) return Value::Null();
    const uint64_t x = (uint64_t)(int64_t)std::llround(num(0)), y = a.size() > 1 ? (uint64_t)(int64_t)std::llround(num(1)) : 0;
    const uint64_t r = f == "BITAND" ? x & y : f == "BITOR" ? x | y : f == "BITXOR" ? x ^ y : f == "SHL" ? (y < 64 ? x << y : 0)
                     : f == "SHR" ? (y < 64 ? x >> y : 0) : ~x;
    return Value::I((int64_t)r);
  }
  if (f == "FIND_IN_SET") {
    if (a[0].isNull() || a[1].isNull()) return Value::Null();
    const std::string needle = a[0].toString(), list = a[1].toString();
    if (needle.find(',') != std::string::npos) return Value::I(0);
    int64_t pos = 1;
    for (size_t b = 0;; ++pos) {
      const size_t e2 = list.find(',', b);
      if (collCmp(list.substr(b, e2 == std::string::npos ? std::string::npos : e2 - b), needle) == 0) return Value::I(list.empty() ? 0 : pos);
      if (e2 == std::string::npos) return Value::I(0);
      b = e2 + 1;
    }
  }
  if (f == "UNHEX") {
    if (a[0].isNull()) return Value::Null();
    const std::string h = a[0].toString();
    std::string r;
    auto hv = [](char ch) { return ch >= '0' && ch <= '9' ? ch - '0' : ch >= 'a' && ch <= 'f' ? ch - 'a' + 10 : ch >= 'A' && ch <= 'F' ? ch - 'A' + 10 : -1; };
    size_t i = h.size() % 2;
    if (i && hv(h[0]) < 0) return Value::Null();
    if (i) r += (char)hv(h[0]);
    for (; i + 1 < h.size(); i += 2) {
      const int x = hv(h[i]), y = hv(h[i + 1]);
      if (x < 0 || y < 0) return Value::Null();
      r += (char)(x * 16 + y);
    }
    return Value::S(r);
  }
  if (f == "INSERT") {  // INSERT(str, pos, len, newstr): characters
    for (auto& x : a)
      if (x.isNull()) return Value::Null();
    const std::string s0 = a[0].toString();
    const int64_t n = (int64_t)utf8Len(s0), pos = (int64_t)num(1), len = (int64_t)num(2);
    if (pos < 1 || pos > n) return Value::S(s0);
    return Value::S(u8Sub(s0, 0, pos - 1) + a[3].toString() + u8Sub(s0, pos - 1 + std::max<int64_t>(len, 0), n));
  }
  if (f == "STR_TO_DATE") {
    if (a[0].isNull() || a[1].isNull()) return Value::Null();
    const std::string str = a[0].toString(), fm = a[1].toString();
    static const char* mon[] = {"jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"};
    int64_t Y = 0, M = 0, D = 0, h = 0, mi = 0, sec = 0, us = 0;
    bool pm = false, ampm = false, hasTime = false;
    size_t i = 0;
    auto num2 = [&](int maxd, int64_t& out) {
      size_t k = 0;
      int64_t v = 0;
      while (i < str.size() && k < (size_t)maxd && isdigit((unsigned char)str[i])) v = v * 10 + (str[i++] - '0'), ++k;
      out = v;
      return k > 0;
    };
    for (size_t j = 0; j < fm.size(); ++j) {
      if (fm[j] != '%' || j + 1 >= fm.size()) {
        if (i >= str.size() || (str[i] != fm[j] && !(isspace((unsigned char)fm[j]) && isspace((unsigned char)str[i])))) return Value::Null();
        ++i;
        continue;
      }
      const char k = fm[++j];
      bool ok = true;
      switch (k) {
        case 'Y': ok = num2(4, Y); break;
        case 'y': ok = num2(2, Y); Y += Y < 70 ? 2000 : 1900; break;
        case 'm': case 'c': ok = num2(2, M); break;
        case 'd': case 'e': ok = num2(2, D); break;
        case 'H': case 'k': ok = num2(2, h); hasTime = true; break;
        case 'h': case 'I': case 'l': ok = num2(2, h); ampm = hasTime = true; break;
        case 'i': ok = num2(2, mi); hasTime = true; break;
        case 's': case 'S': ok = num2(2, sec); hasTime = true; break;
        case 'f': {
          size_t st = i;
          ok = num2(6, us);
          for (size_t z = i - st; z < 6; ++z) us *= 10;
          break;
        }
        case 'p':
          if (i + 1 < str.size() && (toupper(str[i]) == 'A' || toupper(str[i]) == 'P') && toupper(str[i + 1]) == 'M') pm = toupper(str[i]) == 'P', i += 2;
          else ok = false;
          break;
        case 'b': case 'M': {
          ok = false;
          for (int q = 0; q < 12 && !ok; ++q)
            if (i + 3 <= str.size() && strncasecmp(str.c_str() + i, mon[q], 3) == 0) {
              M = q + 1;
              i += 3;
              while (k == 'M' && i < str.size() && isalpha((unsigned char)str[i])) ++i;
              ok = true;
            }
          break;
        }
        case '%': ok = i < str.size() && str[i++] == '%'; break;
        default: return Value::Null();
      }
      if (!ok) return Value::Null();
    }
    if (ampm) h = h % 12 + (pm ? 12 : 0);
    if (M < 1 || M > 12 || D < 1 || D > 31) return Value::Null();
    const int64_t days = daysFromCivil(Y, (unsigned)M, (unsigned)D);
    if (e->ty == LType::DATE) return Value::Dt(days);
    (void)hasTime;
    return Value::Ts(days * kUsPerDay + ((h * 60 + mi) * 60 + sec) * 1000000 + us);
  }
  if (f == "TIMEDIFF" || f == "SEC_TO_TIME" || f == "TIME" || f == "TIME_TO_SEC" || f == "MAKETIME") {
    for (auto& x : a)
      if (x.isNull()) return Value::Null();
    // Times as signed microseconds: a DATETIME/DATE, 'YYYY-MM-DD ..' or '[-]H:MM[:SS[.f]]'.
    auto us = [&](const Value& v, bool timeOfDay, int64_t& out) {
      if (v.kind == Value::DATETIME) { out = v.i; }
      else if (v.kind == Value::DATE) { out = v.i * kUsPerDay; }
      else {
        const std::string t = v.toString();
        if (parseDateTime(t, out)) {
        } else {
          size_t i = 0;
          bool neg = i < t.size() && t[i] == '-';
          i += neg;
          int64_t part[3] = {0, 0, 0}, frac = 0;
          int k = 0;
          bool any = false;
          for (; i < t.size() && k < 3; ++i) {
            if (isdigit((unsigned char)t[i])) part[k] = part[k] * 10 + (t[i] - '0'), any = true;
            else if (t[i] == ':') ++k;
            else break;
          }
          if (!any) return false;
          if (i < t.size() && t[i] == '.') {
            int dg = 0;
            for (++i; i < t.size() && isdigit((unsigned char)t[i]) && dg < 6; ++i, ++dg) frac = frac * 10 + (t[i] - '0');
            for (; dg < 6; ++dg) frac *= 10;
          }
          out = ((part[0] * 60 + part[1]) * 60 + part[2]) * 1000000 + frac;
          if (neg) out = -out;
          return true;
        }
      }
      if (timeOfDay) out = ((out % kUsPerDay) + kUsPerDay) % kUsPerDay;
      return true;
    };
    auto fmt = [](int64_t u) {
      std::string r = u < 0 ? "-" : "";
      u = std::llabs(u);
      char b[48];
      snprintf(b, sizeof b, "%02lld:%02lld:%02lld", (long long)(u / 3600000000LL), (long long)(u / 60000000 % 60), (long long)(u / 1000000 % 60));
      r += b;
      if (u % 1000000) {
        snprintf(b, sizeof b, ".%06lld", (long long)(u % 1000000));
        r += b;
      }
      return Value::S(r);
    };
    int64_t x = 0, y = 0;
    if (f == "SEC_TO_TIME") return fmt((int64_t)std::llround(num(0) * 1e6));
    if (f == "MAKETIME") return fmt((((int64_t)num(0) * 60 + (int64_t)num(1)) * 60) * 1000000 + (int64_t)std::llround(num(2) * 1e6));
    const bool dt0 = a[0].kind == Value::DATETIME || a[0].kind == Value::DATE;
    if (!us(a[0], f != "TIMEDIFF" && (dt0 || a[0].toString().find('-', 1) != std::string::npos), x)) return Value::Null();
    if (f == "TIME") return fmt(x);
    if (f == "TIME_TO_SEC") return Value::I(x / 1000000);
    if (!us(a[1], false, y)) return Value::Null();
    return fmt(x - y);
  }
  if (f == "MAKEDATE") {
    if (a[0].isNull() || a[1].isNull() || num(1) < 1) return Value::Null();
    return Value::Dt(daysFromCivil((int64_t)num(0), 1, 1) + (int64_t)num(1) - 1);
  }
  if (f == "CHAR") {
    std::string r;
    for (auto& x : a) {
      if (x.isNull()) continue;
      uint64_t v = (uint64_t)(int64_t)x.num();
      std::string bytes;
      do bytes.insert(bytes.begin(), (char)(v & 0xff)), v >>= 8; while (v);
      r += bytes;
    }
    return Value::S(r);
  }
  if (f == "BIT_LENGTH") return a[0].isNull() ? Value::Null() : Value::I((int64_t)a[0].toString().size() * 8);
  if (f == "INET_ATON" || f == "IS_IPV4") {
    if (a[0].isNull()) return f == "IS_IPV4" ? Value::I(0) : Value::Null();
    const std::string t = a[0].toString();
    uint64_t r = 0, part = 0;
    int dots = 0, digits = 0;
    bool ok = !t.empty();
    for (char ch : t) {
      if (isdigit((unsigned char)ch)) { part = part * 10 + (ch - '0'); ++digits; ok = ok && part < 256 && digits <= 3; }
      else if (ch == '.' && digits) { r = r * 256 + part; part = 0; digits = 0; ++dots; }
      else ok = false;
    }
    ok = ok && digits && dots <= 3;
    if (f == "IS_IPV4") return Value::I(ok && dots == 3);
    if (!ok) return Value::Null();
    r = r * 256 + part;
    if (dots < 3) {  // MySQL: '127.1' = 127.0.0.1
      const uint64_t last = part, head = r / 256;
      r = (head << (8 * (4 - dots))) | last;
    }
    return Value::I((int64_t)r);
  }
  if (f == "INET_NTOA") {
    if (a[0].isNull() || num(0) < 0 || num(0) > 4294967295.0) return Value::Null();
    const uint64_t v = (uint64_t)num(0);
    char b[20];
    snprintf(b, sizeof b, "%u.%u.%u.%u", (unsigned)(v >> 24), (unsigned)(v >> 16 & 255), (unsigned)(v >> 8 & 255), (unsigned)(v & 255));
    return Value::S(b);
  }
  if (f == "JSON_VALID" || f == "JSON_TYPE" || f == "JSON_KEYS" || f == "JSON_CONTAINS_PATH") {
    if (a[0].isNull()) return Value::Null();
    const std::string js = a[0].toString();
    const size_t b = jsonWs(js, 0);
    const size_t end = b < js.size() ? jsonSkip(js, b) : b;
    const bool valid = b < js.size() && end > b && jsonWs(js, end) == js.size();
    if (f == "JSON_VALID") return Value::I(valid);
    if (!valid) throw SqlError("Invalid JSON text in argument 1 to function " + lower(f), 3141);
    if (f == "JSON_CONTAINS_PATH") {
      if (a.size() < 3) throw SqlError("Incorrect parameter count in the call to native function 'JSON_CONTAINS_PATH'", 1582);
      const std::string mode = lower(a[1].toString());
      const bool all = mode == "all";
      if (!all && mode != "one") throw SqlError("The oneOrAll argument to json_contains_path may take these values: 'one' or 'all'.", 3154);
      for (size_t i = 2; i < a.size(); ++i) {
        std::string out;
        const bool has = jsonFind(js, jsonPathOf(a[i].toString()), out);
        if (has && !all) return Value::I(1);
        if (!has && all) return Value::I(0);
      }
      return Value::I(all);
    }
    std::string v = js;
    if (a.size() > 1 && !jsonFind(js, jsonPathOf(a[1].toString()), v)) return Value::Null();
    const size_t vb = jsonWs(v, 0);
    const char c0 = vb < v.size() ? v[vb] : 0;
    if (f == "JSON_TYPE") {
      if (c0 == '{') return Value::S("OBJECT");
      if (c0 == '[') return Value::S("ARRAY");
      if (c0 == '"') return Value::S("STRING");
      if (c0 == 't' || c0 == 'f') return Value::S("BOOLEAN");
      if (c0 == 'n') return Value::S("NULL");
      const std::string num = v.substr(vb, jsonSkip(v, vb) - vb);
      return Value::S(num.find_first_of(".eE") == std::string::npos ? "INTEGER" : "DOUBLE");
    }
    if (c0 != '{') return Value::Null();  // JSON_KEYS: the keys of an object
    std::string r = "[";
    size_t i = vb + 1;
    for (;;) {
      i = jsonWs(v, i);
      if (i >= v.size() || v[i] == '}') break;
      if (v[i] == ',') { ++i; continue; }
      const size_t ke = jsonSkip(v, i);
      if (r.size() > 1) r += ", ";
      r += v.substr(i, ke - i);
      i = jsonWs(v, ke);
      if (i < v.size() && v[i] == ':') ++i;
      i = jsonSkip(v, jsonWs(v, i));
    }
    return Value::S(r + "]");
  }
  if (f == "HEX") {
    std::string r;
    char b[4];
    if (a[0].kind == Value::STR) {
      for (unsigned char ch : a[0].s) { snprintf(b, sizeof b, "%02X", ch); r += b; }
    } else {
      char h[32];
      snprintf(h, sizeof h, "%llX", (unsigned long long)(int64_t)num(0));
      r = h;
    }
    return Value::S(r);
  }
  if (f.rfind("JSON_EXTRACT", 0) == 0 || f == "JSON_LENGTH" || f == "JSON_UNQUOTE") {
    std::string js = str(0), out;
    if (f == "JSON_UNQUOTE") return Value::S(jsonUnquote(js));
    std::vector<std::string> path;
    if (f == "JSON_EXTRACT" || (a.size() == 2 && str(1).rfind("$", 0) == 0)) path = jsonPathOf(a.size() > 1 ? str(1) : "$");
    else for (size_t i = 1; i < a.size(); ++i) path.push_back(str(i));
    if (!jsonFind(js, path, out)) return f == "JSON_LENGTH" ? Value::Null() : Value::Null();
    if (f == "JSON_LENGTH") {
      size_t b = jsonWs(out, 0);
      if (b >= out.size() || (out[b] != '[' && out[b] != '{')) return Value::I(1);
      int64_t n = 0;
      size_t i = b + 1;
      for (;;) {
        i = jsonWs(out, i);
        if (i >= out.size() || out[i] == ']' || out[i] == '}') break;
        if (out[i] == ',') { ++i; continue; }
        if (out[b] == '{') { i = jsonSkip(out, i); i = jsonWs(out, i); if (i < out.size() && out[i] == ':') ++i; }
        i = jsonSkip(out, i);
        ++n;
      }
      return Value::I(n);
    }
    if (out == "null") return Value::Null();
    if (f == "JSON_EXTRACT_DOUBLE") return Value::D(std::atof(jsonUnquote(out).c_str()));
    if (f == "JSON_EXTRACT_BIGINT") {
      std::string u = jsonUnquote(out);
      return Value::I(u == "true" ? 1 : u == "false" ? 0 : (int64_t)std::atof(u.c_str()));
    }
    if (f == "JSON_EXTRACT_STRING") return Value::S(jsonUnquote(out));
    return Value::S(out);  // JSON_EXTRACT / JSON_EXTRACT_JSON: JSON text
  }
  if (f == "MATCH_AGAINST") {  // full-text relevance: BM25 (boolean mode: +must, -must not, prefix*)
    std::string text;
    for (size_t i = 0; i + 1 < a.size(); ++i) text += str(i) + " ";
    std::map<std::string, int> doc;
    const auto dw = ftsWords(text);
    for (auto& w : dw) ++doc[w];
    const bool boolean = e->castType == "BOOLEAN";
    // Corpus statistics (set by encodeStrings: N, average length, then df per query term); without
    // them every term counts as rare and the document as average length.
    const std::vector<double>* st = e->aux && e->aux->size() >= 2 ? e->aux.get() : nullptr;
    const double N = st ? (*st)[0] : 1e6, avgdl = st ? std::max(1.0, (*st)[1]) : std::max<double>(1, dw.size());
    const double k1 = 1.2, b = 0.75;
    double score = 0;
    int ti = 0;
    for (auto& [mod, term, prefix] : ftsTerms(str(a.size() - 1), boolean)) {
      const int idx = ti++;
      int tf = 0;
      if (prefix) {
        for (auto it = doc.lower_bound(term); it != doc.end() && it->first.rfind(term, 0) == 0; ++it) tf += it->second;
      } else {
        auto it = doc.find(term);
        tf = it == doc.end() ? 0 : it->second;
      }
      if (mod == '+' && !tf) return Value::D(0);
      if (mod == '-' && tf) return Value::D(0);
      if (!tf || mod == '-') continue;
      const double df = st && (size_t)(2 + idx) < st->size() ? (*st)[2 + idx] : 1;
      const double idf = std::log(1 + (N - df + 0.5) / (df + 0.5));
      score += idf * tf / (tf + k1 * (1 - b + b * dw.size() / avgdl));  // Lucene's BM25 (no k1 + 1 factor)
    }
    return Value::D(score);
  }
  if (f == "REGEXP_SUBSTR" || f == "REGEXP_INSTR") {
    if (a[0].isNull() || a[1].isNull()) return Value::Null();
    std::string m;
    int64_t pos = 0;
    const bool hit = regexSubstr(str(0), str(1), ciColl(), m, pos);
    if (f == "REGEXP_INSTR") return Value::I(hit ? pos : 0);
    return hit ? Value::S(m) : Value::Null();
  }
  if (f == "REGEXP_LIKE" || f == "RLIKE" || f == "REGEXP") {
    return Value::I(regexLike(str(0), str(1), ciColl()));
  }
  for (auto& v : a)  // the functions below are NULL on any NULL argument
    if (v.isNull() && f != "RAND") return Value::Null();
  if (f == "MD5" || f == "SHA1" || f == "SHA") {
    const std::string s0 = str(0);
    unsigned char h[20];
    const size_t hl = f == "MD5" ? 16 : 20;
    if (f == "MD5") MD5((const unsigned char*)s0.data(), s0.size(), h);
    else SHA1((const unsigned char*)s0.data(), s0.size(), h);
    static const char* hex = "0123456789abcdef";
    std::string o;
    for (size_t i = 0; i < hl; ++i) o += hex[h[i] >> 4], o += hex[h[i] & 15];
    return Value::S(o);
  }
  if (f == "SHA2") {
    const std::string s0 = str(0);
    const int bits = a.size() > 1 ? (int)num(1) : 256;
    unsigned char h[64];
    size_t hl = 0;
    if (bits == 224) SHA224((const unsigned char*)s0.data(), s0.size(), h), hl = 28;
    else if (bits == 256 || bits == 0) SHA256((const unsigned char*)s0.data(), s0.size(), h), hl = 32;
    else if (bits == 384) SHA384((const unsigned char*)s0.data(), s0.size(), h), hl = 48;
    else if (bits == 512) SHA512((const unsigned char*)s0.data(), s0.size(), h), hl = 64;
    else return Value::Null();
    static const char* hex = "0123456789abcdef";
    std::string o;
    for (size_t i = 0; i < hl; ++i) o += hex[h[i] >> 4], o += hex[h[i] & 15];
    return Value::S(o);
  }
  if (f == "TO_BASE64" || f == "FROM_BASE64") {
    static const char* b64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    const std::string in = str(0);
    std::string o;
    if (f == "TO_BASE64") {
      for (size_t i = 0; i < in.size(); i += 3) {
        uint32_t v = (uint8_t)in[i] << 16 | (i + 1 < in.size() ? (uint8_t)in[i + 1] << 8 : 0) | (i + 2 < in.size() ? (uint8_t)in[i + 2] : 0);
        o += b64[v >> 18 & 63];
        o += b64[v >> 12 & 63];
        o += i + 1 < in.size() ? b64[v >> 6 & 63] : '=';
        o += i + 2 < in.size() ? b64[v & 63] : '=';
      }
      return Value::S(o);
    }
    uint32_t acc = 0;
    int bits = 0;
    for (char ch : in) {
      const char* q = strchr(b64, ch);
      if (!q || !ch) continue;  // '=', whitespace
      acc = acc << 6 | (uint32_t)(q - b64);
      bits += 6;
      if (bits >= 8) o += (char)(acc >> (bits -= 8) & 255);
    }
    return Value::S(o);
  }
  if (f == "CRC32") {
    const std::string s0 = str(0);
    return Value::I((int64_t)crc32(0L, (const Bytef*)s0.data(), (uInt)s0.size()));
  }
  if (f == "BIT_COUNT") return Value::I(__builtin_popcountll((unsigned long long)(int64_t)num(0)));
  if (f == "RAND") {
    if (!a.empty() && !a[0].isNull()) {  // RAND(seed): deterministic
      std::mt19937_64 seeded((uint64_t)(int64_t)num(0));
      return Value::D(std::uniform_real_distribution<double>(0, 1)(seeded));
    }
    if (!tStmtSeed) beginStatementClock();  // (outside a statement)
    return Value::D(std::uniform_real_distribution<double>(0, 1)(tStmtRng));
  }
  if (f == "CONVERT_TZ") {
    int64_t us = a[0].kind == Value::DATETIME || a[0].kind == Value::DATE ? (a[0].kind == Value::DATE ? a[0].i * kUsPerDay : a[0].i) : 0;
    if (a[0].kind == Value::STR && !parseDateTime(a[0].s, us)) return Value::Null();
    int64_t from = 0, to = 0;
    const int64_t secs = floorDiv(us, 1000000);
    if (!tzOffset(str(1), secs, true, from)) return Value::Null();
    const int64_t utc = secs - from;
    if (!tzOffset(str(2), utc, false, to)) return Value::Null();
    return Value::Ts(us + (to - from) * 1000000);
  }
  if (f.rfind("GEOGRAPHY_", 0) == 0) return geoFunction(f, a);
  if (f == "REGEXP_REPLACE") {
    return Value::S(regexReplace(str(0), str(1), str(2)));
  }
  throw SqlError("FUNCTION warpstore." + f + " does not exist", 1305);
}

inline ExprP boundCol(BindCtx& c, int bt, int bc) {
  auto col = std::make_shared<Expr>();
  col->kind = Expr::COL;
  col->bt = bt;
  col->bc = bc;
  col->ty = c.tabs[bt]->cols[bc].lt;
  col->name = c.tabs[bt]->cols[bc].name;
  col->table = c.aliases[bt];
  col->bound = true;
  return col;
}

inline int encSeq = 0;

// Encoded execution of a string expression with exactly one string column: evaluate it once per
// dictionary entry, then the GPU only applies the resulting code -> value map (OP_DICTMAP).
bool colRange(const Table& t, int bc, int64_t& lo, int64_t& hi);

// A host-only expression over one integer / DATE column with a small value range: evaluated once per
// value in [min, max] into a GPU lookup map indexed by (value - min), like the per-dictionary-entry
// encoding of string columns. NULL input: the map's NULL, or a CASE when the result isn't NULL.
// hi - lo + 1 values (hi >= lo), without overflowing for ranges wider than INT64_MAX.
inline uint64_t span64(int64_t lo, int64_t hi) { return (uint64_t)hi - (uint64_t)lo; }

// DATETIME column bt.bc used only through DATE_FORMAT(col, '<literal>'): the result depends only on
// the time bucket of the format's finest field (second / minute / hour / day), in microseconds; 0
// when the column is used otherwise (or the format has finer or unknown fields).
inline int64_t dateFormatBucket(const ExprP& e, int bt, int bc) {
  if (!e) return INT64_MAX;
  if (e->kind == Expr::COL) return e->bt == bt && e->bc == bc ? 0 : INT64_MAX;
  if (e->kind == Expr::FUNC && e->op == "DATE_FORMAT" && e->args.size() == 2 && e->args[0]->kind == Expr::COL && e->args[0]->bt == bt &&
      e->args[0]->bc == bc && e->args[1]->kind == Expr::LIT && e->args[1]->lit.kind == Value::STR) {
    const std::string& f = e->args[1]->lit.s;
    int64_t unit = 86400000000ll;
    for (size_t i = 0; i + 1 < f.size(); ++i) {
      if (f[i] != '%') continue;
      const char k = f[++i];
      if (strchr("sSTr", k)) unit = std::min<int64_t>(unit, 1000000);
      else if (k == 'i') unit = std::min<int64_t>(unit, 60000000);
      else if (strchr("HhIklp", k)) unit = std::min<int64_t>(unit, 3600000000ll);
      else if (!strchr("dejDaWwmcbMYyUuVvXx%", k)) return 0;  // (%f and the rest: per value)
    }
    return unit;
  }
  int64_t u = INT64_MAX;
  for (auto& a : e->args) {
    const int64_t x = dateFormatBucket(a, bt, bc);
    if (x == 0) return 0;
    u = std::min(u, x);
  }
  return u;
}

inline void encodeIntDomain(ExprP& e, BindCtx& c, int bt, int bc) {
  const Table& t = *c.tabs[bt];
  const LType lt = t.cols[bc].lt;
  if (lt == LType::DATETIME) {  // DATE_FORMAT(ts, ..) only: per time bucket
    const int64_t unit = dateFormatBucket(e, bt, bc);
    if (unit <= 1 || unit == INT64_MAX) return;
    int64_t lo, hi;
    if (!colRange(t, bc, lo, hi) || lo < 0 || hi / unit - lo / unit >= (1 << 20)) return;
    const int64_t b0 = lo / unit;
    const size_t N = (size_t)(hi / unit - b0 + 1);
    auto colSlot = [bt = bt, bc = bc](int tt, int cc) { return tt == bt && cc == bc ? 0 : -1; };
    std::vector<Value> res(N);
    parallelFor(N, [&](size_t b, size_t en) {
      PostCtx tpc;
      std::vector<Value> trow(1);
      tpc.row = &trow;
      tpc.colSlot = colSlot;
      for (size_t i = b; i < en; ++i) {
        trow[0] = Value::Ts((b0 + (int64_t)i) * unit);
        res[i] = evalPost(e, tpc);
      }
    });
    Value nullRes;
    {
      PostCtx tpc;
      std::vector<Value> trow(1, Value::Null());
      tpc.row = &trow;
      tpc.colSlot = colSlot;
      nullRes = evalPost(e, tpc);
    }
    std::vector<double> map(N);
    std::shared_ptr<Dict> outShared;
    if (e->ty == LType::VARCHAR) {
      outShared = std::make_shared<Dict>();
      c.sharedDicts.push_back(outShared);
      for (size_t i = 0; i < N; ++i) map[i] = res[i].isNull() ? nullv() : (double)outShared->getOrAdd(res[i].toString());
    } else {
      for (size_t i = 0; i < N; ++i) map[i] = res[i].isNull() ? nullv() : numOf(res[i]);
    }
    ExprP bucket = mkCall(FN_IDIV, {boundCol(c, bt, bc), boundLit(Value::I(unit))}, LType::BIGINT);
    ExprP idx = boundNode(Expr::BIN, "-", {bucket, boundLit(Value::I(b0))}, LType::BIGINT);
    ExprP gpu = boundNode(Expr::FUNC, "#MAP", {idx}, outShared ? LType::BIGINT : e->ty);
    gpu->aux = std::make_shared<std::vector<double>>(std::move(map));
    if (!nullRes.isNull()) {
      ExprP whenNull = boundNode(Expr::FUNC, "ISNULL", {boundCol(c, bt, bc)}, LType::INT);
      ExprP nullVal = outShared ? boundLit(Value::I(outShared->getOrAdd(nullRes.toString()))) : boundLit(Value::D(numOf(nullRes)));
      gpu = boundNode(Expr::CASE, "", {whenNull, nullVal, gpu}, outShared ? LType::BIGINT : e->ty);
    }
    ExprP node = boundNode(Expr::FUNC, "#ENC:" + std::to_string(++encSeq), {gpu, e}, e->ty);
    if (outShared) c.exprDict[node.get()] = outShared.get();
    e = node;
    return;
  }
  if (lt != LType::INT && lt != LType::BIGINT && lt != LType::DATE) return;
  const std::string es = upper(e->str());
  for (const char* f : {"NOW(", "RAND(", "UUID(", "CURDATE(", "CURTIME(", "SYSDATE(", "CURRENT_", "UNIX_TIMESTAMP()", "UTC_"})
    if (es.find(f) != std::string::npos) return;
  const bool udf = es.find("#UDF(") != std::string::npos;  // runs the procedure interpreter: one thread, fewer values
  int64_t lo, hi;
  if (!colRange(t, bc, lo, hi) || span64(lo, hi) >= (udf ? (1u << 16) : (1u << 20))) return;
  const size_t N = (size_t)span64(lo, hi) + 1;
  auto colSlot = [bt = bt, bc = bc](int tt, int cc) { return tt == bt && cc == bc ? 0 : -1; };
  std::vector<Value> res(N);
  auto evalRange = [&](size_t b, size_t en) {
    PostCtx tpc;
    std::vector<Value> trow(1);
    tpc.row = &trow;
    tpc.colSlot = colSlot;
    for (size_t i = b; i < en; ++i) {
      trow[0] = lt == LType::DATE ? Value::Dt(lo + (int64_t)i) : Value::I(lo + (int64_t)i);
      res[i] = evalPost(e, tpc);
    }
  };
  if (udf) evalRange(0, N);
  else parallelFor(N, evalRange);
  Value nullRes;
  {
    PostCtx tpc;
    std::vector<Value> trow(1, Value::Null());
    tpc.row = &trow;
    tpc.colSlot = colSlot;
    nullRes = evalPost(e, tpc);
  }
  std::vector<double> map(N);
  std::shared_ptr<Dict> outShared;
  if (e->ty == LType::VARCHAR) {
    outShared = std::make_shared<Dict>();
    c.sharedDicts.push_back(outShared);
    for (size_t i = 0; i < N; ++i) map[i] = res[i].isNull() ? nullv() : (double)outShared->getOrAdd(res[i].toString());
  } else {
    for (size_t i = 0; i < N; ++i) map[i] = res[i].isNull() ? nullv() : numOf(res[i]);
  }
  ExprP idx = boundNode(Expr::BIN, "-", {boundCol(c, bt, bc), boundLit(Value::I(lo))}, LType::BIGINT);
  ExprP gpu = boundNode(Expr::FUNC, "#MAP", {idx}, outShared ? LType::BIGINT : e->ty);
  gpu->aux = std::make_shared<std::vector<double>>(std::move(map));
  if (!nullRes.isNull()) {  // e.g. COALESCE / IFNULL / a UDF handling NULL
    ExprP whenNull = boundNode(Expr::FUNC, "ISNULL", {boundCol(c, bt, bc)}, LType::INT);
    ExprP nullVal = outShared ? boundLit(Value::I(outShared->getOrAdd(nullRes.toString()))) : boundLit(Value::D(numOf(nullRes)));
    gpu = boundNode(Expr::CASE, "", {whenNull, nullVal, gpu}, outShared ? LType::BIGINT : e->ty);
  }
  ExprP node = boundNode(Expr::FUNC, "#ENC:" + std::to_string(++encSeq), {gpu, e}, e->ty);
  if (outShared) c.exprDict[node.get()] = outShared.get();
  e = node;
}

inline void encodeStrings(ExprP& e, BindCtx& c) {
  if (hasAgg(e)) return;
  if (c.rowWise) return;  // UPDATE .. SET: evaluated for the matched rows only (not a whole dictionary)
  std::set<std::pair<int, int>> cs;
  colsOf(e, cs);
  if (cs.size() != 1) return;  // several columns: evaluated per result row on the host
  auto [bt, bc] = *cs.begin();
  const Table& t = *c.tabs[bt];
  if (t.cols[bc].lt != LType::VARCHAR) {
    encodeIntDomain(e, c, bt, bc);
    return;
  }
  const Dict& src = *t.dicts[bc];
  if (c.lazyStr && src.strs.size() > 4096) return;
  if (e->str().find("MATCH") != std::string::npos && !(e->kind == Expr::FUNC && e->op == "MATCH_AGAINST")) ftsStats(e, src);
  PostCtx pc;
  std::vector<Value> row(1);
  pc.row = &row;
  pc.colSlot = [bt = bt, bc = bc](int tt, int cc) { return tt == bt && cc == bc ? 0 : -1; };
  // Cached per dictionary version and expression (repeated queries skip the evaluation and the upload).
  const std::string ckey = derivKey(src, std::string("ENC|") + (e->ty == LType::VARCHAR ? "s|" : "n|") + e->str());
  const std::string es = upper(e->str());
  bool volatileExpr = false;
  for (const char* f : {"NOW(", "RAND(", "UUID(", "CURDATE(", "CURTIME(", "SYSDATE(", "CURRENT_", "UNIX_TIMESTAMP()", "UTC_"})
    volatileExpr |= es.find(f) != std::string::npos;
  // The result for a NULL input (dictionary maps only see non-NULL strings): non-NULL for COALESCE,
  // IFNULL, UDFs handling NULL, ..: then a CASE in front of the map.
  auto withNull = [&](ExprP gpu, Dict* outDict) {
    PostCtx npc;
    std::vector<Value> nrow(1, Value::Null());
    npc.row = &nrow;
    npc.colSlot = pc.colSlot;
    Value nv;
    try { nv = evalPost(e, npc); } catch (...) { return gpu; }
    if (nv.isNull() || (e->ty == LType::VARCHAR && !outDict)) return gpu;
    ExprP nullVal = outDict ? boundLit(Value::I(outDict->getOrAdd(nv.toString()))) : boundLit(Value::D(numOf(nv)));
    return boundNode(Expr::CASE, "", {boundNode(Expr::FUNC, "ISNULL", {boundCol(c, bt, bc)}, LType::INT), nullVal, gpu}, gpu->ty);
  };
  if (!volatileExpr) {
    DictDerived hit;
    if (derivGet(src, ckey, hit) && hit.map) {
      ExprP gpu = boundNode(Expr::FUNC, "#MAP", {boundCol(c, bt, bc)}, hit.out ? LType::BIGINT : e->ty);
      gpu->aux = hit.map;
      gpu = withNull(gpu, hit.out.get());
      ExprP node = boundNode(Expr::FUNC, "#ENC:" + std::to_string(++encSeq), {gpu, e}, e->ty);
      if (hit.out) {
        c.sharedDicts.push_back(hit.out);
        c.exprDict[node.get()] = hit.out.get();
      }
      e = node;
      return;
    }
  }
  // The dictionary grew since the cached map was built: evaluate only the new entries (copies, as
  // running queries may hold the old map and dictionary).
  DictDerived stale;
  size_t from = 0;
  if (!volatileExpr && derivGetStale(src, ckey, stale) && (e->ty == LType::VARCHAR) == (stale.out != nullptr)) from = stale.n;
  std::vector<double> map(src.strs.size());
  if (from) std::copy(stale.map->begin(), stale.map->begin() + from, map.begin());
  Dict* out = nullptr;
  std::shared_ptr<Dict> outShared;
  if (e->ty == LType::VARCHAR) {
    outShared = from ? std::make_shared<Dict>(*stale.out) : std::make_shared<Dict>();
    c.sharedDicts.push_back(outShared);
    out = outShared.get();
  }
  const auto tEnc = std::chrono::steady_clock::now();
  struct EncTrace {
    std::chrono::steady_clock::time_point t;
    size_t n;
    std::string what;
    ~EncTrace() {
      const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
      if (ms > 20 && envSetting("TRACE")) fprintf(stderr, "[trace] %s over %zu dictionary entries: %.1f ms\n", what.c_str(), n, ms);
    }
  } encTrace{tEnc, src.strs.size() - from, e->str()};
  if (out) {  // string results: computed on all cores, then added to the new dictionary in order
    const size_t N = src.strs.size();
    std::vector<std::string> res(N);
    std::vector<uint8_t> nul(N, 0);
    // REGEXP_REPLACE(col, 'pattern', 'replacement'): straight to the regex engine.
    const bool directRegex = e->kind == Expr::FUNC && e->op == "REGEXP_REPLACE" && e->args.size() == 3 && e->args[0]->kind == Expr::COL &&
                             e->args[1]->kind == Expr::LIT && e->args[1]->lit.kind == Value::STR && e->args[2]->kind == Expr::LIT &&
                             e->args[2]->lit.kind == Value::STR;
    const bool directEmbed = e->kind == Expr::FUNC && e->op == "EMBED" && e->args.size() == 1;
    if (directEmbed) {  // EMBED(f(col)): f per distinct value on the host, then one batched GPU pass
      std::vector<std::string> texts(N - from);
      std::vector<uint8_t> argNull(N - from, 0);
      if (e->args[0]->kind == Expr::COL) {
        std::copy(src.strs.begin() + from, src.strs.end(), texts.begin());
      } else {
        parallelFor(N - from, [&, from](size_t b, size_t en) {
          PostCtx tpc;
          std::vector<Value> trow(1);
          tpc.row = &trow;
          tpc.colSlot = pc.colSlot;
          for (size_t i = b; i < en; ++i) {
            trow[0] = Value::S(std::string(src.strs[i + from]));
            Value v = evalPost(e->args[0], tpc);
            if (v.isNull()) argNull[i] = 1;
            else texts[i] = v.toString();
          }
        });
      }
      std::vector<std::string> vecs;
      embedCached(texts, vecs);
      for (size_t i = from; i < N; ++i) {
        if (argNull[i - from]) nul[i] = 1;
        else res[i] = std::move(vecs[i - from]);
      }
    }
    if (!directEmbed) parallelFor(N - from, [&, from](size_t b, size_t en) {
      b += from, en += from;
      if (directRegex) {
        for (size_t i = b; i < en; ++i) res[i] = regexReplace(std::string(src.strs[i]), e->args[1]->lit.s, e->args[2]->lit.s);
        return;
      }
      PostCtx tpc;
      std::vector<Value> trow(1);
      tpc.row = &trow;
      tpc.colSlot = pc.colSlot;
      for (size_t i = b; i < en; ++i) {
        trow[0] = Value::S(std::string(src.strs[i]));
        Value v = evalPost(e, tpc);
        if (v.isNull()) nul[i] = 1;
        else res[i] = v.toString();
      }
    });
    const auto tAdd = std::chrono::steady_clock::now();
    // Distinct results on all cores (hash-sharded), then only those enter the dictionary, in order of
    // first occurrence (the codes are what a sequential pass would give).
    auto tick = [t = std::chrono::steady_clock::now()]() mutable {
      auto n = std::chrono::steady_clock::now();
      double ms = std::chrono::duration<double, std::milli>(n - t).count();
      t = n;
      return ms;
    };
    constexpr size_t kShards = 64;
    std::vector<uint8_t> shardOf(N);
    parallelFor(N - from, [&, from](size_t b, size_t en) {
      for (size_t i = b + from; i < en + from; ++i) shardOf[i] = (uint8_t)(std::hash<std::string_view>{}(res[i]) % kShards);
    });
    const double tHash = tick();
    std::vector<std::vector<uint32_t>> byShard(kShards);
    for (size_t i = from; i < N; ++i)
      if (!nul[i]) byShard[shardOf[i]].push_back((uint32_t)i);
    const double tBucket = tick();
    std::vector<int64_t> code(N, -1);   // known code (already in the dictionary)
    std::vector<uint32_t> first(N, 0);  // first index with the same result
    parallelFor(kShards, [&](size_t b, size_t en) {
      for (size_t sh = b; sh < en; ++sh) {
        std::unordered_map<std::string_view, uint32_t> seen;
        for (uint32_t i : byShard[sh]) {
          const int32_t have = out->strs.empty() ? -1 : out->find(res[i]);
          if (have >= 0) { code[i] = have; continue; }
          first[i] = seen.emplace(std::string_view(res[i]), i).first->second;
        }
      }
    }, 1);
    const double tShard = tick();
    {
      size_t fresh = 0;
      for (size_t i = from; i < N; ++i) fresh += !nul[i] && code[i] < 0 && first[i] == i;
      out->strs.reserve(out->strs.size() + fresh);
    }
    for (size_t i = from; i < N; ++i)
      if (!nul[i] && code[i] < 0 && first[i] == i) code[i] = out->appendUnique(std::move(res[i]));
    const double tIns = tick();
    if (envSetting("TRACE"))
      fprintf(stderr, "[trace]   dedupe: hash %.1f, bucket %.1f, shards %.1f, inserts %.1f ms\n", tHash, tBucket, tShard, tIns);
    parallelFor(N - from, [&, from](size_t b, size_t en) {
      for (size_t i = b + from; i < en + from; ++i) map[i] = nul[i] ? nullv() : (double)(code[i] >= 0 ? code[i] : code[first[i]]);
    });
    if (envSetting("TRACE"))
      fprintf(stderr, "[trace]   evaluate %.1f ms, dictionary inserts %.1f ms (%zu distinct)\n",
              std::chrono::duration<double, std::milli>(tAdd - tEnc).count(),
              std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tAdd).count(), out->strs.size());
  } else if (ftsMapFromIndex(e, src, map)) {  // MATCH .. AGAINST: from the inverted index
  } else {  // numeric results (length(), predicates, ..): every dictionary entry on all cores
    parallelFor(src.strs.size() - from, [&, from](size_t b, size_t en) {
      b += from, en += from;
      PostCtx tpc;
      std::vector<Value> trow(1);
      tpc.row = &trow;
      tpc.colSlot = pc.colSlot;
      for (size_t i = b; i < en; ++i) {
        trow[0] = Value::S(std::string(src.strs[i]));
        Value v = evalPost(e, tpc);
        map[i] = v.isNull() ? nullv() : numOf(v);
      }
    });
  }
  ExprP gpu = boundNode(Expr::FUNC, "#MAP", {boundCol(c, bt, bc)}, out ? LType::BIGINT : e->ty);
  auto mapPtr = std::make_shared<std::vector<double>>(std::move(map));
  gpu->aux = mapPtr;
  gpu = withNull(gpu, out);
  if (!volatileExpr) {
    DictDerived de;
    de.d = &src;
    de.n = src.strs.size();
    de.map = mapPtr;
    de.out = outShared;
    derivPut(ckey, std::move(de));
  }
  ExprP node = boundNode(Expr::FUNC, "#ENC:" + std::to_string(++encSeq), {gpu, e}, e->ty);
  if (out) c.exprDict[node.get()] = out;
  e = node;
}

// A VARCHAR-valued expression as codes of the merged dictionary `out` (literals and every dictionary
// involved are folded into it), or null when a branch has no dictionary (host evaluation then).
// Maps of dictionaries into a merged dictionary, reused from the cache when the merge is cached.
using CodeMaps = std::map<const Dict*, std::shared_ptr<std::vector<double>>>;
inline ExprP toCodes(const ExprP& x, Dict* out, BindCtx& c, CodeMaps& maps) {
  if (isNullLit(x)) return x;
  if (x->kind == Expr::LIT && x->lit.kind == Value::STR) return boundLit(Value::I(out->getOrAdd(x->lit.s)));
  if (x->kind == Expr::CASE && x->ty == LType::VARCHAR) {
    auto y = std::make_shared<Expr>(*x);
    const size_t n = y->args.size();
    for (size_t i = 1; i < n; i += 2)
      if (!(y->args[i] = toCodes(x->args[i], out, c, maps))) return nullptr;
    if (n % 2 && !(y->args[n - 1] = toCodes(x->args[n - 1], out, c, maps))) return nullptr;  // ELSE
    y->ty = LType::BIGINT;
    return y;
  }
  if (x->kind == Expr::FUNC && (x->op == "COALESCE2" || x->op == "NULLIF") && x->ty == LType::VARCHAR) {
    auto y = std::make_shared<Expr>(*x);
    for (auto& a : y->args)
      if (!(a = toCodes(a, out, c, maps))) return nullptr;
    y->ty = LType::BIGINT;
    return y;
  }
  if (Dict* dx = c.dict(*x)) {
    auto& m = maps[dx];
    if (!m) {
      m = std::make_shared<std::vector<double>>(dx->strs.size());
      if (dx == out) {  // the merge reuses this dictionary: codes stay the same (shared identity map)
        static std::mutex idMu;
        static std::map<size_t, std::shared_ptr<std::vector<double>>> ids;
        std::lock_guard<std::mutex> g(idMu);
        if (ids.size() > 16 && !ids.count(dx->strs.size())) ids.clear();
        auto& id = ids[dx->strs.size()];
        if (!id) {
          id = m;
          for (size_t i = 0; i < m->size(); ++i) (*m)[i] = (double)i;
        }
        m = id;
      } else if (out->strs.empty()) {  // first dictionary merged: copy it whole, codes stay the same
        *out = *dx;
        for (size_t i = 0; i < m->size(); ++i) (*m)[i] = (double)i;
      } else {
        for (size_t i = 0; i < m->size(); ++i) (*m)[i] = out->getOrAdd(dx->strs[i]);
      }
    }
    ExprP g = boundNode(Expr::FUNC, "#MAP", {x}, LType::BIGINT);
    g->aux = m;
    return g;
  }
  return nullptr;
}

// Merged dictionaries of string expressions (CASE / COALESCE over columns and literals), cached by
// expression and source dictionary versions (a merge over a big dictionary is a full pass over it).
struct UnifyEntry {
  std::shared_ptr<Dict> out;
  CodeMaps maps;
};
inline std::mutex gUnifyMu;
inline std::unordered_map<std::string, UnifyEntry> gUnify;

inline void unifyStrings(ExprP& e, BindCtx& c) {
  // Cache key: the expression and every table dictionary it reads (identity and size).
  std::string key = std::string(ciColl() ? "i|" : "b|") + e->str();
  bool cacheable = true;
  std::function<void(const ExprP&)> dicts = [&](const ExprP& x) {
    if (!x) return;
    if (x->kind == Expr::COL && x->ty == LType::VARCHAR) {
      if (Dict* d = c.dict(*x)) key += "|" + std::to_string(d->uid) + ":" + std::to_string(d->strs.size());
      return;
    }
    if (x->kind == Expr::FUNC && x->op.rfind("#", 0) == 0) cacheable = false;  // per-query derived dictionaries
    for (auto& a : x->args) dicts(a);
  };
  dicts(e);
  // One dictionary and literals it already contains (CASE WHEN .. THEN Referer ELSE '' END): use
  // that dictionary as the result's, instead of copying a possibly huge dictionary.
  std::set<Dict*> srcDicts;
  std::vector<std::string> lits;
  std::function<bool(const ExprP&)> leaves = [&](const ExprP& x) -> bool {
    if (isNullLit(x)) return true;
    if (x->kind == Expr::LIT && x->lit.kind == Value::STR) return lits.push_back(x->lit.s), true;
    if ((x->kind == Expr::CASE || (x->kind == Expr::FUNC && (x->op == "COALESCE2" || x->op == "NULLIF"))) && x->ty == LType::VARCHAR) {
      const size_t n = x->args.size();
      for (size_t i = 0; i < n; ++i) {
        const bool result = x->kind != Expr::CASE || (i % 2 == 1) || (n % 2 && i == n - 1);
        if (result && !leaves(x->args[i])) return false;
      }
      return true;
    }
    if (Dict* d = c.dict(*x)) return srcDicts.insert(d), true;
    return false;
  };
  if (leaves(e) && srcDicts.size() == 1) {
    Dict* d = *srcDicts.begin();
    bool all = true;
    for (auto& l : lits) all = all && d->find(l) >= 0;
    if (all) {
      std::shared_ptr<Dict> same(d, [](Dict*) {});  // owned by its table (locked for the query)
      UnifyEntry ent{same, {}};
      ExprP g = toCodes(e, d, c, ent.maps);
      if (g) {
        for (auto& [dd, m] : ent.maps) derivShareMap(m);
        ExprP node = boundNode(Expr::FUNC, "#ENC:" + std::to_string(++encSeq), {g, e}, LType::VARCHAR);
        c.exprDict[node.get()] = d;
        e = node;
        return;
      }
    }
  }
  UnifyEntry ent;
  bool hit = false;
  if (cacheable) {
    std::lock_guard<std::mutex> g(gUnifyMu);
    auto it = gUnify.find(key);
    if (it != gUnify.end()) ent = it->second, hit = true;
  }
  if (!hit) ent.out = std::make_shared<Dict>();
  c.sharedDicts.push_back(ent.out);
  Dict* out = ent.out.get();
  ExprP g = toCodes(e, out, c, ent.maps);
  if (!g) return;  // host-only
  if (cacheable && !hit) {
    for (auto& [d, m] : ent.maps) derivShareMap(m);
    std::lock_guard<std::mutex> lk(gUnifyMu);
    if (gUnify.size() > 256) gUnify.clear();
    gUnify[key] = ent;
  }
  ExprP node = boundNode(Expr::FUNC, "#ENC:" + std::to_string(++encSeq), {g, e}, LType::VARCHAR);
  c.exprDict[node.get()] = out;
  e = node;
}

// Literal that a column is compared against: returns true for `col op lit` forms.
inline bool simplePred(const ExprP& e, int& bt, int& bc, std::string& op, double& v) {
  static const std::map<std::string, std::string> flip = {{"=", "="}, {"<", ">"}, {"<=", ">="}, {">", "<"}, {">=", "<="}};
  if (e->kind != Expr::BIN || !flip.count(e->op)) return false;
  const ExprP& a = e->args[0];
  const ExprP& b = e->args[1];
  if (a->kind == Expr::COL && b->kind == Expr::LIT && b->lit.kind != Value::STR && a->ty != LType::VARCHAR) {
    bt = a->bt, bc = a->bc, op = e->op, v = b->lit.num();
    return true;
  }
  if (b->kind == Expr::COL && a->kind == Expr::LIT && a->lit.kind != Value::STR && b->ty != LType::VARCHAR) {
    bt = b->bt, bc = b->bc, op = flip.at(e->op), v = a->lit.num();
    return true;
  }
  return false;
}

// skipAllDeleted: a segment whose rows are all deleted holds nothing (callers pass false when another
// session's open transaction deleted some of them: those rows are still visible to this one).
inline bool segMayMatch(const Segment& s, const std::vector<ExprP>& preds, bool skipAllDeleted = true) {
  if (skipAllDeleted && s.ndel() == s.n) return false;
  for (auto& p : preds) {
    int bt, bc;
    std::string op;
    double v;
    if (!simplePred(p, bt, bc, op, v)) continue;
    double mn = s.mn[bc], mx = s.mx[bc];
    if ((op == "=" && (v < mn || v > mx)) || (op == "<" && mn >= v) || (op == "<=" && mn > v) ||
        (op == ">" && mx <= v) || (op == ">=" && mx < v))
      return false;
  }
  return true;
}

// Point lookup: finds a top-level `pk = literal` conjunct.
inline bool findPkEq(const std::vector<ExprP>& conj, const Table& t, int bt, int64_t& key) {
  if (t.isShadow()) return false;  // (MVCC snapshots carry no index maps: scan them)
  if (t.pk < 0) return false;
  for (auto& p : conj) {
    int pbt, bc;
    std::string op;
    double v;
    if (simplePred(p, pbt, bc, op, v) && op == "=" && pbt == bt && bc == t.pk) {
      if (v != std::floor(v)) return false;
      key = (int64_t)v;
      return true;
    }
  }
  return false;
}

inline ColView viewOf(const void* p, PType t) { return ColView{p, (uint8_t)t}; }

// A `col = literal` conjunct on the first column of a secondary index: returns the index and the
// raw key (found = false when the literal can't occur, e.g. a string not in the dictionary).
inline const IndexMeta* findIndexEq(const std::vector<ExprP>& conj, const Table& t, int bt, int64_t& key, bool& found) {
  if (t.isShadow()) return nullptr;  // (MVCC snapshots carry no index maps: scan them)
  for (auto& p : conj) {
    if (p->kind != Expr::BIN || p->op != "=") continue;
    const ExprP& a = p->args[0];
    const ExprP& b = p->args[1];
    const ExprP* col = a->kind == Expr::COL && b->kind == Expr::LIT ? &a : b->kind == Expr::COL && a->kind == Expr::LIT ? &b : nullptr;
    if (!col || (*col)->bt != bt) continue;
    const Value& v = (col == &a ? b : a)->lit;
    if (v.isNull()) continue;
    for (auto& ix : t.indexes) {
      if (ix.cols[0] != (*col)->bc || t.cols[ix.cols[0]].lt == LType::DECIMAL) continue;
      const ColumnMeta& cm = t.cols[ix.cols[0]];
      found = true;
      if (cm.lt == LType::VARCHAR) {
        if (v.kind != Value::STR) return nullptr;
        int32_t code = t.dicts[ix.cols[0]]->find(v.s);
        if (code < 0) found = false;
        key = code;
      } else if (cm.pt == PType::F64) {
        double d = numOf(v);
        if (d == 0) d = 0;
        memcpy(&key, &d, 8);
      } else if (cm.lt == LType::DATE && v.kind == Value::STR) {
        int32_t d;
        if (!parseDate(v.s, d)) return nullptr;
        key = d;
      } else if (cm.lt == LType::DATETIME && v.kind == Value::STR) {
        if (!parseDateTime(v.s, key)) return nullptr;
      } else {
        double d = numOf(v);
        if (d != std::floor(d)) found = false;
        key = (int64_t)d;
      }
      return &ix;
    }
  }
  return nullptr;
}

// Dictionary code -> lexicographic rank, in VRAM (for ORDER BY on strings). Dictionaries only
// grow, so (dictionary, size, collation) identifies a version: ranks are cached per version and
// built with a parallel sort.
inline std::shared_ptr<DevVec<int32_t>> dictRankDev(const Dict& d) {
  using Slot = std::pair<size_t, std::shared_ptr<DevVec<int32_t>>>;
  const bool ci = ciColl();
  const std::string ckey = ci ? "rankdev/ci" : "rankdev";
  const size_t n = d.strs.size();
  if (auto c = d.getDerived<Slot>(ckey); c && c->first == n) return c->second;
  auto sorted = sortedDict(d);  // cached (and pre-built in the background after loads)
  const std::vector<int32_t>& order = *sorted.first;
  std::vector<int32_t> rank(n + 1);
  for (size_t i = 0; i < n; ++i)  // case variants share a rank under a _ci collation
    rank[order[i]] = i > 0 && ci && collCmp(d.strs[order[i]], d.strs[order[i - 1]]) == 0 ? rank[order[i - 1]] : (int32_t)i;
  auto r = std::make_shared<DevVec<int32_t>>((int64_t)rank.size());
  h2d(r->p, rank.data(), rank.size() * 4);
  d.setDerived(ckey, std::make_shared<Slot>(n, r));
  return r;
}

// A materialized intermediate relation in VRAM (filtered base table or join result).
struct RelCol {
  int bt, bc;
  PType pt;
  DevVec<uint8_t> data;
  int esize = 0;  // bytes per element (VECTOR columns: 4 * dim)
};

inline void gatherCol(const void* src, PType pt, int esize, const int32_t* idx, int64_t m, void* dst, bool nulls) {
  if (pt == PType::VEC) gpuGatherBytes(src, esize, idx, m, dst, nulls);
  else if (nulls) gpuGatherNull(src, pt, idx, m, dst);
  else gpuGather(src, pt, idx, m, dst);
}
struct Rel {
  int64_t n = 0;
  std::vector<RelCol> cols;
  std::set<int> tabs;
  // Hash partitioning of the rows, when known: rows of partition p are [partOff[p], partOff[p+1]),
  // and every column in partCols is a shard key that determines the partition.
  std::vector<int64_t> partOff;
  std::set<std::pair<int, int>> partCols;
  int find(int bt, int bc) const {
    for (size_t i = 0; i < cols.size(); ++i)
      if (cols[i].bt == bt && cols[i].bc == bc) return (int)i;
    return -1;
  }
};

inline void bindRelCols(Compiler& comp, const Rel& r) {
  for (size_t s = 0; s < comp.slotCols.size(); ++s) {
    int i = r.find(comp.slotCols[s].first, comp.slotCols[s].second);
    if (i < 0) throw SqlError("internal: column not materialized", 1105);
    comp.env.cols[s].p = r.cols[i].data.p;
    comp.env.cols[s].enc = 0;  // materialized columns are decoded
  }
}

using Clock = std::chrono::steady_clock;
inline double msSince(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

// ---- PROFILE support (per session = per connection thread, like SingleStore's SHOW PROFILE).
struct ProfRow {
  std::string stage;
  double ms;
  int64_t rows;
  std::string detail;
};
inline thread_local bool tProfOn = false;
inline thread_local std::vector<ProfRow> tProf, tLastProf;
inline thread_local bool tDense = true;  // SET warpstore_dense_groupby

// Times one stage of a PROFILE'd query. It waits for the GPU at the end of the stage so time is
// attributed to the stage that queued the work (only when profiling).
struct ProfStage {
  std::string stage;
  Clock::time_point t0 = Clock::now();
  explicit ProfStage(std::string s) : stage(std::move(s)) { checkKilled(); }  // (KILL QUERY lands between stages)
  void done(int64_t rows = -1, std::string detail = "") {
    if (!tProfOn) return;
    gpuSync();
    tProf.push_back({stage, msSince(t0), rows, std::move(detail)});
    t0 = Clock::now();
  }
};

struct ScanStats {
  int segsTotal = 0, segsScanned = 0, partsTotal = 0, partsScanned = 0;
  int64_t rowsScanned = 0, bufferRows = 0;
  int64_t segRows = 0;  // rows of the scanned segments before sort-key range narrowing
};

// Partitions that can hold rows matching `preds`: just one when every shard-key column is pinned by
// a `col = literal` conjunct (SingleStore's single-partition routing), otherwise all of them.
inline std::vector<int> partsToScan(const Table& t, int bt, const std::vector<ExprP>& preds) {
  if (tOnlyPart >= 0 && t.nparts > 1 && !t.reference) return tOnlyPart < t.nparts ? std::vector<int>{tOnlyPart} : std::vector<int>{};
  std::vector<int> all(t.nparts);
  for (int p = 0; p < t.nparts; ++p) all[p] = p;
  if (!t.partitioned()) return all;
  std::vector<Value> vals;
  for (int c : t.shardKey) {
    bool found = false;
    for (auto& p : preds) {
      if (p->kind != Expr::BIN || p->op != "=") continue;
      const ExprP& a = p->args[0];
      const ExprP& b = p->args[1];
      const ExprP* col = a->kind == Expr::COL && b->kind == Expr::LIT ? &a : b->kind == Expr::COL && a->kind == Expr::LIT ? &b : nullptr;
      if (!col || (*col)->bt != bt || (*col)->bc != c) continue;
      vals.push_back((col == &a ? b : a)->lit);
      found = true;
      break;
    }
    if (!found) return all;
  }
  int p = t.partOfValues(vals);
  return p < 0 ? all : std::vector<int>{p};
}

struct BatchRef {
  int part;
  int64_t n;
  const std::vector<void*>* dev;
  const uint32_t* del;
  const std::vector<uint8_t>* enc = nullptr;  // per column ColView::enc (segments); null: all raw
  int32_t off = 0;                            // first row, for encoded columns of a range scan
  Segment* seg = nullptr;                     // segment batches: column reads are counted (VRAM cache)
};

// Column view of column c of a batch, for a slot whose type fields are already set.
inline void bindView(ColView& v, const BatchRef& b, int c) {
  if (b.seg && (size_t)c < b.seg->uses.size()) __atomic_fetch_add(&b.seg->uses[c], 1u, __ATOMIC_RELAXED);
  v.p = (*b.dev)[c];
  v.enc = b.enc ? (*b.enc)[c] : 0;
  v.off = v.enc ? b.off : 0;
  if (!v.p && b.seg && (size_t)c < b.seg->d->cold.size()) {  // a cold column (disk tier): read it
    int64_t encOff = 0;
    v.p = segScanColumn(*b.seg->d, c, b.off, b.n, encOff);
    if (v.enc) v.off = (int32_t)encOff;
  }
}

// Visits, partition by partition (ascending), every segment that survives min/max elimination and
// then that partition's write buffer. Stops early when fn returns false.
inline thread_local int tScanDelayMs = 0;  // SET warpstore_debug_scan_delay_ms (tests: a slow reader holding its snapshot)
inline void scanTable(Table& t, const std::vector<int>& parts, const std::vector<ExprP>& preds, ScanStats& st,
               const std::function<bool(const BatchRef&)>& fn, bool dryRun = false) {
  if (tScanDelayMs > 0 && !dryRun) std::this_thread::sleep_for(std::chrono::milliseconds(tScanDelayMs));
  st.segsTotal += (int)t.segs.size();
  st.partsTotal += t.nparts;
  st.partsScanned += (int)parts.size();
  // Range predicates on the SORT KEY: segments are sorted by it, so a binary search on the host
  // mirror narrows each segment to the row range that can match (zone maps at row granularity).
  std::vector<std::pair<std::string, double>> sortPreds;
  if (t.sortKey >= 0 && (t.cols[t.sortKey].pt == PType::I32 || t.cols[t.sortKey].pt == PType::I64))
    for (auto& p : preds) {
      int bt, bc;
      std::string op;
      double v;
      if (simplePred(p, bt, bc, op, v) && bc == t.sortKey) sortPreds.push_back({op, v});
    }
  // Read committed: other sessions' open transactions -> per-batch visibility adjustments (their
  // inserts hidden, their deletes still visible).
  std::map<int32_t, std::vector<std::pair<int32_t, bool>>> adjust;  // batch (segment, or -1-part) -> (row, hide)
  for (auto& [k, v] : t.txnPending) {
    if (k == txnKey()) continue;
    if (v.commitSeq && (!tSnapSeq || v.commitSeq <= tSnapSeq)) continue;  // committed before this snapshot
    for (auto& l : v.hidden) adjust[l.seg].push_back({l.row, true});
    for (auto& l : v.ghosts) adjust[l.seg].push_back({l.row, false});
  }
  auto viewDel = [&](int32_t key, DelBits& del, int64_t n) -> const uint32_t* {
    auto it = adjust.find(key);
    if (dryRun) return nullptr;
    if (it == adjust.end()) return del.device((n + 31) / 32);
    std::vector<uint32_t> bits(del.host.begin(), del.host.begin() + std::min<size_t>(del.host.size(), (size_t)(n + 31) / 32));
    for (auto [r, hide] : it->second) {
      if ((size_t)(r >> 5) >= bits.size()) bits.resize((r >> 5) + 1, 0);
      if (hide) bits[r >> 5] |= 1u << (r & 31);
      else bits[r >> 5] &= ~(1u << (r & 31));
    }
    auto d = std::make_shared<DevVec<uint32_t>>((int64_t)std::max<size_t>(bits.size(), 1));
    h2d(d->p, bits.data(), bits.size() * 4);
    tTxnViews.push_back(d);
    return d->p;
  };
  // The batches to visit, in order: per partition its segments (after elimination, narrowed to
  // sort-key ranges), then its write buffer (sp null).
  struct Visit {
    int p;
    Segment* sp;
    int32_t si;
    int64_t lo, hi;
  };
  std::vector<Visit> plan;
  for (int p : parts) {
    for (auto& sp : t.segs) {
      if (sp->part != p || !segMayMatch(*sp, preds, !adjust.count((int32_t)(&sp - t.segs.data())))) continue;
      int64_t lo = 0, hi = sp->n;
      if (!sortPreds.empty() && (int)sp->host.size() > t.sortKey && (segCold(*sp->d, t.sortKey) || !sp->host[t.sortKey].empty())) {
        const bool i64 = t.cols[t.sortKey].pt == PType::I64;
        auto at = [&](int64_t r) { return (double)segRawInt(*sp->d, t.sortKey, r, i64); };
        auto lowerB = [&](double v) {  // first row >= v
          int64_t a = 0, b = sp->n;
          while (a < b) { int64_t m = (a + b) / 2; if (at(m) < v) a = m + 1; else b = m; }
          return a;
        };
        auto upperB = [&](double v) {  // first row > v
          int64_t a = 0, b = sp->n;
          while (a < b) { int64_t m = (a + b) / 2; if (at(m) <= v) a = m + 1; else b = m; }
          return a;
        };
        for (auto& [op, v] : sortPreds) {
          if (op == ">=" || op == "=") lo = std::max(lo, lowerB(v));
          if (op == ">") lo = std::max(lo, upperB(v));
          if (op == "<") hi = std::min(hi, lowerB(v));
          if (op == "<=" || op == "=") hi = std::min(hi, upperB(v));
        }
        lo &= ~(int64_t)31;  // the deleted-row bitmap is addressed in 32-row words
      }
      if (lo >= hi) continue;
      plan.push_back({p, sp.get(), (int32_t)(&sp - t.segs.data()), lo, hi});
    }
    plan.push_back({p, nullptr, -1, 0, 0});
  }
  // Read-ahead of columns on disk follows this order (restored for an enclosing scan).
  std::vector<ScanAheadItem> ahead;
  for (auto& v : plan)
    if (v.sp && v.sp->d->file) ahead.push_back({v.sp->d.get(), v.lo, v.hi - v.lo});
  size_t aheadPos = 0;
  if (!dryRun) scanAheadExchange(ahead, aheadPos);
  struct Restore {
    std::vector<ScanAheadItem>& a;
    size_t& pos;
    bool on;
    ~Restore() {
      if (on) scanAheadExchange(a, pos);
    }
  } restore{ahead, aheadPos, !dryRun};
  struct Flush {  // GPU decompressions of the columns read by the last batches (before their kernels)
    bool on;
    ~Flush() {
      if (on) scanFlush();
    }
  } flush{!dryRun};
  size_t aheadAt = 0;
  std::vector<void*> adj;
  for (auto& vis : plan) {
    const int p = vis.p;
    if (vis.sp) {
      Segment* sp = vis.sp;
      const int64_t lo = vis.lo, hi = vis.hi;
      ++st.segsScanned;
      st.rowsScanned += hi - lo;
      st.segRows += sp->n;
      if (dryRun) continue;
      if (sp->d->file) scanAheadAt(aheadAt++);
      const uint32_t* segDel = viewDel(vis.si, *sp->del, sp->n);
      if (lo == 0 && hi == sp->n) {
        if (!fn(BatchRef{p, sp->n, &sp->dev, segDel, &sp->enc, 0, sp})) return;
        continue;
      }
      // Raw columns: shift the pointer; encoded ones (header first): pass the row offset.
      adj.assign(sp->dev.size(), nullptr);
      for (size_t c = 0; c < sp->dev.size(); ++c)
        adj[c] = sp->dev[c] && !sp->enc[c] ? (void*)((char*)sp->dev[c] + lo * t.colBytes((int)c)) : sp->dev[c];
      if (!fn(BatchRef{p, hi - lo, &adj, segDel ? segDel + lo / 32 : nullptr, &sp->enc, (int32_t)lo, sp})) return;
      continue;
    }
    WriteBuffer& b = *t.bufs[p];
    if (b.n > 0) {
      st.bufferRows += b.n;
      if (dryRun) continue;
      t.syncBufferToDevice(p);
      const uint32_t* bdel = viewDel(-1 - p, *b.del, b.n);
      if (!fn(BatchRef{p, b.n, &b.dev, bdel})) return;
    }
  }
}

// Runs `fn(env, n, deleted)` over every batch of table bt that may hold matching rows.
inline void forEachBatch(BindCtx& ctx, int bt, Compiler& comp, const std::vector<ExprP>& preds, ScanStats& st,
                  const std::function<bool(ExprEnv&, int64_t, const uint32_t*)>& fn, bool dryRun = false) {
  Table& t = *ctx.tabs[bt];
  scanTable(t, partsToScan(t, bt, preds), preds, st, [&](const BatchRef& b) {
    for (size_t k = 0; k < comp.slotCols.size(); ++k) bindView(comp.env.cols[k], b, comp.slotCols[k].second);
    return fn(comp.env, b.n, b.del);
  }, dryRun);
}

inline std::string scanDesc(const Table& t, const ScanStats& st) {
  std::string r = t.name + ": scanned " + std::to_string(st.segsScanned) + "/" + std::to_string(st.segsTotal) +
                  " GPU segments";
  if (st.rowsScanned < st.segRows)
    r += " (sort-key ranges: " + std::to_string(st.rowsScanned) + " of " + std::to_string(st.segRows) + " rows)";
  if (t.nparts > 1)
    r += " in " + std::to_string(st.partsScanned) + "/" + std::to_string(st.partsTotal) + " partitions" +
         (st.partsScanned < st.partsTotal ? " (shard key routing)" : "");
  return r + " + " + std::to_string(st.bufferRows) + " write-buffer rows";
}

// Value range of an integer-like column from statistics: dictionary size for strings, segment
// min/max plus a pass over the (small) write buffers otherwise. False if unknown / empty.
inline bool colRange(const Table& t, int bc, int64_t& lo, int64_t& hi) {
  if (t.cols[bc].lt == LType::VARCHAR) {
    lo = 0;
    hi = (int64_t)t.dicts[bc]->strs.size() - 1;
    return hi >= lo;
  }
  if (t.cols[bc].lt == LType::DOUBLE || t.cols[bc].lt == LType::DECIMAL) return false;
  double mn = INFINITY, mx = -INFINITY;
  for (auto& sp : t.segs) mn = std::min(mn, sp->mn[bc]), mx = std::max(mx, sp->mx[bc]);
  for (int p = 0; p < t.nparts; ++p) {
    const WriteBuffer& wb = *t.bufs[p];
    BufData& b = *wb.data;  // (the cache is shared with the buffer's other snapshots)
    std::lock_guard<std::mutex> g(b.rangeMu);
    if (b.rangeGen != wb.gen || b.rmn.size() != t.cols.size()) {
      b.rangeGen = wb.gen;
      b.rangeN = 0;
      b.rmn.assign(t.cols.size(), INFINITY);
      b.rmx.assign(t.cols.size(), -INFINITY);
    }
    if (b.rangeN < wb.n) {  // newly appended rows extend every integer-like column's range
      for (size_t c = 0; c < t.cols.size(); ++c) {
        if (t.cols[c].lt == LType::VARCHAR || t.cols[c].lt == LType::DOUBLE || t.cols[c].lt == LType::DECIMAL ||
            t.cols[c].pt == PType::VEC) continue;
        double lo = b.rmn[c], hi = b.rmx[c];
        for (int64_t r = b.rangeN; r < wb.n; ++r) {
          double v = t.cellNum(bufLoc(p, r), (int)c);
          lo = std::min(lo, v), hi = std::max(hi, v);
        }
        b.rmn[c] = lo, b.rmx[c] = hi;
      }
      b.rangeN = wb.n;
    }
    mn = std::min(mn, b.rmn[bc]), mx = std::max(mx, b.rmx[bc]);
  }
  if (!(mx >= mn)) return false;
  lo = (int64_t)mn;
  hi = (int64_t)mx;
  return true;
}

inline thread_local bool tJoinFilters = true;            // SET warpstore_join_filters = ON|OFF
inline thread_local bool tAnn = true;                    // SET warpstore_vector_index = ON|OFF (OFF: exact GPU search)
inline thread_local int tNprobe = 0;                     // SET warpstore_nprobe = N (0: index default)
inline thread_local int tCagraItopk = 0;                 // SET warpstore_cagra_itopk = N (0: index default)

inline thread_local int64_t tLazyRows = 2'000'000;        // SET warpstore_join_filter_rows = N (tables this big are filtered)

// Would this expression compile for the GPU? (Host-only string functions over several columns don't.)
inline bool gpuCompilable(const ExprP& e, BindCtx& ctx) {
  try {
    Compiler c(&ctx);
    c.compile(e);
    return true;
  } catch (const SqlError&) {
    return false;
  }
}

// One stored cell as a typed host value (NULL-aware).
inline Value cellValue(const Table& t, Loc l, int c) {
  if (t.cols[c].pt == PType::VEC) {
    const float* f = (const float*)t.colData(l, c).data() + (int64_t)l.row * t.cols[c].vecDim;
    return f[0] != f[0] ? Value::Null() : Value::S(formatVector(f, t.cols[c].vecDim));
  }
  double v = t.cellNum(l, c);
  if (t.cols[c].lt == LType::DECIMAL) return v != v ? Value::Null() : Value::Dec(v, t.cols[c].scale);
  if (t.cols[c].pt == PType::I64 && v == v) {  // exact for BIGINT / DATETIME
    int64_t x = ((const int64_t*)t.colData(l, c).data())[l.row];
    return t.cols[c].lt == LType::DATETIME ? Value::Ts(x) : Value::I(x);
  }
  return typedVal(v, t.cols[c].lt, t.dicts[c].get());
}

// Semi-join ("join filter") applied while scanning a table: keep rows whose column bc has its
// value set in an exact bitmap of the keys present on the other side of a join (value - lo).
struct JoinFilter {
  int bc;
  int64_t lo, nbits;
  const uint32_t* bits;  // VRAM
  std::string desc;
};

// Filter table bt with pushed-down predicates (and join filters) and gather the needed columns
// into VRAM.
// Raw stored bytes of one cell (a column on disk is read from its segment file).
inline void cellRaw(const Table& t, Loc l, int c, uint8_t* out) {
  if (l.seg >= 0 && segCold(*t.segs[l.seg]->d, c)) {
    segRawCell(*t.segs[l.seg]->d, c, l.row, out);
    return;
  }
  const int w = t.colBytes(c);
  memcpy(out, t.colData(l, c).data() + (size_t)l.row * w, w);
}

// locs: also the location of every output row (late materialization of other columns).
inline Rel materialize(BindCtx& ctx, int bt, const std::vector<ExprP>& preds, const std::set<int>& needCols, int64_t maxRows,
                std::vector<std::string>& notes, const std::vector<JoinFilter>& jfs = {}, std::vector<Loc>* locs = nullptr) {
  Table& t = *ctx.tabs[bt];
  ProfStage prof("scan+filter " + t.name);
  Compiler comp(&ctx);
  ExprP where = andAll(preds);
  Prog wp;
  const bool hasFilter = where || !jfs.empty();
  if (hasFilter) {
    // WHERE AND INSET(col - lo, bitmap) AND ... as one program (one fused, compiled filter).
    comp.depth = 0;
    wp.start = (int16_t)comp.ncode;
    int terms = 0;
    if (where) {
      comp.gen(where);
      ++terms;
    }
    for (auto& f : jfs) {
      comp.emit(OP_COL, comp.slot(bt, f.bc), 1);
      comp.emit(OP_CONST, comp.constantInt(f.lo), 1, true);
      comp.emit(OP_SUB, 0, -1);
      comp.emit(OP_INSET, comp.addDeviceSet(f.bits, (int)f.nbits), 0);
      if (terms++) comp.emit(OP_AND, 0, -1);
    }
    wp.len = (int16_t)(comp.ncode - wp.start);
  }
  comp.uploadSets();
  const JitProgram* jit = nullptr;
  if (hasFilter) {
    AggPlan fp;
    fp.hasWhere = true;
    fp.where = wp;
    jit = jitGet(comp.env, (int)comp.slotCols.size(), fp, JF_FLAGM, "filter");
  }
  // All batches (segments + write buffers) are filtered in one launch with one synchronization;
  // the selected rows (in batch order) are then gathered column by column across batches. Columns
  // read from the disk tier for this scan are bounded: once a window of them is collected, it is
  // filtered and gathered, and the next window starts (the pieces are concatenated at the end).
  int64_t total = 0;
  ScanStats st;
  std::vector<int> need(needCols.begin(), needCols.end());
  std::vector<int64_t> perPart(t.nparts, 0);
  std::vector<int64_t> bn;
  std::vector<const uint32_t*> bdel;
  std::vector<ColView> bcols, srcCols;  // [batch][slot], [batch][need]
  std::vector<int> bpart;
  std::vector<Loc> bloc;  // locs: each batch's first row
  std::unordered_map<const Segment*, int32_t> segIndex;
  if (locs)
    for (size_t i = 0; i < t.segs.size(); ++i) segIndex[t.segs[i].get()] = (int32_t)i;
  const int nslots = (int)comp.slotCols.size();
  std::vector<std::vector<DevVec<uint8_t>>> pieces(need.size());
  std::vector<int64_t> pieceRows;
  int windows = 0;
  auto window = [&] {
    if (bn.empty()) return;
    scanFlush();
    ++windows;
    DevVec<int32_t> sel;
    std::vector<int64_t> perBatch;
    int64_t m = gpuSelectMulti(comp.env, hasFilter, wp, bn, bdel, bcols, nslots, sel, perBatch, jit);
    if (maxRows >= 0 && total + m > maxRows) m = maxRows - total;
    int64_t left = m;
    for (size_t b = 0; b < bn.size(); ++b) {
      const int64_t k = std::min(left, perBatch[b]);
      perPart[bpart[b]] += k;
      left -= k;
    }
    std::vector<int64_t> boff(bn.size(), 0);
    for (size_t b = 1; b < bn.size(); ++b) boff[b] = boff[b - 1] + bn[b - 1];
    if (locs && m > 0) {
      size_t b = 0;
      for (int32_t g : sel.download(m)) {
        while (b + 1 < bn.size() && boff[b + 1] <= g) ++b;
        locs->push_back(Loc{bloc[b].seg, (int32_t)(bloc[b].row + (g - boff[b]))});
      }
    }
    for (size_t ci = 0; ci < need.size(); ++ci) {
      const int c = need[ci];
      DevVec<uint8_t> out(std::max<int64_t>(m, 1) * t.colBytes(c));
      std::vector<ColView> src(bn.size());
      for (size_t b = 0; b < bn.size(); ++b) src[b] = srcCols[b * need.size() + ci];
      if (m > 0) gpuGatherMulti(src, boff, t.cols[c].pt, t.colBytes(c), sel.p, m, out.p);
      pieces[ci].push_back(std::move(out));
    }
    pieceRows.push_back(m);
    total += m;
    bn.clear(), bdel.clear(), bcols.clear(), srcCols.clear(), bpart.clear(), bloc.clear();
    releaseScanTemps();
  };
  scanTable(t, partsToScan(t, bt, preds), preds, st, [&](const BatchRef& b) {
    bn.push_back(b.n);
    bdel.push_back(b.del);
    bpart.push_back(b.part);
    if (locs) bloc.push_back(b.seg ? Loc{segIndex.at(b.seg), b.off} : bufLoc(b.part, 0));
    for (int k = 0; k < nslots; ++k) {
      ColView v = comp.env.cols[k];
      bindView(v, b, comp.slotCols[k].second);
      bcols.push_back(v);
    }
    for (int c : need) {
      ColView v{};
      bindView(v, b, c);
      srcCols.push_back(v);
    }
    if (scanWindowFull()) window();
    return !(maxRows >= 0 && total >= maxRows);
  });
  window();
  Rel r;
  r.n = total;
  r.tabs.insert(bt);
  if (t.partitioned() && t.shardKey.size() == 1) {
    r.partOff.assign(t.nparts + 1, 0);
    for (int p = 0; p < t.nparts; ++p) r.partOff[p + 1] = r.partOff[p] + perPart[p];
    r.partCols.insert({bt, t.shardKey[0]});
  }
  for (size_t ci = 0; ci < need.size(); ++ci) {
    const int c = need[ci];
    RelCol rc{bt, c, t.cols[c].pt, DevVec<uint8_t>(), t.colBytes(c)};
    if (pieces[ci].size() == 1) {
      rc.data = std::move(pieces[ci][0]);
    } else {
      rc.data = DevVec<uint8_t>(std::max<int64_t>(total, 1) * rc.esize);
      int64_t off = 0;
      for (size_t w = 0; w < pieces[ci].size(); ++w) {
        d2d(rc.data.p + off * rc.esize, pieces[ci][w].p, (size_t)pieceRows[w] * rc.esize);
        off += pieceRows[w];
      }
      pieces[ci].clear();
    }
    r.cols.push_back(std::move(rc));
  }
  std::string jdesc;
  for (auto& f : jfs) jdesc += ", join filter " + f.desc;
  notes.push_back("  materialize " + scanDesc(t, st) + (where ? ", filter " + where->str() : "") + jdesc + " -> " +
                  std::to_string(total) + " rows");
  prof.done(total, std::to_string(st.segsScanned) + "/" + std::to_string(st.segsTotal) + " segments, " +
                       std::to_string(st.rowsScanned + st.bufferRows) + " rows scanned" + jdesc +
                       (windows > 1 ? ", " + std::to_string(windows) + " disk-tier windows" : "") +
                       (hasFilter ? jit ? ", filter compiled (plan #" + std::to_string(jit->id) + ")" : ", filter interpreted" : ""));
  return r;
}

// String join keys: b's dictionary codes -> a's code space (strings a doesn't have -> -2, never
// equal to a code). Under a case-insensitive collation both sides map to a's canonical codes.
// Returns (map for a or empty, map for b).
inline std::pair<std::vector<int32_t>, std::vector<int32_t>> joinCodeMaps(const Dict& a, const Dict& b) {
  std::vector<int32_t> ma, mb(b.strs.size(), -2);
  if (ciColl()) {
    std::unordered_map<std::string, int32_t> folded;  // folded text -> first code in a
    ma.resize(a.strs.size());
    for (size_t i = 0; i < a.strs.size(); ++i) ma[i] = folded.emplace(foldCase(a.strs[i]), (int32_t)i).first->second;
    parallelFor(b.strs.size(), [&](size_t s, size_t e) {
      for (size_t i = s; i < e; ++i) {
        auto it = folded.find(foldCase(b.strs[i]));
        if (it != folded.end()) mb[i] = it->second;
      }
    });
  } else {
    parallelFor(b.strs.size(), [&](size_t s, size_t e) {
      for (size_t i = s; i < e; ++i) mb[i] = a.find(b.strs[i]);
      for (size_t i = s; i < e; ++i)
        if (mb[i] < 0) mb[i] = -2;
    });
  }
  return {ma, mb};
}

inline Rel hashJoin(Rel& a, Rel& b, int abt, int abc, int bbt, int bbc, std::vector<std::string>& notes,
             const Dict* da = nullptr, const Dict* db = nullptr) {
  ProfStage prof("hash join");
  bool aBuild = a.n <= b.n;
  Rel& build = aBuild ? a : b;
  Rel& probe = aBuild ? b : a;
  int bk = aBuild ? a.find(abt, abc) : b.find(bbt, bbc);
  int pk = aBuild ? b.find(bbt, bbc) : a.find(abt, abc);
  // String keys from different dictionaries: join on remapped codes (temporary key columns).
  DevVec<int32_t> remapA, remapB;
  const void* keyA = (aBuild ? build : probe).cols[aBuild ? bk : pk].data.p;
  const void* keyB = (aBuild ? probe : build).cols[aBuild ? pk : bk].data.p;
  if (da && db && da != db) {
    auto [ma, mb] = joinCodeMaps(*da, *db);
    auto apply = [&](const void* keys, int64_t n, const std::vector<int32_t>& m, DevVec<int32_t>& out) {
      DevVec<int32_t> md((int64_t)std::max<size_t>(m.size(), 1));
      if (!m.empty()) h2d(md.p, m.data(), m.size() * 4);
      out = DevVec<int32_t>(std::max<int64_t>(n, 1));
      gpuRemapCodes((const int32_t*)keys, n, md.p, (int64_t)m.size(), out.p);
    };
    const int64_t na = aBuild ? build.n : probe.n, nb = aBuild ? probe.n : build.n;
    if (!ma.empty()) apply(keyA, na, ma, remapA), keyA = remapA.p;
    apply(keyB, nb, mb, remapB), keyB = remapB.p;
  }
  const void* buildKey = aBuild ? keyA : keyB;
  const void* probeKey = aBuild ? keyB : keyA;
  // DOUBLE keys (on either side): join on the float64 bit patterns (-0.0 = 0.0; NULL never matches),
  // the integer side converted to double first.
  PType buildPt = build.cols[bk].pt, probePt = probe.cols[pk].pt;
  DevVec<int64_t> fbits[2];
  if (buildPt == PType::F64 || probePt == PType::F64) {
    for (int sd = 0; sd < 2; ++sd) {
      Rel& r = sd == 0 ? build : probe;
      const void* key = sd == 0 ? buildKey : probeKey;
      const PType pt = sd == 0 ? buildPt : probePt;
      std::vector<int64_t> k(r.n);
      if (pt == PType::F64) {
        std::vector<double> h(r.n);
        if (r.n) d2h(h.data(), key, r.n * 8);
        for (int64_t i = 0; i < r.n; ++i) {
          double d = h[i] == 0 ? 0.0 : h[i];
          if (d != d) k[i] = kNullI64;
          else memcpy(&k[i], &d, 8);
        }
      } else {
        const bool i64 = pt == PType::I64;
        std::vector<uint8_t> raw(r.n * (i64 ? 8 : 4));
        if (r.n) d2h(raw.data(), key, raw.size());
        for (int64_t i = 0; i < r.n; ++i) {
          const int64_t v = i64 ? ((const int64_t*)raw.data())[i] : ((const int32_t*)raw.data())[i];
          if (v == (i64 ? kNullI64 : (int64_t)kNullI32)) { k[i] = kNullI64; continue; }
          double d = (double)v;
          if (d == 0) d = 0;
          memcpy(&k[i], &d, 8);
        }
      }
      fbits[sd] = DevVec<int64_t>(std::max<int64_t>(r.n, 1));
      if (r.n) h2d(fbits[sd].p, k.data(), r.n * 8);
    }
    buildKey = fbits[0].p, probeKey = fbits[1].p;
    buildPt = probePt = PType::I64;
  }
  // Co-located join: both inputs are hash-partitioned the same way on exactly these join columns,
  // so partition p of one side can only match partition p of the other.
  auto intLike = [](PType t) { return t != PType::F64; };
  const bool coloc = !build.partOff.empty() && build.partOff.size() == probe.partOff.size() &&
                     a.partCols.count({abt, abc}) && b.partCols.count({bbt, bbc}) && intLike(build.cols[bk].pt) &&
                     intLike(probe.cols[pk].pt);
  std::vector<int64_t> bOff = coloc ? build.partOff : std::vector<int64_t>{0, build.n};
  std::vector<int64_t> pOff = probe.partOff.empty() ? std::vector<int64_t>{0, probe.n} : probe.partOff;
  std::vector<int64_t> outOff;
  DevVec<int32_t> ib, ip;
  int64_t m = gpuHashJoinParts(viewOf(buildKey, buildPt), bOff, viewOf(probeKey, probePt), pOff, ib, ip, &outOff);
  Rel out;
  out.n = m;
  if (!probe.partOff.empty()) {  // output keeps the probe side's partitioning
    out.partOff = outOff;
    out.partCols = probe.partCols;
    if (coloc) out.partCols.insert(build.partCols.begin(), build.partCols.end());
  }
  out.tabs = a.tabs;
  out.tabs.insert(b.tabs.begin(), b.tabs.end());
  for (auto* side : {&build, &probe}) {
    const int32_t* idx = side == &build ? ib.p : ip.p;
    for (auto& c : side->cols) {
      RelCol rc{c.bt, c.bc, c.pt, DevVec<uint8_t>(std::max<int64_t>(m, 1) * c.esize), c.esize};
      gatherCol(c.data.p, c.pt, c.esize, idx, m, rc.data.p, false);
      out.cols.push_back(std::move(rc));
    }
  }
  notes.push_back(std::string("  GPU hash join") +
                  (coloc ? " (co-located, " + std::to_string(bOff.size() - 1) + " partitions)" : " (broadcast build side)") +
                  ": build " + std::to_string(build.n) + " rows, probe " + std::to_string(probe.n) + " rows -> " +
                  std::to_string(m) + " rows");
  prof.done(m, std::string(coloc ? "co-located" : "broadcast") + ", build " + std::to_string(build.n) + " x probe " +
                   std::to_string(probe.n));
  return out;
}

// Outer hash join of two relations on lt.lc = rt.rc. type 1: LEFT (keep every row of a), 2: RIGHT
// (keep every row of b), 3: FULL. Columns of the side that can be missing come out NULL.
inline Rel outerJoin(Rel& a, Rel& b, int lt, int lc, int rt, int rc, int type, std::vector<std::string>& notes,
              const Dict* da = nullptr, const Dict* db = nullptr) {
  ProfStage prof("outer hash join");
  Rel& build = type == 2 ? a : b;
  Rel& probe = type == 2 ? b : a;
  int bk = type == 2 ? a.find(lt, lc) : b.find(rt, rc);
  int pk = type == 2 ? b.find(rt, rc) : a.find(lt, lc);
  if (bk < 0 || pk < 0) throw SqlError("internal: outer join key not materialized", 1105);
  // String keys from different dictionaries: b's codes (and a's, under _ci) remapped to a's.
  const int ak = type == 2 ? bk : pk, bkk = type == 2 ? pk : bk;
  const void* keyA = a.cols[ak].data.p;
  const void* keyB = b.cols[bkk].data.p;
  DevVec<int32_t> remapA, remapB;
  if (da && db && da != db) {
    auto [ma, mb] = joinCodeMaps(*da, *db);
    auto apply = [&](const void* keys, int64_t n, const std::vector<int32_t>& m, DevVec<int32_t>& out) {
      DevVec<int32_t> md((int64_t)std::max<size_t>(m.size(), 1));
      if (!m.empty()) h2d(md.p, m.data(), m.size() * 4);
      out = DevVec<int32_t>(std::max<int64_t>(n, 1));
      gpuRemapCodes((const int32_t*)keys, n, md.p, (int64_t)m.size(), out.p);
    };
    if (!ma.empty()) apply(keyA, a.n, ma, remapA), keyA = remapA.p;
    apply(keyB, b.n, mb, remapB), keyB = remapB.p;
  }
  const void* buildKey = type == 2 ? keyA : keyB;
  const void* probeKey = type == 2 ? keyB : keyA;
  DevVec<int32_t> ib, ip;
  int64_t m = gpuHashJoinOuter(viewOf(buildKey, build.cols[bk].pt), build.n, viewOf(probeKey, probe.cols[pk].pt), probe.n, ib, ip, type == 3);
  Rel out;
  out.n = m;
  out.tabs = a.tabs;
  out.tabs.insert(b.tabs.begin(), b.tabs.end());
  for (auto* side : {&build, &probe}) {
    const int32_t* idx = side == &build ? ib.p : ip.p;
    const bool nul = side == &build || type == 3;
    for (auto& c : side->cols) {
      RelCol rc2{c.bt, c.bc, c.pt, DevVec<uint8_t>(std::max<int64_t>(m, 1) * c.esize), c.esize};
      gatherCol(c.data.p, c.pt, c.esize, idx, m, rc2.data.p, nul);
      out.cols.push_back(std::move(rc2));
    }
  }
  static const char* names[] = {"", "LEFT", "RIGHT", "FULL"};
  notes.push_back(std::string("  GPU ") + names[type] + " OUTER hash join: build " + std::to_string(build.n) +
                  " rows, probe " + std::to_string(probe.n) + " rows -> " + std::to_string(m) + " rows");
  prof.done(m, std::string(names[type]) + " outer");
  return out;
}

inline Rel crossJoin(Rel& a, Rel& b, std::vector<std::string>& notes) {
  ProfStage prof("cross join");
  DevVec<int32_t> ia, ib;
  int64_t m = gpuCrossJoin(a.n, b.n, ia, ib);
  Rel out;
  out.n = m;
  out.tabs = a.tabs;
  out.tabs.insert(b.tabs.begin(), b.tabs.end());
  for (auto* side : {&a, &b}) {
    const int32_t* idx = side == &a ? ia.p : ib.p;
    for (auto& c : side->cols) {
      RelCol rc{c.bt, c.bc, c.pt, DevVec<uint8_t>(std::max<int64_t>(m, 1) * c.esize), c.esize};
      gatherCol(c.data.p, c.pt, c.esize, idx, m, rc.data.p, false);
      out.cols.push_back(std::move(rc));
    }
  }
  notes.push_back("  GPU cross join: " + std::to_string(a.n) + " x " + std::to_string(b.n) + " -> " + std::to_string(m) + " rows");
  prof.done(m);
  return out;
}

// Column defaults for an INSERT that doesn't name every column: DEFAULT value, CURRENT_TIMESTAMP, or NULL.
inline std::vector<Value> defaultRow(const Table& t) {
  std::vector<Value> vals(t.cols.size(), Value::Null());
  for (size_t i = 0; i < t.cols.size(); ++i) {
    const ColumnMeta& c = t.cols[i];
    if (!c.hasDefault) continue;
    if (c.def.kind == Value::STR && c.def.s == "CURRENT_TIMESTAMP")
      vals[i] = c.lt == LType::DATE ? Value::Dt(floorDiv(stmtNowUs(), kUsPerDay)) : Value::Ts(stmtNowUs() / 1000000 * 1000000);
    else
      vals[i] = c.def;
  }
  return vals;
}

inline ResultSet okResult(int64_t affected, std::string info = "") {
  ResultSet r;
  r.affected = affected;
  r.info = std::move(info);
  return r;
}

// (}  // namespace: removed, engine_core.h is shared by the engine's sources)

// (used across the engine's sources)
struct Pipeline {
  std::string name, db, table, glob, delim, cols;
  int64_t skip = 0;
  std::atomic<bool> running{false};
  bool wantRun = false;
  std::thread th;
  std::mutex mu;  // guards the fields below
  std::set<std::string> done;
  int64_t files = 0, rows = 0;
  std::string state = "Stopped", lastError, createdSql;
};
struct Txn {
  bool active = false;
  bool autocommit = true;
  bool needSync = false;  // a write to a SYNC DURABILITY database (or a sync session): COMMIT waits for the disk
  std::vector<std::string> wal;
  std::vector<UndoRec> undo;
  std::vector<std::pair<std::string, std::pair<size_t, size_t>>> savepoints;  // name -> (undo size, wal size)
};
inline thread_local Txn tTxn;
inline std::string fmtMs(double ms) {
  char b[32];
  snprintf(b, sizeof b, "%.3f", ms);
  return b;
}
struct SnapScope {
  std::map<const Table*, std::shared_ptr<Table>> shadows;
  std::vector<std::shared_ptr<Table>> bases;  // the live tables, kept alive while their snapshots are used
  uint64_t ticket = rcuEnter();               // memory the VRAM balancer frees waits for this statement
  ~SnapScope() {
    shadows.clear();
    rcuExit(ticket);
  }
};
inline thread_local SnapScope* tSnapScope = nullptr;
// (defined in engine.cpp, used by the other sources)
void waitTxnFree(Table& t, std::unique_lock<FairSharedMutex>& lk, const std::vector<Loc>* rows = nullptr);
bool rowsPending(const Table& t, const std::vector<Loc>& rows);
extern std::shared_mutex gCommitGate;
void forEachLiveRow(const Table& t, const std::function<void(Loc)>& fn);
bool keyLookup(const Table& t, const ExprP& w);
}  // namespace gdb
