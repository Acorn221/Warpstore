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
bool foreignPending(const Table& t) {
  for (auto& [k, v] : t.txnPending)
    if (k != txnKey() && !v.commitSeq && (!v.hidden.empty() || !v.ghosts.empty())) return true;
  return false;
}
// Snapshot isolation (REPEATABLE READ): a snapshot transaction sees the commits up to tSnapSeq; the
// changes of later commits stay in Table::txnPending (retired, with their commitSeq) while it runs.
thread_local uint64_t tSnapSeq = 0;
std::atomic<uint64_t> gCommitSeq{1};
std::mutex gSnapMu;
std::multiset<uint64_t> gSnaps;  // active snapshots
bool snapshotsActive() {
  std::lock_guard<std::mutex> g(gSnapMu);
  return !gSnaps.empty();
}
thread_local std::vector<std::shared_ptr<DevVec<uint32_t>>> tTxnViews;  // visibility bitmaps of the current statement
bool regexLike(const std::string& s, const std::string& pat, bool ci);  // regex.cpp (RE2)
std::string regexReplace(const std::string& s, const std::string& pat, const std::string& rep);
bool tzOffset(const std::string& zone, int64_t secs, bool local, int64_t& offsetSecs);  // regex.cpp
bool regexSubstr(const std::string& s, const std::string& pat, bool ci, std::string& out, int64_t& pos);  // regex.cpp
Value geoFunction(const std::string& f, const std::vector<Value>& a);                   // regex.cpp

ExprP cloneExpr(const ExprP& e);  // query layer

const std::string kNullCell("\0NULL", 5);
ExprP cloneExpr(const ExprP& e);  // defined with the query layer

// ---- Collation. Default utf8mb4_general_ci (case-insensitive, like SingleStore); per session
// SET collation_connection = utf8mb4_bin for binary. Strings are dictionary encoded, so case
// folding happens once per dictionary entry, not per row.
std::atomic<bool> gCiDefault{true};
thread_local int tCi = -1;
bool ciColl() { return tCi >= 0 ? tCi != 0 : gCiDefault.load(); }
std::string foldCase(std::string_view x) {
  std::string r(x);
  for (auto& ch : r) ch = (char)tolower((unsigned char)ch);
  return r;
}
// Runs f(begin, end) over [0, n) on all cores in chunks (multiples of 32, so bitmap words are never
// shared). Workers inherit the session's collation.
void parallelFor(size_t n, const std::function<void(size_t, size_t)>& f, size_t minChunk = 32768) {
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
std::function<bool(std::string_view)> likeMatcher(const std::string& pat) {
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
std::shared_ptr<const std::vector<int32_t>> sortedOrderCached(const Dict& d, bool ci, size_t n) {
  auto c = d.getDerived<SortedOrder>(ci ? "sorted/ci" : "sorted");
  return c && c->n == n ? c->order : nullptr;
}
// An order read back from a dictionary file's side file (snapshot load).
void sortedOrderInstall(const Dict& d, bool ci, std::shared_ptr<const std::vector<int32_t>> order) {
  auto rank = std::make_shared<std::vector<double>>(order->size());
  for (size_t r = 0; r < order->size(); ++r) (*rank)[(*order)[r]] = (double)r;
  d.setDerived(ci ? "sorted/ci" : "sorted", std::make_shared<SortedOrder>(SortedOrder{order->size(), order, rank}));
  derivShareMap(rank);
}

std::pair<std::shared_ptr<const std::vector<int32_t>>, std::shared_ptr<std::vector<double>>> sortedDict(const Dict& d) {
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
void (*gFtsWarm)(const Dict& d) = nullptr;  // builds a FULLTEXT column's inverted index (set by the full-text code)
// Background dictionary warm-up yields the CPU to queries: the thread, and the threads it starts,
// run at nice 10.
static void lowPriority() { setpriority(PRIO_PROCESS, (id_t)syscall(SYS_gettid), 10); }

void warmDicts(const std::shared_ptr<Table>& t) {
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
std::mutex gDerivMu;
std::unordered_map<std::string, DictDerived> gDeriv;
// Host map -> (the map, kept alive so its address is never reused; its device copy, made lazily).
std::unordered_map<const void*, std::pair<std::shared_ptr<std::vector<double>>, std::shared_ptr<DevVec<double>>>> gMapDev;
size_t gDerivBytes = 0;
std::string derivKey(const Dict& d, const std::string& what) {
  return std::to_string(d.uid) + "|" + (ciColl() ? "i" : "b") + "|" + what;
}
bool derivGet(const Dict& d, const std::string& key, DictDerived& out) {
  std::lock_guard<std::mutex> g(gDerivMu);
  auto it = gDeriv.find(key);
  if (it == gDeriv.end() || it->second.d != &d || it->second.n != d.strs.size()) return false;
  out = it->second;
  return true;
}
// An entry for an older (smaller) version of the dictionary: it can be extended with the new entries.
bool derivGetStale(const Dict& d, const std::string& key, DictDerived& out) {
  std::lock_guard<std::mutex> g(gDerivMu);
  auto it = gDeriv.find(key);
  if (it == gDeriv.end() || it->second.d != &d || it->second.n >= d.strs.size() || !(it->second.map || it->second.bits)) return false;
  out = it->second;
  return true;
}
void derivPut(const std::string& key, DictDerived e) {
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
std::shared_ptr<DevVec<double>> derivMapDev(const std::vector<double>* m) {
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
void derivShareMap(const std::shared_ptr<std::vector<double>>& m) {
  std::lock_guard<std::mutex> g(gDerivMu);
  if (!gMapDev.count(m.get())) {
    gMapDev[m.get()] = {m, nullptr};
    gDerivBytes += m->size() * 8;
  }
}

int collCmp(std::string_view a, std::string_view b) {
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
std::shared_ptr<std::vector<double>> ciCanon(const Dict& d) {
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
std::pair<std::shared_ptr<std::vector<double>>, std::shared_ptr<std::vector<double>>> crossRank(const Dict& a, const Dict& b) {
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

namespace {

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

bool isAggName(const std::string& n) { return n == "SUM" || n == "COUNT" || n == "AVG" || n == "MIN" || n == "MAX"; }
// Aggregates computed by the host aggregation path of the query layer.
bool isHostAggName(const std::string& n) {
  return n == "GROUP_CONCAT" || n == "JSON_ARRAYAGG" || n == "JSON_AGG" || n == "JSON_OBJECTAGG" || n == "STRING_AGG" ||
         n == "STDDEV" || n == "STD" || n == "STDDEV_POP" || n == "STDDEV_SAMP" || n == "VARIANCE" || n == "VAR_POP" ||
         n == "VAR_SAMP" || n == "ANY_VALUE" || n == "MEDIAN" || n == "BIT_AND" || n == "BIT_OR" || n == "BIT_XOR" ||
         n == "FIRST" || n == "LAST" || n == "PERCENTILE_CONT" || n == "PERCENTILE_DISC" || n == "APPROX_PERCENTILE";
}
bool isAgg(const ExprP& e) { return e->kind == Expr::FUNC && isAggName(e->op); }
bool isCmp(const std::string& o) { return o == "=" || o == "<>" || o == "<" || o == "<=" || o == ">" || o == ">="; }

bool hasAgg(const ExprP& e) {
  if (!e) return false;
  if (isAgg(e)) return true;
  for (auto& a : e->args)
    if (hasAgg(a)) return true;
  return false;
}

// Canonical form after binding: columns are identified by (table, column) index.
std::string canon(const ExprP& e) {
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

void bindChildren(ExprP& e, BindCtx& c) {
  for (auto& a : e->args) bind(a, c);
}

int64_t addInterval(int64_t days, int64_t n, const std::string& unit) {
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

thread_local int64_t tLastInsertId = 0;  // per connection, like MySQL
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
thread_local int64_t tFoundRows = 0;
thread_local bool tSnapTryOnly = false;  // automatic snapshots: skip the round rather than wait for writers
thread_local bool tWriteGateHeld = false;  // this thread's statement holds Engine::writeGate_ (shared)
thread_local uint64_t tAckLsn = 0;          // semi-sync: the replica acknowledgement to wait for once the gate is let go     // FOUND_ROWS(): rows of the last SELECT (without its LIMIT if SQL_CALC_FOUND_ROWS)
thread_local std::vector<std::string> tTempTables;  // CREATE TEMPORARY TABLE: dropped when the connection ends
std::string dbOf(const std::string& name) {
  size_t d = name.find('.');
  return d == std::string::npos ? currentDb() : name.substr(0, d);
}
std::string plainName(const std::string& name) {
  size_t d = name.find('.');
  return d == std::string::npos ? name : name.substr(d + 1);
}
thread_local bool tWantVals = false;     // doSelect returns typed rows (for nested queries)
thread_local bool tWinOuterCols = false;  // windows(): its final projection may come back as columns

int64_t nowUs() {
  return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

// The statement's clock and random numbers: NOW() / CURRENT_TIMESTAMP defaults read tStmtTimeUs, and
// RAND() / UUID() draw from tStmtRng, seeded with tStmtSeed. Both are set when a statement starts
// and written with it to the WAL ("/*ws <us> <seed>*/ " in front), so a replay (crash recovery,
// replicas, point-in-time restores) computes the same values, as MySQL's binlog does.
thread_local int64_t tStmtTimeUs = 0;
thread_local uint64_t tStmtSeed = 0;
thread_local std::mt19937_64 tStmtRng;
int64_t stmtNowUs() { return tStmtTimeUs ? tStmtTimeUs : nowUs(); }
static void beginStatementClock() {
  static std::atomic<uint64_t> salt{std::random_device{}() ^ ((uint64_t)std::random_device{}() << 32)};
  tStmtTimeUs = nowUs();
  tStmtSeed = salt.fetch_add(0x9E3779B97F4A7C15ull) ^ (uint64_t)tStmtTimeUs;
  tStmtRng.seed(tStmtSeed);
}
static std::string clockPrefix() {
  char b[64];
  snprintf(b, sizeof b, "/*ws %lld %llu*/ ", (long long)tStmtTimeUs, (unsigned long long)tStmtSeed);
  return b;
}
// A logged statement's clock prefix (replay): taken off, and the statement's clock set from it.
static std::string takeClockPrefix(const std::string& sql) {
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

ExprP boundNode(Expr::Kind k, const std::string& op, std::vector<ExprP> args, LType ty) {
  auto e = std::make_shared<Expr>();
  e->kind = k;
  e->op = op;
  e->args = std::move(args);
  e->ty = ty;
  e->bound = true;
  return e;
}
ExprP mkCall(uint16_t id, std::vector<ExprP> args, LType ty) {
  ExprP e = boundNode(Expr::FUNC, (args.size() == 1 ? "#CALL1:" : "#CALL2:") + std::to_string(id), std::move(args), ty);
  e->fnId = id;
  return e;
}
ExprP boundLit(Value v) {
  ExprP e = mkLit(std::move(v));
  e->ty = e->lit.isNull() ? LType::DOUBLE
        : e->lit.kind == Value::INT ? LType::BIGINT : e->lit.kind == Value::DBL ? LType::DOUBLE
        : e->lit.kind == Value::DATE ? LType::DATE : e->lit.kind == Value::DATETIME ? LType::DATETIME : LType::VARCHAR;
  e->bound = true;
  return e;
}
bool isNullLit(const ExprP& e) { return e->kind == Expr::LIT && e->lit.isNull(); }
bool allLits(const ExprP& e) {
  if (e->kind == Expr::COL || e->kind == Expr::STAR) return false;
  if (e->kind == Expr::FUNC && isAggName(e->op)) return false;
  for (auto& a : e->args)
    if (!allLits(a)) return false;
  return true;
}
Value evalConst(const ExprP& e);  // constant folding (evalPost without context)

// A string literal used where a date / datetime is expected.
ExprP coerceTemporal(const ExprP& x, LType want) {
  if (x->kind != Expr::LIT || x->lit.kind != Value::STR) return x;
  if (want == LType::DATE) {
    int32_t d;
    if (parseDate(x->lit.s, d)) return boundLit(Value::Dt(d));
  }
  int64_t us;
  if (parseDateTime(x->lit.s, us)) return want == LType::DATE ? boundLit(Value::Dt(floorDiv(us, kUsPerDay))) : boundLit(Value::Ts(us));
  throw SqlError("Incorrect " + std::string(want == LType::DATE ? "DATE" : "DATETIME") + " value: '" + x->lit.s + "'", 1292);
}
ExprP asDate(const ExprP& x) {
  ExprP y = coerceTemporal(x, LType::DATE);
  return y->ty == LType::DATETIME ? mkCall(FN_DT_DATE, {y}, LType::DATE) : y;
}
ExprP asDateTime(const ExprP& x) {
  ExprP y = coerceTemporal(x, LType::DATETIME);
  return y->ty == LType::DATE ? mkCall(FN_DATE_TO_DT, {y}, LType::DATETIME) : y;
}

// date/datetime +/- INTERVAL n unit
ExprP intervalArith(ExprP dt, int64_t n, const std::string& unit) {
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
void bindFunc(ExprP& e, BindCtx& c) {
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
ExprP strToNum(const ExprP& x, BindCtx& c, bool integer) {
  if (x->kind == Expr::LIT && x->lit.kind == Value::STR)
    return boundLit(integer ? Value::I(std::strtoll(x->lit.s.c_str(), nullptr, 10)) : Value::D(std::strtod(x->lit.s.c_str(), nullptr)));
  ExprP n = boundNode(Expr::FUNC, integer ? "CASTINT" : "CASTNUM", {x}, integer ? LType::BIGINT : LType::DOUBLE);
  encodeStrings(n, c);
  return n;
}

void bind(ExprP& e, BindCtx& c) {
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
bool nullable(const ExprP& e, const BindCtx& c) {
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

void tablesOf(const ExprP& e, std::set<int>& out) {
  if (e->kind == Expr::COL) out.insert(e->bt);
  for (auto& a : e->args) tablesOf(a, out);
}
void colsOf(const ExprP& e, std::set<std::pair<int, int>>& out) {
  if (!e) return;
  if (e->kind == Expr::COL) out.insert({e->bt, e->bc});
  for (auto& a : e->args) colsOf(a, out);
}
void flattenAnd(const ExprP& e, std::vector<ExprP>& out) {
  if (!e) return;
  if (e->kind == Expr::BIN && e->op == "AND") {
    flattenAnd(e->args[0], out);
    flattenAnd(e->args[1], out);
  } else {
    out.push_back(e);
  }
}
ExprP andAll(const std::vector<ExprP>& v) {
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
int* ovfFlagDev() {
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

Value typedVal(double v, LType t, const Dict* d, int scale = -1) {
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

std::string cell(const Value& v) { return v.isNull() ? kNullCell : v.toString(); }

int cmpVal(const Value& a, const Value& b) {
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

double numOf(const Value& v) {
  if (v.isNull()) return nullv();
  switch (v.kind) {
    case Value::DBL: return v.d;
    case Value::STR: return std::atof(v.s.c_str());
    default: return (double)v.i;
  }
}
bool isTrue(const Value& v) { return !v.isNull() && (v.kind == Value::STR ? std::atof(v.s.c_str()) != 0 : numOf(v) != 0); }
Value boolVal(double v) { return v != v ? Value::Null() : Value::I(v != 0); }

Value evalPost(const ExprP& e, const PostCtx& c) {
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

Value evalConst(const ExprP& e) {
  PostCtx pc;
  return evalPost(e, pc);
}

// ---- Minimal JSON navigation for JSON_EXTRACT_* (SingleStore) and JSON_EXTRACT (MySQL).
size_t jsonWs(const std::string& s, size_t i) {
  while (i < s.size() && isspace((unsigned char)s[i])) ++i;
  return i;
}
size_t jsonSkip(const std::string& s, size_t i) {  // index just past the value starting at i
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
std::string jsonUnquote(const std::string& v) {
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
bool jsonFind(const std::string& js, const std::vector<std::string>& path, std::string& out) {
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
std::vector<std::string> jsonPathOf(const std::string& p) {  // '$.a.b[2]' -> {a, b, 2}
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
static std::string jsonEdit(const std::string& v, const std::vector<std::string>& path, size_t k, const std::string& val, int mode) {
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

std::string dateFormat(const Value& v, const std::string& fmt) {
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

size_t utf8Len(const std::string& s) {
  size_t n = 0;
  for (unsigned char ch : s) n += (ch & 0xc0) != 0x80;
  return n;
}

// Character (not byte) positions in UTF-8 text, as MySQL's string functions count them.
static bool asciiOnly(const std::string& s) {
  for (unsigned char ch : s)
    if (ch & 0x80) return false;
  return true;
}
// Byte offset of character k (clamped to the end).
static size_t u8Off(const std::string& s, int64_t k) {
  if (k <= 0) return 0;
  size_t i = 0;
  for (int64_t c = 0; i < s.size(); ++i)
    if (((unsigned char)s[i] & 0xc0) != 0x80 && c++ == k) return i;
  return s.size();
}
// Characters [from, from + n) (from 0-based).
static std::string u8Sub(const std::string& s, int64_t from, int64_t n) {
  if (asciiOnly(s)) return from >= (int64_t)s.size() || n <= 0 ? std::string() : s.substr(from, n);
  const size_t b = u8Off(s, from), e = n >= (int64_t)s.size() ? s.size() : u8Off(s, from + n);
  return b >= e ? std::string() : s.substr(b, e - b);
}
// Code points <-> UTF-8, for case mapping.
static void u8Decode(const std::string& s, std::vector<uint32_t>& out) {
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
static void u8Append(std::string& o, uint32_t cp) {
  if (cp < 0x80) o += (char)cp;
  else if (cp < 0x800) o += (char)(0xc0 | (cp >> 6)), o += (char)(0x80 | (cp & 0x3f));
  else if (cp < 0x10000) o += (char)(0xe0 | (cp >> 12)), o += (char)(0x80 | ((cp >> 6) & 0x3f)), o += (char)(0x80 | (cp & 0x3f));
  else o += (char)(0xf0 | (cp >> 18)), o += (char)(0x80 | ((cp >> 12) & 0x3f)), o += (char)(0x80 | ((cp >> 6) & 0x3f)), o += (char)(0x80 | (cp & 0x3f));
}
// Upper / lower case: ASCII, Latin-1, Latin Extended-A, Greek and Cyrillic.
static uint32_t cpUpper(uint32_t c) {
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
static uint32_t cpLower(uint32_t c) {
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
void embedCached(const std::vector<std::string>& texts, std::vector<std::string>& out) {
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
std::string jsonQuoteStr(const std::string& x) {
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
std::string jsonOfValue(const Value& v, bool raw) {
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
bool isJsonExpr(const ExprP& x, const BindCtx* c) {
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
std::vector<std::string> ftsWords(std::string_view x) {
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
std::vector<std::tuple<char, std::string, bool>> ftsTerms(const std::string& qs, bool boolean) {
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
static FtsIndex& ftsIndexFor(const Dict& d) {
  static std::mutex mu;
  std::lock_guard<std::mutex> g(mu);
  auto p = d.getDerived<FtsIndex>("fts");
  if (!p) d.setDerived("fts", p = std::make_shared<FtsIndex>());
  return *p;  // lives as long as the dictionary
}
// Brings ix up to date with d (caller holds ix.mu; the table is locked, so d does not change).
static void ftsIndexUpdate(FtsIndex& ix, const Dict& d) {
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
void ftsWarm(const Dict& d) {
  FtsIndex& ix = ftsIndexFor(d);
  std::lock_guard<std::mutex> g(ix.mu);
  ftsIndexUpdate(ix, d);
}
const bool ftsWarmRegistered = (gFtsWarm = ftsWarm, true);
// MATCH (col) AGAINST ('..' [IN BOOLEAN MODE]) for every code of col's dictionary, from the index
// (the same BM25 as the per-document evaluation).
static bool ftsMapFromIndex(const ExprP& e, const Dict& d, std::vector<double>& map) {
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
void ftsStats(const ExprP& e, const Dict& d) {
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

Value evalHostFunc(const ExprP& e, const PostCtx& c) {
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

ExprP boundCol(BindCtx& c, int bt, int bc) {
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

int encSeq = 0;

// Encoded execution of a string expression with exactly one string column: evaluate it once per
// dictionary entry, then the GPU only applies the resulting code -> value map (OP_DICTMAP).
bool colRange(const Table& t, int bc, int64_t& lo, int64_t& hi);

// A host-only expression over one integer / DATE column with a small value range: evaluated once per
// value in [min, max] into a GPU lookup map indexed by (value - min), like the per-dictionary-entry
// encoding of string columns. NULL input: the map's NULL, or a CASE when the result isn't NULL.
// hi - lo + 1 values (hi >= lo), without overflowing for ranges wider than INT64_MAX.
static inline uint64_t span64(int64_t lo, int64_t hi) { return (uint64_t)hi - (uint64_t)lo; }

// DATETIME column bt.bc used only through DATE_FORMAT(col, '<literal>'): the result depends only on
// the time bucket of the format's finest field (second / minute / hour / day), in microseconds; 0
// when the column is used otherwise (or the format has finer or unknown fields).
static int64_t dateFormatBucket(const ExprP& e, int bt, int bc) {
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

void encodeIntDomain(ExprP& e, BindCtx& c, int bt, int bc) {
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

void encodeStrings(ExprP& e, BindCtx& c) {
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
ExprP toCodes(const ExprP& x, Dict* out, BindCtx& c, CodeMaps& maps) {
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
std::mutex gUnifyMu;
std::unordered_map<std::string, UnifyEntry> gUnify;

void unifyStrings(ExprP& e, BindCtx& c) {
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
bool simplePred(const ExprP& e, int& bt, int& bc, std::string& op, double& v) {
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
bool segMayMatch(const Segment& s, const std::vector<ExprP>& preds, bool skipAllDeleted = true) {
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
bool findPkEq(const std::vector<ExprP>& conj, const Table& t, int bt, int64_t& key) {
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

ColView viewOf(const void* p, PType t) { return ColView{p, (uint8_t)t}; }

// A `col = literal` conjunct on the first column of a secondary index: returns the index and the
// raw key (found = false when the literal can't occur, e.g. a string not in the dictionary).
const IndexMeta* findIndexEq(const std::vector<ExprP>& conj, const Table& t, int bt, int64_t& key, bool& found) {
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
std::shared_ptr<DevVec<int32_t>> dictRankDev(const Dict& d) {
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

void gatherCol(const void* src, PType pt, int esize, const int32_t* idx, int64_t m, void* dst, bool nulls) {
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

void bindRelCols(Compiler& comp, const Rel& r) {
  for (size_t s = 0; s < comp.slotCols.size(); ++s) {
    int i = r.find(comp.slotCols[s].first, comp.slotCols[s].second);
    if (i < 0) throw SqlError("internal: column not materialized", 1105);
    comp.env.cols[s].p = r.cols[i].data.p;
    comp.env.cols[s].enc = 0;  // materialized columns are decoded
  }
}

using Clock = std::chrono::steady_clock;
double msSince(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

// ---- PROFILE support (per session = per connection thread, like SingleStore's SHOW PROFILE).
struct ProfRow {
  std::string stage;
  double ms;
  int64_t rows;
  std::string detail;
};
thread_local bool tProfOn = false;
thread_local std::vector<ProfRow> tProf, tLastProf;
thread_local bool tDense = true;  // SET warpstore_dense_groupby

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
std::vector<int> partsToScan(const Table& t, int bt, const std::vector<ExprP>& preds) {
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
thread_local int tScanDelayMs = 0;  // SET warpstore_debug_scan_delay_ms (tests: a slow reader holding its snapshot)
void scanTable(Table& t, const std::vector<int>& parts, const std::vector<ExprP>& preds, ScanStats& st,
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
void forEachBatch(BindCtx& ctx, int bt, Compiler& comp, const std::vector<ExprP>& preds, ScanStats& st,
                  const std::function<bool(ExprEnv&, int64_t, const uint32_t*)>& fn, bool dryRun = false) {
  Table& t = *ctx.tabs[bt];
  scanTable(t, partsToScan(t, bt, preds), preds, st, [&](const BatchRef& b) {
    for (size_t k = 0; k < comp.slotCols.size(); ++k) bindView(comp.env.cols[k], b, comp.slotCols[k].second);
    return fn(comp.env, b.n, b.del);
  }, dryRun);
}

std::string scanDesc(const Table& t, const ScanStats& st) {
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
bool colRange(const Table& t, int bc, int64_t& lo, int64_t& hi) {
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

thread_local bool tJoinFilters = true;            // SET warpstore_join_filters = ON|OFF
thread_local bool tAnn = true;                    // SET warpstore_vector_index = ON|OFF (OFF: exact GPU search)
thread_local int tNprobe = 0;                     // SET warpstore_nprobe = N (0: index default)
thread_local int tCagraItopk = 0;                 // SET warpstore_cagra_itopk = N (0: index default)

thread_local int64_t tLazyRows = 2'000'000;        // SET warpstore_join_filter_rows = N (tables this big are filtered)

// Would this expression compile for the GPU? (Host-only string functions over several columns don't.)
bool gpuCompilable(const ExprP& e, BindCtx& ctx) {
  try {
    Compiler c(&ctx);
    c.compile(e);
    return true;
  } catch (const SqlError&) {
    return false;
  }
}

// One stored cell as a typed host value (NULL-aware).
Value cellValue(const Table& t, Loc l, int c) {
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
void cellRaw(const Table& t, Loc l, int c, uint8_t* out) {
  if (l.seg >= 0 && segCold(*t.segs[l.seg]->d, c)) {
    segRawCell(*t.segs[l.seg]->d, c, l.row, out);
    return;
  }
  const int w = t.colBytes(c);
  memcpy(out, t.colData(l, c).data() + (size_t)l.row * w, w);
}

// locs: also the location of every output row (late materialization of other columns).
Rel materialize(BindCtx& ctx, int bt, const std::vector<ExprP>& preds, const std::set<int>& needCols, int64_t maxRows,
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
std::pair<std::vector<int32_t>, std::vector<int32_t>> joinCodeMaps(const Dict& a, const Dict& b) {
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

Rel hashJoin(Rel& a, Rel& b, int abt, int abc, int bbt, int bbc, std::vector<std::string>& notes,
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
Rel outerJoin(Rel& a, Rel& b, int lt, int lc, int rt, int rc, int type, std::vector<std::string>& notes,
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

Rel crossJoin(Rel& a, Rel& b, std::vector<std::string>& notes) {
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
std::vector<Value> defaultRow(const Table& t) {
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

ResultSet okResult(int64_t affected, std::string info = "") {
  ResultSet r;
  r.affected = affected;
  r.info = std::move(info);
  return r;
}

}  // namespace

// ================================================================== Engine

// ---- Pipelines (SingleStore CREATE PIPELINE .. AS LOAD DATA FS '/dir/*.csv' INTO TABLE t): a
// background thread polls the glob and loads each new file once, through the normal (logged) LOAD.
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


// WAL fsync counters (/metrics).
static std::atomic<uint64_t> gWalSyncs{0}, gWalSyncUs{0}, gWalSyncMaxUs{0};
static const int64_t gStartUs = nowUs();
static void timedSync(int fd) {
  const auto t0 = Clock::now();
  fdatasync(fd);
  const uint64_t us = (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - t0).count();
  ++gWalSyncs;
  gWalSyncUs += us;
  for (uint64_t m = gWalSyncMaxUs; us > m && !gWalSyncMaxUs.compare_exchange_weak(m, us);) {}
}

// Prometheus metrics (GET /metrics on the HTTP port): the session / statement counters from the
// monitor plus storage, WAL, VRAM, replication and plan cache gauges.
std::string Engine::sessionUserName() const { return tUserName(); }

std::string Engine::metrics() {
  std::string o = monitor::prometheus();
  char b[512];
  auto metric = [&](const char* name, const char* type, const char* help, double v, const std::string& labels = "") {
    if (o.find(std::string("# TYPE ") + name + " ") == std::string::npos) {
      snprintf(b, sizeof b, "# HELP %s %s\n# TYPE %s %s\n", name, help, name, type);
      o += b;
    }
    snprintf(b, sizeof b, "%s%s %.17g\n", name, labels.empty() ? "" : ("{" + labels + "}").c_str(), v);
    o += b;
  };
  metric("warpstore_up", "gauge", "1 while the server runs.", 1);
  metric("warpstore_uptime_seconds", "gauge", "Seconds since start.", (nowUs() - gStartUs) / 1e6);
  std::vector<std::shared_ptr<Table>> tabs;
  size_t dbs;
  {
    std::lock_guard<std::mutex> g(cat_.mu);
    for (auto& [k, t] : cat_.tables)
      if (t->db != "__tmp") tabs.push_back(t);
    dbs = cat_.dbs.size();
  }
  int64_t rows = 0, segs = 0, bufRows = 0;
  for (auto& t : tabs) {
    std::shared_lock<FairSharedMutex> lk(t->mu, std::try_to_lock);  // (skip tables a writer holds)
    if (!lk.owns_lock()) continue;
    rows += t->liveRows();
    segs += (int64_t)t->segs.size();
    for (auto& wb : t->bufs) bufRows += (int64_t)wb->n;
  }
  metric("warpstore_databases", "gauge", "Databases.", (double)dbs);
  metric("warpstore_tables", "gauge", "Tables.", (double)tabs.size());
  metric("warpstore_rows", "gauge", "Live rows in all tables.", (double)rows);
  metric("warpstore_segments", "gauge", "Columnar segments.", (double)segs);
  metric("warpstore_write_buffer_rows", "gauge", "Rows not yet flushed to segments.", (double)bufRows);
  const int64_t lim = gpuVramLimit(), freeB = gpuVramFree();
  metric("warpstore_vram_free_bytes", "gauge", "Free device memory.", (double)freeB);
  if (lim != INT64_MAX) metric("warpstore_vram_limit_bytes", "gauge", "--vram-limit.", (double)lim);
  {
    std::lock_guard<std::mutex> g(walMu_);
    metric("warpstore_wal_generation", "gauge", "Current WAL generation (goes up at each SNAPSHOT).", (double)walGen_);
    metric("warpstore_wal_bytes", "gauge", "Bytes in the current WAL.", (double)walOff_);
    metric("warpstore_wal_records_total", "counter", "WAL records written since start.", (double)walWritten_);
    metric("warpstore_replication_lsn", "gauge", "Replication log position (primary).", (double)lsn_);
  }
  metric("warpstore_wal_syncs_total", "counter", "WAL fdatasync calls.", (double)gWalSyncs.load());
  metric("warpstore_wal_sync_seconds_total", "counter", "Time spent in WAL fdatasync.", gWalSyncUs.load() / 1e6);
  metric("warpstore_wal_sync_max_seconds", "gauge", "Slowest WAL fdatasync since start.", gWalSyncMaxUs.load() / 1e6);
  metric("warpstore_replica", "gauge", "1 on a replica, 0 on a primary.", isReplica_ ? 1 : 0);
  if (isReplica_) {
    metric("warpstore_replica_applied_lsn", "gauge", "Last applied primary log position.", (double)appliedLsn_.load());
    const int64_t cu = replCaughtUpUs_;
    if (cu) metric("warpstore_replica_lag_seconds", "gauge", "Time since the replica was last caught up.", std::max<int64_t>(0, nowUs() - cu - 1000000) / 1e6);
  } else {
    std::lock_guard<std::mutex> g(replMu_);
    int live = 0;
    for (auto& [id, a] : replAcks_) live += nowUs() - a.second < 10000000;
    metric("warpstore_replicas_connected", "gauge", "Replicas seen in the last 10 s.", live);
  }
  {
    auto pc = jitPlanCache();
    double compileMs = 0;
    for (auto& p : pc) compileMs += p.compileMs;
    metric("warpstore_plan_cache_plans", "gauge", "Compiled query plans cached.", (double)pc.size());
    metric("warpstore_plan_compile_seconds_total", "counter", "Time spent compiling the cached plans.", compileMs / 1000);
  }
  {
    std::lock_guard<std::mutex> g(pipesMu_);
    metric("warpstore_pipelines", "gauge", "Pipelines defined.", (double)pipes_.size());
  }
  metric("warpstore_table_memory_bytes", "gauge", "Segment column data in host memory.", (double)tierHostBytes());
  metric("warpstore_disk_read_bytes_total", "counter", "Column data read from segment files by scans.", (double)tierReadBytes());
  if (tierEnabled() && tierMemLimit() != INT64_MAX) metric("warpstore_table_memory_limit_bytes", "gauge", "--memory-limit.", (double)tierMemLimit());
  if (FILE* f = fopen("/proc/self/statm", "r")) {
    long pages = 0, res = 0;
    if (fscanf(f, "%ld %ld", &pages, &res) == 2) metric("warpstore_resident_memory_bytes", "gauge", "Resident host memory.", (double)res * sysconf(_SC_PAGESIZE));
    fclose(f);
  }
  return o;
}

Engine::Engine(const std::string& dataDir, bool syncWal, int partitions)
    : dir_(dataDir), sync_(syncWal), nparts_(std::max(partitions, 1)) {
  vramTh_ = std::thread([this] {
    while (!vramStop_) {
      std::this_thread::sleep_for(std::chrono::milliseconds(250));
      try { tierBalance(); } catch (std::exception& e) { fprintf(stderr, "[warpstore] memory tier: %s\n", e.what()); }
      try { vramBalance(); } catch (...) {}
    }
  });
  maintTh_ = std::thread([this] {
    for (int tick = 1; !vramStop_; ++tick) {
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
      if (!recovered_) continue;
      if (tick % 10 == 0 && !envSetting("NO_COMPACT")) {
        try {
          compactTables();
        } catch (std::exception& e) {
          fprintf(stderr, "[warpstore] compaction: %s\n", e.what());
        }
      }
      if (dir_.empty() || isReplica_ || gSnapshotTriggerBytes <= 0) continue;
      int64_t w;
      {
        std::lock_guard<std::mutex> g(walMu_);
        w = walOff_;
      }
      const int64_t loaded = loadedBytes_.load();
      if (w + loaded < gSnapshotTriggerBytes) continue;  // (a big LOAD would replay from its file copy)
      const auto t0 = Clock::now();
      try {
        tSnapTryOnly = true;
        struct Off {
          ~Off() { tSnapTryOnly = false; }
        } off;
        writeSnapshot("", "");
        if (lastSnapSkipped_) continue;
        loadedBytes_ -= loaded;
        fprintf(stderr, "[warpstore] automatic snapshot (WAL at %.0f MB) in %.0f ms, writes paused %.0f ms\n", w / 1048576.0, msSince(t0),
                lastSnapPauseMs_.load());
      } catch (std::exception& e) {
        fprintf(stderr, "[warpstore] automatic snapshot failed: %s\n", e.what());
        for (int i = 0; i < 60 && !vramStop_; ++i) std::this_thread::sleep_for(std::chrono::seconds(1));  // (retry in a minute)
      }
    }
  });
  flushTh_ = std::thread([this] {  // async durability: everything committed is on disk within ~interval
    const int ms = envSetting("FLUSH_MS") ? std::max(1, atoi(envSetting("FLUSH_MS"))) : 10;
    while (!vramStop_) {
      std::this_thread::sleep_for(std::chrono::milliseconds(ms));
      bool behind;
      {
        std::lock_guard<std::mutex> g(walMu_);
        std::lock_guard<std::mutex> g2(syncMu_);
        behind = wal_ && walWritten_ > walSynced_;
      }
      if (behind) walSyncNow();
    }
  });
}

// Merges segments full of deleted rows, or small ones (Table::compactPlan): one partition at a
// time, built outside the table lock.
void Engine::compactTables() {
  std::vector<std::shared_ptr<Table>> tabs;
  {
    std::lock_guard<std::mutex> g(cat_.mu);
    for (auto& [k, t] : cat_.tables)
      if (t->db != "__tmp") tabs.push_back(t);
  }
  for (auto& t : tabs) {
    std::vector<CompactJob> jobs;
    {
      std::shared_lock<FairSharedMutex> lk(t->mu);
      jobs = t->compactPlan();
    }
    for (auto& j : jobs) {
      if (vramStop_) return;
      const auto t0 = Clock::now();
      int64_t rows = 0, dead = 0;
      for (size_t k = 0; k < j.olds.size(); ++k) {
        rows += j.olds[k]->n;
        for (uint32_t w : j.bits[k]) dead += __builtin_popcount(w);
      }
      t->compactBuild(j);
      bool ok;
      {
        std::unique_lock<FairSharedMutex> lk(t->mu);
        ok = t->compactInstall(j);
      }
      if (ok && envSetting("TRACE"))
        fprintf(stderr, "[trace] compacted %s partition %d: %zu segments (%lld rows, %lld dead) -> %zu in %.0f ms\n", t->name.c_str(), j.part,
                j.olds.size(), (long long)rows, (long long)dead, j.news.size(), msSince(t0));
    }
  }
}

// Host memory under --memory-limit (see SegFile): over 90% of the limit, the coldest column data is
// evicted to the segment files (written first if needed) until 80% is used; under 75%, cold columns
// are read back, hottest first (after a restart everything is cold: the cache fills up again).
void Engine::tierBalance() {
  if (!tierEnabled()) return;
  const int64_t limit = tierMemLimit();
  const int64_t used = tierHostBytes() - tierPendingBytes();
  struct Ent {
    std::shared_ptr<Table> t;
    std::shared_ptr<Segment> seg;
    int c;
    float heat;
    int64_t bytes;
  };
  const bool evict = limit != INT64_MAX && used > limit / 10 * 9;
  const bool warm = limit == INT64_MAX || used < limit / 4 * 3;
  if (!evict && !warm) return;
  std::vector<std::shared_ptr<Table>> tabs;
  {
    std::lock_guard<std::mutex> g(cat_.mu);
    for (auto& [k, t] : cat_.tables)
      if (t->db != "__tmp") tabs.push_back(t);
  }
  std::vector<Ent> ents;
  for (auto& t : tabs) {
    std::shared_lock<FairSharedMutex> lk(t->mu);
    std::vector<uint8_t> pinned(t->cols.size(), 0);
    for (size_t c = 0; c < t->cols.size(); ++c) pinned[c] = t->vramHint((int)c) == 1;
    for (auto& ix : t->indexes)
      if (ix.vec && ix.algo == 1) pinned[ix.cols[0]] = 1;  // CAGRA graphs refer to the column's VRAM
    for (auto& sp : t->segs) {
      Segment& sg = *sp;
      if (sg.uses.size() != sg.dev.size() || sg.n == 0 || sg.d->cold.size() != sg.dev.size()) continue;
      for (size_t c = 0; c < sg.dev.size(); ++c) {
        const bool cold = segCold(*sg.d, (int)c);
        if (cold) {  // (present columns' heat is kept by vramBalance)
          const uint32_t u = __atomic_exchange_n(&sg.uses[c], 0u, __ATOMIC_RELAXED);
          sg.heat[c] = sg.heat[c] * 0.9f + (float)u;
        }
        if (evict == cold || (evict && pinned[c])) continue;
        const auto& fc = cold ? sg.d->file->cols[c] : SegFile::Col{};
        const int64_t b = cold ? (int64_t)fc.len + (fc.enc ? sg.n * (int64_t)fc.width : 0)
                               : (int64_t)sg.host[c].size() + (sg.enc[c] ? (int64_t)sg.encHost[c].size() : 0);
        ents.push_back({t, sp, (int)c, sg.heat[c], b});
      }
    }
  }
  if (evict) {
    std::stable_sort(ents.begin(), ents.end(), [](const Ent& a, const Ent& b) { return a.heat < b.heat; });  // coldest first
    int64_t need = used - limit / 10 * 8, got = 0;
    size_t k = 0;
    while (k < ents.size() && got < need) got += ents[k++].bytes;
    ents.resize(k);
    for (auto& e : ents) segFileEnsure(*e.seg->d, e.seg->n, e.t->sortKey);  // (outside the table lock: the data is immutable)
    std::map<Table*, std::vector<const Ent*>> byTable;
    for (auto& e : ents) byTable[e.t.get()].push_back(&e);
    for (auto& [tp, list] : byTable) {
      std::vector<std::tuple<std::shared_ptr<SegData>, int, uint32_t>> done;
      {
        std::unique_lock<FairSharedMutex> lk(tp->mu);  // no reader of the live table uses the columns now
        std::set<const Segment*> alive;
        for (auto& sp : tp->segs) alive.insert(sp.get());
        for (auto* e : list) {
          if (!alive.count(e->seg.get())) continue;
          uint32_t gen = 0;
          if (segEvictBegin(*e->seg->d, e->c, gen)) done.push_back({e->seg->d, e->c, gen});
        }
      }
      // Queries on snapshots may still read the data: freed once those that started earlier finish.
      if (!done.empty())
        rcuDefer([done] {
          for (auto& [d, c, gen] : done) segEvictFinish(*d, c, gen);
        });
    }
    return;
  }
  std::stable_sort(ents.begin(), ents.end(), [](const Ent& a, const Ent& b) { return a.heat > b.heat; });  // hottest first
  int64_t room = limit == INT64_MAX ? INT64_MAX : limit / 4 * 3 - used, loaded = 0;
  for (auto& e : ents) {
    if (loaded >= (256ll << 20) || e.bytes > room - loaded) break;
    segLoad(*e.seg->d, e.c);
    loaded += e.bytes;
  }
}

// When columns live on the host tier (over --vram-limit, or VRAM was full), keep the columns queries
// actually read in VRAM: per-column read counts decay into heat; the hottest columns that fit the
// budget are wanted in VRAM, resident columns outside that set are moved out first to make room.
// Copies and mappings are made outside the table lock; only the pointer swap takes it exclusively
// (running queries hold it shared, so none still reads an old pointer when it is freed).
void Engine::vramBalance(int64_t shed) {
  struct Ent {
    std::shared_ptr<Table> t;
    Segment* seg;
    int c;
    size_t bytes;
    bool resident;
    float heat;  // ordering key: PIN first, NEVER last, otherwise the decayed read count
    int hint;    // 0 AUTO, 1 PIN, 2 NEVER
  };
  static const bool off = envSetting("NO_VRAM_CACHE") != nullptr;  // static placement (A/B comparisons)
  if (off) return;
  if (shed <= 0 && nowUs() < vramHoldUntilUs_.load()) return;
  std::vector<std::shared_ptr<Table>> tabs;
  {
    std::lock_guard<std::mutex> g(cat_.mu);
    for (auto& [k, t] : cat_.tables) tabs.push_back(t);
  }
  std::vector<Ent> ents;
  bool anyMapped = false, neverResident = false;
  int64_t resident = 0;
  for (auto& t : tabs) {
    std::shared_lock<FairSharedMutex> lk(t->mu);
    for (auto& sp : t->segs) {
      Segment& sg = *sp;
      if (sg.uses.size() != sg.dev.size()) continue;  // GPU-built temp segments: not cacheable
      for (size_t c = 0; c < sg.dev.size(); ++c) {
        if (segCold(*sg.d, (int)c)) continue;  // (disk tier: tierBalance)
        const std::vector<uint8_t>& h = sg.enc[c] ? sg.encHost[c] : sg.host[c];
        if (!sg.dev[c] || h.empty()) continue;
        const uint32_t u = __atomic_exchange_n(&sg.uses[c], 0u, __ATOMIC_RELAXED);
        sg.heat[c] = sg.heat[c] * 0.9f + (float)u;
        int hint = t->vramHint((int)c);
        for (auto& ix : t->indexes)  // a CAGRA graph refers to the column's VRAM: it must stay there
          if (ix.vec && ix.algo == 1 && ix.cols[0] == (int)c) hint = 1;
        anyMapped |= sg.mapped[c] != 0;
        neverResident |= hint == 2 && !sg.mapped[c];
        if (!sg.mapped[c]) resident += (int64_t)h.size();
        ents.push_back({t, &sg, (int)c, h.size(), !sg.mapped[c], hint == 1 ? 1e30f : hint == 2 ? -1.f : sg.heat[c], hint});
      }
    }
  }
  if (shed <= 0 && !anyMapped && !neverResident) return;
  const int64_t limit = gpuVramLimit();
  const int64_t budget = limit != INT64_MAX ? limit : resident + std::max<int64_t>(0, gpuVramFree() - (1ll << 30));
  // Hottest first; at equal heat resident columns stay (no churn).
  std::sort(ents.begin(), ents.end(), [](const Ent& a, const Ent& b) {
    if (a.heat != b.heat) return a.heat > b.heat;
    return a.resident > b.resident;
  });
  int64_t cum = 0;
  std::vector<const Ent*> promote, demote;
  for (auto& e : ents) {
    const bool want = e.hint != 2 && e.heat > 0.05f && cum + (int64_t)e.bytes <= budget;
    if (want) cum += (int64_t)e.bytes;
    if (want && !e.resident) promote.push_back(&e);
    // Resident and not wanted: only moved out when the space is needed for a hotter column (VRAM =
    // NEVER: always).
    if (!want && e.resident) demote.push_back(&e);
  }
  if (shed > 0) {  // a query needs the room: every resident column is a candidate (coldest first, below)
    promote.clear();
    demote.clear();
    for (auto& e : ents)
      if (e.resident && e.hint != 1) demote.push_back(&e);
  }
  if (promote.empty() && !neverResident && shed <= 0) return;
  int64_t need = 0;
  for (auto* e : promote) need += (int64_t)e->bytes;
  need = shed > 0 ? shed : std::min<int64_t>(need, 256ll << 20);  // bounded work per round
  const int64_t room = shed > 0 ? 0 : budget - resident;
  // Coldest first among the demotion candidates; a column is only moved out for one at least 1.5x
  // hotter (no ping-pong when the working set is larger than the budget).
  std::reverse(demote.begin(), demote.end());
  {
    float coldestIn = INFINITY;
    int64_t acc = 0;
    for (auto* e : promote) {
      if (acc >= need) break;
      coldestIn = std::min(coldestIn, e->heat);
      acc += (int64_t)e->bytes;
    }
    if (shed <= 0)
      demote.erase(std::remove_if(demote.begin(), demote.end(), [&](const Ent* e) { return e->hint != 2 && e->heat * 1.5f >= coldestIn; }), demote.end());
    std::stable_partition(demote.begin(), demote.end(), [](const Ent* e) { return e->hint == 2; });  // NEVER first
  }
  struct Move {
    const Ent* e;
    void* nptr;
  };
  std::map<Table*, std::vector<Move>> moves;
  int64_t freed = 0;
  auto hostOf = [](Segment& sg, int c) -> std::vector<uint8_t>& { return sg.enc[c] ? sg.encHost[c] : sg.host[c]; };
  auto alive = [](Table& t, const Segment* sg) {
    for (auto& sp : t.segs)
      if (sp.get() == sg) return true;
    return false;
  };
  for (auto* e : demote) {
    if (e->hint != 2 && room + freed >= need) break;
    std::shared_lock<FairSharedMutex> lk(e->t->mu);  // the segment stays alive while it is mapped
    if (!alive(*e->t, e->seg)) continue;
    std::vector<uint8_t>& h = hostOf(*e->seg, e->c);
    void* d = hostMap(h.data(), h.size());
    if (!d) continue;
    moves[e->t.get()].push_back({e, d});
    freed += (int64_t)e->bytes;
  }
  // Swap demotions in first (frees VRAM), then copy the promotions in.
  auto apply = [&](std::map<Table*, std::vector<Move>>& mv, bool toVram) {
    for (auto& [tp, list] : mv) {
      std::vector<std::pair<void*, size_t>> oldDev;  // freed after the swap
      std::vector<void*> oldHost;
      {
        std::unique_lock<FairSharedMutex> lk(tp->mu);
        for (auto& m : list) {
          const bool live = alive(*tp, m.e->seg);
          if (!live) {  // segment gone: undo (its host bytes went with it)
            if (toVram) vramRelease(m.nptr, m.e->bytes);
            continue;
          }
          Segment& sg = *m.e->seg;
          std::vector<uint8_t>& h = hostOf(sg, m.e->c);
          if ((sg.mapped[m.e->c] != 0) == !toVram) {  // already moved: undo
            if (toVram) vramRelease(m.nptr, m.e->bytes);
            else hostUnmap(h.data());
            continue;
          }
          void* old = sg.dev[m.e->c];
          sg.dev[m.e->c] = m.nptr;
          sg.mapped[m.e->c] = toVram ? 0 : 1;
          if (toVram) oldHost.push_back(h.data());
          else oldDev.push_back({old, m.e->bytes});
        }
      }
      // MVCC: queries reading snapshots don't hold the table lock, so a statement that started
      // before the swap may still read the old copy: free it once those have finished.
      if (!oldDev.empty() || !oldHost.empty())
        rcuDefer([oldDev, oldHost] {
          for (auto& [p, b] : oldDev) vramRelease(p, b);
          for (void* p : oldHost) hostUnmap(p);
        });
    }
  };
  apply(moves, false);
  std::map<Table*, std::vector<Move>> ups;
  int64_t done = 0;
  for (auto* e : promote) {
    if (done >= need) break;
    std::shared_lock<FairSharedMutex> lk(e->t->mu);
    if (!alive(*e->t, e->seg)) continue;
    const std::vector<uint8_t>& h = hostOf(*e->seg, e->c);
    void* d = vramAcquire(h.data(), h.size());
    if (!d) break;  // budget reached
    ups[e->t.get()].push_back({e, d});
    done += (int64_t)e->bytes;
  }
  apply(ups, true);
  if (envSetting("TRACE") && (freed || done))
    fprintf(stderr, "[trace] vram cache: moved %.0f MB out, %.0f MB in (budget %.0f MB)\n", freed / 1048576.0, done / 1048576.0, budget / 1048576.0);
}

Engine::~Engine() {
  vramStop_ = true;
  if (vramTh_.joinable()) vramTh_.join();
  if (maintTh_.joinable()) maintTh_.join();
  if (flushTh_.joinable()) flushTh_.join();
  if (wal_) {
    fflush(wal_);
    timedSync(fileno(wal_));
  }
  std::vector<std::shared_ptr<Pipeline>> ps;
  {
    std::lock_guard<std::mutex> g(pipesMu_);
    for (auto& [n, p] : pipes_) ps.push_back(p);
  }
  for (auto& p : ps) stopPipeline(*p);
  if (wal_) fclose(wal_);
}

// Replays one WAL file (format 1: u64 length + payload; format 2: header, u32 length + u32 CRC32 +
// payload). Stops at the first torn or corrupt record; repairTail truncates the file there, so
// records appended later are not hidden behind garbage. Returns the number of records applied.
int64_t Engine::replayWal(const std::string& path, bool repairTail) {
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) return 0;
  fseeko(f, 0, SEEK_END);
  const int64_t size = ftello(f);
  fseeko(f, 0, SEEK_SET);
  char hdr[16];
  const size_t got = fread(hdr, 1, 16, f);
  const bool v3 = got == 16 && memcmp(hdr, "GDBWAL3\n", 8) == 0;
  const bool v2 = v3 || (got >= 8 && memcmp(hdr, "GDBWAL2\n", 8) == 0);  // (records: same layout)
  fseeko(f, v3 ? 16 : v2 ? 8 : 0, SEEK_SET);
  int64_t good = ftello(f), n = 0;
  replaying_ = true;
  auto apply = [&](std::string sql) {
    size_t sep = sql.find('\x1f');
    if (sep != std::string::npos) {  // "<db>\x1f<statement>"
      currentDb() = sql.substr(0, sep);
      if (currentDb() == "gpudb") currentDb() = "warpstore";  // (WALs from before the rename)
      sql = sql.substr(sep + 1);
    }
    try {
      execute(sql);
    } catch (std::exception& e) {
      // A statement that failed when it ran is only in the WAL if it changed rows before failing (a
      // partial load, marked so): its replay ends in the same error. Any other error means the
      // replay would lose data: recovery stops (WARPSTORE_REPLAY_IGNORE_ERRORS=1: go on anyway).
      static const bool ignore = envSetting("REPLAY_IGNORE_ERRORS") != nullptr;
      const bool expected = sql.find("-partial.") != std::string::npos;
      fprintf(stderr, "WAL replay error%s: %s\n  in: %.300s\n", expected ? " (as when it ran)" : "", e.what(), sql.c_str());
      if (!expected && !ignore)
        throw std::runtime_error(std::string("WAL replay failed (") + e.what() +
                                 "); the data dir is left untouched. WARPSTORE_REPLAY_IGNORE_ERRORS=1 starts without that statement");
    }
  };
  for (;;) {
    const int64_t at = ftello(f);
    uint64_t len = 0;
    uint32_t crc = 0;
    if (v2) {
      uint32_t l32;
      if (fread(&l32, 4, 1, f) != 1 || fread(&crc, 4, 1, f) != 1) break;
      len = l32;
    } else if (fread(&len, 8, 1, f) != 1) {
      break;
    }
    if (v2 && len == 0) break;  // the pre-zeroed space after the last record
    if (len > (uint64_t)(size - at)) break;  // torn (or garbage) length
    std::string rec(len, '\0');
    if (fread(rec.data(), 1, len, f) != len) break;
    if (v2 && (uint32_t)crc32(0, (const Bytef*)rec.data(), (uInt)len) != crc) {
      fprintf(stderr, "[warpstore] WAL %s: checksum mismatch at offset %lld; ignoring the rest\n", path.c_str(), (long long)at);
      break;
    }
    if (rec.rfind("\x1eTS", 0) == 0) {  // time marker
    } else if (rec.rfind("\x1eTXN", 0) == 0) {  // a committed transaction: its statements, applied together
      size_t i = 4;
      while (i + 4 <= rec.size()) {
        uint32_t l;
        memcpy(&l, rec.data() + i, 4);
        i += 4;
        if (i + l > rec.size()) break;
        apply(rec.substr(i, l));
        i += l;
      }
    } else {
      apply(rec);
    }
    ++n;
    good = ftello(f);
  }
  fclose(f);
  replaying_ = false;
  if (repairTail && good < size) {  // pre-zeroed space (normal) or a torn / corrupt tail: cut it off either way
    bool zeros = true;
    if (FILE* g = fopen(path.c_str(), "rb")) {
      fseeko(g, good, SEEK_SET);
      char buf[65536];
      size_t k;
      while (zeros && (k = fread(buf, 1, sizeof buf, g)) > 0)
        for (size_t i = 0; i < k && zeros; ++i) zeros = buf[i] == 0;
      fclose(g);
    }
    if (!zeros) fprintf(stderr, "[warpstore] WAL %s: dropping incomplete records at the end (offset %lld)\n", path.c_str(), (long long)good);
    if (truncate(path.c_str(), good) != 0) perror("truncate WAL");
  }
  if (repairTail && !v2 && size > 0) {  // upgrade a format-1 log: rewrite it in format 2
    std::vector<std::string> keep;
    if (FILE* g = fopen(path.c_str(), "rb")) {
      for (;;) {
        uint64_t len;
        if (fread(&len, 8, 1, g) != 1) break;
        std::string rec(len, '\0');
        if (fread(rec.data(), 1, len, g) != len) break;
        keep.push_back(std::move(rec));
      }
      fclose(g);
    }
    const std::string tmp = path + ".v2";
    if (FILE* g = fopen(tmp.c_str(), "wb")) {
      fwrite("GDBWAL2\n", 1, 8, g);
      for (auto& rec : keep) {
        const uint32_t l = (uint32_t)rec.size(), c = (uint32_t)crc32(0, (const Bytef*)rec.data(), (uInt)rec.size());
        fwrite(&l, 4, 1, g);
        fwrite(&c, 4, 1, g);
        fwrite(rec.data(), 1, rec.size(), g);
      }
      fflush(g);
      fdatasync(fileno(g));
      fclose(g);
      rename(tmp.c_str(), path.c_str());
    }
  }
  return n;
}

// The WAL is written into space zeroed ahead of time (64 MB at a time), so a commit's fdatasync only
// flushes data blocks: the file size, and with it the filesystem journal, change once per chunk.
void Engine::walOpen(bool truncate) {
  const std::string path = dir_ + "/wal.log";
  wal_ = fopen(path.c_str(), truncate ? "w+b" : "r+b");
  if (!wal_ && !truncate) wal_ = fopen(path.c_str(), "w+b");
  if (!wal_) return;
  fseeko(wal_, 0, SEEK_END);
  walOff_ = walCap_ = ftello(wal_);  // after replay the file ends at its last record
  if (walOff_ == 0) {
    fwrite("GDBWAL3\n", 1, 8, wal_);
    fwrite(&walGen_, 8, 1, wal_);
    fflush(wal_);
    walOff_ = walCap_ = 16;
  } else {
    char hdr[16];
    fseeko(wal_, 0, SEEK_SET);
    if (fread(hdr, 1, 16, wal_) == 16 && memcmp(hdr, "GDBWAL3\n", 8) == 0) memcpy(&walGen_, hdr + 8, 8);
    fseeko(wal_, 0, SEEK_END);
  }
  walTsSec_ = 0;  // the next record gets a time marker
}

void Engine::walAppendLocked(const std::string& rec) {
  if (!wal_) return;
  const int64_t us = nowUs();
  if (us / 1000000 != walTsSec_) {  // time marker for point-in-time restores (at most one a second)
    walTsSec_ = us / 1000000;
    std::string ts = "\x1eTS";
    ts.append((const char*)&us, 8);
    walWriteLocked(ts);
  }
  walWriteLocked(rec);
}

void Engine::walWriteLocked(const std::string& rec) {
  const uint32_t len = (uint32_t)rec.size(), crc = (uint32_t)crc32(0, (const Bytef*)rec.data(), (uInt)rec.size());
  const int64_t need = 8 + (int64_t)rec.size();
  if (walOff_ + need > walCap_) {  // zero the next chunk (and make its size durable) before using it
    const int64_t chunk = std::max<int64_t>(64ll << 20, need);
    static const std::vector<char> zeros(1 << 20, 0);
    fseeko(wal_, walCap_, SEEK_SET);
    bool ok = true;
    for (int64_t w = 0; w < chunk && ok; w += (int64_t)zeros.size()) {
      const size_t k = (size_t)std::min<int64_t>(zeros.size(), chunk - w);
      ok = fwrite(zeros.data(), 1, k, wal_) == k;
    }
    ok = ok && fflush(wal_) == 0 && fdatasync(fileno(wal_)) == 0;
    if (!ok) {  // (a full disk shows here: records only ever go into space zeroed ahead)
      const std::string why = strerror(errno);
      clearerr(wal_);
      throw SqlError("cannot extend the WAL: " + why + " (disk full?)", 1021);
    }
    walCap_ += chunk;
  }
  fseeko(wal_, walOff_, SEEK_SET);
  const bool ok = fwrite(&len, 4, 1, wal_) == 1 && fwrite(&crc, 4, 1, wal_) == 1 && fwrite(rec.data(), 1, rec.size(), wal_) == rec.size() &&
                  fflush(wal_) == 0;
  if (!ok) {
    const std::string why = strerror(errno);
    clearerr(wal_);
    throw SqlError("cannot write the WAL: " + why, 1021);
  }
  walOff_ += need;
  ++walWritten_;
}

// Durability, as in SingleStore: async by default (a commit is acknowledged once it is in the WAL in
// the OS page cache, which survives a process crash; a background thread makes it durable on disk
// within ~10 ms, and writers are throttled if that falls behind), sync for databases created WITH
// SYNC DURABILITY, sessions with SET warpstore_sync_durability = ON, or every write with --sync-durability.
thread_local int tSyncDurability = -1;  // SET warpstore_sync_durability: -1 database / server default, 0 off, 1 on
bool Engine::needsSync(const std::string& db) {
  if (tSyncDurability >= 0) return tSyncDurability == 1;
  if (sync_) return true;
  std::lock_guard<std::mutex> g(cat_.mu);
  return !cat_.syncDbs.empty() && cat_.syncDbs.count(lower(db.empty() ? currentDb() : db)) > 0;
}
void Engine::walSyncNow() { walDurable(true); }

// Group commit: the first waiter syncs everything written so far; the others wait for that sync.
void Engine::walDurable(bool wait) {
  if (!wal_) return;
  if (!wait) {  // async durability: only throttle when the disk is far behind (64K records)
    uint64_t written;
    {
      std::lock_guard<std::mutex> g(walMu_);
      written = walWritten_;
    }
    {
      std::lock_guard<std::mutex> g(syncMu_);
      if (written - walSynced_ < 65536) return;
    }
  }
  uint64_t mine;
  {
    std::lock_guard<std::mutex> g(walMu_);
    mine = walWritten_;
  }
  std::unique_lock<std::mutex> lk(syncMu_);
  while (walSynced_ < mine) {
    if (syncing_) {
      syncCv_.wait(lk);
      continue;
    }
    syncing_ = true;
    lk.unlock();
    uint64_t target;
    int fd;
    {
      std::lock_guard<std::mutex> g(walMu_);
      target = walWritten_;
      fd = wal_ ? fileno(wal_) : -1;
    }
    if (fd >= 0) timedSync(fd);
    lk.lock();
    walSynced_ = std::max(walSynced_, target);
    syncing_ = false;
    syncCv_.notify_all();
  }
}

void Engine::recover() {
  if (dir_.empty()) {
    recovered_ = true;
    return;
  }
  replaying_ = true;
  loadSnapshot(dir_ + "/snapshot.bin");
  replaying_ = false;
  std::string path = dir_ + "/wal.log";
  // An interrupted SNAPSHOT (see writeSnapshot): snapshot.bin.tmp left over means the new snapshot was
  // not installed, so the rotated-out wal.old still has to be replayed; otherwise wal.old is covered.
  const std::string old = dir_ + "/wal.old";
  const bool tmpLeft = access((dir_ + "/snapshot.bin.tmp").c_str(), F_OK) == 0;
  if (tmpLeft) unlink((dir_ + "/snapshot.bin.tmp").c_str());
  {
    auto t0 = Clock::now();
    int64_t n = 0;
    if (access(old.c_str(), F_OK) == 0) {
      if (tmpLeft) n += replayWal(old, false);
      archiveWal(old);
    }
    n += replayWal(path, true);
    fprintf(stderr, "[warpstore] replayed %lld WAL records in %.0f ms\n", (long long)n, msSince(t0));
  }
  {
    std::vector<std::shared_ptr<Table>> all;
    {
      std::lock_guard<std::mutex> g(cat_.mu);
      for (auto& [k, t] : cat_.tables) all.push_back(t);
    }
    for (auto& t : all) warmDicts(t);
  }
  walOpen(false);
  segFilesCleanup();  // segment files of an interrupted snapshot or of segments dropped since
  // CAGRA vector indexes: compile cuVS's kernels and build the segment graphs in the background, so
  // the first vector query doesn't pay for either.
  {
    std::vector<std::shared_ptr<Table>> vt;
    {
      std::lock_guard<std::mutex> g(cat_.mu);
      for (auto& [k, t] : cat_.tables)
        for (auto& ix : t->indexes)
          if (ix.vec && ix.algo == 1) { vt.push_back(t); break; }
    }
    if (!vt.empty()) std::thread([vt] {
      try {
        for (auto& t : vt) {
          std::shared_lock<FairSharedMutex> lk(t->mu);
          for (auto& ix : t->indexes) {
            if (!(ix.vec && ix.algo == 1 && ix.cagra)) continue;
            cagraWarm(ix.dim, ix.metric == 0);
            const int vc = ix.cols[0];
            for (auto& sp : t->segs) {
              Segment& sg = *sp;
              if (segCold(*sg.d, vc)) segLoad(*sg.d, vc);  // (graph columns stay in memory)
              if (sg.n < 2048 || !sg.dev[vc] || sg.mapped[vc]) continue;
              std::lock_guard<std::mutex> g(ix.cagra->mu);
              if (ix.cagra->graphs.count(sg.id) || ix.cagra->failed.count(sg.id)) continue;
              std::string err;
              auto gr = cagraBuild((const float*)sg.dev[vc], sg.n, ix.dim, ix.metric == 0, ix.graphDegree, err);
              if (gr) ix.cagra->graphs[sg.id] = gr;
              else ix.cagra->failed.insert(sg.id);
            }
          }
        }
        fprintf(stderr, "[warpstore] CAGRA vector indexes ready (%s)\n", cagraStatus().c_str());
      } catch (...) {
      }
    }).detach();
  }
  // Pipelines that were running when the server stopped resume now (after replay, so their
  // loads are logged and files loaded before the restart are not loaded again).
  std::vector<std::shared_ptr<Pipeline>> resume;
  {
    std::lock_guard<std::mutex> g(pipesMu_);
    for (auto& [n, p] : pipes_)
      if (p->wantRun) resume.push_back(p);
  }
  for (auto& p : resume) startPipeline(p);
  if (!wal_) fprintf(stderr, "[warpstore] warning: cannot open WAL %s\n", path.c_str());
  recovered_ = true;
}

// Per-connection transaction: writes are visible immediately (read uncommitted), WAL records are
// held until COMMIT, and ROLLBACK replays the undo log backwards.
struct Txn {
  bool active = false;
  bool autocommit = true;
  bool needSync = false;  // a write to a SYNC DURABILITY database (or a sync session): COMMIT waits for the disk
  std::vector<std::string> wal;
  std::vector<UndoRec> undo;
  std::vector<std::pair<std::string, std::pair<size_t, size_t>>> savepoints;  // name -> (undo size, wal size)
};
thread_local Txn tTxn;

// ---- Read committed: an open transaction's changes are applied to the tables (so it reads its own
// writes) and recorded in Table::txnPending; other sessions' scans see the committed state through
// per-batch visibility bitmaps, and their writes to those tables wait for the COMMIT / ROLLBACK.
// Write conflicts wait for the blocking transactions (all open ones on the table, or only those that
// changed the given rows). Waits are registered in a wait-for graph: a wait that would close a cycle
// is a deadlock (1213, and the transaction is rolled back, as in MySQL); otherwise 10 s at most (1205).
std::mutex gWaitMu;
std::map<const void*, std::set<const void*>> gWaitsFor;  // waiting transaction -> transactions it waits for
std::set<const void*> blockersOf(const Table& t, const std::vector<Loc>* rows) {
  std::set<const void*> out;
  for (auto& [k, v] : t.txnPending) {
    if (k == txnKey() || v.commitSeq || (v.hidden.empty() && v.ghosts.empty())) continue;
    if (!rows) {
      out.insert(k);
      continue;
    }
    // (the transaction's rows hashed, usually far fewer than the statement's: no set of a big
    // UPDATE's matched rows when nothing is pending)
    std::unordered_set<uint64_t> theirs;
    theirs.reserve(v.hidden.size() + v.ghosts.size());
    for (const Loc& l : v.hidden) theirs.insert(TxnPending::key(l));
    for (const Loc& l : v.ghosts) theirs.insert(TxnPending::key(l));
    for (const Loc& l : *rows)
      if (theirs.count(TxnPending::key(l))) {
        out.insert(k);
        break;
      }
  }
  return out;
}
void waitTxnFree(Table& t, std::unique_lock<FairSharedMutex>& lk, const std::vector<Loc>* rows = nullptr) {
  const auto t0 = Clock::now();
  const void* me = txnKey();
  struct Unregister {
    const void* me;
    ~Unregister() {
      if (!me) return;
      std::lock_guard<std::mutex> g(gWaitMu);
      gWaitsFor.erase(me);
    }
  } unreg{me};
  for (;;) {
    const std::set<const void*> bl = blockersOf(t, rows);
    if (bl.empty()) return;
    if (me) {  // (outside a transaction a statement holds nothing anyone waits for: no cycle through it)
      std::lock_guard<std::mutex> g(gWaitMu);
      gWaitsFor[me] = bl;
      std::set<const void*> seen;
      std::vector<const void*> stack(bl.begin(), bl.end());
      while (!stack.empty()) {
        const void* x = stack.back();
        stack.pop_back();
        if (x == me) throw SqlError("Deadlock found when trying to get lock; try restarting transaction", 1213);
        if (!seen.insert(x).second) continue;
        auto it = gWaitsFor.find(x);
        if (it != gWaitsFor.end()) stack.insert(stack.end(), it->second.begin(), it->second.end());
      }
    }
    if (msSince(t0) > 10000) throw SqlError("Lock wait timeout exceeded; another transaction has uncommitted changes to " + t.name, 1205);
    lk.unlock();
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    lk.lock();
  }
}
// Row-level write conflicts: does another open transaction have uncommitted changes to these rows?
bool rowsPending(const Table& t, const std::vector<Loc>& rows) { return !blockersOf(t, &rows).empty(); }
// Inserts only conflict through unique keys (another transaction's uncommitted row with the same key).
bool hasUniqueKey(const Table& t) {
  if (t.pk >= 0 || !t.pkCols.empty()) return true;
  for (auto& ix : t.indexes)
    if (ix.unique) return true;
  return false;
}

// Commit gate: a COMMIT stamps all its change sets with its sequence number under the exclusive
// gate (no table locks: the stamp is atomic), and statements reading several tables take all their
// snapshots under the shared gate, so a transaction's commit shows in every table at once.
std::shared_mutex gCommitGate;

uint64_t oldestSnapshotSeq() {
  std::lock_guard<std::mutex> g(gSnapMu);
  return gSnaps.empty() ? UINT64_MAX : *gSnaps.begin();
}
static void beginTxnState() {
  static std::atomic<uintptr_t> n{0};
  tTxn.active = true;
  undoLog() = &tTxn.undo;
  tTxnToken = (++n << 1) | 1;  // (odd: never an address)
  tTxnEntries.clear();
}
// Drops committed change sets no snapshot needs, in every table (after a REPEATABLE READ snapshot ends).
static void gcRetired(Catalog& cat) {
  std::vector<std::shared_ptr<Table>> ts;
  {
    std::lock_guard<std::mutex> g(cat.mu);
    for (auto& [k, t] : cat.tables) ts.push_back(t);
  }
  for (auto& t : ts) {
    {
      std::shared_lock<FairSharedMutex> sl(t->mu);
      bool any = false;
      for (auto& [k, v] : t->txnPending) any |= v.commitSeq != 0;
      if (!any) continue;
    }
    std::unique_lock<FairSharedMutex> lk(t->mu);
    t->gcPending();
  }
}
static void endSnapshot(Catalog& cat) {
  if (!tSnapSeq) return;
  {
    std::lock_guard<std::mutex> g(gSnapMu);
    auto it = gSnaps.find(tSnapSeq);
    if (it != gSnaps.end()) gSnaps.erase(it);
  }
  tSnapSeq = 0;
  gcRetired(cat);
}

void Engine::commitTxn() {
  if (!tTxn.active) return;
  undoLog() = nullptr;
  // Durable (and acknowledged by semi-sync replicas) first, visible after: nobody reads a commit a
  // crash could take back. A failed WAL write (disk full) rolls the transaction back.
  if (!replaying_ && !tTxn.wal.empty()) {
    uint64_t myLsn;
    try {
      std::lock_guard<std::mutex> g(walMu_);
      std::string bundle = "\x1eTXN";  // one WAL record: replayed all or nothing
      for (auto& rec : tTxn.wal) {
        const uint32_t l = (uint32_t)rec.size();
        bundle.append((const char*)&l, 4);
        bundle += rec;
      }
      walAppendLocked(tTxn.wal.size() == 1 ? tTxn.wal[0] : bundle);
      for (auto& [t, e] : tTxnEntries) e->logged = true;  // (under walMu_: see TxnPending::logged)
      for (auto& rec : tTxn.wal) replAppend(rec);
      myLsn = lsn_;
    } catch (...) {
      undoLog() = &tTxn.undo;
      rollbackTxn();
      throw;
    }
    walDurable(tTxn.needSync || needsSync(""));
    replWaitAckAfterGate(myLsn);  // semi-sync replication
    static const int delayMs = envSetting("DEBUG_COMMIT_DELAY_MS") ? std::atoi(envSetting("DEBUG_COMMIT_DELAY_MS")) : 0;
    if (delayMs) std::this_thread::sleep_for(std::chrono::milliseconds(delayMs));  // (tests: widens logged -> visible)
  }
  {
    std::unique_lock<std::shared_mutex> g(gCommitGate);
    const uint64_t seq = ++gCommitSeq;
    for (auto& [t, e] : tTxnEntries) e->commitSeq = seq;
  }
  for (auto& [t, e] : tTxnEntries) {  // tidy up now where the table is free (else later: flush, GC)
    std::unique_lock<FairSharedMutex> lk(t->mu, std::try_to_lock);
    if (lk.owns_lock()) t->gcPending();
  }
  tTxnEntries.clear();
  tTxnToken = 0;
  endSnapshot(cat_);
  tTxn.needSync = false;
  tTxn.wal.clear();
  tTxn.savepoints.clear();
  tTxn.undo.clear();
  tTxn.active = false;
}

// Replays undo records [mark, end) newest first.
static void undoRecords(std::vector<UndoRec>& undo, size_t mark) {
  // Re-inserted rows join the transaction's hidden rows (txnKey() stays set; rows it inserted are
  // hidden already), so other sessions keep seeing the pre-transaction state.
  for (size_t i = undo.size(); i-- > mark;) {
    UndoRec& u = undo[i];
    Table& t = *u.t;
    std::unique_lock<FairSharedMutex> lk(t.mu);
    if (!u.inserted) {
      t.appendRow(u.row);
      continue;
    }
    // Find the inserted row again (by primary key, else by value, newest first) and delete it.
    auto same = [&](Loc l) {
      std::vector<Value> v = t.rowValues(l);
      for (size_t c = 0; c < v.size(); ++c)
        if (v[c].isNull() != u.row[c].isNull() || (!v[c].isNull() && cmpVal(v[c], u.row[c]) != 0)) return false;
      return true;
    };
    bool done = false;
    if (t.pk >= 0 && !u.row[t.pk].isNull()) {
      auto it = t.pkIndex.find((int64_t)numOf(u.row[t.pk]));
      if (it != t.pkIndex.end()) { t.deleteAt(it->second); done = true; }
    }
    for (int p = 0; p < t.nparts && !done; ++p)
      for (int64_t r = t.bufs[p]->n; r-- > 0 && !done;)
        if (!t.bufs[p]->del->test(r) && same(bufLoc(p, r))) { t.deleteAt(bufLoc(p, r)); done = true; }
    for (size_t si = t.segs.size(); si-- > 0 && !done;)
      for (int64_t r = t.segs[si]->n; r-- > 0 && !done;)
        if (!t.segs[si]->del->test(r) && same(Loc{(int32_t)si, (int32_t)r})) {
          t.deleteAt(Loc{(int32_t)si, (int32_t)r});
          done = true;
        }
  }
}

void Engine::rollbackTxn() {
  if (!tTxn.active) return;
  undoLog() = nullptr;
  undoRecords(tTxn.undo, 0);  // (re-inserted rows join the transaction's hidden rows)
  for (auto& [t, e] : tTxnEntries) {  // then the change sets go: the table shows its state from before
    std::unique_lock<FairSharedMutex> lk(t->mu);
    t->txnPending.erase(txnKey());
  }
  tTxnEntries.clear();
  tTxnToken = 0;
  endSnapshot(cat_);
  tTxn.wal.clear();
  tTxn.undo.clear();
  tTxn.savepoints.clear();
  tTxn.active = false;
}

thread_local bool tIsoSnapshot = false;  // SET TRANSACTION ISOLATION LEVEL REPEATABLE READ / SERIALIZABLE
std::string isolationLevel() { return tIsoSnapshot ? "REPEATABLE-READ" : "READ-COMMITTED"; }

ResultSet Engine::doTxn(const std::string& what) {
  if (what == "BEGIN" || what == "BEGIN SNAPSHOT") {
    commitTxn();  // MySQL: BEGIN implicitly commits an open transaction
    beginTxnState();
    if (what == "BEGIN SNAPSHOT" || tIsoSnapshot) {  // the snapshot: every commit so far
      std::shared_lock<std::shared_mutex> gate(gCommitGate);  // (no commit half published)
      std::lock_guard<std::mutex> g(gSnapMu);
      tSnapSeq = gCommitSeq.load();
      gSnaps.insert(tSnapSeq);
    }
  } else if (what == "COMMIT") {
    commitTxn();
  } else if (what == "ROLLBACK") {
    rollbackTxn();
  } else if (what.rfind("SAVEPOINT ", 0) == 0 || what.rfind("ROLLBACK TO ", 0) == 0 || what.rfind("RELEASE ", 0) == 0) {
    const std::string name = lower(what.substr(what.rfind(' ') + 1));
    auto& sp = tTxn.savepoints;
    auto it = std::find_if(sp.begin(), sp.end(), [&](const auto& x) { return x.first == name; });
    if (what.rfind("SAVEPOINT ", 0) == 0) {
      if (!tTxn.active) return ResultSet{};  // autocommit: nothing to roll back to
      if (it != sp.end()) sp.erase(it);
      sp.push_back({name, {tTxn.undo.size(), tTxn.wal.size()}});
      return ResultSet{};
    }
    if (it == sp.end()) throw SqlError("SAVEPOINT " + name + " does not exist", 1305);
    if (what.rfind("RELEASE ", 0) == 0) {
      sp.erase(it, sp.end());
      return ResultSet{};
    }
    const size_t undoMark = it->second.first, walMark = it->second.second;
    undoLog() = nullptr;
    undoRecords(tTxn.undo, undoMark);
    tTxn.undo.resize(undoMark);
    tTxn.wal.resize(walMark);
    sp.erase(it + 1, sp.end());  // later savepoints are gone; this one stays
    undoLog() = &tTxn.undo;
    return ResultSet{};
  }
  if (!tTxn.autocommit && !tTxn.active) beginTxnState();  // autocommit = 0: the next transaction starts right away
  ResultSet r;
  return r;
}

// ---- MVCC: a SELECT reads per-statement snapshots of its tables (Table::shadow), taken under the
// table lock for microseconds; the query itself runs without the live tables' locks, so writers
// never wait for it. Lookups by key (PRIMARY KEY / secondary index equality) and the IVF vector
// index read the live table under its lock instead (they hold it for microseconds and need the
// index maps).
struct SnapScope {
  std::map<const Table*, std::shared_ptr<Table>> shadows;
  std::vector<std::shared_ptr<Table>> bases;  // the live tables, kept alive while their snapshots are used
  uint64_t ticket = rcuEnter();               // memory the VRAM balancer frees waits for this statement
  ~SnapScope() {
    shadows.clear();
    rcuExit(ticket);
  }
};
thread_local SnapScope* tSnapScope = nullptr;
const bool gMvcc = envSetting("NO_MVCC") == nullptr;
struct SnapGuard {  // a top-level SELECT statement: open a scope (nested queries share it)
  std::unique_ptr<SnapScope> own;
  SnapGuard() {
    if (gMvcc && !tSnapScope) {
      own = std::make_unique<SnapScope>();
      tSnapScope = own.get();
    }
  }
  ~SnapGuard() {
    if (own) tSnapScope = nullptr;
  }
};

// Equality between a key column (PRIMARY KEY, or the first column of a secondary index) and a literal.
static bool keyLookup(const Table& t, const ExprP& w) {
  if (!w) return false;
  if (w->kind == Expr::BIN && w->op == "AND") return keyLookup(t, w->args[0]) || keyLookup(t, w->args[1]);
  ExprP col;
  if (w->kind == Expr::INLIST && !w->neg && w->args.size() <= 33 && w->args[0]->kind == Expr::COL) {  // k IN (1, 2, ..)
    for (size_t i = 1; i < w->args.size(); ++i)
      if (w->args[i]->kind != Expr::LIT) return false;
    col = w->args[0];
  } else if (w->kind == Expr::BIN && w->op == "=") {
    const ExprP& a = w->args[0];
    const ExprP& b = w->args[1];
    col = a->kind == Expr::COL && b->kind == Expr::LIT ? a : b->kind == Expr::COL && a->kind == Expr::LIT ? b : nullptr;
  }
  if (!col) return false;
  const int c = t.colIndex(col->name);
  if (c < 0) return false;
  if (c == t.pk) return true;
  for (auto& ix : t.indexes)
    if (!ix.vec && !ix.cols.empty() && ix.cols[0] == c) return true;
  return false;
}

thread_local std::string tLoadLogPath;  // doLoad loaded a copy of the file: the WAL names the copy
static uint32_t copyFile(const std::string& from, const std::string& to);
static uint32_t fileCrc(const std::string& path);
static bool readFile(const std::string& path, std::string& out);
// The statement as logged: LOAD DATA INFILE of a copy (doLoad) names the copy's path.
static std::string loggedSql(const std::string& sql, const Stmt& s) {
  std::string logged = sql;
  if (!tLoadLogPath.empty()) {
    auto quote = [](const std::string& x) {
      std::string q = "'";
      for (char ch : x) q += ch == '\'' ? std::string("''") : std::string(1, ch);
      return q + "'";
    };
    const size_t at = logged.find(quote(s.path));
    if (at != std::string::npos) logged.replace(at, quote(s.path).size(), quote(tLoadLogPath));
    tLoadLogPath.clear();
  }
  return logged;
}
static void mkdirs(const std::string& d);

void Engine::logWrite(const std::string& sql, const std::string& db) {
  if (replaying_) return;
  const bool sync = needsSync(db);
  if (tTxn.active) {  // held until COMMIT
    tTxn.wal.push_back(currentDb() + "\x1f" + clockPrefix() + sql);
    tTxn.needSync = tTxn.needSync || sync;
    return;
  }
  uint64_t myLsn;
  {
    std::lock_guard<std::mutex> g(walMu_);
    const std::string rec = currentDb() + "\x1f" + clockPrefix() + sql;
    try {
      walAppendLocked(rec);
    } catch (SqlError& e) {  // the change is applied but not logged: refuse writes until a restart
      walBroken_ = true;
      throw SqlError(std::string(e.what()) + "; the server is read-only until it is restarted", e.code);
    }
    replAppend(rec);
    myLsn = lsn_;
  }
  walDurable(sync);
  replWaitAckAfterGate(myLsn);  // semi-sync replication
}

namespace {

ResultSet table(std::vector<std::string> names, std::vector<LType> types) {
  ResultSet r;
  r.isResult = true;
  r.names = std::move(names);
  r.types = std::move(types);
  return r;
}

std::string fmtMs(double ms) {
  char b[32];
  snprintf(b, sizeof b, "%.3f", ms);
  return b;
}

// Runs a SELECT with plan-cache bookkeeping (and stage timings when profiling).
ResultSet runSelect(Engine* e, const std::function<ResultSet()>& fn, const std::string& sql, bool profile) {
  jitBeginQuery(sql);
  tProfOn = profile;
  tProf.clear();
  auto t0 = Clock::now();
  ResultSet r;
  try {
    r = fn();
  } catch (...) {
    tProfOn = false;
    jitEndQuery(msSince(t0));
    throw;
  }
  if (profile) gpuSync();
  const double ms = msSince(t0);
  JitQueryInfo ji = jitEndQuery(ms);
  if (profile) {
    std::string plans;
    for (int id : ji.ids) plans += (plans.empty() ? "#" : ", #") + std::to_string(id);
    tProf.insert(tProf.begin(), {"codegen summary", ji.compileMs, -1,
                                 std::to_string(ji.hits) + " plan-cache hits, " + std::to_string(ji.fromDisk) +
                                     " loaded from disk, " + std::to_string(ji.compiled) + " compiled now, " +
                                     std::to_string(ji.background) + " compiling in background" +
                                     (ji.background ? " (those parts ran interpreted)" : "") +
                                     (plans.empty() ? "" : "; plans used: " + plans)});
    tProf.push_back({"total", ms, (int64_t)r.rows.size(), r.info});
    tLastProf = tProf;
    tProfOn = false;
  }
  (void)e;
  return r;
}

}  // namespace
Stmt cloneStmt(const Stmt& s);

thread_local int tExecDepth = 0;
// Concurrent queries share the GPU's memory: their temporary buffers (hash tables, gathered columns)
// can add up to more than there is. A query that runs out takes this gate exclusively, once the ones
// in flight are done, and runs again alone; queries take it shared. (Writes don't: they aren't
// retried, and a long LOAD would hold up every query behind a waiting retry.)
static FairSharedMutex gGpuGate;
// And at most this many run at once (WARPSTORE_GPU_QUERIES, default 4): more only queue on the GPU
// while their temporaries add up, and running out means retries.
static std::counting_semaphore<1024>& gpuSlots() {
  static std::counting_semaphore<1024> s(std::clamp(envSetting("GPU_QUERIES") ? atoi(envSetting("GPU_QUERIES")) : 4, 1, 1024));
  return s;
}

ResultSet Engine::execute(const std::string& sql) {
  if (tExecDepth == 0) beginStatementClock();
  const std::string fw = tExecDepth == 0 ? firstWord(sql) : std::string();
  const bool query = fw == "SELECT" || fw == "WITH" || fw == "EXPLAIN" || fw == "PROFILE" || fw == "(";
  try {
    if (!query) return executeStmt(sql);
    gpuSlots().acquire();
    struct Slot {
      ~Slot() { gpuSlots().release(); }
    } slot;
    try {
      std::shared_lock<FairSharedMutex> gate(gGpuGate);
      return executeStmt(sql);
    } catch (SqlError& e) {
      if (e.code != 1041 || sql.find("INTO OUTFILE") != std::string::npos) throw;
    }
    // Out of GPU memory: once every other query is done, alone (this attempt's buffers are freed).
    std::unique_lock<FairSharedMutex> alone(gGpuGate);
    gpuTrimPool();
    // Still out of memory alone: table columns cached in VRAM make room (read over PCIe from host
    // memory meanwhile; the VRAM cache brings the hot ones back later), more each time.
    struct Release {  // once the query is done the VRAM cache may refill (no need to wait for the hold)
      std::atomic<int64_t>& hold;
      bool shed = false;
      ~Release() { if (shed) hold = 0; }
    } release{vramHoldUntilUs_};
    for (int round = 0;; ++round) {
      try {
        return executeStmt(sql);
      } catch (SqlError& e) {
        if (e.code != 1041 || round >= 3 || gpuVramUsed() < (256ll << 20)) {
          if (e.code == 1041) goto oom;
          throw;
        }
      }
      vramHoldUntilUs_ = nowUs() + 5000000;  // (no promotions meanwhile)
      release.shed = true;
      vramBalance((768ll << 20) << round);  // as little as needed: 768 MB, then 1.5 GB, 3 GB, 6 GB
      gpuTrimPool();
    }
  oom:
    try {
      throw SqlError("GPU out of memory", 1041);
    } catch (SqlError& e) {
      if (e.code == 1041) {
        size_t mapDev = 0;
        {
          std::lock_guard<std::mutex> g(gDerivMu);
          for (auto& [k, v] : gMapDev)
            if (v.second) mapDev += v.second->n * 8;
          for (auto& [k, v] : gDeriv)
            if (v.bitsDev) mapDev += v.bitsDev->n * 4;
        }
        fprintf(stderr, "[warpstore] out of GPU memory even alone (%.0f MB free; columns %.0f MB, derived maps %.0f MB): %.200s\n",
                gpuVramFree() / 1048576.0, gpuVramUsed() / 1048576.0, mapDev / 1048576.0, sql.c_str());
      }
      throw;
    }
  } catch (SqlError& e) {
    if (e.code == 1213 && tExecDepth == 0 && tTxn.active) rollbackTxn();  // deadlock victim: the whole transaction (MySQL)
    throw;
  }
}

void Engine::replWaitAckAfterGate(uint64_t lsn) {
  if (tWriteGateHeld) tAckLsn = std::max(tAckLsn, lsn);
  else replWaitAck(lsn);
}
ResultSet Engine::executeStmt(const std::string& sqlIn) {
  const std::string sql = takeClockPrefix(sqlIn);  // (a logged statement: its clock)
  if (tExecDepth == 0) tTxnViews.clear();  // visibility bitmaps of the previous statement
  struct AckAfterGate {  // (destroyed after writeGate below: waits for replicas without holding it)
    Engine* e;
    bool top;
    ~AckAfterGate() {
      if (!top || !tAckLsn) return;
      const uint64_t l = tAckLsn;
      tAckLsn = 0;
      e->replWaitAck(l);
    }
  } ackAfter{this, tExecDepth == 0};
  std::shared_lock<FairSharedMutex> writeGate;
  if (tExecDepth == 0 && !replaying_) {
    // (not: reads, and statements made durable by a snapshot of their own instead of the WAL)
    static const std::set<std::string> reads = {"SELECT", "SHOW", "EXPLAIN", "DESCRIBE", "DESC", "WITH", "SET", "USE", "(",
                                                "SNAPSHOT", "BACKUP", "RESTORE", "OPTIMIZE", "PROFILE", "KILL", "PROMOTE", "HELP"};
    if (!reads.count(firstWord(sql))) writeGate = std::shared_lock<FairSharedMutex>(writeGate_);
  }
  struct GateHeld {
    bool prev;
    explicit GateHeld(bool now) : prev(tWriteGateHeld) { tWriteGateHeld = tWriteGateHeld || now; }
    ~GateHeld() { tWriteGateHeld = prev; }
  } gateHeld{writeGate.owns_lock()};
  struct LoadCount {
    std::atomic<int>* c = nullptr;
    ~LoadCount() { if (c) --*c; }
  } loadCount;
  if (writeGate.owns_lock() && firstWord(sql) == "LOAD") loadCount.c = &loadsInFlight_, ++loadsInFlight_;
  ++tExecDepth;
  struct Depth {
    ~Depth() {
      if (--tExecDepth == 0) releaseStatementTemps();  // columns read from the disk tier for this statement
    }
  } depthGuard;
  {
    ResultSet rr;
    if (routineStatement(sql, rr)) return rr;
  }
  {
    static const std::regex replStatusRe(R"(^\s*SHOW\s+(REPLICATION|REPLICA|SLAVE)\s+STATUS\s*;?\s*$)", std::regex::icase);
    static const std::regex promoteRe(R"(^\s*PROMOTE(\s+\w+)?\s*;?\s*$)", std::regex::icase);
    const std::string fw = firstWord(sql);
    if (fw == "SHOW" && std::regex_match(sql, replStatusRe)) return replicationStatus();
    static const std::regex killRe(R"(^\s*KILL\s+(QUERY\s+|CONNECTION\s+)?(\d+)\s*;?\s*$)", std::regex::icase);
    std::smatch km;
    if (firstWord(sql) == "KILL" && std::regex_match(sql, km, killRe)) {
      const uint32_t id = (uint32_t)std::stoul(km[2].str());
      bool mine = false;
      for (auto& se : monitor::sessions()) mine = mine || (se->id == id && se->user == tUserName());
      if (!mine && !isSuper(tUserName())) throw SqlError("You are not owner of thread " + std::to_string(id), 1095);
      if (!monitor::kill(id, upper(km[1].str()).rfind("QUERY", 0) == 0)) throw SqlError("Unknown thread id: " + std::to_string(id), 1094);
      return okResult(0);
    }
    static const std::regex sleepRe(R"(^\s*SELECT\s+SLEEP\s*\(\s*([0-9]*\.?[0-9]+)\s*\)\s*(?:AS\s+`?(\w+)`?)?\s*;?\s*$)", std::regex::icase);
    if (firstWord(sql) == "SELECT" && std::regex_match(sql, km, sleepRe)) {  // SELECT SLEEP(n): 1 if KILL QUERY cut it short (MySQL)
      const auto until = Clock::now() + std::chrono::microseconds((int64_t)(std::atof(km[1].str().c_str()) * 1e6));
      int interrupted = 0;
      while (Clock::now() < until) {
        if (tSession && tSession->killQuery) {
          tSession->killQuery = false;
          interrupted = 1;
          break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(std::min<int64_t>(10, std::chrono::duration_cast<std::chrono::milliseconds>(until - Clock::now()).count() + 1)));
      }
      ResultSet sr = table({km[2].matched ? km[2].str() : "SLEEP(" + km[1].str() + ")"}, {LType::BIGINT});
      sr.rows.push_back({std::to_string(interrupted)});
      return sr;
    }
    static const std::regex flushStatsRe(R"(^\s*FLUSH\s+QUERY_STATS\s*;?\s*$)", std::regex::icase);
    if (firstWord(sql) == "FLUSH" && std::regex_match(sql, flushStatsRe)) {
      if (!isSuper(tUserName())) throw SqlError("Access denied; you need the SUPER privilege for this operation", 1227);
      monitor::resetQueryStats();
      return okResult(0);
    }
    if (fw == "PROMOTE" && std::regex_match(sql, promoteRe)) {
      if (!isSuper(tUserName())) throw SqlError("Access denied; you need the SUPER privilege for this operation", 1227);
      promote();
      return okResult(0, "promoted to primary");
    }
  }
  if (!leaves_.empty()) {  // cluster administration (not SQL the parser knows)
    static const std::regex leavesRe(R"(^\s*SHOW\s+(LEAVES|CLUSTER\s+STATUS)\s*;?\s*$)", std::regex::icase);
    if (std::regex_match(sql, leavesRe)) return clusterStatus();
    static const std::regex adminRe(R"(^\s*((ALTER\s+CLUSTER\s+)?ADD\s+LEAF|REBALANCE(\s+ALL)?\s+PARTITIONS)\b[\s\S]*$)", std::regex::icase);
    if (std::regex_match(sql, adminRe)) {
      if (!isSuper(tUserName())) throw SqlError("Access denied; you need the SUPER privilege for this operation", 1227);
      return clusterAdmin(sql);
    }
  }
  Stmt s = parseSql(sql);
  if (!replaying_ && !tApplying) checkPrivileges(s);
  if (!leaves_.empty()) {  // cluster aggregator: sharded tables live on the leaves
    ResultSet cr;
    if (clusterStatement(sql, s, cr)) return cr;
  }
  if ((isReplica_ && !tApplying) || (walBroken_ && !replaying_)) {
    switch (s.type) {
      case Stmt::SELECT: if (s.outfile.empty()) break; [[fallthrough]];
      case Stmt::INSERT: case Stmt::UPDATE: case Stmt::DELETE: case Stmt::LOAD: case Stmt::CREATE: case Stmt::DROP:
      case Stmt::ALTER: case Stmt::TRUNCATE: case Stmt::RENAME: case Stmt::CREATE_DB: case Stmt::DROP_DB:
      case Stmt::CREATE_VIEW: case Stmt::DROP_VIEW: case Stmt::PIPELINE: case Stmt::USER: case Stmt::BACKUP:
        if (s.type == Stmt::CREATE && s.temporary) break;
        if (isReplica_) throw SqlError("The server is a read-only replica (--replica-of); run writes on the primary or PROMOTE it", 1290);
        throw SqlError("The server is read-only: a WAL write failed (disk full?); free space and restart it", 1290);
      default: break;
    }
  }
  if (s.type == Stmt::SET && (s.setName == "TRANSACTION" || s.setName == "TRANSACTION_ISOLATION" || s.setName == "TX_ISOLATION")) {
    static const std::regex snapRe(R"(REPEATABLE[\s_-]+READ|SERIALIZABLE|SNAPSHOT)", std::regex::icase);
    static const std::regex rcRe(R"(READ[\s_-]+(COMMITTED|UNCOMMITTED))", std::regex::icase);
    if (std::regex_search(sql, snapRe)) tIsoSnapshot = true;
    else if (std::regex_search(sql, rcRe)) tIsoSnapshot = false;
    return ResultSet{};
  }
  if (s.type == Stmt::TXN && s.setName == "BEGIN") {
    static const std::regex consistentRe(R"(WITH\s+CONSISTENT\s+SNAPSHOT)", std::regex::icase);
    if (std::regex_search(sql, consistentRe)) return doTxn("BEGIN SNAPSHOT");
  }
  // Autocommit writes while snapshots run: an implicit transaction, so its changes are retired with a
  // commit sequence like any other (the snapshots keep not seeing them).
  thread_local bool tImplicitTxn = false;
  if (!tTxn.active && !tImplicitTxn && !replaying_ && tExecDepth == 1 &&
      (s.type == Stmt::INSERT || s.type == Stmt::UPDATE || s.type == Stmt::DELETE || s.type == Stmt::LOAD) && snapshotsActive()) {
    tImplicitTxn = true;
    struct Flag { ~Flag() { tImplicitTxn = false; } } flag;
    beginTxnState();
    try {
      --tExecDepth;  // (the nested execute counts again)
      struct Inc { ~Inc() { ++tExecDepth; } } inc;
      ResultSet rr = execute(sql);
      commitTxn();
      return rr;
    } catch (...) {
      rollbackTxn();
      throw;
    }
  }
  ResultSet r;
  switch (s.type) {  // DDL commits an open transaction first (as in MySQL)
    case Stmt::CREATE: case Stmt::DROP: case Stmt::ALTER: case Stmt::TRUNCATE: case Stmt::RENAME:
    case Stmt::CREATE_DB: case Stmt::DROP_DB: case Stmt::CREATE_VIEW: case Stmt::DROP_VIEW: case Stmt::PIPELINE:
      if (tTxn.active) {
        commitTxn();
        if (!tTxn.autocommit) doTxn("");
      }
      break;
    default: break;
  }
  switch (s.type) {
    case Stmt::TXN: return doTxn(s.setName);
    case Stmt::SELECT: {
      SnapGuard snap;  // MVCC: the statement reads snapshots of its tables
      snapshotStatement(s);
      if (s.outfile.empty()) {
        const bool usesFound = s.items.size() == 1 && s.items[0].e->kind == Expr::FUNC && upper(s.items[0].e->op) == "FOUND_ROWS";
        std::optional<int64_t> total;
        if (s.calcFoundRows && (s.limit >= 0 || s.offset > 0)) {  // the rows without LIMIT: COUNT(*) over the query
          Stmt c;
          c.type = Stmt::SELECT;
          auto cnt = std::make_shared<Expr>();
          cnt->kind = Expr::FUNC;
          cnt->op = "COUNT";
          auto star = std::make_shared<Expr>();
          star->kind = Expr::STAR;
          cnt->args = {star};
          c.items = {SelectItem{cnt, ""}};
          TableRef ref;
          ref.alias = "__found";
          ref.sub = std::make_shared<Stmt>(cloneStmt(s));
          ref.sub->limit = -1;
          ref.sub->offset = 0;
          ref.sub->orderBy.clear();
          ref.sub->calcFoundRows = false;
          c.from = {ref};
          QueryCtx qc;
          ResultSet n = query(c, qc, false, true);
          total = n.vals.empty() ? 0 : (int64_t)n.vals[0][0].num();
        }
        ResultSet r = runSelect(this, [&] { return runStatementQuery(s, false); }, sql, false);
        if (!usesFound) tFoundRows = total ? *total : (int64_t)std::max(r.rows.size(), r.vals.size());
        return r;
      }
      ResultSet q = runStatementQuery(s, false);  // SELECT .. INTO OUTFILE: tab-separated, \N for NULL
      FILE* f = fopen(s.outfile.c_str(), "wx");
      if (!f) throw SqlError("Can't create/write to file '" + s.outfile + "' (it may already exist)", 1086);
      for (auto& row : q.rows) {
        for (size_t i = 0; i < row.size(); ++i) {
          if (i) fputc(s.outDelim, f);
          if (row[i] == kNullCell) fputs("\\N", f);
          else fwrite(row[i].data(), 1, row[i].size(), f);
        }
        fputc('\n', f);
      }
      fclose(f);
      return okResult((int64_t)q.rows.size());
    }
    case Stmt::EXPLAIN: {
      SnapGuard snap;
      if (s.inner) snapshotStatement(*s.inner);
      return runSelect(this, [&] { return runStatementQuery(*s.inner, true); }, sql, false);
    }
    case Stmt::PROFILE: {
      SnapGuard snap;
      if (s.inner) snapshotStatement(*s.inner);
      return runSelect(this, [&] { return runStatementQuery(*s.inner, false); }, sql, true);
    }
    case Stmt::SHOW_PROFILE: {
      r = table({"stage", "time_ms", "rows", "detail"}, {LType::VARCHAR, LType::DOUBLE, LType::BIGINT, LType::VARCHAR});
      for (auto& p : tLastProf)
        r.rows.push_back({p.stage, fmtMs(p.ms), p.rows < 0 ? kNullCell : std::to_string(p.rows), p.detail});
      return r;
    }
    case Stmt::SHOW_PLANCACHE: {
      r = table({"plan_id", "kind", "origin", "compile_ms", "executions", "avg_query_ms", "last_query_ms", "source_bytes",
                 "query_text"},
                {LType::BIGINT, LType::VARCHAR, LType::VARCHAR, LType::DOUBLE, LType::BIGINT, LType::DOUBLE, LType::DOUBLE,
                 LType::BIGINT, LType::VARCHAR});
      for (auto& p : jitPlanCache())
        r.rows.push_back({std::to_string(p.id), p.kind, p.origin, fmtMs(p.compileMs), std::to_string(p.execs),
                          fmtMs(p.avgMs), fmtMs(p.lastMs), std::to_string(p.sourceBytes), p.sql});
      return r;
    }
    case Stmt::SHOW_PLAN: {
      std::string src = jitSourceOf((int)s.planId);
      if (src.empty()) throw SqlError("Unknown plan id " + std::to_string(s.planId), 1105);
      r = table({"generated_source"}, {LType::VARCHAR});
      size_t a = 0;
      for (size_t b; (b = src.find('\n', a)) != std::string::npos; a = b + 1) r.rows.push_back({src.substr(a, b - a)});
      return r;
    }
    case Stmt::DROP_PLANCACHE: jitClearCache(); return okResult(0);
    case Stmt::SET: {
      if (s.setName.rfind("GPUDB_", 0) == 0) s.setName = "WARPSTORE_" + s.setName.substr(6);  // (pre-rename names)
      auto on = [&] { return s.setValue == "ON" || s.setValue == "1" || s.setValue == "TRUE"; };
      if (s.setName == "WARPSTORE_CODEGEN") jitSetMode(s.setValue == "SYNC" ? JIT_SYNC : on() ? JIT_ON : JIT_OFF);
      else if (s.setName == "WARPSTORE_DENSE_GROUPBY") tDense = on();
      else if (s.setName == "WARPSTORE_JOIN_FILTERS") tJoinFilters = on();
      else if (s.setName == "WARPSTORE_VECTOR_INDEX") tAnn = on();
      else if (s.setName == "WARPSTORE_NPROBE") tNprobe = std::atoi(s.setValue.c_str());
      else if (s.setName == "WARPSTORE_CAGRA_ITOPK") tCagraItopk = std::atoi(s.setValue.c_str());
      else if (s.setName == "WARPSTORE_DEBUG_SCAN_DELAY_MS") tScanDelayMs = std::atoi(s.setValue.c_str());
      else if (s.setName == "WARPSTORE_JOIN_FILTER_ROWS") tLazyRows = std::atoll(s.setValue.c_str());
      else if (s.setName == "WARPSTORE_SYNC_DURABILITY") tSyncDurability = s.setValue == "DEFAULT" ? -1 : on() ? 1 : 0;
      else if (s.setName == "LONG_QUERY_TIME" || s.setName == "SLOW_QUERY_LOG") {  // server-wide (SET GLOBAL)
        if (!isSuper(tUserName())) throw SqlError("Access denied; you need the SUPER privilege for this operation", 1227);
        static double lastMs = 1000;
        if (s.setName == "LONG_QUERY_TIME") {
          lastMs = std::atof(s.setValue.c_str()) * 1000;
          if (monitor::slowMs.load() >= 0) monitor::slowMs = lastMs;
        } else {
          monitor::slowMs = on() ? lastMs : -1;
        }
      }
      else if (s.setName == "COLLATION_CONNECTION" || s.setName == "COLLATION_SERVER" || s.setName == "WARPSTORE_COLLATION")
        tCi = s.setValue.find("_BIN") != std::string::npos || s.setValue == "BINARY" ? 0 : 1;
      else if (s.setName == "NAMES" && s.setValue.find("BIN") != std::string::npos) tCi = 0;
      else if (s.setName == "AUTOCOMMIT") {
        tTxn.autocommit = on();
        if (tTxn.autocommit) commitTxn();  // SET autocommit = 1 commits
        else doTxn("");
      }
      return okResult(0);
    }
    case Stmt::INSERT: r = doInsert(s); break;
    case Stmt::CREATE:
      r = doCreate(s);
      if (s.temporary) tTempTables.push_back(dbOf(s.table) + "." + plainName(s.table));
      break;
    case Stmt::DROP: r = doDrop(s); break;
    case Stmt::DELETE:
    case Stmt::UPDATE: r = doModify(s); break;
    case Stmt::LOAD:
      if ((s.setValue != "FS" && !s.setValue.empty()) || s.path.rfind("http://", 0) == 0 || s.path.rfind("https://", 0) == 0 ||
          s.format == "JSON" || (s.format == "PARQUET" && s.path.find_first_of("*?[") != std::string::npos) || s.format == "AVRO" || s.format == "ORC" ||
          (s.path.size() > 4 && lower(s.path.substr(s.path.size() - 4)) == ".orc") ||
          ((s.path.size() > 8 && lower(s.path.substr(s.path.size() - 8)) == ".parquet") && s.path.find_first_of("*?[") != std::string::npos) ||
          (s.path.size() > 5 && lower(s.path.substr(s.path.size() - 5)) == ".avro"))
        return loadRemote(s);  // logs the staged LOAD DATA INFILE statements it runs, not this one
      try {
        r = doLoad(s);
      } catch (SqlError&) {
        if (!tLoadLogPath.empty()) {  // a load that failed after loading rows (doLoad): log it all the same
          logWrite(loggedSql(sql, s), s.table.find('.') != std::string::npos ? s.table.substr(0, s.table.find('.')) : std::string());
        }
        throw;
      }
      break;
    case Stmt::OPTIMIZE: r = doOptimize(s); break;
    case Stmt::SHOW_TABLES: return doShowTables(s);
    case Stmt::SHOW_MISC:
      if (s.setName == "PIPELINES") return showPipelines();
      return doShowMisc(s);
    case Stmt::PIPELINE: r = doPipeline(s); break;
    case Stmt::SNAPSHOT: return doSnapshot();
    case Stmt::USER: r = doUser(s); break;
    case Stmt::BACKUP: return doBackup(s);
    case Stmt::ALTER: r = doAlter(s); break;
    case Stmt::TRUNCATE: r = doTruncate(s); break;
    case Stmt::RENAME: r = doRename(s); break;
    case Stmt::USE: {
      std::lock_guard<std::mutex> g(cat_.mu);
      if (!cat_.dbs.count(lower(s.table)) && lower(s.table) != "information_schema")
        throw SqlError("Unknown database '" + s.table + "'", 1049);
      currentDb() = lower(s.table);
      return okResult(0);
    }
    case Stmt::CREATE_DB: {
      std::lock_guard<std::mutex> g(cat_.mu);
      if (cat_.dbs.count(lower(s.table))) {
        if (!s.ifNotExists) throw SqlError("Can't create database '" + s.table + "'; database exists", 1007);
      } else {
        cat_.dbs.insert(lower(s.table));
        if (s.setValue == "SYNC") cat_.syncDbs.insert(lower(s.table));
      }
      r = okResult(1);
      break;
    }
    case Stmt::DROP_DB: {
      std::lock_guard<std::mutex> g(cat_.mu);
      const std::string db = lower(s.table);
      if (!cat_.dbs.count(db)) {
        if (!s.ifExists) throw SqlError("Can't drop database '" + s.table + "'; database doesn't exist", 1008);
      } else {
        for (auto it = cat_.tables.begin(); it != cat_.tables.end();)
          it = it->first.rfind(db + ".", 0) == 0 ? cat_.tables.erase(it) : std::next(it);
        for (auto it = cat_.views.begin(); it != cat_.views.end();)
          it = it->first.rfind(db + ".", 0) == 0 ? cat_.views.erase(it) : std::next(it);
        cat_.dbs.erase(db);
        cat_.syncDbs.erase(db);
        if (currentDb() == db) currentDb() = "warpstore";
      }
      r = okResult(0);
      break;
    }
    case Stmt::CREATE_VIEW: {
      std::lock_guard<std::mutex> g(cat_.mu);
      const std::string k = Catalog::key(s.table);
      if (cat_.tables.count(k)) throw SqlError("Table '" + s.table + "' already exists", 1050);
      std::string sql2 = s.viewSql;
      if (!s.insertCols.empty()) {  // CREATE VIEW v (a, b) AS ..: column names applied when the view is used
        std::string names;
        for (auto& c : s.insertCols) names += (names.empty() ? "" : ",") + c;
        sql2 = "/*cols:" + names + "*/ " + sql2;
      }
      cat_.views[k] = sql2;
      r = okResult(0);
      break;
    }
    case Stmt::DROP_VIEW: {
      std::lock_guard<std::mutex> g(cat_.mu);
      for (auto& n : s.names)
        if (!cat_.views.erase(Catalog::key(n)) && !s.ifExists) throw SqlError("Unknown view '" + n + "'", 1051);
      r = okResult(0);
      break;
    }
    case Stmt::SHOW_DATABASES: {
      r.isResult = true;
      r.names = {"Database"};
      r.types = {LType::VARCHAR};
      std::set<std::string> dbs;
      {
        std::lock_guard<std::mutex> g(cat_.mu);
        dbs = cat_.dbs;
      }
      dbs.insert("information_schema");
      for (auto& d : dbs)
        if (s.path.empty() || likeMatch(d, lower(s.path))) r.rows.push_back({d});
      return r;
    }
    case Stmt::SHOW_SEGMENTS: return doShowSegments(s);
    case Stmt::DESCRIBE: return doDescribe(s);
    default: return okResult(0);
  }
  if (s.type != Stmt::OPTIMIZE)
    logWrite(loggedSql(sql, s), s.table.find('.') != std::string::npos ? s.table.substr(0, s.table.find('.')) : std::string());
  return r;
}

// IndexMeta for a parsed index definition (VECTOR INDEX: IVF options).
static IndexMeta indexMetaOf(const IndexDef& ix, const Table& t, const std::vector<int>& cols) {
  IndexMeta m;
  m.name = ix.name.empty() ? ix.cols.at(0) : ix.name;
  m.unique = ix.unique;
  m.cols = cols;
  if (ix.vector) {
    if (cols.size() != 1 || t.cols[cols[0]].pt != PType::VEC)
      throw SqlError("VECTOR INDEX needs exactly one VECTOR column", 1235);
    m.vec = true;
    m.options = ix.options;
    m.dim = t.cols[cols[0]].vecDim;
    std::string o = ix.options;
    for (auto& ch : o) ch = (char)toupper((unsigned char)ch);
    m.metric = o.find("EUCLIDEAN") != std::string::npos ? 1 : 0;
    size_t p = ix.options.find("\"nprobe\"");
    m.nprobe = p == std::string::npos ? 0 : std::atoi(ix.options.c_str() + ix.options.find(':', p) + 1);
    auto num = [&](const char* key) {
      size_t q = ix.options.find(key);
      return q == std::string::npos ? 0 : std::atoi(ix.options.c_str() + ix.options.find(':', q) + 1);
    };
    if (int g = num("\"graph_degree\"")) m.graphDegree = g;
    m.itopk = num("\"itopk\"");
    // index_type: CAGRA (and SingleStore's HNSW_* / AUTO, or no type) use the GPU graph when cuVS is
    // installed and the dimension suits it; IVF_* and everything else the IVF lists.
    const bool ivf = o.find("IVF") != std::string::npos;
    const bool graph = o.find("CAGRA") != std::string::npos || o.find("HNSW") != std::string::npos || o.find("AUTO") != std::string::npos ||
                       o.find("INDEX_TYPE") == std::string::npos;
    if (!ivf && graph && m.dim % 4 == 0 && cagraAvailable()) {
      m.algo = 1;
      m.cagra = std::make_shared<CagraState>();
    }
  }
  return m;
}

ResultSet Engine::doCreate(Stmt& s) {
  if (s.select) {  // CREATE TABLE t AS SELECT ..: schema from the result, then insert it
    {
      std::lock_guard<std::mutex> g(cat_.mu);
      if (cat_.tables.count(Catalog::key(s.table))) {
        if (s.ifNotExists) return okResult(0);
        throw SqlError("Table '" + s.table + "' already exists", 1050);
      }
    }
    QueryCtx q;
    ResultSet r;
    try {
      r = query(*s.select, q, false, true);
    } catch (...) {
      dropTemps(q);
      throw;
    }
    dropTemps(q);
    Stmt c;
    c.type = Stmt::CREATE;
    c.table = s.table;
    c.pkCols = s.pkCols;
    c.shardKey = s.shardKey;
    c.hasShardKey = s.hasShardKey;
    c.sortKey = s.sortKey;
    std::set<std::string> seen;
    for (size_t i = 0; i < r.names.size(); ++i) {
      ColumnDef d;
      d.name = r.names[i];
      for (int k = 2; seen.count(lower(d.name)); ++k) d.name = r.names[i] + "_" + std::to_string(k);
      seen.insert(lower(d.name));
      d.type = r.types[i];
      d.typeText = ltypeName(d.type);
      c.cols.push_back(d);
    }
    doCreate(c);
    auto t = cat_.get(s.table);
    std::unique_lock<FairSharedMutex> lk(t->mu);
    for (auto& row : r.vals) t->appendRow(row);
    return okResult((int64_t)r.vals.size());
  }
  if (!s.path.empty()) {  // CREATE TABLE t LIKE other
    auto o = cat_.get(s.path);
    std::lock_guard<std::mutex> g(cat_.mu);
    if (cat_.tables.count(Catalog::key(s.table))) {
      if (s.ifNotExists) return okResult(0);
      throw SqlError("Table '" + s.table + "' already exists", 1050);
    }
    auto nt = std::make_shared<Table>(plainName(s.table), o->cols, o->pk, o->sortKey, o->pkCols, o->shardKey, o->nparts,
                                      o->reference);
    nt->db = dbOf(s.table);
    nt->vram = o->vram;
    nt->fks = o->fks;
    nt->fulltextCols = o->fulltextCols;
    cat_.tables[Catalog::key(s.table)] = nt;
    return okResult(0);
  }
  std::lock_guard<std::mutex> g(cat_.mu);
  std::string key = Catalog::key(s.table);
  if (!cat_.dbs.count(dbOf(s.table))) throw SqlError("Unknown database '" + dbOf(s.table) + "'", 1049);
  if (cat_.tables.count(key)) {
    if (s.ifNotExists) return okResult(0);
    throw SqlError("Table '" + s.table + "' already exists", 1050);
  }
  std::vector<ColumnMeta> cols;
  for (auto& c : s.cols) {
    ColumnMeta m;
    m.name = c.name;
    m.lt = c.type;
    m.pt = physOf(c.type);
    m.notNull = c.notNull;
    m.autoInc = c.autoInc;
    m.typeText = c.typeText;
    m.vecDim = c.vecDim;
    m.scale = c.scale;
    m.vram = c.vram;
    m.defText = c.defText;
    if (c.type == LType::VECTOR && c.vecDim <= 0) throw SqlError("VECTOR columns need a dimension: VECTOR(n)", 1235);
    if (c.def) {
      m.hasDefault = true;
      if (c.def->kind == Expr::LIT && c.def->lit.kind == Value::STR && c.def->lit.s == "CURRENT_TIMESTAMP") {
        m.def = c.def->lit;  // evaluated at insert time
      } else {
        BindCtx bc;
        PostCtx pc;
        ExprP d = c.def;
        bind(d, bc);
        m.def = evalPost(d, pc);
      }
    }
    cols.push_back(m);
  }
  if (cols.size() > 4096) throw SqlError("too many columns", 1117);  // (a query may reference up to kMaxCols)
  auto find = [&](const std::string& n) {
    if (n.empty()) return -1;
    for (size_t i = 0; i < cols.size(); ++i)
      if (strcasecmp(cols[i].name.c_str(), n.c_str()) == 0) return (int)i;
    throw SqlError("Key column '" + n + "' doesn't exist in table", 1072);
  };
  int sk = find(s.sortKey);
  std::vector<int> pkCols, shard;
  for (auto& n : s.pkCols) pkCols.push_back(find(n));
  for (auto& n : s.shardKey) shard.push_back(find(n));
  // Single-column integer-like primary keys get the hash index (point lookups, uniqueness).
  int pk = pkCols.size() == 1 ? pkCols[0] : -1;
  if (pk >= 0 && cols[pk].lt == LType::DOUBLE) throw SqlError("PRIMARY KEY on DOUBLE is not supported", 1235);
  if (pk >= 0 && cols[pk].lt == LType::VARCHAR) pk = -1;
  // SingleStore semantics: the shard key defaults to the primary key, and a primary key must
  // contain every shard-key column (so uniqueness can be checked inside one partition).
  if (s.reference) {
    if (s.hasShardKey && !shard.empty()) throw SqlError("Reference tables cannot have a SHARD KEY", 1235);
    shard.clear();
  } else if (!s.hasShardKey) {
    shard = pkCols;
  } else if (!pkCols.empty()) {
    for (int c : shard)
      if (std::find(pkCols.begin(), pkCols.end(), c) == pkCols.end())
        throw SqlError("The unique key named: 'PRIMARY' must contain all columns specified in the shard key", 1705);
  }
  auto nt = std::make_shared<Table>(plainName(s.table), cols, pk, sk, pkCols, shard, nparts_, s.reference);
  nt->db = dbOf(s.table);
  nt->vram = s.vram;
  for (auto& c : s.fulltextCols) nt->fulltextCols.push_back(find(c));
  for (auto& f : s.fks) {
    for (auto& c : f.cols) find(c);
    Table::FkMeta m{f.name, f.refTable, f.onDelete, f.onUpdate, f.cols, f.refCols};
    if (m.name.empty()) m.name = plainName(s.table) + "_ibfk_" + std::to_string(nt->fks.size() + 1);
    nt->fks.push_back(m);
  }
  if (pk < 0 && !pkCols.empty()) {  // composite or string PRIMARY KEY: enforced by a unique index
    IndexMeta m;
    m.name = "PRIMARY";
    m.unique = true;
    m.cols = pkCols;
    nt->indexes.push_back(m);
    for (int c : pkCols) nt->cols[c].notNull = true;
  }
  for (auto& ix : s.indexes) {
    std::vector<int> ic;
    for (auto& c : ix.cols) ic.push_back(find(c));
    nt->indexes.push_back(indexMetaOf(ix, *nt, ic));
  }
  cat_.tables[key] = nt;
  return okResult(0);
}

ResultSet Engine::doDrop(Stmt& s) {
  for (auto& name : s.names.empty() ? std::vector<std::string>{s.table} : s.names) {
    std::shared_ptr<Table> t;
    {
      std::lock_guard<std::mutex> g(cat_.mu);
      auto it = cat_.tables.find(Catalog::key(name));
      if (it == cat_.tables.end()) {
        if (cat_.views.erase(Catalog::key(name))) continue;  // DROP TABLE on a view
        if (s.ifExists) continue;
        throw SqlError("Unknown table '" + dbOf(name) + "." + plainName(name) + "'", 1051);
      }
      t = it->second;
      cat_.tables.erase(it);
    }
    std::unique_lock<FairSharedMutex> lk(t->mu);  // wait for in-flight readers
  }
  return okResult(0);
}

namespace {
void forEachLiveRow(const Table& t, const std::function<void(Loc)>& fn) {
  for (size_t si = 0; si < t.segs.size(); ++si) {
    const Segment& sg = *t.segs[si];
    for (int64_t r = 0; r < sg.n; ++r)
      if (!sg.del->test(r)) fn(Loc{(int32_t)si, (int32_t)r});
  }
  for (int p = 0; p < t.nparts; ++p) {
    const WriteBuffer& b = *t.bufs[p];
    for (int64_t r = 0; r < b.n; ++r)
      if (!b.del->test(r)) fn(bufLoc(p, r));
  }
}

ColumnMeta metaOf(const ColumnDef& c) {
  ColumnMeta m;
  m.name = c.name;
  m.lt = c.type;
  m.pt = physOf(c.type);
  m.notNull = c.notNull;
  m.autoInc = c.autoInc;
  m.typeText = c.typeText;
  m.vecDim = c.vecDim;
  m.scale = c.scale;
  m.vram = c.vram;
  m.defText = c.defText;
  if (c.def) {
    m.hasDefault = true;
    if (c.def->kind == Expr::LIT && c.def->lit.kind == Value::STR && c.def->lit.s == "CURRENT_TIMESTAMP") {
      m.def = c.def->lit;
    } else {
      ExprP d = cloneExpr(c.def);
      BindCtx bc;
      bind(d, bc);
      m.def = evalConst(d);
    }
  }
  return m;
}

std::string colTypeText(const ColumnMeta& c) {
  if (!c.typeText.empty()) {  // MySQL's canonical spelling of the type as written
    std::string t = lower(c.typeText);
    std::string base = t.substr(0, t.find_first_of("( ")), rest = t.substr(base.size());
    static const std::map<std::string, std::string> canon = {
        {"integer", "int"}, {"bool", "tinyint(1)"}, {"boolean", "tinyint(1)"}, {"numeric", "decimal"}, {"dec", "decimal"},
        {"fixed", "decimal"}, {"real", "double"}, {"float8", "double"}, {"float4", "float"}, {"int4", "int"}, {"int8", "bigint"},
        {"int2", "smallint"}, {"int1", "tinyint"}, {"middleint", "mediumint"}, {"character", "char"}, {"nchar", "char"},
        {"nvarchar", "varchar"}, {"serial", "bigint unsigned"}};
    auto it = canon.find(base);
    if (it != canon.end()) {
      if (base == "bool" || base == "boolean" || base == "serial") return it->second;
      t = it->second + rest;
    }
    if (t == "decimal") t = "decimal(10,0)";
    for (size_t i; (i = t.find(", ")) != std::string::npos;) t.erase(i + 1, 1);  // decimal(10, 2) -> decimal(10,2)
    if (t.rfind("double precision", 0) == 0) t = "double";
    if (t.rfind("character varying", 0) == 0) t = "varchar" + t.substr(17);
    return t;
  }
  switch (c.lt) {
    case LType::INT: return "int";
    case LType::BIGINT: return "bigint";
    case LType::DOUBLE: return "double";
    case LType::DATE: return "date";
    case LType::DATETIME: return "datetime";
    case LType::VARCHAR: return "text";
  }
  return "text";
}

bool likeCI(const std::string& s, const std::string& pat) { return pat.empty() || likeMatch(lower(s), lower(pat)); }
}  // namespace

// Rebuilds a table with a new schema (ALTER TABLE): `from[i]` is the old column feeding new column i,
// or -1 for a new column (filled with its default).
std::shared_ptr<Table> Engine::rebuild(Table& t, const std::vector<ColumnMeta>& cols, const std::vector<int>& from) {
  auto mapCol = [&](int old) {
    for (size_t i = 0; i < from.size(); ++i)
      if (from[i] == old) return (int)i;
    return -1;
  };
  std::vector<int> pkCols, shard;
  for (int c : t.pkCols)
    if (mapCol(c) >= 0) pkCols.push_back(mapCol(c));
  for (int c : t.shardKey) {
    if (mapCol(c) < 0) throw SqlError("can't drop a SHARD KEY column", 1235);
    shard.push_back(mapCol(c));
  }
  int pk = t.pk >= 0 ? mapCol(t.pk) : -1;
  int sk = t.sortKey >= 0 ? mapCol(t.sortKey) : -1;
  auto nt = std::make_shared<Table>(t.name, cols, pk, sk, pkCols, shard, t.nparts, t.reference);
  nt->db = t.db;
  nt->vram = t.vram;
  nt->fks = t.fks;
  nt->fulltextCols = t.fulltextCols;
  for (auto& ix : t.indexes) {
    IndexMeta m = ix;
    m.map.clear();
    m.cols.clear();
    for (int c : ix.cols)
      if (mapCol(c) >= 0) m.cols.push_back(mapCol(c));
    if (!m.cols.empty()) nt->indexes.push_back(m);
  }
  std::vector<Value> defaults = defaultRow(*nt);
  forEachLiveRow(t, [&](Loc l) {
    std::vector<Value> row = defaults;
    for (size_t i = 0; i < from.size(); ++i)
      if (from[i] >= 0) row[i] = cellValue(t, l, from[i]);
    for (size_t i = 0; i < cols.size(); ++i) {  // type changes: convert through text / numbers
      Value& v = row[i];
      if (v.isNull()) continue;
      if (cols[i].lt == LType::VARCHAR && v.kind != Value::STR) v = Value::S(v.toString());
      else if (cols[i].lt != LType::VARCHAR && v.kind == Value::STR) {
        if (cols[i].lt == LType::DATE || cols[i].lt == LType::DATETIME) continue;
        v = cols[i].lt == LType::DOUBLE ? Value::D(std::atof(v.s.c_str())) : Value::I(std::atoll(v.s.c_str()));
      } else if ((cols[i].lt == LType::INT || cols[i].lt == LType::BIGINT) && v.kind == Value::DBL) {
        v = Value::I((int64_t)std::llround(v.d));
      }
    }
    nt->appendRow(row);
  });
  nt->autoNext = std::max(nt->autoNext, t.autoNext);
  nt->flush(true);
  return nt;
}

ResultSet Engine::doAlter(Stmt& s) {
  auto t = cat_.get(s.table);
  std::unique_lock<FairSharedMutex> lk(t->mu);
  std::shared_ptr<Table> cur = t;
  for (auto& a : s.alters) {
    std::vector<ColumnMeta> cols = cur->cols;
    std::vector<int> from(cols.size());
    for (size_t i = 0; i < from.size(); ++i) from[i] = (int)i;
    switch (a.kind) {
      case Stmt::Alter::NOOP: continue;
      case Stmt::Alter::ADD_COL: {
        if (cur->colIndex(a.col.name) >= 0) throw SqlError("Duplicate column name '" + a.col.name + "'", 1060);
        size_t pos = cols.size();
        if (a.newName == "\x01FIRST") pos = 0;
        else if (!a.newName.empty()) {
          int k = cur->colIndex(a.newName);
          if (k < 0) throw SqlError("Unknown column '" + a.newName + "' in 'table'", 1054);
          pos = (size_t)k + 1;
        }
        cols.insert(cols.begin() + pos, metaOf(a.col));
        from.insert(from.begin() + pos, -1);
        break;
      }
      case Stmt::Alter::DROP_COL: {
        int k = cur->colIndex(a.name);
        if (k < 0) throw SqlError("Can't DROP '" + a.name + "'; check that column/key exists", 1091);
        cols.erase(cols.begin() + k);
        from.erase(from.begin() + k);
        break;
      }
      case Stmt::Alter::RENAME_COL:
      case Stmt::Alter::MODIFY_COL: {
        int k = cur->colIndex(a.name);
        if (k < 0) throw SqlError("Unknown column '" + a.name + "' in 'table'", 1054);
        if (a.kind == Stmt::Alter::RENAME_COL) cols[k].name = a.newName;
        else cols[k] = metaOf(a.col);
        break;
      }
      case Stmt::Alter::ADD_FK: {  // metadata only (not enforced)
        for (auto& c : a.fk.cols)
          if (cur->colIndex(c) < 0) throw SqlError("Key column '" + c + "' doesn't exist in table", 1072);
        Table::FkMeta m{a.fk.name, a.fk.refTable, a.fk.onDelete, a.fk.onUpdate, a.fk.cols, a.fk.refCols};
        if (m.name.empty()) m.name = cur->name + "_ibfk_" + std::to_string(cur->fks.size() + 1);
        for (auto& f : cur->fks)
          if (strcasecmp(f.name.c_str(), m.name.c_str()) == 0) throw SqlError("Duplicate foreign key constraint name '" + m.name + "'", 1826);
        cur->fks.push_back(m);
        continue;
      }
      case Stmt::Alter::DROP_FK: {  // DROP FOREIGN KEY / DROP CONSTRAINT name: a foreign key, else a unique key
        auto& f = cur->fks;
        auto it = std::find_if(f.begin(), f.end(), [&](const Table::FkMeta& m) { return strcasecmp(m.name.c_str(), a.name.c_str()) == 0; });
        if (it != f.end()) { f.erase(it); continue; }
        auto& ix = cur->indexes;
        auto jt = std::find_if(ix.begin(), ix.end(), [&](const IndexMeta& m) { return strcasecmp(m.name.c_str(), a.name.c_str()) == 0; });
        if (jt == ix.end()) throw SqlError("Can't DROP '" + a.name + "'; check that column/key exists", 1091);
        ix.erase(jt);
        continue;
      }
      case Stmt::Alter::SET_AUTOINC:  // the next AUTO_INCREMENT value (never below the values in use)
        cur->autoNext = std::max<int64_t>(cur->autoNext, a.autoInc);
        continue;
      case Stmt::Alter::SET_VRAM: {  // placement hint only: the VRAM cache applies it within a second
        if (a.name.empty()) cur->vram = a.vram;
        else {
          int ci = cur->colIndex(a.name);
          if (ci < 0) throw SqlError("Unknown column '" + a.name + "' in '" + cur->name + "'", 1054);
          cur->cols[ci].vram = a.vram;
        }
        continue;
      }
      case Stmt::Alter::RENAME_TABLE: {
        std::lock_guard<std::mutex> g(cat_.mu);
        if (cat_.tables.count(Catalog::key(a.newName))) throw SqlError("Table '" + a.newName + "' already exists", 1050);
        cat_.tables.erase(Catalog::key(cur->db + "." + cur->name));
        cur->name = plainName(a.newName);
        cur->db = dbOf(a.newName);
        cat_.tables[Catalog::key(a.newName)] = cur;
        continue;
      }
      case Stmt::Alter::ADD_INDEX: {
        std::vector<int> ic;
        for (auto& c : a.index.cols) {
          int k = cur->colIndex(c);
          if (k < 0) throw SqlError("Key column '" + c + "' doesn't exist in table", 1072);
          ic.push_back(k);
        }
        cur->indexes.push_back(indexMetaOf(a.index, *cur, ic));
        cur->buildIndex(cur->indexes.back());
        continue;
      }
      case Stmt::Alter::DROP_INDEX: {
        auto& ix = cur->indexes;
        ix.erase(std::remove_if(ix.begin(), ix.end(), [&](const IndexMeta& m) { return strcasecmp(m.name.c_str(), a.name.c_str()) == 0; }), ix.end());
        continue;
      }
    }
    auto nt = rebuild(*cur, cols, from);
    std::lock_guard<std::mutex> g(cat_.mu);
    cat_.tables[Catalog::key(cur->db + "." + cur->name)] = nt;
    cur = nt;
  }
  return okResult(0);
}

ResultSet Engine::doTruncate(Stmt& s) {
  auto t = cat_.get(s.table);
  std::unique_lock<FairSharedMutex> lk(t->mu);
  waitTxnFree(*t, lk);  // another session's open transaction changed this table
  auto nt = std::make_shared<Table>(t->name, t->cols, t->pk, t->sortKey, t->pkCols, t->shardKey, t->nparts, t->reference);
  nt->db = t->db;
  nt->vram = t->vram;
  nt->fks = t->fks;
  nt->fulltextCols = t->fulltextCols;
  nt->indexes = t->indexes;
  for (auto& ix : nt->indexes) ix.map.clear();
  std::lock_guard<std::mutex> g(cat_.mu);
  cat_.tables[Catalog::key(t->db + "." + t->name)] = nt;
  return okResult(0);
}

ResultSet Engine::doRename(Stmt& s) {
  std::lock_guard<std::mutex> g(cat_.mu);
  for (size_t i = 0; i + 1 < s.names.size(); i += 2) {
    auto it = cat_.tables.find(Catalog::key(s.names[i]));
    if (it == cat_.tables.end()) throw SqlError("Table '" + s.names[i] + "' doesn't exist", 1146);
    if (cat_.tables.count(Catalog::key(s.names[i + 1]))) throw SqlError("Table '" + s.names[i + 1] + "' already exists", 1050);
    auto t = it->second;
    cat_.tables.erase(it);
    t->name = plainName(s.names[i + 1]);
    t->db = dbOf(s.names[i + 1]);
    cat_.tables[Catalog::key(s.names[i + 1])] = t;
  }
  return okResult(0);
}

std::string Engine::showCreate(const Table& t) {
  std::string r = "CREATE " + std::string(t.reference ? "REFERENCE " : "") + "TABLE `" + t.name + "` (\n";
  for (size_t i = 0; i < t.cols.size(); ++i) {
    const ColumnMeta& c = t.cols[i];
    r += "  `" + c.name + "` " + colTypeText(c);
    if (c.notNull || (int)i == t.pk) r += " NOT NULL";
    if (c.autoInc) r += " AUTO_INCREMENT";
    if (c.hasDefault && !c.defText.empty()) r += " DEFAULT (" + c.defText + ")";
    else if (c.hasDefault) r += " DEFAULT " + (c.def.kind == Value::STR && c.def.s != "CURRENT_TIMESTAMP" ? "'" + c.def.s + "'" : c.def.isNull() ? std::string("NULL") : c.def.toString());
    if (c.vram) r += c.vram == 1 ? " VRAM=PIN" : " VRAM=NEVER";
    r += ",\n";
  }
  auto names = [&](const std::vector<int>& cs) {
    std::string x;
    for (int c : cs) x += (x.empty() ? "`" : ", `") + t.cols[c].name + "`";
    return x;
  };
  if (!t.pkCols.empty()) r += "  PRIMARY KEY (" + names(t.pkCols) + "),\n";
  for (auto& ix : t.indexes)
    if (ix.name != "PRIMARY")
      r += std::string("  ") + (ix.unique ? "UNIQUE " : "") + (ix.vec ? "VECTOR " : "") + "KEY `" + ix.name + "` (" + names(ix.cols) + ")" +
           (ix.vec && !ix.options.empty() ? " INDEX_OPTIONS '" + ix.options + "'" : "") + ",\n";
  for (auto& f : t.fks) {
    std::string lc, rc;
    for (auto& c : f.cols) lc += (lc.empty() ? "`" : ", `") + c + "`";
    for (auto& c : f.refCols) rc += (rc.empty() ? "`" : ", `") + c + "`";
    r += "  CONSTRAINT `" + f.name + "` FOREIGN KEY (" + lc + ") REFERENCES `" + f.refTable + "` (" + rc + ") ON DELETE " + f.onDelete +
         " ON UPDATE " + f.onUpdate + ",\n";
  }
  // SHARD KEY only when it isn't the default (the primary key; keyless without one), so MySQL tools
  // reading SHOW CREATE TABLE see plain MySQL DDL.
  if (!t.fulltextCols.empty()) {
    std::vector<int> ft = t.fulltextCols;
    r += "  FULLTEXT KEY (" + names(ft) + "),\n";
  }
  if (!t.reference && t.shardKey != t.pkCols) r += "  SHARD KEY (" + names(t.shardKey) + "),\n";
  if (t.sortKey >= 0) r += "  SORT KEY (`" + t.cols[t.sortKey].name + "`),\n";
  r = r.substr(0, r.size() - 2) + "\n)";
  if (t.vram) r += t.vram == 1 ? " VRAM=PIN" : " VRAM=NEVER";
  return r;
}

// information_schema tables, generated from the catalog.
// SHOW PROCESSLIST rows: every session for SUPER (PROCESS) users, else the user's own.
std::vector<std::vector<std::string>> Engine::processRows(bool truncateInfo) {
  std::vector<std::vector<std::string>> out;
  const bool all = isSuper(tUserName());
  const int64_t now = nowUs();
  for (auto& se : monitor::sessions()) {
    if (!all && se->user != tUserName()) continue;
    std::lock_guard<std::mutex> g(se->mu);
    std::string info = se->info;
    if (truncateInfo && info.size() > 100) info.resize(100);
    out.push_back({std::to_string(se->id), se->user, se->host, se->db.empty() ? kNullCell : se->db, se->command,
                   std::to_string((now - se->sinceUs) / 1000000), se->command == "Sleep" ? "" : "executing", info.empty() ? kNullCell : info});
  }
  std::sort(out.begin(), out.end(), [](auto& a, auto& b) { return std::atoll(a[0].c_str()) < std::atoll(b[0].c_str()); });
  return out;
}

ResultSet Engine::infoSchema(const std::string& table) {
  ResultSet r;
  r.isResult = true;
  r.typed = true;
  std::vector<std::shared_ptr<Table>> tabs;
  std::map<std::string, std::string> views;
  std::set<std::string> dbs;
  {
    std::lock_guard<std::mutex> g(cat_.mu);
    for (auto& [k, t] : cat_.tables)
      if (t->db != "__tmp") tabs.push_back(t);
    views = cat_.views;
    dbs = cat_.dbs;
  }
  if (tUserName() != "root") {  // only what the user may read (MySQL filters information_schema the same way)
    std::lock_guard<std::mutex> ug(usersMu_);
    tabs.erase(std::remove_if(tabs.begin(), tabs.end(), [&](const std::shared_ptr<Table>& t) {
      return !hasPriv(tUserName(), "SELECT", lower(t->db), lower(t->name)) && !hasPriv(tUserName(), "SELECT", lower(t->db), "*");
    }), tabs.end());
  }
  auto S = [](const std::string& x) { return Value::S(x); };
  auto I = [](int64_t x) { return Value::I(x); };
  const std::string tn = lower(table);
  auto cols = [&](std::vector<std::string> names, std::vector<LType> types) {
    r.names = std::move(names);
    r.types = std::move(types);
  };
  const LType V = LType::VARCHAR, B = LType::BIGINT;
  if (tn == "schemata") {
    cols({"CATALOG_NAME", "SCHEMA_NAME", "DEFAULT_CHARACTER_SET_NAME", "DEFAULT_COLLATION_NAME", "SQL_PATH"}, {V, V, V, V, V});
    dbs.insert("information_schema");
    for (auto& d : dbs) r.vals.push_back({S("def"), S(d), S("utf8mb4"), S("utf8mb4_bin"), Value::Null()});
  } else if (tn == "tables") {
    cols({"TABLE_CATALOG", "TABLE_SCHEMA", "TABLE_NAME", "TABLE_TYPE", "ENGINE", "VERSION", "ROW_FORMAT", "TABLE_ROWS",
          "AVG_ROW_LENGTH", "DATA_LENGTH", "MAX_DATA_LENGTH", "INDEX_LENGTH", "DATA_FREE", "AUTO_INCREMENT", "CREATE_TIME",
          "UPDATE_TIME", "CHECK_TIME", "TABLE_COLLATION", "CHECKSUM", "CREATE_OPTIONS", "TABLE_COMMENT"},
         {V, V, V, V, V, B, V, B, B, B, B, B, B, B, LType::DATETIME, LType::DATETIME, LType::DATETIME, V, B, V, V});
    for (auto& t : tabs) {
      std::shared_lock<FairSharedMutex> lk(t->mu);
      r.vals.push_back({S("def"), S(t->db), S(t->name), S("BASE TABLE"), S("Columnstore"), I(10), S("Dynamic"), I(t->liveRows()),
                        I(0), I(t->vramBytes()), I(0), I(0), I(0), t->autoIncCol >= 0 ? I(t->autoNext) : Value::Null(), Value::Null(),
                        Value::Null(), Value::Null(), S("utf8mb4_bin"), Value::Null(), S(""), S("")});
    }
    for (auto& [k, sql] : views) {
      size_t d = k.find('.');
      std::vector<Value> row(21, Value::Null());
      row[0] = S("def"), row[1] = S(k.substr(0, d)), row[2] = S(k.substr(d + 1)), row[3] = S("VIEW"), row[20] = S("VIEW");
      r.vals.push_back(row);
    }
  } else if (tn == "columns") {
    cols({"TABLE_CATALOG", "TABLE_SCHEMA", "TABLE_NAME", "COLUMN_NAME", "ORDINAL_POSITION", "COLUMN_DEFAULT", "IS_NULLABLE",
          "DATA_TYPE", "CHARACTER_MAXIMUM_LENGTH", "CHARACTER_OCTET_LENGTH", "NUMERIC_PRECISION", "NUMERIC_SCALE", "DATETIME_PRECISION",
          "CHARACTER_SET_NAME", "COLLATION_NAME", "COLUMN_TYPE", "COLUMN_KEY", "EXTRA", "PRIVILEGES", "COLUMN_COMMENT",
          "GENERATION_EXPRESSION", "SRS_ID"},
         {V, V, V, V, B, V, V, V, B, B, B, B, B, V, V, V, V, V, V, V, V, B});
    for (auto& t : tabs)
      for (size_t i = 0; i < t->cols.size(); ++i) {
        const ColumnMeta& c = t->cols[i];
        std::string ct = colTypeText(c), dt = lower(ct.substr(0, ct.find_first_of("( ")));
        bool str = c.lt == LType::VARCHAR, num = c.lt == LType::INT || c.lt == LType::BIGINT || c.lt == LType::DOUBLE || c.lt == LType::DECIMAL;
        std::string key = (int)i == t->pk || std::count(t->pkCols.begin(), t->pkCols.end(), (int)i) ? "PRI" : "";
        for (auto& ix : t->indexes)
          if (key.empty() && !ix.cols.empty() && ix.cols[0] == (int)i) key = ix.unique && ix.cols.size() == 1 ? "UNI" : "MUL";
        // Declared length / precision from the type text: varchar(255), decimal(12,2), char(3)
        int64_t n1 = -1, n2 = -1;
        if (size_t lp = ct.find('('); lp != std::string::npos && (dt == "varchar" || dt == "char" || dt == "decimal" || dt == "numeric" ||
                                                                    dt == "varbinary" || dt == "binary")) {
          n1 = std::atoll(ct.c_str() + lp + 1);
          if (size_t cm = ct.find(',', lp); cm != std::string::npos) n2 = std::atoll(ct.c_str() + cm + 1);
        }
        Value maxLen = Value::Null();
        if (str) maxLen = I(n1 >= 0 ? n1 : dt == "tinytext" ? 255 : dt == "mediumtext" ? 16777215 : dt == "longtext" || dt == "json" ? 4294967295LL : 65535);
        Value prec = Value::Null(), scl = Value::Null();
        if (c.lt == LType::DECIMAL || dt == "decimal") prec = I(n1 >= 0 ? n1 : 10), scl = I(n2 >= 0 ? n2 : c.scale);
        else if (num) prec = I(c.lt == LType::DOUBLE ? 22 : dt == "tinyint" ? 3 : dt == "smallint" ? 5 : dt == "mediumint" ? 7 : c.lt == LType::INT ? 10 : 19),
                      scl = c.lt == LType::DOUBLE ? Value::Null() : I(0);
        const bool nowDef = c.hasDefault && c.def.kind == Value::STR && c.def.s == "CURRENT_TIMESTAMP";
        std::string extra = c.autoInc ? "auto_increment" : nowDef ? "DEFAULT_GENERATED" : "";
        Value def = !c.hasDefault ? Value::Null() : c.def.isNull() ? Value::Null() : !c.defText.empty() ? S(c.defText) : S(c.def.toString());
        r.vals.push_back({S("def"), S(t->db), S(t->name), S(c.name), I((int64_t)i + 1), def, S(c.notNull || (int)i == t->pk ? "NO" : "YES"), S(dt),
                          maxLen, str ? Value::I(maxLen.i * 4) : Value::Null(), prec, scl,
                          c.lt == LType::DATETIME ? I(0) : Value::Null(), str ? S("utf8mb4") : Value::Null(), str ? S("utf8mb4_bin") : Value::Null(),
                          S(ct), S(key), S(extra), S("select,insert,update,references"), S(""), S(""), Value::Null()});
      }
  } else if (tn == "statistics") {
    cols({"TABLE_CATALOG", "TABLE_SCHEMA", "TABLE_NAME", "NON_UNIQUE", "INDEX_SCHEMA", "INDEX_NAME", "SEQ_IN_INDEX",
          "COLUMN_NAME", "COLLATION", "CARDINALITY", "SUB_PART", "PACKED", "NULLABLE", "INDEX_TYPE", "COMMENT", "INDEX_COMMENT",
          "IS_VISIBLE", "EXPRESSION"},
         {V, V, V, B, V, V, B, V, V, B, B, V, V, V, V, V, V, V});
    for (auto& t : tabs) {
      auto nullable = [&](int c) { return S(t->cols[c].notNull || c == t->pk ? "" : "YES"); };
      for (size_t k = 0; k < t->pkCols.size(); ++k)
        r.vals.push_back({S("def"), S(t->db), S(t->name), I(0), S(t->db), S("PRIMARY"), I((int64_t)k + 1), S(t->cols[t->pkCols[k]].name),
                          S("A"), I(t->liveRows()), Value::Null(), Value::Null(), S(""), S("BTREE"), S(""), S(""), S("YES"), Value::Null()});
      for (auto& ix : t->indexes)
        for (size_t k = 0; k < ix.cols.size(); ++k)
          r.vals.push_back({S("def"), S(t->db), S(t->name), I(!ix.unique), S(t->db), S(ix.name), I((int64_t)k + 1),
                            S(t->cols[ix.cols[k]].name), S("A"), Value::Null(), Value::Null(), Value::Null(), nullable(ix.cols[k]),
                            S(ix.vec ? "VECTOR" : "BTREE"), S(""), S(""), S("YES"), Value::Null()});
    }
  } else if (tn == "key_column_usage") {
    cols({"CONSTRAINT_CATALOG", "CONSTRAINT_SCHEMA", "CONSTRAINT_NAME", "TABLE_CATALOG", "TABLE_SCHEMA", "TABLE_NAME",
          "COLUMN_NAME", "ORDINAL_POSITION", "POSITION_IN_UNIQUE_CONSTRAINT", "REFERENCED_TABLE_SCHEMA", "REFERENCED_TABLE_NAME",
          "REFERENCED_COLUMN_NAME"},
         {V, V, V, V, V, V, V, B, B, V, V, V});
    for (auto& t : tabs) {
      for (size_t k = 0; k < t->pkCols.size(); ++k)
        r.vals.push_back({S("def"), S(t->db), S("PRIMARY"), S("def"), S(t->db), S(t->name), S(t->cols[t->pkCols[k]].name),
                          I((int64_t)k + 1), Value::Null(), Value::Null(), Value::Null(), Value::Null()});
      for (auto& ix : t->indexes)
        if (ix.unique && ix.name != "PRIMARY")
          for (size_t k = 0; k < ix.cols.size(); ++k)
            r.vals.push_back({S("def"), S(t->db), S(ix.name), S("def"), S(t->db), S(t->name), S(t->cols[ix.cols[k]].name),
                              I((int64_t)k + 1), Value::Null(), Value::Null(), Value::Null(), Value::Null()});
      for (auto& f : t->fks)
        for (size_t k = 0; k < f.cols.size(); ++k) {
          const std::string rdb = f.refTable.find('.') != std::string::npos ? f.refTable.substr(0, f.refTable.find('.')) : t->db;
          const std::string rt = f.refTable.substr(f.refTable.find('.') == std::string::npos ? 0 : f.refTable.find('.') + 1);
          r.vals.push_back({S("def"), S(t->db), S(f.name), S("def"), S(t->db), S(t->name), S(f.cols[k]), I((int64_t)k + 1), I((int64_t)k + 1),
                            S(rdb), S(rt), k < f.refCols.size() ? S(f.refCols[k]) : Value::Null()});
        }
    }
  } else if (tn == "referential_constraints") {
    cols({"CONSTRAINT_CATALOG", "CONSTRAINT_SCHEMA", "CONSTRAINT_NAME", "UNIQUE_CONSTRAINT_CATALOG", "UNIQUE_CONSTRAINT_SCHEMA",
          "UNIQUE_CONSTRAINT_NAME", "MATCH_OPTION", "UPDATE_RULE", "DELETE_RULE", "TABLE_NAME", "REFERENCED_TABLE_NAME"},
         {V, V, V, V, V, V, V, V, V, V, V});
    for (auto& t : tabs)
      for (auto& f : t->fks) {
        const std::string rt = f.refTable.substr(f.refTable.find('.') == std::string::npos ? 0 : f.refTable.find('.') + 1);
        r.vals.push_back({S("def"), S(t->db), S(f.name), S("def"), S(t->db), S("PRIMARY"), S("NONE"), S(f.onUpdate), S(f.onDelete),
                          S(t->name), S(rt)});
      }
  } else if (tn == "check_constraints") {
    cols({"CONSTRAINT_CATALOG", "CONSTRAINT_SCHEMA", "CONSTRAINT_NAME", "CHECK_CLAUSE"}, {V, V, V, V});
  } else if (tn == "views") {
    cols({"TABLE_CATALOG", "TABLE_SCHEMA", "TABLE_NAME", "VIEW_DEFINITION", "CHECK_OPTION", "IS_UPDATABLE", "DEFINER", "SECURITY_TYPE",
          "CHARACTER_SET_CLIENT", "COLLATION_CONNECTION"}, {V, V, V, V, V, V, V, V, V, V});
    for (auto& [k, sql] : views) {
      size_t d = k.find('.');
      r.vals.push_back({S("def"), S(k.substr(0, d)), S(k.substr(d + 1)), S(sql), S("NONE"), S("NO"), S("root@%"), S("DEFINER"),
                        S("utf8mb4"), S("utf8mb4_bin")});
    }
  } else if (tn == "table_constraints") {
    cols({"CONSTRAINT_CATALOG", "CONSTRAINT_SCHEMA", "CONSTRAINT_NAME", "TABLE_SCHEMA", "TABLE_NAME", "CONSTRAINT_TYPE", "ENFORCED"},
         {V, V, V, V, V, V, V});
    for (auto& t : tabs) {
      if (!t->pkCols.empty()) r.vals.push_back({S("def"), S(t->db), S("PRIMARY"), S(t->db), S(t->name), S("PRIMARY KEY"), S("YES")});
      for (auto& ix : t->indexes)
        if (ix.unique && ix.name != "PRIMARY") r.vals.push_back({S("def"), S(t->db), S(ix.name), S(t->db), S(t->name), S("UNIQUE"), S("YES")});
      for (auto& f : t->fks) r.vals.push_back({S("def"), S(t->db), S(f.name), S(t->db), S(t->name), S("FOREIGN KEY"), S("NO")});
    }
  } else {  // the other information_schema tables: MySQL 8's columns (empty, or a few fixed rows)
    static const std::map<std::string, std::vector<std::string>> layouts = {
        {"routines", {"SPECIFIC_NAME", "ROUTINE_CATALOG", "ROUTINE_SCHEMA", "ROUTINE_NAME", "ROUTINE_TYPE", "DATA_TYPE",
                      "CHARACTER_MAXIMUM_LENGTH", "NUMERIC_PRECISION", "NUMERIC_SCALE", "DTD_IDENTIFIER", "ROUTINE_BODY",
                      "ROUTINE_DEFINITION", "EXTERNAL_NAME", "EXTERNAL_LANGUAGE", "PARAMETER_STYLE", "IS_DETERMINISTIC",
                      "SQL_DATA_ACCESS", "SQL_PATH", "SECURITY_TYPE", "CREATED", "LAST_ALTERED", "SQL_MODE", "ROUTINE_COMMENT",
                      "DEFINER", "CHARACTER_SET_CLIENT", "COLLATION_CONNECTION", "DATABASE_COLLATION"}},
        {"parameters", {"SPECIFIC_CATALOG", "SPECIFIC_SCHEMA", "SPECIFIC_NAME", "ORDINAL_POSITION", "PARAMETER_MODE",
                        "PARAMETER_NAME", "DATA_TYPE", "CHARACTER_MAXIMUM_LENGTH", "NUMERIC_PRECISION", "NUMERIC_SCALE",
                        "DTD_IDENTIFIER", "ROUTINE_TYPE"}},
        {"triggers", {"TRIGGER_CATALOG", "TRIGGER_SCHEMA", "TRIGGER_NAME", "EVENT_MANIPULATION", "EVENT_OBJECT_CATALOG",
                      "EVENT_OBJECT_SCHEMA", "EVENT_OBJECT_TABLE", "ACTION_ORDER", "ACTION_CONDITION", "ACTION_STATEMENT",
                      "ACTION_ORIENTATION", "ACTION_TIMING", "CREATED", "SQL_MODE", "DEFINER", "CHARACTER_SET_CLIENT",
                      "COLLATION_CONNECTION", "DATABASE_COLLATION"}},
        {"events", {"EVENT_CATALOG", "EVENT_SCHEMA", "EVENT_NAME", "DEFINER", "TIME_ZONE", "EVENT_BODY", "EVENT_DEFINITION",
                    "EVENT_TYPE", "EXECUTE_AT", "INTERVAL_VALUE", "INTERVAL_FIELD", "SQL_MODE", "STARTS", "ENDS", "STATUS",
                    "ON_COMPLETION", "CREATED", "LAST_ALTERED", "LAST_EXECUTED", "EVENT_COMMENT"}},
        {"partitions", {"TABLE_CATALOG", "TABLE_SCHEMA", "TABLE_NAME", "PARTITION_NAME", "SUBPARTITION_NAME",
                        "PARTITION_ORDINAL_POSITION", "PARTITION_METHOD", "PARTITION_EXPRESSION", "PARTITION_DESCRIPTION",
                        "TABLE_ROWS", "PARTITION_COMMENT"}},
        {"character_sets", {"CHARACTER_SET_NAME", "DEFAULT_COLLATE_NAME", "DESCRIPTION", "MAXLEN"}},
        {"collations", {"COLLATION_NAME", "CHARACTER_SET_NAME", "ID", "IS_DEFAULT", "IS_COMPILED", "SORTLEN", "PAD_ATTRIBUTE"}},
        {"engines", {"ENGINE", "SUPPORT", "COMMENT", "TRANSACTIONS", "XA", "SAVEPOINTS"}},
        {"user_privileges", {"GRANTEE", "TABLE_CATALOG", "PRIVILEGE_TYPE", "IS_GRANTABLE"}},
        {"schema_privileges", {"GRANTEE", "TABLE_CATALOG", "TABLE_SCHEMA", "PRIVILEGE_TYPE", "IS_GRANTABLE"}},
        {"table_privileges", {"GRANTEE", "TABLE_CATALOG", "TABLE_SCHEMA", "TABLE_NAME", "PRIVILEGE_TYPE", "IS_GRANTABLE"}},
        {"column_privileges", {"GRANTEE", "TABLE_CATALOG", "TABLE_SCHEMA", "TABLE_NAME", "COLUMN_NAME", "PRIVILEGE_TYPE", "IS_GRANTABLE"}},
        {"processlist", {"ID", "USER", "HOST", "DB", "COMMAND", "TIME", "STATE", "INFO"}},
        {"keywords", {"WORD", "RESERVED"}},
        {"plugins", {"PLUGIN_NAME", "PLUGIN_VERSION", "PLUGIN_STATUS", "PLUGIN_TYPE", "PLUGIN_LIBRARY", "PLUGIN_LICENSE"}},
        {"global_variables", {"VARIABLE_NAME", "VARIABLE_VALUE"}},
        {"session_variables", {"VARIABLE_NAME", "VARIABLE_VALUE"}},
        {"innodb_tables", {"TABLE_ID", "NAME", "FLAG", "N_COLS", "SPACE", "ROW_FORMAT"}},
        {"st_geometry_columns", {"TABLE_CATALOG", "TABLE_SCHEMA", "TABLE_NAME", "COLUMN_NAME", "SRS_NAME", "SRS_ID", "GEOMETRY_TYPE_NAME"}},
        {"resource_groups", {"RESOURCE_GROUP_NAME", "RESOURCE_GROUP_TYPE", "RESOURCE_GROUP_ENABLED", "VCPU_IDS", "THREAD_PRIORITY"}},
    };
    if (tn == "query_stats") {  // per statement shape (literals replaced by ?), heaviest first
      cols({"DIGEST_TEXT", "COUNT_STAR", "ERRORS", "TOTAL_MS", "AVG_MS", "MAX_MS", "ROWS", "LAST_SEEN"},
           {V, LType::BIGINT, LType::BIGINT, LType::DOUBLE, LType::DOUBLE, LType::DOUBLE, LType::BIGINT, LType::DATETIME});
      if (!isSuper(tUserName())) return r;
      for (auto& q : monitor::queryStats())
        r.vals.push_back({S(q.digest), I((int64_t)q.count), I((int64_t)q.errors), Value::D(q.totalMs), Value::D(q.count ? q.totalMs / q.count : 0),
                          Value::D(q.maxMs), I(q.rows), Value::Ts(q.lastUs)});
      return r;
    }
    if (tn == "processlist") {
      cols({"ID", "USER", "HOST", "DB", "COMMAND", "TIME", "STATE", "INFO"}, {LType::BIGINT, V, V, V, V, LType::BIGINT, V, V});
      for (auto& row : processRows(false)) {
        std::vector<Value> out;
        for (size_t i = 0; i < row.size(); ++i)
          out.push_back(row[i] == kNullCell ? Value::Null() : i == 0 || i == 5 ? I(std::atoll(row[i].c_str())) : S(row[i]));
        r.vals.push_back(out);
      }
      return r;
    }
    auto it = layouts.find(tn);
    if (it == layouts.end()) throw SqlError("Unknown table '" + table + "' in information_schema", 1109);
    cols(it->second, std::vector<LType>(it->second.size(), V));
    auto row = [&](std::vector<std::string> v) {
      std::vector<Value> out;
      for (auto& x : v) out.push_back(S(x));
      out.resize(r.names.size(), Value::Null());
      r.vals.push_back(out);
    };
    if (tn == "character_sets") row({"utf8mb4", "utf8mb4_0900_ai_ci", "UTF-8 Unicode", "4"});
    if (tn == "collations")
      for (auto* c : {"utf8mb4_0900_ai_ci", "utf8mb4_bin", "utf8mb4_general_ci", "utf8mb4_unicode_ci"})
        row({c, "utf8mb4", "0", std::string(c) == "utf8mb4_0900_ai_ci" ? "Yes" : "", "Yes", "1", "PAD SPACE"});
    if (tn == "engines") {
      row({"Columnstore", "DEFAULT", "warpstore GPU columnstore with a row write buffer", "YES", "NO", "YES"});
      row({"InnoDB", "YES", "Alias of the warpstore columnstore", "YES", "NO", "YES"});
    }
    if (tn == "global_variables" || tn == "session_variables")
      for (auto& [k, v] : systemVariables()) row({upper(k), v});
    if (tn == "routines")
      for (auto& d : routineDefs()) {
        std::string u = upper(d.substr(0, 200));
        const bool fn = u.find("FUNCTION") != std::string::npos && (u.find("PROCEDURE") == std::string::npos || u.find("FUNCTION") < u.find("PROCEDURE"));
        size_t k = u.find(fn ? "FUNCTION" : "PROCEDURE") + (fn ? 8 : 9);
        while (k < d.size() && isspace((unsigned char)d[k])) ++k;
        size_t e = k;
        while (e < d.size() && (isalnum((unsigned char)d[e]) || d[e] == '_' || d[e] == '.' || d[e] == '`')) ++e;
        std::string name = d.substr(k, e - k);
        name.erase(std::remove(name.begin(), name.end(), '`'), name.end());
        std::string db = currentDb();
        if (name.find('.') != std::string::npos) db = name.substr(0, name.find('.')), name = name.substr(name.find('.') + 1);
        row({name, "def", db, name, fn ? "FUNCTION" : "PROCEDURE", "", "", "", "", "", "SQL", d, "", "SQL", "SQL", "NO",
             "CONTAINS SQL", "", "DEFINER", "", "", "", "", "root@%", "utf8mb4", "utf8mb4_bin", "utf8mb4_bin"});
      }
  }
  return r;
}

// System variables (SHOW VARIABLES, @@name): MySQL 8 defaults where warpstore behaves the same, the
// session's settings, and warpstore_* knobs.
std::vector<std::pair<std::string, std::string>> systemVariables() {
  const std::string iso = tIsoSnapshot ? "REPEATABLE-READ" : "READ-COMMITTED";
  return {{"version", "8.0.32-warpstore"}, {"version_comment", "warpstore (GPU-accelerated HTAP)"}, {"version_compile_os", "Linux"},
          {"version_compile_machine", "x86_64"}, {"protocol_version", "10"}, {"license", "GPL"}, {"autocommit", "ON"},
          {"character_set_client", "utf8mb4"}, {"character_set_connection", "utf8mb4"}, {"character_set_results", "utf8mb4"},
          {"character_set_server", "utf8mb4"}, {"character_set_database", "utf8mb4"}, {"character_set_system", "utf8mb3"},
          {"collation_connection", ciColl() ? "utf8mb4_0900_ai_ci" : "utf8mb4_bin"}, {"collation_server", "utf8mb4_0900_ai_ci"},
          {"collation_database", "utf8mb4_0900_ai_ci"}, {"lower_case_table_names", "1"}, {"lower_case_file_system", "OFF"},
          {"max_allowed_packet", "1073741824"}, {"net_buffer_length", "16384"}, {"sql_mode", ""}, {"time_zone", "SYSTEM"},
          {"system_time_zone", "UTC"}, {"transaction_isolation", iso}, {"tx_isolation", iso}, {"transaction_read_only", "0"},
          {"tx_read_only", "0"}, {"wait_timeout", "28800"}, {"interactive_timeout", "28800"}, {"net_write_timeout", "60"},
          {"net_read_timeout", "30"}, {"connect_timeout", "10"}, {"max_connections", "1000"}, {"auto_increment_increment", "1"},
          {"auto_increment_offset", "1"}, {"sql_auto_is_null", "0"}, {"sql_select_limit", "18446744073709551615"},
          {"sql_safe_updates", "0"}, {"foreign_key_checks", "1"}, {"unique_checks", "1"}, {"explicit_defaults_for_timestamp", "ON"},
          {"default_storage_engine", "InnoDB"}, {"default_authentication_plugin", "mysql_native_password"},
          {"performance_schema", "OFF"}, {"have_ssl", "YES"}, {"have_openssl", "YES"}, {"query_cache_size", "0"},
          {"query_cache_type", "OFF"}, {"init_connect", ""}, {"lc_messages", "en_US"}, {"lc_time_names", "en_US"},
          {"default_week_format", "0"}, {"div_precision_increment", "4"}, {"group_concat_max_len", "1048576"},
          {"innodb_version", "8.0.32"}, {"warning_count", "0"}, {"error_count", "0"}, {"session_track_gtids", "OFF"},
          {"last_insert_id", std::to_string(tLastInsertId)}, {"identity", std::to_string(tLastInsertId)}, {"hostname", "localhost"},
          {"port", "3306"}, {"socket", "/tmp/mysql.sock"}, {"datadir", "/var/lib/warpstore/"}, {"basedir", "/usr/"},
          {"tmpdir", "/tmp"}, {"server_id", "1"}, {"log_bin", "OFF"}, {"binlog_format", "ROW"}, {"gtid_mode", "OFF"},
          {"read_only", "0"}, {"super_read_only", "0"}, {"max_execution_time", "0"}, {"innodb_lock_wait_timeout", "50"},
          {"lock_wait_timeout", "31536000"}, {"max_join_size", "18446744073709551615"}, {"sql_big_selects", "1"},
          {"local_infile", "ON"}, {"secure_file_priv", ""}, {"skip_name_resolve", "ON"}, {"require_secure_transport", "OFF"},
          {"default_collation_for_utf8mb4", "utf8mb4_0900_ai_ci"}, {"information_schema_stats_expiry", "86400"},
          {"warpstore_sync_durability", tSyncDurability < 0 ? "DEFAULT" : tSyncDurability ? "ON" : "OFF"},
          {"warpstore_codegen", jitMode() == JIT_OFF ? "OFF" : jitMode() == JIT_SYNC ? "SYNC" : "ON"},
          {"warpstore_dense_groupby", tDense ? "ON" : "OFF"}, {"warpstore_join_filters", tJoinFilters ? "ON" : "OFF"},
          {"warpstore_join_filter_rows", std::to_string(tLazyRows)}, {"warpstore_gpu", gpuDescribe()}};
}
Value sysVarValue(const std::string& name) {
  const std::string n = lower(name);
  for (auto& [k, v] : systemVariables())
    if (k == n) {
      if (v == "ON" || v == "OFF") return n == "autocommit" ? Value::I(v == "ON") : Value::S(v);
      char* end = nullptr;
      long long x = std::strtoll(v.c_str(), &end, 10);
      if (!v.empty() && end && *end == 0 && v.size() < 19) return Value::I(x);
      return Value::S(v);
    }
  return Value::Null();
}

ResultSet Engine::doShowMisc(Stmt& s) {
  const std::string k = s.setName;
  ResultSet r = table({"Variable_name", "Value"}, {LType::VARCHAR, LType::VARCHAR});
  if (k == "CREATE TABLE") {
    if (auto t = cat_.find(s.table)) {
      std::shared_lock<FairSharedMutex> lk(t->mu);
      r = table({"Table", "Create Table"}, {LType::VARCHAR, LType::VARCHAR});
      r.rows.push_back({t->name, showCreate(*t)});
      return r;
    }
    k == "CREATE TABLE" ? void() : void();
  }
  if (k == "CREATE TABLE" || k == "CREATE VIEW") {
    std::lock_guard<std::mutex> g(cat_.mu);
    auto it = cat_.views.find(Catalog::key(s.table));
    if (it == cat_.views.end()) throw SqlError("Table '" + s.table + "' doesn't exist", 1146);
    r = table({"View", "Create View", "character_set_client", "collation_connection"}, {LType::VARCHAR, LType::VARCHAR, LType::VARCHAR, LType::VARCHAR});
    r.rows.push_back({plainName(s.table), "CREATE VIEW `" + plainName(s.table) + "` AS " + it->second, "utf8mb4", "utf8mb4_bin"});
    return r;
  }
  if (k == "CREATE DATABASE") {
    r = table({"Database", "Create Database"}, {LType::VARCHAR, LType::VARCHAR});
    r.rows.push_back({s.table, "CREATE DATABASE `" + s.table + "`"});
    return r;
  }
  if (k == "INDEX") {
    r = table({"Table", "Non_unique", "Key_name", "Seq_in_index", "Column_name", "Collation", "Cardinality", "Sub_part",
               "Packed", "Null", "Index_type", "Comment"},
              {LType::VARCHAR, LType::BIGINT, LType::VARCHAR, LType::BIGINT, LType::VARCHAR, LType::VARCHAR, LType::BIGINT,
               LType::VARCHAR, LType::VARCHAR, LType::VARCHAR, LType::VARCHAR, LType::VARCHAR});
    auto t = cat_.get(s.table);
    for (size_t i = 0; i < t->pkCols.size(); ++i)
      r.rows.push_back({t->name, "0", "PRIMARY", std::to_string(i + 1), t->cols[t->pkCols[i]].name, "A", std::to_string(t->liveRows()),
                        kNullCell, kNullCell, "", "HASH", ""});
    for (auto& ix : t->indexes)
      for (size_t i = 0; i < ix.cols.size(); ++i)
        r.rows.push_back({t->name, ix.unique ? "0" : "1", ix.name, std::to_string(i + 1), t->cols[ix.cols[i]].name, "A",
                          kNullCell, kNullCell, kNullCell, "YES", "HASH", ""});
    if (!t->shardKey.empty())
      for (size_t i = 0; i < t->shardKey.size(); ++i)
        r.rows.push_back({t->name, "1", "__SHARDKEY", std::to_string(i + 1), t->cols[t->shardKey[i]].name, kNullCell,
                          kNullCell, kNullCell, kNullCell, "", "METADATA_ONLY", ""});
    return r;
  }
  if (k == "VARIABLES" || k == "STATUS") {
    std::vector<std::pair<std::string, std::string>> vars;
    if (k == "VARIABLES")
      vars = systemVariables();
    else
      vars = {{"Uptime", "1"}, {"Threads_connected", "1"}, {"Queries", "0"}};
    for (auto& [n, v] : vars)
      if (likeCI(n, s.path)) r.rows.push_back({n, v});
    return r;
  }
  if (k == "TABLE STATUS") {
    r = table({"Name", "Engine", "Version", "Row_format", "Rows", "Avg_row_length", "Data_length", "Max_data_length",
               "Index_length", "Data_free", "Auto_increment", "Create_time", "Update_time", "Check_time", "Collation",
               "Checksum", "Create_options", "Comment"},
              std::vector<LType>(18, LType::VARCHAR));
    std::vector<std::shared_ptr<Table>> tabs;
    {
      std::lock_guard<std::mutex> g(cat_.mu);
      const std::string db = s.table.empty() ? currentDb() : s.table;
      for (auto& [key, t] : cat_.tables)
        if (lower(t->db) == lower(db)) tabs.push_back(t);
    }
    for (auto& t : tabs)
      if (likeCI(t->name, s.path))
        r.rows.push_back({t->name, "Columnstore", "10", "Dynamic", std::to_string(t->liveRows()), "0", std::to_string(t->vramBytes()),
                          "0", "0", "0", t->autoIncCol >= 0 ? std::to_string(t->autoNext) : kNullCell, kNullCell, kNullCell,
                          kNullCell, "utf8mb4_bin", kNullCell, "", ""});
    return r;
  }
  if (k == "ENGINES") {
    r = table({"Engine", "Support", "Comment", "Transactions", "XA", "Savepoints"}, std::vector<LType>(6, LType::VARCHAR));
    r.rows.push_back({"Columnstore", "DEFAULT", "GPU-resident columnstore with a CPU write buffer", "YES", "NO", "NO"});
    return r;
  }
  if (k == "COLLATION") {
    r = table({"Collation", "Charset", "Id", "Default", "Compiled", "Sortlen"}, std::vector<LType>(6, LType::VARCHAR));
    r.rows.push_back({"utf8mb4_bin", "utf8mb4", "46", "Yes", "Yes", "1"});
    return r;
  }
  if (k == "CHARSET") {
    r = table({"Charset", "Description", "Default collation", "Maxlen"}, std::vector<LType>(4, LType::VARCHAR));
    r.rows.push_back({"utf8mb4", "UTF-8 Unicode", "utf8mb4_bin", "4"});
    return r;
  }
  if (k == "VRAM") {  // SHOW VRAM: where each column's segments live (VRAM cache / hints)
    r = table({"Database", "Table", "Column", "Hint", "VRAM_MB", "Host_MB", "Heat"},
              {LType::VARCHAR, LType::VARCHAR, LType::VARCHAR, LType::VARCHAR, LType::DOUBLE, LType::DOUBLE, LType::DOUBLE});
    std::vector<std::shared_ptr<Table>> tabs;
    {
      std::lock_guard<std::mutex> g(cat_.mu);
      for (auto& [key, t] : cat_.tables) tabs.push_back(t);
    }
    std::sort(tabs.begin(), tabs.end(), [](auto& a, auto& b) { return a->db + "." + a->name < b->db + "." + b->name; });
    for (auto& t : tabs) {
      if (t->db == "__tmp") continue;
      if (sessionUser() != "root") {
        std::lock_guard<std::mutex> ug(usersMu_);
        if (!hasPriv(sessionUser(), "SELECT", lower(t->db), lower(t->name)) && !hasPriv(sessionUser(), "SELECT", lower(t->db), "*")) continue;
      }
      std::shared_lock<FairSharedMutex> lk(t->mu);
      for (size_t c = 0; c < t->cols.size(); ++c) {
        double vram = 0, host = 0, heat = 0;
        for (auto& sp : t->segs) {
          if (c >= sp->dev.size() || !sp->dev[c] || segCold(*sp->d, (int)c)) continue;
          const double mb = (double)(sp->enc[c] ? sp->encHost[c].size() : sp->host[c].size()) / 1048576.0;
          (sp->mapped[c] ? host : vram) += mb;
          if (c < sp->heat.size()) heat += sp->heat[c];
        }
        const int h = t->vramHint((int)c);
        auto f2 = [](double x) { char b[32]; snprintf(b, sizeof b, "%.2f", x); return std::string(b); };
        r.rows.push_back({t->db, t->name, t->cols[c].name, h == 1 ? "PIN" : h == 2 ? "NEVER" : "AUTO", f2(vram), f2(host), f2(heat)});
      }
    }
    return r;
  }
  if (k == "PROCESSLIST") {
    r = table({"Id", "User", "Host", "db", "Command", "Time", "State", "Info"}, {LType::BIGINT, LType::VARCHAR, LType::VARCHAR, LType::VARCHAR,
                                                                              LType::VARCHAR, LType::BIGINT, LType::VARCHAR, LType::VARCHAR});
    for (auto& row : processRows(s.setValue != "FULL")) r.rows.push_back(row);
    return r;
  }
  if (k == "GRANTS") {
    r = table({"Grants for root@%"}, {LType::VARCHAR});
    std::lock_guard<std::mutex> g(usersMu_);
    if (sessionUser() == "root") r.rows.push_back({"GRANT ALL PRIVILEGES ON *.* TO 'root'@'%'"});
    for (auto& e : grants_[sessionUser()]) {
      if (e[0] == '@') r.rows.push_back({"GRANT '" + e.substr(1) + "' TO '" + sessionUser() + "'@'%'"});
      else r.rows.push_back({"GRANT " + e.substr(0, e.find('|')) + " ON " + e.substr(e.find('|') + 1) + " TO '" + sessionUser() + "'@'%'"});
    }
    return r;
  }
  if (k == "WARNINGS" || k == "ERRORS") {
    r = table({"Level", "Code", "Message"}, {LType::VARCHAR, LType::BIGINT, LType::VARCHAR});
    return r;
  }
  r = table({"Name", "Value"}, {LType::VARCHAR, LType::VARCHAR});  // TRIGGERS, EVENTS, PLUGINS, ..: empty
  return r;
}


// A columnar INSERT .. SELECT result (ResultSet::insertCols) as rows of values.
static std::vector<std::vector<Value>> columnsToRows(const ResultSet& r) {
  const size_t ni = r.numCols.size();
  int64_t n = 0;
  for (size_t i = 0; i < ni; ++i) n = std::max<int64_t>(n, !r.intCols[i].empty() ? (int64_t)r.intCols[i].size() : (int64_t)r.numCols[i].size());
  std::vector<std::vector<Value>> rows((size_t)n);
  for (int64_t k = 0; k < n; ++k) {
    rows[k].reserve(ni);
    for (size_t i = 0; i < ni; ++i) {
      if (!r.intCols[i].empty()) {
        const int64_t v = r.intCols[i][k];
        rows[k].push_back(v == kNullI64 ? Value::Null() : r.types[i] == LType::DATETIME ? Value::Ts(v) : Value::I(v));
      } else {
        rows[k].push_back(typedVal(r.numCols[i][k], r.types[i], r.colDicts[i].get(), r.colScales[i]));
      }
    }
  }
  return rows;
}

static ExprP boundLitOrdinal(int64_t i) {  // an (unbound) integer literal: ORDER BY <ordinal>
  auto e = std::make_shared<Expr>();
  e->kind = Expr::LIT;
  e->lit = Value::I(i);
  return e;
}

// INSERT .. SELECT from a columnar result: each target column is converted on its own (strings
// recoded once per distinct value, matching types copied, anything else through a Value), then the
// rows are appended in bulk (Table::appendColumns). Caller holds the table's exclusive lock.
static int64_t insertColumns(Table& t, const std::vector<int>& map, const ResultSet& r) {
  const size_t nc = t.cols.size(), ni = r.numCols.size();
  if (ni != map.size()) throw SqlError("Column count doesn't match value count at row 1", 1136);
  int64_t n = 0;
  for (size_t i = 0; i < ni; ++i) n = std::max<int64_t>(n, !r.intCols[i].empty() ? (int64_t)r.intCols[i].size() : (int64_t)r.numCols[i].size());
  if (n == 0) return 0;
  std::vector<int> src(nc, -1);
  for (size_t i = 0; i < ni; ++i) src[map[i]] = (int)i;
  const std::vector<Value> defs = defaultRow(t);
  std::vector<std::vector<int64_t>> iv(nc);
  std::vector<std::vector<double>> dv(nc);
  auto valueOf = [&](size_t i, int64_t k) {  // result cell k of item i as a Value
    if (!r.intCols[i].empty()) {
      const int64_t v = r.intCols[i][k];
      return v == kNullI64 ? Value::Null() : r.types[i] == LType::DATETIME ? Value::Ts(v) : Value::I(v);
    }
    return typedVal(r.numCols[i][k], r.types[i], r.colDicts[i].get(), r.colScales[i]);
  };
  std::vector<std::exception_ptr> errs(nc);
  ioParallel(nc, [&](size_t c) {  // one target column per task (each has its own dictionary)
    try {
      const ColumnMeta& cm = t.cols[c];
      const bool f64 = cm.pt == PType::F64;
      if (f64) dv[c].resize(n);
      else iv[c].resize(n);
      const int i = src[c];
      if ((int)c == t.autoIncCol) {  // NULL / 0: the next number (assigned in order below)
        for (int64_t k = 0; k < n; ++k) {
          const Value v = i < 0 ? Value::Null() : valueOf(i, k);
          iv[c][k] = v.isNull() ? 0 : (int64_t)v.num();
        }
        return;
      }
      if (i < 0) {  // not in the column list: the default
        int64_t i0 = 0;
        double d0 = 0;
        t.rawOf((int)c, defs[c], i0, d0);
        if (f64) std::fill(dv[c].begin(), dv[c].end(), d0);
        else std::fill(iv[c].begin(), iv[c].end(), i0);
        return;
      }
      const LType st = r.types[i];
      const int64_t nul = cm.pt == PType::I64 ? kNullI64 : kNullI32;
      auto nullable = [&](bool isNull) {
        if (!isNull) return;
        if (cm.notNull || (int)c == t.pk) throw SqlError("Column '" + cm.name + "' cannot be null", 1048);
        t.hasNull[c] = 1;
      };
      if (st == LType::VARCHAR && cm.lt == LType::VARCHAR && r.colDicts[i]) {
        const Dict& sd = *r.colDicts[i];
        Dict& td = *t.dicts[c];
        // the distinct source strings in order of first appearance, coded in one batch
        std::vector<int32_t> code(sd.strs.size(), -1);
        std::vector<std::string_view> firsts;
        for (int64_t k = 0; k < n; ++k) {
          const double x = r.numCols[i][k];
          if (x != x) continue;
          int32_t& m = code.at((size_t)x);
          if (m == -1) m = -2 - (int32_t)firsts.size(), firsts.push_back(sd.strs[(size_t)x]);
        }
        std::vector<int64_t> fc(firsts.size());
        td.getOrAddMany(firsts.data(), firsts.size(), fc.data(), kNullI32);
        for (int64_t k = 0; k < n; ++k) {
          const double x = r.numCols[i][k];
          if (x != x) {
            nullable(true);
            iv[c][k] = kNullI32;
            continue;
          }
          iv[c][k] = fc[(size_t)(-2 - code[(size_t)x])];
        }
      } else if (!r.intCols[i].empty() && st == cm.lt && cm.pt == PType::I64) {  // BIGINT / DATETIME as is
        for (int64_t k = 0; k < n; ++k) {
          const int64_t v = r.intCols[i][k];
          if (v == kNullI64) nullable(true);
          iv[c][k] = v;
        }
      } else if (r.intCols[i].empty() && st == cm.lt && (st == LType::INT || st == LType::DATE || st == LType::DOUBLE)) {
        for (int64_t k = 0; k < n; ++k) {
          const double x = r.numCols[i][k];
          if (x != x) nullable(true);
          if (f64) dv[c][k] = x;
          else iv[c][k] = x != x ? nul : std::llround(x);
        }
      } else {
        int64_t i0;
        double d0;
        for (int64_t k = 0; k < n; ++k) {
          Value v = valueOf(i, k);
          if (!v.isNull() && cm.lt == LType::VARCHAR && v.kind != Value::STR) v = Value::S(v.toString());
          t.rawOf((int)c, v, i0, d0);
          if (f64) dv[c][k] = d0;
          else iv[c][k] = i0;
        }
      }
    } catch (...) {
      errs[c] = std::current_exception();
    }
  });
  for (auto& e : errs)
    if (e) std::rethrow_exception(e);
  if (t.autoIncCol >= 0) {
    for (int64_t k = 0; k < n; ++k) {
      int64_t& v = iv[t.autoIncCol][k];
      if (v == 0) v = t.autoNext++;
      else t.autoNext = std::max<int64_t>(t.autoNext, v + 1);
    }
    t.lastInsertId = iv[t.autoIncCol][n - 1];
  }
  t.appendColumns(iv, dv, n);
  return n;
}

ResultSet Engine::doInsert(Stmt& s) {
  std::vector<std::vector<Value>> rows;
  ResultSet cols;  // INSERT .. SELECT whose projection came back as columns
  if (s.select) {  // INSERT .. SELECT: run the query first (it may read the target table)
    QueryCtx q;
    ResultSet r;
    // Plain inserts outside a transaction take the result as columns (no per-row values).
    tInsertCols = !undoLog() && !s.insertIgnore && !s.replaceInto && s.onDup.empty();
    // Rows whose order matters (LIMIT without ORDER BY; AUTO_INCREMENT numbers): ordered by all their
    // values, so a WAL replay (recovery, replicas), whatever its physical row order, gets the same.
    if (!s.select->unionNext && s.select->orderBy.empty() && !s.select->items.empty()) {
      bool numbered = false;
      if (auto tt = cat_.find(Catalog::key(s.table)); tt && tt->autoIncCol >= 0) {
        numbered = s.insertCols.empty();
        for (auto& c : s.insertCols) numbered = numbered || strcasecmp(c.c_str(), tt->cols[tt->autoIncCol].name.c_str()) == 0;
      }
      bool star = false;
      for (auto& it : s.select->items) star = star || it.e->kind == Expr::STAR;
      std::shared_ptr<Table> src;
      if (star && s.select->from.size() == 1 && !s.select->from[0].sub && s.select->joinConds.empty())
        src = cat_.find(Catalog::key(s.select->from[0].name));
      if ((s.select->limit >= 0 || numbered) && !star)
        for (size_t i = 0; i < s.select->items.size(); ++i)
          s.select->orderBy.push_back(OrderItem{boundLitOrdinal((int64_t)i + 1), false});
      else if ((s.select->limit >= 0 || numbered) && src && s.select->items.size() == 1)  // SELECT *: by the table's columns
        for (auto& c : src->cols)
          if (c.pt != PType::VEC) s.select->orderBy.push_back(OrderItem{mkCol(s.select->from[0].alias.empty() ? s.select->from[0].name : s.select->from[0].alias, c.name), false});
    }
    try {
      r = query(*s.select, q, false, true);
    } catch (...) {
      tInsertCols = false;
      dropTemps(q);
      throw;
    }
    tInsertCols = false;
    dropTemps(q);
    if (r.insertCols) cols = std::move(r);
    else rows = std::move(r.vals);
  }
  auto t = cat_.get(s.table);
  std::unique_lock<FairSharedMutex> lk(t->mu);
  if (hasUniqueKey(*t)) waitTxnFree(*t, lk);  // another open transaction may hold a row with the same key
  std::vector<int> map;
  if (s.insertCols.empty()) {
    for (size_t i = 0; i < t->cols.size(); ++i) map.push_back((int)i);
  } else {
    for (auto& c : s.insertCols) {
      int i = t->colIndex(c);
      if (i < 0) throw SqlError("Unknown column '" + c + "' in 'field list'", 1054);
      map.push_back(i);
    }
  }
  if (!s.select) {
    BindCtx bc;
    PostCtx pc;
    for (auto& row : s.rows) {
      if (row.empty() && s.insertCols.empty()) { rows.push_back({}); continue; }  // VALUES (): all defaults
      if (row.size() != map.size())
        throw SqlError("Column count doesn't match value count at row " + std::to_string(rows.size() + 1), 1136);
      std::vector<Value> vals;
      for (auto& x : row) {
        if (x->kind == Expr::COL && upper(x->name) == "DEFAULT" && x->table.empty()) { vals.push_back(Value::S("\x01DEFAULT")); continue; }
        bind(x, bc);
        vals.push_back(evalPost(x, pc));
      }
      rows.push_back(std::move(vals));
    }
  }
  int64_t n = 0;
  if (cols.insertCols) {
    bool vec = false;
    for (auto& cm : t->cols) vec = vec || cm.pt == PType::VEC;
    if (!vec) n = insertColumns(*t, map, cols);
    else {  // (VECTOR columns: as rows after all)
      rows = columnsToRows(cols);
      n = insertRows(*t, map, rows, s);
    }
  } else {
    n = insertRows(*t, map, rows, s);
  }
  ResultSet r = okResult(n);
  r.lastInsertId = t->autoIncCol >= 0 ? t->lastInsertId : 0;
  if (r.lastInsertId) tLastInsertId = r.lastInsertId;
  return r;
}

// Appends rows (values for the columns in `map`), handling INSERT IGNORE, REPLACE and
// ON DUPLICATE KEY UPDATE against the PRIMARY KEY. Caller holds the table's exclusive lock.
int64_t Engine::insertRows(Table& t, const std::vector<int>& map, std::vector<std::vector<Value>>& rows, const Stmt& s) {
  int64_t affected = 0;
  BindCtx bc;
  bc.tabs = {cat_.get(t.db + "." + t.name)};
  bc.aliases = {t.name};
  std::vector<ExprP> dupExprs;
  std::vector<int> dupCols;
  for (auto& [name, e] : s.onDup) {
    int ci = t.colIndex(name);
    if (ci < 0) throw SqlError("Unknown column '" + name + "' in 'field list'", 1054);
    dupCols.push_back(ci);
    dupExprs.push_back(e);
  }
  for (auto& in : rows) {
    std::vector<Value> vals = defaultRow(t);
    if (!in.empty() && in.size() != map.size()) throw SqlError("Column count doesn't match value count", 1136);
    for (size_t i = 0; i < in.size(); ++i)
      if (!(in[i].kind == Value::STR && in[i].s == "\x01DEFAULT")) vals[map[i]] = in[i];
    // A conflict with an existing row (PRIMARY KEY or UNIQUE key): IGNORE skips the new row, REPLACE
    // deletes the old one, ON DUPLICATE KEY UPDATE updates it. True: the new row is consumed.
    auto onConflict = [&](Loc old, const std::string& err) -> bool {
      if (s.insertIgnore) return true;
      if (s.replaceInto) {
        t.deleteAt(old);
        affected += 1;
        return false;
      }
      if (s.onDup.empty()) throw SqlError(err, 1062);
      std::vector<Value> cur;
      for (size_t c = 0; c < t.cols.size(); ++c) cur.push_back(cellValue(t, old, (int)c));
      PostCtx pc;
      pc.row = &cur;
      pc.colSlot = [](int, int c) { return c; };
      std::vector<Value> upd = cur;
      std::function<ExprP(const ExprP&)> subst = [&](const ExprP& x) -> ExprP {  // VALUES(col): the new value
        if (x->kind == Expr::FUNC && x->op == "VALUES" && x->args.size() == 1 && x->args[0]->kind == Expr::COL) {
          int ci = t.colIndex(x->args[0]->name);
          if (ci < 0) throw SqlError("Unknown column '" + x->args[0]->name + "' in 'field list'", 1054);
          return mkLit(vals[ci]);
        }
        auto y = std::make_shared<Expr>(*x);
        y->bound = false;
        for (auto& a : y->args) a = subst(a);
        return y;
      };
      for (size_t k = 0; k < dupExprs.size(); ++k) {
        ExprP e = subst(dupExprs[k]);
        bind(e, bc);
        upd[dupCols[k]] = evalPost(e, pc);
      }
      bool same = true;
      for (size_t c = 0; c < upd.size(); ++c) same = same && cmpVal(upd[c], cur[c]) == 0 && upd[c].isNull() == cur[c].isNull();
      if (same) return true;  // MySQL: 0 rows affected when nothing changes
      t.deleteAt(old);
      t.appendRow(upd);
      affected += 2;
      return true;
    };
    if (t.pk >= 0 && !vals[t.pk].isNull()) {
      auto it = t.pkIndex.find((int64_t)numOf(vals[t.pk]));
      if (it != t.pkIndex.end() && onConflict(it->second, "Duplicate entry '" + vals[t.pk].toString() + "' for key 'PRIMARY'")) continue;
    }
    for (int tries = 0;; ++tries) {
      try {
        t.appendRow(vals);
        ++affected;
        break;
      } catch (SqlError& e) {  // UNIQUE key (or composite PRIMARY KEY) conflict
        if (e.code != 1062 || t.lastDup.seg == INT32_MIN || tries > 16 || (!s.insertIgnore && !s.replaceInto && s.onDup.empty())) throw;
        if (onConflict(t.lastDup, e.what())) break;
      }
    }
  }
  return affected;
}


static bool isParquetFile(const std::string& path) {
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) return false;
  char m[4];
  const bool pq = fread(m, 1, 4, f) == 4 && memcmp(m, "PAR1", 4) == 0;
  fclose(f);
  return pq;
}

// LOAD DATA INFILE of a Parquet file: flat columns are decoded to typed arrays (parquetScan) and
// appended a column at a time; nested ones go through text (parquetToTsv + the text loader). The
// WAL names a copy of the file (it is compressed already), which replays the same way.
// A load copy's name carries its CRC32C (checked when the WAL replays it) and "-partial" when the load
// failed after loading rows (its replay ends in the same error): <base>-c<crc>[-partial].parquet.
static std::string crcName(const std::string& copy, uint32_t crc, bool partial) {
  char b[24];
  snprintf(b, sizeof b, "-c%08x", crc);
  const size_t dot = copy.rfind('.');
  return copy.substr(0, dot) + b + (partial ? "-partial" : "") + copy.substr(dot);
}
ResultSet Engine::loadParquet(Stmt& s, const std::shared_ptr<Table>& t, const std::vector<int>& srcOf) {
  const auto t0 = Clock::now();
  if (replaying_) {  // a copy in the data dir: unchanged since it was written?
    static const std::regex named(R"(-c([0-9a-f]{8})(-partial)?\.parquet$)");
    std::smatch m;
    if (std::regex_search(s.path, m, named) && fileCrc(s.path) != (uint32_t)std::stoul(m[1].str(), nullptr, 16))
      throw SqlError("the load copy " + s.path + " is damaged (CRC mismatch)", 1105);
  }
  // Mapped, not read: the file stays in the page cache (no heap copy of a multi-GB file).
  struct Map {
    void* p = MAP_FAILED;
    size_t n = 0;
    ~Map() { if (p != MAP_FAILED && n) munmap(p, n); }
  } map;
  {
    const int fd = ::open(s.path.c_str(), O_RDONLY | O_CLOEXEC);
    struct stat st;
    if (fd < 0 || fstat(fd, &st) != 0) {
      if (fd >= 0) ::close(fd);
      throw SqlError("File '" + s.path + "' not found", 29);
    }
    map.n = (size_t)st.st_size;
    if (map.n) map.p = mmap(nullptr, map.n, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (map.n && map.p == MAP_FAILED) throw SqlError("cannot map '" + s.path + "'", 29);
    if (map.n) madvise(map.p, map.n, MADV_SEQUENTIAL);
  }
  const std::string_view data(map.n ? (const char*)map.p : "", map.n);
  std::vector<std::string> wanted;  // per table column: the Parquet column it comes from ("": none)
  for (size_t c = 0; c < t->cols.size(); ++c)
    wanted.push_back(srcOf.empty() ? t->cols[c].name : srcOf[c] >= 0 ? s.insertCols[srcOf[c]] : std::string());
  std::string copyTo;
  if (!replaying_ && !dir_.empty() && s.path.rfind(dir_ + "/", 0) != 0) {
    static std::atomic<uint64_t> seq{0};
    uint64_t gen;
    {
      std::lock_guard<std::mutex> g(walMu_);
      gen = walGen_;
    }
    mkdirs(dir_ + "/loads");
    copyTo = dir_ + "/loads/g" + std::to_string(gen) + "-" + std::to_string(nowUs()) + "-" + std::to_string(++seq) + ".parquet";
  }
  std::unique_lock<FairSharedMutex> lk(t->mu);
  waitTxnFree(*t, lk);
  bool vec = false;
  for (auto& cm : t->cols) vec = vec || cm.pt == PType::VEC;
  int64_t n = 0;
  const std::vector<Value> defs = defaultRow(*t);
  // Rows loaded before an error stay (as with text files): the statement is logged anyway, with a
  // copy of the file, so a restart replays to the same rows and the same error.
  // The WAL's copy is made (and checked) while the file loads; waited for before the statement is done.
  std::future<uint32_t> copying;
  if (!copyTo.empty()) copying = std::async(std::launch::async, [path = s.path, copyTo] { return copyFile(path, copyTo); });
  struct FailLog {
    const std::string& copyTo;
    std::future<uint32_t>& copying;
    const int64_t& n;
    bool armed = true;
    ~FailLog() {
      if (!armed || copyTo.empty()) return;
      try {  // (unwinding: never throw from here)
        const uint32_t crc = copying.get();
        if (n <= 0) {  // nothing loaded: no WAL record, no copy
          unlink(copyTo.c_str());
          return;
        }
        const std::string named = crcName(copyTo, crc, true);
        if (rename(copyTo.c_str(), named.c_str()) == 0) tLoadLogPath = named;
      } catch (...) {
      }
    }
  } failLog{copyTo, copying, n};
  std::vector<double> colMs(t->cols.size());
  double cbMs = 0, appMs = 0;
  const bool direct = !vec && parquetScan(data, wanted, [&](int64_t rows, std::vector<PqChunk>& in) {
    const auto cb0 = Clock::now();
    struct CbTime {
      double& acc;
      Clock::time_point t0;
      ~CbTime() { acc += msSince(t0); }
    } cbTime{cbMs, cb0};
    const size_t nc = t->cols.size();
    std::vector<std::vector<int64_t>> iv(nc);
    std::vector<std::vector<double>> dv(nc);
    std::vector<std::exception_ptr> errs(nc);
    // String columns first (the long tasks: dictionary coding), so they don't start last.
    std::vector<size_t> order(nc);
    for (size_t c = 0; c < nc; ++c) order[c] = c;
    std::stable_partition(order.begin(), order.end(), [&](size_t c) { return t->cols[c].lt == LType::VARCHAR && in[c].dict.empty(); });
    ioParallel(nc, [&](size_t oc) {  // one column per task (each its own dictionary)
      const size_t c = order[oc];
      struct ColTime {  // (WARPSTORE_TRACE: per-column conversion time)
        double& acc;
        Clock::time_point t0 = Clock::now();
        ~ColTime() { acc += msSince(t0); }
      } colTime{colMs[c]};
      try {
        const ColumnMeta& cm = t->cols[c];
        const PqChunk& p = in[c];
        const bool f64 = cm.pt == PType::F64;
        {  // No NULLs and the values as they are: the decoded array becomes the column (no copy).
          const bool exactI = p.kind == 1 && ((p.unit == 0 && (cm.lt == LType::INT || cm.lt == LType::BIGINT)) ||
                                              (p.unit == 1 && cm.lt == LType::DATE) || (p.unit == 2 && cm.lt == LType::DATETIME) ||
                                              (p.unit == 3 && cm.lt == LType::DECIMAL && p.scale == cm.scale));
          const bool exactD = p.kind == 2 && cm.lt == LType::DOUBLE;
          if ((exactI || exactD) && (int)c != t->autoIncCol && memchr(p.valid.data(), 0, (size_t)rows) == nullptr) {
            bool fits = true;
            if (exactI && cm.pt == PType::I32)
              for (int64_t r = 0; r < rows && fits; ++r) fits = p.i[r] > INT32_MIN && p.i[r] <= INT32_MAX;
            if (fits && exactD && (int64_t)p.d.size() >= rows) {
              in[c].d.resize((size_t)rows);
              dv[c].swap(in[c].d);
              return;
            }
            if (fits && exactI && (int64_t)p.i.size() >= rows) {
              in[c].i.resize((size_t)rows);
              iv[c].swap(in[c].i);
              return;
            }
          }
        }
        if (f64) dv[c].resize(rows);
        else iv[c].resize(rows);
        const int64_t nul = cm.pt == PType::I64 ? kNullI64 : kNullI32;
        auto null = [&](int64_t r) {
          if ((int)c == t->autoIncCol) {
            iv[c][r] = 0;  // (numbered below)
            return;
          }
          if (cm.notNull || (int)c == t->pk) throw SqlError("Column '" + cm.name + "' cannot be null (row " + std::to_string(n + r + 1) + ")", 1048);
          t->hasNull[c] = 1;
          if (f64) dv[c][r] = nullv();
          else iv[c][r] = nul;
        };
        if (p.kind == 0) {
          for (int64_t r = 0; r < rows; ++r) null(r);
          return;
        }
        const double pw = pow10i(p.scale);
        if (p.kind == 3 && cm.lt == LType::VARCHAR && !p.dict.empty() && p.dict.size() * 4 <= (size_t)rows) {
          // dictionary-encoded: its entries coded once, rows mapped by index (first-appearance order
          // of the entries, so codes follow the rows as the per-row path would)
          std::vector<uint8_t> used(p.dict.size(), 0);
          std::vector<std::string_view> order;
          for (int64_t r = 0; r < rows; ++r)
            if (p.valid[r] && !used[p.code[r]]) used[p.code[r]] = 1, order.push_back(p.dict[p.code[r]]);
          std::vector<int64_t> oc(order.size());
          t->dicts[c]->getOrAddMany(order.data(), order.size(), oc.data(), kNullI32);
          std::vector<int64_t> map(p.dict.size(), kNullI32);
          size_t o = 0;
          std::fill(used.begin(), used.end(), 0);
          for (int64_t r = 0; r < rows; ++r)
            if (p.valid[r] && !used[p.code[r]]) used[p.code[r]] = 1, map[p.code[r]] = oc[o++];
          for (int64_t r = 0; r < rows; ++r) {
            if (p.valid[r]) iv[c][r] = map[p.code[r]];
            else null(r);
          }
          return;
        }
        if (p.kind == 3 && cm.lt == LType::VARCHAR) {  // strings: coded in row order, one lock
          std::vector<std::string_view> sv(p.s);
          for (int64_t r = 0; r < rows; ++r)
            if (!p.valid[r]) sv[r] = std::string_view();
          t->dicts[c]->getOrAddMany(sv.data(), (size_t)rows, iv[c].data(), kNullI32);
          for (int64_t r = 0; r < rows; ++r)
            if (!p.valid[r]) null(r);
          return;
        }
        const bool intExact = p.kind == 1 && p.unit == 0 && (cm.lt == LType::INT || cm.lt == LType::BIGINT);
        const bool dblExact = p.kind == 2 && cm.lt == LType::DOUBLE;
        const bool dateExact = p.kind == 1 && p.unit == 1 && cm.lt == LType::DATE;
        const bool tsExact = p.kind == 1 && p.unit == 2 && cm.lt == LType::DATETIME;
        const bool decExact = p.kind == 1 && p.unit == 3 && cm.lt == LType::DECIMAL && p.scale == cm.scale;
        int64_t i0;
        double d0;
        for (int64_t r = 0; r < rows; ++r) {
          if (!p.valid[r]) {
            null(r);
            continue;
          }
          if (intExact || dateExact || tsExact || decExact) {
            iv[c][r] = p.i[r];
            continue;
          }
          if (dblExact) {
            dv[c][r] = p.d[r];
            continue;
          }
          Value v = p.kind == 2 ? Value::D(p.d[r])
                  : p.kind == 3 ? Value::S(std::string(p.s[r]))
                  : p.unit == 1 ? Value::Dt(p.i[r])
                  : p.unit == 2 ? Value::Ts(p.i[r])
                  : p.unit == 3 ? Value::Dec((double)p.i[r] / pw, p.scale)
                                : Value::I(p.i[r]);
          if (p.unit == 3 && cm.lt == LType::DECIMAL) {  // exact rescale (no detour through a double)
            const int64_t u = p.i[r];
            iv[c][r] = cm.scale >= p.scale ? u * (int64_t)pow10i(cm.scale - p.scale)
                                           : (int64_t)std::llround((double)u / pow10i(p.scale - cm.scale));
            continue;
          }
          if (cm.lt == LType::VARCHAR && v.kind != Value::STR) v = Value::S(v.toString());
          t->rawOf((int)c, v, i0, d0);
          if (f64) dv[c][r] = d0;
          else iv[c][r] = i0;
          if (f64 ? d0 != d0 : i0 == nul) null(r);  // (a value converted to NULL, e.g. an invalid date)
        }
      } catch (...) {
        errs[c] = std::current_exception();
      }
    }, true);
    for (auto& e : errs)
      if (e) std::rethrow_exception(e);
    if (t->autoIncCol >= 0) {
      for (int64_t r = 0; r < rows; ++r) {
        int64_t& v = iv[t->autoIncCol][r];
        if (v == 0) v = t->autoNext++;
        else t->autoNext = std::max<int64_t>(t->autoNext, v + 1);
      }
    }
    const auto ap0 = Clock::now();
    t->appendColumns(iv, dv, rows, nullptr, true);  // (null() set hasNull while converting)
    appMs += msSince(ap0);
    n += rows;
    checkKilled();
    lk.unlock();  // (readers get in between row groups)
    // Over --memory-limit: evict cold column data to the segment files now (the maintenance thread
    // can't get the table lock while a long load holds it).
    if (tierEnabled() && tierHostBytes() - tierPendingBytes() > tierMemLimit() / 10 * 9) tierBalance();
    std::this_thread::yield();
    lk.lock();
  });
  if (!direct) {  // nested columns etc.: as text, through the text loader
    std::vector<std::string> names;
    for (auto& w : wanted) names.push_back(w.empty() ? std::string("\x01") : w);
    const std::string tsv = parquetToTsv(data, names);
    const std::string staged = (dir_.empty() ? std::string("/tmp/warpstore-") : dir_ + "/loads/staged-") + std::to_string(nowUs()) + ".tsv";
    {
      FILE* f = fopen(staged.c_str(), "wb");
      if (!f) throw SqlError("cannot stage " + staged, 1105);
      fwrite(tsv.data(), 1, tsv.size(), f);
      fclose(f);
    }
    try {
      n = t->loadFile(staged, '\t', 0, {}, {}, '\\');
    } catch (...) {
      unlink(staged.c_str());
      throw;
    }
    unlink(staged.c_str());
  }
  failLog.armed = false;
  if (envSetting("TRACE")) {
    std::vector<size_t> ord(colMs.size());
    for (size_t c = 0; c < ord.size(); ++c) ord[c] = c;
    std::sort(ord.begin(), ord.end(), [&](size_t a, size_t b) { return colMs[a] > colMs[b]; });
    std::string top;
    double sum = 0;
    for (double x : colMs) sum += x;
    top += " [all columns " + std::to_string((int)sum) + " ms]";
    for (size_t k = 0; k < std::min<size_t>(20, ord.size()); ++k) {
      char b[96];
      snprintf(b, sizeof b, " %s %.0f", t->cols[ord[k]].name.c_str(), colMs[ord[k]]);
      top += b;
    }
    fprintf(stderr, "[trace] parquet load%s: %.0f ms (conversion callbacks %.0f, appends %.0f); slowest columns (ms):%s\n", direct ? "" : " (text path)",
            msSince(t0), cbMs, appMs, top.c_str());
  }
  lk.unlock();
  const auto cp0 = Clock::now();
  if (!copyTo.empty()) {
    const uint32_t crc = copying.get();
    const std::string named = crcName(copyTo, crc, false);
    if (rename(copyTo.c_str(), named.c_str()) != 0) throw SqlError("cannot rename " + copyTo, 1105);
    tLoadLogPath = named;
    loadedBytes_ += (int64_t)map.n;
  }
  if (envSetting("TRACE")) fprintf(stderr, "[trace] parquet load: WAL copy %.0f ms\n", msSince(cp0));
  warmDicts(t);
  char info[128];
  snprintf(info, sizeof info, "Records: %lld  Loaded in %.1f s", (long long)n, msSince(t0) / 1000);
  return okResult(n, info);
}

ResultSet Engine::doLoad(Stmt& s) {
  if (s.setValue != "FS" && !s.setValue.empty())
    throw SqlError("LOAD DATA " + s.setValue + " is not supported (use a local path / FS, S3, KAFKA or MYSQL)", 1235);
  auto t = cat_.get(s.table);
  std::unique_lock<FairSharedMutex> lk(t->mu);
  waitTxnFree(*t, lk);  // another session's open transaction changed this table
  auto t0 = Clock::now();
  std::vector<int> srcOf;
  if (!s.insertCols.empty()) {  // (col, @skip, ..): which field feeds which column
    srcOf.assign(t->cols.size(), -1);
    for (size_t i = 0; i < s.insertCols.size(); ++i) {
      if (s.insertCols[i].rfind("@", 0) == 0) continue;
      int c = t->colIndex(s.insertCols[i]);
      if (c < 0) throw SqlError("Unknown column '" + s.insertCols[i] + "' in 'field list'", 1054);
      srcOf[c] = (int)i;
    }
  }
  if (replaying_) replayedLoads_.insert(s.path);
  if (isParquetFile(s.path)) {  // (also a copy the WAL names, on replay)
    lk.unlock();
    return loadParquet(s, t, srcOf);
  }
  std::string path = s.path;
  if (replaying_ && access(path.c_str(), F_OK) != 0 && !gWalArchive.empty() && path.find("/loads/g") != std::string::npos) {
    const std::string alt = gWalArchive + path.substr(path.rfind("/loads/"));  // moved to the archive already
    if (access(alt.c_str(), F_OK) == 0) path = alt;
  }
  std::string copyTo;
  if (!replaying_ && !dir_.empty() && s.path.rfind(dir_ + "/", 0) != 0) {
    // the WAL must not depend on a file outside the data directory: load (and log) a copy
    static std::atomic<uint64_t> seq{0};
    uint64_t gen;
    {
      std::lock_guard<std::mutex> g(walMu_);
      gen = walGen_;
    }
    mkdirs(dir_ + "/loads");
    // (zstd-compressed while it is loaded: the file is read once)
    copyTo = dir_ + "/loads/g" + std::to_string(gen) + "-" + std::to_string(nowUs()) + "-" + std::to_string(++seq) + ".zst";
  }
  // The exclusive lock is released briefly every 256K rows so queries are not stalled by a long load
  // (they see the rows loaded so far, like other uncommitted writes).
  int64_t n = 0, loaded = 0;
  try {
    n = t->loadFile(path, s.delim, s.skipLines, srcOf, [&] {
      checkKilled();
      lk.unlock();
      if (tierEnabled() && tierHostBytes() - tierPendingBytes() > tierMemLimit() / 10 * 9) tierBalance();  // (as for Parquet)
      std::this_thread::yield();
      lk.lock();
    }, s.escape, copyTo, &loaded);
  } catch (...) {
    // Rows were loaded before the error: the statement is logged anyway (with the copy, it replays
    // to the same rows and the same error), or a restart would lose them.
    if (loaded > 0 && !copyTo.empty() && access(copyTo.c_str(), F_OK) == 0) {
      const std::string partial = copyTo.substr(0, copyTo.size() - 4) + "-partial.zst";  // (its replay fails the same way)
      tLoadLogPath = rename(copyTo.c_str(), partial.c_str()) == 0 ? partial : copyTo;
    }
    throw;
  }
  if (!copyTo.empty()) tLoadLogPath = copyTo;
  {
    struct stat st;
    if (stat(path.c_str(), &st) == 0) loadedBytes_ += (int64_t)st.st_size;
  }
  lk.unlock();
  warmDicts(t);
  char info[128];
  snprintf(info, sizeof info, "Records: %lld  Loaded in %.1f s", (long long)n, msSince(t0) / 1000);
  return okResult(n, info);
}

// Share of a string column's dictionary that live rows still use (dictionaries only grow: strings
// of deleted / updated rows stay). (table lock held)
static double dictUse(const Table& t, int c) {
  const Dict& d = *t.dicts[c];
  const size_t n = d.strs.size();
  if (n == 0) return 1;
  std::vector<uint8_t> used(n, 0);
  auto mark = [&](const uint8_t* raw, int64_t rows, const DelBits& del) {
    const int32_t* v = (const int32_t*)raw;
    for (int64_t r = 0; r < rows; ++r)
      if (!del.test(r) && v[r] >= 0 && (size_t)v[r] < n) used[v[r]] = 1;
  };
  std::vector<uint8_t> tmp;
  for (auto& sg : t.segs) mark(segRawColumn(*sg->d, c, tmp), sg->n, *sg->del);
  for (auto& b : t.bufs) mark(b->cols[c].data(), b->n, *b->del);
  size_t k = 0;
  for (uint8_t u : used) k += u;
  return (double)k / (double)n;
}

ResultSet Engine::doOptimize(Stmt& s) {
  auto t = cat_.get(s.table);
  std::unique_lock<FairSharedMutex> lk(t->mu);
  int64_t n = t->bufferRows();
  t->flush(true);
  for (auto& ix : t->indexes)
    if (ix.vec) t->buildIndex(ix);  // retrain vector indexes on the current data
  // Dictionaries mostly made of strings no row uses any more: the table is rebuilt (as MySQL's
  // OPTIMIZE TABLE does), which codes only the live strings.
  std::string garbage;
  if (t->txnPending.empty() && t->db != "__tmp")
    for (size_t c = 0; c < t->cols.size(); ++c)
      if (t->cols[c].lt == LType::VARCHAR && t->dicts[c] && t->dicts[c]->strs.size() >= 4096) {
        const double u = dictUse(*t, (int)c);
        if (u < 0.5) {
          char b[64];
          snprintf(b, sizeof b, "%s%s %.0f%% used", garbage.empty() ? "" : ", ", t->cols[c].name.c_str(), u * 100);
          garbage += b;
        }
      }
  const int64_t autoNext = t->autoNext;
  lk.unlock();
  std::string msg = "flushed " + std::to_string(n) + " buffered rows to GPU segments";
  if (!garbage.empty() && !dir_.empty()) {
    // Each step is a logged statement (a replay rebuilds the same way); writes wait meanwhile.
    std::unique_lock<FairSharedMutex> gate(writeGate_);
    const std::string q = "`" + t->db + "`.`", base = q + t->name, tmp = q + t->name + "__ws_rebuild`", old = q + t->name + "__ws_old`";
    const auto t0 = Clock::now();
    execute("DROP TABLE IF EXISTS " + tmp);  // (left over by a crash during an earlier rebuild)
    execute("DROP TABLE IF EXISTS " + old);
    execute("CREATE TABLE " + tmp + " LIKE " + base + "`");
    execute("INSERT INTO " + tmp + " SELECT * FROM " + base + "`");
    if (t->autoIncCol >= 0) execute("ALTER TABLE " + tmp + " AUTO_INCREMENT = " + std::to_string(autoNext));
    execute("RENAME TABLE " + base + "` TO " + old + ", " + tmp + " TO " + base + "`");
    execute("DROP TABLE " + old);
    t = cat_.get(s.table);
    msg += "; rebuilt the table in " + fmtMs(msSince(t0)) + " ms to drop unused dictionary strings (" + garbage + ")";
  }
  warmDicts(t);
  ResultSet r;
  r.isResult = true;
  r.names = {"Table", "Op", "Msg_type", "Msg_text"};
  r.types = {LType::VARCHAR, LType::VARCHAR, LType::VARCHAR, LType::VARCHAR};
  r.rows = {{t->name, "optimize", "status", msg}};
  return r;
}

ResultSet Engine::doShowTables(Stmt& s) {
  const std::string db = s.table.empty() ? currentDb() : s.table;
  ResultSet r;
  r.isResult = true;
  r.names = {"Tables_in_" + db};
  r.types = {LType::VARCHAR};
  if (s.setName == "FULL") {
    r.names.push_back("Table_type");
    r.types.push_back(LType::VARCHAR);
  }
  std::lock_guard<std::mutex> g(cat_.mu);
  std::vector<std::pair<std::string, std::string>> out;
  for (auto& [k, t] : cat_.tables)
    if (lower(t->db) == lower(db) && likeCI(t->name, s.path)) out.push_back({t->name, "BASE TABLE"});
  for (auto& [k, v] : cat_.views)
    if (k.substr(0, k.find('.')) == lower(db) && likeCI(k.substr(k.find('.') + 1), s.path)) out.push_back({k.substr(k.find('.') + 1), "VIEW"});
  std::sort(out.begin(), out.end());
  for (auto& [n, ty] : out) r.rows.push_back(s.setName == "FULL" ? std::vector<std::string>{n, ty} : std::vector<std::string>{n});
  return r;
}

ResultSet Engine::doDescribe(Stmt& s) {
  auto t = cat_.get(s.table);
  ResultSet r;
  r.isResult = true;
  r.names = {"Field", "Type", "Null", "Key", "Default", "Extra"};
  r.types.assign(6, LType::VARCHAR);
  for (size_t i = 0; i < t->cols.size(); ++i) {
    const ColumnMeta& c = t->cols[i];
    std::string key = (int)i == t->pk || std::count(t->pkCols.begin(), t->pkCols.end(), (int)i) ? "PRI" : "";
    for (auto& ix : t->indexes)
      if (key.empty() && !ix.cols.empty() && ix.cols[0] == (int)i) key = ix.unique ? "UNI" : "MUL";
    r.rows.push_back({c.name, colTypeText(c), c.notNull || (int)i == t->pk ? "NO" : "YES", key,
                      c.hasDefault ? (c.def.isNull() ? kNullCell : c.def.toString()) : kNullCell, c.autoInc ? "auto_increment" : ""});
  }
  return r;
}

ResultSet Engine::doShowSegments(Stmt& s) {
  auto t = cat_.get(s.table);
  std::shared_lock<FairSharedMutex> lk(t->mu);
  ResultSet r;
  r.isResult = true;
  r.names = {"partition", "segment", "tier", "rows", "deleted", "vram_bytes", "sort_key_min", "sort_key_max"};
  r.types = {LType::BIGINT, LType::BIGINT, LType::VARCHAR, LType::BIGINT, LType::BIGINT, LType::BIGINT, LType::VARCHAR, LType::VARCHAR};
  auto fmtKey = [&](double v) {
    if (t->sortKey < 0) return kNullCell;
    return typedVal(v, t->cols[t->sortKey].lt, t->dicts[t->sortKey].get()).toString();
  };
  int64_t rowBytes = 0;
  for (size_t c = 0; c < t->cols.size(); ++c) rowBytes += t->colBytes((int)c);
  for (int p = 0; p < t->nparts; ++p) {
    for (size_t i = 0; i < t->segs.size(); ++i) {
      auto& sg = *t->segs[i];
      if (sg.part != p) continue;
      bool host = std::any_of(sg.mapped.begin(), sg.mapped.end(), [](uint8_t m) { return m != 0; });
      bool disk = std::any_of(sg.d->cold.begin(), sg.d->cold.end(), [](uint8_t m) { return m != 0; });
      r.rows.push_back({std::to_string(p), std::to_string(i), disk ? "disk-columnstore" : host ? "host-columnstore (PCIe)" : "gpu-columnstore",
                        std::to_string(sg.n), std::to_string(sg.ndel()),
                        std::to_string(sg.n * rowBytes + (sg.n + 31) / 32 * 4),
                        t->sortKey >= 0 ? fmtKey(sg.mn[t->sortKey]) : kNullCell,
                        t->sortKey >= 0 ? fmtKey(sg.mx[t->sortKey]) : kNullCell});
    }
    const WriteBuffer& b = *t->bufs[p];
    if (b.n > 0)
      r.rows.push_back({std::to_string(p), "-1", "cpu-write-buffer", std::to_string(b.n), std::to_string(b.ndel()), "0", kNullCell, kNullCell});
  }
  return r;
}

// ------------------------------------------------------------------ DELETE / UPDATE

// UPDATE / DELETE with joins (UPDATE a JOIN b .. SET, DELETE a FROM a JOIN b ..), correlated
// subqueries, ORDER BY / LIMIT: the rows to change (and their new values) come from a SELECT over
// the same FROM / WHERE, then the target rows with exactly those column values are modified (rows
// identical in every column are interchangeable, so matching by value is exact).
ResultSet Engine::modifyViaSelect(Stmt& s) {
  auto t = cat_.get(s.table);
  Stmt sel;
  sel.type = Stmt::SELECT;
  sel.from = s.from;
  if (sel.from.empty()) sel.from = {TableRef{s.table, plainName(s.table), nullptr, 0, nullptr}};
  sel.joinConds = s.joinConds;
  sel.where = s.where;
  sel.orderBy = s.orderBy;
  sel.limit = s.limit;
  const std::string alias = s.targetAlias.empty() ? sel.from[0].alias : s.targetAlias;
  for (auto& c : t->cols) sel.items.push_back({mkCol(alias, c.name), "__c" + std::to_string(sel.items.size())});
  if (s.limit >= 0 && sel.orderBy.empty())  // LIMIT without ORDER BY: rows chosen by value, so a WAL replay picks the same ones
    for (auto& c : t->cols)
      if (c.pt != PType::VEC) sel.orderBy.push_back(OrderItem{mkCol(alias, c.name), false});
  std::vector<int> setCols;
  for (auto& [name, e] : s.sets) {
    const int c = t->colIndex(name);
    if (c < 0) throw SqlError("Unknown column '" + name + "' in 'field list'", 1054);
    setCols.push_back(c);
    sel.items.push_back({cloneExpr(e), "__s" + std::to_string(sel.items.size())});
  }
  QueryCtx q;
  ResultSet r;
  try {
    r = query(sel, q, false, true);
  } catch (...) {
    dropTemps(q);
    throw;
  }
  dropTemps(q);
  const size_t nc = t->cols.size();
  auto keyOf = [&](const std::vector<Value>& row) {
    std::string k;
    for (size_t c = 0; c < nc; ++c) {
      const Value& v = row[c];
      if (v.isNull()) { k += "\x02\x1f"; continue; }
      if (v.kind == Value::DBL) {
        char b[40];
        snprintf(b, sizeof b, "%.15g", v.d);
        k += b;
      } else if (v.kind == Value::STR) {
        k += v.s;
      } else {
        k += std::to_string(v.i);
      }
      k += '\x1f';
    }
    return k;
  };
  std::unordered_map<std::string, size_t> want;  // row values -> result row (its SET values)
  for (size_t i = 0; i < r.vals.size(); ++i) want.emplace(keyOf(r.vals[i]), i);
  std::unique_lock<FairSharedMutex> lk(t->mu);
  std::vector<Loc> matched;
  std::vector<std::vector<Value>> newRows;
  std::unordered_map<std::string, int64_t> budget;  // LIMIT: how many rows with these values may change
  if (s.limit >= 0)
    for (auto& row : r.vals) ++budget[keyOf(row)];
  forEachLiveRow(*t, [&](Loc l) {
    std::vector<Value> cur;
    for (size_t c = 0; c < nc; ++c) cur.push_back(cellValue(*t, l, (int)c));
    const std::string k = keyOf(cur);
    auto it = want.find(k);
    if (it == want.end()) return;
    if (s.limit >= 0 && budget[k]-- <= 0) return;
    matched.push_back(l);
    if (s.type == Stmt::UPDATE) {
      std::vector<Value> row = cur;
      for (size_t j = 0; j < setCols.size(); ++j) {
        Value v = r.vals[it->second][nc + j];
        if (!v.isNull() && t->cols[setCols[j]].lt == LType::VARCHAR && v.kind != Value::STR) v = Value::S(v.toString());
        row[setCols[j]] = v;
      }
      newRows.push_back(std::move(row));
    }
  });
  if (rowsPending(*t, matched)) {  // another open transaction changed some of these rows: wait, then redo
    waitTxnFree(*t, lk, &matched);
    lk.unlock();
    return modifyViaSelect(s);
  }
  return applyModify(*t, matched, newRows);
}

static void markDeleted(Table* t, const std::vector<Loc>& matched);

// UPDATE outside a transaction, without VECTOR columns: each new row starts as the old row's stored
// cells (strings stay codes); Values are made only for the columns the SET expressions read, and
// only the SET columns are converted back. Matched locations stay valid while the new rows are
// appended (full write buffers are flushed at the end).
static ResultSet updateRaw(Table& t, const std::vector<Loc>& matched, const std::vector<std::pair<int, ExprP>>& sets,
                           const std::set<int>& reads) {
  const bool tr = envSetting("TRACE") != nullptr;
  auto ph = Clock::now();
  double tg = 0, ts = 0, ta = 0;
  markDeleted(&t, matched);
  if (tr) fprintf(stderr, "[trace] update: markDeleted %.1f ms\n", msSince(ph));
  t.deferFlush = true;
  struct Flush {
    Table& t;
    ~Flush() {
      t.deferFlush = false;
      try {
        for (int p = 0; p < t.nparts; ++p)
          if (t.bufs[p]->n >= kSegRows) t.flushPart(p);
      } catch (std::exception& e) {
        fprintf(stderr, "[warpstore] flush after UPDATE: %s\n", e.what());
      }
    }
  } flush{t};
  const size_t nc = t.cols.size();
  std::vector<Value> cur(nc), vals(sets.size());
  PostCtx pc;
  pc.row = &cur;
  pc.colSlot = [](int bt, int bc) { return bt == 0 ? bc : -1; };
  // In chunks: the matched rows' cells, a column per task; the SET values, row by row; then the new
  // rows appended a column at a time.
  const size_t kChunk = 262144;
  std::vector<std::vector<int64_t>> iv(nc);
  std::vector<std::vector<double>> dv(nc);
  for (size_t from = 0; from < matched.size(); from += kChunk) {
    const size_t n = std::min(kChunk, matched.size() - from);
    const Loc* ls = matched.data() + from;
    auto gather = [&](size_t c) {
      const PType pt = t.cols[c].pt;
      if (pt == PType::F64) dv[c].resize(n);
      else iv[c].resize(n);
      const uint8_t* base = nullptr;
      int32_t baseSeg = INT32_MIN;
      for (size_t i = 0; i < n; ++i) {
        const Loc l = ls[i];
        if (l.seg != baseSeg) base = t.colData(l, (int)c).data(), baseSeg = l.seg;
        switch (pt) {
          case PType::I32: iv[c][i] = ((const int32_t*)base)[l.row]; break;
          case PType::I64: iv[c][i] = ((const int64_t*)base)[l.row]; break;
          default: dv[c][i] = ((const double*)base)[l.row]; break;
        }
      }
    };
    ph = Clock::now();
    // Tables without keys / indexes / open transactions: the new rows are copied from the old ones
    // straight into the write buffers (appendCopy), so only the columns the SETs read are gathered.
    bool shardSet = false;
    for (auto& [c, e] : sets)
      for (int k : t.shardKey) shardSet = shardSet || c == k;
    const bool copyRows = t.pk < 0 && t.indexes.empty() && !txnKey() && !shardSet;
    std::vector<uint8_t> need(nc, copyRows ? 0 : 1);
    if (copyRows) {
      std::set<std::pair<int, int>> used;
      for (auto& [c, e] : sets) colsOf(e, used);
      for (auto& [bt, bc] : used)
        if (bt == 0 && bc >= 0 && bc < (int)nc) need[bc] = 1;
      for (auto& [c, e] : sets) {  // (the SET columns' value arrays)
        if (t.cols[c].pt == PType::F64) dv[c].resize(n);
        else iv[c].resize(n);
      }
    }
    std::vector<size_t> gcols;
    for (size_t c = 0; c < nc; ++c)
      if (need[c]) gcols.push_back(c);
    if (n < 4096)  // (a few rows, e.g. by primary key: not worth a task per column)
      for (size_t c : gcols) gather(c);
    else
      ioParallel(gcols.size(), [&](size_t k) { gather(gcols[k]); }, true);
    tg += msSince(ph), ph = Clock::now();
    // The SET values, row by row: on all cores unless a string column is set (new strings are coded in
    // row order, as a replay of the statement codes them).
    // MySQL evaluates single-table SETs left to right: a SET that reads a column an earlier one assigns
    // sees the new value (then everything goes row by row, in order).
    bool chained = false;
    {
      std::set<int> assigned;
      for (auto& [c, e] : sets) {
        std::set<std::pair<int, int>> used;
        colsOf(e, used);
        for (auto& [bt, bc] : used) chained = chained || (bt == 0 && assigned.count(bc));
        assigned.insert(c);
      }
    }
    std::vector<std::pair<int, ExprP>> rs = sets;  // the SETs evaluated row by row
    auto setRows = [&](size_t b, size_t e, std::vector<Value>& cur, std::vector<Value>& vals, PostCtx& pc) {
      int64_t i0;
      double d0;
      for (size_t i = b; i < e; ++i) {
        for (int c : reads) cur[c] = cellValue(t, ls[i], c);
        for (size_t j = 0; j < rs.size(); ++j) {  // (left to right: a later SET reads the new values)
          vals[j] = evalPost(rs[j].second, pc);
          if (chained) cur[rs[j].first] = vals[j];
        }
        for (size_t j = 0; j < rs.size(); ++j) {
          const int c = rs[j].first;
          Value& v = vals[j];
          if (!v.isNull() && t.cols[c].lt == LType::VARCHAR && v.kind != Value::STR) v = Value::S(v.toString());
          t.rawOf(c, v, i0, d0);
          if (t.cols[c].pt == PType::F64) dv[c][i] = d0;
          else iv[c][i] = i0;
        }
      }
    };
    // Vectorized SET forms on the raw values (computed against the old row first): a constant,
    // another column, column +/- constant. Others (or an overflow) go row by row.
    std::vector<int> fastC;
    std::vector<std::vector<int64_t>> fi;
    std::vector<std::vector<double>> fd;
    rs.clear();
    for (auto& [c, e] : sets) {
      if (chained) {
        rs.push_back({c, e});
        continue;
      }
      const ColumnMeta& cm = t.cols[c];
      std::vector<int64_t> oi;
      std::vector<double> od;
      bool ok = false;
      const bool f64 = cm.pt == PType::F64;
      auto colOf = [&](const ExprP& x) {
        return x->kind == Expr::COL && x->bt == 0 && x->bc >= 0 && t.cols[x->bc].lt == cm.lt && t.cols[x->bc].pt == cm.pt &&
                       t.cols[x->bc].scale == cm.scale && cm.lt != LType::VARCHAR ? x->bc : -1;
      };
      if (e->kind == Expr::LIT && !e->lit.isNull() && (cm.lt != LType::VARCHAR || e->lit.kind == Value::STR)) {
        int64_t i0 = 0;  // (a string constant: coded once)
        double d0 = 0;
        t.rawOf(c, e->lit, i0, d0);
        if (f64) od.assign(n, d0);
        else oi.assign(n, i0);
        ok = true;
      } else if (const int s2 = colOf(e); s2 >= 0) {
        if (f64) od = dv[s2];
        else oi = iv[s2];
        ok = true;
      } else if (e->kind == Expr::BIN && (e->op == "+" || e->op == "-") && colOf(e->args[0]) >= 0 && e->args[1]->kind == Expr::LIT &&
                 !e->args[1]->lit.isNull() && e->args[1]->lit.kind != Value::STR &&
                 (cm.lt == LType::INT || cm.lt == LType::BIGINT || cm.lt == LType::DECIMAL || cm.lt == LType::DOUBLE)) {
        const int s2 = colOf(e->args[0]);
        const double lv = e->args[1]->lit.num() * (e->op == "-" ? -1 : 1);
        if (f64) {
          od = dv[s2];
          for (auto& x : od)
            if (x == x) x += lv;
          ok = true;
        } else {
          const double scaled = cm.lt == LType::DECIMAL ? lv * pow10i(cm.scale) : lv;
          const int64_t add = (int64_t)std::llround(scaled);
          const int64_t nul = cm.pt == PType::I64 ? kNullI64 : kNullI32;
          const int64_t lo = cm.pt == PType::I64 ? INT64_MIN + 1 : INT32_MIN + 1, hi = cm.pt == PType::I64 ? INT64_MAX : INT32_MAX;
          if (std::fabs(scaled - (double)add) < 1e-9 && std::fabs(scaled) < 9e15) {
            oi = iv[s2];
            ok = true;
            for (auto& x : oi) {
              if (x == nul) continue;
              int64_t y;
              if (__builtin_add_overflow(x, add, &y) || y < lo || y > hi) { ok = false; break; }
              x = y;
            }
          }
        }
      }
      if (ok && (cm.notNull || c == t.pk)) {  // a NULL into NOT NULL: the row path reports it
        const int64_t nul = cm.pt == PType::I64 ? kNullI64 : kNullI32;
        for (int64_t x : oi) ok = ok && x != nul;
        for (double x : od) ok = ok && x == x;
      }
      if (ok) {
        fastC.push_back(c);
        fi.push_back(std::move(oi));
        fd.push_back(std::move(od));
      } else {
        rs.push_back({c, e});
      }
    }
    vals.resize(rs.size());
    bool strSet = false;
    for (auto& [c, e] : rs) strSet = strSet || t.cols[c].lt == LType::VARCHAR;
    if (rs.empty()) {
    } else if (n < 16384) {
      setRows(0, n, cur, vals, pc);
    } else if (strSet) {
      // Values on all cores; strings then coded in one batch in row order (first appearance), as the
      // row-by-row path and a replay code them.
      const size_t R = rs.size();
      std::vector<Value> vb(n * R);
      parallelFor(n, [&](size_t b, size_t e) {
        std::vector<Value> lcur(nc);
        PostCtx lpc = pc;
        lpc.row = &lcur;
        for (size_t i = b; i < e; ++i) {
          for (int c : reads) lcur[c] = cellValue(t, ls[i], c);
          for (size_t j = 0; j < R; ++j) {
            Value v = evalPost(rs[j].second, lpc);
            if (!v.isNull() && t.cols[rs[j].first].lt == LType::VARCHAR && v.kind != Value::STR) v = Value::S(v.toString());
            if (chained) lcur[rs[j].first] = v;
            vb[i * R + j] = std::move(v);
          }
        }
      }, 4096);
      for (size_t j = 0; j < R; ++j) {
        const int c = rs[j].first;
        if (t.cols[c].lt == LType::VARCHAR) {
          std::vector<std::string_view> sv(n);
          bool anyNull = false;
          for (size_t i = 0; i < n; ++i) {
            const Value& v = vb[i * R + j];
            if (v.isNull()) anyNull = true;
            else sv[i] = std::string_view(v.s.data() ? v.s.data() : "", v.s.size());
          }
          int64_t nullCode = kNullI32;
          if (anyNull) {  // (the NOT NULL check and the NULL code, as row by row)
            double d0;
            t.rawOf(c, Value::Null(), nullCode, d0);
          }
          t.dicts[c]->getOrAddMany(sv.data(), n, iv[c].data(), nullCode);
        } else {
          parallelFor(n, [&](size_t b, size_t e) {
            int64_t i0;
            double d0;
            for (size_t i = b; i < e; ++i) {
              t.rawOf(c, vb[i * R + j], i0, d0);
              if (t.cols[c].pt == PType::F64) dv[c][i] = d0;
              else iv[c][i] = i0;
            }
          }, 4096);
        }
      }
    } else {
      parallelFor(n, [&](size_t b, size_t e) {
        std::vector<Value> lcur(nc), lvals(sets.size());
        PostCtx lpc = pc;
        lpc.row = &lcur;
        setRows(b, e, lcur, lvals, lpc);
      }, 4096);
    }
    for (size_t j = 0; j < fastC.size(); ++j) {
      if (t.cols[fastC[j]].pt == PType::F64) dv[fastC[j]].swap(fd[j]);
      else iv[fastC[j]].swap(fi[j]);
    }
    // Rows stay in their partition when no shard-key column is set (no re-hashing).
    bool keepPart = true;
    for (auto& [c, e] : sets)
      for (int k : t.shardKey) keepPart = keepPart && c != k;
    std::vector<int32_t> partOf;
    if (keepPart && t.nparts > 1 && !t.shardKey.empty()) {
      partOf.resize(n);
      for (size_t i = 0; i < n; ++i) partOf[i] = ls[i].seg >= 0 ? t.segs[ls[i].seg]->part : -1 - ls[i].seg;
    }
    ts += msSince(ph), ph = Clock::now();
    if (copyRows) {
      std::vector<int32_t> part(n);
      for (size_t i = 0; i < n; ++i) part[i] = ls[i].seg >= 0 ? t.segs[ls[i].seg]->part : -1 - ls[i].seg;
      std::vector<uint8_t> given(nc, 0);
      for (auto& [c, e] : sets) given[c] = 1;
      t.appendCopy(ls, n, part.data(), given, iv, dv);
    } else {
      t.appendColumns(iv, dv, (int64_t)n, partOf.empty() ? nullptr : partOf.data());
    }
    ta += msSince(ph);
  }
  if (tr) fprintf(stderr, "[trace] update: gather %.1f ms, set %.1f ms, append %.1f ms\n", tg, ts, ta);
  return okResult((int64_t)matched.size(), "Rows matched: " + std::to_string(matched.size()));
}

ResultSet Engine::doModify(Stmt& s) {
  std::function<bool(const ExprP&)> hasSub = [&](const ExprP& e) {
    if (!e) return false;
    if (e->sub || e->kind == Expr::SUBQUERY) return true;
    for (auto& a : e->args)
      if (hasSub(a)) return true;
    return false;
  };
  bool viaSelect = s.from.size() > 1 || !s.joinConds.empty() || s.limit >= 0 || hasSub(s.where) ||
                   (s.from.size() == 1 && (s.from[0].sub || strcasecmp(s.from[0].alias.c_str(), plainName(s.from[0].name).c_str()) != 0));
  for (auto& [n, e] : s.sets) viaSelect = viaSelect || hasSub(e);
  if (viaSelect) return modifyViaSelect(s);
  // Subqueries in WHERE / SET run first (they may read this table).
  QueryCtx q;
  {
    Stmt tmp;
    tmp.where = s.where;
    for (auto& [n, e] : s.sets) tmp.items.push_back({e, n});
    try {
      resolveSubqueries(tmp, q);
    } catch (...) {
      dropTemps(q);
      throw;
    }
    s.where = tmp.where;
    for (size_t i = 0; i < s.sets.size(); ++i) s.sets[i].second = tmp.items[i].e;
    dropTemps(q);
  }
  auto t = cat_.get(s.table);
  std::unique_lock<FairSharedMutex> lk(t->mu);
  BindCtx ctx;
  ctx.tabs = {t};
  ctx.aliases = {t->name};
  if (s.where) bind(s.where, ctx);
  std::vector<std::pair<int, ExprP>> sets;
  ctx.rowWise = true;
  for (auto& [name, e] : s.sets) {
    int c = t->colIndex(name);
    if (c < 0) throw SqlError("Unknown column '" + name + "' in 'field list'", 1054);
    bind(e, ctx);
    sets.push_back({c, e});
  }
  std::vector<ExprP> conj;
  flattenAnd(s.where, conj);
  Compiler comp(&ctx);
  Prog wp;
  if (s.where) wp = comp.compile(s.where);

  ExprEnv henv = comp.hostEnv();
  auto hostCols = [&](int seg) {
    for (size_t k = 0; k < comp.slotCols.size(); ++k) {
      int c = comp.slotCols[k].second;
      henv.cols[k] = comp.env.cols[k];
      henv.cols[k].p = t->colData(Loc{(int32_t)seg, 0}, c).data();
      henv.cols[k].enc = 0;  // host mirror is raw
    }
  };

  const auto modT0 = Clock::now();
  std::vector<Loc> matched;
  int64_t key;
  bool ixFound = false;
  const IndexMeta* ixm = nullptr;
  if (findPkEq(conj, *t, 0, key)) {
    auto it = t->pkIndex.find(key);
    if (it != t->pkIndex.end()) {
      hostCols(it->second.seg);
      if (!s.where || truthy(evalProg(henv, wp, it->second.row))) matched.push_back(it->second);
    }
  } else if ((ixm = findIndexEq(conj, *t, 0, key, ixFound))) {
    auto it = ixFound ? ixm->map.find(key) : ixm->map.end();
    if (it != ixm->map.end())
      for (const Loc& l : it->second) {
        hostCols(l.seg);
        if (!s.where || truthy(evalProg(henv, wp, l.row))) matched.push_back(l);
      }
  } else {
    comp.uploadSets();
    std::vector<int> parts = partsToScan(*t, 0, conj);
    std::vector<bool> inParts(t->nparts, false);
    for (int p : parts) inParts[p] = true;
    for (size_t si = 0; si < t->segs.size(); ++si) {
      Segment& sg = *t->segs[si];
      if (!inParts[sg.part] || !segMayMatch(sg, conj)) continue;
      const uint32_t* sdel = sg.delDev();
      const BatchRef br{sg.part, sg.n, &sg.dev, sdel, &sg.enc, 0, &sg};
      struct Temps { ~Temps() { releaseScanTemps(); } } temps;
      for (size_t k = 0; k < comp.slotCols.size(); ++k) bindView(comp.env.cols[k], br, comp.slotCols[k].second);
      scanFlush();
      DevVec<int32_t> idx(sg.n);
      int64_t m = gpuSelect(comp.env, (bool)s.where, wp, sg.n, sdel, idx.p);
      for (int32_t r : idx.download(m)) matched.push_back(Loc{(int32_t)si, r});
    }
    for (int p : parts) {  // write-buffer rows: on the host, on all cores (chunks kept in row order)
      const WriteBuffer& b = *t->bufs[p];
      hostCols(bufLoc(p, 0).seg);
      const size_t nch = std::max<size_t>(1, std::min<size_t>(64, (size_t)b.n / 32768));
      std::vector<std::vector<Loc>> part(nch);
      parallelFor(nch, [&](size_t cb, size_t ce) {
        for (size_t ch = cb; ch < ce; ++ch)
          for (int64_t r = (int64_t)(b.n * ch / nch); r < (int64_t)(b.n * (ch + 1) / nch); ++r) {
            if (b.del->test(r)) continue;
            if (s.where && !truthy(evalProg(henv, wp, r))) continue;  // (NULL: not a match)
            part[ch].push_back(bufLoc(p, r));
          }
      }, 1);
      for (auto& v : part) matched.insert(matched.end(), v.begin(), v.end());
    }
  }

  if (rowsPending(*t, matched)) {  // row-level conflict with another open transaction: wait, then redo
    waitTxnFree(*t, lk, &matched);
    lk.unlock();
    return doModify(s);
  }
  // UPDATE = delete + re-insert of the modified row (like SingleStore's columnstore updates).
  if (s.type == Stmt::UPDATE && !undoLog()) {
    bool raw = true;
    for (auto& cm : t->cols) raw = raw && cm.pt != PType::VEC;
    std::set<int> reads;  // columns the SET expressions read
    std::function<bool(const ExprP&)> plain = [&](const ExprP& x) {
      if (!x) return true;
      if (x->kind == Expr::FUNC && x->op == "EMBED") return false;
      if (x->kind == Expr::COL) {
        if (x->bt != 0 || x->bc < 0) return false;
        reads.insert(x->bc);
      }
      for (auto& a : x->args)
        if (!plain(a)) return false;
      return true;
    };
    for (auto& [c, e] : sets) raw = raw && c != t->autoIncCol && plain(e);
    if (raw) {
      const auto u0 = Clock::now();
      ResultSet r = updateRaw(*t, matched, sets, reads);
      if (envSetting("TRACE")) fprintf(stderr, "[trace] update: match %.1f ms, rewrite %.1f ms (%zu rows)\n", std::chrono::duration<double, std::milli>(u0 - modT0).count(), msSince(u0), matched.size());
      return r;
    }
  }
  std::vector<std::vector<Value>> newRows;
  if (s.type == Stmt::UPDATE) {
    // EMBED(..) in SET: the texts of all matched rows go through the model in one batch first (the
    // per-row evaluation below then finds them in the cache).
    std::vector<ExprP> embeds;
    std::function<void(const ExprP&)> findEmbed = [&](const ExprP& x) {
      if (!x) return;
      if (x->kind == Expr::FUNC && x->op == "EMBED" && x->args.size() == 1) embeds.push_back(x->args[0]);
      for (auto& a : x->args) findEmbed(a);
    };
    for (auto& [c, e] : sets) findEmbed(e);
    if (!embeds.empty() && !matched.empty()) {
      std::vector<std::string> texts;
      for (const Loc& l : matched) {
        std::vector<Value> cur;
        for (size_t c = 0; c < t->cols.size(); ++c) cur.push_back(cellValue(*t, l, (int)c));
        PostCtx pc;
        pc.row = &cur;
        pc.colSlot = [](int bt, int bc) { return bt == 0 ? bc : -1; };
        for (auto& a : embeds) {
          Value v = evalPost(a, pc);
          if (!v.isNull()) texts.push_back(v.toString());
        }
      }
      std::vector<std::string> ignored;
      embedCached(texts, ignored);
    }
    for (const Loc& l : matched) {  // SET expressions on the host, left to right (MySQL: a later one sees the new values)
      std::vector<Value> cur;
      for (size_t c = 0; c < t->cols.size(); ++c) cur.push_back(cellValue(*t, l, (int)c));
      PostCtx pc;
      pc.row = &cur;
      pc.colSlot = [](int bt, int bc) { return bt == 0 ? bc : -1; };
      std::vector<Value> row = cur;
      for (auto& [c, e] : sets) {
        Value v = evalPost(e, pc);
        if (!v.isNull() && t->cols[c].lt == LType::VARCHAR && v.kind != Value::STR) v = Value::S(v.toString());
        row[c] = v;
        cur[c] = v;
      }
      newRows.push_back(std::move(row));
    }
  }
  return applyModify(*t, matched, newRows);
}

// Deletes the matched rows (and inserts their new versions, for UPDATE). Holds the table lock.
ResultSet Engine::applyModify(Table& tt, const std::vector<Loc>& matched, const std::vector<std::vector<Value>>& newRows) {
  markDeleted(&tt, matched);
  for (auto& row : newRows) tt.appendRow(row);
  return okResult((int64_t)matched.size(), "Rows matched: " + std::to_string(matched.size()));
}

// Marks the matched rows deleted (undo records, PK and index entries, deleted-row bitmaps).
static void markDeleted(Table* t, const std::vector<Loc>& matched) {
  std::set<int> touched, touchedBufs;
  for (const Loc& l : matched) {
    if (undoLog()) {
      undoLog()->push_back({t, false, t->rowValues(l)});
      t->txnNoteDelete(l);
    }
    if (t->pk >= 0) t->pkIndex.erase(t->pkValue(l));
    if (!t->indexes.empty()) t->indexRemove(l);
    DelBits& d = delForWrite(l.seg < 0 ? t->bufs[bufPart(l)]->del : t->segs[l.seg]->del);  // (a private copy if a snapshot holds it)
    uint32_t& w = d.host[l.row >> 5];
    if (!(w & (1u << (l.row & 31)))) w |= 1u << (l.row & 31), ++d.ndel;
    if (l.seg < 0) ++t->bufs[bufPart(l)]->version, touchedBufs.insert(bufPart(l));
    else touched.insert(l.seg);
  }
  for (int si : touched) {  // bits changed: the device copy is re-uploaded on the next scan
    std::lock_guard<std::mutex> g(t->segs[si]->del->mu);
    ++t->segs[si]->del->ver;
  }
  for (int p : touchedBufs) {
    std::lock_guard<std::mutex> g(t->bufs[p]->del->mu);
    ++t->bufs[p]->del->ver;
  }
}

// ------------------------------------------------------------------ SELECT

// Every table a statement reads (FROM entries, derived tables, CTE bodies, subqueries, UNION arms,
// views), as catalog keys.
static void collectTables(const Stmt& s, const std::string& db, std::set<std::string>& keys, std::set<std::string> ctes, Catalog& cat, int depth);
static void collectExpr(const ExprP& e, const std::string& db, std::set<std::string>& keys, const std::set<std::string>& ctes, Catalog& cat, int depth) {
  if (!e) return;
  if (e->sub) collectTables(*e->sub, db, keys, ctes, cat, depth);
  for (auto& a : e->args) collectExpr(a, db, keys, ctes, cat, depth);
  for (auto& a : e->partBy) collectExpr(a, db, keys, ctes, cat, depth);
  for (auto& o : e->winOrder) collectExpr(o.first, db, keys, ctes, cat, depth);
}
static void collectTables(const Stmt& s, const std::string& db, std::set<std::string>& keys, std::set<std::string> ctes, Catalog& cat, int depth) {
  if (depth > 8) return;
  for (auto& [name, body] : s.ctes) {
    if (body) collectTables(*body, db, keys, ctes, cat, depth + 1);
    ctes.insert(lower(name));
  }
  for (auto& f : s.from) {
    if (f.sub) collectTables(*f.sub, db, keys, ctes, cat, depth + 1);
    else if (!f.name.empty() && !ctes.count(lower(f.name))) {
      const std::string k = f.name.find('.') != std::string::npos ? lower(f.name) : db + "." + lower(f.name);
      std::string view;
      {
        std::lock_guard<std::mutex> g(cat.mu);
        auto it = cat.views.find(k);
        if (it != cat.views.end()) view = it->second;
      }
      if (!view.empty()) {
        if (view.rfind("/*cols:", 0) == 0) view = view.substr(view.find("*/") + 2);
        try {
          collectTables(parseSql(view), k.substr(0, k.find('.')), keys, {}, cat, depth + 1);
        } catch (std::exception&) {
        }
      } else {
        keys.insert(k);
      }
    }
    collectExpr(f.on, db, keys, ctes, cat, depth + 1);
  }
  for (auto& it : s.items) collectExpr(it.e, db, keys, ctes, cat, depth + 1);
  collectExpr(s.where, db, keys, ctes, cat, depth + 1);
  collectExpr(s.having, db, keys, ctes, cat, depth + 1);
  for (auto& g : s.groupBy) collectExpr(g, db, keys, ctes, cat, depth + 1);
  for (auto& o : s.orderBy) collectExpr(o.e, db, keys, ctes, cat, depth + 1);
  for (auto& j : s.joinConds) collectExpr(j, db, keys, ctes, cat, depth + 1);
  if (s.unionNext) collectTables(*s.unionNext, db, keys, ctes, cat, depth + 1);
}

// A statement reading several tables takes all their snapshots at one moment (under the shared commit
// gate), so it sees each transaction in all of its tables or in none. (One table: lazily, as read.)
void Engine::snapshotStatement(const Stmt& s) {
  if (!tSnapScope) return;
  std::set<std::string> keys;
  collectTables(s, lower(currentDb()), keys, {}, cat_, 0);
  std::vector<std::shared_ptr<Table>> tabs;
  for (auto& k : keys) {
    if (k.rfind("information_schema.", 0) == 0) continue;
    auto t = cat_.find(k);
    if (t && t->db != "__tmp" && !t->isShadow() && !tSnapScope->shadows.count(t.get())) tabs.push_back(t);
  }
  if (tabs.size() + tSnapScope->shadows.size() < 2) return;
  std::sort(tabs.begin(), tabs.end());
  std::shared_lock<std::shared_mutex> gate(gCommitGate);
  for (auto& t : tabs) {
    std::shared_ptr<Table> sh;
    {
      FairSharedMutex::barge = true;
      std::shared_lock<FairSharedMutex> lk(t->mu);
      FairSharedMutex::barge = false;
      sh = t->shadow();
    }
    tSnapScope->bases.push_back(t);
    tSnapScope->shadows[t.get()] = sh;
  }
}

std::shared_ptr<Table> Engine::tableForRead(const std::string& name, const Stmt& s) {
  auto t = cat_.get(name);
  if (!tSnapScope || t->db == "__tmp" || t->isShadow()) return t;
  auto it = tSnapScope->shadows.find(t.get());
  if (it != tSnapScope->shadows.end()) return it->second;
  if (s.from.size() == 1 && !s.unionNext && keyLookup(*t, s.where)) return t;  // microsecond lock on the live table
  for (auto& ix : t->indexes)
    if (ix.vec && ix.algo == 0) return t;  // IVF lists live in the index map
  std::shared_ptr<Table> sh;
  {
    FairSharedMutex::barge = true;  // (never queue behind a waiting writer for this)
    std::shared_lock<FairSharedMutex> lk(t->mu);
    FairSharedMutex::barge = false;
    sh = t->shadow();
  }
  tSnapScope->bases.push_back(t);
  tSnapScope->shadows[t.get()] = sh;
  return sh;
}

ResultSet Engine::doSelect(Stmt& s, bool explain) {
  auto t0 = Clock::now();
  std::vector<std::string> notes;
  BindCtx ctx;
  for (auto& f : s.from) {
    ctx.tabs.push_back(tableForRead(f.name, s));
    ctx.aliases.push_back(f.alias);
    ctx.usingCols.push_back(f.usingCols);
  }
  // Shared locks in a stable order (writers only ever take one table lock).
  std::vector<std::shared_ptr<Table>> lockOrder = ctx.tabs;
  std::sort(lockOrder.begin(), lockOrder.end());
  lockOrder.erase(std::unique(lockOrder.begin(), lockOrder.end()), lockOrder.end());
  // A PRIMARY KEY point lookup holds the lock for microseconds: it may pass writers waiting behind
  // long analytic queries instead of queueing with them.
  FairSharedMutex::barge = false;
  if (ctx.tabs.size() == 1 && ctx.tabs[0]->pk >= 0 && s.where && s.groupBy.empty() && !s.unionNext) {
    const std::string& pkName = ctx.tabs[0]->cols[ctx.tabs[0]->pk].name;
    std::function<bool(const ExprP&)> pkEq = [&](const ExprP& e) {
      if (e->kind == Expr::BIN && e->op == "AND") return pkEq(e->args[0]) || pkEq(e->args[1]);
      if (e->kind != Expr::BIN || e->op != "=") return false;
      const ExprP& a = e->args[0];
      const ExprP& b = e->args[1];
      return (a->kind == Expr::COL && b->kind == Expr::LIT && strcasecmp(a->name.c_str(), pkName.c_str()) == 0) ||
             (b->kind == Expr::COL && a->kind == Expr::LIT && strcasecmp(b->name.c_str(), pkName.c_str()) == 0);
    };
    FairSharedMutex::barge = pkEq(s.where);
  }
  std::vector<std::shared_lock<FairSharedMutex>> locks;
  const auto tLock = Clock::now();
  for (auto& t : lockOrder) locks.emplace_back(t->mu);
  FairSharedMutex::barge = false;
  if (envSetting("TRACE") && msSince(tLock) > 20) fprintf(stderr, "[trace] waited %.1f ms for table locks\n", msSince(tLock));

  // Expand * and bind the select list.
  std::vector<SelectItem> items;
  for (auto& it : s.items) {
    if (it.e->kind == Expr::STAR) {
      if (ctx.tabs.empty()) throw SqlError("No tables used", 1096);
      for (size_t ti = 0; ti < ctx.tabs.size(); ++ti) {
        if (!it.e->table.empty() && strcasecmp(it.e->table.c_str(), ctx.aliases[ti].c_str()) != 0) continue;
        for (auto& c : ctx.tabs[ti]->cols) items.push_back({mkCol(ctx.aliases[ti], c.name), c.name});
      }
    } else {
      items.push_back(it);
    }
  }
  // Result column names of unaliased items, from the expressions as written (binding rewrites them).
  std::vector<std::string> itemNames;
  for (auto& it : items)
    itemNames.push_back(it.e->kind == Expr::COL || (it.e->kind == Expr::LIT && !it.e->name.empty()) ? it.e->name : it.e->str());
  auto resolveAlias = [&](ExprP e) -> ExprP {
    if (e->kind == Expr::LIT && e->lit.kind == Value::INT) {
      if (e->lit.i < 1 || e->lit.i > (int64_t)items.size()) throw SqlError("Unknown column '" + e->str() + "' in 'order clause'", 1054);
      return items[e->lit.i - 1].e;
    }
    if (e->kind == Expr::COL && e->table.empty())
      for (auto& it : items)
        if (!it.alias.empty() && strcasecmp(it.alias.c_str(), e->name.c_str()) == 0) return it.e;
    return e;
  };
  std::function<void(ExprP&)> substAliases = [&](ExprP& e) {
    if (e->kind == Expr::COL && e->table.empty()) {
      for (auto& it : items)
        if (!it.alias.empty() && strcasecmp(it.alias.c_str(), e->name.c_str()) == 0) { e = it.e; return; }
    }
    for (auto& a : e->args) substAliases(a);
  };
  {
    bool plain = s.groupBy.empty() && !s.distinct && s.orderBy.empty() && !s.having && s.limit >= 0 && s.offset + s.limit <= 100000;
    for (auto& it : items) plain = plain && !hasAgg(it.e);
    ctx.lazyStr = plain;
    for (auto& it : items) bind(it.e, ctx);
    ctx.lazyStr = false;
  }
  std::vector<ExprP> groupBy;
  for (auto& g : s.groupBy) {
    ExprP e = resolveAlias(g);
    bind(e, ctx);
    groupBy.push_back(e);
  }
  ExprP having = s.having;
  if (having) { substAliases(having); bind(having, ctx); }
  std::vector<OrderItem> orderBy;
  std::function<void(ExprP&)> substOrder = [&](ExprP& e) {  // select-list aliases inside ORDER BY expressions (not in aggregates)
    if (e->kind == Expr::COL && e->table.empty()) {
      for (auto& it : items)
        if (!it.alias.empty() && strcasecmp(it.alias.c_str(), e->name.c_str()) == 0) { e = it.e; return; }
      return;
    }
    if (e->kind == Expr::FUNC && (isAggName(e->op) || isHostAggName(e->op))) return;
    for (auto& a : e->args) substOrder(a);
  };
  for (auto& o : s.orderBy) {
    OrderItem oi{resolveAlias(o.e), o.desc};
    if (oi.e == o.e && !oi.e->args.empty() && !oi.e->bound) {
      oi.e = cloneExpr(oi.e);
      substOrder(oi.e);
    }
    bind(oi.e, ctx);
    orderBy.push_back(oi);
  }
  ExprP where = s.where;
  if (where) bind(where, ctx);
  std::vector<ExprP> conj;
  flattenAnd(where, conj);
  for (auto& jc : s.joinConds) {
    bind(jc, ctx);
    flattenAnd(jc, conj);
  }
  // Outer joins: which tables can come out NULL-extended; ON conditions are bound per table.
  bool anyOuter = false;
  for (size_t i = 0; i < s.from.size(); ++i) {
    const int j = s.from[i].join;
    if (j == 1) ctx.outerTabs.insert((int)i);
    else if (j == 2) for (size_t k = 0; k < i; ++k) ctx.outerTabs.insert((int)k);
    else if (j == 3) for (size_t k = 0; k <= i; ++k) ctx.outerTabs.insert((int)k);
    anyOuter |= j != 0;
    if (s.from[i].on) bind(s.from[i].on, ctx);
  }
  bool aggQuery = !groupBy.empty() || (having && hasAgg(having));
  for (auto& it : items) aggQuery |= hasAgg(it.e);
  for (auto& o : orderBy) aggQuery |= hasAgg(o.e);
  if (s.distinct && !aggQuery) {
    for (auto& it : items) groupBy.push_back(it.e);
    aggQuery = true;
  }
  if (groupBy.size() > (size_t)kMaxKeys) throw SqlError("at most 8 GROUP BY expressions are supported", 1235);
  // Case-insensitive collation: string group keys whose dictionary has case variants are grouped
  // by a canonical code ('Bob' and 'bob' form one group), everywhere the key appears.
  if (ciColl()) {
    for (auto& g : groupBy) {
      Dict* d = g->ty == LType::VARCHAR ? ctx.dict(*g) : nullptr;
      if (!d) continue;
      auto cmap = ciCanon(*d);
      if (!cmap) continue;
      const std::string orig = canon(g);
      ExprP m = boundNode(Expr::FUNC, "#MAP", {g}, LType::VARCHAR);
      m->aux = cmap;
      ctx.exprDict[m.get()] = d;
      std::function<void(ExprP&)> sub = [&](ExprP& e) {
        if (!e || e == m) return;
        if (isAgg(e)) return;
        if (canon(e) == orig) { e = m; return; }
        for (auto& a : e->args) sub(a);
      };
      for (auto& it : items) sub(it.e);
      if (having) sub(having);
      for (auto& o : orderBy) sub(o.e);
      g = m;
    }
  }

  ResultSet res;
  res.isResult = true;
  for (size_t ii = 0; ii < items.size(); ++ii) {
    auto& it = items[ii];
    // information_schema columns keep their upper-case names (MySQL 8 returns TABLE_NAME for `select table_name`)
    const bool isCol = it.e->kind == Expr::COL && it.e->bt >= 0 && it.e->bt < (int)ctx.tabs.size();
    const bool isI_S = isCol && ctx.tabs[it.e->bt]->name.rfind("__tmp", 0) == 0 && ctx.tabs[it.e->bt]->name.find("_is_") != std::string::npos;
    res.jsonCols.push_back(isJsonExpr(it.e, &ctx));
    res.names.push_back(!it.alias.empty() ? it.alias : isI_S ? ctx.tabs[it.e->bt]->cols[it.e->bc].name
                        : ii < itemNames.size() ? itemNames[ii] : it.e->kind == Expr::COL ? it.e->name : it.e->str());
    res.types.push_back(it.e->ty);
  }
  if (tDescribeOnly) return res;  // COM_STMT_PREPARE: the result columns only
  auto finish = [&](std::vector<std::vector<Value>>& rows, std::vector<std::vector<Value>>& sortKeys) {
    if (!orderBy.empty()) {
      std::vector<size_t> perm(rows.size());
      for (size_t i = 0; i < perm.size(); ++i) perm[i] = i;
      std::stable_sort(perm.begin(), perm.end(), [&](size_t a, size_t b) {
        for (size_t k = 0; k < orderBy.size(); ++k) {
          int c = cmpVal(sortKeys[a][k], sortKeys[b][k]);
          if (c) return orderBy[k].desc ? c > 0 : c < 0;
        }
        return false;
      });
      std::vector<std::vector<Value>> sorted;
      sorted.reserve(rows.size());
      for (size_t i : perm) sorted.push_back(std::move(rows[i]));
      rows.swap(sorted);
    }
    int64_t start = std::min<int64_t>(s.offset, (int64_t)rows.size());
    int64_t end = s.limit < 0 ? (int64_t)rows.size() : std::min<int64_t>((int64_t)rows.size(), start + s.limit);
    for (int64_t i = start; i < end; ++i) {
      if (tWantVals) {
        res.vals.push_back(std::move(rows[i]));
        continue;
      }
      std::vector<std::string> out;
      for (auto& v : rows[i]) out.push_back(cell(v));
      res.rows.push_back(std::move(out));
    }
    res.typed = tWantVals;
  };
  auto explainResult = [&]() {
    ResultSet r;
    r.isResult = true;
    r.names = {"plan"};
    r.types = {LType::VARCHAR};
    for (auto& n : notes) r.rows.push_back({n});
    return r;
  };

  // ---- No FROM: constant expressions.
  if (ctx.tabs.empty()) {
    PostCtx pc;
    std::vector<std::vector<Value>> rows(1), keys(1);
    for (auto& it : items) rows[0].push_back(evalPost(it.e, pc));
    finish(rows, keys);
    return res;
  }

  // ---- Route 1: primary-key point lookup on the CPU (no GPU launch, microseconds).
  int64_t pkKey;
  if (ctx.tabs.size() == 1 && !aggQuery && !foreignPending(*ctx.tabs[0]) && findPkEq(conj, *ctx.tabs[0], 0, pkKey)) {
    Table& t = *ctx.tabs[0];
    notes.push_back("CPU point lookup via PRIMARY KEY hash index on " + t.name + "." + t.cols[t.pk].name + " = " +
                    std::to_string(pkKey));
    if (explain) return explainResult();
    std::vector<std::vector<Value>> rows, keys;
    auto f = t.pkIndex.find(pkKey);
    if (f != t.pkIndex.end()) {
      Loc l = f->second;
      std::vector<Value> vals;
      for (size_t c = 0; c < t.cols.size(); ++c) vals.push_back(cellValue(t, l, (int)c));
      PostCtx pc;
      pc.row = &vals;
      pc.colSlot = [](int bt, int bc) { return bt == 0 ? bc : -1; };
      if (!where || isTrue(evalPost(where, pc))) {
        std::vector<Value> row;
        for (auto& it : items) row.push_back(evalPost(it.e, pc));
        rows.push_back(row);
        keys.push_back({});
      }
    }
    finish(rows, keys);
    res.info = "cpu-point-lookup";
    return res;
  }

  // ---- Route 1b: equality on a secondary index (CPU hash lookup, no GPU scan).
  int64_t ixKey;
  bool ixFound = false;
  const IndexMeta* ixm = nullptr;
  if (ctx.tabs.size() == 1 && !aggQuery && !anyOuter && !foreignPending(*ctx.tabs[0]) &&
      (ixm = findIndexEq(conj, *ctx.tabs[0], 0, ixKey, ixFound))) {
    Table& t = *ctx.tabs[0];
    notes.push_back("CPU lookup via secondary index " + ixm->name + " on " + t.name + "." + t.cols[ixm->cols[0]].name);
    if (explain) return explainResult();
    ProfStage pl("index lookup");
    std::vector<std::vector<Value>> rows, keys;
    auto it = ixFound ? ixm->map.find(ixKey) : ixm->map.end();
    if (it != ixm->map.end()) {
      for (const Loc& l : it->second) {
        std::vector<Value> vals;
        for (size_t c = 0; c < t.cols.size(); ++c) vals.push_back(cellValue(t, l, (int)c));
        PostCtx pc;
        pc.row = &vals;
        pc.colSlot = [](int bt, int bc) { return bt == 0 ? bc : -1; };
        if (where && !isTrue(evalPost(where, pc))) continue;
        std::vector<Value> row, key;
        for (auto& itm : items) row.push_back(evalPost(itm.e, pc));
        for (auto& o : orderBy) key.push_back(evalPost(o.e, pc));
        rows.push_back(std::move(row));
        keys.push_back(std::move(key));
      }
    }
    finish(rows, keys);
    pl.done((int64_t)res.rows.size() + (int64_t)res.vals.size(), ixm->name);
    res.info = "cpu-index-lookup";
    return res;
  }

  // ---- Route 1b: ORDER BY v <*> q DESC / v <-> q LIMIT k on a CAGRA VECTOR INDEX: the GPU graph of
  // every large VRAM-resident segment is searched in one call (deleted rows masked out), the write
  // buffers and other segments exactly; candidates are re-scored exactly, then filtered by WHERE.
  if (ctx.tabs.size() == 1 && !aggQuery && !anyOuter && orderBy.size() == 1 && s.limit >= 0 && s.limit + s.offset <= 1024 &&
      orderBy[0].e->kind == Expr::FUNC && (orderBy[0].e->op == "#VDOT" || orderBy[0].e->op == "#VEUC") &&
      orderBy[0].e->args[0]->kind == Expr::COL && (orderBy[0].e->op == "#VDOT") == orderBy[0].desc && tAnn &&
      !foreignPending(*ctx.tabs[0])) {
    Table& t = *ctx.tabs[0];
    const ExprP& oe = orderBy[0].e;
    const IndexMeta* vix = nullptr;
    for (auto& ix : t.indexes)
      if (ix.vec && ix.algo == 1 && ix.cagra && ix.cols[0] == oe->args[0]->bc) vix = &ix;
    const std::vector<double>& qd = *oe->aux;
    if (vix && (int)qd.size() == vix->dim) {
      const bool dot = oe->op == "#VDOT";
      const int d = vix->dim, vc = vix->cols[0];
      const size_t want = (size_t)(s.limit + s.offset);
      // WHERE the GPU can evaluate becomes a per-segment keep-bitset (filtered graph search);
      // otherwise candidates are over-fetched and filtered on the host.
      const bool gpuWhere = where && gpuCompilable(where, ctx);
      Compiler wcomp(&ctx);
      Prog wprog;
      if (gpuWhere) {
        wprog = wcomp.compile(where);
        wcomp.uploadSets();
      }
      const int k = (int)std::min<size_t>(1024, where && !gpuWhere ? std::max<size_t>(want * 8, 64) : want);
      const int itopk = std::max(k, tCagraItopk > 0 ? tCagraItopk : vix->itopk > 0 ? vix->itopk : 64);
      std::vector<float> q(qd.begin(), qd.end());
      ProfStage pl("vector index");
      // Graphs for the segments that have (or can get) one; everything else is scored exactly.
      std::vector<CagraGraph*> graphs;
      std::vector<int> graphSeg;
      std::vector<const uint32_t*> keep;
      std::vector<std::shared_ptr<DevVec<uint32_t>>> keepBufs;
      std::vector<int> exactSegs;
      std::vector<std::shared_ptr<CagraGraph>> hold;
      std::vector<std::pair<int, std::vector<uint32_t>>> selRows;  // gpuWhere: per graph segment, rows passing WHERE
      int64_t selected = 0;
      int built = 0;
      for (size_t si = 0; si < t.segs.size(); ++si) {
        Segment& sg = *t.segs[si];
        if (sg.n == 0 || sg.ndel() == sg.n) continue;
        std::shared_ptr<CagraGraph> g;
        if (segCold(*sg.d, vc)) segLoad(*sg.d, vc);  // (graph columns stay in memory)
        if (sg.n >= 2048 && sg.dev[vc] && !sg.mapped[vc]) {
          std::lock_guard<std::mutex> lk(vix->cagra->mu);
          auto it = vix->cagra->graphs.find(sg.id);
          if (it != vix->cagra->graphs.end()) g = it->second;
          else if (!vix->cagra->failed.count(sg.id)) {
            std::string err;
            g = cagraBuild((const float*)sg.dev[vc], sg.n, d, dot, vix->graphDegree, err);
            if (g) vix->cagra->graphs[sg.id] = g, ++built;
            else {
              vix->cagra->failed.insert(sg.id);
              fprintf(stderr, "[warpstore] CAGRA build failed for %s.%s: %s (exact search used)\n", t.name.c_str(), vix->name.c_str(), err.c_str());
            }
          }
        }
        if (!g) {
          exactSegs.push_back((int)si);
          continue;
        }
        if (gpuWhere) {  // rows passing WHERE (and not deleted) -> keep-bitset
          const uint32_t* sdel = sg.delDev();
          const BatchRef br{sg.part, sg.n, &sg.dev, sdel, &sg.enc, 0, &sg};
          struct Temps { ~Temps() { releaseScanTemps(); } } temps;
          for (size_t kk = 0; kk < wcomp.slotCols.size(); ++kk) bindView(wcomp.env.cols[kk], br, wcomp.slotCols[kk].second);
          scanFlush();
          DevVec<int32_t> idx(sg.n);
          const int64_t m = gpuSelect(wcomp.env, true, wprog, sg.n, sdel, idx.p);
          if (m == 0) continue;
          std::vector<uint32_t> bits((sg.n + 31) / 32, 0);
          for (int32_t r : idx.download(m)) bits[r >> 5] |= 1u << (r & 31);
          selected += m;
          selRows.push_back({(int)si, std::move(bits)});
          hold.push_back(g);
          graphs.push_back(g.get());
          graphSeg.push_back((int)si);
          auto b = std::make_shared<DevVec<uint32_t>>((int64_t)selRows.back().second.size());
          h2d(b->p, selRows.back().second.data(), selRows.back().second.size() * 4);
          keepBufs.push_back(b);
          keep.push_back(b->p);
          continue;
        }
        hold.push_back(g);
        graphs.push_back(g.get());
        graphSeg.push_back((int)si);
        if (sg.ndel() > 0) {  // keep-bitset: the complement of the deleted-row bitmap
          std::vector<uint32_t> bits(sg.delHost().size());
          for (size_t w = 0; w < bits.size(); ++w) bits[w] = ~sg.delHost()[w];
          auto b = std::make_shared<DevVec<uint32_t>>((int64_t)bits.size());
          h2d(b->p, bits.data(), bits.size() * 4);
          keepBufs.push_back(b);
          keep.push_back(b->p);
        } else {
          keep.push_back(nullptr);
        }
      }
      notes.push_back("ANN via VECTOR INDEX " + vix->name + " (CAGRA GPU graph: " + std::to_string(graphs.size()) + " segment graphs" +
                      (built ? " (" + std::to_string(built) + " built now)" : "") + ", itopk " + std::to_string(itopk) + ", " +
                      std::to_string(exactSegs.size()) + " segments + write buffers exact) on " + t.name + "." + t.cols[vc].name);
      if (explain) return explainResult();
      std::vector<CagraHit> hits;
      std::string err;
      // A selective filter leaves few rows: scoring them exactly beats a graph walk that must skip
      // most of its nodes.
      const bool exactSelected = gpuWhere && selected < 8192;
      if (exactSelected) notes.push_back("WHERE keeps " + std::to_string(selected) + " rows: scored exactly");
      if (exactSelected || cagraSearch(graphs, keep, q.data(), d, k, itopk, hits, err)) {
        auto score = [&](const float* v) {
          double acc = 0;
          for (int i = 0; i < d; ++i) acc += dot ? (double)v[i] * qd[i] : ((double)v[i] - qd[i]) * ((double)v[i] - qd[i]);
          return dot ? -acc : acc;  // lower is better
        };
        std::vector<std::pair<double, Loc>> cand;
        if (exactSelected) {
          for (auto& [si, bits] : selRows)
            for (size_t w = 0; w < bits.size(); ++w)
              for (uint32_t m = bits[w]; m; m &= m - 1) {
                const Loc l{si, (int32_t)(w * 32 + __builtin_ctz(m))};
                cand.push_back({score((const float*)t.colData(l, vc).data() + (size_t)l.row * d), l});
              }
        }
        for (auto& h : hits) {
          const Loc l{graphSeg[h.graph], (int32_t)h.row};
          cand.push_back({score((const float*)t.colData(l, vc).data() + (size_t)l.row * d), l});
        }
        auto exactRows = [&](Loc first, int64_t n, const DelBits& del) {
          for (int64_t r = 0; r < n; ++r) {
            if (del.test(r)) continue;
            const Loc l{first.seg, (int32_t)r};
            cand.push_back({score((const float*)t.colData(l, vc).data() + (size_t)r * d), l});
          }
        };
        for (int si : exactSegs) exactRows(Loc{si, 0}, t.segs[si]->n, *t.segs[si]->del);
        for (int p = 0; p < t.nparts; ++p) exactRows(bufLoc(p, 0), t.bufs[p]->n, *t.bufs[p]->del);
        std::sort(cand.begin(), cand.end(), [](auto& a, auto& b) { return a.first < b.first; });
        std::vector<std::vector<Value>> rows, keys;
        for (auto& [sc, l] : cand) {
          if (rows.size() >= want) break;
          std::vector<Value> vals;
          for (size_t c = 0; c < t.cols.size(); ++c) vals.push_back(cellValue(t, l, (int)c));
          PostCtx pc;
          pc.row = &vals;
          pc.colSlot = [](int bt, int bc) { return bt == 0 ? bc : -1; };
          if (where && !isTrue(evalPost(where, pc))) continue;
          std::vector<Value> row;
          for (auto& itm : items) row.push_back(evalPost(itm.e, pc));
          rows.push_back(std::move(row));
          keys.push_back({Value::D(dot ? -sc : std::sqrt(sc))});
        }
        if (rows.size() >= want || !where) {  // a selective WHERE that left too few: exact path below
          finish(rows, keys);
          pl.done((int64_t)cand.size(), vix->name);
          res.info = "ann-cagra";
          return res;
        }
        notes.push_back("CAGRA candidates too few after WHERE: exact search");
      } else {
        fprintf(stderr, "[warpstore] CAGRA search failed: %s (exact search used)\n", err.c_str());
      }
    }
  }

  // ---- Route 1c: ORDER BY v <*> q DESC / v <-> q LIMIT k on a VECTOR INDEX: probe the nprobe
  // nearest IVF lists on the CPU, score only their vectors, then evaluate the select list for the best.
  if (ctx.tabs.size() == 1 && !aggQuery && !anyOuter && orderBy.size() == 1 && s.limit >= 0 && s.limit <= 100000 &&
      orderBy[0].e->kind == Expr::FUNC && (orderBy[0].e->op == "#VDOT" || orderBy[0].e->op == "#VEUC") &&
      orderBy[0].e->args[0]->kind == Expr::COL && (orderBy[0].e->op == "#VDOT") == orderBy[0].desc && tAnn &&
      !foreignPending(*ctx.tabs[0])) {
    Table& t = *ctx.tabs[0];
    const ExprP& oe = orderBy[0].e;
    const IndexMeta* vix = nullptr;
    for (auto& ix : t.indexes)
      if (ix.vec && ix.nlist > 0 && ix.cols[0] == oe->args[0]->bc) vix = &ix;
    if (vix) {
      const bool dot = oe->op == "#VDOT";
      const int d = vix->dim;
      const std::vector<double>& q = *oe->aux;
      if ((int)q.size() == d) {
        int nprobe = tNprobe > 0 ? tNprobe : vix->nprobe > 0 ? vix->nprobe : std::max(8, vix->nlist / 32);
        nprobe = std::min(nprobe, vix->nlist);
        notes.push_back("ANN via VECTOR INDEX " + vix->name + " (IVF, " + std::to_string(vix->nlist) + " lists, nprobe " +
                        std::to_string(nprobe) + ") on " + t.name + "." + t.cols[vix->cols[0]].name);
        if (explain) return explainResult();
        ProfStage pl("vector index");
        auto score = [&](const float* v) {
          double acc = 0;
          for (int i = 0; i < d; ++i) acc += dot ? v[i] * q[i] : (v[i] - q[i]) * (v[i] - q[i]);
          return dot ? -acc : acc;  // lower is better
        };
        std::vector<std::pair<double, int>> cs(vix->nlist);
        for (int c = 0; c < vix->nlist; ++c) cs[c] = {score(vix->cent.data() + (size_t)c * d), c};
        std::partial_sort(cs.begin(), cs.begin() + nprobe, cs.end());
        std::vector<std::pair<double, Loc>> cand;
        for (int i = 0; i < nprobe; ++i) {
          auto it = vix->map.find(cs[i].second);
          if (it == vix->map.end()) continue;
          for (const Loc& l : it->second)
            cand.push_back({score((const float*)t.colData(l, vix->cols[0]).data() + (size_t)l.row * d), l});
        }
        // Rows not yet assigned to the trained lists (key 0 before training) are covered by the lists.
        std::sort(cand.begin(), cand.end(), [](auto& a, auto& b) { return a.first < b.first; });
        std::vector<std::vector<Value>> rows, keys;
        const size_t want = (size_t)(s.limit + s.offset);
        for (auto& [sc, l] : cand) {
          if (rows.size() >= want) break;
          std::vector<Value> vals;
          for (size_t c = 0; c < t.cols.size(); ++c) vals.push_back(cellValue(t, l, (int)c));
          PostCtx pc;
          pc.row = &vals;
          pc.colSlot = [](int bt, int bc) { return bt == 0 ? bc : -1; };
          if (where && !isTrue(evalPost(where, pc))) continue;
          std::vector<Value> row;
          for (auto& itm : items) row.push_back(evalPost(itm.e, pc));
          rows.push_back(std::move(row));
          keys.push_back({Value::D(dot ? -sc : std::sqrt(sc))});
        }
        finish(rows, keys);
        pl.done((int64_t)cand.size(), vix->name);
        res.info = "ann-ivf";
        return res;
      }
    }
  }

  // ---- Split predicates: per-table pushdown, equi-join edges, residual.
  std::vector<std::vector<ExprP>> pushdown(ctx.tabs.size());
  struct Edge { int at, ac, bt, bc; ExprP e; };
  std::vector<Edge> edges;
  std::vector<ExprP> residual;
  for (auto& p : conj) {
    std::set<int> ts;
    tablesOf(p, ts);
    bool touchesOuter = false;
    for (int t : ts) touchesOuter |= ctx.outerTabs.count(t) > 0;
    if (touchesOuter) {  // WHERE on a NULL-extended table filters after the outer join
      residual.push_back(p);
    } else if (ts.size() <= 1) {
      pushdown[ts.empty() ? 0 : *ts.begin()].push_back(p);
    } else if (ts.size() == 2 && p->kind == Expr::BIN && p->op == "=" && p->args[0]->kind == Expr::COL &&
               p->args[1]->kind == Expr::COL &&
               (p->args[0]->ty != LType::VARCHAR || (p->args[1]->ty == LType::VARCHAR && ctx.dict(*p->args[0]) && ctx.dict(*p->args[1])))) {
      edges.push_back({p->args[0]->bt, p->args[0]->bc, p->args[1]->bt, p->args[1]->bc, p});
    } else {
      residual.push_back(p);
    }
  }

  // Aggregate slots: slot 0 is a hidden COUNT(*) used for COUNT/AVG and empty-group detection.
  std::map<std::string, AggRef> aggs;
  AggPlan plan;
  std::vector<ExprP> aggArgs;  // per slot (null for COUNT)
  plan.naggs = 1;
  plan.ops[0] = AGG_COUNT;
  aggArgs.push_back(nullptr);
  std::map<std::string, int> slotByCanon;
  // Slot counting the non-NULL rows of an aggregate argument (0 = COUNT(*) when it can't be NULL).
  auto countSlot = [&](const ExprP& x) -> int {
    if (x->kind == Expr::STAR || !nullable(x, ctx)) return 0;
    std::string sk = "cnt" + canon(x);
    auto it = slotByCanon.find(sk);
    if (it != slotByCanon.end()) return it->second;
    if (plan.naggs >= kMaxAggs) throw SqlError("too many aggregates", 1105);
    plan.ops[plan.naggs] = AGG_SUM;
    aggArgs.push_back(boundNode(Expr::FUNC, "ISNOTNULL", {x}, LType::INT));
    slotByCanon.emplace(sk, plan.naggs);
    return plan.naggs++;
  };
  ExprP distX;  // COUNT(DISTINCT x): computed by the two-phase GPU path below
  bool anyDistinct = false;  // (grouped by x in its first phase, so BIGINT sums split as with GROUP BY)
  std::function<void(const ExprP&)> findDistinct = [&](const ExprP& e) {
    if (!e) return;
    if (isAgg(e) && e->distinct) anyDistinct = true;
    for (auto& a : e->args) findDistinct(a);
  };
  for (auto& it : items) findDistinct(it.e);
  findDistinct(having);
  for (auto& o : orderBy) findDistinct(o.e);
  std::function<void(const ExprP&)> collectAggs = [&](const ExprP& e) {
    if (isAgg(e)) {
      if (e->distinct) {
        if (distX && canon(distX) != canon(e->args[0])) throw SqlError("internal: several DISTINCT arguments", 1105);
        distX = e->args[0];
      }
      std::string k = canon(e);
      if (aggs.count(k)) return;
      AggRef r;
      r.distinct = e->distinct;
      r.fn = e->op;
      r.argTy = e->args[0]->ty;
      r.dict = ctx.dict(*e);
      if (auto u = ctx.unrank.find(e.get()); u != ctx.unrank.end()) r.unrank = u->second;
      if (r.dict) r.argTy = LType::VARCHAR;
      r.scale = e->scale;
      r.cntSlot = countSlot(e->args[0]);
      if (r.fn == "COUNT") r.slot = r.cntSlot;
      if (r.fn != "COUNT") {
        uint8_t op = r.fn == "MIN" ? AGG_MIN : r.fn == "MAX" ? AGG_MAX : AGG_SUM;
        const ExprP& arg = e->args[0];
        const bool rawDec = op == AGG_SUM && arg->kind == Expr::COL && arg->ty == LType::DECIMAL;
        if (rawDec) r.decScale = arg->scale;
        std::string sk = std::to_string(op) + (rawDec ? "raw" : "") + canon(e->args[0]);
        auto it = slotByCanon.find(sk);
        if (it == slotByCanon.end()) {
          if (plan.naggs >= kMaxAggs) throw SqlError("too many aggregates", 1105);
          plan.ops[plan.naggs] = op;
          aggArgs.push_back(rawDec ? boundNode(Expr::FUNC, "#RAW", {arg}, LType::BIGINT) : e->args[0]);
          it = slotByCanon.emplace(sk, plan.naggs++).first;
          // Grouped SUM of a 64-bit integer column: exact as two int64 sums of its 32-bit halves.
          if (op == AGG_SUM && (!groupBy.empty() || anyDistinct) && arg->kind == Expr::COL && (arg->ty == LType::BIGINT || arg->ty == LType::DATETIME)) {
            if (plan.naggs >= kMaxAggs) throw SqlError("too many aggregates", 1105);
            plan.ops[plan.naggs] = AGG_SUM;
            plan.aggPart[it->second] = 1;
            plan.aggPart[plan.naggs] = 2;
            aggArgs.push_back(arg);
            slotByCanon.emplace(sk + "#hi", plan.naggs++);
          }
        }
        r.slot = it->second;
        if (auto h = slotByCanon.find(sk + "#hi"); h != slotByCanon.end()) r.hiSlot = h->second;
      }
      aggs[k] = r;
      return;
    }
    for (auto& a : e->args) collectAggs(a);
  };

  // Columns needed per table (for materialization paths).
  std::set<std::pair<int, int>> needed;
  for (auto& it : items) colsOf(it.e, needed);
  for (auto& g : groupBy) colsOf(g, needed);
  for (auto& o : orderBy) colsOf(o.e, needed);
  if (having) colsOf(having, needed);
  for (auto& r : residual) colsOf(r, needed);
  for (auto& e : edges) colsOf(e.e, needed);
  for (auto& f : s.from)
    if (f.on) colsOf(f.on, needed);
  auto needOf = [&](int bt) {
    std::set<int> c;
    for (auto& [t, col] : needed)
      if (t == bt) c.insert(col);
    if (c.empty()) c.insert(0);
    return c;
  };

  // Build the joined relation (multi-table). Small tables are filtered up front. Big ones are
  // scanned only when the join order reaches them, with join filters: exact bitmaps of the join
  // keys already in the joined relation, applied inside the big table's scan (sideways
  // information passing), so rows that cannot match are never gathered or joined.
  // With outer joins the tables are joined in FROM order (join order matters for their semantics).
  // Equality between expressions of the two sides (a.x = b.y + 1, a.d = DATE(b.ts), ..): both key
  // expressions evaluated on the GPU into temporary key columns, then a hash join on them.
  auto exprEquiCond = [&](const std::set<int>& lt, const std::set<int>& rt, const std::vector<ExprP>& conds, ExprP& le, ExprP& re) {
    for (auto& c : conds) {
      if (c->kind != Expr::BIN || c->op != "=" || hasAgg(c)) continue;
      std::set<int> t0, t1;
      tablesOf(c->args[0], t0);
      tablesOf(c->args[1], t1);
      if (t0.empty() || t1.empty()) continue;
      auto within = [](const std::set<int>& a, const std::set<int>& b) { return std::includes(b.begin(), b.end(), a.begin(), a.end()); };
      if (c->args[0]->ty == LType::VARCHAR || c->args[1]->ty == LType::VARCHAR) continue;
      if (within(t0, lt) && within(t1, rt)) { le = c->args[0], re = c->args[1]; return true; }
      if (within(t1, lt) && within(t0, rt)) { le = c->args[1], re = c->args[0]; return true; }
    }
    return false;
  };
  auto addKeyCol = [&](Rel& r, const ExprP& e, int tag, bool integral) {
    Compiler comp(&ctx);
    Prog pr = comp.compile(e);
    comp.uploadSets();
    bindRelCols(comp, r);
    DevVec<double> v(std::max<int64_t>(r.n, 1));
    gpuEvalAt(comp.env, pr, nullptr, r.n, v.p);
    std::vector<double> h = v.download(r.n);
    std::vector<int64_t> k(r.n);
    for (int64_t i = 0; i < r.n; ++i) {
      double d = h[i];
      if (d != d) { k[i] = kNullI64; continue; }
      if (integral) { k[i] = std::llround(d); continue; }
      if (d == 0) d = 0;  // -0.0 = 0.0
      memcpy(&k[i], &d, 8);
    }
    RelCol c{tag, 0, PType::I64, DevVec<uint8_t>(std::max<int64_t>(r.n, 1) * 8), 8};
    if (r.n) h2d(c.data.p, k.data(), r.n * 8);
    r.cols.push_back(std::move(c));
  };
  auto exprEquiJoin = [&](Rel& left, Rel& right, const std::vector<ExprP>& conds, Rel& out) {
    ExprP le, re;
    if (!exprEquiCond(left.tabs, right.tabs, conds, le, re)) return false;
    const int kA = -2000, kB = -2001;
    auto intTy = [](LType t) { return t == LType::INT || t == LType::BIGINT || t == LType::DATE || t == LType::DATETIME; };
    const bool integral = intTy(le->ty) && intTy(re->ty);  // else both as float64 bit patterns
    addKeyCol(left, le, kA, integral);
    addKeyCol(right, re, kB, integral);
    out = hashJoin(left, right, kA, 0, kB, 0, notes);
    for (auto* r : {&out, &left, &right})
      r->cols.erase(std::remove_if(r->cols.begin(), r->cols.end(), [&](const RelCol& c) { return c.bt == kA || c.bt == kB; }), r->cols.end());
    notes.push_back("    (hash join on expression keys: " + le->str() + " = " + re->str() + ")");
    return true;
  };

  // Joins tables [from, end) onto cur in FROM order (outer joins: the order matters).
  auto buildOrdered = [&](Rel cur, size_t from, std::vector<bool> edgeUsed) -> Rel {
    for (size_t i = from; i < ctx.tabs.size(); ++i) {
      const TableRef& ref = s.from[i];
      if (ref.join == 0) {
        Rel ri = materialize(ctx, (int)i, pushdown[i], needOf((int)i), -1, notes);
        int ei = -1;
        for (size_t k = 0; k < edges.size() && ei < 0; ++k) {
          auto& e = edges[k];
          if (!edgeUsed[k] && ((cur.tabs.count(e.at) && e.bt == (int)i) || (cur.tabs.count(e.bt) && e.at == (int)i))) ei = (int)k;
        }
        if (ei < 0) {
          Rel j;
          if (exprEquiJoin(cur, ri, residual, j)) cur = std::move(j);
          else cur = crossJoin(cur, ri, notes);
        } else {
          edgeUsed[ei] = true;
          auto& e = edges[ei];
          const Dict* da = ctx.tabs[e.at]->cols[e.ac].lt == LType::VARCHAR ? ctx.tabs[e.at]->dicts[e.ac].get() : nullptr;
          const Dict* db = ctx.tabs[e.bt]->cols[e.bc].lt == LType::VARCHAR ? ctx.tabs[e.bt]->dicts[e.bc].get() : nullptr;
          cur = cur.tabs.count(e.at) ? hashJoin(cur, ri, e.at, e.ac, e.bt, e.bc, notes, da, db) : hashJoin(cur, ri, e.bt, e.bc, e.at, e.ac, notes, db, da);
        }
        continue;
      }
      std::vector<ExprP> on, rightOnly, other;
      flattenAnd(ref.on, on);
      int ct = -1, cc = -1, rc = -1;
      for (auto& c : on) {
        std::set<int> ts;
        tablesOf(c, ts);
        if (ct < 0 && c->kind == Expr::BIN && c->op == "=" && c->args[0]->kind == Expr::COL && c->args[1]->kind == Expr::COL &&
            (c->args[0]->ty == LType::VARCHAR) == (c->args[1]->ty == LType::VARCHAR) &&
            c->args[0]->ty != LType::DOUBLE && c->args[1]->ty != LType::DOUBLE) {  // (DOUBLE keys: the general path)
          const ExprP &a = c->args[0], &b = c->args[1];
          if (cur.tabs.count(a->bt) && b->bt == (int)i) { ct = a->bt; cc = a->bc; rc = b->bc; continue; }
          if (cur.tabs.count(b->bt) && a->bt == (int)i) { ct = b->bt; cc = b->bc; rc = a->bc; continue; }
        }
        if (ts.size() == 1 && *ts.begin() == (int)i) rightOnly.push_back(c);
        else other.push_back(c);
      }
      if (ct < 0 || !other.empty() || (ref.join != 1 && !rightOnly.empty())) {
        // General ON (no equality, or conditions beyond one equality plus right-side filters): inner
        // join on the equality (else cross join), the rest of ON as a GPU filter over the pairs, then
        // the preserved side's unmatched rows NULL-extended.
        std::vector<ExprP> rest = other;
        rest.insert(rest.end(), rightOnly.begin(), rightOnly.end());
        Rel ri = materialize(ctx, (int)i, {}, needOf((int)i), -1, notes);
        const int kIdA = -1000, kIdB = -1001;
        auto addId = [](Rel& r, int tag) {
          std::vector<int32_t> seq((size_t)r.n);
          for (int64_t k = 0; k < r.n; ++k) seq[k] = (int32_t)k;
          RelCol c{tag, 0, PType::I32, DevVec<uint8_t>(std::max<int64_t>(r.n, 1) * 4), 4};
          if (r.n) h2d(c.data.p, seq.data(), r.n * 4);
          r.cols.push_back(std::move(c));
        };
        addId(cur, kIdA);
        addId(ri, kIdB);
        Rel inner;
        if (ct >= 0) {
          const Dict* dl = ctx.tabs[ct]->cols[cc].lt == LType::VARCHAR ? ctx.tabs[ct]->dicts[cc].get() : nullptr;
          const Dict* dr = ctx.tabs[i]->cols[rc].lt == LType::VARCHAR ? ctx.tabs[i]->dicts[rc].get() : nullptr;
          inner = hashJoin(cur, ri, ct, cc, (int)i, rc, notes, dl, dr);
        } else if (exprEquiJoin(cur, ri, rest, inner)) {
        } else {
          if ((double)cur.n * (double)ri.n > 4e8)
            throw SqlError("outer join without an equality would compare " + std::to_string(cur.n) + " x " + std::to_string(ri.n) + " rows; add an equality to ON", 1235);
          inner = crossJoin(cur, ri, notes);
        }
        std::vector<int32_t> pa, pb;  // matching (left row, right row) pairs
        {
          DevVec<int32_t> sel;
          int64_t m = inner.n;
          if (!rest.empty() && inner.n > 0) {
            Compiler comp(&ctx);
            Prog fp = comp.compile(andAll(rest));
            comp.uploadSets();
            bindRelCols(comp, inner);
            sel = DevVec<int32_t>(inner.n);
            m = gpuSelect(comp.env, true, fp, inner.n, nullptr, sel.p);
          }
          const std::pair<int, std::vector<int32_t>*> idCols[] = {{kIdA, &pa}, {kIdB, &pb}};
          for (auto [tag, out] : idCols) {
            const RelCol& c = inner.cols[inner.find(tag, 0)];
            if (sel.p) {
              DevVec<int32_t> g(std::max<int64_t>(m, 1));
              gpuGather(c.data.p, PType::I32, sel.p, m, g.p);
              *out = g.download(m);
            } else {
              out->resize(m);
              if (m) d2h(out->data(), c.data.p, m * 4);
            }
          }
        }
        std::vector<uint8_t> hitA(cur.n, 0), hitB(ri.n, 0);
        for (size_t k = 0; k < pa.size(); ++k) hitA[pa[k]] = hitB[pb[k]] = 1;
        std::vector<int32_t> fa = pa, fb = pb;
        if (ref.join == 1 || ref.join == 3)
          for (int64_t k = 0; k < cur.n; ++k)
            if (!hitA[k]) fa.push_back((int32_t)k), fb.push_back(-1);
        if (ref.join == 2 || ref.join == 3)
          for (int64_t k = 0; k < ri.n; ++k)
            if (!hitB[k]) fa.push_back(-1), fb.push_back((int32_t)k);
        const int64_t m = (int64_t)fa.size();
        DevVec<int32_t> da(std::max<int64_t>(m, 1)), db(std::max<int64_t>(m, 1));
        if (m) h2d(da.p, fa.data(), m * 4), h2d(db.p, fb.data(), m * 4);
        Rel out;
        out.n = m;
        out.tabs = cur.tabs;
        out.tabs.insert(ri.tabs.begin(), ri.tabs.end());
        for (auto* side : {&cur, &ri})
          for (auto& c : side->cols) {
            if (c.bt <= kIdA) continue;
            RelCol rc2{c.bt, c.bc, c.pt, DevVec<uint8_t>(std::max<int64_t>(m, 1) * c.esize), c.esize};
            gatherCol(c.data.p, c.pt, c.esize, side == &cur ? da.p : db.p, m, rc2.data.p, true);
            out.cols.push_back(std::move(rc2));
          }
        static const char* kind[] = {"", "LEFT", "RIGHT", "FULL"};
        notes.push_back(std::string("  GPU ") + kind[ref.join] + " OUTER join on a general ON condition: " + std::to_string(pa.size()) +
                        " matching pairs -> " + std::to_string(m) + " rows");
        cur = std::move(out);
        continue;
      }
      Rel ri = materialize(ctx, (int)i, ref.join == 1 ? rightOnly : std::vector<ExprP>{}, needOf((int)i), -1, notes);
      const Dict* dl = ctx.tabs[ct]->cols[cc].lt == LType::VARCHAR ? ctx.tabs[ct]->dicts[cc].get() : nullptr;
      const Dict* dr = ctx.tabs[i]->cols[rc].lt == LType::VARCHAR ? ctx.tabs[i]->dicts[rc].get() : nullptr;
      cur = outerJoin(cur, ri, ct, cc, (int)i, rc, ref.join, notes, dl, dr);
    }
    for (size_t k = 0; k < edges.size(); ++k)
      if (!edgeUsed[k]) residual.push_back(edges[k].e);
    return cur;
  };

  auto buildJoin = [&]() -> Rel {
    // With outer joins only at the end of FROM (e.g. decorrelated subqueries), the inner-join prefix
    // is ordered by cost and the outer joins follow in order; other outer joins: FROM order.
    size_t prefixN = ctx.tabs.size();
    if (anyOuter) {
      size_t k = 0;
      while (k < s.from.size() && s.from[k].join == 0) ++k;
      bool trailing = k >= 2;
      for (size_t i = k; i < s.from.size(); ++i) trailing = trailing && s.from[i].join != 0;
      if (!trailing) return buildOrdered(materialize(ctx, 0, pushdown[0], needOf(0), -1, notes), 1, std::vector<bool>(edges.size(), false));
      prefixN = k;
    }
    const size_t nt = ctx.tabs.size();
    std::vector<Rel> rels(nt);
    std::vector<bool> mat(nt, false);
    for (size_t i = 0; i < prefixN; ++i)
      if (!tJoinFilters || ctx.tabs[i]->liveRows() < tLazyRows) {
        rels[i] = materialize(ctx, (int)i, pushdown[i], needOf((int)i), -1, notes);
        mat[i] = true;
      }
    if (std::find(mat.begin(), mat.end(), true) == mat.end()) {
      size_t sm = 0;
      for (size_t i = 1; i < prefixN; ++i)
        if (ctx.tabs[i]->liveRows() < ctx.tabs[sm]->liveRows()) sm = i;
      rels[sm] = materialize(ctx, (int)sm, pushdown[sm], needOf((int)sm), -1, notes);
      mat[sm] = true;
    }
    auto est = [&](size_t i) { return mat[i] ? (double)rels[i].n : (double)ctx.tabs[i]->liveRows(); };
    // Greedy left-deep join order. Start from the smallest relation; at each step pick the edge
    // with the smallest estimated output, using PRIMARY KEY uniqueness to recognise FK->PK joins
    // (which never grow the input). Build side is always the smaller input.
    std::vector<bool> used(nt, false);
    for (size_t i = prefixN; i < nt; ++i) used[i] = true;  // (joined afterwards, in order)
    size_t start = nt;
    for (size_t i = 0; i < nt; ++i)
      if (mat[i] && (start == nt || rels[i].n < rels[start].n)) start = i;
    Rel cur = std::move(rels[start]);
    used[start] = true;
    std::set<std::pair<int, int>> uniq;  // columns known to be unique in `cur`
    if (ctx.tabs[start]->pk >= 0) uniq.insert({(int)start, ctx.tabs[start]->pk});
    std::vector<bool> edgeUsed(edges.size(), false);
    for (size_t step = 1; step < prefixN; ++step) {
      int best = -1, bestEdge = -1;
      double bestEst = 0;
      for (size_t ei = 0; ei < edges.size(); ++ei) {
        auto& e = edges[ei];
        int other = cur.tabs.count(e.at) && !cur.tabs.count(e.bt) ? e.bt : cur.tabs.count(e.bt) && !cur.tabs.count(e.at) ? e.at : -1;
        if (other < 0 || used[other]) continue;
        bool aIsCur = other == e.bt;
        std::pair<int, int> curCol = aIsCur ? std::make_pair(e.at, e.ac) : std::make_pair(e.bt, e.bc);
        int otherCol = aIsCur ? e.bc : e.ac;
        // Without key information: rows x rows / the larger number of distinct join values (from
        // dictionary sizes / value ranges; a many-to-many join on a low-cardinality column is huge).
        auto ndv = [&](int bt, int bc, double rows) {
          int64_t lo, hi;
          double n = colRange(*ctx.tabs[bt], bc, lo, hi) ? (double)span64(lo, hi) + 1 : rows;
          return std::max(1.0, std::min(n, rows));
        };
        const double nd = std::max(ndv(curCol.first, curCol.second, (double)cur.n), ndv(other, otherCol, (double)ctx.tabs[other]->liveRows()));
        double est1 = otherCol == ctx.tabs[other]->pk ? (double)cur.n
                    : uniq.count(curCol) ? est(other) : (double)cur.n * est(other) / nd;
        if (best < 0 || est1 < bestEst || (est1 == bestEst && est(other) < est(best)))
          best = other, bestEdge = (int)ei, bestEst = est1;
      }
      if (best < 0) {  // no column equality to any remaining table: an expression equality, else a cross join
        ExprP le, re;
        for (size_t i = 0; i < nt && best < 0; ++i)
          if (!used[i] && exprEquiCond(cur.tabs, std::set<int>{(int)i}, residual, le, re)) best = (int)i;
        if (best < 0)
          for (size_t i = 0; i < nt; ++i)
            if (!used[i] && (best < 0 || est(i) < est(best))) best = (int)i;
        if (!mat[best]) {
          rels[best] = materialize(ctx, best, pushdown[best], needOf(best), -1, notes);
          mat[best] = true;
        }
        Rel j;
        if (exprEquiJoin(cur, rels[best], residual, j)) cur = std::move(j);
        else cur = crossJoin(cur, rels[best], notes);
        used[best] = true;
        rels[best] = Rel();
        uniq.clear();
        continue;
      }
      if (!mat[best]) {
        std::vector<JoinFilter> jfs;
        std::vector<DevVec<uint32_t>> bitmaps;
        bitmaps.reserve(edges.size());
        for (auto& e : edges) {
          const bool aCur = cur.tabs.count(e.at) && e.bt == best, bCur = cur.tabs.count(e.bt) && e.at == best;
          if (!aCur && !bCur) continue;
          const int ct = aCur ? e.at : e.bt, cc = aCur ? e.ac : e.bc, oc = aCur ? e.bc : e.ac;
          // Only worth it when the keys feeding the filter were already reduced (by filters or
          // joins) to under half of their table; a filter built from every key removes nothing.
          if ((double)cur.n > 0.5 * (double)ctx.tabs[ct]->liveRows()) continue;
          const int ri = cur.find(ct, cc);
          int64_t lo, hi;
          if (ri < 0 || cur.cols[ri].pt == PType::F64 || ctx.tabs[best]->cols[oc].pt == PType::F64 ||
              ctx.tabs[ct]->cols[cc].lt == LType::VARCHAR ||  // dictionary codes differ per table
              !colRange(*ctx.tabs[ct], cc, lo, hi) || span64(lo, hi) >= (uint64_t)1 << 30)
            continue;
          const int64_t nbits = (int64_t)span64(lo, hi) + 1;
          bitmaps.emplace_back((nbits + 31) / 32);
          gpuKeyBitmap(viewOf(cur.cols[ri].data.p, cur.cols[ri].pt), cur.n, lo, nbits, bitmaps.back().p);
          jfs.push_back({oc, lo, nbits, bitmaps.back().p,
                         ctx.tabs[best]->cols[oc].name + " IN (" + std::to_string(cur.n) + " keys of " +
                             ctx.tabs[ct]->cols[cc].name + ", " + std::to_string((nbits + 7) / 8 / 1024) + " KiB bitmap)"});
        }
        rels[best] = materialize(ctx, best, pushdown[best], needOf(best), -1, notes, jfs);
        mat[best] = true;
      }
      auto& e = edges[bestEdge];
      edgeUsed[bestEdge] = true;
      bool curHasA = cur.tabs.count(e.at) > 0;
      {
        std::pair<int, int> curCol = curHasA ? std::make_pair(e.at, e.ac) : std::make_pair(e.bt, e.bc);
        int otherCol = curHasA ? e.bc : e.ac;
        bool newUnique = otherCol == ctx.tabs[best]->pk, curUnique = uniq.count(curCol) > 0;
        std::set<std::pair<int, int>> nu;
        if (newUnique) nu = uniq;
        if (curUnique && ctx.tabs[best]->pk >= 0) nu.insert({best, ctx.tabs[best]->pk});
        uniq.swap(nu);
      }
      const Dict* da = ctx.tabs[e.at]->cols[e.ac].lt == LType::VARCHAR ? ctx.tabs[e.at]->dicts[e.ac].get() : nullptr;
      const Dict* db = ctx.tabs[e.bt]->cols[e.bc].lt == LType::VARCHAR ? ctx.tabs[e.bt]->dicts[e.bc].get() : nullptr;
      cur = curHasA ? hashJoin(cur, rels[best], e.at, e.ac, e.bt, e.bc, notes, da, db)
                    : hashJoin(cur, rels[best], e.bt, e.bc, e.at, e.ac, notes, db, da);
      used[best] = true;
      rels[best] = Rel();
    }
    if (prefixN < nt) return buildOrdered(std::move(cur), prefixN, edgeUsed);
    for (size_t ei = 0; ei < edges.size(); ++ei)
      if (!edgeUsed[ei]) residual.push_back(edges[ei].e);
    return cur;
  };

  // ================= Aggregation queries
  if (aggQuery) {
    for (auto& it : items) collectAggs(it.e);
    if (having) collectAggs(having);
    for (auto& o : orderBy) collectAggs(o.e);
    for (auto& g : groupBy)
      if (hasAgg(g)) throw SqlError("Can't group on aggregate", 1056);
    plan.nkeys = (int)groupBy.size();
    for (auto& g : groupBy)
      if (g->ty == LType::VARCHAR && !ctx.dict(*g))
        throw SqlError("GROUP BY on this string expression is not supported (it combines several columns)", 1235);
    if (distX) {  // only COUNT(DISTINCT x) aggregates: each is the count of distinct x per group (slot 0)
      if (distX->ty == LType::VARCHAR && !ctx.dict(*distX))
        throw SqlError("GROUP BY on this string expression is not supported (it combines several columns)", 1235);
      if (ctx.tabs.size() != 1) throw SqlError("internal: COUNT(DISTINCT) GPU path needs one table", 1105);
      if (ciColl() && distX->ty == LType::VARCHAR) {  // case variants are one value
        Dict* d = ctx.dict(*distX);
        if (auto cmap = d ? ciCanon(*d) : nullptr) {
          ExprP m = boundNode(Expr::FUNC, "#MAP", {distX}, LType::VARCHAR);
          m->aux = cmap;
          ctx.exprDict[m.get()] = d;
          distX = m;
        }
      }
      // COUNT(DISTINCT x) reads the extra slot phase 2 adds after the ordinary aggregates.
      for (auto& [k, r] : aggs)
        if (r.distinct) {
          r.fn = "COUNT";
          r.slot = plan.naggs;
          r.cntSlot = 0;
        }
    }

    bool single = ctx.tabs.size() == 1;
    Rel joined;
    if (!single) {
      notes.push_back("GPU join pipeline (EXPLAIN runs the joins to report the actual strategy):");
      joined = buildJoin();
      if (explain) return explainResult();
    }
    ExprP filter = single ? where : andAll(residual);
    Compiler comp(&ctx);
    plan.hasWhere = (bool)filter;
    if (filter) plan.where = comp.compile(filter);
    for (int k = 0; k < plan.nkeys; ++k) {
      // DECIMAL arithmetic runs in doubles: round the key to its scale so equal decimals group together
      // (-30.76 can come out as -30.759999.. for one row and -30.760000..1 for another).
      ExprP kx = groupBy[k];
      if (kx->ty == LType::DECIMAL && kx->kind != Expr::COL && kx->scale >= 0) {
        kx = mkCall(FN_ROUND2, {kx, boundLit(Value::I(kx->scale))}, LType::DECIMAL);
        kx->scale = groupBy[k]->scale;
      }
      plan.keys[k] = comp.compile(kx);
      plan.keyInt[k] = groupBy[k]->ty != LType::DOUBLE && groupBy[k]->ty != LType::DECIMAL && progIsInt(comp.env, plan.keys[k]);
    }
    for (int a = 1; a < plan.naggs; ++a) plan.aggs[a] = comp.compile(aggArgs[a]);
    Prog distProg;
    bool distInt = false;
    if (distX) {
      distProg = comp.compile(distX);
      distInt = distX->ty != LType::DOUBLE && distX->ty != LType::DECIMAL && progIsInt(comp.env, distProg);
    }
    comp.uploadSets();
    plan.aggInt[0] = 1;
    for (int a = 1; a < plan.naggs; ++a) {
      plan.aggInt[a] = plan.ops[a] != AGG_COUNT && progIsInt(comp.env, plan.aggs[a]);
      plan.aggNull[a] = 0;
      if (plan.ops[a] != AGG_COUNT && plan.aggs[a].len == 1 && comp.env.code[plan.aggs[a].start].op == OP_COL) {
        const uint8_t t = comp.env.cols[comp.env.code[plan.aggs[a].start].arg].t;
        if (t == 3 || t == 4) plan.aggNull[a] = t, plan.aggInt[a] = 1;  // nullable int column: exact, NULLs skipped
      }
      // GROUP BY over 64-bit integer columns: exact int64 results as bits (a SUM as its two halves,
      // see aggPart). Without GROUP BY integer sums are 128-bit and exact anyway.
      plan.aggBits[a] = 0;
      if ((plan.nkeys || distX) && plan.aggInt[a] && aggArgs[a] && aggArgs[a]->kind == Expr::COL &&
          (aggArgs[a]->ty == LType::BIGINT || aggArgs[a]->ty == LType::DATETIME) &&
          (plan.ops[a] == AGG_MIN || plan.ops[a] == AGG_MAX || plan.aggPart[a]))
        plan.aggBits[a] = 1;
    }
    for (int a = 1; a < plan.naggs; ++a)  // both halves exact, or neither (then the low slot sums whole values)
      if (plan.aggPart[a] == 1 && plan.aggBits[a] != plan.aggBits[a + 1]) plan.aggBits[a] = plan.aggBits[a + 1] = 0;
    bool anyBits = false;
    for (int a = 1; a < plan.naggs; ++a) anyBits |= plan.aggBits[a] != 0;
    const int ncols = (int)comp.slotCols.size();

    std::vector<double> keys, accs;
    int64_t groups = 0;
    // Size the GPU hash table from statistics: dictionary sizes and segment min/max ranges give
    // an upper bound on the number of groups; fall back to the input row count.
    int64_t rowsIn = single ? ctx.tabs[0]->liveRows() : joined.n;
    double bound = 1;
    for (auto& g : groupBy) {
      if (g->kind != Expr::COL || g->ty == LType::DOUBLE) { bound = 1e18; break; }
      Table& gt = *ctx.tabs[g->bt];
      if (g->ty == LType::VARCHAR) { bound *= std::max<size_t>(gt.dicts[g->bc]->strs.size(), 1); continue; }
      double mn = INFINITY, mx = -INFINITY;
      for (auto& sp : gt.segs) mn = std::min(mn, sp->mn[g->bc]), mx = std::max(mx, sp->mx[g->bc]);
      bound *= mx >= mn ? mx - mn + 1 : 1;
      bound += (double)gt.bufferTotal();
    }
    double est = std::min<double>(bound, (double)std::max<int64_t>(rowsIn, 1));

    // Dense ("encoded") GROUP BY when every key is an integer-like column with a small known range:
    // dictionary codes, small ints, dates. Ranges come from dictionary sizes, segment min/max and
    // the write buffers.
    DenseSpec dspec;
    if (plan.nkeys && tDense) {
      int64_t G = 1;
      bool ok = true;
      for (int k = 0; k < plan.nkeys && ok; ++k) {
        const ExprP& g = groupBy[k];
        if (g->kind != Expr::COL || g->ty == LType::DOUBLE) { ok = false; break; }
        int64_t lo, hi;
        if (ctx.tabs[g->bt]->hasNull[g->bc] || ctx.outerTabs.count(g->bt) || !colRange(*ctx.tabs[g->bt], g->bc, lo, hi)) {
          ok = false;
          break;
        }
        if (span64(lo, hi) >= (uint64_t)1 << 26) { ok = false; break; }
        const int64_t range = (int64_t)span64(lo, hi) + 1;
        if (G > (int64_t)(1ll << 26) / range) { ok = false; break; }
        dspec.kmin[k] = lo;
        dspec.range[k] = range;
        dspec.stride[k] = G;
        G *= range;
      }
      if (ok) {
        int off = 0;
        for (int a = 0; a < plan.naggs; ++a) {
          const uint8_t op = plan.ops[a];
          const bool in = plan.aggInt[a];
          dspec.kind[a] = op == AGG_COUNT ? DK_COUNT
                        : op == AGG_SUM   ? (in ? DK_SUM_I : DK_SUM_D)
                        : op == AGG_MIN   ? (in ? DK_MIN_I : DK_MIN_D)
                                          : (in ? DK_MAX_I : DK_MAX_D);
          dspec.slotOff[a] = off;
          int64_t bytes = G * (dspec.kind[a] == DK_COUNT ? 4 : 8);
          off += (int)std::min<int64_t>((bytes + 7) / 8 * 8, 1 << 30);
        }
        const int64_t copyBytes = off;
        // Shared-memory atomics on a handful of hot groups serialize, and FP64 atomics are
        // compare-and-swap loops on consumer GPUs: with few groups and floating-point sums the
        // warp-sort hash path (no atomics per row) is faster (Q1: 13 ms vs 42 ms).
        bool intOnly = true;
        for (int a = 0; a < plan.naggs; ++a)
          intOnly &= dspec.kind[a] == DK_COUNT || dspec.kind[a] == DK_SUM_I || dspec.kind[a] == DK_MIN_I ||
                     dspec.kind[a] == DK_MAX_I;
        if (G * plan.naggs <= 88) {  // 88 x 8 B x 128 threads = 88 KB, leaving room for static shared memory
          // Few groups: per-thread private accumulators (no atomics per row, see aggDensePrivBody).
          dspec.priv = 1;
          dspec.G = G;
        } else if (copyBytes <= 48 * 1024 && G < 64 && !intOnly) {
          // leave dspec.G = 0: hash aggregation
        } else if (copyBytes <= 48 * 1024) {
          dspec.copies = (int)std::min<int64_t>(8, (48 * 1024) / copyBytes);
          dspec.copyBytes = (int)copyBytes;
          dspec.G = G;
        } else if (G <= std::max<int64_t>(4 * rowsIn, 1 << 16) &&
                   G * (8 + 8 * plan.nkeys + 8 * plan.naggs) <= (int64_t)512 << 20) {
          dspec.copies = 0;  // straight to global memory (still no hashing)
          dspec.G = G;
        }
      }
    }
    int64_t cap = 1;
    if (plan.nkeys) {
      cap = 1024;
      int64_t maxCap = (1ll << 30) / (8 + 8 * plan.nkeys + 8 * plan.naggs);  // keep the table <= 1 GiB
      while (cap < est * 2 && cap * 2 <= maxCap) cap <<= 1;
    }
    // HAVING / ORDER BY .. LIMIT over many groups: evaluate on the GPU over the compacted groups
    // and download only the surviving (top-N) groups. The host code below then re-checks and
    // orders that small set exactly. Returns false (host fallback) for anything it can't compile.
    auto gpuPostAggregate = [&](GpuAggregator& agg, std::vector<double>& hk, std::vector<double>& ha, int64_t& out) -> bool {
      std::vector<std::string> keyCanon;
      for (auto& g : groupBy) keyCanon.push_back(canon(g));
      Compiler pc(&ctx);
      const int nk = plan.nkeys;
      pc.hook = [&](const ExprP& e) {
        std::string c = canon(e);
        for (int k = 0; k < nk; ++k)
          if (keyCanon[k] == c) { pc.emit(OP_COL, k, 1); return true; }
        if (e->kind == Expr::FUNC && isAggName(e->op)) {
          const AggRef& r = aggs.at(c);
          if (r.fn == "COUNT") pc.emit(OP_COL, nk + r.slot, 1);
          else if (r.fn == "AVG") { pc.emit(OP_COL, nk + r.slot, 1); pc.emit(OP_COL, nk + r.cntSlot, 1); pc.emit(OP_DIV, 0, -1); }
          else if (r.argTy == LType::VARCHAR && !r.dict) throw SqlError("string aggregate", 1235);
          else {
            pc.emit(OP_COL, nk + r.slot, 1);
            if (r.decScale >= 0) {
              pc.emit(OP_CONST, pc.constant(pow10i(r.decScale)), 1);
              pc.emit(OP_DIV, 0, -1);
            }
            // No non-NULL value in the group: NULL (the slot holds the accumulator's start value):
            // + 0 / count, NaN when the count is 0.
            pc.emit(OP_CONST, pc.constant(0.0), 1);
            pc.emit(OP_COL, nk + r.cntSlot, 1);
            pc.emit(OP_DIV, 0, -1);
            pc.emit(OP_ADD, 0, -1);
          }
          return true;
        }
        if (e->kind == Expr::COL) throw SqlError("not a group key", 1055);
        return false;
      };
      Prog hp;
      std::vector<Prog> ops;
      bool topN = !orderBy.empty() && s.limit >= 0;
      const bool firstN = orderBy.empty() && s.limit >= 0 && !s.distinct;
      if (!having && !topN && !firstN) return false;
      try {
        if (having) hp = pc.compile(having);
        if (topN)
          for (auto& o : orderBy) {
            if (o.e->ty == LType::VARCHAR && !ctx.dict(*o.e)) return false;
            ops.push_back(pc.compile(o.e));
          }
      } catch (const SqlError&) {
        return false;
      }
      DevVec<double> dk, da;
      const int64_t g = agg.compact(dk, da);
      std::vector<DevVec<double>> keyConv;  // integer keys (stored as int64 bits) as doubles for evaluation
      for (int k = 0; k < nk; ++k) {
        if (plan.keyInt[k]) {
          keyConv.emplace_back(std::max<int64_t>(g, 1));
          gpuBitsToDouble(dk.p + (int64_t)k * g, g, keyConv.back().p);
          pc.env.cols[k] = viewOf(keyConv.back().p, PType::F64);
        } else {
          pc.env.cols[k] = viewOf(dk.p + (int64_t)k * g, PType::F64);
        }
      }
      // Exact int64 slots (AggPlan::aggBits: SUM / MIN / MAX of integers) hold int64 bit patterns:
      // as doubles for the evaluation, like the integer keys.
      const auto bitsSlots = agg.sumBitsSlots();
      std::vector<DevVec<double>> accConv;
      accConv.reserve(plan.naggs);
      for (int a = 0; a < plan.naggs; ++a) {
        if (a < (int)bitsSlots.size() && bitsSlots[a]) {
          accConv.emplace_back(std::max<int64_t>(g, 1));
          gpuBitsToDouble(da.p + (int64_t)a * g, g, accConv.back().p);
          pc.env.cols[nk + a] = viewOf(accConv.back().p, PType::F64);
        } else {
          pc.env.cols[nk + a] = viewOf(da.p + (int64_t)a * g, PType::F64);
        }
      }
      pc.uploadSets();
      DevVec<int32_t> sel;
      int64_t m = g;
      if (having) {
        sel = DevVec<int32_t>(std::max<int64_t>(g, 1));
        m = gpuSelect(pc.env, true, hp, g, nullptr, sel.p);
      }
      DevVec<int32_t> cand;
      int64_t mc = m;
      if (topN && m > 0) {
        mc = std::min<int64_t>(m, s.offset + s.limit);
        std::vector<DevVec<double>> bufs;
        std::vector<std::shared_ptr<DevVec<int32_t>>> ranks;
        std::vector<std::pair<const double*, bool>> sk;
        for (size_t k = 0; k < orderBy.size(); ++k) {
          bufs.emplace_back(m);
          gpuEvalAt(pc.env, ops[k], having ? sel.p : nullptr, m, bufs.back().p);
          const ExprP& oe = orderBy[k].e;
          bool rankAlready = false;  // string MIN / MAX: the slot holds a rank already (not a code)
          if (oe->kind == Expr::FUNC && isAggName(oe->op))
            if (auto it = aggs.find(canon(oe)); it != aggs.end() && it->second.unrank) rankAlready = true;
          if (oe->ty == LType::VARCHAR && !rankAlready) {
            ranks.push_back(dictRankDev(*ctx.dict(*orderBy[k].e)));
            gpuMapRank(bufs.back().p, ranks.back()->p, m);
          }
          sk.push_back({bufs.back().p, orderBy[k].desc});
        }
        DevVec<int32_t> perm = gpuSortPerm(sk, m);
        cand = DevVec<int32_t>(std::max<int64_t>(mc, 1));
        if (having) gpuGather(sel.p, PType::I32, perm.p, mc, cand.p);
        else d2d(cand.p, perm.p, mc * 4);
      } else if (firstN) {  // LIMIT without ORDER BY: any offset + limit groups do
        mc = std::min<int64_t>(m, s.offset + s.limit);
        if (having) {
          cand = std::move(sel);
        } else {
          std::vector<int32_t> first((size_t)std::max<int64_t>(mc, 1));
          for (int64_t i = 0; i < mc; ++i) first[i] = (int32_t)i;
          cand = DevVec<int32_t>((int64_t)first.size());
          h2d(cand.p, first.data(), first.size() * 4);
        }
      } else {
        cand = std::move(sel);
      }
      // Download the candidate groups in one transfer, then lay them out row-major like fetch().
      const int W = nk + plan.naggs;
      DevVec<double> tmp(std::max<int64_t>(mc * W, 1));
      for (int c = 0; c < W; ++c)
        gpuGather((c < nk ? dk.p + (int64_t)c * g : da.p + (int64_t)(c - nk) * g), PType::F64, cand.p, mc, tmp.p + (int64_t)c * mc);
      const std::vector<double> all = tmp.download(mc * W);
      hk.assign((size_t)mc * nk, 0.0);
      ha.assign((size_t)mc * plan.naggs, 0.0);
      for (int c = 0; c < W; ++c)
        for (int64_t i = 0; i < mc; ++i) {
          const double v = all[(size_t)c * mc + i];
          if (c < nk) hk[(size_t)i * nk + c] = v;
          else ha[(size_t)i * plan.naggs + (c - nk)] = v;
        }
      out = mc;
      notes.push_back("  GPU post-aggregation: " + std::to_string(g) + " groups -> " + std::to_string(mc) + " sent to host");
      return true;
    };
    const std::string method =
        !plan.nkeys ? "no GROUP BY"
        : dspec.G   ? "dense GROUP BY on encoded keys (" + std::to_string(dspec.G) + " slots, " +
                        (dspec.priv ? std::string("per-thread accumulators") : dspec.copies ? std::to_string(dspec.copies) + " shared-memory copies per block" : "global arrays") + ")"
                    : "hash GROUP BY";
    const JitProgram* jit = nullptr;
    std::string codegen = jitEnabled() ? "compiled" : "interpreted (SET warpstore_codegen = OFF)";
    if (!explain && !distX) {
      ProfStage pcg("codegen");
      const JitFn fn = !plan.nkeys ? JF_NOKEYS : dspec.G ? JF_DENSE : JF_GROUPED;
      plan.densePriv = dspec.G ? (dspec.priv ? 1 : 2) : 0;
      // Plans beyond the interpreter's limits must be compiled before they can run.
      const bool mustCompile = plan.naggs > kInterpAggs || plan.nkeys > kInterpKeys;
      const JitMode savedMode = jitMode();
      if (mustCompile && savedMode != JIT_SYNC) jitSetMode(JIT_SYNC);
      jit = jitGet(comp.env, ncols, plan, fn,
                   !plan.nkeys ? "aggregate (no group by)" : dspec.G ? "aggregate (dense group by)" : "aggregate (hash group by)");
      if (mustCompile && savedMode != JIT_SYNC) jitSetMode(savedMode);
      if (jit) codegen = "compiled plan #" + std::to_string(jit->id);
      else if (jitEnabled()) codegen = "interpreted while the plan compiles in the background";
      pcg.done(-1, codegen);
    }
    ScanStats st;
    // COUNT(DISTINCT x): phase 1 groups by (keys, x) on the GPU, phase 2 aggregates the compacted
    // distinct pairs by the keys (COUNT, skipping NULL x), still on the GPU.
    std::unique_ptr<GpuAggregator> agg2;
    std::vector<uint8_t> sumBits;  // slots holding exact sums as bits (no GROUP BY)
    std::vector<int64_t> sumHi;    // their high 64 bits
    DevVec<double> distKeys, distAccs;
    if (distX && !explain) {
      AggPlan plan1 = plan;  // every ordinary aggregate as a partial per (keys, x)
      plan1.nkeys = plan.nkeys + 1;
      plan1.keys[plan.nkeys] = distProg;
      plan1.keyInt[plan.nkeys] = distInt;
      plan1.densePriv = 0;
      if (plan.naggs + 1 > kInterpAggs) throw SqlError("too many aggregates next to COUNT(DISTINCT)", 1235);
      if (plan1.nkeys > kMaxKeys) throw SqlError("too many GROUP BY expressions", 1235);
      const JitProgram* jit1 = nullptr;
      {
        ProfStage pcg("codegen");
        jit1 = jitGet(comp.env, ncols, plan1, JF_GROUPED, "aggregate (distinct pairs)");
        pcg.done(-1, jit1 ? "compiled plan #" + std::to_string(jit1->id) : "interpreted");
      }
      int64_t cap1 = 1024;
      while (cap1 < std::min<int64_t>(std::max<int64_t>(rowsIn, 1), 1 << 22)) cap1 <<= 1;
      for (;;) {
        GpuAggregator a1(plan1, cap1, nullptr);
        ProfStage p1("scan+distinct pairs " + ctx.tabs[0]->name);
        st = ScanStats();
        forEachBatch(ctx, 0, comp, conj, st, [&](ExprEnv& env, int64_t n, const uint32_t* del) {
          a1.addBatch(env, ncols, n, del);
          if (scanWindowFull()) scanFlush(), a1.runWindow(comp.env, jit1), releaseScanTemps();  // (disk tier)
          return true;
        });
        a1.run(comp.env, jit1);
        if (a1.overflowed()) {
          int64_t want = std::max<int64_t>(cap1 * 4, a1.groupCount() * 4);
          cap1 = 1;
          while (cap1 < want) cap1 <<= 1;
          continue;
        }
        const int64_t g1 = a1.compact(distKeys, distAccs);
        p1.done(a1.rowsIn(), std::to_string(g1) + " distinct (group, value) pairs, hash GROUP BY");
        ProfStage p2("count distinct");
        const int nk = plan.nkeys;
        ExprEnv env2{};
        int nc = 0;
        for (int k = 0; k <= nk; ++k) {
          const bool in = k < nk ? plan.keyInt[k] != 0 : distInt;
          env2.cols[k] = ColView{distKeys.p + (int64_t)k * g1, (uint8_t)(in ? 1 : 2)};
          env2.code[nc++] = Instr{OP_COL, (uint8_t)in, (uint16_t)k};
        }
        AggPlan plan2{};
        plan2.nkeys = nk;
        for (int k = 0; k < nk; ++k) {
          plan2.keys[k] = Prog{(int16_t)k, 1};
          plan2.keyInt[k] = plan.keyInt[k];
        }
        // Combine the partials: COUNT and SUM by SUM, MIN by MIN, MAX by MAX; the extra slot counts
        // the pairs whose x is not NULL (= COUNT(DISTINCT x) per group).
        const int na = plan1.naggs;
        for (int a = 0; a < na; ++a) {
          // (exact int64 partials, AggPlan::aggBits, are read as int64 and merged exactly)
          env2.cols[nk + 1 + a] = ColView{distAccs.p + (int64_t)a * g1, (uint8_t)(plan1.aggBits[a] ? 1 : 2)};
          plan2.aggs[a] = Prog{(int16_t)nc, 1};
          env2.code[nc++] = Instr{OP_COL, 0, (uint16_t)(nk + 1 + a)};
          plan2.ops[a] = plan1.ops[a] == AGG_COUNT ? AGG_SUM : plan1.ops[a];
          plan2.aggInt[a] = plan2.aggBits[a] = plan1.aggBits[a];
          plan2.aggPart[a] = 0;
        }
        plan2.aggs[na] = Prog{(int16_t)nc, distInt ? (int16_t)1 : (int16_t)3};
        if (distInt) {
          env2.code[nc++] = Instr{OP_CONST, 0, 0};
          env2.consts[0] = 1.0;
        } else {
          env2.code[nc++] = Instr{OP_COL, 0, (uint16_t)nk};
          env2.code[nc++] = Instr{OP_ISNULL, 0, 0};
          env2.code[nc++] = Instr{OP_NOT, 0, 0};
        }
        plan2.ops[na] = AGG_SUM;
        plan2.aggInt[na] = 0;
        plan2.naggs = na + 1;
        int64_t cap2 = 1024;
        while (cap2 < 2 * g1) cap2 <<= 1;
        agg2 = std::make_unique<GpuAggregator>(plan2, cap2, nullptr);
        agg2->addBatch(env2, nk + 1 + na, g1, nullptr);
        agg2->run(env2, nullptr);
        agg2->overflowed();
        p2.done(g1, "second GPU aggregation over the distinct pairs");
        break;
      }
      plan.naggs += 1;  // the result layout is plan2's: ordinary slots + the distinct count
      ProfStage ppost("post-aggregate");
      if (!(plan.nkeys && (having || s.limit >= 0) && gpuPostAggregate(*agg2, keys, accs, groups)))
        groups = agg2->fetch(keys, accs);
      sumBits = agg2->sumBitsSlots();
      sumHi = agg2->sumHigh();
      ppost.done(groups, plan.nkeys ? "groups sent to host" : "");
    }
    for (;;) {
      if (distX) break;
      GpuAggregator agg(plan, cap, dspec.G ? &dspec : nullptr);
      ProfStage pscan(single ? "scan+aggregate " + ctx.tabs[0]->name : "aggregate join result");
      if (single) {
        st = ScanStats();
        if (explain) {
          forEachBatch(ctx, 0, comp, conj, st, nullptr, true);
        } else {
          forEachBatch(ctx, 0, comp, conj, st, [&](ExprEnv& env, int64_t n, const uint32_t* del) {
            agg.addBatch(env, ncols, n, del);
            if (scanWindowFull()) scanFlush(), agg.runWindow(comp.env, jit), releaseScanTemps();  // (disk tier)
            return true;
          });
        }
      } else {
        bindRelCols(comp, joined);
        agg.addBatch(comp.env, ncols, joined.n, nullptr);
      }
      if (explain) break;
      agg.run(comp.env, jit);
      const bool overflow = agg.overflowed();
      pscan.done(agg.rowsIn(), std::to_string(agg.batches()) + " batches in " + std::to_string(agg.windows() + 1) + (agg.windows() ? " launches (disk tier windows), " : " launch, ") + method + ", " + codegen +
                                   (overflow ? ", hash table overflowed: retrying bigger" : ""));
      if (overflow) {
        int64_t want = std::max<int64_t>(cap * 4, agg.groupCount() * 4);
        cap = 1;
        while (cap < want) cap <<= 1;
        continue;
      }
      ProfStage ppost("post-aggregate");
      // A derived table's GROUP BY whose items are keys and plain aggregates: built on the GPU from the
      // compacted groups, never leaving VRAM (makeTemp takes the columns over).
      if (tColumnar && plan.nkeys && !having && orderBy.empty() && s.limit < 0 && s.offset == 0 && !explain) {
        std::vector<TempColSpec> specs;
        std::vector<std::shared_ptr<Dict>> dicts(items.size());
        bool ok = true;
        for (size_t i = 0; i < items.size() && ok; ++i) {
          const ExprP& e = items[i].e;
          TempColSpec sp;
          const LType ty = e->ty == LType::DECIMAL ? LType::DOUBLE : e->ty;
          if (ty == LType::VECTOR) { ok = false; break; }
          sp.out = ty == LType::DOUBLE ? 2 : physOf(ty) == PType::I64 ? 1 : 0;
          const std::string k = canon(e);
          int key = -1;
          for (int g = 0; g < plan.nkeys; ++g)
            if (canon(groupBy[g]) == k) key = g;
          if (key >= 0) {
            if (ty == LType::VARCHAR) {  // the derived table shares the dictionary (it only grows), or a copy
              const Dict* d = ctx.dict(*e);
              if (!d) { ok = false; break; }
              if (e->kind == Expr::COL && e->bt >= 0) dicts[i] = ctx.tabs[e->bt]->dicts[e->bc];
              for (auto& sd : ctx.sharedDicts)
                if (!dicts[i] && sd.get() == d) dicts[i] = sd;
              for (auto& tb : ctx.tabs)  // (e.g. canonical case-insensitive codes of a column: its dictionary)
                for (auto& td : tb->dicts)
                  if (!dicts[i] && td.get() == d) dicts[i] = td;
              if (!dicts[i]) {
                if (d->strs.size() > (1u << 20)) { ok = false; break; }
                dicts[i] = std::make_shared<Dict>(*d);
              }
            }
            sp.src = 0;
            sp.idx = key;
            sp.keyInt = plan.keyInt[key];
          } else if ((e->kind == Expr::FUNC && isAggName(e->op) && ty != LType::VARCHAR) ||
                     (e->kind == Expr::BIN && e->op == "*" && ty == LType::DOUBLE &&
                      ((e->args[0]->kind == Expr::LIT && e->args[1]->kind == Expr::FUNC && isAggName(e->args[1]->op)) ||
                       (e->args[1]->kind == Expr::LIT && e->args[0]->kind == Expr::FUNC && isAggName(e->args[0]->op))))) {
            ExprP ag = e;
            if (e->kind == Expr::BIN) {  // constant * aggregate
              const bool litFirst = e->args[0]->kind == Expr::LIT;
              const Value& lv = e->args[litFirst ? 0 : 1]->lit;
              if (lv.isNull() || lv.kind == Value::STR) { ok = false; break; }
              sp.mul = lv.num();
              ag = e->args[litFirst ? 1 : 0];
            }
            auto it = aggs.find(canon(ag));
            if (it == aggs.end() || it->second.unrank || it->second.distinct || it->second.dict) { ok = false; break; }
            const AggRef& r = it->second;
            const bool bits = plan.aggBits[r.slot];
            sp.src = 1;
            sp.idx = r.slot;
            sp.cnt = r.fn == "COUNT" ? -1 : r.cntSlot;
            sp.div = r.decScale >= 0 ? pow10i(r.decScale) : 1.0;
            if (r.fn == "COUNT") sp.kind = 3;
            else if (r.fn == "AVG") sp.kind = 4, sp.sumBits = bits, sp.hi = bits ? r.hiSlot : -1;
            else if (bits) sp.kind = r.hiSlot >= 0 ? 2 : 1, sp.hi = r.hiSlot;
            else sp.kind = 0;
          } else {
            ok = false;
          }
          specs.push_back(sp);
        }
        if (ok) {
          DevVec<double> dk, da;
          const int64_t g = agg.compact(dk, da);
          for (auto& sp : specs) {
            auto col = std::make_shared<ResultSet::DevColumn>();
            col->p = gpuTempColumn(dk.p, plan.nkeys, da.p, plan.naggs, g, sp, col->mn, col->mx);
            res.devCols.push_back(col);
          }
          res.devRows = g;
          res.colDicts = dicts;
          for (auto& t : res.types)
            if (t == LType::DECIMAL) t = LType::DOUBLE;
          res.isResult = true;
          res.typed = true;
          ppost.done(g, "derived table built on the GPU");
          res.info = single ? "gpu-aggregate" : "gpu-join-aggregate";
          return res;
        }
      }
      if (!(plan.nkeys && !anyBits && (having || s.limit >= 0) &&
            gpuPostAggregate(agg, keys, accs, groups)))
        groups = agg.fetch(keys, accs);
      sumBits = agg.sumBitsSlots();
      sumHi = agg.sumHigh();
      ppost.done(groups, plan.nkeys ? "groups sent to host" : "");
      break;
    }
    notes.insert(notes.begin(), std::string(single ? "GPU fused scan+filter+aggregate" : "GPU aggregate") + ": " + method +
                                    ", " + std::to_string(plan.naggs) + " agg slots, codegen " +
                                    (jitEnabled() ? "on" : "off"));
    if (single) {
      notes.insert(notes.begin() + 1, "  " + scanDesc(*ctx.tabs[0], st));
      if (where) notes.insert(notes.begin() + 2, "  filter: " + where->str());
    }
    if (explain) return explainResult();
    ProfStage phost("host finalize");

    // Non-NULL counts (SUM(ISNOTNULL x) slots) are read as plain doubles everywhere: convert their exact bits.
    for (size_t a = 1; a < sumBits.size() && a < aggArgs.size() && a < accs.size(); ++a)
      if (sumBits[a] && aggArgs[a] && aggArgs[a]->kind == Expr::FUNC && aggArgs[a]->op == "ISNOTNULL")
        accs[a] = (double)bitsOf(accs[a]), sumBits[a] = 0;
    PostCtx pc;
    pc.bc = &ctx;
    pc.aggs = &aggs;
    if (!sumBits.empty()) pc.accBits = sumBits.data();
    if (!sumHi.empty() && !plan.nkeys) pc.accHi = sumHi.data();
    for (int k = 0; k < plan.nkeys; ++k) pc.keyInt.push_back(plan.keyInt[k]);
    for (auto& g : groupBy) {
      pc.keyCanon.push_back(canon(g));
      pc.keyTy.push_back(g->ty);
      pc.keyDict.push_back(ctx.dict(*g));
      pc.keyScale.push_back(g->scale);
    }
    // Filter groups, compute ORDER BY keys, pick the top rows, and only then evaluate the
    // select list (keeps large GROUP BY ... ORDER BY .. LIMIT cheap on the host).
    // On all cores (millions of groups when a GROUP BY feeds a derived table): each chunk with its
    // own evaluation context, results concatenated in group order.
    std::vector<int64_t> keep;
    std::vector<std::vector<Value>> sortKeys;
    // Every fetched group is kept and in order (no HAVING / ORDER BY, compacted groups): no index arrays.
    bool identity = !having && orderBy.empty() && plan.nkeys;
    for (int64_t gi = 0; identity && gi < groups; gi += std::max<int64_t>(1, groups / 64))
      identity = accs[gi * plan.naggs] != 0;
    // ORDER BY a string GROUP BY key: compare its rank in the sorted dictionary, not the strings.
    std::vector<std::shared_ptr<std::vector<double>>> rankOf(orderBy.size());
    std::vector<int> rankKey(orderBy.size(), -1);
    for (size_t k = 0; k < orderBy.size(); ++k) {
      const ExprP& oe = orderBy[k].e;
      if (oe->ty != LType::VARCHAR || !ctx.dict(*oe)) continue;
      const std::string oc = canon(oe);
      for (int g = 0; g < plan.nkeys; ++g)
        if (pc.keyCanon[g] == oc && plan.keyInt[g]) {
          rankKey[k] = g;
          rankOf[k] = sortedDict(*ctx.dict(*oe)).second;
        }
    }
    ProfStage psub("  finalize: setup");
    psub.done();
    if (!identity) {
      const size_t nchunks = std::max<size_t>(1, std::min<size_t>(64, (size_t)groups / 16384));
      std::vector<std::vector<int64_t>> keepC(nchunks);
      std::vector<std::vector<std::vector<Value>>> skC(nchunks);
      std::exception_ptr err;
      std::mutex errMu;
      parallelFor(nchunks, [&](size_t cb, size_t ce) {
        for (size_t ch = cb; ch < ce; ++ch) {
          try {
            PostCtx tpc = pc;
            const int64_t lo = (int64_t)(groups * ch / nchunks), hi = (int64_t)(groups * (ch + 1) / nchunks);
            for (int64_t gi = lo; gi < hi; ++gi) {
              tpc.keys = keys.data() + gi * plan.nkeys;
              tpc.accs = accs.data() + gi * plan.naggs;
              if (plan.nkeys && tpc.accs[0] == 0) continue;
              if (having && !isTrue(evalPost(having, tpc))) continue;
              keepC[ch].push_back(gi);
              if (!orderBy.empty()) {
                std::vector<Value> sk;
                for (size_t k = 0; k < orderBy.size(); ++k) {
                  if (rankOf[k]) {
                    const long long code = bitsOf(tpc.keys[rankKey[k]]);
                    sk.push_back(code < 0 || code >= (long long)rankOf[k]->size() ? Value::Null() : Value::D((*rankOf[k])[code]));
                    continue;
                  }
                  sk.push_back(evalPost(orderBy[k].e, tpc));
                }
                skC[ch].push_back(std::move(sk));
              }
            }
          } catch (...) {
            std::lock_guard<std::mutex> g(errMu);
            err = std::current_exception();
          }
        }
      }, 1);
      if (err) std::rethrow_exception(err);
      for (size_t ch = 0; ch < nchunks; ++ch) {
        keep.insert(keep.end(), keepC[ch].begin(), keepC[ch].end());
        for (auto& k : skC[ch]) sortKeys.push_back(std::move(k));
      }
    }
    // (identity: verify the sampled assumption on all cores, falling back to the index arrays)
    if (identity) {
      std::atomic<bool> empty{false};
      parallelFor((size_t)groups, [&](size_t b, size_t e) {
        for (size_t gi = b; gi < e && !empty; ++gi)
          if (accs[gi * plan.naggs] == 0) empty = true;
      });
      if (empty) {
        identity = false;
        for (int64_t gi = 0; gi < groups; ++gi)
          if (accs[gi * plan.naggs] != 0) keep.push_back(gi);
      }
    }
    const size_t nkeep = identity ? (size_t)groups : keep.size();
    psub.stage = "  finalize: filter + sort keys";
    psub.done(nkeep);
    // A VARCHAR GROUP BY key can be a column of dictionary codes in the columnar result (its dictionary,
    // when small, is copied into the temp table).
    auto codeKey = [&](const ExprP& e) {
      if (e->ty != LType::VARCHAR) return false;
      const Dict* d = ctx.dict(*e);
      if (!d || d->strs.size() > (1u << 20)) return false;
      const std::string k = canon(e);
      for (auto& kc : pc.keyCanon)
        if (kc == k) return true;
      return false;
    };
    bool columnarOk = tColumnar && s.limit < 0 && s.offset == 0;  // (the condition of the columnar result below)
    for (auto& it : items) columnarOk = columnarOk && (it.e->ty != LType::VARCHAR || codeKey(it.e)) && it.e->ty != LType::VECTOR;
    if (identity && !columnarOk) {  // the general paths below index through keep
      keep.resize(nkeep);
      for (size_t i = 0; i < nkeep; ++i) keep[i] = (int64_t)i;
      identity = false;
    }
    std::vector<size_t> order(identity ? 0 : keep.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = i;
    size_t want = s.limit < 0 ? nkeep : std::min<size_t>(nkeep, (size_t)(s.offset + s.limit));
    if (!orderBy.empty()) {
      // All-numeric keys (numbers, dates, string ranks): compared as one flat array of doubles.
      // String keys: their distinct values sorted once (collation order), compared by rank.
      const size_t K = orderBy.size();
      bool flat = true;
      std::vector<std::unordered_map<std::string, double>> strRank(K);
      for (size_t k = 0; k < K && flat; ++k) {
        bool str = false, num = false;
        for (auto& sk : sortKeys) {
          const Value& v = sk[k];
          if (v.isNull()) continue;
          if (v.kind == Value::STR) str = true;
          else num = true;
          if (v.kind != Value::DBL && v.kind != Value::STR && std::llabs(v.i) > (1ll << 53)) flat = false;
        }
        if (str && num) flat = false;
        if (!str || !flat) continue;
        std::vector<const Value*> u;
        for (auto& sk : sortKeys)
          if (!sk[k].isNull() && strRank[k].emplace(sk[k].s, 0).second) u.push_back(&sk[k]);
        std::sort(u.begin(), u.end(), [](const Value* a, const Value* b) { return cmpVal(*a, *b) < 0; });
        double r = 0;
        for (size_t i = 0; i < u.size(); ++i) {
          if (i && cmpVal(*u[i - 1], *u[i]) != 0) ++r;  // (equal under the collation: same rank)
          strRank[k][u[i]->s] = r;
        }
      }
      std::vector<double> fk;
      if (flat) {
        fk.resize(sortKeys.size() * K);
        for (size_t i = 0; i < sortKeys.size(); ++i)
          for (size_t k = 0; k < K; ++k) {
            const Value& v = sortKeys[i][k];
            fk[i * K + k] = v.isNull() ? NAN : v.kind == Value::STR ? strRank[k][v.s] : v.num();
          }
      }
      auto less = [&](size_t a, size_t b) {
        if (flat) {
          const double* x = &fk[a * K];
          const double* y = &fk[b * K];
          for (size_t k = 0; k < K; ++k) {
            const bool nx = x[k] != x[k], ny = y[k] != y[k];
            const int c = nx || ny ? (nx == ny ? 0 : nx ? -1 : 1) : (x[k] < y[k] ? -1 : x[k] > y[k] ? 1 : 0);  // (NULLs first)
            if (c) return orderBy[k].desc ? c > 0 : c < 0;
          }
          return a < b;
        }
        for (size_t k = 0; k < K; ++k) {
          int c = cmpVal(sortKeys[a][k], sortKeys[b][k]);
          if (c) return orderBy[k].desc ? c > 0 : c < 0;
        }
        return a < b;
      };
      if (want * 4 >= order.size()) {
        std::sort(order.begin(), order.end(), less);
      } else {
        std::partial_sort(order.begin(), order.begin() + want, order.end(), less);
      }
    }
    psub.stage = "  finalize: sort";
    psub.done();
    std::vector<std::vector<Value>> rows, noKeys;
    bool allNumeric = tColumnar && orderBy.empty() && !having && s.limit < 0 && s.offset == 0;
    for (auto& it : items) allNumeric = allNumeric && (it.e->ty != LType::VARCHAR || codeKey(it.e)) && it.e->ty != LType::VECTOR;
    if (allNumeric) {  // a derived table's GROUP BY: numeric columns, no row objects
      const size_t nr = want;
      res.numCols.resize(items.size());
      parallelFor(items.size(), [&](size_t b, size_t e) {  // (zero-filling 10s of MB: one column per core)
        for (size_t i = b; i < e; ++i) res.numCols[i].resize(nr);
      }, 1);
      // Items that are just a GROUP BY key or a plain aggregate read the result arrays directly.
      std::vector<int> keyOf(items.size(), -1);
      std::vector<const AggRef*> aggOf(items.size(), nullptr);
      for (size_t i = 0; i < items.size(); ++i) {
        const std::string k = canon(items[i].e);
        for (size_t g = 0; g < pc.keyCanon.size(); ++g)
          if (pc.keyCanon[g] == k) keyOf[i] = (int)g;
        if (keyOf[i] < 0 && items[i].e->kind == Expr::FUNC && isAggName(items[i].e->op)) {
          auto it = aggs.find(k);
          if (it != aggs.end() && !it->second.unrank && !it->second.distinct) aggOf[i] = &it->second;
        }
      }
      std::exception_ptr err;
      std::mutex errMu;
      parallelFor(nr, [&](size_t b, size_t e) {
        try {
          PostCtx tpc = pc;
          for (size_t j = b; j < e; ++j) {
            const int64_t g = identity ? (int64_t)j : keep[order[j]];
            tpc.keys = keys.data() + g * plan.nkeys;
            tpc.accs = accs.data() + g * plan.naggs;
            for (size_t i = 0; i < items.size(); ++i) {
              if (keyOf[i] >= 0) {
                const double kv = tpc.keys[keyOf[i]];
                res.numCols[i][j] = (size_t)keyOf[i] < pc.keyInt.size() && pc.keyInt[keyOf[i]] ? (double)bitsOf(kv) : kv;  // (NULL keys: the sentinel round-trips)
                continue;
              }
              if (const AggRef* r = aggOf[i]) {
                if (r->fn == "COUNT") { res.numCols[i][j] = tpc.accs[r->slot]; continue; }
                const double cnt = tpc.accs[r->cntSlot];
                double v = tpc.accs[r->slot];
                if (pc.accBits && pc.accBits[r->slot]) {  // exact int64 bits (a BIGINT SUM: two 32-bit halves)
                  v = r->hiSlot >= 0 && pc.accBits[r->hiSlot] ? (double)bitsOf(tpc.accs[r->hiSlot]) * 4294967296.0 + (double)bitsOf(v)
                    : pc.accHi && (r->fn == "SUM" || r->fn == "AVG") ? (double)(((__int128)pc.accHi[r->slot] << 64) | (uint64_t)bitsOf(v))
                                                                    : (double)bitsOf(v);
                }
                if (r->decScale >= 0) v /= pow10i(r->decScale);
                res.numCols[i][j] = cnt == 0 ? nullv() : r->fn == "AVG" ? v / cnt : v;
                continue;
              }
              const Value v = evalPost(items[i].e, tpc);
              res.numCols[i][j] = v.isNull() ? nullv() : v.kind == Value::DBL ? v.d : (double)v.i;
            }
          }
        } catch (...) {
          std::lock_guard<std::mutex> g(errMu);
          err = std::current_exception();
        }
      }, 16384);
      if (err) std::rethrow_exception(err);
      res.typed = true;
      for (auto& t : res.types)  // (a DECIMAL expression's double value: a DOUBLE column of the temp table)
        if (t == LType::DECIMAL) t = LType::DOUBLE;
      res.colDicts.assign(items.size(), nullptr);
      for (size_t i = 0; i < items.size(); ++i)
        if (items[i].e->ty == LType::VARCHAR) {  // (shared when it is a column's or a cached derived one)
          const ExprP& e = items[i].e;
          const Dict* d = ctx.dict(*e);
          if (e->kind == Expr::COL && e->bt >= 0) res.colDicts[i] = ctx.tabs[e->bt]->dicts[e->bc];
          for (auto& sd : ctx.sharedDicts)
            if (!res.colDicts[i] && sd.get() == d) res.colDicts[i] = sd;
          for (auto& tb : ctx.tabs)
            for (auto& td : tb->dicts)
              if (!res.colDicts[i] && td.get() == d) res.colDicts[i] = td;
          if (!res.colDicts[i]) res.colDicts[i] = std::make_shared<Dict>(*d);
        }
      phost.done((int64_t)nr, "columnar result for a derived table");
      res.info = single ? "gpu-aggregate" : "gpu-join-aggregate";
      return res;
    }
    {
      const size_t first = std::min<size_t>(s.offset, want), nr = want - first;
      rows.resize(nr);
      std::exception_ptr err;
      std::mutex errMu;
      parallelFor(nr, [&](size_t b, size_t e) {
        try {
          PostCtx tpc = pc;
          for (size_t j = b; j < e; ++j) {
            const size_t i = first + j;
            tpc.keys = keys.data() + keep[order[i]] * plan.nkeys;
            tpc.accs = accs.data() + keep[order[i]] * plan.naggs;
            rows[j].reserve(items.size());
            for (auto& it : items) rows[j].push_back(evalPost(it.e, tpc));
          }
        } catch (...) {
          std::lock_guard<std::mutex> g(errMu);
          err = std::current_exception();
        }
      }, 16384);
      if (err) std::rethrow_exception(err);
    }
    orderBy.clear();
    s.offset = 0;
    s.limit = -1;
    sortKeys.clear();
    finish(rows, noKeys);
    phost.done((int64_t)res.rows.size(), "HAVING / ORDER BY / LIMIT on the candidate groups, formatting");
    res.info = single ? "gpu-aggregate" : "gpu-join-aggregate";
    if (envSetting("TRACE")) fprintf(stderr, "[trace] %.2f ms %s\n", msSince(t0), res.info.c_str());
    return res;
  }

  // ================= Plain projections (optionally joined / ordered / limited)
  // Predicates and expressions the GPU can't evaluate (host-only functions over several columns)
  // run on the host over the rows the GPU filters produce.
  std::vector<ExprP> hostFilt;
  for (auto& pd : pushdown) {
    std::vector<ExprP> keep;
    for (auto& p : pd) (gpuCompilable(p, ctx) ? keep : hostFilt).push_back(p);
    pd.swap(keep);
  }
  {
    std::vector<ExprP> keep;
    for (auto& p : residual) (gpuCompilable(p, ctx) ? keep : hostFilt).push_back(p);
    residual.swap(keep);
  }
  bool hostMode = !hostFilt.empty();
  for (auto& it : items) hostMode = hostMode || !gpuCompilable(it.e, ctx);
  for (auto& o : orderBy) hostMode = hostMode || !gpuCompilable(o.e, ctx);
  if (hostMode) {
    for (auto& f : hostFilt) colsOf(f, needed);
    notes.push_back("host evaluation of: " + [&] {
      std::string r;
      for (auto& f : hostFilt) r += (r.empty() ? "" : ", ") + f->str();
      for (auto& it : items) if (!gpuCompilable(it.e, ctx)) r += (r.empty() ? "" : ", ") + it.e->str();
      return r;
    }());
  }
  Rel rel;
  // Top-N over a table with columns on disk: only the filter / ORDER BY columns are scanned (with
  // each row's location); the other columns are read for the N rows that are left.
  std::set<int> deferred;
  std::vector<Loc> lateLocs;
  if (ctx.tabs.size() == 1 && !hostMode && !orderBy.empty() && s.limit >= 0 && s.offset + s.limit <= 10000 && !tRawNumber) {
    std::set<std::pair<int, int>> early;
    for (auto& o : orderBy) colsOf(o.e, early);
    for (auto& p : pushdown[0]) colsOf(p, early);
    for (auto& p : residual) colsOf(p, early);
    const Table& t0 = *ctx.tabs[0];
    bool cold = false;
    for (int c : needOf(0)) {
      if (early.count({0, c})) continue;
      deferred.insert(c);
      for (auto& sp : t0.segs) cold = cold || segCold(*sp->d, c);
    }
    if (!cold || deferred.size() == needOf(0).size()) deferred.clear();
  }
  if (ctx.tabs.size() == 1) {
    int64_t maxRows = orderBy.empty() && s.limit >= 0 && hostFilt.empty() ? s.offset + s.limit : -1;
    notes.push_back("GPU filter + gather (late materialization)" + std::string(maxRows >= 0 ? ", early stop at LIMIT" : ""));
    if (explain) return explainResult();
    std::set<int> need = needOf(0);
    for (int c : deferred) need.erase(c);
    rel = materialize(ctx, 0, pushdown[0], need, maxRows, notes, {}, deferred.empty() ? nullptr : &lateLocs);
    for (int c : deferred) rel.cols.push_back(RelCol{0, c, ctx.tabs[0]->cols[c].pt, DevVec<uint8_t>(16), ctx.tabs[0]->colBytes(c)});  // (filled for the top N below)
  } else {
    notes.push_back("GPU join pipeline (EXPLAIN runs the joins to report the actual strategy):");
    rel = buildJoin();
    if (explain) return explainResult();
  }
  Compiler comp(&ctx);
  ExprP filter = ctx.tabs.size() == 1 ? nullptr : andAll(residual);
  Prog fp;
  if (filter) fp = comp.compile(filter);
  std::vector<Prog> projs, ords;
  std::vector<std::pair<int, int>> hostCols;
  std::vector<Prog> hostProgs;
  if (hostMode) {
    std::set<std::pair<int, int>> hc;
    for (auto& it : items) colsOf(it.e, hc);
    for (auto& o : orderBy) colsOf(o.e, hc);
    for (auto& f : hostFilt) colsOf(f, hc);
    for (auto& pr : hc) {
      hostCols.push_back(pr);
      const bool vec = ctx.tabs[pr.first]->cols[pr.second].pt == PType::VEC;
      hostProgs.push_back(vec ? Prog{} : comp.compile(boundCol(ctx, pr.first, pr.second)));
    }
  } else {
    for (auto& it : items) projs.push_back(comp.compile(it.e));
    for (auto& o : orderBy) ords.push_back(comp.compile(o.e));
  }
  comp.uploadSets();
  bindRelCols(comp, rel);
  DevVec<int32_t> sel;
  int64_t m = rel.n;
  if (filter) {
    ProfStage pf("residual filter");
    sel = DevVec<int32_t>(rel.n);
    m = gpuSelect(comp.env, true, fp, rel.n, nullptr, sel.p);
    pf.done(m, filter->str());
  }
  if (hostMode) {
    if (m > 20'000'000) throw SqlError("too many rows for host evaluation (" + std::to_string(m) + "); add filters", 1105);
    ProfStage ph("host evaluation");
    std::vector<std::vector<double>> cv;
    std::vector<std::vector<uint8_t>> vecBytes(hostCols.size());
    for (size_t i = 0; i < hostProgs.size(); ++i) {
      const Table& tb = *ctx.tabs[hostCols[i].first];
      if (tb.cols[hostCols[i].second].pt == PType::VEC) {  // raw float32 vectors straight from the relation
        const int ri = rel.find(hostCols[i].first, hostCols[i].second);
        const int es = rel.cols[ri].esize;
        DevVec<uint8_t> tmp(std::max<int64_t>(m, 1) * es);
        if (filter) gpuGatherBytes(rel.cols[ri].data.p, es, sel.p, m, tmp.p, false);
        else if (m) d2d(tmp.p, rel.cols[ri].data.p, m * es);
        vecBytes[i] = tmp.download(m * es);
        cv.emplace_back();
        continue;
      }
      DevVec<double> out(std::max<int64_t>(m, 1));
      gpuEvalAt(comp.env, hostProgs[i], filter ? sel.p : nullptr, m, out.p);
      cv.push_back(out.download(m));
    }
    std::vector<Value> rowVals(hostCols.size());
    PostCtx pc;
    pc.row = &rowVals;
    pc.colSlot = [&](int bt, int bc) {
      for (size_t i = 0; i < hostCols.size(); ++i)
        if (hostCols[i].first == bt && hostCols[i].second == bc) return (int)i;
      return -1;
    };
    std::vector<std::vector<Value>> rows, keys;
    for (int64_t r = 0; r < m; ++r) {
      for (size_t i = 0; i < hostCols.size(); ++i) {
        const Table& tb = *ctx.tabs[hostCols[i].first];
        const ColumnMeta& cm = tb.cols[hostCols[i].second];
        if (cm.pt == PType::VEC) {
          const float* f = (const float*)vecBytes[i].data() + r * cm.vecDim;
          rowVals[i] = f[0] != f[0] ? Value::Null() : Value::S(formatVector(f, cm.vecDim));
          continue;
        }
        rowVals[i] = typedVal(cv[i][r], cm.lt, tb.dicts[hostCols[i].second].get(), cm.scale);
      }
      bool keep = true;
      for (auto& f : hostFilt) keep = keep && isTrue(evalPost(f, pc));
      if (!keep) continue;
      std::vector<Value> row, key;
      for (auto& it : items) row.push_back(evalPost(it.e, pc));
      for (auto& o : orderBy) key.push_back(evalPost(o.e, pc));
      rows.push_back(std::move(row));
      keys.push_back(std::move(key));
    }
    finish(rows, keys);
    ph.done((int64_t)res.rows.size(), std::to_string(m) + " rows from the GPU");
    res.info = "gpu-scan+host";
    return res;
  }
  DevVec<int32_t> finalIdx;
  int64_t start = std::min<int64_t>(s.offset, m);
  int64_t cnt = s.limit < 0 ? m - start : std::min<int64_t>(m - start, s.limit);
  if (tInsertCols ? cnt * (int64_t)projs.size() > 1'000'000'000 : cnt > 20'000'000)
    throw SqlError("result set too large (" + std::to_string(cnt) + " rows); add a LIMIT", 1105);
  ProfStage psort(orderBy.empty() ? "limit" : "GPU sort");
  if (!orderBy.empty() && m > 0) {
    std::vector<DevVec<double>> keyBufs;
    std::vector<std::shared_ptr<DevVec<int32_t>>> ranks;
    std::vector<std::pair<const double*, bool>> keys;
    for (size_t k = 0; k < orderBy.size(); ++k) {
      keyBufs.emplace_back(m);
      gpuEvalAt(comp.env, ords[k], filter ? sel.p : nullptr, m, keyBufs.back().p);
      if (orderBy[k].e->ty == LType::VARCHAR) {
        ranks.push_back(dictRankDev(*ctx.dict(*orderBy[k].e)));
        gpuMapRank(keyBufs.back().p, ranks.back()->p, m);
      }
      keys.push_back({keyBufs.back().p, orderBy[k].desc});
    }
    DevVec<int32_t> perm = gpuSortPerm(keys, m);
    finalIdx = DevVec<int32_t>(std::max<int64_t>(cnt, 1));
    if (filter) gpuGather(sel.p, PType::I32, perm.p + start, cnt, finalIdx.p);
    else d2d(finalIdx.p, perm.p + start, cnt * 4);
    notes.push_back("  GPU sort of " + std::to_string(m) + " rows");
  } else if (cnt > 0) {
    finalIdx = DevVec<int32_t>(cnt);
    if (filter) d2d(finalIdx.p, sel.p + start, cnt * 4);
    else {
      std::vector<int32_t> seq(cnt);
      for (int64_t i = 0; i < cnt; ++i) seq[i] = (int32_t)(start + i);
      h2d(finalIdx.p, seq.data(), cnt * 4);
    }
  }
  psort.done(cnt, std::to_string(m) + " rows in");
  if (!deferred.empty() && cnt > 0) {  // the top N rows, with the deferred columns read for them
    ProfStage plate("late materialization");
    const Table& t0 = *ctx.tabs[0];
    std::vector<int32_t> fi = finalIdx.download(cnt);
    Rel top;
    top.n = cnt;
    std::vector<CellRead> lateReads;  // all deferred cells on disk, read together
    std::vector<std::pair<void*, std::vector<uint8_t>>> lateUploads;
    for (auto& rc : rel.cols) {
      RelCol nc{rc.bt, rc.bc, rc.pt, DevVec<uint8_t>(cnt * rc.esize), rc.esize};
      if (!deferred.count(rc.bc)) {
        gpuGatherBytes(rc.data.p, rc.esize, finalIdx.p, cnt, nc.data.p, false);
      } else {
        std::vector<uint8_t> h((size_t)cnt * rc.esize);
        std::vector<CellRead> cold;
        for (int64_t r = 0; r < cnt; ++r) {
          const Loc l = lateLocs[fi[r]];
          if (l.seg >= 0 && segCold(*t0.segs[l.seg]->d, rc.bc)) cold.push_back({t0.segs[l.seg]->d.get(), rc.bc, l.row, h.data() + r * rc.esize});
          else cellRaw(t0, l, rc.bc, h.data() + r * rc.esize);
        }
        lateReads.insert(lateReads.end(), cold.begin(), cold.end());
        lateUploads.push_back({nc.data.p, std::move(h)});
      }
      top.cols.push_back(std::move(nc));
    }
    segRawCells(lateReads);
    for (auto& [dp, h] : lateUploads) h2d(dp, h.data(), h.size());
    rel = std::move(top);
    bindRelCols(comp, rel);
    std::vector<int32_t> seq(cnt);
    for (int64_t i = 0; i < cnt; ++i) seq[i] = (int32_t)i;
    h2d(finalIdx.p, seq.data(), cnt * 4);
    notes.push_back("  late materialization: " + std::to_string(deferred.size()) + " columns read for " + std::to_string(cnt) + " rows");
    plate.done(cnt, std::to_string(deferred.size()) + " columns read for the top rows");
  }
  ProfStage pproj("project + format");
  // IN (SELECT int_col ..): only the distinct values matter: deduplicated on the GPU through a bitmap
  // over the column's value range, so only those come back.
  if (tRawNumber && projs.size() == 1 && items[0].e->kind == Expr::COL && items[0].e->ty != LType::VARCHAR &&
      items[0].e->ty != LType::DOUBLE && items[0].e->ty != LType::DECIMAL && items[0].e->ty != LType::VECTOR && cnt > 4096) {
    int64_t lo, hi;
    const ExprP& ce = items[0].e;
    if (colRange(*ctx.tabs[ce->bt], ce->bc, lo, hi) && span64(lo, hi) < (uint64_t)1 << 31) {
      std::vector<int64_t> d = gpuDistinctInts(comp.env, projs[0], finalIdx.p, cnt, lo, (int64_t)span64(lo, hi) + 1);
      auto nc = std::make_shared<std::vector<double>>(d.begin(), d.end());
      res.numCol = nc;
      res.typed = true;
      pproj.done(cnt, std::to_string(d.size()) + " distinct values (GPU bitmap)");
      res.info = "gpu-scan";
      return res;
    }
  }
  std::vector<std::vector<double>> cols;
  std::vector<std::vector<int64_t>> icols(projs.size());  // exact 64-bit integer columns
  for (size_t i = 0; i < projs.size(); ++i) {
    const ExprP& e = items[i].e;
    if ((e->kind == Expr::COL || e->kind == Expr::BIN || e->kind == Expr::UNARY) && (e->ty == LType::BIGINT || e->ty == LType::DATETIME)) {
      DevVec<int64_t> out(std::max<int64_t>(cnt, 1));
      gpuEvalAtInt(comp.env, projs[i], finalIdx.p, cnt, out.p);
      icols[i] = out.download(std::max<int64_t>(cnt, 0));
      cols.emplace_back();
      continue;
    }
    DevVec<double> out(std::max<int64_t>(cnt, 1));
    gpuEvalAt(comp.env, projs[i], finalIdx.p, cnt, out.p);
    cols.push_back(out.download(std::max<int64_t>(cnt, 0)));
  }
  if (tRawNumber && projs.size() == 1 && items[0].e->ty != LType::VARCHAR && items[0].e->ty != LType::VECTOR) {
    auto nc = std::make_shared<std::vector<double>>();
    if (!icols[0].empty()) {
      nc->reserve(icols[0].size());
      for (int64_t v : icols[0]) nc->push_back(v == kNullI64 ? nullv() : (double)v);
    } else {
      *nc = std::move(cols[0]);
    }
    res.numCol = nc;
    res.typed = true;
    pproj.done(cnt);
    res.info = "gpu-scan";
    return res;
  }
  if (tInsertCols) {  // INSERT .. SELECT: the columns as they are (strings: codes of plain columns)
    bool ok = true;
    for (auto& it : items)
      ok = ok && it.e->ty != LType::VECTOR && (it.e->ty != LType::VARCHAR || (it.e->kind == Expr::COL && it.e->bt >= 0));
    if (ok) {
      res.insertCols = res.typed = true;
      res.colDicts.assign(items.size(), nullptr);
      res.colScales.assign(items.size(), -1);
      for (size_t i = 0; i < items.size(); ++i) {
        const ExprP& e = items[i].e;
        if (e->ty == LType::VARCHAR) res.colDicts[i] = ctx.tabs[e->bt]->dicts[e->bc];
        res.colScales[i] = e->scale;
      }
      res.numCols = std::move(cols);
      res.intCols = std::move(icols);
      pproj.done(cnt, "columns for INSERT .. SELECT");
      res.info = "gpu-scan";
      return res;
    }
  }
  auto valueAt = [&](size_t i, int64_t r) {
    if (!icols[i].empty()) {
      const int64_t v = icols[i][r];
      if (v == kNullI64) return Value::Null();
      return items[i].e->ty == LType::DATETIME ? Value::Ts(v) : Value::I(v);
    }
    return typedVal(cols[i][r], items[i].e->ty, ctx.dict(*items[i].e), items[i].e->scale);
  };
  res.typed = tWantVals;
  // Rows are built on all cores (big results: formatting dominates the GPU work).
  if (tWantVals) res.vals.resize(std::max<int64_t>(cnt, 0));
  else res.rows.resize(std::max<int64_t>(cnt, 0));
  const bool wantVals = tWantVals;
  parallelFor((size_t)std::max<int64_t>(cnt, 0), [&](size_t b, size_t e) {
    for (size_t r = b; r < e; ++r) {
      if (wantVals) {
        std::vector<Value>& row = res.vals[r];
        row.reserve(items.size());
        for (size_t i = 0; i < items.size(); ++i) row.push_back(valueAt(i, (int64_t)r));
      } else {
        std::vector<std::string>& row = res.rows[r];
        row.reserve(items.size());
        for (size_t i = 0; i < items.size(); ++i) row.push_back(cell(valueAt(i, (int64_t)r)));
      }
    }
  }, 4096);
  pproj.done(cnt);
  res.info = ctx.tabs.size() == 1 ? "gpu-scan" : "gpu-join";
  if (envSetting("TRACE")) fprintf(stderr, "[trace] %.2f ms %s\n", msSince(t0), res.info.c_str());
  return res;
}

// ================================================================== query layer
// CTEs, derived tables, subqueries, UNION and COUNT(DISTINCT) on top of doSelect: inner queries run
// first and their results become query-scoped GPU temp tables (or literals / value sets).


ExprP cloneExpr(const ExprP& e) {
  if (!e) return e;
  auto y = std::make_shared<Expr>(*e);
  y->bound = false;
  y->bt = y->bc = -1;
  for (auto& a : y->args) a = cloneExpr(a);
  for (auto& a : y->partBy) a = cloneExpr(a);
  for (auto& o : y->winOrder) o.first = cloneExpr(o.first);
  if (e->sub) y->sub = std::make_shared<Stmt>(*e->sub);  // statements are cloned again before use
  return y;
}

Stmt cloneStmt(const Stmt& s) {
  Stmt c = s;
  for (auto& it : c.items) it.e = cloneExpr(it.e);
  for (auto& f : c.from) {
    f.on = cloneExpr(f.on);
    if (f.sub) f.sub = std::make_shared<Stmt>(cloneStmt(*f.sub));
  }
  for (auto& j : c.joinConds) j = cloneExpr(j);
  c.where = cloneExpr(c.where);
  c.having = cloneExpr(c.having);
  for (auto& g : c.groupBy) g = cloneExpr(g);
  for (auto& o : c.orderBy) o.e = cloneExpr(o.e);
  if (c.unionNext) c.unionNext = std::make_shared<Stmt>(cloneStmt(*c.unionNext));
  for (auto& [n, q] : c.ctes) q = std::make_shared<Stmt>(cloneStmt(*q));
  return c;
}

void walkStmtExprs(Stmt& s, const std::function<void(ExprP&)>& fn) {
  for (auto& it : s.items) fn(it.e);
  if (s.where) fn(s.where);
  if (s.having) fn(s.having);
  for (auto& j : s.joinConds) fn(j);
  for (auto& f : s.from)
    if (f.on) fn(f.on);
  for (auto& g : s.groupBy) fn(g);
  for (auto& o : s.orderBy) fn(o.e);
}

void formatRows(ResultSet& r) {
  if (!r.typed) return;
  for (auto& v : r.vals) {
    std::vector<std::string> row;
    for (auto& x : v) row.push_back(cell(x));
    r.rows.push_back(std::move(row));
  }
  r.vals.clear();
  r.typed = false;
}

std::string Engine::makeTemp(const ResultSet& r, QueryCtx& q, const std::string& hint) {
  static std::atomic<int> seq{0};
  std::string name = "__tmp" + std::to_string(++seq) + "_" + hint;
  std::vector<ColumnMeta> cols;
  std::set<std::string> seen;
  for (size_t i = 0; i < r.names.size(); ++i) {
    std::string n = r.names[i];
    for (int k = 2; seen.count(lower(n)); ++k) n = r.names[i] + "_" + std::to_string(k);
    seen.insert(lower(n));
    ColumnMeta m;
    m.name = n;
    m.lt = i < r.types.size() ? r.types[i] : LType::VARCHAR;
    if (m.lt == LType::DECIMAL) {  // the scale travels with the values (Value::scale); none known: DOUBLE
      int sc = -1;
      for (auto& row : r.vals)
        if (i < row.size() && !row[i].isNull() && row[i].scale >= 0) sc = std::max(sc, (int)row[i].scale);
      if (sc < 0) m.lt = LType::DOUBLE;
      else m.scale = sc;
    }
    if (m.lt == LType::VECTOR) {  // the dimension comes from the values (all NULL: kept as text)
      m.lt = LType::VARCHAR;
      for (auto& row : r.vals) {
        std::vector<float> f;
        if (i < row.size() && !row[i].isNull() && parseVector(row[i].toString(), f) && !f.empty()) {
          m.lt = LType::VECTOR;
          m.vecDim = (int)f.size();
          break;
        }
      }
    }
    m.pt = physOf(m.lt);
    cols.push_back(m);
  }
  if (cols.empty()) throw SqlError("derived table has no columns", 1105);
  if (r.insertCols)  // typed columns (a projection): DECIMAL scales from the result
    for (size_t i = 0; i < cols.size() && i < r.colScales.size(); ++i)
      if (r.types[i] == LType::DECIMAL) {
        cols[i].lt = r.colScales[i] >= 0 ? LType::DECIMAL : LType::DOUBLE;
        cols[i].scale = std::max(0, r.colScales[i]);
        cols[i].pt = physOf(cols[i].lt);
      }
  auto t = std::make_shared<Table>(name, cols, -1, -1, std::vector<int>{}, std::vector<int>{}, 1, false);
  t->db = "__tmp";  // (before the rows: temporary tables stay out of the session's transaction)
  if (r.insertCols) {  // a projection's columns: appended as they are (strings keep their codes and dictionary)
    const size_t nc = cols.size();
    int64_t n = 0;
    for (size_t i = 0; i < nc; ++i) n = std::max<int64_t>(n, !r.intCols[i].empty() ? (int64_t)r.intCols[i].size() : (int64_t)r.numCols[i].size());
    std::vector<std::vector<int64_t>> iv(nc);
    std::vector<std::vector<double>> dv(nc);
    for (size_t i = 0; i < nc; ++i) {
      const ColumnMeta& m = cols[i];
      if (m.lt == LType::VARCHAR) t->dicts[i] = r.colDicts[i];
      if (!r.intCols[i].empty()) {
        iv[i] = r.intCols[i];
        continue;
      }
      const std::vector<double>& x = r.numCols[i];
      if (m.pt == PType::F64) {
        dv[i] = x;
        continue;
      }
      const int64_t nul = m.pt == PType::I64 ? kNullI64 : kNullI32;
      const double mul = m.lt == LType::DECIMAL ? pow10i(m.scale) : 1.0;
      iv[i].resize(x.size());
      for (size_t k = 0; k < x.size(); ++k) iv[i][k] = x[k] != x[k] ? nul : std::llround(x[k] * mul);
    }
    for (size_t i = 0; i < nc; ++i) t->hasNull[i] = 1;
    if (n > 0) {
      t->appendColumns(iv, dv, n);
      t->flush(true);
    }
    std::lock_guard<std::mutex> g(cat_.mu);
    cat_.tables["__tmp." + lower(name)] = t;
    q.temps.push_back("__tmp." + lower(name));
    return "__tmp." + name;
  }
  if (!r.devCols.empty()) {  // built on the GPU: the columns become the table's one segment (no host copy)
    const int64_t n = r.devRows;
    const size_t nc = cols.size();
    auto seg = std::make_unique<Segment>();
    seg->part = 0;
    seg->n = n;
    seg->host.resize(nc);  // (filled on demand: Table::colData)
    seg->dev.resize(nc, nullptr);
    seg->mapped.assign(nc, 0);
    seg->enc.assign(nc, 0);
    seg->encHost.resize(nc);
    seg->mn.resize(nc);
    seg->mx.resize(nc);
    for (size_t c = 0; c < nc; ++c) {
      seg->dev[c] = r.devCols[c]->p;
      r.devCols[c]->p = nullptr;
      seg->mn[c] = r.devCols[c]->mn;
      seg->mx[c] = r.devCols[c]->mx;
      seg->devBytes += n * t->colBytes((int)c);
      if (cols[c].lt == LType::VARCHAR) t->dicts[c] = r.colDicts[c];  // (shared: temp tables add no strings)
      t->hasNull[c] = 1;
    }
    seg->del->host.assign((size_t)(n + 31) / 32, 0);
    if (n > 0) t->segs.push_back(std::move(seg));
    std::lock_guard<std::mutex> g(cat_.mu);
    cat_.tables["__tmp." + lower(name)] = t;
    q.temps.push_back("__tmp." + lower(name));
    return "__tmp." + name;
  }
  if (!r.numCols.empty()) {  // numeric columns (a GROUP BY result): straight into the write buffer
    const size_t n = r.numCols[0].size();
    WriteBuffer& b = *t->bufs[0];
    for (size_t c = 0; c < cols.size(); ++c) {
      const ColumnMeta& m = cols[c];
      if (m.lt == LType::VARCHAR) {  // dictionary codes: the column takes a copy of their dictionary
        if (c >= r.colDicts.size() || !r.colDicts[c]) throw SqlError("internal: string column without a dictionary", 1105);
        t->dicts[c] = r.colDicts[c];  // (shared: temp tables add no strings)
      }
      const int sz = t->colBytes((int)c);
      std::vector<uint8_t> bytes(n * sz);
      std::atomic<bool> nul{false};
      parallelFor(n, [&](size_t lo, size_t hi) {
        for (size_t i = lo; i < hi; ++i) {
          const double d = r.numCols[c][i];
          if (m.pt == PType::F64) { memcpy(&bytes[i * 8], &d, 8); if (d != d) nul = true; continue; }
          int64_t x = d != d ? (m.pt == PType::I64 ? kNullI64 : kNullI32) : std::llround(d);
          if (d != d) nul = true;
          if (m.pt == PType::I64) memcpy(&bytes[i * 8], &x, 8);
          else { const int32_t y = (int32_t)x; memcpy(&bytes[i * 4], &y, 4); }
        }
      });
      if (nul) t->hasNull[c] = 1;
      b.cols[c] = std::move(bytes);
    }
    b.n = (int64_t)n;
    b.del = std::make_shared<DelBits>();
    b.del->host.assign((n + 31) / 32, 0);
    ++b.version;
    ++b.gen;
    t->flush(true);
    std::lock_guard<std::mutex> g(cat_.mu);
    cat_.tables["__tmp." + lower(name)] = t;
    q.temps.push_back("__tmp." + lower(name));
    return "__tmp." + name;
  }
  bool bulk = true;
  for (auto& c : cols) bulk = bulk && c.pt != PType::VEC;
  if (bulk) {  // column by column straight into the write buffer (on all cores), then one flush
    const size_t n = r.vals.size(), nc = cols.size();
    WriteBuffer& b = *t->bufs[0];
    for (size_t c = 0; c < nc; ++c) {
      const ColumnMeta& m = cols[c];
      const int sz = t->colBytes((int)c);
      std::vector<uint8_t> bytes(n * sz);
      std::atomic<bool> nul{false};
      if (m.lt == LType::VARCHAR) {
        Dict& d = *t->dicts[c];
        for (size_t i = 0; i < n; ++i) {
          const Value& v = r.vals[i][c];
          int32_t code = kNullI32;
          if (v.isNull()) nul = true;
          else code = d.getOrAdd(v.kind == Value::STR ? v.s : v.toString());
          memcpy(&bytes[i * 4], &code, 4);
        }
      } else {
        const double unscale = m.lt == LType::DECIMAL ? pow10i(m.scale) : 1.0;
        parallelFor(n, [&](size_t lo, size_t hi) {
          for (size_t i = lo; i < hi; ++i) {
            const Value& v = r.vals[i][c];
            if (m.pt == PType::F64) {
              const double d = v.isNull() ? nullv() : v.kind == Value::STR ? std::strtod(v.s.c_str(), nullptr) : v.num();
              if (v.isNull()) nul = true;
              memcpy(&bytes[i * 8], &d, 8);
              continue;
            }
            int64_t x;
            if (v.isNull()) {
              nul = true;
              x = m.pt == PType::I64 ? kNullI64 : kNullI32;
            } else if (m.lt == LType::DECIMAL) {
              x = std::llround(v.num() * unscale);
            } else if (m.lt == LType::DATE && v.kind == Value::DATETIME) {
              x = floorDiv(v.i, kUsPerDay);
            } else if (m.lt == LType::DATETIME && v.kind == Value::DATE) {
              x = v.i * kUsPerDay;
            } else if (v.kind == Value::STR) {
              x = std::strtoll(v.s.c_str(), nullptr, 10);
            } else {
              x = v.kind == Value::DBL ? std::llround(v.d) : v.i;
            }
            if (m.pt == PType::I64) memcpy(&bytes[i * 8], &x, 8);
            else { const int32_t y = (int32_t)x; memcpy(&bytes[i * 4], &y, 4); }
          }
        });
      }
      if (nul) t->hasNull[c] = 1;
      b.cols[c] = std::move(bytes);
    }
    b.n = (int64_t)n;
    b.del = std::make_shared<DelBits>();
    b.del->host.assign((n + 31) / 32, 0);
    ++b.version;
    ++b.gen;
  } else {
    for (auto& row : r.vals) t->appendRow(row);
  }
  t->flush(true);
  std::lock_guard<std::mutex> g(cat_.mu);
  cat_.tables["__tmp." + lower(name)] = t;
  q.temps.push_back("__tmp." + lower(name));
  return "__tmp." + name;
}

void Engine::dropTemps(QueryCtx& q) {
  std::lock_guard<std::mutex> g(cat_.mu);
  for (auto& k : q.temps) cat_.tables.erase(k);
  q.temps.clear();
}

namespace {
// Column names of a FROM entry (catalog tables only; derived tables unknown here).
bool fromHasColumn(Catalog& cat, const Stmt& s, const std::string& col) {
  for (auto& f : s.from) {
    if (f.sub) continue;
    std::shared_ptr<Table> t;
    {
      std::lock_guard<std::mutex> g(cat.mu);
      auto it = cat.tables.find(Catalog::key(f.name));
      if (it != cat.tables.end()) t = it->second;
    }
    if (t && t->colIndex(col) >= 0) return true;
  }
  return false;
}
bool fromHasAlias(const Stmt& s, const std::string& a) {
  for (auto& f : s.from)
    if (strcasecmp(f.alias.c_str(), a.c_str()) == 0 || (f.alias.empty() && strcasecmp(f.name.c_str(), a.c_str()) == 0))
      return true;  // (an aliased table is only visible by its alias)
  return false;
}
}  // namespace

// Is alias a defined anywhere in s, including its derived tables and subqueries in FROM?
static bool fromHasAliasDeep(const Stmt& s, const std::string& a) {
  if (fromHasAlias(s, a)) return true;
  for (auto& f : s.from)
    if (f.sub && fromHasAliasDeep(*f.sub, a)) return true;
  return s.unionNext && fromHasAliasDeep(*s.unionNext, a);
}
// Expressions of s and of the derived tables in its FROM (correlated references may sit in there).
void walkStmtExprsDeep(Stmt& s, const std::function<void(ExprP&)>& fn) {
  walkStmtExprs(s, fn);
  for (auto& f : s.from)
    if (f.sub) walkStmtExprsDeep(*f.sub, fn);
  if (s.unionNext) walkStmtExprsDeep(*s.unionNext, fn);
}

// Does `col` (inside subquery `sub`) refer to the outer statement?
static bool outerRef(Catalog& cat, const ExprP& col, const Stmt& sub, const Stmt& outer,
                     const std::map<std::string, std::string>& ctes) {
  if (col->kind != Expr::COL) return false;
  if (!col->table.empty()) return !fromHasAliasDeep(sub, col->table) && fromHasAlias(outer, col->table);
  // Unqualified: outer if no table of the subquery has it but an outer table does.
  Stmt subR = sub;
  for (auto& f : subR.from) {
    auto it = ctes.find(lower(f.name));
    if (it != ctes.end()) f.name = it->second;
  }
  for (auto& f : subR.from)
    if (f.sub) return false;  // unknown columns: assume local
  return !fromHasColumn(cat, subR, col->name) && fromHasColumn(cat, outer, col->name);
}

void Engine::resolveSubqueries(Stmt& s, QueryCtx& q) {
  auto correlated = [&](Stmt& sub) {
    bool any = false;
    std::function<void(const ExprP&)> scan = [&](const ExprP& e) {
      if (!e || any) return;
      if (outerRef(cat_, e, sub, s, q.ctes)) any = true;
      for (auto& a : e->args) scan(a);
    };
    walkStmtExprsDeep(sub, [&](ExprP& e) { scan(e); });
    return any;
  };
  // EXISTS (SELECT .. WHERE inner = outer AND ..) -> outer IN (SELECT inner FROM .. WHERE ..)
  auto decorrelateExists = [&](ExprP& e) -> bool {
    Stmt sub = cloneStmt(*e->sub);
    std::vector<ExprP> conj;
    std::function<void(const ExprP&)> flat = [&](const ExprP& x) {
      if (x && x->kind == Expr::BIN && x->op == "AND") { flat(x->args[0]); flat(x->args[1]); }
      else if (x) conj.push_back(x);
    };
    flat(sub.where);
    ExprP outerCol, innerCol;
    std::vector<ExprP> rest;
    for (auto& c : conj) {
      if (!outerCol && c->kind == Expr::BIN && c->op == "=" && c->args[0]->kind == Expr::COL && c->args[1]->kind == Expr::COL) {
        bool o0 = outerRef(cat_, c->args[0], sub, s, q.ctes), o1 = outerRef(cat_, c->args[1], sub, s, q.ctes);
        if (o0 != o1) {
          outerCol = o0 ? c->args[0] : c->args[1];
          innerCol = o0 ? c->args[1] : c->args[0];
          continue;
        }
      }
      rest.push_back(c);
    }
    if (!outerCol) return false;
    // NULLs never satisfy the equality: inner NULLs are dropped and a NULL outer value gives false
    // (not NULL, which NOT EXISTS would keep as NULL).
    auto notNull = [](ExprP x) {
      auto n = std::make_shared<Expr>();
      n->kind = Expr::FUNC;
      n->op = "ISNOTNULL";
      n->args = {x};
      return n;
    };
    ExprP w = notNull(cloneExpr(innerCol));
    for (auto& r : rest) w = mkBin("AND", w, r);
    sub.where = w;
    sub.items = {SelectItem{innerCol, ""}};
    sub.orderBy.clear();
    sub.limit = -1;
    if (correlated(sub)) return false;
    auto in = std::make_shared<Expr>();
    in->kind = Expr::INLIST;
    in->args = {cloneExpr(outerCol)};
    in->sub = std::make_shared<Stmt>(std::move(sub));
    e = mkBin("AND", notNull(cloneExpr(outerCol)), in);
    return true;
  };
  // Any other correlated scalar subquery: evaluate it once per distinct combination of the outer
  // columns it references (from the outer FROM), keep the results in a temporary table and LEFT JOIN
  // it on those columns.
  auto applyPerOuterValue = [&](ExprP& e) {
    Stmt sub = cloneStmt(*e->sub);
    std::vector<ExprP> outerCols;
    std::set<std::string> seen;
    std::function<void(const ExprP&)> collect = [&](const ExprP& x) {
      if (!x) return;
      if (x->kind == Expr::COL && outerRef(cat_, x, sub, s, q.ctes)) {
        if (seen.insert(lower(x->table + "." + x->name)).second) outerCols.push_back(x);
        return;
      }
      for (auto& a : x->args) collect(a);
    };
    walkStmtExprsDeep(sub, [&](ExprP& x) { collect(x); });
    Stmt dq = cloneStmt(s);
    dq.items.clear();
    for (auto& c : outerCols) dq.items.push_back(SelectItem{cloneExpr(c), ""});
    dq.distinct = true;
    dq.where.reset();
    if (s.from.size() == 1) {  // one outer table: only the outer values its other filters let through
      std::function<bool(const ExprP&)> hasSub = [&](const ExprP& x) {
        if (!x) return false;
        if (x->sub) return true;
        for (auto& a : x->args)
          if (hasSub(a)) return true;
        return false;
      };
      std::function<void(const ExprP&)> keep = [&](const ExprP& x) {
        if (!x) return;
        if (x->kind == Expr::BIN && x->op == "AND") {
          keep(x->args[0]);
          keep(x->args[1]);
        } else if (!hasSub(x)) {
          dq.where = dq.where ? mkBin("AND", dq.where, cloneExpr(x)) : cloneExpr(x);
        }
      };
      keep(s.where);
    }
    dq.having.reset();
    dq.groupBy.clear();
    dq.orderBy.clear();
    dq.limit = -1;
    dq.offset = 0;
    dq.unionNext.reset();
    ResultSet keys = query(dq, q, false, true);
    if (keys.vals.size() > 20000)
      throw SqlError("correlated subquery: " + std::to_string(keys.vals.size()) + " distinct outer values (at most 20000)", 1235);
    ResultSet tab;
    tab.typed = true;
    for (size_t k = 0; k < outerCols.size(); ++k) tab.names.push_back("__o" + std::to_string(k));
    tab.types = keys.types;
    tab.names.push_back("__v");
    LType vt = LType::BIGINT;
    for (auto& row : keys.vals) {
      Stmt one = cloneStmt(sub);
      std::function<void(ExprP&)> bindOuter = [&](ExprP& x) {
        if (!x) return;
        if (x->kind == Expr::COL && outerRef(cat_, x, sub, s, q.ctes)) {
          const std::string key = lower(x->table + "." + x->name);
          size_t k = 0;
          while (k < outerCols.size() && lower(outerCols[k]->table + "." + outerCols[k]->name) != key) ++k;
          x = mkLit(row[k]);
          return;
        }
        for (auto& a : x->args) bindOuter(a);
      };
      walkStmtExprsDeep(one, bindOuter);
      ResultSet r = query(one, q, false, true);
      if (r.names.size() != 1) throw SqlError("Operand should contain 1 column(s)", 1241);
      if (r.vals.size() > 1) throw SqlError("Subquery returns more than 1 row", 1242);
      if (!r.types.empty()) vt = r.types[0];
      std::vector<Value> out = row;
      out.push_back(r.vals.empty() ? Value::Null() : r.vals[0][0]);
      tab.vals.push_back(std::move(out));
    }
    tab.types.push_back(vt);
    const std::string tmp = makeTemp(tab, q, "apply");
    static std::atomic<int> sq{0};
    const std::string alias = "__ap" + std::to_string(++sq);
    TableRef ref;
    ref.name = tmp;
    ref.alias = alias;
    ref.join = 1;
    // NULL outer values never match "=": with one outer column, its result is picked by IF(col IS
    // NULL, ..); with more, the columns holding NULLs are joined NULL-safely.
    auto isNull = [](ExprP x) {
      auto n = std::make_shared<Expr>();
      n->kind = Expr::FUNC;
      n->op = "ISNULL";
      n->args = {x};
      return n;
    };
    std::vector<bool> nullKey(outerCols.size());
    std::optional<Value> nullVal;
    for (auto& row : tab.vals)
      for (size_t k = 0; k < outerCols.size(); ++k)
        if (row[k].isNull()) {
          nullKey[k] = true;
          if (outerCols.size() == 1) nullVal = row.back();
        }
    for (size_t k = 0; k < outerCols.size(); ++k) {
      ExprP l = mkCol(alias, "__o" + std::to_string(k)), r = cloneExpr(outerCols[k]);
      ExprP eq = mkBin("=", l, r);
      if (nullKey[k] && outerCols.size() > 1) {
        auto co = std::make_shared<Expr>();
        co->kind = Expr::FUNC;
        co->op = "COALESCE";
        co->args = {eq, mkBin("AND", isNull(cloneExpr(l)), isNull(cloneExpr(r)))};
        eq = co;
      }
      ref.on = ref.on ? mkBin("AND", ref.on, eq) : eq;
    }
    s.from.push_back(ref);
    e = mkCol(alias, "__v");
    if (nullVal && !nullVal->isNull()) {
      auto f = std::make_shared<Expr>();
      f->kind = Expr::FUNC;
      f->op = "IF";
      f->args = {isNull(cloneExpr(outerCols[0])), mkLit(*nullVal), e};
      e = f;
    }
  };
  // Any other correlated EXISTS: (SELECT 1 .. LIMIT 1) IS NOT NULL, evaluated per outer value.
  auto existsPerOuterValue = [&](ExprP& e) {
    Stmt sub = cloneStmt(*e->sub);
    sub.items = {SelectItem{mkLit(Value::I(1)), ""}};
    sub.orderBy.clear();
    sub.limit = 1;
    sub.offset = 0;
    auto sq = std::make_shared<Expr>();
    sq->kind = Expr::SUBQUERY;
    sq->sub = std::make_shared<Stmt>(std::move(sub));
    ExprP v = sq;
    applyPerOuterValue(v);
    auto nn = std::make_shared<Expr>();
    nn->kind = Expr::FUNC;
    nn->op = "ISNOTNULL";
    nn->args = {v};
    e = nn;
  };
  std::function<void(ExprP&)> fix = [&](ExprP& e) {
    if (!e) return;
    for (auto& a : e->args) fix(a);
    if (e->kind == Expr::SUBQUERY) {
      Stmt sub = cloneStmt(*e->sub);
      if (correlated(sub)) {
        // (SELECT agg(x) FROM t2 WHERE t2.k = outer.k AND ..) -> LEFT JOIN (SELECT k, agg(x) .. GROUP BY k)
        std::vector<ExprP> conj, rest;
        std::function<void(const ExprP&)> flat = [&](const ExprP& x) {
          if (x && x->kind == Expr::BIN && x->op == "AND") { flat(x->args[0]); flat(x->args[1]); }
          else if (x) conj.push_back(x);
        };
        flat(sub.where);
        std::vector<ExprP> outerCols, innerCols;  // every inner.col = outer.col equality
        for (auto& c : conj) {
          if (c->kind == Expr::BIN && c->op == "=" && c->args[0]->kind == Expr::COL && c->args[1]->kind == Expr::COL) {
            bool o0 = outerRef(cat_, c->args[0], sub, s, q.ctes), o1 = outerRef(cat_, c->args[1], sub, s, q.ctes);
            if (o0 != o1) {
              outerCols.push_back(o0 ? c->args[0] : c->args[1]);
              innerCols.push_back(o0 ? c->args[1] : c->args[0]);
              continue;
            }
          }
          rest.push_back(c);
        }
        if (outerCols.empty() || sub.items.size() != 1 || !sub.groupBy.empty()) {
          applyPerOuterValue(e);
          return;
        }
        ExprP w;
        for (auto& r : rest) w = w ? mkBin("AND", w, r) : r;
        sub.where = w;
        ExprP item = sub.items[0].e;
        // Aggregates (GPU and host ones) in the item: each becomes a column of the grouped derived
        // table; the expression around them (COALESCE(COUNT(*), 0), ..) is evaluated after the LEFT
        // JOIN, so outer rows without matches see the empty-set values (COUNT 0, others NULL).
        std::vector<ExprP> aggs;
        std::function<bool(const ExprP&)> anyAgg = [&](const ExprP& x) {
          if (!x) return false;
          if (x->kind == Expr::FUNC && !x->window && (isAggName(x->op) || isHostAggName(x->op))) return true;
          for (auto& a : x->args)
            if (anyAgg(a)) return true;
          return false;
        };
        if (!anyAgg(item)) {  // plain value: one row per key expected; MAX keeps it single-valued
          auto m = std::make_shared<Expr>();
          m->kind = Expr::FUNC;
          m->op = "MAX";
          m->args = {item};
          item = m;
        }
        static std::atomic<int> sq{0};
        const std::string alias = "__sq" + std::to_string(++sq);
        std::function<ExprP(const ExprP&)> lift = [&](const ExprP& x) -> ExprP {
          if (x->kind == Expr::FUNC && !x->window && (isAggName(x->op) || isHostAggName(x->op))) {
            const std::string col = "__a" + std::to_string(aggs.size());
            aggs.push_back(x);
            ExprP v = mkCol(alias, col);
            if (x->op == "COUNT") {  // no matching rows: COUNT is 0, not NULL
              auto f = std::make_shared<Expr>();
              f->kind = Expr::FUNC;
              f->op = "COALESCE";
              f->args = {v, mkLit(Value::I(0))};
              v = f;
            }
            return v;
          }
          auto y = std::make_shared<Expr>(*x);
          for (auto& a : y->args) a = lift(a);
          return y;
        };
        ExprP outerExpr = lift(item);
        sub.items.clear();
        sub.groupBy.clear();
        for (size_t k = 0; k < innerCols.size(); ++k) {
          sub.items.push_back(SelectItem{innerCols[k], "__k" + std::to_string(k)});
          sub.groupBy.push_back(cloneExpr(innerCols[k]));
        }
        for (size_t k = 0; k < aggs.size(); ++k) sub.items.push_back(SelectItem{aggs[k], "__a" + std::to_string(k)});
        sub.orderBy.clear();
        sub.limit = -1;
        if (correlated(sub)) {
          applyPerOuterValue(e);
          return;
        }
        TableRef ref;
        ref.alias = alias;
        ref.sub = std::make_shared<Stmt>(std::move(sub));
        ref.join = 1;
        for (size_t k = 0; k < outerCols.size(); ++k) {
          ExprP eq = mkBin("=", mkCol(alias, "__k" + std::to_string(k)), cloneExpr(outerCols[k]));
          ref.on = ref.on ? mkBin("AND", ref.on, eq) : eq;
        }
        s.from.push_back(ref);
        e = outerExpr;
        return;
      }
      ResultSet r = query(sub, q, false, true);
      if (r.names.size() != 1) throw SqlError("Operand should contain 1 column(s)", 1241);
      if (r.vals.size() > 1) throw SqlError("Subquery returns more than 1 row", 1242);
      e = mkLit(r.vals.empty() ? Value::Null() : r.vals[0][0]);
      return;
    }
    if (e->kind == Expr::FUNC && e->op == "EXISTS") {
      Stmt sub = cloneStmt(*e->sub);
      if (correlated(sub)) {
        if (decorrelateExists(e)) {
          fix(e);
          return;
        } else {
          // EXISTS (.. WHERE <equalities with the outer query> AND inner.c <> outer.d AND <local filters>)
          //   = COUNT(.. equalities, filters, c IS NOT NULL) - COUNT(.. equalities, filters, c = d) > 0
          // (both equality-correlated counts, decorrelated into grouped LEFT JOINs).
          std::vector<ExprP> conj;
          std::function<void(const ExprP&)> flat = [&](const ExprP& x) {
            if (x && x->kind == Expr::BIN && x->op == "AND") { flat(x->args[0]); flat(x->args[1]); }
            else if (x) conj.push_back(x);
          };
          flat(sub.where);
          ExprP neq;
          std::vector<ExprP> keep;
          bool ok = true;
          for (auto& c : conj) {
            bool corr = false;
            std::function<void(const ExprP&)> chk = [&](const ExprP& x) {
              if (!x) return;
              if (x->kind == Expr::COL && outerRef(cat_, x, sub, s, q.ctes)) corr = true;
              for (auto& a : x->args) chk(a);
            };
            chk(c);
            const bool isEq = c->kind == Expr::BIN && c->op == "=" && c->args[0]->kind == Expr::COL && c->args[1]->kind == Expr::COL;
            if (corr && !isEq) {
              if (neq || !(c->kind == Expr::BIN && c->op == "<>" && c->args[0]->kind == Expr::COL && c->args[1]->kind == Expr::COL)) ok = false;
              else neq = c;
            } else {
              keep.push_back(c);
            }
          }
          if (!ok || !neq) {
            existsPerOuterValue(e);
            return;
          }
          // EXISTS (.. inner.c <> outer.d ..) <=> MIN(c) <> d OR MAX(c) <> d over the rows matching the
          // equalities and filters (if any c differs from d, the smallest or the largest does): one
          // grouped derived table on the equality keys, LEFT JOINed, instead of a correlated scan.
          const bool o0 = outerRef(cat_, neq->args[0], sub, s, q.ctes);
          const ExprP inner = o0 ? neq->args[1] : neq->args[0], outer = o0 ? neq->args[0] : neq->args[1];
          std::vector<ExprP> innerKeys, outerKeys, local;
          for (auto& k : keep) {
            const bool isEq = k->kind == Expr::BIN && k->op == "=" && k->args[0]->kind == Expr::COL && k->args[1]->kind == Expr::COL;
            const bool a0 = isEq && outerRef(cat_, k->args[0], sub, s, q.ctes), a1 = isEq && outerRef(cat_, k->args[1], sub, s, q.ctes);
            if (isEq && a0 != a1) {
              outerKeys.push_back(a0 ? k->args[0] : k->args[1]);
              innerKeys.push_back(a0 ? k->args[1] : k->args[0]);
            } else {
              local.push_back(k);
            }
          }
          if (innerKeys.empty()) {
            existsPerOuterValue(e);
            return;
          }
          Stmt ds = cloneStmt(sub);
          ExprP w;
          for (auto& l : local) w = w ? mkBin("AND", w, cloneExpr(l)) : cloneExpr(l);
          ds.where = w;
          ds.items.clear();
          ds.groupBy.clear();
          for (size_t k = 0; k < innerKeys.size(); ++k) {
            ds.items.push_back(SelectItem{cloneExpr(innerKeys[k]), "__k" + std::to_string(k)});
            ds.groupBy.push_back(cloneExpr(innerKeys[k]));
          }
          auto agg = [&](const char* fn) {
            auto f = std::make_shared<Expr>();
            f->kind = Expr::FUNC;
            f->op = fn;
            f->args = {cloneExpr(inner)};
            return f;
          };
          ds.items.push_back(SelectItem{agg("MIN"), "__mn"});
          ds.items.push_back(SelectItem{agg("MAX"), "__mx"});
          ds.orderBy.clear();
          ds.limit = -1;
          if (correlated(ds)) {
            existsPerOuterValue(e);
            return;
          }
          static std::atomic<int> ex{0};
          const std::string alias = "__ex" + std::to_string(++ex);
          TableRef ref;
          ref.alias = alias;
          ref.sub = std::make_shared<Stmt>(std::move(ds));
          ref.join = 1;
          for (size_t k = 0; k < outerKeys.size(); ++k) {
            ExprP eq = mkBin("=", mkCol(alias, "__k" + std::to_string(k)), cloneExpr(outerKeys[k]));
            ref.on = ref.on ? mkBin("AND", ref.on, eq) : eq;
          }
          s.from.push_back(ref);
          ExprP differs = mkBin("OR", mkBin("<>", mkCol(alias, "__mn"), cloneExpr(outer)), mkBin("<>", mkCol(alias, "__mx"), cloneExpr(outer)));
          auto co = std::make_shared<Expr>();
          co->kind = Expr::FUNC;
          co->op = "COALESCE";
          co->args = {differs, mkLit(Value::I(0))};
          e = co;
          return;
        }
      } else {
        sub.limit = 1;
        ResultSet r = query(sub, q, false, true);
        e = mkLit(Value::I(!r.vals.empty()));
        return;
      }
    }
    if (e->kind == Expr::INLIST && e->sub) {
      Stmt sub = cloneStmt(*e->sub);
      // x [NOT] IN (set): membership f; with a NULL in the set, "not found" is NULL (IF(f, 1, NULL)).
      auto setTest = [&](ExprP f, bool hasNull) {
        if (hasNull) {
          auto i = std::make_shared<Expr>();
          i->kind = Expr::FUNC;
          i->op = "IF";
          i->args = {f, mkLit(Value::I(1)), mkLit(Value::Null())};
          f = i;
        }
        if (!e->neg) return f;
        auto n = std::make_shared<Expr>();
        n->kind = Expr::UNARY;
        n->op = "NOT";
        n->args = {f};
        return ExprP(n);
      };
      if (correlated(sub)) {
        // x IN (SELECT c .. WHERE w) -> EXISTS (SELECT 1 .. WHERE w AND c = x); NOT IN ->
        // NOT EXISTS (.. AND (c = x OR c IS NULL OR x IS NULL)) (true only when nothing could match).
        bool agg = !sub.groupBy.empty() || sub.having || sub.items.size() != 1 || sub.limit >= 0 || sub.unionNext || sub.distinct;
        for (auto& it : sub.items) agg = agg || hasAgg(it.e);
        if (agg || e->args.size() != 1) throw SqlError("this correlated IN subquery is not supported", 1235);
        const ExprP c = sub.items[0].e;
        ExprP x = cloneExpr(e->args[0]);
        std::function<void(ExprP&)> qualify = [&](ExprP& y) {  // outer columns, named by their outer table
          if (!y) return;
          if (y->kind == Expr::COL && y->table.empty()) {
            for (auto& f : s.from) {
              Stmt one;
              one.from = {f};
              if (!f.sub && fromHasColumn(cat_, one, y->name)) {
                y->table = f.alias.empty() ? f.name : f.alias;
                break;
              }
            }
          }
          for (auto& a : y->args) qualify(a);
        };
        qualify(x);
        ExprP cond = mkBin("=", cloneExpr(c), cloneExpr(x));
        if (e->neg) {
          auto isNull = [](ExprP a) {
            auto n = std::make_shared<Expr>();
            n->kind = Expr::FUNC;
            n->op = "ISNULL";
            n->args = {a};
            return n;
          };
          cond = mkBin("OR", mkBin("OR", cond, isNull(cloneExpr(c))), isNull(cloneExpr(x)));
        }
        sub.where = sub.where ? mkBin("AND", sub.where, cond) : cond;
        auto ex = std::make_shared<Expr>();
        ex->kind = Expr::FUNC;
        ex->op = "EXISTS";
        ex->sub = std::make_shared<Stmt>(std::move(sub));
        ExprP r = ex;
        if (e->neg) {
          auto nt = std::make_shared<Expr>();
          nt->kind = Expr::UNARY;
          nt->op = "NOT";
          nt->args = {ex};
          r = nt;
        }
        e = r;
        fix(e);
        return;
      }
      // Integer values: the distinct values computed on the GPU stay there, as a bitmap the filter
      // tests (no host round trip), when their range is compact.
      std::shared_ptr<std::vector<double>> pre;  // else: the distinct numbers, if they came back to the host
      {
        bool plain = sub.items.size() == 1 && sub.groupBy.empty() && !sub.having && sub.limit < 0 && !sub.unionNext;
        for (auto& it : sub.items) plain = plain && !hasAgg(it.e) && !it.e->window;
        if (plain) {
          Stmt ds = cloneStmt(sub);
          ds.distinct = true;
          ds.orderBy.clear();
          tColumnar = true;
          ResultSet r;
          try {
            r = query(ds, q, false, true);
          } catch (...) {
            tColumnar = false;
            throw;
          }
          tColumnar = false;
          const bool intTy = !r.types.empty() && (r.types[0] == LType::INT || r.types[0] == LType::BIGINT || r.types[0] == LType::DATE);
          if (r.devCols.size() == 1 && intTy) {
            const auto& dc = r.devCols[0];
            const bool neg = e->neg;
            if (r.devRows == 0) { e = mkLit(Value::I(neg ? 1 : 0)); return; }
            if (!(dc->mn <= dc->mx)) {  // only NULL: never true; NULL unless x is .. (x IN (NULL) is NULL)
              e = setTest(mkLit(Value::I(0)), true);
              return;
            }
            const double span = dc->mx - dc->mn + 1;
            if (span <= (double)(1 << 28) && span <= 64.0 * (double)std::max<int64_t>(r.devRows, 1 << 16)) {
              auto b = std::make_shared<DevBits>();
              b->base = (int64_t)dc->mn;
              b->nbits = (int64_t)span;
              bool hasNull = false;
              b->p = gpuBitsFromColumn(dc->p, r.types[0] == LType::BIGINT ? 1 : 0, r.devRows, b->base, b->nbits, &hasNull);
              auto f = std::make_shared<Expr>();
              f->kind = Expr::FUNC;
              f->op = "#INBITS";
              f->args = {mkBin("-", e->args[0], mkLit(Value::I(b->base)))};
              f->devAux = b;
              e = setTest(f, hasNull);
              return;
            }
          }
          if (r.numCols.size() == 1 && !r.types.empty() && r.types[0] != LType::VARCHAR && r.devCols.empty())
            pre = std::make_shared<std::vector<double>>(std::move(r.numCols[0]));
        }
      }
      // Only the set of values matters: numbers come back as one raw array (no row objects).
      ResultSet r;
      if (pre) {
        r.numCol = pre;
      } else {
        tRawNumber = true;
        try {
          r = query(sub, q, false, true);
        } catch (...) {
          tRawNumber = false;
          throw;
        }
        tRawNumber = false;
      }
      if (r.numCol) {
        auto vals = r.numCol;
        const size_t before = vals->size();
        vals->erase(std::remove_if(vals->begin(), vals->end(), [](double v) { return v != v; }), vals->end());
        const bool neg = e->neg, hasNull = vals->size() != before;
        if (before == 0) { e = mkLit(Value::I(neg ? 1 : 0)); return; }
        if (vals->empty()) { e = setTest(mkLit(Value::I(0)), true); return; }
        // Sorted distinct values: integers in a compact range through a bitmap (no sort), else a sort.
        double lo = (*vals)[0], hi = lo;
        bool integral = true;
        for (double v : *vals) {
          lo = std::min(lo, v), hi = std::max(hi, v);
          integral = integral && v == std::floor(v);
        }
        if (integral && hi - lo < (double)(1 << 28)) {
          const int64_t base = (int64_t)lo, span = (int64_t)(hi - lo) + 1;
          std::vector<uint64_t> bits((size_t)(span + 63) / 64, 0);
          for (double v : *vals) {
            const int64_t k = (int64_t)v - base;
            bits[(size_t)k >> 6] |= 1ull << (k & 63);
          }
          vals->clear();
          for (size_t w = 0; w < bits.size(); ++w)
            for (uint64_t b = bits[w]; b; b &= b - 1) vals->push_back((double)(base + (int64_t)(w * 64 + __builtin_ctzll(b))));
        } else {
          std::sort(vals->begin(), vals->end());
          vals->erase(std::unique(vals->begin(), vals->end()), vals->end());
        }
        auto f = std::make_shared<Expr>();
        f->kind = Expr::FUNC;
        f->op = "#INSORTED";
        f->args = {e->args[0]};
        f->aux = vals;
        e = setTest(f, hasNull);
        return;
      }
      if (r.names.size() != 1) throw SqlError("Operand should contain 1 column(s)", 1241);
      const bool neg = e->neg;
      ExprP x = e->args[0];
      bool hasNull = false;
      for (auto& row : r.vals) hasNull = hasNull || row[0].isNull();
      if (r.vals.empty()) { e = mkLit(Value::I(neg ? 1 : 0)); return; }
      if (r.types[0] == LType::VARCHAR) {
        std::set<std::string> vs;
        for (auto& row : r.vals)
          if (!row[0].isNull()) vs.insert(row[0].toString());
        if (vs.empty()) { e = setTest(mkLit(Value::I(0)), true); return; }
        auto in = std::make_shared<Expr>();
        in->kind = Expr::INLIST;
        in->args = {x};
        for (auto& v : vs) in->args.push_back(mkLit(Value::S(v)));
        e = setTest(in, hasNull);
        return;
      }
      auto vals = std::make_shared<std::vector<double>>();
      for (auto& row : r.vals)
        if (!row[0].isNull()) vals->push_back(numOf(row[0]));
      if (vals->empty()) { e = setTest(mkLit(Value::I(0)), true); return; }
      std::sort(vals->begin(), vals->end());
      vals->erase(std::unique(vals->begin(), vals->end()), vals->end());
      auto f = std::make_shared<Expr>();
      f->kind = Expr::FUNC;
      f->op = "#INSORTED";
      f->args = {x};
      f->aux = vals;
      e = setTest(f, hasNull);
    }
  };
  walkStmtExprs(s, fix);
}


// SUM(x + c) / SUM(x - c) / SUM(c + x) -> SUM(x) +/- c * COUNT(x) (exact, NULLs included): many such
// aggregates over one column (ClickBench Q30) collapse to two.
static void rewriteSumOffsets(Stmt& s) {
  std::function<ExprP(const ExprP&)> rw = [&](const ExprP& e) -> ExprP {
    if (!e) return e;
    if (e->kind == Expr::FUNC && e->op == "SUM" && !e->window && !e->distinct && e->args.size() == 1 &&
        e->args[0]->kind == Expr::BIN && (e->args[0]->op == "+" || e->args[0]->op == "-")) {
      const ExprP& a = e->args[0]->args[0];
      const ExprP& b = e->args[0]->args[1];
      auto numLit = [](const ExprP& x) { return x->kind == Expr::LIT && (x->lit.kind == Value::INT || x->lit.kind == Value::DBL) && !x->lit.isNull(); };
      const bool cRight = numLit(b) && !numLit(a), cLeft = e->args[0]->op == "+" && numLit(a) && !numLit(b);
      if (cRight || cLeft) {
        const ExprP x = cRight ? a : b, c = cRight ? b : a;
        auto sum = std::make_shared<Expr>();
        sum->kind = Expr::FUNC;
        sum->op = "SUM";
        sum->args = {x};
        auto cnt = std::make_shared<Expr>();
        cnt->kind = Expr::FUNC;
        cnt->op = "COUNT";
        cnt->args = {x};
        return mkBin(e->args[0]->op, sum, mkBin("*", c, cnt));
      }
    }
    for (auto& a : e->args) a = rw(a);
    return e;
  };
  for (auto& it : s.items) {
    ExprP n = rw(it.e);
    if (n != it.e && it.alias.empty()) it.alias = it.e->str();
    it.e = n;
  }
  s.having = rw(s.having);
  for (auto& o : s.orderBy) o.e = rw(o.e);
}

// A derived table that only projects and filters one base table is merged into the outer query:
// no temporary table, and (on a cluster aggregator) the outer aggregation can run on the leaves.
void Engine::flattenDerived(Stmt& s) {
  while (s.from.size() == 1 && s.from[0].sub && s.joinConds.empty()) {
    Stmt& in = *s.from[0].sub;
    if (in.type != Stmt::SELECT || in.from.size() != 1 || in.from[0].sub || in.from[0].join || !in.joinConds.empty() ||
        !in.groupBy.empty() || in.having || in.distinct || in.keepDerived || in.limit >= 0 || in.unionNext || !in.ctes.empty() || in.items.empty() ||
        !cat_.find(in.from[0].name))
      return;
    std::function<bool(const ExprP&)> complex = [&](const ExprP& e) -> bool {
      if (!e) return false;
      if (e->kind == Expr::STAR || e->sub || e->kind == Expr::SUBQUERY || e->window || hasAgg(e)) return true;
      for (auto& a : e->args)
        if (complex(a)) return true;
      return false;
    };
    std::map<std::string, ExprP> byName;
    for (auto& it : in.items) {
      if (complex(it.e)) return;
      const std::string n = lower(!it.alias.empty() ? it.alias : it.e->kind == Expr::COL ? it.e->name : it.e->str());
      if (byName.count(n)) return;  // duplicate output names: leave it to the general path
      byName[n] = it.e;
    }
    std::set<std::string> outAliases;
    bool star = false;
    for (auto& it : s.items) {
      star = star || it.e->kind == Expr::STAR;
      if (!it.alias.empty()) outAliases.insert(lower(it.alias));
    }
    if (star) return;
    const std::string alias = s.from[0].alias;
    bool unknown = false;
    std::function<void(ExprP&, bool)> rw = [&](ExprP& e, bool orderLevel) {
      if (!e) return;
      if (e->kind == Expr::COL && (e->table.empty() || strcasecmp(e->table.c_str(), alias.c_str()) == 0)) {
        if (orderLevel && e->table.empty() && outAliases.count(lower(e->name))) return;  // a select-list alias
        auto it = byName.find(lower(e->name));
        if (it == byName.end()) { unknown = true; return; }
        e = cloneExpr(it->second);
        return;
      }
      if (e->sub) unknown = true;  // correlated references into the derived table: keep it
      for (auto& a : e->args) rw(a, false);
    };
    Stmt merged = cloneStmt(s);
    for (auto& it : merged.items) {
      if (it.alias.empty()) it.alias = it.e->kind == Expr::COL ? it.e->name : it.e->str();
      rw(it.e, false);
    }
    rw(merged.where, false);
    rw(merged.having, true);
    for (auto& g : merged.groupBy) rw(g, true);
    for (auto& o : merged.orderBy) rw(o.e, true);
    if (unknown) return;
    if (in.where) merged.where = merged.where ? mkBin("AND", cloneExpr(in.where), merged.where) : cloneExpr(in.where);
    merged.from = {in.from[0]};
    s = std::move(merged);
  }
}

// (a AND b AND x) OR (a AND b AND y) -> a AND b AND (x OR y): conditions every OR branch repeats
// (typically the join equality, TPC-H Q19) become plain conjuncts the planner can join on.
ExprP factorOr(const ExprP& e) {
  if (!e) return e;
  if (e->kind == Expr::BIN && e->op == "AND") {
    auto y = std::make_shared<Expr>(*e);
    y->args = {factorOr(e->args[0]), factorOr(e->args[1])};
    return y;
  }
  if (!(e->kind == Expr::BIN && e->op == "OR")) return e;
  std::vector<ExprP> branches;
  std::function<void(const ExprP&)> ors = [&](const ExprP& x) {
    if (x->kind == Expr::BIN && x->op == "OR") { ors(x->args[0]); ors(x->args[1]); }
    else branches.push_back(x);
  };
  ors(e);
  std::vector<std::vector<ExprP>> conj(branches.size());
  for (size_t i = 0; i < branches.size(); ++i) {
    std::function<void(const ExprP&)> ands = [&](const ExprP& x) {
      if (x->kind == Expr::BIN && x->op == "AND") { ands(x->args[0]); ands(x->args[1]); }
      else conj[i].push_back(x);
    };
    ands(branches[i]);
  }
  std::vector<ExprP> common;
  for (auto& c : conj[0]) {
    const std::string k = c->str();
    bool all = true;
    for (size_t i = 1; i < conj.size() && all; ++i)
      all = std::any_of(conj[i].begin(), conj[i].end(), [&](const ExprP& x) { return x->str() == k; });
    if (all && std::none_of(common.begin(), common.end(), [&](const ExprP& x) { return x->str() == k; })) common.push_back(c);
  }
  if (common.empty()) return e;
  ExprP rest;
  bool anyEmpty = false;
  for (auto& cs : conj) {
    ExprP b;
    for (auto& c : cs)
      if (std::none_of(common.begin(), common.end(), [&](const ExprP& x) { return x->str() == c->str(); })) b = b ? mkBin("AND", b, c) : c;
    if (!b) anyEmpty = true;
    else rest = rest ? mkBin("OR", rest, b) : b;
  }
  ExprP out;
  for (auto& c : common) out = out ? mkBin("AND", out, c) : c;
  if (!anyEmpty && rest) out = mkBin("AND", out, rest);  // (a branch that was only common terms: OR is true)
  return out;
}

// ROLLUP / CUBE / GROUPING SETS: one GROUP BY per set (keys outside the set are NULL there,
// GROUPING(k) is 1 for them), concatenated with UNION ALL; ORDER BY / LIMIT apply to the whole.
ResultSet Engine::groupingSetsQuery(Stmt& s, QueryCtx& q, bool explain, bool typed) {
  for (auto& it : s.items)
    if (it.alias.empty()) it.alias = it.e->kind == Expr::COL ? it.e->name : it.e->str();
  std::vector<OrderItem> outerOrder;
  for (auto& o : s.orderBy) {
    int pos = -1;
    if (o.e->kind == Expr::LIT && o.e->lit.kind == Value::INT) pos = (int)o.e->lit.i;
    for (size_t i = 0; i < s.items.size() && pos < 0; ++i)
      if ((o.e->kind == Expr::COL && o.e->table.empty() && strcasecmp(o.e->name.c_str(), s.items[i].alias.c_str()) == 0) ||
          o.e->str() == s.items[i].e->str())
        pos = (int)i + 1;
    if (pos < 0) throw SqlError("ORDER BY with ROLLUP / CUBE / GROUPING SETS must name a select-list column", 1235);
    outerOrder.push_back({mkLit(Value::I(pos)), o.desc});
  }
  auto sets = s.groupingSets;
  std::stable_sort(sets.begin(), sets.end(), [](const auto& a, const auto& b) { return a.size() > b.size(); });
  std::shared_ptr<Stmt> head;
  Stmt* last = nullptr;
  for (auto& set : sets) {
    auto b = std::make_shared<Stmt>(cloneStmt(s));
    b->groupingSets.clear();
    b->orderBy.clear();
    b->limit = -1;
    b->offset = 0;
    b->groupBy.clear();
    std::vector<std::string> in, out;
    for (size_t i = 0; i < s.groupBy.size(); ++i) {
      const bool keep = std::find(set.begin(), set.end(), (int)i) != set.end();
      (keep ? in : out).push_back(s.groupBy[i]->str());
      if (keep) b->groupBy.push_back(cloneExpr(s.groupBy[i]));
    }
    std::function<void(ExprP&)> rw = [&](ExprP& e) {
      if (!e) return;
      if (e->kind == Expr::FUNC && upper(e->op) == "GROUPING" && e->args.size() == 1) {
        const std::string k = e->args[0]->str();
        e = mkLit(Value::I(std::find(out.begin(), out.end(), k) != out.end() ? 1 : 0));
        return;
      }
      if (e->kind == Expr::FUNC && (isAggName(e->op) || isHostAggName(e->op))) return;  // aggregates keep their arguments
      if (std::find(out.begin(), out.end(), e->str()) != out.end()) {
        e = mkLit(Value::Null());
        return;
      }
      for (auto& a : e->args) rw(a);
    };
    for (auto& it : b->items) rw(it.e);
    rw(b->having);
    if (!head) head = b;
    else {
      last->unionNext = b;
      last->unionAll = true;
    }
    last = b.get();
  }
  Stmt outer;
  outer.type = Stmt::SELECT;
  auto star = std::make_shared<Expr>();
  star->kind = Expr::STAR;
  outer.items = {SelectItem{star, ""}};
  TableRef ref;
  ref.alias = "__gs";
  ref.sub = head;
  outer.from = {ref};
  outer.orderBy = outerOrder;
  outer.limit = s.limit;
  outer.offset = s.offset;
  return query(outer, q, explain, typed);
}

std::shared_ptr<Stmt> tvfStatement(const std::string& name, const std::vector<ExprP>& args);  // proc.cpp
void Engine::expandTableFunctions(Stmt& s) {
  for (auto& f : s.from)
    if (f.tvf) {
      f.sub = tvfStatement(f.name, f.tvfArgs);
      f.tvf = false;
      f.tvfArgs.clear();
      f.name.clear();
    }
}

void Engine::resolveNaturalJoins(Stmt& s) {
  auto colsOf = [&](const TableRef& f) {
    std::vector<std::string> out;
    if (f.sub) {
      for (auto& it : f.sub->items) out.push_back(it.alias.empty() ? it.e->name : it.alias);
    } else if (auto t = cat_.find(f.name)) {
      for (auto& c : t->cols) out.push_back(c.name);
    }
    return out;
  };
  for (size_t i = 1; i < s.from.size(); ++i) {
    if (!s.from[i].natural) continue;
    s.from[i].natural = false;
    ExprP cond;
    for (auto& c : colsOf(s.from[i]))
      for (size_t k = 0; k < i; ++k) {
        auto ck = colsOf(s.from[k]);
        if (std::none_of(ck.begin(), ck.end(), [&](const std::string& x) { return strcasecmp(x.c_str(), c.c_str()) == 0; })) continue;
        ExprP eq = mkBin("=", mkCol(s.from[k].alias, c), mkCol(s.from[i].alias, c));
        cond = cond ? mkBin("AND", cond, eq) : eq;
        s.from[i].usingCols.push_back(c);
        break;
      }
    if (!cond) continue;  // no shared columns: a cross join
    if (s.from[i].join == 0) s.joinConds.push_back(cond);
    else s.from[i].on = cond;
  }
}

// [LEFT] JOIN LATERAL (SELECT a, b ..) x ON TRUE, where the subquery yields at most one row (an
// aggregate, LIMIT 1: what ORMs such as Drizzle generate for relations): every x.a becomes the
// correlated scalar subquery (SELECT a ..), evaluated once per distinct outer value.
static void rewriteLateral(Stmt& s) {
  for (size_t i = 0; i < s.from.size(); ++i) {
    TableRef& f = s.from[i];
    if (!f.lateral || !f.sub) continue;
    const bool onTrue = !f.on || (f.on->kind == Expr::LIT && !f.on->lit.isNull() && f.on->lit.num() != 0);
    if (!onTrue || f.join >= 2) throw SqlError("LATERAL derived tables are supported with ON TRUE (inner or LEFT JOIN)", 1235);
    std::map<std::string, ExprP> repl;
    for (auto& it : f.sub->items) {
      if (it.e->kind == Expr::STAR) throw SqlError("SELECT * in a LATERAL derived table is not supported", 1235);
      Stmt one = cloneStmt(*f.sub);
      one.items = {SelectItem{cloneExpr(it.e), it.alias}};
      auto q = std::make_shared<Expr>();
      q->kind = Expr::SUBQUERY;
      q->sub = std::make_shared<Stmt>(std::move(one));
      repl[lower(!it.alias.empty() ? it.alias : it.e->kind == Expr::COL ? it.e->name : it.e->str())] = q;
    }
    const std::string alias = lower(f.alias);
    std::function<void(ExprP&)> fix = [&](ExprP& e) {
      if (!e) return;
      if (e->kind == Expr::COL && lower(e->table) == alias) {
        auto it = repl.find(lower(e->name));
        if (it == repl.end()) throw SqlError("Unknown column '" + e->table + "." + e->name + "' in 'field list'", 1054);
        auto c = std::make_shared<Expr>(*it->second);
        c->sub = std::make_shared<Stmt>(cloneStmt(*it->second->sub));
        e = c;
        return;
      }
      for (auto& a : e->args) fix(a);
    };
    s.from.erase(s.from.begin() + i);
    --i;
    walkStmtExprs(s, fix);
  }
}

// A single-table query whose WHERE / GROUP BY / aggregates use expressions the GPU can't run (host
// functions over several columns, ..) runs again as
//   SELECT .. FROM (SELECT *, f(a, b) AS __e0, .. FROM t WHERE ..) AS t GROUP BY .. :
// the inner query filters and evaluates them on the host over the rows the GPU produces.
static bool hostRewrite(const Stmt& s, Stmt& out) {
  if (s.from.empty() || s.unionNext || !s.groupingSets.empty() || !s.ctes.empty()) return false;
  for (auto& f : s.from)
    if (f.tvf) return false;
  const bool multi = s.from.size() > 1 || !s.joinConds.empty();
  out = cloneStmt(s);
  const std::string alias = multi ? std::string("__hj") : s.from[0].alias.empty() ? s.from[0].name : s.from[0].alias;
  auto inner = std::make_shared<Stmt>();
  inner->type = Stmt::SELECT;
  inner->keepDerived = true;  // (merging it back would undo the rewrite)
  auto star = std::make_shared<Expr>();
  star->kind = Expr::STAR;
  if (!multi) inner->items.push_back(SelectItem{star, ""});
  inner->from = out.from;
  inner->joinConds = out.joinConds;
  out.joinConds.clear();
  std::map<std::string, std::string> done;
  std::function<bool(const ExprP&)> hasCol = [&](const ExprP& x) {
    if (!x) return false;
    if (x->kind == Expr::COL) return true;
    for (auto& a : x->args)
      if (hasCol(a)) return true;
    return false;
  };
  std::function<bool(const ExprP&)> hasColFunc = [&](const ExprP& x) {
    if (!x || x->kind == Expr::SUBQUERY) return false;
    if (x->kind == Expr::FUNC && !isAgg(x) && hasCol(x)) return true;
    for (auto& a : x->args)
      if (hasColFunc(a)) return true;
    return false;
  };
  // Joins can't filter on the host: conjuncts with functions of columns move to the outer query.
  ExprP outerWhere;
  if (multi && out.where) {
    std::vector<ExprP> conj;
    std::function<void(const ExprP&)> split = [&](const ExprP& x) {
      if (x->kind == Expr::BIN && x->op == "AND") { split(x->args[0]); split(x->args[1]); }
      else conj.push_back(x);
    };
    split(out.where);
    ExprP in;
    for (auto& x : conj) {
      ExprP& dst = hasColFunc(x) ? outerWhere : in;
      dst = dst ? mkBin("AND", dst, x) : x;
    }
    inner->where = in;
  } else if (!multi) {
    inner->where = out.where;
  }
  if (multi) {  // (inner join conditions may hold WHERE conjuncts too)
    std::vector<ExprP> keep;
    for (auto& j : inner->joinConds) {
      if (hasColFunc(j)) outerWhere = outerWhere ? mkBin("AND", outerWhere, j) : j;
      else keep.push_back(j);
    }
    inner->joinConds = keep;
  }
  out.where = outerWhere;
  std::function<void(ExprP&)> lift = [&](ExprP& x) {
    if (!x || x->kind == Expr::SUBQUERY) return;
    if (x->kind == Expr::FUNC && !isAgg(x) && !hasAgg(x) && hasCol(x)) {
      const std::string k = x->str();
      auto it = done.find(k);
      if (it == done.end()) {
        it = done.emplace(k, "__e" + std::to_string(done.size())).first;
        inner->items.push_back(SelectItem{x, it->second});
      }
      auto c = std::make_shared<Expr>();
      c->kind = Expr::COL;
      c->name = it->second;
      c->table = alias;
      x = c;
      return;
    }
    for (auto& a : x->args) lift(a);
  };
  for (auto& it : out.items) {
    const bool bare = it.alias.empty() && it.e && it.e->kind == Expr::FUNC && !isAgg(it.e);
    const std::string name = it.e ? it.e->str() : "";
    lift(it.e);
    if (bare) it.alias = name;  // keep the column header
  }
  for (auto& g : out.groupBy) lift(g);
  lift(out.having);
  lift(out.where);
  for (auto& o : out.orderBy) lift(o.e);
  if (done.empty() && !inner->where) return false;
  if (!multi) {  // only the columns the outer query still uses (not *: every column would be evaluated on the host)
    std::set<std::string> itemAliases, lifted, cols;
    for (auto& it : out.items)
      if (!it.alias.empty()) itemAliases.insert(lower(it.alias));
    for (auto& [k, v] : done) lifted.insert(lower(v));
    bool star = false;
    std::function<void(const ExprP&)> scan = [&](const ExprP& x) {
      if (!x || x->kind == Expr::SUBQUERY) return;
      if (x->kind == Expr::COL) {
        const std::string n = lower(x->name);
        if (!lifted.count(n) && !(x->table.empty() && itemAliases.count(n))) cols.insert(x->name);
        return;
      }
      for (auto& a : x->args) scan(a);
    };
    for (auto& it : out.items) {
      if (it.e && it.e->kind == Expr::STAR) star = true;
      scan(it.e);
    }
    for (auto& g : out.groupBy) scan(g);
    scan(out.having);
    scan(out.where);
    for (auto& o : out.orderBy) scan(o.e);
    if (!star) {
      inner->items.erase(inner->items.begin());  // (the *)
      std::set<std::string> seen;
      for (auto& n : cols) {
        if (!seen.insert(lower(n)).second) continue;
        auto c = std::make_shared<Expr>();
        c->kind = Expr::COL;
        c->name = n;
        inner->items.push_back(SelectItem{c, ""});
      }
      if (inner->items.empty()) {
        auto one = std::make_shared<Expr>();
        one->kind = Expr::LIT;
        one->lit = Value::I(1);
        inner->items.push_back(SelectItem{one, "__one"});
      }
    }
  }
  if (multi) {  // the join's columns the outer query still uses, each under a unique name
    std::map<std::string, std::string> cols;
    std::set<std::string> itemAliases;  // GROUP BY / ORDER BY / HAVING may name select items
    for (auto& it : out.items)
      if (!it.alias.empty()) itemAliases.insert(lower(it.alias));
    bool star = false;
    std::function<void(ExprP&)> mapCols = [&](ExprP& x) {
      if (!x || x->kind == Expr::SUBQUERY) return;
      if (x->kind == Expr::STAR) return;  // COUNT(*) (a SELECT * item is checked below)
      if (x->kind == Expr::COL && x->table.empty() && itemAliases.count(lower(x->name))) return;
      if (x->kind == Expr::COL && x->table != alias) {
        const std::string k = (x->table.empty() ? "" : x->table + ".") + x->name;
        auto it = cols.find(k);
        if (it == cols.end()) {
          it = cols.emplace(k, "__c" + std::to_string(cols.size())).first;
          auto src = std::make_shared<Expr>(*x);
          inner->items.push_back(SelectItem{src, it->second});
        }
        const std::string orig = x->name;
        auto c = std::make_shared<Expr>();
        c->kind = Expr::COL;
        c->name = it->second;
        c->table = alias;
        x = c;
        return;
      }
      for (auto& a : x->args) mapCols(a);
    };
    for (auto& it : out.items) {
      if (it.e && it.e->kind == Expr::STAR) star = true;
      if (it.e && it.e->kind == Expr::COL && it.alias.empty()) it.alias = it.e->name;  // keep the column header
      mapCols(it.e);
    }
    for (auto& g : out.groupBy) mapCols(g);
    mapCols(out.having);
    mapCols(out.where);
    for (auto& o : out.orderBy) mapCols(o.e);
    if (star) return false;
    if (inner->items.empty()) inner->items.push_back(SelectItem{std::make_shared<Expr>(Expr{}), ""}), inner->items.back().e->kind = Expr::LIT;
  }
  TableRef d;
  d.sub = inner;
  d.alias = alias;
  out.from = {d};
  return true;
}

ResultSet Engine::query(Stmt& s, QueryCtx& q, bool explain, bool typed) {
  thread_local int tQueryDepth = 0;
  if (tQueryDepth == 0) cudaMemsetAsync(ovfFlagDev(), 0, sizeof(int), 0);
  ++tQueryDepth;
  struct QD { ~QD() { --tQueryDepth; } } qd;
  if (tQueryDepth == 1) {  // the outermost query: compiled int64 arithmetic that overflowed is an error (MySQL)
    ResultSet r = queryRetry(s, q, explain, typed);
    int ovf = 0;
    cudaMemcpy(&ovf, ovfFlagDev(), sizeof(int), cudaMemcpyDeviceToHost);
    if (ovf) throw SqlError("BIGINT value is out of range", 1690);
    return r;
  }
  return queryRetry(s, q, explain, typed);
}

ResultSet Engine::queryRetry(Stmt& s, QueryCtx& q, bool explain, bool typed) {
  thread_local int tHostRewrite = 0;
  Stmt orig;
  const bool canRetry = !explain && tHostRewrite == 0 && !s.from.empty() && !s.unionNext;
  if (canRetry) orig = cloneStmt(s);
  try {
    return queryImpl(s, q, explain, typed);
  } catch (SqlError& e) {
    Stmt re;
    if (!canRetry || e.code != 1235 || !hostRewrite(orig, re)) throw;
    ++tHostRewrite;
    struct Dec { int& n; ~Dec() { --n; } } dec{tHostRewrite};
    return queryImpl(re, q, explain, typed);
  }
}

ResultSet Engine::queryImpl(Stmt& s, QueryCtx& q, bool explain, bool typed) {
  // tRawNumber applies to this statement's own projection only, not to the queries run on the way.
  const bool rawNumber = tRawNumber && !s.unionNext && !explain, columnar = tColumnar && !s.unionNext && !explain;
  const bool insertCols = tInsertCols && !s.unionNext && !explain;
  tRawNumber = tColumnar = tInsertCols = false;
  struct RawRestore {
    bool v;
    ~RawRestore() { tRawNumber = tColumnar = tInsertCols = false; }
  } rawRestore{rawNumber};
  expandTableFunctions(s);
  if (!s.groupingSets.empty()) return groupingSetsQuery(s, q, explain, typed);
  rewriteLateral(s);
  resolveNaturalJoins(s);
  flattenDerived(s);
  s.where = factorOr(s.where);
  for (auto& j : s.joinConds) j = factorOr(j);
  rewriteSumOffsets(s);
  dropDependentKeys(s);
  auto savedCtes = q.ctes;
  struct Restore {
    QueryCtx& q;
    std::map<std::string, std::string> saved;
    ~Restore() { q.ctes = saved; }
  } restore{q, savedCtes};
  for (auto& [name, cs] : s.ctes) {
    // WITH RECURSIVE: anchor UNION [ALL] parts that reference the CTE itself, iterated to a fixpoint.
    bool recursive = false;
    for (Stmt* p = cs->unionNext.get(); p && !recursive; p = p->unionNext.get())
      for (auto& f : p->from) recursive |= !f.sub && strcasecmp(f.name.c_str(), name.c_str()) == 0;
    if (!recursive) {
      Stmt c = cloneStmt(*cs);
      ResultSet r = query(c, q, false, true);
      q.ctes[lower(name)] = makeTemp(r, q, name);
      continue;
    }
    Stmt anchor = cloneStmt(*cs);
    anchor.unionNext.reset();
    const bool all = cs->unionAll;
    ResultSet acc = query(anchor, q, false, true);
    std::set<std::string> seen;
    auto key = [](const std::vector<Value>& v) {
      std::string k;
      for (auto& x : v) k += (x.isNull() ? std::string("\x02") : x.toString()) + '\x1f';
      return k;
    };
    if (!all) {
      std::vector<std::vector<Value>> u;
      for (auto& v : acc.vals)
        if (seen.insert(key(v)).second) u.push_back(v);
      acc.vals.swap(u);
    }
    ResultSet delta = acc;
    for (int iter = 0; iter < 1000 && !delta.vals.empty(); ++iter) {
      q.ctes[lower(name)] = makeTemp(delta, q, name);
      Stmt rec = cloneStmt(*cs->unionNext);
      ResultSet r = query(rec, q, false, true);
      delta.vals.clear();
      for (auto& v : r.vals)
        if (all || seen.insert(key(v)).second) delta.vals.push_back(v);
      for (auto& v : delta.vals) acc.vals.push_back(v);
      if (iter == 999) throw SqlError("Recursive query aborted after 1000 iterations", 3636);
    }
    q.ctes[lower(name)] = makeTemp(acc, q, name);
  }
  if (s.unionNext) {
    std::vector<Stmt*> parts;
    for (Stmt* p = &s; p; p = p->unionNext.get()) parts.push_back(p);
    bool dedupe = false;
    size_t dedupeUpTo = 0;  // rows of parts [0, dedupeUpTo) are deduplicated (a UNION DISTINCT covers all to its left; UNION ALL after it doesn't)
    for (size_t i = 0; i + 1 < parts.size(); ++i)
      if (!parts[i]->unionAll || parts[i]->setOp != 0) dedupe = true, dedupeUpTo = i + 2;
    auto rowKey = [](const std::vector<Value>& v) {
      std::string k;
      for (auto& x : v) k += (x.isNull() ? std::string("\x02") : x.toString()) + '\x1f';
      return k;
    };
    ResultSet out;
    out.isResult = true;
    out.typed = true;
    bool allUnion = dedupe;  // UNION (no INTERSECT / EXCEPT): each part can drop its duplicates first, on the GPU
    for (size_t i = 0; i + 1 < parts.size(); ++i) allUnion = allUnion && parts[i]->setOp == 0 && !parts[i]->unionAll;
    std::vector<bool> typeKnown;
    size_t dedupeRows = 0;  // rows from the parts the deduplication covers
    for (size_t i = 0; i < parts.size(); ++i) {
      Stmt ps = cloneStmt(*parts[i]);
      ps.unionNext.reset();
      ps.ctes.clear();
      if (allUnion && ps.limit < 0 && !(i == 0 && s.unionOrder && !ps.orderBy.empty())) ps.distinct = true;
      if (i == 0 && s.unionOrder) {
        ps.orderBy.clear();
        ps.limit = -1;
        ps.offset = 0;
      }
      ResultSet r = query(ps, q, false, true);
      // A column that is NULL in every row of a part (a NULL literal, ROLLUP totals) takes the type of the others.
      auto allNull = [&](size_t c) {
        for (auto& v : r.vals)
          if (!v[c].isNull()) return false;
        return true;
      };
      if (i == 0) {
        out.names = r.names;
        out.types = r.types;
        typeKnown.assign(r.types.size(), false);
        for (size_t c = 0; c < r.types.size(); ++c) typeKnown[c] = !allNull(c);
      } else {
        if (r.names.size() != out.names.size())
          throw SqlError("The used SELECT statements have a different number of columns", 1222);
        for (size_t c = 0; c < out.types.size(); ++c) {
          LType a = out.types[c], b = r.types[c];
          if (a == b || allNull(c)) continue;
          if (!typeKnown[c]) {
            out.types[c] = b;
            typeKnown[c] = true;
            continue;
          }
          bool na = a == LType::INT || a == LType::BIGINT || a == LType::DOUBLE, nb = b == LType::INT || b == LType::BIGINT || b == LType::DOUBLE;
          out.types[c] = na && nb ? (a == LType::DOUBLE || b == LType::DOUBLE ? LType::DOUBLE : LType::BIGINT) : LType::VARCHAR;
        }
      }
      const int op = i == 0 ? 0 : parts[i - 1]->setOp;
      if (op == 0) {
        for (auto& v : r.vals) out.vals.push_back(std::move(v));
        if (i < dedupeUpTo) dedupeRows = out.vals.size();
      } else {  // INTERSECT / EXCEPT against the rows accumulated so far
        std::unordered_set<std::string> other;
        other.reserve(r.vals.size() * 2);
        for (auto& v : r.vals) other.insert(rowKey(v));
        std::vector<std::vector<Value>> keep;
        for (auto& v : out.vals)
          if ((other.count(rowKey(v)) > 0) == (op == 1)) keep.push_back(std::move(v));
        out.vals.swap(keep);
        if (i < dedupeUpTo) dedupeRows = out.vals.size();
      }
    }
    if (dedupe) {
      // keys built on all cores, then a hash set (the first of equal rows stays, in order)
      const size_t m = std::min(dedupeRows, out.vals.size());
      std::vector<std::string> keyOf(m);
      parallelFor(m, [&](size_t b, size_t e) {
        for (size_t i = b; i < e; ++i) keyOf[i] = rowKey(out.vals[i]);
      });
      std::unordered_set<std::string_view> seen;
      seen.reserve(m * 2);
      std::vector<std::vector<Value>> keep;
      for (size_t i = 0; i < m; ++i)
        if (seen.insert(keyOf[i]).second) keep.push_back(std::move(out.vals[i]));
      for (size_t i = m; i < out.vals.size(); ++i) keep.push_back(std::move(out.vals[i]));  // (after the last UNION DISTINCT: kept)
      out.vals.swap(keep);
    }
    if (!s.orderBy.empty()) {
      std::vector<std::pair<int, bool>> keys;
      for (auto& o : s.orderBy) {
        int idx = -1;
        if (o.e->kind == Expr::LIT && o.e->lit.kind == Value::INT) idx = (int)o.e->lit.i - 1;
        else if (o.e->kind == Expr::COL)
          for (size_t c = 0; c < out.names.size(); ++c)
            if (strcasecmp(out.names[c].c_str(), o.e->name.c_str()) == 0) idx = (int)c;
        if (idx < 0 || idx >= (int)out.names.size()) throw SqlError("Unknown column '" + o.e->str() + "' in 'order clause'", 1054);
        keys.push_back({idx, o.desc});
      }
      std::stable_sort(out.vals.begin(), out.vals.end(), [&](const std::vector<Value>& a, const std::vector<Value>& b) {
        for (auto& [i, d] : keys) {
          int c = cmpVal(a[i], b[i]);
          if (c) return d ? c > 0 : c < 0;
        }
        return false;
      });
    }
    if (s.offset > 0 || s.limit >= 0) {
      size_t st = std::min<size_t>(s.offset, out.vals.size());
      size_t en = s.limit < 0 ? out.vals.size() : std::min<size_t>(out.vals.size(), st + s.limit);
      out.vals = std::vector<std::vector<Value>>(out.vals.begin() + st, out.vals.begin() + en);
    }
    if (!typed) formatRows(out);
    return out;
  }
  // FROM: derived tables, CTE references, views and information_schema become temp tables.
  auto resolveFrom = [&]() {
  std::string viewDb;
  for (auto& f : s.from) {
    if (!f.sub && !q.ctes.count(lower(f.name))) {
      if (lower(f.name).rfind("information_schema.", 0) == 0 || (lower(currentDb()) == "information_schema" && f.name.find('.') == std::string::npos)) {
        ResultSet r = infoSchema(plainName(f.name));
        f.name = makeTemp(r, q, "is_" + plainName(f.name));
        continue;
      }
      std::string view;
      {
        std::lock_guard<std::mutex> g(cat_.mu);
        auto it = cat_.views.find(Catalog::key(f.name));
        if (it != cat_.views.end()) view = it->second, viewDb = it->first.substr(0, it->first.find('.'));
      }
      if (!view.empty()) {
        std::vector<std::string> names;
        if (view.rfind("/*cols:", 0) == 0) {
          size_t e = view.find("*/");
          std::string list = view.substr(7, e - 7);
          for (size_t a = 0, b; a <= list.size(); a = b + 1) {
            b = list.find(',', a);
            if (b == std::string::npos) b = list.size();
            names.push_back(list.substr(a, b - a));
          }
          view = view.substr(e + 2);
        }
        f.sub = std::make_shared<Stmt>(parseSql(view));
        for (size_t i = 0; i < names.size() && i < f.sub->items.size(); ++i) f.sub->items[i].alias = names[i];
      }
    }
    if (f.sub) {
      Stmt sub = cloneStmt(*f.sub);
      tColumnar = true;  // a numeric GROUP BY result may come back as columns (straight into the temp table)
      tInsertCols = true;  // ... and a projection too
      struct DbSwitch {  // a view's tables resolve in the view's database
        std::string saved;
        bool on;
        DbSwitch(const std::string& d) : saved(currentDb()), on(!d.empty()) { if (on) currentDb() = d; }
        ~DbSwitch() { if (on) currentDb() = saved; }
      } sw(viewDb);
      viewDb.clear();
      ResultSet r = query(sub, q, false, true);
      tColumnar = tInsertCols = false;
      f.name = makeTemp(r, q, f.alias);
      f.sub.reset();
    } else {
      auto it = q.ctes.find(lower(f.name));
      if (it != q.ctes.end()) f.name = it->second;
    }
  }
  };
  resolveFrom();
  resolveSubqueries(s, q);
  resolveFrom();  // derived tables added by subquery decorrelation
  {
    bool win = false;
    std::function<void(const ExprP&)> find = [&](const ExprP& e) {
      if (!e || win) return;
      if (e->window) win = true;
      for (auto& a : e->args) find(a);
    };
    for (auto& it : s.items) find(it.e);
    for (auto& o : s.orderBy) find(o.e);
    if (win) {
      tWinOuterCols = insertCols;  // (its final projection may come back as columns)
      return windows(s, q, explain, typed);
    }
  }
  {  // aggregates only the host computes (GROUP_CONCAT, JSON_ARRAYAGG, STDDEV, ..): host aggregation
    bool hostAgg = false;
    std::function<void(const ExprP&)> find = [&](const ExprP& e) {
      if (!e || hostAgg) return;
      if (e->kind == Expr::FUNC && isHostAggName(e->op) && !e->window) hostAgg = true;
      for (auto& a : e->args) find(a);
    };
    for (auto& it : s.items) find(it.e);
    find(s.having);
    for (auto& o : s.orderBy) find(o.e);
    if (hostAgg) return hostAggregate(s, q, explain, typed);
  }
  // DISTINCT aggregates (COUNT/SUM/AVG(DISTINCT x), mixed with ordinary aggregates): two-level
  // aggregation. The inner query groups by the GROUP BY keys plus x and computes partial aggregates;
  // the outer one aggregates x (now distinct per group) and combines the partials.
  std::vector<ExprP> dist;
  std::vector<ExprP> plainAggs;
  std::function<void(const ExprP&)> findAggs = [&](const ExprP& e) {
    if (!e) return;
    if (e->kind == Expr::FUNC && isAggName(e->op) && !e->window) {
      (e->distinct ? dist : plainAggs).push_back(e);
      return;
    }
    for (auto& a : e->args) findAggs(a);
  };
  for (auto& it : s.items) findAggs(it.e);
  findAggs(s.having);
  for (auto& o : s.orderBy) findAggs(o.e);
  bool gpuDistinct = !dist.empty() && s.from.size() == 1 && !s.from[0].sub && s.from[0].join == 0;
  for (auto& d : dist) gpuDistinct &= d->op == "COUNT" && d->args[0]->kind != Expr::STAR && d->args[0]->str() == dist[0]->args[0]->str();
  if (!dist.empty() && !gpuDistinct) {
    const std::string x = dist[0]->args[0]->str();
    bool mixed = false;
    for (auto& d : dist) mixed = mixed || d->args[0]->str() != x;
    if (mixed && s.groupBy.empty() && !s.having && !s.unionNext) {
      // Without GROUP BY, each DISTINCT aggregate is a scalar subquery over the same rows (each runs on
      // the GPU); with no other aggregate left, the outer query needs no rows at all.
      Stmt outer = cloneStmt(s);
      std::function<void(ExprP&)> rw = [&](ExprP& e) {
        if (!e) return;
        if (e->kind == Expr::FUNC && isAggName(e->op) && !e->window && e->distinct) {
          auto sub = std::make_shared<Stmt>();
          sub->type = Stmt::SELECT;
          sub->from = s.from;
          sub->joinConds = s.joinConds;
          sub->where = s.where ? cloneExpr(s.where) : nullptr;
          sub->items.push_back({cloneExpr(e), "v"});
          auto sq = std::make_shared<Expr>();
          sq->kind = Expr::SUBQUERY;
          sq->sub = sub;
          e = sq;
          return;
        }
        for (auto& a : e->args) rw(a);
      };
      for (auto& it : outer.items) {
        if (it.alias.empty()) it.alias = it.e->str();
        rw(it.e);
      }
      for (auto& o : outer.orderBy) rw(o.e);
      bool agg = false;
      for (auto& it : outer.items) agg = agg || hasAgg(it.e);
      if (!agg) {
        outer.from.clear();
        outer.joinConds.clear();
        outer.where.reset();
      }
      return queryImpl(outer, q, explain, typed);
    }
    if (mixed && !explain && !s.unionNext && s.groupingSets.empty()) {
      try {
        Stmt copy = cloneStmt(s);
        return multiDistinct(copy, q, typed);
      } catch (SqlError&) {
      }
    }
    if (mixed) return hostAggregate(s, q, explain, typed);  // mixed DISTINCT with GROUP BY: host path
    for (auto& d : dist)
      if (d->op == "MIN" || d->op == "MAX") d->distinct = false;  // MIN/MAX(DISTINCT x) = MIN/MAX(x)
    auto resolveGroup = [&](const ExprP& g) -> ExprP {
      if (g->kind == Expr::LIT && g->lit.kind == Value::INT && g->lit.i >= 1 && g->lit.i <= (int64_t)s.items.size())
        return s.items[g->lit.i - 1].e;
      if (g->kind == Expr::COL && g->table.empty())
        for (auto& it : s.items)
          if (!it.alias.empty() && strcasecmp(it.alias.c_str(), g->name.c_str()) == 0) return it.e;
      return g;
    };
    std::vector<ExprP> groups;
    for (auto& g : s.groupBy) groups.push_back(resolveGroup(g));
    Stmt inner = cloneStmt(s);
    inner.items.clear();
    inner.groupBy.clear();
    for (size_t i = 0; i < groups.size(); ++i) {
      inner.items.push_back({cloneExpr(groups[i]), "__g" + std::to_string(i)});
      inner.groupBy.push_back(cloneExpr(groups[i]));
    }
    inner.items.push_back({cloneExpr(dist[0]->args[0]), "__d"});
    inner.groupBy.push_back(cloneExpr(dist[0]->args[0]));
    // Partial aggregates of the ordinary aggregates, per (keys, x).
    std::map<std::string, ExprP> outerOf;  // canonical text of an ordinary aggregate -> its outer form
    auto mkAgg = [](const std::string& fn, ExprP arg) {
      auto f = std::make_shared<Expr>();
      f->kind = Expr::FUNC;
      f->op = fn;
      f->args = {arg};
      return f;
    };
    int np = 0;
    for (auto& a : plainAggs) {
      const std::string k = a->str();
      if (outerOf.count(k)) continue;
      auto part = [&](const std::string& fn, const ExprP& arg) {
        std::string name = "__p" + std::to_string(np++);
        inner.items.push_back({mkAgg(fn, arg->kind == Expr::STAR ? arg : cloneExpr(arg)), name});
        return mkCol("", name);
      };
      ExprP o;
      if (a->op == "COUNT") {
        auto c = std::make_shared<Expr>();
        c->kind = Expr::FUNC;
        c->op = "COALESCE";
        c->args = {mkAgg("SUM", part("COUNT", a->args[0])), mkLit(Value::I(0))};
        o = c;
      } else if (a->op == "SUM") o = mkAgg("SUM", part("SUM", a->args[0]));
      else if (a->op == "MIN") o = mkAgg("MIN", part("MIN", a->args[0]));
      else if (a->op == "MAX") o = mkAgg("MAX", part("MAX", a->args[0]));
      else {  // AVG = SUM / COUNT
        ExprP sm = mkAgg("SUM", part("SUM", a->args[0])), cn = mkAgg("SUM", part("COUNT", a->args[0]));
        o = mkBin("/", sm, cn);
      }
      outerOf[k] = o;
    }
    inner.having.reset();
    inner.orderBy.clear();
    inner.limit = -1;
    inner.offset = 0;
    inner.distinct = false;
    tColumnar = true;  // numeric / dictionary-code GROUP BY results come back as columns
    ResultSet r;
    try {
      r = query(inner, q, false, true);
    } catch (...) {
      tColumnar = false;
      throw;
    }
    tColumnar = false;
    std::string tmp = makeTemp(r, q, "dist");
    std::function<void(ExprP&)> rw = [&](ExprP& e) {
      if (!e) return;
      for (size_t i = 0; i < groups.size(); ++i)
        if (e->str() == groups[i]->str()) { e = mkCol("", "__g" + std::to_string(i)); return; }
      if (e->kind == Expr::FUNC && isAggName(e->op)) {
        if (e->distinct) {
          auto c = std::make_shared<Expr>(*e);
          c->distinct = false;
          c->args = {mkCol("", "__d")};
          e = c;
        } else {
          e = cloneExpr(outerOf.at(e->str()));
        }
        return;
      }
      for (auto& a : e->args) rw(a);
    };
    for (auto& it : s.items) {
      if (it.alias.empty()) it.alias = it.e->str();
      rw(it.e);
    }
    rw(s.having);
    for (auto& o : s.orderBy) rw(o.e);
    s.groupBy.clear();
    for (size_t i = 0; i < groups.size(); ++i) s.groupBy.push_back(mkCol("", "__g" + std::to_string(i)));
    s.from = {TableRef{tmp, "__dist", nullptr, 0, nullptr}};
    s.joinConds.clear();
    s.where.reset();
  }
  bool savedWant = tWantVals;
  tWantVals = typed;
  ResultSet out;
  std::unique_ptr<Stmt> pristine;
  if (!s.groupBy.empty() || s.distinct) pristine = std::make_unique<Stmt>(cloneStmt(s));
  try {
    tRawNumber = rawNumber;
    tColumnar = columnar;
    tInsertCols = insertCols;
    out = doSelect(s, explain);
    tRawNumber = tColumnar = tInsertCols = false;
  } catch (SqlError& e) {
    tWantVals = savedWant;
    const std::string msg = e.what();
    // GROUP BY the GPU can't run (a string key it can't group, more than 8 keys): on the host.
    if (pristine && (msg.find("GROUP BY on this string") != std::string::npos || msg.find("GROUP BY expressions are supported") != std::string::npos)) {
      if (pristine->groupBy.empty() && pristine->distinct) {  // SELECT DISTINCT ..: GROUP BY its items
        bool star = false;
        for (auto& it : pristine->items) star = star || it.e->kind == Expr::STAR;
        if (star) {  // (DISTINCT *: every column of the one table)
          if (pristine->from.size() != 1 || pristine->from[0].sub || !pristine->joinConds.empty()) throw;
          auto t = cat_.get(pristine->from[0].name);
          pristine->items.clear();
          for (auto& c : t->cols) {
            auto col = std::make_shared<Expr>();
            col->kind = Expr::COL;
            col->name = c.name;
            pristine->items.push_back({col, ""});
          }
        }
        for (auto& it : pristine->items)
          if (hasAgg(it.e)) throw;
        for (auto& it : pristine->items) pristine->groupBy.push_back(cloneExpr(it.e));
        pristine->distinct = false;
      }
      if (msg.find("GROUP BY expressions are supported") != std::string::npos) {
        Stmt packed = cloneStmt(*pristine);
        if (packGroupKeys(packed)) {  // integer keys packed into one: on the GPU after all
          tWantVals = typed;
          try {
            out = doSelect(packed, explain);
            tWantVals = savedWant;
            return out;
          } catch (SqlError&) {
            tWantVals = savedWant;
          }
        }
      }
      return hostAggregate(*pristine, q, explain, typed);
    }
    throw;
  } catch (...) {
    tWantVals = savedWant;
    throw;
  }
  tWantVals = savedWant;
  if (!typed) formatRows(out);
  return out;
}

void Engine::expandStars(Stmt& s) {
  std::vector<SelectItem> items;
  for (auto& it : s.items) {
    if (it.e->kind != Expr::STAR) {
      items.push_back(it);
      continue;
    }
    for (auto& f : s.from) {
      if (!it.e->table.empty() && strcasecmp(it.e->table.c_str(), f.alias.c_str()) != 0) continue;
      auto t = cat_.get(f.name);
      for (auto& c : t->cols) items.push_back({mkCol(f.alias, c.name), c.name});
    }
  }
  s.items = std::move(items);
}

// Window functions: run the query without them (keeping every expression they need as hidden
// columns), compute the windows over the result on the host, then run the final projection,
// ORDER BY and LIMIT over a temp table with the results.
ResultSet Engine::windows(Stmt& s, QueryCtx& q, bool explain, bool typed) {
  const bool outerCols = tWinOuterCols;
  tWinOuterCols = false;
  expandStars(s);
  Stmt inner = cloneStmt(s);
  inner.items.clear();
  inner.orderBy.clear();
  inner.limit = -1;
  inner.offset = 0;
  inner.distinct = false;
  std::map<std::string, int> hiddenOf;
  auto hidden = [&](const ExprP& e) -> int {
    std::string k = e->str();
    auto it = hiddenOf.find(k);
    if (it != hiddenOf.end()) return it->second;
    int i = (int)inner.items.size();
    inner.items.push_back({cloneExpr(e), "__c" + std::to_string(i)});
    hiddenOf[k] = i;
    return i;
  };
  struct Win {
    ExprP node;
    std::vector<int> args, part;
    std::vector<std::pair<int, bool>> ord;
  };
  std::vector<Win> wins;
  std::map<const Expr*, int> winOf;
  std::function<ExprP(const ExprP&)> outerOf = [&](const ExprP& e) -> ExprP {
    if (e->window) {
      auto it = winOf.find(e.get());
      int w;
      if (it != winOf.end()) {
        w = it->second;
      } else {
        Win wn;
        wn.node = e;
        for (auto& a0 : e->args) {
          if (a0->kind == Expr::STAR) continue;
          // (MIN / MAX over a string bound before: its argument became the string's rank; the
          // window needs the strings themselves)
          const ExprP a = (e->op == "MIN" || e->op == "MAX") && a0->kind == Expr::FUNC && a0->op == "#MAP" && a0->args.size() == 1 ? a0->args[0] : a0;
          wn.args.push_back(hidden(a));
        }
        for (auto& p : e->partBy) wn.part.push_back(hidden(p));
        for (auto& [o, d] : e->winOrder) wn.ord.push_back({hidden(o), d});
        w = (int)wins.size();
        wins.push_back(wn);
        winOf[e.get()] = w;
      }
      return mkCol("", "__w" + std::to_string(w));
    }
    if (e->kind == Expr::COL || (e->kind == Expr::FUNC && isAggName(e->op))) return mkCol("", "__c" + std::to_string(hidden(e)));
    for (auto& g : inner.groupBy)  // a GROUP BY expression: as a whole (its columns alone are not grouped)
      if (g->str() == e->str()) return mkCol("", "__c" + std::to_string(hidden(e)));
    auto y = std::make_shared<Expr>(*e);
    y->bound = false;
    for (auto& a : y->args) a = outerOf(a);
    return y;
  };
  Stmt outer;
  outer.type = Stmt::SELECT;
  for (auto& it : s.items) outer.items.push_back({outerOf(it.e), it.alias.empty() ? (it.e->kind == Expr::COL ? it.e->name : it.e->str()) : it.alias});
  for (auto& o : s.orderBy) {
    bool alias = o.e->kind == Expr::COL && o.e->table.empty() &&
                 std::any_of(s.items.begin(), s.items.end(), [&](const SelectItem& it) { return !it.alias.empty() && strcasecmp(it.alias.c_str(), o.e->name.c_str()) == 0; });
    bool pos = o.e->kind == Expr::LIT && o.e->lit.kind == Value::INT;
    outer.orderBy.push_back({alias || pos ? o.e : outerOf(o.e), o.desc});
  }
  outer.limit = s.limit;
  outer.offset = s.offset;
  outer.distinct = s.distinct;
  if (inner.items.empty()) inner.items.push_back({mkLit(Value::I(1)), "__c0"});
  const auto tW0 = Clock::now();
  Stmt innerRows = cloneStmt(inner);  // (in case the result comes back as numbers only: run again for rows)
  tColumnar = true;  // a GROUP BY inner query may come back built on the GPU (exact, typed columns)
  tInsertCols = true;  // ... and a projection as typed columns
  ResultSet r = query(inner, q, false, true);
  tColumnar = tInsertCols = false;
  if (r.devCols.empty() && !r.insertCols && (!r.numCols.empty() || r.numCol)) r = query(innerRows, q, false, true);
  if ((!r.devCols.empty() || r.insertCols) && !explain) {
    // Columnar: the inner result as a temp table; keys, sort and window values from its typed
    // columns (64-bit integer keys, strings by collation rank); the output appended a column at a time.
    const std::string srcName = makeTemp(r, q, "winsrc");
    auto src = cat_.get(srcName);
    const size_t nin = src->cols.size();
    const size_t n = src->segs.empty() ? 0 : (size_t)src->segs[0]->n;
    std::vector<const uint8_t*> raw(nin, nullptr);
    if (n)
      for (size_t c = 0; c < nin; ++c) raw[c] = src->colData(Loc{0, 0}, (int)c).data();
    auto isF = [&](int c) { return src->cols[c].pt == PType::F64; };
    auto rawI = [&](int c, size_t i) -> int64_t {
      return src->cols[c].pt == PType::I32 ? (int64_t)((const int32_t*)raw[c])[i] : ((const int64_t*)raw[c])[i];
    };
    auto rawD = [&](int c, size_t i) { return ((const double*)raw[c])[i]; };
    auto isNullAt = [&](int c, size_t i) {
      if (isF(c)) return rawD(c, i) != rawD(c, i);
      return rawI(c, i) == (src->cols[c].pt == PType::I32 ? (int64_t)kNullI32 : kNullI64);
    };
    struct KeyCol {
      bool dbl = false;
      std::vector<int64_t> i;
      std::vector<double> d;
    };
    std::map<int, KeyCol> keys;
    auto keyCol = [&](int col) -> const KeyCol& {
      auto it = keys.find(col);
      if (it != keys.end()) return it->second;
      KeyCol k;
      const ColumnMeta& cm = src->cols[col];
      std::shared_ptr<std::vector<double>> rank;
      if (cm.lt == LType::VARCHAR) rank = sortedDict(*src->dicts[col]).second;
      if (isF(col)) {
        k.dbl = true;
        k.d.resize(n);
        parallelFor(n, [&](size_t b, size_t e) {
          for (size_t i = b; i < e; ++i) k.d[i] = isNullAt(col, i) ? -HUGE_VAL : rawD(col, i);
        });
      } else {
        k.i.resize(n);
        parallelFor(n, [&](size_t b, size_t e) {
          for (size_t i = b; i < e; ++i) {
            if (isNullAt(col, i)) k.i[i] = INT64_MIN;
            else k.i[i] = rank ? (int64_t)(*rank)[(size_t)rawI(col, i)] : rawI(col, i);
          }
        });
      }
      return keys[col] = std::move(k);
    };
    std::vector<std::vector<int64_t>> outI;   // per window: integer / code results
    std::vector<std::vector<double>> outD;    // per window: double results
    std::vector<ColumnMeta> outMeta;
    std::vector<std::shared_ptr<Dict>> outDict;
    std::vector<size_t> lastOrderC;
    for (size_t w = 0; w < wins.size(); ++w) {
      const Win& wn = wins[w];
      const std::string f = wn.node->op;
      static const std::set<std::string> kFns = {"ROW_NUMBER", "RANK", "DENSE_RANK", "PERCENT_RANK", "CUME_DIST", "NTILE", "LAG", "LEAD",
                                                 "FIRST_VALUE", "LAST_VALUE", "NTH_VALUE", "SUM", "AVG", "COUNT", "MIN", "MAX"};
      if (!kFns.count(f)) throw SqlError("window function " + f + " is not supported", 1235);
      std::vector<const KeyCol*> pk;
      std::vector<std::pair<const KeyCol*, bool>> ok;
      for (int p : wn.part) pk.push_back(&keyCol(p));
      for (auto& [o, d] : wn.ord) ok.push_back({&keyCol(o), d});
      auto cmpK = [](const KeyCol& k, size_t a, size_t b) { return k.dbl ? (k.d[a] < k.d[b] ? -1 : k.d[a] > k.d[b]) : (k.i[a] < k.i[b] ? -1 : k.i[a] > k.i[b]); };
      auto cmpKeys = [&](size_t a, size_t b, bool withOrder) {
        for (auto* k : pk)
          if (int c = cmpK(*k, a, b)) return c;
        if (withOrder)
          for (auto& [k, d] : ok)
            if (int c = cmpK(*k, a, b)) return d ? -c : c;
        return 0;
      };
      std::vector<size_t> idx(n);
      for (size_t i = 0; i < n; ++i) idx[i] = i;
      {  // stable sort on all cores: sorted chunks, then pairwise merges
        auto less = [&](size_t a, size_t b) { return cmpKeys(a, b, true) < 0; };
        const size_t nth = std::max<size_t>(1, std::min<size_t>(std::thread::hardware_concurrency(), n / 32768));
        std::vector<size_t> bounds;
        for (size_t t = 0; t <= nth; ++t) bounds.push_back(n * t / nth);
        parallelFor(nth, [&](size_t b, size_t e) {
          for (size_t t = b; t < e; ++t) std::stable_sort(idx.begin() + bounds[t], idx.begin() + bounds[t + 1], less);
        }, 1);
        for (size_t step = 1; step < nth; step *= 2) {
          std::vector<size_t> starts;
          for (size_t t = 0; t + step < nth; t += 2 * step) starts.push_back(t);
          parallelFor(starts.size(), [&](size_t b, size_t e) {
            for (size_t k = b; k < e; ++k) {
              size_t t = starts[k];
              std::inplace_merge(idx.begin() + bounds[t], idx.begin() + bounds[t + step], idx.begin() + bounds[std::min(nth, t + 2 * step)], less);
            }
          }, 1);
        }
      }
      // Result type: ranks and counts BIGINT, ratios / AVG DOUBLE, SUM by its argument, value
      // functions the argument's own type (strings: codes of its dictionary).
      const int a0 = wn.args.empty() ? -1 : wn.args[0];
      ColumnMeta om;
      om.name = "__w" + std::to_string(w);
      om.lt = LType::BIGINT;
      std::shared_ptr<Dict> od;
      const bool decArg = a0 >= 0 && src->cols[a0].lt == LType::DECIMAL;
      const double argUnit = decArg ? pow10i(src->cols[a0].scale) : 1.0;  // (DECIMAL cells are scaled integers)
      if (f == "PERCENT_RANK" || f == "CUME_DIST" || f == "AVG") om.lt = LType::DOUBLE;
      else if (f == "SUM") {
        om.lt = a0 >= 0 && isF(a0) ? LType::DOUBLE : decArg ? LType::DECIMAL : LType::BIGINT;
        if (decArg) om.scale = src->cols[a0].scale;
      } else if ((f == "MIN" || f == "MAX" || f == "LAG" || f == "LEAD" || f == "FIRST_VALUE" || f == "LAST_VALUE" || f == "NTH_VALUE") && a0 >= 0) {
        om.lt = src->cols[a0].lt;
        om.scale = src->cols[a0].scale;
        if (om.lt == LType::VARCHAR) od = src->dicts[a0];
      }
      om.pt = physOf(om.lt);
      const bool fOut = om.pt == PType::F64;
      std::vector<int64_t> oi(fOut ? 0 : n);
      std::vector<double> od2(fOut ? n : 0);
      const int64_t onul = om.pt == PType::I32 ? (int64_t)kNullI32 : kNullI64;
      auto constArg = [&](size_t k, double def) {
        if (k >= wn.node->args.size()) return def;
        const ExprP& a = wn.node->args[k];
        return a->kind == Expr::LIT ? numOf(a->lit) : def;
      };
      // a cell of the argument column into the output (same physical type, or converted)
      auto putArg = [&](size_t row, int col, size_t at) {
        if (col < 0 || isNullAt(col, at)) {
          if (fOut) od2[row] = nullv();
          else oi[row] = onul;
          return;
        }
        if (fOut) od2[row] = isF(col) ? rawD(col, at) : (double)rawI(col, at);
        else oi[row] = isF(col) ? (int64_t)std::llround(rawD(col, at)) : rawI(col, at);
      };
      const KeyCol* valKey = a0 >= 0 && (f == "MIN" || f == "MAX") ? &keyCol(a0) : nullptr;  // (strings: by rank)
      std::vector<std::pair<size_t, size_t>> parts;
      for (size_t a = 0; a < n;) {
        size_t b = a + 1;
        while (b < n && cmpKeys(idx[a], idx[b], false) == 0) ++b;
        parts.push_back({a, b});
        a = b;
      }
      parallelFor(parts.size(), [&](size_t pb, size_t pe) {
        for (size_t pi = pb; pi < pe; ++pi) {
          const size_t a = parts[pi].first, b = parts[pi].second, cnt = b - a;
          std::vector<size_t> peerStart(cnt), peerEnd(cnt);
          for (size_t i = 0; i < cnt;) {
            size_t j = i + 1;
            while (j < cnt && cmpKeys(idx[a + i], idx[a + j], true) == 0) ++j;
            for (size_t k = i; k < j; ++k) peerStart[k] = i, peerEnd[k] = j;
            i = j;
          }
          // running aggregates over frames that start at the partition (prefix sums), or per frame
          double pAcc = 0;
          int64_t pAccI = 0, pCnt = 0;
          size_t pBest = SIZE_MAX, pUpTo = 0;
          int64_t dense = 0;
          for (size_t i = 0; i < cnt; ++i) {
            const size_t row = idx[a + i];
            if (i == 0 || peerStart[i] == i) ++dense;
            size_t lo, hi;
            if (wn.node->rowsFrame) {
              int64_t L = wn.node->frameLo == INT64_MIN ? 0 : std::max<int64_t>(0, (int64_t)i + wn.node->frameLo);
              int64_t H = wn.node->frameHi == INT64_MAX ? (int64_t)cnt : std::min<int64_t>((int64_t)cnt, (int64_t)i + wn.node->frameHi + 1);
              lo = (size_t)L;
              hi = (size_t)std::max<int64_t>(L, H);
            } else if (wn.ord.empty()) {
              lo = 0, hi = cnt;
            } else {
              lo = 0, hi = wn.node->frameHi == INT64_MAX ? cnt : peerEnd[i];
            }
            if (f == "ROW_NUMBER") oi[row] = (int64_t)i + 1;
            else if (f == "RANK") oi[row] = (int64_t)peerStart[i] + 1;
            else if (f == "DENSE_RANK") oi[row] = dense;
            else if (f == "PERCENT_RANK") od2[row] = cnt > 1 ? (double)peerStart[i] / (double)(cnt - 1) : 0.0;
            else if (f == "CUME_DIST") od2[row] = (double)peerEnd[i] / (double)cnt;
            else if (f == "NTILE") {
              int64_t k = std::max<int64_t>(1, (int64_t)constArg(0, 1)), sz = (int64_t)cnt / k, rem = (int64_t)cnt % k;
              int64_t pos = (int64_t)i, bucket = pos < rem * (sz + 1) ? pos / (sz + 1) : rem + (pos - rem * (sz + 1)) / std::max<int64_t>(sz, 1);
              oi[row] = bucket + 1;
            } else if (f == "LAG" || f == "LEAD") {
              int64_t off = (int64_t)constArg(1, 1), j = f == "LAG" ? (int64_t)i - off : (int64_t)i + off;
              if (j >= 0 && j < (int64_t)cnt) putArg(row, a0, idx[a + j]);
              else putArg(row, wn.args.size() > 2 ? wn.args[2] : -1, row);
            } else if (f == "FIRST_VALUE") {
              if (hi > lo) putArg(row, a0, idx[a + lo]);
              else putArg(row, -1, row);
            } else if (f == "LAST_VALUE") {
              if (hi > lo) putArg(row, a0, idx[a + hi - 1]);
              else putArg(row, -1, row);
            } else if (f == "NTH_VALUE") {
              size_t k = (size_t)std::max<double>(1, constArg(1, 1));
              if (lo + k - 1 < hi) putArg(row, a0, idx[a + lo + k - 1]);
              else putArg(row, -1, row);
            } else {  // SUM / AVG / COUNT / MIN / MAX over [lo, hi)
              const bool star = a0 < 0;
              double acc = 0;
              int64_t accI = 0, c = 0;
              size_t best = SIZE_MAX;
              auto take = [&](size_t at, double& sacc, int64_t& sI, int64_t& sc, size_t& sbest) {
                if (star) { ++sc; return; }
                if (isNullAt(a0, at)) return;
                ++sc;
                if (isF(a0)) sacc += rawD(a0, at);
                else sI += rawI(a0, at), sacc += (double)rawI(a0, at);
                if (valKey) {
                  const int cmp = sbest == SIZE_MAX ? 1 : cmpK(*valKey, at, sbest);
                  if (sbest == SIZE_MAX || (f == "MIN" ? cmp < 0 : cmp > 0)) sbest = at;
                }
              };
              if (lo == 0) {  // a prefix of the partition: extend the running totals
                for (; pUpTo < hi; ++pUpTo) take(idx[a + pUpTo], pAcc, pAccI, pCnt, pBest);
                acc = pAcc, accI = pAccI, c = pCnt, best = pBest;
              } else {
                for (size_t j = lo; j < hi; ++j) take(idx[a + j], acc, accI, c, best);
              }
              if (f == "COUNT") oi[row] = c;
              else if (c == 0) putArg(row, -1, row);
              else if (f == "SUM") {
                if (fOut) od2[row] = acc;
                else oi[row] = accI;  // (DECIMAL: the scaled sum, same scale)
              } else if (f == "AVG") od2[row] = acc / (double)c / argUnit;
              else putArg(row, a0, best);
            }
          }
        }
      }, std::max<size_t>(1, parts.size() / (4 * std::thread::hardware_concurrency())));
      if (w + 1 == wins.size() && s.orderBy.empty() && !s.distinct && (!wn.part.empty() || !wn.ord.empty())) lastOrderC = idx;
      outI.push_back(std::move(oi));
      outD.push_back(std::move(od2));
      outMeta.push_back(om);
      outDict.push_back(od);
    }
    // The output table: the inner columns, the window values (and the window order, without ORDER BY).
    std::vector<ColumnMeta> cols = src->cols;
    for (auto& m : outMeta) cols.push_back(m);
    if (!lastOrderC.empty()) {
      ColumnMeta m;
      m.name = "__wpos";
      m.lt = LType::BIGINT;
      m.pt = PType::I64;
      cols.push_back(m);
    }
    static std::atomic<int> wseq{0};
    const std::string name = "__tmp" + std::to_string(++wseq) + "_wincol";
    auto t = std::make_shared<Table>(name, cols, -1, -1, std::vector<int>{}, std::vector<int>{}, 1, false);
    t->db = "__tmp";
    std::vector<std::vector<int64_t>> iv(cols.size());
    std::vector<std::vector<double>> dv(cols.size());
    for (size_t c = 0; c < nin; ++c) {
      if (src->cols[c].lt == LType::VARCHAR) t->dicts[c] = src->dicts[c];
      if (isF((int)c)) dv[c].assign((const double*)raw[c], (const double*)raw[c] + n);
      else {
        iv[c].resize(n);
        for (size_t i = 0; i < n; ++i) iv[c][i] = rawI((int)c, i);
      }
    }
    for (size_t w = 0; w < outMeta.size(); ++w) {
      const size_t c = nin + w;
      if (outDict[w]) t->dicts[c] = outDict[w];
      if (outMeta[w].pt == PType::F64) dv[c] = std::move(outD[w]);
      else iv[c] = std::move(outI[w]);
    }
    if (!lastOrderC.empty()) {
      auto& pos = iv[cols.size() - 1];
      pos.resize(n);
      for (size_t j = 0; j < n; ++j) pos[lastOrderC[j]] = (int64_t)j;
      outer.orderBy.push_back({mkCol("__win", "__wpos"), false});
    }
    if (n) {
      t->appendColumns(iv, dv, (int64_t)n);
      t->flush(true);
    }
    {
      std::lock_guard<std::mutex> g(cat_.mu);
      cat_.tables["__tmp." + lower(name)] = t;
      q.temps.push_back("__tmp." + lower(name));
    }
    if (envSetting("TRACE")) fprintf(stderr, "[trace] windows (columnar): %.1f ms, %zu rows\n", msSince(tW0), n);
    outer.from = {TableRef{"__tmp." + name, "__win", nullptr, 0, nullptr}};
    tInsertCols = outerCols;
    return query(outer, q, explain, typed);
  }
  const auto tW1 = Clock::now();

  const size_t n = r.vals.size(), base = r.names.size();
  std::vector<size_t> lastOrder;  // rows in the last window's (PARTITION BY, ORDER BY) order
  for (size_t w = 0; w < wins.size(); ++w) {
    const Win& wn = wins[w];
    const std::string f = wn.node->op;
    std::vector<size_t> idx(n);
    for (size_t i = 0; i < n; ++i) idx[i] = i;
    // Sort keys as doubles (strings: rank among the distinct values; NULL lowest), so the sort and
    // the partition / peer scans compare numbers instead of Values.
    std::map<int, std::vector<double>> keys;
    auto keyCol = [&](int col) -> const std::vector<double>& {
      auto it = keys.find(col);
      if (it != keys.end()) return it->second;
      std::vector<double> k(n);
      bool str = false;
      for (size_t i = 0; i < n && !str; ++i) str = r.vals[i][col].kind == Value::STR;
      if (str) {  // rank the distinct strings, then look each row up
        std::unordered_map<std::string_view, size_t> id;
        std::vector<size_t> first;
        std::vector<uint32_t> rowId(n, 0);
        for (size_t i = 0; i < n; ++i) {
          const Value& v = r.vals[i][col];
          if (v.isNull()) continue;
          auto [it, fresh] = id.try_emplace(v.s, first.size());
          if (fresh) first.push_back(i);
          rowId[i] = (uint32_t)it->second;
        }
        std::vector<size_t> o(first.size());
        for (size_t j = 0; j < o.size(); ++j) o[j] = j;
        std::sort(o.begin(), o.end(), [&](size_t a, size_t b) { return cmpVal(r.vals[first[a]][col], r.vals[first[b]][col]) < 0; });
        std::vector<double> rank(o.size());
        double rk = 0;
        for (size_t j = 0; j < o.size(); ++j) {
          if (j && cmpVal(r.vals[first[o[j - 1]]][col], r.vals[first[o[j]]][col]) != 0) ++rk;
          rank[o[j]] = rk;
        }
        for (size_t i = 0; i < n; ++i)
          if (!r.vals[i][col].isNull()) k[i] = rank[rowId[i]];
      }
      parallelFor(n, [&](size_t b, size_t e) {
        for (size_t i = b; i < e; ++i)
          if (r.vals[i][col].isNull()) k[i] = -HUGE_VAL;
          else if (!str) k[i] = numOf(r.vals[i][col]);
      });
      return keys[col] = std::move(k);
    };
    std::vector<const std::vector<double>*> pk;
    std::vector<std::pair<const std::vector<double>*, bool>> ok;
    for (int p : wn.part) pk.push_back(&keyCol(p));
    for (auto& [o, d] : wn.ord) ok.push_back({&keyCol(o), d});
    auto cmpKeys = [&](size_t a, size_t b, bool withOrder) {
      for (auto* k : pk) {
        double x = (*k)[a], y = (*k)[b];
        if (x != y) return x < y ? -1 : 1;
      }
      if (withOrder)
        for (auto& [k, d] : ok) {
          double x = (*k)[a], y = (*k)[b];
          if (x != y) return (x < y) != d ? -1 : 1;
        }
      return 0;
    };
    const auto tK = Clock::now();
    {  // stable sort on all cores: sorted chunks, then pairwise merges
      auto less = [&](size_t a, size_t b) { return cmpKeys(a, b, true) < 0; };
      const size_t nth = std::max<size_t>(1, std::min<size_t>(std::thread::hardware_concurrency(), n / 32768));
      std::vector<size_t> bounds;
      for (size_t t = 0; t <= nth; ++t) bounds.push_back(n * t / nth);
      parallelFor(nth, [&](size_t b, size_t e) {
        for (size_t t = b; t < e; ++t) std::stable_sort(idx.begin() + bounds[t], idx.begin() + bounds[t + 1], less);
      }, 1);
      for (size_t step = 1; step < nth; step *= 2) {
        std::vector<size_t> starts;
        for (size_t t = 0; t + step < nth; t += 2 * step) starts.push_back(t);
        parallelFor(starts.size(), [&](size_t b, size_t e) {
          for (size_t k = b; k < e; ++k) {
            size_t t = starts[k];
            std::inplace_merge(idx.begin() + bounds[t], idx.begin() + bounds[t + step], idx.begin() + bounds[std::min(nth, t + 2 * step)], less);
          }
        }, 1);
      }
    }
    if (envSetting("TRACE")) fprintf(stderr, "[trace] window %zu: keys+sort %.1f ms (sort %.1f)\n", w, msSince(tW1), msSince(tK));
    std::vector<Value> out(n);
    LType ty = LType::BIGINT;
    const LType argTy = wn.args.empty() ? LType::BIGINT : r.types[wn.args[0]];
    if (f == "PERCENT_RANK" || f == "CUME_DIST" || f == "AVG") ty = LType::DOUBLE;
    else if (f == "SUM") ty = argTy == LType::DOUBLE ? LType::DOUBLE : LType::BIGINT;
    else if (f == "MIN" || f == "MAX" || f == "LAG" || f == "LEAD" || f == "FIRST_VALUE" || f == "LAST_VALUE" || f == "NTH_VALUE") ty = argTy;
    auto argv = [&](size_t row, size_t k) -> Value { return k < wn.args.size() ? r.vals[row][wn.args[k]] : Value::Null(); };
    auto constArg = [&](size_t k, double def) {  // LAG/LEAD offset, NTILE buckets, NTH_VALUE n: constants
      if (k >= wn.node->args.size()) return def;
      const ExprP& a = wn.node->args[k];
      return a->kind == Expr::LIT ? numOf(a->lit) : def;
    };
    static const std::set<std::string> kWinFns = {"ROW_NUMBER", "RANK", "DENSE_RANK", "PERCENT_RANK", "CUME_DIST", "NTILE", "LAG", "LEAD",
                                                  "FIRST_VALUE", "LAST_VALUE", "NTH_VALUE", "SUM", "AVG", "COUNT", "MIN", "MAX"};
    if (!kWinFns.count(f)) throw SqlError("window function " + f + " is not supported", 1235);
    // Partitions are independent: computed on all cores.
    std::vector<std::pair<size_t, size_t>> parts;
    for (size_t a = 0; a < n;) {
      size_t b = a + 1;
      while (b < n && cmpKeys(idx[a], idx[b], false) == 0) ++b;
      parts.push_back({a, b});
      a = b;
    }
    parallelFor(parts.size(), [&](size_t pb, size_t pe) {
    for (size_t pi = pb; pi < pe; ++pi) {
      const size_t a = parts[pi].first, b = parts[pi].second;
      const size_t cnt = b - a;
      // peer groups (equal ORDER BY keys)
      std::vector<size_t> peerStart(cnt), peerEnd(cnt);
      for (size_t i = 0; i < cnt;) {
        size_t j = i + 1;
        while (j < cnt && cmpKeys(idx[a + i], idx[a + j], true) == 0) ++j;
        for (size_t k = i; k < j; ++k) peerStart[k] = i, peerEnd[k] = j;
        i = j;
      }
      int64_t dense = 0;
      for (size_t i = 0; i < cnt; ++i) {
        const size_t row = idx[a + i];
        if (i == 0 || peerStart[i] == i) ++dense;
        // frame [lo, hi) within the partition
        size_t lo, hi;
        if (wn.node->rowsFrame) {
          int64_t L = wn.node->frameLo == INT64_MIN ? 0 : std::max<int64_t>(0, (int64_t)i + wn.node->frameLo);
          int64_t H = wn.node->frameHi == INT64_MAX ? (int64_t)cnt : std::min<int64_t>((int64_t)cnt, (int64_t)i + wn.node->frameHi + 1);
          lo = (size_t)L;
          hi = (size_t)std::max<int64_t>(L, H);
        } else if (wn.ord.empty()) {
          lo = 0, hi = cnt;
        } else {
          lo = 0, hi = wn.node->frameHi == INT64_MAX ? cnt : peerEnd[i];
        }
        Value v;
        if (f == "ROW_NUMBER") v = Value::I((int64_t)i + 1);
        else if (f == "RANK") v = Value::I((int64_t)peerStart[i] + 1);
        else if (f == "DENSE_RANK") v = Value::I(dense);
        else if (f == "PERCENT_RANK") v = Value::D(cnt > 1 ? (double)peerStart[i] / (double)(cnt - 1) : 0.0);
        else if (f == "CUME_DIST") v = Value::D((double)peerEnd[i] / (double)cnt);
        else if (f == "NTILE") {
          int64_t k = std::max<int64_t>(1, (int64_t)constArg(0, 1)), sz = (int64_t)cnt / k, rem = (int64_t)cnt % k;
          int64_t pos = (int64_t)i, bucket = pos < rem * (sz + 1) ? pos / (sz + 1) : rem + (pos - rem * (sz + 1)) / std::max<int64_t>(sz, 1);
          v = Value::I(bucket + 1);
        } else if (f == "LAG" || f == "LEAD") {
          int64_t off = (int64_t)constArg(1, 1), j = f == "LAG" ? (int64_t)i - off : (int64_t)i + off;
          v = j >= 0 && j < (int64_t)cnt ? argv(idx[a + j], 0) : (wn.args.size() > 2 ? argv(row, 2) : Value::Null());
        } else if (f == "FIRST_VALUE") v = hi > lo ? argv(idx[a + lo], 0) : Value::Null();
        else if (f == "LAST_VALUE") v = hi > lo ? argv(idx[a + hi - 1], 0) : Value::Null();
        else if (f == "NTH_VALUE") {
          size_t k = (size_t)std::max<double>(1, constArg(1, 1));
          v = lo + k - 1 < hi ? argv(idx[a + lo + k - 1], 0) : Value::Null();
        } else if (f == "SUM" || f == "AVG" || f == "COUNT" || f == "MIN" || f == "MAX") {
          double acc = 0;
          int64_t c = 0;
          Value best = Value::Null();  // (a default Value is the number 0, not NULL)
          bool star = wn.args.empty();
          for (size_t j = lo; j < hi; ++j) {
            if (star) { ++c; continue; }
            Value x = argv(idx[a + j], 0);
            if (x.isNull()) continue;
            ++c;
            acc += numOf(x);
            if (best.isNull() || (f == "MIN" ? cmpVal(x, best) < 0 : cmpVal(x, best) > 0)) best = x;
          }
          if (f == "COUNT") v = Value::I(c);
          else if (c == 0) v = Value::Null();
          else if (f == "SUM") v = ty == LType::DOUBLE ? Value::D(acc) : Value::I((int64_t)std::llround(acc));
          else if (f == "AVG") v = Value::D(acc / c);
          else v = best;
        } else {
          throw SqlError("window function " + f + " is not supported", 1235);
        }
        out[row] = v;
      }
    }
    }, std::max<size_t>(1, parts.size() / (4 * std::thread::hardware_concurrency())));
    if (w + 1 == wins.size() && s.orderBy.empty() && !s.distinct && (!wn.part.empty() || !wn.ord.empty())) lastOrder = idx;
    r.names.push_back("__w" + std::to_string(w));
    r.types.push_back(ty);
    parallelFor(n, [&](size_t b, size_t e) {
      for (size_t i = b; i < e; ++i) {
        if (w == 0) r.vals[i].reserve(base + wins.size());
        r.vals[i].push_back(std::move(out[i]));
      }
    });
  }
  (void)base;
  if (!lastOrder.empty()) {  // no ORDER BY: rows come out in window order, as in MySQL (ORMs rely on it)
    std::vector<int64_t> pos(n);
    for (size_t j = 0; j < n; ++j) pos[lastOrder[j]] = (int64_t)j;
    r.names.push_back("__wpos");
    r.types.push_back(LType::BIGINT);
    for (size_t i = 0; i < n; ++i) r.vals[i].push_back(Value::I(pos[i]));
    outer.orderBy.push_back({mkCol("__win", "__wpos"), false});
  }
  const auto tW2 = Clock::now();
  std::string tmp = makeTemp(r, q, "win");
  if (envSetting("TRACE"))
    fprintf(stderr, "[trace] windows: inner %.1f ms, compute %.1f ms, temp %.1f ms (%zu rows)\n", msSince(tW0) - msSince(tW1),
            msSince(tW1) - msSince(tW2), msSince(tW2), n);
  outer.from = {TableRef{tmp, "__win", nullptr, 0, nullptr}};
  return query(outer, q, explain, typed);
}

void Engine::stopPipeline(Pipeline& p) {
  p.running = false;
  if (p.th.joinable()) p.th.join();
  std::lock_guard<std::mutex> g(p.mu);
  p.state = "Stopped";
}

void Engine::startPipeline(const std::shared_ptr<Pipeline>& p) {
  if (p->running) return;
  {
    std::lock_guard<std::mutex> g(p->mu);
    for (auto& f : replayedLoads_) p->done.insert(f);
    p->state = "Running";
  }
  p->running = true;
  if (p->glob.rfind("MYSQL\x1e", 0) == 0) {  // MySQL CDC: snapshot, then follow the binlog
    p->th = std::thread([this, p] {
      currentDb() = p->db;
      const std::string url = p->glob.substr(6, p->glob.find('\x1e', 6) - 6);  // MYSQL \x1e url \x1e ..
      try {
        mysqlCdcRun(*this, url, p->table, p->name, p->running, true, [&](const CdcStatus& st) {
          std::lock_guard<std::mutex> lk(p->mu);
          p->state = st.state;
          p->lastError = st.error;
          p->rows = st.rows;
          p->files = st.txns;
        });
      } catch (std::exception& e) {
        std::lock_guard<std::mutex> lk(p->mu);
        p->state = "Error";
        p->lastError = e.what();
      }
    });
    return;
  }
  if (p->glob.find('\x1e') != std::string::npos) {
    p->th = std::thread([this, p] {
      currentDb() = p->db;
      runRemotePipeline(p);
    });
    return;
  }
  p->th = std::thread([this, p] {
    currentDb() = p->db;
    while (p->running) {
      glob_t g{};
      std::vector<std::string> files;
      if (::glob(p->glob.c_str(), 0, nullptr, &g) == 0)
        for (size_t i = 0; i < g.gl_pathc; ++i) files.push_back(g.gl_pathv[i]);
      globfree(&g);
      std::sort(files.begin(), files.end());
      for (auto& f : files) {
        if (!p->running) break;
        {
          std::lock_guard<std::mutex> lk(p->mu);
          if (p->done.count(f)) continue;
        }
        std::string sql = "LOAD DATA INFILE '" + f + "' INTO TABLE " + p->table + " FIELDS TERMINATED BY '" +
                          (p->delim == "\t" ? std::string("\\t") : p->delim) + "' IGNORE " + std::to_string(p->skip) + " LINES" +
                          (p->cols.empty() ? "" : " (" + p->cols + ")");
        try {
          ResultSet r = execute(sql);
          std::lock_guard<std::mutex> lk(p->mu);
          p->done.insert(f);
          ++p->files;
          p->rows += r.affected;
        } catch (const std::exception& e) {
          std::lock_guard<std::mutex> lk(p->mu);
          p->done.insert(f);  // don't retry a bad file forever
          p->lastError = f + ": " + e.what();
          p->state = "Running (with errors)";
        }
      }
      for (int i = 0; i < 10 && p->running; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
  });
}

ResultSet Engine::doPipeline(Stmt& s) {
  const std::string key = lower(s.table);
  if (s.setName == "CREATE") {
    const Stmt& l = *s.inner;
    if (l.setValue != "FS" && l.setValue != "S3" && l.setValue != "KAFKA" && l.setValue != "LINK" && l.setValue != "MYSQL")
      throw SqlError("pipelines from " + l.setValue + " are not supported (use FS, S3, KAFKA or MYSQL)", 1235);
    if (l.setValue == "MYSQL" && l.format != "DATABASE") throw SqlError("LOAD DATA MYSQL loads INTO DATABASE <name>", 1064);
    if (l.setValue != "MYSQL") cat_.get(l.table);  // the target table must exist
    auto p = std::make_shared<Pipeline>();
    p->name = s.table;
    p->db = currentDb();
    p->table = l.setValue == "MYSQL" ? lower(l.table) : l.table.find('.') == std::string::npos ? currentDb() + "." + l.table : l.table;
    p->glob = l.path;
    if (l.setValue != "FS" || l.format == "JSON") {  // remote source: packed spec (see unpackSource)
      std::string paths;
      for (auto& jp : l.jsonPaths) paths += (paths.empty() ? "" : "\x1f") + jp;
      p->glob = l.setValue + "\x1e" + l.path + "\x1e" + l.config + "\x1e" + l.credentials + "\x1e" + l.format + "\x1e" + paths;
    }
    p->delim = std::string(1, l.delim);
    p->skip = l.skipLines;
    for (auto& c : l.insertCols) p->cols += (p->cols.empty() ? "" : ", ") + c;
    std::lock_guard<std::mutex> g(pipesMu_);
    if (pipes_.count(key)) {
      if (s.ifNotExists) return okResult(0);
      throw SqlError("Pipeline '" + s.table + "' already exists", 1050);
    }
    pipes_[key] = p;
    return okResult(0);
  }
  std::shared_ptr<Pipeline> p;
  {
    std::lock_guard<std::mutex> g(pipesMu_);
    auto it = pipes_.find(key);
    if (it == pipes_.end()) {
      if (s.ifExists) return okResult(0);
      throw SqlError("Pipeline '" + s.table + "' doesn't exist", 1146);
    }
    p = it->second;
  }
  if (s.setName == "START") {
    p->wantRun = true;
    if (!replaying_ && !isReplica_) startPipeline(p);  // replicas get the pipeline's loads from the WAL
  } else if (s.setName == "STOP") {
    p->wantRun = false;
    stopPipeline(*p);
  } else if (s.setName == "DROP") {
    stopPipeline(*p);
    std::lock_guard<std::mutex> g(pipesMu_);
    pipes_.erase(key);
  }
  return okResult(0);
}

namespace {
struct SourceSpec {
  std::string src, path, config, creds, format;
  std::vector<std::string> jsonPaths;
};
SourceSpec unpackSource(const std::string& g) {
  std::vector<std::string> f;
  for (size_t a = 0;;) {
    size_t b = g.find('\x1e', a);
    f.push_back(g.substr(a, b == std::string::npos ? std::string::npos : b - a));
    if (b == std::string::npos) break;
    a = b + 1;
  }
  f.resize(6);
  SourceSpec s{f[0], f[1], f[2], f[3], f[4], {}};
  for (size_t a = 0; !f[5].empty();) {
    size_t b = f[5].find('\x1f', a);
    s.jsonPaths.push_back(f[5].substr(a, b == std::string::npos ? std::string::npos : b - a));
    if (b == std::string::npos) break;
    a = b + 1;
  }
  return s;
}
std::string sanitizeName(const std::string& s) {
  std::string o;
  for (char c : s) o += isalnum((unsigned char)c) || c == '.' || c == '-' || c == '_' ? c : '_';
  return o.size() > 180 ? o.substr(o.size() - 180) : o;
}
std::string baseName(const std::string& p) { return p.substr(p.rfind('/') + 1); }
}  // namespace

int64_t Engine::ingest(const std::string& table, char delim, int64_t skip, const std::vector<std::string>& cols,
                       const std::vector<std::string>& jsonPaths, bool json, const std::string& data, const std::string& stagedName) {
  std::string body = data;
  std::vector<std::string> loadCols = cols;
  bool escaped = false;  // the staged text uses backslash escapes (Parquet values can hold tabs / newlines)
  const bool parquet = data.size() >= 12 && data.compare(0, 4, "PAR1") == 0 && data.compare(data.size() - 4, 4, "PAR1") == 0;
  const bool avro = data.size() >= 4 && data.compare(0, 4, std::string("Obj\x01", 4)) == 0;
  const bool orc = data.size() >= 16 && data.compare(0, 3, "ORC") == 0;
  if (parquet) {  // staged as it is: LOAD DATA reads Parquet directly (doLoad -> loadParquet), by column name
    delim = '\t';
    skip = 0;
  } else if (avro || orc) {  // row / columnar binary formats: as text, by column name
    std::vector<std::string> names = cols;
    if (names.empty()) {
      auto t = cat_.get(table);
      for (auto& c : t->cols) names.push_back(c.name);
    }
    body = avro ? avroToTsv(data, names) : orcToTsv(data, names);
    delim = '\t';
    skip = 0;
    escaped = true;
  } else if (json) {
    std::vector<std::string> paths = jsonPaths;
    if (cols.empty()) {
      auto t = cat_.get(table);
      for (auto& c : t->cols) paths.push_back(c.name);
    }
    body = jsonLinesToTsv(data, paths);
    delim = '\t';
    skip = 0;
  }
  const std::string dir = dir_.empty() ? std::string("/tmp") : dir_ + "/pipeline-loads";
  if (!dir_.empty()) mkdir(dir.c_str(), 0755);
  const std::string path = dir + "/" + (dir_.empty() ? "warpstore-" : "") + stagedName + ".tsv";  // (never re-routed by its name)
  {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) throw SqlError("cannot stage " + path, 1105);
    fwrite(body.data(), 1, body.size(), f);
    fclose(f);
  }
  std::string colList;
  for (auto& c : loadCols) colList += (colList.empty() ? "" : ", ") + c;
  std::string sql = "LOAD DATA INFILE '" + path + "' INTO TABLE " + table + " FIELDS TERMINATED BY '" +
                    (delim == '\t' ? std::string("\\t") : std::string(1, delim)) + "'" + (escaped ? " ESCAPED BY '\\\\'" : "") +
                    " IGNORE " + std::to_string(skip) + " LINES" +
                    (colList.empty() ? "" : " (" + colList + ")");
  ResultSet r;
  try {
    r = execute(sql);
  } catch (...) {
    if (dir_.empty()) unlink(path.c_str());
    throw;
  }
  if (dir_.empty()) unlink(path.c_str());  // without a data dir nothing replays it
  return r.affected;
}

ResultSet Engine::loadRemote(Stmt& s) {
  auto t0 = Clock::now();
  const bool json = s.format == "JSON";
  const std::string table = s.table.find('.') == std::string::npos ? currentDb() + "." + s.table : s.table;
  if (s.setValue != "MYSQL") cat_.get(table);
  int64_t total = 0, objects = 0;
  const std::string tag = std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
  if (s.setValue == "MYSQL") {  // one-off copy of a MySQL database (a pipeline keeps following it)
    std::atomic<bool> run{true};
    CdcStatus last;
    mysqlCdcRun(*this, s.path, s.table, "load-" + std::to_string(nowUs()), run, false, [&](const CdcStatus& st) { last = st; });
    return okResult(last.rows, "copied " + std::to_string(last.rows) + " rows from MySQL (binlog position " + last.position + ")");
  }
  if (s.setValue == "S3") {
    const std::string bucket = s3Bucket(s.path);
    for (auto& o : s3List(s.path, s.config, s.credentials)) {
      std::string data = decompressByName(o.key, s3Get(bucket, o.key, s.config, s.credentials));
      total += ingest(table, s.delim, s.skipLines, s.insertCols, s.jsonPaths, json, data, "s3-" + tag + "-" + sanitizeName(o.key));
      ++objects;
    }
  } else if (s.setValue == "KAFKA") {
    std::map<int, int64_t> next;
    for (int round = 0; round < 100000; ++round) {
      auto msgs = kafkaPoll(s.path, next, 8 << 20);
      if (msgs.empty()) break;
      std::string data;
      for (auto& m : msgs) {
        data += m.value;
        if (data.empty() || data.back() != '\n') data += '\n';
      }
      total += ingest(table, s.delim, 0, s.insertCols, s.jsonPaths, json, data, "kafka-" + tag + "-" + std::to_string(round));
      objects += (int64_t)msgs.size();
    }
  } else if (s.path.rfind("http://", 0) == 0 || s.path.rfind("https://", 0) == 0) {
    std::string data = decompressByName(s.path, httpGet(s.path));
    total = ingest(table, s.delim, s.skipLines, s.insertCols, s.jsonPaths, json, data, "http-" + tag);
    objects = 1;
  } else {  // FS + FORMAT JSON
    glob_t g{};
    std::vector<std::string> files;
    if (::glob(s.path.c_str(), 0, nullptr, &g) == 0)
      for (size_t i = 0; i < g.gl_pathc; ++i) files.push_back(g.gl_pathv[i]);
    globfree(&g);
    if (files.empty()) throw SqlError("File '" + s.path + "' not found", 29);
    for (auto& f : files) {
      FILE* fp = fopen(f.c_str(), "rb");
      if (!fp) throw SqlError("cannot read " + f, 29);
      std::string data;
      char buf[1 << 16];
      for (size_t n; (n = fread(buf, 1, sizeof buf, fp)) > 0;) data.append(buf, n);
      fclose(fp);
      total += ingest(table, s.delim, s.skipLines, s.insertCols, s.jsonPaths, json, decompressByName(f, data),
                      "fs-" + tag + "-" + sanitizeName(baseName(f)));
      ++objects;
    }
  }
  char info[160];
  snprintf(info, sizeof info, "Records: %lld  Sources: %lld  Loaded in %.1f s", (long long)total, (long long)objects, msSince(t0) / 1000);
  return okResult(total, info);
}

void Engine::runRemotePipeline(const std::shared_ptr<Pipeline>& p) {
  const SourceSpec sp = unpackSource(p->glob);
  const bool json = sp.format == "JSON";
  std::vector<std::string> cols;
  for (size_t a = 0; !p->cols.empty();) {
    size_t b = p->cols.find(", ", a);
    cols.push_back(p->cols.substr(a, b == std::string::npos ? std::string::npos : b - a));
    if (b == std::string::npos) break;
    a = b + 2;
  }
  const char delim = p->delim.empty() ? '\t' : p->delim[0];
  // Kafka offsets come back from the staged file names already loaded: kafka-<pipe>-<part>-<from>-<next>.
  std::map<int, int64_t> next;
  const std::string kpre = "kafka-" + sanitizeName(lower(p->name)) + "-";
  {
    std::lock_guard<std::mutex> lk(p->mu);
    for (auto& d : p->done) {
      std::string b = baseName(d);
      if (b.rfind("warpstore-", 0) == 0) b = b.substr(6);
      if (b.rfind(kpre, 0) != 0) continue;
      int part;
      long long from, nx;
      if (sscanf(b.c_str() + kpre.size(), "%d-%lld-%lld", &part, &from, &nx) == 3) next[part] = std::max<int64_t>(next[part], nx);
    }
  }
  auto stagedPath = [&](const std::string& name) {  // the file ingest() stages (and the WAL names)
    return (dir_.empty() ? std::string("/tmp/warpstore-") : dir_ + "/pipeline-loads/") + name + ".tsv";
  };
  auto fail = [&](const std::string& what) {
    std::lock_guard<std::mutex> lk(p->mu);
    p->lastError = what;
    p->state = "Running (with errors)";
  };
  while (p->running) {
    try {
      if (sp.src == "S3" || sp.src == "FS") {
        std::vector<std::string> keys;
        if (sp.src == "S3") {
          for (auto& o : s3List(sp.path, sp.config, sp.creds)) keys.push_back(o.key);
        } else {
          glob_t g{};
          if (::glob(sp.path.c_str(), 0, nullptr, &g) == 0)
            for (size_t i = 0; i < g.gl_pathc; ++i) keys.push_back(g.gl_pathv[i]);
          globfree(&g);
        }
        std::sort(keys.begin(), keys.end());
        for (auto& k : keys) {
          if (!p->running) break;
          const std::string name = lower(sp.src) + "-" + sanitizeName(lower(p->name)) + "-" + sanitizeName(k);
          {
            std::lock_guard<std::mutex> lk(p->mu);
            const std::string sp2 = stagedPath(name);
            if (p->done.count(sp2) || p->done.count(sp2.substr(0, sp2.size() - 4))) continue;  // (older snapshots: no .tsv)
          }
          try {
            std::string data;
            if (sp.src == "S3") {
              data = s3Get(s3Bucket(sp.path), k, sp.config, sp.creds);
            } else {
              FILE* fp = fopen(k.c_str(), "rb");
              if (!fp) continue;
              char buf[1 << 16];
              for (size_t n; (n = fread(buf, 1, sizeof buf, fp)) > 0;) data.append(buf, n);
              fclose(fp);
            }
            int64_t n = ingest(p->table, delim, p->skip, cols, sp.jsonPaths, json, decompressByName(k, data), name);
            std::lock_guard<std::mutex> lk(p->mu);
            p->done.insert(stagedPath(name));
            ++p->files;
            p->rows += n;
          } catch (std::exception& e) {
            std::lock_guard<std::mutex> lk(p->mu);
            p->done.insert(stagedPath(name));  // don't retry a bad object forever
            p->lastError = k + ": " + e.what();
            p->state = "Running (with errors)";
          }
        }
      } else if (sp.src == "KAFKA") {
        auto before = next;
        auto msgs = kafkaPoll(sp.path, next, 8 << 20);
        std::map<int, std::string> byPart;
        std::map<int, int64_t> cnt;
        for (auto& m : msgs) {
          std::string& d = byPart[m.partition];
          d += m.value;
          if (d.empty() || d.back() != '\n') d += '\n';
          ++cnt[m.partition];
        }
        for (auto& [part, data] : byPart) {
          int64_t from = before.count(part) ? before[part] : 0;
          const std::string name = kpre + std::to_string(part) + "-" + std::to_string(from) + "-" + std::to_string(next[part]);
          int64_t n = ingest(p->table, delim, 0, cols, sp.jsonPaths, json, data, name);
          std::lock_guard<std::mutex> lk(p->mu);
          for (auto it = p->done.begin(); it != p->done.end();)  // keep only the latest batch per partition
            it = baseName(*it).find(kpre + std::to_string(part) + "-") != std::string::npos ? p->done.erase(it) : std::next(it);
          p->done.insert(stagedPath(name));
          p->files += cnt[part];
          p->rows += n;
        }
        if (!msgs.empty()) continue;  // more may be waiting
      }
      {
        std::lock_guard<std::mutex> lk(p->mu);
        if (p->state == "Running (with errors)" && p->lastError.empty()) p->state = "Running";
      }
    } catch (std::exception& e) {
      fail(e.what());
    }
    for (int i = 0; i < 10 && p->running; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
}

ResultSet Engine::showPipelines() {
  ResultSet r;
  r.isResult = true;
  r.names = {"Pipelines_in_" + currentDb(), "State", "Table", "Source", "Files_Loaded", "Rows_Loaded", "Last_Error"};
  r.types = {LType::VARCHAR, LType::VARCHAR, LType::VARCHAR, LType::VARCHAR, LType::BIGINT, LType::BIGINT, LType::VARCHAR};
  std::lock_guard<std::mutex> g(pipesMu_);
  for (auto& [k, p] : pipes_) {
    std::lock_guard<std::mutex> lk(p->mu);
    std::string src = p->glob.find('\x1e') == std::string::npos ? "FS " + p->glob
                      : p->glob.substr(0, p->glob.find('\x1e')) + " " + p->glob.substr(p->glob.find('\x1e') + 1, p->glob.find('\x1e', p->glob.find('\x1e') + 1) - p->glob.find('\x1e') - 1);
    if (size_t sch = src.find("://"); sch != std::string::npos)  // never show a password in the source URL
      if (size_t at = src.find('@', sch); at != std::string::npos)
        if (size_t colon = src.find(':', sch + 3); colon != std::string::npos && colon < at) src.replace(colon + 1, at - colon - 1, "***");
    r.rows.push_back({p->name, p->state, p->table, src, std::to_string(p->files), std::to_string(p->rows),
                      p->lastError.empty() ? kNullCell : p->lastError});
  }
  return r;
}

// Host aggregation: the GPU produces the filtered / joined rows (group keys and aggregate arguments),
// the host groups them and computes every aggregate, then HAVING / ORDER BY / LIMIT run over a temp
// table of the groups. Used for aggregates without a GPU implementation and as a fallback.
// GROUP BY with more keys than the GPU path takes: plain INT / BIGINT columns of one table whose value
// ranges are known are packed into one BIGINT key (each value's offset from its minimum, NULL as one
// more value, times the product of the ranges before it); select list, HAVING and ORDER BY read them
// back from it. False if that doesn't bring the keys down to kMaxKeys.
// GROUP BY keys determined by other keys (functional dependencies): a table's columns by its primary
// key, and a column by one it equals in WHERE / inner-join conditions. They are dropped from GROUP BY
// and read as MIN(col) (one value per group): GROUP BY c_custkey, c_name, n_name .. groups on the key
// alone (fewer keys: cheaper hashing, a dense key range, no compile-only plans).
void Engine::dropDependentKeys(Stmt& s) {
  if (s.groupBy.size() < 2 || !s.groupingSets.empty() || s.from.empty()) return;
  for (auto& it : s.items)
    if (it.e->kind == Expr::STAR || it.e->window) return;
  struct T {
    std::shared_ptr<Table> t;
    std::string alias;
  };
  std::vector<T> tabs;
  for (auto& f : s.from) {
    T x;
    if (!f.sub && !f.tvf) x.t = cat_.find(f.name.find('.') == std::string::npos ? currentDb() + "." + f.name : f.name);
    x.alias = f.alias.empty() ? f.name : f.alias;
    tabs.push_back(x);
  }
  using TC = std::pair<int, int>;
  auto colOf = [&](const ExprP& e) -> TC {  // a plain column of a FROM table, or (-1, -1)
    if (!e || e->kind != Expr::COL) return {-1, -1};
    TC r{-1, -1};
    for (size_t i = 0; i < tabs.size(); ++i) {
      if (!e->table.empty() && strcasecmp(e->table.c_str(), tabs[i].alias.c_str()) != 0) continue;
      if (!tabs[i].t) {
        if (e->table.empty()) return {-1, -1};  // (a derived table might have it: ambiguous)
        continue;
      }
      const int c = tabs[i].t->colIndex(e->name);
      if (c < 0) continue;
      if (r.first >= 0) return {-1, -1};  // ambiguous
      r = {(int)i, c};
    }
    return r;
  };
  auto resolve = [&](const ExprP& g) -> ExprP {  // GROUP BY 2 / GROUP BY alias
    if (g->kind == Expr::LIT && g->lit.kind == Value::INT && g->lit.i >= 1 && g->lit.i <= (int64_t)s.items.size()) return s.items[g->lit.i - 1].e;
    if (g->kind == Expr::COL && g->table.empty())
      for (auto& it : s.items)
        if (!it.alias.empty() && strcasecmp(it.alias.c_str(), g->name.c_str()) == 0) return it.e;
    return g;
  };
  std::vector<TC> key(s.groupBy.size());
  bool anyPk = false;
  for (size_t i = 0; i < s.groupBy.size(); ++i) key[i] = colOf(resolve(s.groupBy[i]));
  for (auto& x : tabs) anyPk = anyPk || (x.t && !x.t->pkCols.empty());
  if (!anyPk) return;
  // Equalities between columns: WHERE and inner-join conditions (top-level AND).
  std::vector<std::pair<TC, TC>> eqs;
  std::function<void(const ExprP&)> conj = [&](const ExprP& x) {
    if (!x) return;
    if (x->kind == Expr::BIN && x->op == "AND") { conj(x->args[0]); conj(x->args[1]); return; }
    if (x->kind == Expr::BIN && x->op == "=") {
      const TC a = colOf(x->args[0]), b = colOf(x->args[1]);
      if (a.first >= 0 && b.first >= 0) eqs.push_back({a, b});
    }
  };
  conj(s.where);
  for (auto& j : s.joinConds) conj(j);
  for (size_t i = 0; i < s.from.size(); ++i)
    if (s.from[i].join == 0) conj(s.from[i].on);
  auto closure = [&](const std::vector<bool>& use) {
    std::set<TC> d;
    std::vector<bool> full(tabs.size());
    for (size_t i = 0; i < key.size(); ++i)
      if (use[i] && key[i].first >= 0) d.insert(key[i]);
    for (bool changed = true; changed;) {
      changed = false;
      for (auto& [a, b] : eqs) {
        if (d.count(a) && d.insert(b).second) changed = true;
        if (d.count(b) && d.insert(a).second) changed = true;
      }
      for (size_t t = 0; t < tabs.size(); ++t) {
        if (full[t] || !tabs[t].t || tabs[t].t->pkCols.empty()) continue;
        bool all = true;
        for (int c : tabs[t].t->pkCols) all = all && d.count({(int)t, c});
        if (!all) continue;
        full[t] = true;
        changed = true;
        for (int c = 0; c < (int)tabs[t].t->cols.size(); ++c) d.insert({(int)t, c});
      }
    }
    return d;
  };
  std::vector<bool> use(key.size(), true);
  int dropped = 0;
  for (size_t i = key.size(); i-- > 0;) {
    if (key[i].first < 0) continue;
    use[i] = false;
    if (closure(use).count(key[i])) ++dropped;
    else use[i] = true;
  }
  if (!dropped) return;
  int aggs = dropped;
  std::function<void(const ExprP&)> countAggs = [&](const ExprP& x) {
    if (!x) return;
    if (x->kind == Expr::FUNC && (isAggName(x->op) || isHostAggName(x->op))) { aggs += 1 + (x->op == "AVG"); return; }
    for (auto& a : x->args) countAggs(a);
  };
  for (auto& it : s.items) countAggs(it.e);
  countAggs(s.having);
  if (aggs > kInterpAggs - 2) return;
  std::set<TC> gone;
  for (size_t i = 0; i < key.size(); ++i)
    if (!use[i]) gone.insert(key[i]);
  for (size_t i = 0; i < key.size(); ++i)
    if (use[i]) gone.erase(key[i]);  // (a repeated key: still grouped on)
  bool mixed = false;  // a kept expression key over a dropped column: left alone
  std::function<void(const ExprP&)> refs = [&](const ExprP& x) {
    if (!x || mixed) return;
    if (x->kind == Expr::COL && gone.count(colOf(x))) mixed = true;
    for (auto& a : x->args) refs(a);
  };
  for (size_t i = 0; i < key.size(); ++i)
    if (use[i] && key[i].first < 0) refs(resolve(s.groupBy[i]));
  if (mixed) return;
  for (size_t i = 0; i < key.size(); ++i)
    if (use[i]) s.groupBy[i] = cloneExpr(resolve(s.groupBy[i]));  // (an alias / ordinal: its item is rewritten below)
  std::function<void(ExprP&)> wrap = [&](ExprP& x) {
    if (!x || x->kind == Expr::SUBQUERY || x->sub) return;
    if (x->kind == Expr::FUNC && (isAggName(x->op) || isHostAggName(x->op))) return;
    if (x->kind == Expr::COL && gone.count(colOf(x))) {
      auto m = std::make_shared<Expr>();
      m->kind = Expr::FUNC;
      m->op = "MIN";
      m->args = {x};
      x = m;
      return;
    }
    for (auto& a : x->args) wrap(a);
  };
  for (auto& it : s.items) {
    const std::string before = it.e->str(), name = it.e->kind == Expr::COL ? it.e->name : before;
    wrap(it.e);
    if (it.alias.empty() && it.e->str() != before) it.alias = name;  // (the output column keeps its name)
  }
  wrap(s.having);
  for (auto& o : s.orderBy) {
    if (o.e->kind == Expr::LIT) continue;  // ORDER BY 2
    bool alias = false;
    if (o.e->kind == Expr::COL && o.e->table.empty())
      for (auto& it : s.items) alias = alias || strcasecmp(it.alias.c_str(), o.e->name.c_str()) == 0;
    if (!alias) wrap(o.e);
  }
  std::vector<ExprP> keep;
  for (size_t i = 0; i < key.size(); ++i)
    if (use[i]) keep.push_back(s.groupBy[i]);
  s.groupBy = keep;
}

bool Engine::packGroupKeys(Stmt& s) {
  if (s.from.size() != 1 || s.from[0].sub || !s.joinConds.empty() || s.unionNext) return false;
  auto t = cat_.find(s.from[0].name.find('.') == std::string::npos ? currentDb() + "." + s.from[0].name : s.from[0].name);
  if (!t) return false;
  const std::string alias = s.from[0].alias.empty() ? s.from[0].name : s.from[0].alias;
  auto resolve = [&](const ExprP& g) -> ExprP {  // GROUP BY 2 / GROUP BY alias
    if (g->kind == Expr::LIT && g->lit.kind == Value::INT && g->lit.i >= 1 && g->lit.i <= (int64_t)s.items.size()) return s.items[g->lit.i - 1].e;
    if (g->kind == Expr::COL && g->table.empty())
      for (auto& it : s.items)
        if (!it.alias.empty() && strcasecmp(it.alias.c_str(), g->name.c_str()) == 0) return it.e;
    return g;
  };
  struct Key {
    int col;
    int64_t lo;
    uint64_t width;  // values + 1 (NULL)
  };
  std::vector<Key> pack;
  std::vector<ExprP> rest;
  long double product = 1;
  for (auto& g0 : s.groupBy) {
    const ExprP g = resolve(g0);
    int c = -1;
    if (g->kind == Expr::COL && (g->table.empty() || strcasecmp(g->table.c_str(), alias.c_str()) == 0)) c = t->colIndex(g->name);
    int64_t lo, hi;
    bool dup = false;
    for (auto& k : pack) dup = dup || k.col == c;
    if (c >= 0 && !dup && (t->cols[c].lt == LType::INT || t->cols[c].lt == LType::BIGINT) && colRange(*t, c, lo, hi) && hi >= lo &&
        (uint64_t)hi - (uint64_t)lo < ((uint64_t)1 << 40) && product * ((long double)((uint64_t)hi - (uint64_t)lo) + 2) < 9.0e15L) {
      pack.push_back({c, lo, (uint64_t)hi - (uint64_t)lo + 2});
      product *= (long double)pack.back().width;
    } else {
      rest.push_back(g0);
    }
  }
  if (pack.size() < 2 || rest.size() + 1 > (size_t)kMaxKeys) return false;
  auto q = [&](int c) { return "`" + t->cols[c].name + "`"; };
  std::string p;
  uint64_t mult = 1;
  std::vector<uint64_t> mults;
  for (auto& k : pack) {
    mults.push_back(mult);
    p += (p.empty() ? "" : " + ") + std::string("COALESCE(") + q(k.col) + " - (" + std::to_string(k.lo) + "), " + std::to_string(k.width - 1) + ") * " + std::to_string(mult);
    mult *= k.width;
  }
  p = "(" + p + ")";
  auto parseExpr = [](const std::string& x) { return parseSql("SELECT " + x).items.at(0).e; };
  std::vector<ExprP> decode;
  for (size_t i = 0; i < pack.size(); ++i) {
    const std::string slot = "((" + p + " DIV " + std::to_string(mults[i]) + ") % " + std::to_string(pack[i].width) + ")";
    decode.push_back(parseExpr("IF(" + slot + " = " + std::to_string(pack[i].width - 1) + ", NULL, " + slot + " + (" + std::to_string(pack[i].lo) + "))"));
  }
  std::function<void(ExprP&)> rw = [&](ExprP& x) {
    if (!x || x->kind == Expr::SUBQUERY || x->sub) return;
    if (x->kind == Expr::COL && (x->table.empty() || strcasecmp(x->table.c_str(), alias.c_str()) == 0)) {
      const int c = t->colIndex(x->name);
      for (size_t i = 0; i < pack.size(); ++i)
        if (pack[i].col == c) {
          x = cloneExpr(decode[i]);
          return;
        }
      return;
    }
    if (x->kind == Expr::FUNC && (isAggName(x->op) || isHostAggName(x->op))) return;  // (aggregates read the rows)
    for (auto& a : x->args) rw(a);
  };
  for (auto& it : s.items) {
    if (it.alias.empty() && it.e->kind == Expr::COL) it.alias = it.e->name;  // (keep the column header)
    rw(it.e);
  }
  rw(s.having);
  for (auto& o : s.orderBy) {
    if (o.e->kind == Expr::LIT && o.e->lit.kind == Value::INT) continue;  // ORDER BY 1: the item
    rw(o.e);
  }
  s.groupBy = rest;
  s.groupBy.insert(s.groupBy.begin(), parseExpr(p));
  return true;
}

// GROUP BY with DISTINCT aggregates over different arguments: one GPU aggregation for the plain
// aggregates and one per DISTINCT aggregate (each the usual single-DISTINCT path), merged per group
// key (NULL keys included) into a temp table; the select list, HAVING, ORDER BY and LIMIT then run
// over it.
ResultSet Engine::multiDistinct(Stmt& s, QueryCtx& q, bool typed) {
  auto resolve = [&](const ExprP& g) -> ExprP {
    if (g->kind == Expr::LIT && g->lit.kind == Value::INT && g->lit.i >= 1 && g->lit.i <= (int64_t)s.items.size()) return s.items[g->lit.i - 1].e;
    if (g->kind == Expr::COL && g->table.empty())
      for (auto& it : s.items)
        if (!it.alias.empty() && strcasecmp(it.alias.c_str(), g->name.c_str()) == 0) return it.e;
    return g;
  };
  std::vector<ExprP> keys;
  for (auto& g : s.groupBy) keys.push_back(cloneExpr(resolve(g)));
  std::vector<ExprP> plain, dist;
  std::map<std::string, std::string> colOf;  // aggregate text -> merged column
  std::function<void(const ExprP&)> collect = [&](const ExprP& e) {
    if (!e || e->kind == Expr::SUBQUERY || e->sub) return;
    if (e->kind == Expr::FUNC && e->window) throw SqlError("window", 1235);
    if (e->kind == Expr::FUNC && (isAggName(e->op) || isHostAggName(e->op))) {
      const std::string k = e->str();
      if (!colOf.count(k)) {
        if (e->distinct) colOf[k] = "__d" + std::to_string(dist.size()), dist.push_back(e);
        else colOf[k] = "__a" + std::to_string(plain.size()), plain.push_back(e);
      }
      return;
    }
    for (auto& a : e->args) collect(a);
  };
  for (auto& it : s.items) collect(it.e);
  collect(s.having);
  for (auto& o : s.orderBy) collect(o.e);
  auto part = [&](const std::vector<ExprP>& aggs, const std::string& prefix) {
    Stmt p;
    p.type = Stmt::SELECT;
    p.from = s.from;
    p.joinConds = s.joinConds;
    p.where = s.where ? cloneExpr(s.where) : nullptr;
    for (size_t i = 0; i < keys.size(); ++i) {
      p.items.push_back({cloneExpr(keys[i]), "__g" + std::to_string(i)});
      p.groupBy.push_back(cloneExpr(keys[i]));
    }
    for (size_t j = 0; j < aggs.size(); ++j) p.items.push_back({cloneExpr(aggs[j]), prefix + std::to_string(j)});
    return p;
  };
  auto keyOf = [&](const std::vector<Value>& row) {  // (numbers by value: the parts may type a key differently, 19 / 19.0)
    std::string k;
    char b[40];
    for (size_t i = 0; i < keys.size(); ++i) {
      const Value& v = row[i];
      if (v.isNull()) k += "\x02";
      else if (v.kind == Value::STR) k += v.s;
      else snprintf(b, sizeof b, "%.17g", v.num()), k += b;
      k += '\x1f';
    }
    return k;
  };
  Stmt base = part(plain, "__a");
  ResultSet m = query(base, q, false, true);
  std::unordered_map<std::string, size_t> at;
  for (size_t r = 0; r < m.vals.size(); ++r) at.emplace(keyOf(m.vals[r]), r);
  for (size_t k = 0; k < dist.size(); ++k) {
    Stmt dk = part({dist[k]}, "__v");
    ResultSet rk = query(dk, q, false, true);
    for (auto& row : m.vals) row.push_back(Value::Null());
    for (auto& row : rk.vals) {
      auto it = at.find(keyOf(row));
      if (it != at.end()) m.vals[it->second].back() = row[keys.size()];
    }
    m.names.push_back("__d" + std::to_string(k));
    m.types.push_back(rk.types.at(keys.size()));
  }
  const std::string tmp = makeTemp(m, q, "mdist");
  // the original query over the merged groups
  std::function<ExprP(const ExprP&)> rw = [&](const ExprP& e) -> ExprP {
    if (!e) return e;
    if (e->kind == Expr::SUBQUERY || e->sub) return e;
    const std::string k = e->str();
    for (size_t i = 0; i < keys.size(); ++i)
      if (keys[i]->str() == k) return mkCol("__mg", "__g" + std::to_string(i));
    if (auto it = colOf.find(k); it != colOf.end() && e->kind == Expr::FUNC) return mkCol("__mg", it->second);
    auto y = std::make_shared<Expr>(*e);
    y->bound = false;
    for (auto& a : y->args) a = rw(a);
    return y;
  };
  Stmt outer;
  outer.type = Stmt::SELECT;
  for (auto& it : s.items) outer.items.push_back({rw(it.e), it.alias.empty() ? (it.e->kind == Expr::COL ? it.e->name : it.e->str()) : it.alias});
  outer.from = {TableRef{tmp, "__mg", nullptr, 0, nullptr}};
  outer.where = rw(s.having);
  for (auto& o : s.orderBy) {
    const bool pos = o.e->kind == Expr::LIT && o.e->lit.kind == Value::INT;
    const bool alias = o.e->kind == Expr::COL && o.e->table.empty() &&
                       std::any_of(s.items.begin(), s.items.end(), [&](const SelectItem& it) { return !it.alias.empty() && strcasecmp(it.alias.c_str(), o.e->name.c_str()) == 0; });
    outer.orderBy.push_back({pos || alias ? o.e : rw(o.e), o.desc});
  }
  outer.limit = s.limit;
  outer.offset = s.offset;
  outer.distinct = s.distinct;
  return query(outer, q, false, typed);
}

ResultSet Engine::hostAggregate(Stmt& s, QueryCtx& q, bool explain, bool typed) {
  auto resolveGroup = [&](const ExprP& g) -> ExprP {
    if (g->kind == Expr::LIT && g->lit.kind == Value::INT && g->lit.i >= 1 && g->lit.i <= (int64_t)s.items.size())
      return s.items[g->lit.i - 1].e;
    if (g->kind == Expr::COL && g->table.empty())
      for (auto& it : s.items)
        if (!it.alias.empty() && strcasecmp(it.alias.c_str(), g->name.c_str()) == 0) return it.e;
    return g;
  };
  std::vector<ExprP> groups;
  for (auto& g : s.groupBy) groups.push_back(resolveGroup(g));
  std::vector<ExprP> aggs;
  std::map<std::string, int> aggOf;
  std::function<void(const ExprP&)> collect = [&](const ExprP& e) {
    if (!e) return;
    if (e->kind == Expr::FUNC && (isAggName(e->op) || isHostAggName(e->op)) && !e->window) {
      if (!aggOf.count(e->str() + (e->distinct ? "!" : ""))) {
        aggOf[e->str() + (e->distinct ? "!" : "")] = (int)aggs.size();
        aggs.push_back(e);
      }
      return;
    }
    for (auto& a : e->args) collect(a);
  };
  for (auto& it : s.items) collect(it.e);
  collect(s.having);
  for (auto& o : s.orderBy) collect(o.e);
  // Inner query: group keys and every argument / ORDER BY expression of every aggregate.
  Stmt inner = cloneStmt(s);
  inner.items.clear();
  inner.groupBy.clear();
  inner.having.reset();
  inner.orderBy.clear();
  inner.limit = -1;
  inner.offset = 0;
  inner.distinct = false;
  for (size_t i = 0; i < groups.size(); ++i) inner.items.push_back({cloneExpr(groups[i]), "__g" + std::to_string(i)});
  struct AggCols {
    std::vector<int> args;
    std::vector<std::pair<int, bool>> ord;
  };
  std::vector<AggCols> ac(aggs.size());
  for (size_t j = 0; j < aggs.size(); ++j) {
    for (auto& a : aggs[j]->args) {
      if (a->kind == Expr::STAR) continue;
      ac[j].args.push_back((int)inner.items.size());
      inner.items.push_back({cloneExpr(a), "__x" + std::to_string(inner.items.size())});
    }
    for (auto& [o, d] : aggs[j]->winOrder) {
      ac[j].ord.push_back({(int)inner.items.size(), d});
      inner.items.push_back({cloneExpr(o), "__x" + std::to_string(inner.items.size())});
    }
  }
  if (inner.items.empty()) inner.items.push_back({mkLit(Value::I(1)), "__x0"});
  ResultSet r = query(inner, q, false, true);
  if (r.vals.size() > 50'000'000) throw SqlError("too many rows for host aggregation", 1105);
  // Group.
  std::unordered_map<std::string, size_t> gi;
  std::vector<std::vector<size_t>> members;
  std::vector<size_t> firstRow;
  for (size_t i = 0; i < r.vals.size(); ++i) {
    std::string k;
    for (size_t g = 0; g < groups.size(); ++g) {
      const Value& v = r.vals[i][g];
      k += (v.isNull() ? std::string("\x02") : v.kind == Value::STR && ciColl() ? foldCase(v.s) : v.toString()) + '\x1f';
    }
    auto [it, fresh] = gi.emplace(k, members.size());
    if (fresh) {
      members.emplace_back();
      firstRow.push_back(i);
    }
    members[it->second].push_back(i);
  }
  if (groups.empty() && members.empty()) {  // aggregate over no rows: one group
    members.emplace_back();
    firstRow.push_back(SIZE_MAX);
  }
  auto jsonVal = [](const Value& v) -> std::string {
    if (v.isNull()) return "null";
    if (v.kind == Value::INT || v.kind == Value::DBL) return v.toString();
    std::string o = "\"";
    for (char ch : v.toString()) {
      if (ch == '"' || ch == '\\') o += '\\';
      o += ch;
    }
    return o + "\"";
  };
  ResultSet out;
  out.isResult = true;
  out.typed = true;
  for (size_t g = 0; g < groups.size(); ++g) {
    out.names.push_back("__g" + std::to_string(g));
    out.types.push_back(r.types[g] == LType::DECIMAL ? LType::DOUBLE : r.types[g]);
  }
  for (size_t j = 0; j < aggs.size(); ++j) {
    const std::string f = aggs[j]->op;
    LType at = ac[j].args.empty() ? LType::BIGINT : r.types[ac[j].args[0]];
    if (at == LType::DECIMAL) at = LType::DOUBLE;
    LType t = f == "COUNT" ? LType::BIGINT
            : f == "SUM" ? (at == LType::DOUBLE ? LType::DOUBLE : LType::BIGINT)
            : f == "MIN" || f == "MAX" || f == "ANY_VALUE" || f == "FIRST" || f == "LAST" || f == "PERCENTILE_DISC" ? at
            : f == "GROUP_CONCAT" || f == "STRING_AGG" || f == "JSON_ARRAYAGG" || f == "JSON_AGG" || f == "JSON_OBJECTAGG" ? LType::VARCHAR
            : f.rfind("BIT_", 0) == 0 ? LType::BIGINT : LType::DOUBLE;
    out.names.push_back("__a" + std::to_string(j));
    out.types.push_back(t);
  }
  for (size_t gidx = 0; gidx < members.size(); ++gidx) {
    std::vector<Value> row;
    for (size_t g = 0; g < groups.size(); ++g) row.push_back(r.vals[firstRow[gidx]][g]);
    for (size_t j = 0; j < aggs.size(); ++j) {
      const ExprP& a = aggs[j];
      const std::string f = a->op;
      std::vector<size_t> rows = members[gidx];
      if (!ac[j].ord.empty())
        std::stable_sort(rows.begin(), rows.end(), [&](size_t x, size_t y) {
          for (auto& [c, d] : ac[j].ord) {
            int k = cmpVal(r.vals[x][c], r.vals[y][c]);
            if (k) return d ? k > 0 : k < 0;
          }
          return false;
        });
      // Values of the first argument (rows with a NULL argument are skipped, as in SQL).
      std::vector<Value> vals;
      std::set<std::string> seen;
      size_t starCount = rows.size();
      for (size_t rr : rows) {
        if (ac[j].args.empty()) continue;
        bool nul = false;
        for (int c : ac[j].args) nul |= r.vals[rr][c].isNull() && (f != "JSON_OBJECTAGG" || c == ac[j].args[0]);
        if (nul && f != "JSON_ARRAYAGG" && f != "JSON_AGG") continue;
        Value v = r.vals[rr][ac[j].args[0]];
        if ((f == "GROUP_CONCAT") && ac[j].args.size() > 1) {  // GROUP_CONCAT(a, b): concatenated
          std::string cat;
          for (int c : ac[j].args) cat += r.vals[rr][c].toString();
          v = Value::S(cat);
        }
        if (f == "JSON_OBJECTAGG") v = Value::S(jsonVal(Value::S(v.toString())) + ":" + jsonVal(r.vals[rr][ac[j].args[1]]));
        if (a->distinct) {
          std::string k = v.isNull() ? "\x02" : v.kind == Value::STR && ciColl() ? foldCase(v.s) : v.toString();
          if (!seen.insert(k).second) continue;
        }
        vals.push_back(v);
      }
      Value res = Value::Null();
      if (f == "COUNT") res = Value::I(ac[j].args.empty() ? (int64_t)starCount : (int64_t)vals.size());
      else if (f == "SUM" || f == "AVG") {
        if (!vals.empty()) {
          double acc = 0;
          for (auto& v : vals) acc += numOf(v);
          res = f == "AVG" ? Value::D(acc / vals.size()) : out.types[groups.size() + j] == LType::DOUBLE ? Value::D(acc) : Value::I((int64_t)std::llround(acc));
        }
      } else if (f == "MIN" || f == "MAX") {
        for (auto& v : vals)
          if (res.isNull() || (f == "MIN" ? cmpVal(v, res) < 0 : cmpVal(v, res) > 0)) res = v;
      } else if (f == "ANY_VALUE") {
        if (!vals.empty()) res = vals[0];
      } else if (f == "GROUP_CONCAT" || f == "STRING_AGG") {
        std::string sep = a->castType == "SEP" ? a->name : ",";
        if (f == "STRING_AGG" && a->args.size() > 1 && a->args[1]->kind == Expr::LIT) sep = a->args[1]->lit.toString();
        if (!vals.empty()) {
          std::string cat;
          for (size_t i = 0; i < vals.size(); ++i) cat += (i ? sep : "") + vals[i].toString();
          res = Value::S(cat);
        }
      } else if (f == "JSON_ARRAYAGG" || f == "JSON_AGG") {
        if (!vals.empty()) {
          std::string o = "[";
          const bool raw = !a->args.empty() && isJsonExpr(a->args[0], nullptr);
          for (size_t i = 0; i < vals.size(); ++i) o += (i ? ", " : "") + (raw ? jsonOfValue(vals[i], true) : jsonVal(vals[i]));
          res = Value::S(o + "]");
        }
      } else if (f == "JSON_OBJECTAGG") {
        if (!vals.empty()) {
          std::string o = "{";
          for (size_t i = 0; i < vals.size(); ++i) o += (i ? "," : "") + vals[i].s;
          res = Value::S(o + "}");
        }
      } else if (f.rfind("BIT_", 0) == 0) {
        int64_t acc = f == "BIT_AND" ? -1 : 0;
        for (auto& v : vals) {
          int64_t x = (int64_t)numOf(v);
          acc = f == "BIT_AND" ? acc & x : f == "BIT_OR" ? acc | x : acc ^ x;
        }
        res = Value::I(acc);
      } else if (f == "FIRST" || f == "LAST") {  // FIRST(v, t) / LAST(v, t): v at the smallest / largest t
        Value best, bestT;
        bool have = false;
        for (size_t rr : rows) {
          const Value& v = r.vals[rr][ac[j].args[0]];
          const Value t = ac[j].args.size() > 1 ? r.vals[rr][ac[j].args[1]] : Value::I((int64_t)rr);
          if (t.isNull()) continue;
          const int c = have ? cmpVal(t, bestT) : 0;
          if (!have || (f == "FIRST" ? c < 0 : c > 0)) best = v, bestT = t, have = true;
        }
        if (have) res = best;
      } else if (f == "PERCENTILE_CONT" || f == "PERCENTILE_DISC" || f == "APPROX_PERCENTILE") {
        if (!vals.empty()) {
          double p = 0.5;
          if (ac[j].args.size() > 1 && !rows.empty()) p = numOf(r.vals[rows[0]][ac[j].args[1]]);
          p = std::min(1.0, std::max(0.0, p));
          std::vector<double> d;
          for (auto& v : vals) d.push_back(numOf(v));
          std::sort(d.begin(), d.end());
          if (f == "PERCENTILE_DISC") {
            size_t k = (size_t)std::ceil(p * d.size());
            res = Value::D(d[k == 0 ? 0 : k - 1]);
          } else {
            const double pos = p * (d.size() - 1);
            const size_t lo = (size_t)std::floor(pos), hi = (size_t)std::ceil(pos);
            res = Value::D(d[lo] + (d[hi] - d[lo]) * (pos - lo));
          }
        }
      } else if (f == "MEDIAN") {
        if (!vals.empty()) {
          std::vector<double> d;
          for (auto& v : vals) d.push_back(numOf(v));
          std::sort(d.begin(), d.end());
          res = Value::D(d.size() % 2 ? d[d.size() / 2] : (d[d.size() / 2 - 1] + d[d.size() / 2]) / 2);
        }
      } else {  // STDDEV / VARIANCE family
        const bool samp = f == "STDDEV_SAMP" || f == "VAR_SAMP";
        if (vals.size() > (samp ? 1u : 0u)) {
          double mean = 0, m2 = 0;
          for (auto& v : vals) mean += numOf(v);
          mean /= vals.size();
          for (auto& v : vals) m2 += (numOf(v) - mean) * (numOf(v) - mean);
          double var = m2 / (vals.size() - (samp ? 1 : 0));
          res = Value::D(f.rfind("VAR", 0) == 0 ? var : std::sqrt(var));
        }
      }
      row.push_back(res);
    }
    out.vals.push_back(std::move(row));
  }
  std::string tmp = makeTemp(out, q, "hagg");
  // Outer query over the groups.
  std::function<ExprP(const ExprP&)> rw = [&](const ExprP& e) -> ExprP {
    if (!e) return e;
    for (size_t i = 0; i < groups.size(); ++i)
      if (e->str() == groups[i]->str()) return mkCol("", "__g" + std::to_string(i));
    if (e->kind == Expr::FUNC && (isAggName(e->op) || isHostAggName(e->op)) && !e->window)
      return mkCol("", "__a" + std::to_string(aggOf.at(e->str() + (e->distinct ? "!" : ""))));
    auto y = std::make_shared<Expr>(*e);
    y->bound = false;
    for (auto& a : y->args) a = rw(a);
    return y;
  };
  Stmt outer;
  outer.type = Stmt::SELECT;
  for (auto& it : s.items) outer.items.push_back({rw(it.e), it.alias.empty() ? (it.e->kind == Expr::COL ? it.e->name : it.e->str()) : it.alias});
  outer.where = rw(s.having);
  for (auto& o : s.orderBy) {
    bool alias = o.e->kind == Expr::COL && o.e->table.empty() &&
                 std::any_of(s.items.begin(), s.items.end(), [&](const SelectItem& it) { return !it.alias.empty() && strcasecmp(it.alias.c_str(), o.e->name.c_str()) == 0; });
    bool pos = o.e->kind == Expr::LIT && o.e->lit.kind == Value::INT;
    outer.orderBy.push_back({alias || pos ? o.e : rw(o.e), o.desc});
  }
  outer.limit = s.limit;
  outer.offset = s.offset;
  outer.distinct = s.distinct;
  outer.from = {TableRef{tmp, "__hagg", nullptr, 0, nullptr}};
  return query(outer, q, explain, typed);
}

thread_local bool tTypedResult = false;
thread_local bool tRawNumber = false;
thread_local bool tColumnar = false;  // a single numeric column comes back as ResultSet::numCol (IN subqueries)
thread_local bool tInsertCols = false;
thread_local int tOnlyPart = -1;
ResultSet Engine::runStatementQuery(Stmt& s, bool explain) {
  QueryCtx q;
  ResultSet r;
  try {
    r = query(s, q, explain, tTypedResult && !explain);
  } catch (...) {
    dropTemps(q);
    throw;
  }
  dropTemps(q);
  return r;
}

// ---- Snapshots (SNAPSHOT DATABASE): every database, table (schema + rows), view and pipeline in one
// file; the WAL then only holds what happened after it.
namespace {
void putU64(FILE* f, uint64_t v) { fwrite(&v, 8, 1, f); }
void putStr(FILE* f, std::string_view s) { putU64(f, s.size()); fwrite(s.data(), 1, s.size(), f); }
bool getU64(FILE* f, uint64_t& v) { return fread_unlocked(&v, 8, 1, f) == 1; }
bool getStr(FILE* f, std::string& s) {
  uint64_t n;
  if (!getU64(f, n) || n > (1ull << 32)) return false;
  s.resize(n);
  return fread_unlocked(s.data(), 1, n, f) == n;
}
void putVal(FILE* f, const Value& v) {
  uint8_t k = v.isNull() ? 0 : v.kind == Value::INT ? 1 : v.kind == Value::DBL ? 2 : v.kind == Value::STR ? 3 : v.kind == Value::DATE ? 4 : 5;
  fwrite(&k, 1, 1, f);
  if (k == 2) fwrite(&v.d, 8, 1, f);
  else if (k == 3) putStr(f, v.s);
  else if (k) fwrite(&v.i, 8, 1, f);
}
bool getVal(FILE* f, Value& v) {
  uint8_t k;
  if (fread(&k, 1, 1, f) != 1) return false;
  if (k == 0) { v = Value::Null(); return true; }
  if (k == 3) { std::string s; if (!getStr(f, s)) return false; v = Value::S(std::move(s)); return true; }
  int64_t i;
  double d;
  if (k == 2) { if (fread(&d, 8, 1, f) != 1) return false; v = Value::D(d); return true; }
  if (fread(&i, 8, 1, f) != 1) return false;
  v = k == 1 ? Value::I(i) : k == 4 ? Value::Dt(i) : Value::Ts(i);
  return true;
}
}  // namespace

ResultSet Engine::doSnapshot() {
  if (dir_.empty()) throw SqlError("SNAPSHOT needs a data directory (the server runs with --no-wal)", 1235);
  if (tTxn.active) commitTxn();
  const auto t0 = Clock::now();
  const int64_t loaded = loadedBytes_.load();
  writeSnapshot("", "");
  loadedBytes_ -= loaded;
  return okResult(0, "snapshot written in " + fmtMs(msSince(t0)) + " ms (writes paused " + fmtMs(lastSnapPauseMs_) + " ms)");
}

// Columnar table image: dictionaries, NULL flags, then per partition the live rows' physical column
// bytes (string columns as dictionary codes). Restoring it skips parsing and per-row inserts.
void putDbl(FILE* f, double v) {
  uint64_t u;
  memcpy(&u, &v, 8);
  putU64(f, u);
}
double getDbl(FILE* f) {
  uint64_t u = 0;
  getU64(f, u);
  double v;
  memcpy(&v, &u, 8);
  return v;
}

// ---- Dictionary files (snapshot format 5): <data dir>/dicts/<name>.wdict, append-only: "WDICT001",
// then frames of u32 "WDFR", u32 strings, u32 raw bytes, u32 compressed bytes, u32 CRC32 of the
// compressed bytes, and the zstd-compressed strings (u32 length + bytes each). A snapshot records
// (file, n, bytes): the first n strings end at byte `bytes`. Each snapshot appends only the strings
// added since the last one (before writes are paused); bytes after `bytes` (from a snapshot that
// was never installed) are cut off when the dictionary is loaded.
static constexpr uint32_t kDictFrame = 0x52464457;  // "WDFR"
static bool writeAt(int fd, const char* p, size_t len, uint64_t off) {
  while (len) {
    const ssize_t k = ::pwrite(fd, p, len, (off_t)off);
    if (k <= 0) return false;
    p += k, len -= (size_t)k, off += (uint64_t)k;
  }
  return true;
}
static void dictOrders(Dict& d, size_t upto, const std::string& dir);
static void dictPersist(Dict& d, size_t upto, const std::string& dir) {
  if (!d.fileName.empty() && upto <= d.fileStrs) {
    dictOrders(d, d.fileStrs, dir);
    return;
  }
  if (d.fileName.empty()) {
    mkdir(dir.c_str(), 0755);
    for (int tries = 0; d.fileName.empty(); ++tries) {
      char name[40];
      snprintf(name, sizeof name, "%016llx", (unsigned long long)(nowUs() * 6364136223846793005ull + d.uid * 1442695040888963407ull + tries));
      const int fd = ::open((dir + "/" + name + ".wdict").c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
      if (fd < 0) {
        if (errno == EEXIST && tries < 8) continue;
        throw SqlError("cannot create a dictionary file in " + dir, 1105);
      }
      const bool ok = ::write(fd, "WDICT001", 8) == 8 && fdatasync(fd) == 0;
      ::close(fd);
      if (!ok) throw SqlError("cannot write a dictionary file in " + dir, 1105);
      d.fileName = name;
      d.fileStrs = 0;
      d.fileBytes = 8;
    }
  }
  std::vector<std::pair<size_t, size_t>> ranges;  // frames: up to 64K strings / 8 MB
  for (size_t i = d.fileStrs; i < upto;) {
    size_t j = i, bytes = 0;
    while (j < upto && j - i < 65536 && bytes < (8u << 20)) bytes += 4 + d.strs[j++].size();
    ranges.push_back({i, j});
    i = j;
  }
  std::vector<std::string> frames(ranges.size());
  std::atomic<bool> bad{false};
  ioParallel(ranges.size(), [&](size_t k) {
    std::string raw;
    for (size_t i = ranges[k].first; i < ranges[k].second; ++i) {
      const std::string_view x = d.strs[i];
      const uint32_t len = (uint32_t)x.size();
      raw.append((const char*)&len, 4).append(x);
    }
    std::string& fr = frames[k];
    fr.resize(20 + ZSTD_compressBound(raw.size()));
    const size_t z = ZSTD_compress(fr.data() + 20, fr.size() - 20, raw.data(), raw.size(), 3);
    if (ZSTD_isError(z)) {
      bad = true;
      return;
    }
    const uint32_t hdr[5] = {kDictFrame, (uint32_t)(ranges[k].second - ranges[k].first), (uint32_t)raw.size(), (uint32_t)z,
                             (uint32_t)crc32(0, (const Bytef*)fr.data() + 20, (uInt)z)};
    memcpy(fr.data(), hdr, 20);
    fr.resize(20 + z);
  });
  if (bad) throw SqlError("cannot compress a dictionary", 1105);
  const std::string path = dir + "/" + d.fileName + ".wdict";
  const int fd = ::open(path.c_str(), O_WRONLY | O_CLOEXEC);
  if (fd < 0) throw SqlError("cannot open " + path, 1105);
  uint64_t off = d.fileBytes;
  bool ok = true;
  for (auto& fr : frames) {
    ok = ok && writeAt(fd, fr.data(), fr.size(), off);
    off += fr.size();
  }
  ok = ok && fdatasync(fd) == 0;
  ::close(fd);
  if (!ok) throw SqlError("cannot write " + path, 1105);
  d.fileBytes = off;
  d.fileStrs = upto;
  dictOrders(d, upto, dir);
}

// Side files with the collation orders (when cached for exactly the strings in the file), so a
// restart doesn't sort the dictionary again: "WORD0001", u64 strings, u32 CRC32C of the rest,
// zstd(order).
static void dictOrders(Dict& d, size_t upto, const std::string& dir) {
  for (int ci : {0, 1}) {
    if (d.orderFileStrs[ci] == upto) continue;
    auto order = sortedOrderCached(d, ci, upto);
    if (!order) continue;
    std::string z(ZSTD_compressBound(order->size() * 4), '\0');
    const size_t k = ZSTD_compress(z.data(), z.size(), order->data(), order->size() * 4, 1);
    if (ZSTD_isError(k)) continue;
    z.resize(k);
    std::string out = "WORD0001";
    const uint64_t nn = upto;
    const uint32_t crc = crc32c(z.data(), z.size());
    out.append((const char*)&nn, 8).append((const char*)&crc, 4).append(z);
    const std::string op = dir + "/" + d.fileName + ".o" + std::to_string(ci), tmp = op + ".tmp";
    const int ofd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    if (ofd < 0) continue;
    const bool wrote = writeAt(ofd, out.data(), out.size(), 0) && fdatasync(ofd) == 0;
    ::close(ofd);
    if (wrote && rename(tmp.c_str(), op.c_str()) == 0) d.orderFileStrs[ci] = upto;
    else unlink(tmp.c_str());
  }
}

static bool readFile(const std::string& path, std::string& out) {
  const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) return false;
  struct stat st;
  bool ok = fstat(fd, &st) == 0;
  if (ok) {
    out.resize((size_t)st.st_size);
    size_t got = 0;  // (one read returns at most ~2 GB)
    while (ok && got < out.size()) {
      const ssize_t r = pread(fd, out.data() + got, out.size() - got, (off_t)got);
      if (r < 0 && errno == EINTR) continue;
      ok = r > 0;
      if (ok) got += (size_t)r;
    }
  }
  ::close(fd);
  return ok;
}

// The first n strings of a dictionary file (ending at byte `bytes`) into an empty dictionary.
static void dictLoad(Dict& d, const std::string& dir, const std::string& name, uint64_t n, uint64_t bytes) {
  const std::string path = dir + "/" + name + ".wdict";
  const int fd = ::open(path.c_str(), O_RDWR | O_CLOEXEC);
  if (fd < 0) throw SqlError("snapshot: dictionary file " + path + " is missing", 1105);
  struct Close {
    int fd;
    ~Close() { ::close(fd); }
  } closer{fd};
  std::string buf(bytes, '\0');
  size_t have = 0;  // (one read returns at most ~2 GB)
  for (ssize_t r; have < bytes && ((r = pread(fd, buf.data() + have, bytes - have, (off_t)have)) > 0 || (r < 0 && errno == EINTR));)
    if (r > 0) have += (size_t)r;
  if (bytes < 8 || have != bytes || memcmp(buf.data(), "WDICT001", 8) != 0)
    throw SqlError("snapshot: dictionary file " + path + " is truncated", 1105);
  struct Fr {
    size_t off, count, raw, z;
  };
  std::vector<Fr> frs;
  uint64_t got = 0;
  for (size_t off = 8; off < bytes;) {
    uint32_t hdr[5];
    if (off + 20 > bytes) throw SqlError("snapshot: dictionary file " + path + " is corrupt", 1105);
    memcpy(hdr, buf.data() + off, 20);
    if (hdr[0] != kDictFrame || off + 20 + hdr[3] > bytes || (uint32_t)crc32(0, (const Bytef*)buf.data() + off + 20, (uInt)hdr[3]) != hdr[4])
      throw SqlError("snapshot: dictionary file " + path + " is corrupt (frame at byte " + std::to_string(off) + ")", 1105);
    frs.push_back({off + 20, hdr[1], hdr[2], hdr[3]});
    got += hdr[1];
    off += 20 + hdr[3];
  }
  if (got != n) throw SqlError("snapshot: dictionary file " + path + " has " + std::to_string(got) + " strings, not " + std::to_string(n), 1105);
  std::vector<std::vector<std::string>> out(frs.size());
  std::atomic<bool> bad{false};
  ioParallel(frs.size(), [&](size_t k) {
    std::string raw(frs[k].raw, '\0');
    if (ZSTD_isError(ZSTD_decompress(raw.data(), raw.size(), buf.data() + frs[k].off, frs[k].z))) {
      bad = true;
      return;
    }
    auto& v = out[k];
    v.reserve(frs[k].count);
    for (size_t p = 0; p + 4 <= raw.size() && v.size() < frs[k].count;) {
      uint32_t len;
      memcpy(&len, raw.data() + p, 4);
      v.emplace_back(raw.data() + p + 4, len);
      p += 4 + len;
    }
    if (v.size() != frs[k].count) bad = true;
  });
  if (bad) throw SqlError("snapshot: dictionary file " + path + " is corrupt", 1105);
  for (auto& v : out)
    for (auto& x : v) d.appendUnique(std::move(x));
  struct stat st;
  if (fstat(fd, &st) == 0 && (uint64_t)st.st_size > bytes && ftruncate(fd, (off_t)bytes) != 0) {}  // (an uninstalled snapshot's tail)
  d.fileName = name;
  d.fileStrs = n;
  d.fileBytes = bytes;
  for (int ci : {0, 1}) {  // the collation orders: read when first needed (sortedDict)
    const std::string op = dir + "/" + name + ".o" + std::to_string(ci);
    if (access(op.c_str(), F_OK) == 0) d.orderFile[ci] = op;
  }
}

// A saved collation order (dictOrders) of exactly the dictionary's strings, or null.
std::shared_ptr<std::vector<int32_t>> loadOrderFile(const std::string& path, size_t n) {
  std::string o;
  if (!readFile(path, o) || o.size() < 20 || o.compare(0, 8, "WORD0001") != 0) return nullptr;
  uint64_t on;
  uint32_t crc;
  memcpy(&on, o.data() + 8, 8);
  memcpy(&crc, o.data() + 16, 4);
  if (on != n || crc32c(o.data() + 20, o.size() - 20) != crc) return nullptr;
  auto order = std::make_shared<std::vector<int32_t>>(n);
  const size_t k = ZSTD_decompress(order->data(), n * 4, o.data() + 20, o.size() - 20);
  if (ZSTD_isError(k) || k != n * 4) return nullptr;
  return order;
}

// After a snapshot is installed: dictionary files it doesn't refer to go.
static void dictFilesInstalled(const std::string& dir, const std::set<std::string>& used) {
  DIR* dp = opendir(dir.c_str());
  if (!dp) return;
  std::vector<std::string> gone;
  while (dirent* e = readdir(dp)) {
    std::string n = e->d_name;
    const size_t dot = n.find('.');
    if (dot != std::string::npos && dot == 16 && !used.count(n.substr(0, dot))) gone.push_back(n);  // (<name>.wdict / .o0 / .o1)
  }
  closedir(dp);
  for (auto& n : gone) unlink((dir + "/" + n).c_str());
}

// refs (snapshot.bin with a data dir): segments that are mostly live are not copied but referred to
// by their segment file (written now if they have none), with their deleted-row bitmap; their file
// ids are added to refs. Everything else is written row by row as before (and compacted).
// dictDir (format 5): dictionaries go to their files (dictPersist); their names are added to dictRefs.
void writeColumnar(FILE* f, const Table& t, std::vector<uint64_t>* refs = nullptr, const std::string* dictDir = nullptr,
                   std::set<std::string>* dictRefs = nullptr) {
  const size_t nc = t.cols.size();
  putU64(f, (uint64_t)t.nparts);
  for (size_t c = 0; c < nc; ++c) {
    Dict* d = t.dicts[c].get();
    const size_t n = d ? d->strs.size() : 0;
    putU64(f, n);
    if (dictDir) {
      if (n) {
        dictPersist(*d, n, *dictDir);  // (the strings added since the pre-pass)
        dictRefs->insert(d->fileName);
      }
      putStr(f, n ? d->fileName : std::string());
      putU64(f, n ? d->fileBytes : 0);
    } else if (d) {
      for (size_t i = 0; i < n; ++i) putStr(f, d->strs[i]);
    }
    putU64(f, t.hasNull[c]);
  }
  // The committed state: rows inserted by still-open transactions are left out, rows they deleted
  // are kept (their WAL records only appear at COMMIT, so recovery stays consistent).
  std::set<std::pair<int32_t, int32_t>> hidden, ghosts;
  for (auto& [k, v] : t.txnPending) {
    if (v.commitSeq || v.logged) continue;  // committed (or logged: its WAL record is rotated away with this snapshot)
    for (auto& l : v.hidden) hidden.insert({l.seg, l.row});
    for (auto& l : v.ghosts) ghosts.insert({l.seg, l.row});
  }
  auto keep = [&](int32_t seg, int64_t r, bool deleted) {
    if (hidden.empty() && ghosts.empty()) return !deleted;
    if (ghosts.count({seg, (int32_t)r})) return true;
    return !deleted && !hidden.count({seg, (int32_t)r});
  };
  std::vector<std::vector<int32_t>> live(t.segs.size());
  std::vector<bool> byRef(t.segs.size(), false);
  // (inside the snapshot's write pause: no per-row work for segments written by reference when no
  // transaction is open, their deleted-row bitmap is the answer)
  const bool plain = hidden.empty() && ghosts.empty();
  std::vector<int64_t> nlive(t.segs.size());
  for (size_t si = 0; si < t.segs.size(); ++si) {
    const auto& sg = t.segs[si];
    if (plain) {
      nlive[si] = sg->n - sg->del->ndel;
    } else {
      for (int64_t r = 0; r < sg->n; ++r)
        if (keep((int32_t)si, r, sg->del->test(r))) live[si].push_back((int32_t)r);
      nlive[si] = (int64_t)live[si].size();
    }
    byRef[si] = refs && tierEnabled() && sg->uses.size() == sg->dev.size() && sg->n > 0 && nlive[si] * 2 >= sg->n;
    if (plain && !byRef[si])
      for (int64_t r = 0; r < sg->n; ++r)
        if (!sg->del->test(r)) live[si].push_back((int32_t)r);
  }
  if (refs) {
    std::vector<size_t> rs;
    for (size_t si = 0; si < t.segs.size(); ++si)
      if (byRef[si]) rs.push_back(si);
    putU64(f, rs.size());
    for (size_t si : rs) {
      Segment& sg = *t.segs[si];
      auto file = segFileEnsure(*sg.d, sg.n, t.sortKey);
      segFileSync(*file);
      refs->push_back(file->id);
      putU64(f, (uint64_t)sg.part);
      putU64(f, (uint64_t)sg.n);
      putU64(f, file->id);
      for (size_t c = 0; c < nc; ++c) {
        putDbl(f, sg.mn[c]);
        putDbl(f, sg.mx[c]);
      }
      std::vector<uint32_t> bits((size_t)(sg.n + 31) / 32, ~0u);  // deleted unless live
      if (plain) {
        const auto& h = sg.del->host;
        for (size_t w = 0; w < bits.size(); ++w) bits[w] = w < h.size() ? h[w] : 0u;
      } else {
        for (int32_t r : live[si]) bits[r >> 5] &= ~(1u << (r & 31));
      }
      if (sg.n % 32) bits.back() &= (1u << (sg.n % 32)) - 1;
      fwrite(bits.data(), 4, bits.size(), f);
    }
  }
  std::vector<uint8_t> tmp;
  for (int p = 0; p < t.nparts; ++p) {
    std::vector<std::pair<const Segment*, const std::vector<int32_t>*>> parts;  // live rows per segment
    int64_t n = 0;
    for (size_t si = 0; si < t.segs.size(); ++si) {
      if (t.segs[si]->part != p || byRef[si]) continue;
      n += (int64_t)live[si].size();
      parts.push_back({t.segs[si].get(), &live[si]});
    }
    const WriteBuffer& b = *t.bufs[p];
    std::vector<int32_t> blive;
    for (int64_t r = 0; r < b.n; ++r)
      if (keep(-1 - p, r, b.del->test(r))) blive.push_back((int32_t)r);
    n += (int64_t)blive.size();
    putU64(f, (uint64_t)n);
    std::vector<uint8_t> out;
    for (size_t c = 0; c < nc; ++c) {
      const int sz = t.colBytes((int)c);
      out.resize((size_t)n * sz);
      size_t o = 0;
      auto copy = [&](const uint8_t* src, const std::vector<int32_t>& rows, int64_t total) {
        if ((int64_t)rows.size() == total) {
          memcpy(out.data() + o, src, (size_t)total * sz);
          o += (size_t)total * sz;
          return;
        }
        for (int32_t r : rows) memcpy(out.data() + o, src + (size_t)r * sz, sz), o += sz;
      };
      for (auto& [sg, lv] : parts) copy(segRawColumn(*sg->d, (int)c, tmp), *lv, sg->n);
      copy(b.cols[c].data(), blive, b.n);
      fwrite(out.data(), 1, out.size(), f);
    }
  }
}

void readColumnar(FILE* f, Table& t, bool refs = false, const std::string* dictDir = nullptr) {
  const size_t nc = t.cols.size();
  uint64_t np, v;
  getU64(f, np);
  for (size_t c = 0; c < nc; ++c) {
    getU64(f, v);
    // (codes are positions: appended as they are, indexed when first looked up)
    if (v && (!t.dicts[c] || t.dicts[c]->strs.size() != 0)) throw SqlError("snapshot: unexpected dictionary for " + t.cols[c].name, 1105);
    if (dictDir) {  // format 5: in its dictionary file
      std::string name;
      uint64_t bytes = 0;
      getStr(f, name);
      getU64(f, bytes);
      if (v) dictLoad(*t.dicts[c], *dictDir, name, v, bytes);
    } else {
      for (uint64_t i = 0; i < v; ++i) {
        std::string x;
        getStr(f, x);
        t.dicts[c]->appendUnique(std::move(x));
      }
    }
    getU64(f, v);
    t.hasNull[c] = (uint8_t)v;
  }
  const bool samePlacement = (int)np == t.nparts || t.nparts == 1;
  auto place = [&](const std::vector<std::vector<uint8_t>>& colsIn, int dst, const std::vector<int64_t>* rows, size_t cnt) {
    WriteBuffer& b = *t.bufs[dst];
    for (size_t c = 0; c < nc; ++c) {
      const int sz = t.colBytes((int)c);
      auto& bc = b.cols[c];
      if (!rows) bc.insert(bc.end(), colsIn[c].begin(), colsIn[c].begin() + cnt * sz);
      else
        for (int64_t r : *rows) bc.insert(bc.end(), colsIn[c].begin() + r * sz, colsIn[c].begin() + (r + 1) * sz);
    }
    b.n += (int64_t)(rows ? rows->size() : cnt);
    b.del->host.resize((b.n + 31) / 32, 0);
    ++b.version;
  };
  // Rows of partition p of the snapshot (or of any, when keep is given: only those rows) -> write buffers.
  auto ingest = [&](const std::vector<std::vector<uint8_t>>& colsIn, uint64_t p, uint64_t n, const std::vector<int64_t>* keepRows) {
    if (samePlacement) {
      place(colsIn, t.nparts == 1 ? 0 : (int)p, keepRows, n);
      return;
    }
    // a different partition count: re-hash each row by its shard key
    std::vector<std::vector<int64_t>> to(t.nparts);
    std::vector<int64_t> ints(nc);
    std::vector<double> dbls(nc);
    auto route = [&](int64_t r) {
      for (int c : t.shardKey) {
        const uint8_t* x = colsIn[c].data() + r * t.colBytes(c);
        if (t.cols[c].pt == PType::F64) memcpy(&dbls[c], x, 8);
        else if (t.cols[c].pt == PType::I64) memcpy(&ints[c], x, 8);
        else { int32_t i32; memcpy(&i32, x, 4); ints[c] = i32; }
      }
      to[t.partOf(ints, dbls)].push_back(r);
    };
    if (keepRows)
      for (int64_t r : *keepRows) route(r);
    else
      for (uint64_t r = 0; r < n; ++r) route((int64_t)r);
    for (int q = 0; q < t.nparts; ++q) place(colsIn, q, &to[q], 0);
  };
  if (refs) {  // segments kept in their files: back as cold segments (columns read when used)
    uint64_t nref = 0;
    getU64(f, nref);
    for (uint64_t i = 0; i < nref; ++i) {
      uint64_t part, n, id;
      getU64(f, part);
      getU64(f, n);
      getU64(f, id);
      auto seg = std::make_unique<Segment>();
      seg->id = nextSegmentId();
      seg->part = t.nparts == 1 ? 0 : (int)part;
      seg->n = (int64_t)n;
      seg->host.resize(nc);
      seg->dev.assign(nc, nullptr);
      seg->mapped.assign(nc, 0);
      seg->uses.assign(nc, 0);
      seg->heat.assign(nc, 0.f);
      seg->mn.resize(nc);
      seg->mx.resize(nc);
      seg->encHost.resize(nc);
      for (size_t c = 0; c < nc; ++c) {
        seg->mn[c] = getDbl(f);
        seg->mx[c] = getDbl(f);
      }
      std::vector<uint32_t> bits((size_t)(n + 31) / 32);
      if (fread(bits.data(), 4, bits.size(), f) != bits.size()) throw SqlError("truncated snapshot", 1105);
      seg->d->file = segFileOpen(id, nc, (int64_t)n);
      seg->enc.resize(nc);
      for (size_t c = 0; c < nc; ++c) {
        seg->enc[c] = seg->d->file->cols[c].enc;
        if (seg->d->file->cols[c].width != (uint32_t)t.colBytes((int)c)) throw SqlError("segment file does not match the table", 1105);
      }
      segAccount(*seg->d);
      seg->d->cold.assign(nc, 1);
      int64_t ndel = 0;
      for (uint32_t w : bits) ndel += __builtin_popcount(w);
      if (!samePlacement) {  // rows go to their new partitions through the write buffers
        std::vector<std::vector<uint8_t>> colsIn(nc);
        std::vector<uint8_t> tmp;
        for (size_t c = 0; c < nc; ++c) {
          const uint8_t* raw = segRawColumn(*seg->d, (int)c, tmp);
          colsIn[c].assign(raw, raw + n * t.colBytes((int)c));
        }
        std::vector<int64_t> keepRows;
        for (uint64_t r = 0; r < n; ++r)
          if (!((bits[r >> 5] >> (r & 31)) & 1u)) keepRows.push_back((int64_t)r);
        ingest(colsIn, part, n, &keepRows);
        continue;
      }
      seg->del->host = std::move(bits);
      seg->del->ndel = ndel;
      const int32_t segId = (int32_t)t.segs.size();
      if (t.pk >= 0) {
        std::vector<uint8_t> tmp;
        const uint8_t* raw = segRawColumn(*seg->d, t.pk, tmp);
        const PType pt = t.cols[t.pk].pt;
        for (int64_t r = 0; r < (int64_t)n; ++r) {
          if (seg->del->test(r)) continue;
          const int64_t k = pt == PType::I32 ? ((const int32_t*)raw)[r] : pt == PType::I64 ? ((const int64_t*)raw)[r] : (int64_t)((const double*)raw)[r];
          t.pkIndex[k] = Loc{segId, (int32_t)r};
        }
      }
      t.segs.push_back(std::move(seg));
    }
  }
  for (uint64_t p = 0; p < np; ++p) {
    uint64_t n;
    getU64(f, n);
    std::vector<std::vector<uint8_t>> colsIn(nc);
    for (size_t c = 0; c < nc; ++c) {
      colsIn[c].resize(n * t.colBytes((int)c));
      if (fread(colsIn[c].data(), 1, colsIn[c].size(), f) != colsIn[c].size()) throw SqlError("truncated snapshot", 1105);
    }
    ingest(colsIn, p, n, nullptr);
    t.flush(true);  // segments (sorted by the SORT KEY, compressed, primary-key index) right away
  }
  for (auto& ix : t.indexes) t.buildIndex(ix);
}

// Writes a snapshot of everything (path "" = the data dir's snapshot.bin, which also truncates the
// WAL) or of one database (BACKUP DATABASE).
// zstd-compressed snapshot / backup streams: a FILE* (fopencookie) the snapshot code writes and
// reads as usual.
namespace {
struct ZW {
  FILE* raw;
  ZSTD_CCtx* c;
  std::vector<char> out;
};
ssize_t zwWrite(void* ck, const char* buf, size_t n) {
  auto* z = (ZW*)ck;
  ZSTD_inBuffer ib{buf, n, 0};
  while (ib.pos < ib.size) {
    ZSTD_outBuffer ob{z->out.data(), z->out.size(), 0};
    size_t rc = ZSTD_compressStream2(z->c, &ob, &ib, ZSTD_e_continue);
    if (ZSTD_isError(rc)) return -1;
    if (fwrite(z->out.data(), 1, ob.pos, z->raw) != ob.pos) return -1;
  }
  return (ssize_t)n;
}
int zwClose(void* ck) {
  auto* z = (ZW*)ck;
  ZSTD_inBuffer ib{nullptr, 0, 0};
  size_t rc;
  do {
    ZSTD_outBuffer ob{z->out.data(), z->out.size(), 0};
    rc = ZSTD_compressStream2(z->c, &ob, &ib, ZSTD_e_end);
    fwrite(z->out.data(), 1, ob.pos, z->raw);
  } while (rc > 0 && !ZSTD_isError(rc));
  fflush(z->raw);
  fdatasync(fileno(z->raw));
  int r = fclose(z->raw);
  ZSTD_freeCCtx(z->c);
  delete z;
  return r;
}
FILE* zstdWriter(FILE* raw) {
  auto* z = new ZW{raw, ZSTD_createCCtx(), std::vector<char>(ZSTD_CStreamOutSize())};
  ZSTD_CCtx_setParameter(z->c, ZSTD_c_compressionLevel, 3);
  ZSTD_CCtx_setParameter(z->c, ZSTD_c_nbWorkers, 4);
  ZSTD_CCtx_setParameter(z->c, ZSTD_c_checksumFlag, 1);  // (a damaged snapshot / backup fails to load instead of loading garbage)
  cookie_io_functions_t io{nullptr, zwWrite, nullptr, zwClose};
  FILE* f = fopencookie(z, "w", io);
  if (f) setvbuf(f, nullptr, _IOFBF, 1 << 20);
  return f;
}
struct ZR {
  FILE* raw;
  ZSTD_DCtx* d;
  std::vector<char> in;
  ZSTD_inBuffer ib{nullptr, 0, 0};
};
ssize_t zrRead(void* ck, char* buf, size_t n) {
  auto* z = (ZR*)ck;
  ZSTD_outBuffer ob{buf, n, 0};
  while (ob.pos == 0) {
    if (z->ib.pos == z->ib.size) {
      size_t got = fread(z->in.data(), 1, z->in.size(), z->raw);
      if (got == 0) break;
      z->ib = {z->in.data(), got, 0};
    }
    size_t rc = ZSTD_decompressStream(z->d, &ob, &z->ib);
    if (ZSTD_isError(rc)) return -1;
  }
  return (ssize_t)ob.pos;
}
int zrClose(void* ck) {
  auto* z = (ZR*)ck;
  int r = fclose(z->raw);
  ZSTD_freeDCtx(z->d);
  delete z;
  return r;
}
FILE* zstdReader(FILE* raw) {
  auto* z = new ZR{raw, ZSTD_createDCtx(), std::vector<char>(ZSTD_DStreamInSize())};
  cookie_io_functions_t io{zrRead, nullptr, nullptr, zrClose};
  FILE* f = fopencookie(z, "r", io);
  if (f) setvbuf(f, nullptr, _IOFBF, 1 << 20);  // (millions of small reads: dictionary strings)
  return f;
}
}  // namespace

void Engine::writeSnapshot(const std::string& outPath, const std::string& onlyDb, uint64_t* lsnOut, std::pair<uint64_t, int64_t>* posOut) {
  std::vector<std::shared_ptr<Table>> tabs;
  std::set<std::string> dbs;
  std::map<std::string, std::string> views;
  {
    std::lock_guard<std::mutex> g(cat_.mu);
    for (auto& [k, t] : cat_.tables)
      if (t->db != "__tmp" && (onlyDb.empty() || lower(t->db) == lower(onlyDb))) tabs.push_back(t);
    for (auto& d : cat_.dbs)
      if (onlyDb.empty() || d == lower(onlyDb)) dbs.insert(d);
    for (auto& [k, v] : cat_.views)
      if (onlyDb.empty() || k.rfind(lower(onlyDb) + ".", 0) == 0) views[k] = v;
  }
  std::lock_guard<std::mutex> snapGuard(snapMu_);
  const bool v4 = outPath.empty() && tierEnabled();  // snapshot.bin: segments by reference (their files)
  std::vector<uint64_t> refs;
  const std::string dictDir = dir_ + "/dicts";
  std::set<std::string> dictRefs;
  // The pre-pass runs twice: what a concurrent load added during the first one (segments, strings)
  // is written by the second, so little is left for the pause.
  for (int pass = 0; pass < (v4 ? 2 : 0); ++pass) {
  if (v4)  // big write buffers become segments first, so they too are written outside the pause below
    for (auto& t : tabs) {
      std::unique_lock<FairSharedMutex> lk(t->mu);
      for (int p = 0; p < t->nparts; ++p)
        if (t->bufs[p]->n - t->bufs[p]->ndel() >= 65536) t->flushPart(p);
    }
  if (v4)  // write the files first, without quiescing anything (segment data is immutable)
    for (auto& t : tabs) {
      std::vector<std::shared_ptr<Segment>> segs;
      {
        std::shared_lock<FairSharedMutex> lk(t->mu);
        for (auto& sp : t->segs)
          if (sp->uses.size() == sp->dev.size() && sp->n > 0 && sp->ndel() * 2 <= sp->n) segs.push_back(sp);
      }
      for (auto& sp : segs) segFileSync(*segFileEnsure(*sp->d, sp->n, t->sortKey));
    }
  if (v4)  // dictionaries: the strings added since the last snapshot go to their files (no pause)
    for (auto& t : tabs) {
      std::vector<std::shared_ptr<Dict>> ds;
      {
        std::shared_lock<FairSharedMutex> lk(t->mu);
        ds = t->dicts;
      }
      for (auto& d : ds)
        if (d && d->strs.size()) dictPersist(*d, d->strs.size(), dictDir);
    }
  }
  const auto pause0 = Clock::now();
  std::unique_lock<FairSharedMutex> gate(writeGate_, std::defer_lock);  // (statements in flight log first)
  lastSnapSkipped_ = false;
  if (tSnapTryOnly && loadsInFlight_ > 0) {  // (automatic: not behind a long LOAD, writers would queue behind it)
    lastSnapSkipped_ = true;
    return;
  }
  if (tWriteGateHeld) throw SqlError("internal: a snapshot inside a statement that logs to the WAL (would wait for itself)", 1105);
  gate.lock();  // (fair: statements in flight finish, new ones wait for the snapshot)
  std::vector<std::unique_lock<FairSharedMutex>> locks;  // quiesce every table
  std::sort(tabs.begin(), tabs.end());
  for (auto& t : tabs) locks.emplace_back(t->mu);
  std::lock_guard<std::mutex> wg(walMu_);
  if (lsnOut) *lsnOut = lsn_;  // the snapshot contains exactly the records up to this lsn
  if (posOut) *posOut = {walGen_, walOff_};  // ... and exactly the WAL up to here
  const std::string fin = outPath.empty() ? dir_ + "/snapshot.bin" : outPath, tmp = fin + ".tmp";
  FILE* f = fopen(tmp.c_str(), "wb");
  if (!f) throw SqlError("cannot write " + tmp, 1105);
  // GDBSNAP2+: the rest is one zstd stream; 3: columnar tables; 4: + segment files; 5: + dictionary files
  fwrite(v4 ? "GDBSNAP5" : "GDBSNAP3", 1, 8, f);
  f = zstdWriter(f);
  putU64(f, dbs.size());
  for (auto& d : dbs) putStr(f, cat_.syncDbs.count(d) ? d + "\x1fSYNC" : d);  // (cat_.mu not needed: dbs only grow under writes, quiesced here)
  putU64(f, views.size());
  for (auto& [k, v] : views) { putStr(f, k); putStr(f, v); }
  int64_t rows = 0;
  putU64(f, tabs.size());
  for (auto& t : tabs) {
    putStr(f, t->db);
    putStr(f, showCreate(*t));
    putU64(f, (uint64_t)t->autoNext);
    putU64(f, (uint64_t)t->liveRows());
    writeColumnar(f, *t, v4 ? &refs : nullptr, v4 ? &dictDir : nullptr, &dictRefs);
    rows += t->liveRows();
  }
  std::vector<std::shared_ptr<Pipeline>> ps;
  {
    std::lock_guard<std::mutex> g(pipesMu_);
    for (auto& [k, p] : pipes_)
      if (onlyDb.empty() || lower(p->db) == lower(onlyDb)) ps.push_back(p);
  }
  putU64(f, ps.size());
  for (auto& p : ps) {
    std::lock_guard<std::mutex> g(p->mu);
    for (auto* x : {&p->name, &p->db, &p->table, &p->glob, &p->delim, &p->cols}) putStr(f, *x);
    putU64(f, (uint64_t)p->skip);
    putU64(f, p->wantRun ? 1 : 0);
    putU64(f, p->done.size());
    for (auto& d : p->done) putStr(f, d);
  }
  fflush(f);
  fdatasync(fileno(f));
  fclose(f);
  auto syncDir = [&](const std::string& file) {
    std::string d = file.substr(0, file.rfind('/') == std::string::npos ? 0 : file.rfind('/'));
    int fd = open(d.empty() ? "." : d.c_str(), O_RDONLY);
    if (fd >= 0) { fsync(fd); ::close(fd); }
  };
  {
    std::lock_guard<std::mutex> ug(usersMu_);  // users go into a side file next to the snapshot (tmp + rename)
    const std::string uf = fin + ".users", ut = uf + ".tmp";
    if (FILE* u = fopen(ut.c_str(), "wb")) {
      putU64(u, users_.size());
      for (auto& [name, h] : users_) { putStr(u, name); putStr(u, h); }
      auto defs = routineDefs();
      putU64(u, defs.size());
      for (auto& d : defs) putStr(u, d);
      putU64(u, grants_.size());
      for (auto& [name, es] : grants_) {
        putStr(u, name);
        putU64(u, es.size());
        for (auto& e : es) putStr(u, e);
      }
      fflush(u);
      fdatasync(fileno(u));
      fclose(u);
      rename(ut.c_str(), uf.c_str());
    }
  }
  if (outPath.empty()) {
    // Crash-safe order: (1) the WAL so far moves to wal.old and a new WAL starts, (2) the snapshot is
    // installed, (3) wal.old is deleted. Recovery replays wal.old only if snapshot.bin.tmp is still
    // there (the snapshot was not installed), so every change is applied exactly once.
    if (wal_) fclose(wal_);
    wal_ = nullptr;
    rename((dir_ + "/wal.log").c_str(), (dir_ + "/wal.old").c_str());
    ++walGen_;
    walOpen(true);
    if (wal_) fdatasync(fileno(wal_));
    syncDir(fin);
    segDirSync();
  }
  if (rename(tmp.c_str(), fin.c_str()) != 0) throw SqlError("cannot install snapshot " + fin, 1105);
  syncDir(fin);
  if (v4) segFilesInstalled(refs), dictFilesInstalled(dictDir, dictRefs);  // files only the previous snapshot used can go
  if (outPath.empty()) {
    archiveWal(dir_ + "/wal.old");
    archiveLoads(walGen_);
  }
  lastSnapPauseMs_ = msSince(pause0);
  (void)rows;
}

std::string gWalArchive;
int64_t gSnapshotTriggerBytes = (int64_t)256 << 20;  // (replay runs ~150K records/s: a crash costs ~20 s at most)

static uint32_t fileCrc(const std::string& path) {  // CRC32C of a file's bytes (0xFFFFFFFF: unreadable)
  const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) return 0xFFFFFFFFu;
  std::vector<char> buf(8 << 20);
  uint32_t c = 0;
  for (ssize_t k; (k = ::read(fd, buf.data(), buf.size())) > 0;) c = crc32c(buf.data(), (size_t)k, c);
  ::close(fd);
  return c;
}
// Copies a file, checks the copy against the source (a damaged copy would replay to other rows) and
// returns its CRC32C; one more attempt on a mismatch.
static void copyFileOnce(const std::string& from, const std::string& to);
static uint32_t copyFile(const std::string& from, const std::string& to) {
  const uint32_t want = fileCrc(from);
  for (int attempt = 0;; ++attempt) {
    copyFileOnce(from, to);
    const uint32_t got = fileCrc(to);
    if (got == want) return got;
    fprintf(stderr, "[warpstore] copy of %s differs from the file (CRC %08x, want %08x)%s\n", from.c_str(), got, want, attempt ? "" : ": copying again");
    if (attempt) {
      unlink(to.c_str());
      throw SqlError("the copy of " + from + " kept differing from the file (memory or disk errors?)", 1105);
    }
  }
}
static void copyFileOnce(const std::string& from, const std::string& to) {
  int in = open(from.c_str(), O_RDONLY);
  if (in < 0) throw SqlError("cannot read " + from, 1105);
  int out = open((to + ".tmp").c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (out < 0) {
    ::close(in);
    throw SqlError("cannot write " + to, 1105);
  }
  struct stat st;
  fstat(in, &st);
  off_t left = st.st_size;
  bool ok = true;
  while (left > 0) {  // in-kernel copy (a reflink on filesystems that have them)
    ssize_t k = copy_file_range(in, nullptr, out, nullptr, (size_t)std::min<off_t>(left, 1 << 30), 0);
    if (k <= 0) {
      ok = false;
      break;
    }
    left -= k;
  }
  if (!ok) {  // e.g. not supported across these filesystems: plain copy
    lseek(in, 0, SEEK_SET);
    lseek(out, 0, SEEK_SET);
    if (ftruncate(out, 0) != 0) {}
    std::vector<char> buf(1 << 20);
    ok = true;
    for (ssize_t k; ok && (k = read(in, buf.data(), buf.size())) > 0;) ok = write(out, buf.data(), (size_t)k) == k;
  }
  ok = ok && fdatasync(out) == 0;
  ::close(in);
  ::close(out);
  if (!ok || rename((to + ".tmp").c_str(), to.c_str()) != 0) {
    unlink((to + ".tmp").c_str());
    throw SqlError("cannot copy " + from + " to " + to, 1105);
  }
}

static void mkdirs(const std::string& d) {
  for (size_t p = 1; p <= d.size(); ++p)
    if (p == d.size() || d[p] == '/') mkdir(d.substr(0, p).c_str(), 0755);
}

static uint64_t walFileGen(const std::string& path) {
  uint64_t g = 0;
  if (FILE* f = fopen(path.c_str(), "rb")) {
    char hdr[16];
    if (fread(hdr, 1, 16, f) == 16 && memcmp(hdr, "GDBWAL3\n", 8) == 0) memcpy(&g, hdr + 8, 8);
    fclose(f);
  }
  return g;
}

void Engine::archiveWal(const std::string& path) {
  if (access(path.c_str(), F_OK) != 0) return;
  if (!gWalArchive.empty()) {
    const uint64_t g = walFileGen(path);
    if (g) {
      try {
        mkdirs(gWalArchive);
        char name[64];
        snprintf(name, sizeof name, "/wal-%020llu.log", (unsigned long long)g);
        const std::string to = gWalArchive + name;
        if (rename(path.c_str(), to.c_str()) != 0) copyFile(path, to);
      } catch (std::exception& e) {
        fprintf(stderr, "[warpstore] WAL archive: %s (keeping %s)\n", e.what(), path.c_str());
        return;
      }
    }
  }
  unlink(path.c_str());
}

// Server-side LOAD DATA INFILE copies its file to <data dir>/loads/g<gen>-.. before logging it, so the
// WAL never depends on files outside the data directory. Once a snapshot covers a WAL generation its
// copies go to the archive (they are needed to replay archived WALs) or are deleted.
void Engine::archiveLoads(uint64_t belowGen) {
  const std::string d = dir_ + "/loads";
  DIR* dp = opendir(d.c_str());
  if (!dp) return;
  std::vector<std::string> names;
  while (dirent* e = readdir(dp))
    if (e->d_name[0] == 'g') names.push_back(e->d_name);
  closedir(dp);
  for (auto& n : names) {
    if (strtoull(n.c_str() + 1, nullptr, 10) >= belowGen) continue;
    const std::string from = d + "/" + n;
    if (!gWalArchive.empty()) {
      try {
        mkdirs(gWalArchive + "/loads");
        const std::string to = gWalArchive + "/loads/" + n;
        if (rename(from.c_str(), to.c_str()) != 0) copyFile(from, to);
      } catch (std::exception& e) {
        fprintf(stderr, "[warpstore] WAL archive: %s\n", e.what());
        continue;
      }
    }
    unlink(from.c_str());
  }
}

// ---- Users: mysql_native_password (token = SHA1(pw) XOR SHA1(scramble + SHA1(SHA1(pw)))).
static std::string sha256(const std::string& x) {
  unsigned char d[32];
  SHA256((const unsigned char*)x.data(), x.size(), d);
  return std::string((const char*)d, 32);
}
static std::string sha1(const std::string& x) {
  unsigned char d[20];
  SHA1((const unsigned char*)x.data(), x.size(), d);
  return std::string((const char*)d, 20);
}

int Engine::checkAuthSha2(const std::string& user, const std::string& scramble, const std::string& token) {
  std::lock_guard<std::mutex> g(usersMu_);
  auto it = users_.find(user.empty() ? "root" : user);
  if (it == users_.end() || it->second == "!role") return 0;
  if (it->second.empty()) return token.empty() || (token.size() == 1 && token[0] == 0) ? 1 : 0;  // no password
  if (it->second.size() != 52) return -1;  // created before SHA256 hashes were kept
  const std::string stage2 = it->second.substr(20);
  if (token.size() != 32) return 0;
  const std::string x = sha256(stage2 + scramble);
  std::string stage1(32, '\0');
  for (int i = 0; i < 32; ++i) stage1[i] = (char)(token[i] ^ x[i]);
  return sha256(stage1) == stage2 ? 1 : 0;
}

bool Engine::checkAuth(const std::string& user, const std::string& scramble, const std::string& token) {
  std::lock_guard<std::mutex> g(usersMu_);
  auto it = users_.find(user.empty() ? "root" : user);
  if (it == users_.end()) return false;
  const std::string stage2 = it->second.substr(0, std::min<size_t>(20, it->second.size()));
  if (stage2.empty()) return user == "root" || user.empty() ? true : token.empty();  // root without a password: open
  if (token.size() != 20) return false;
  std::string h = sha1(scramble + stage2), stage1(20, '\0');
  for (int i = 0; i < 20; ++i) stage1[i] = (char)(token[i] ^ h[i]);
  return sha1(stage1) == stage2;
}

thread_local std::string tUser = "root";
thread_local bool tLocalInfile = false;
thread_local bool tDescribeOnly = false;
const std::string& sessionUser() { return tUser; }
const std::string& tUserName() { return tUser; }

bool Engine::checkPassword(const std::string& user, const std::string& pw) {
  std::lock_guard<std::mutex> g(usersMu_);
  auto it = users_.find(user.empty() ? "root" : user);
  if (it == users_.end() || it->second == "!role") return false;
  return it->second.empty() ? pw.empty() : it->second.substr(0, 20) == sha1(sha1(pw));
}

bool Engine::isSuper(const std::string& user) {
  if (user == "root") return true;
  std::lock_guard<std::mutex> g(usersMu_);
  return grants_[user].count("ALL|*.*") > 0;
}
void Engine::setUser(const std::string& user) { tUser = user.empty() ? "root" : user; }

bool Engine::hasPriv(const std::string& user, const std::string& priv, const std::string& db, const std::string& table) {
  std::set<std::string> seen;
  std::function<bool(const std::string&)> rec = [&](const std::string& u) {
    if (u == "root") return true;
    if (!seen.insert(u).second) return false;
    auto it = grants_.find(u);
    if (it == grants_.end()) return false;
    for (const std::string& e : it->second) {
      if (e[0] == '@') {
        if (rec(e.substr(1))) return true;
        continue;
      }
      const std::string pv = e.substr(0, e.find('|')), sc = e.substr(e.find('|') + 1);
      if (pv != "ALL" && pv != priv) continue;
      if (sc == "*.*" || sc == db + ".*" || sc == db + "." + table || (sc.find('.') == std::string::npos && sc == table)) return true;
    }
    return false;
  };
  return rec(user);
}

// Statement-level privilege checks (SELECT / INSERT / UPDATE / DELETE / CREATE / DROP / ALTER per
// table or database; account management, pipelines, backups need ALL ON *.*). Views are checked by
// their own name, so a view over a filtered table is how row-level security is expressed.
void Engine::checkPrivileges(const Stmt& s) {
  if (tUser == "root") return;
  std::lock_guard<std::mutex> g(usersMu_);
  if (hasPriv(tUser, "ALL", "*", "*") && grants_[tUser].count("ALL|*.*")) return;
  auto need = [&](const std::string& priv, const std::string& name) {
    if (name.empty()) return;
    std::string db = lower(dbOf(name)), t = lower(plainName(name));
    if (db == "information_schema" || db == "__tmp") return;
    if (!hasPriv(tUser, priv, db, t) && !hasPriv(tUser, priv, db, "*"))
      throw SqlError(priv + " command denied to user '" + tUser + "'@'%' for table '" + t + "'", 1142);
  };
  auto needDb = [&](const std::string& priv, const std::string& db) {
    if (!hasPriv(tUser, priv, lower(db), "*") && !grants_[tUser].count(priv + "|" + lower(db) + ".*") &&
        !hasPriv(tUser, priv, lower(db), ""))
      throw SqlError("Access denied for user '" + tUser + "'@'%' to database '" + db + "'", 1044);
  };
  auto super = [&] {
    if (!grants_[tUser].count("ALL|*.*") && !hasPriv(tUser, "SUPER", "", ""))
      throw SqlError("Access denied; you need (at least one of) the SUPER privilege(s) for this operation", 1227);
  };
  std::function<void(const Stmt&, std::set<std::string>)> reads = [&](const Stmt& q, std::set<std::string> ctes) {
    for (auto& [n, c] : q.ctes) {
      ctes.insert(lower(n));
      reads(*c, ctes);
    }
    std::function<void(const ExprP&)> ex = [&](const ExprP& e) {
      if (!e) return;
      if (e->sub) reads(*e->sub, ctes);
      for (auto& a : e->args) ex(a);
    };
    for (auto& f : q.from) {
      if (f.sub) reads(*f.sub, ctes);
      else if (!ctes.count(lower(f.name))) need("SELECT", f.name);
      ex(f.on);
    }
    for (auto& it : q.items) ex(it.e);
    ex(q.where);
    ex(q.having);
    for (auto& c : q.joinConds) ex(c);
    if (q.unionNext) reads(*q.unionNext, ctes);
    if (q.select) reads(*q.select, ctes);
  };
  switch (s.type) {
    case Stmt::SELECT:
      reads(s, {});
      if (!s.outfile.empty()) super();  // SELECT .. INTO OUTFILE writes server files: FILE privilege
      break;
    case Stmt::EXPLAIN: case Stmt::PROFILE: if (s.inner) reads(*s.inner, {}); break;
    case Stmt::INSERT: need("INSERT", s.table); reads(s, {}); break;
    case Stmt::LOAD:  // reading server-side files / remote sources needs FILE (SUPER here), as in MySQL
      need("INSERT", s.table);
      if (!tLocalInfile) super();
      break;
    case Stmt::UPDATE: need("UPDATE", s.table); reads(s, {}); break;
    case Stmt::DELETE: need("DELETE", s.table); reads(s, {}); break;
    case Stmt::CREATE: if (!s.temporary) need("CREATE", s.table); if (s.select) reads(*s.select, {}); break;
    case Stmt::CREATE_VIEW: need("CREATE VIEW", s.table); break;
    case Stmt::DROP: for (auto& n : s.names) need("DROP", n); break;
    case Stmt::DROP_VIEW: for (auto& n : s.names) need("DROP", n); break;
    case Stmt::TRUNCATE: need("DROP", s.table); break;
    case Stmt::ALTER: need("ALTER", s.table); break;
    case Stmt::RENAME: for (auto& n : s.names) need("ALTER", n); break;
    case Stmt::CREATE_DB: case Stmt::DROP_DB: needDb(s.type == Stmt::CREATE_DB ? "CREATE" : "DROP", s.table); break;
    case Stmt::USER:
      if (s.setName == "ALTER" && (s.table.empty() || s.table == tUser)) break;  // own password
      super();
      break;
    case Stmt::PIPELINE: case Stmt::BACKUP: case Stmt::SNAPSHOT: case Stmt::DROP_PLANCACHE: super(); break;
    default: break;
  }
}

ResultSet Engine::doUser(Stmt& s) {
  std::lock_guard<std::mutex> g(usersMu_);
  if (s.setName.rfind("GRANT", 0) == 0 || s.setName.rfind("REVOKE", 0) == 0) {
    const bool grant = s.setName[0] == 'G', role = s.setName.find("_ROLE") != std::string::npos;
    for (auto& who : s.grantees) {
      if (!users_.count(who)) throw SqlError("Can't find any matching row in the user table for '" + who + "'", 1133);
      for (auto& pv : s.names) {
        std::string e = role ? "@" + pv : pv + "|" + s.path;
        if (role) {
          std::string r = pv;
          if (!r.empty() && (r[0] == '\'' || r[0] == '"')) r = r.substr(1, r.size() - 2);
          if (!users_.count(lower(r)) && !users_.count(r)) throw SqlError("Unknown role '" + r + "'", 3523);
          e = "@" + (users_.count(r) ? r : lower(r));
        }
        if (grant) grants_[who].insert(e);
        else if (!role && pv == "ALL") {  // REVOKE ALL ON scope: every privilege on that scope
          for (auto it = grants_[who].begin(); it != grants_[who].end();)
            it = it->size() > s.path.size() && it->substr(it->size() - s.path.size() - 1) == "|" + s.path ? grants_[who].erase(it) : std::next(it);
        } else grants_[who].erase(e);
      }
    }
    return okResult(0);
  }
  const std::string u = s.table.empty() ? tUser : s.table;
  // Stored hash: SHA1(SHA1(pw)) (mysql_native_password) followed by SHA256(SHA256(pw)) (caching_sha2_password).
  const std::string hash = s.path == "\x01" || s.path.empty() ? "" : s.path == "\x03role" ? "!role" : sha1(sha1(s.path)) + sha256(sha256(s.path));
  if (s.setName == "CREATE") {
    if (users_.count(u)) {
      if (s.ifNotExists) return okResult(0);
      throw SqlError("Operation CREATE USER failed for '" + u + "'@'%'", 1396);
    }
    users_[u] = hash;
  } else if (s.setName == "DROP") {
    if (!users_.erase(u) && !s.ifExists) throw SqlError("Operation DROP USER failed for '" + u + "'@'%'", 1396);
    grants_.erase(u);
    for (auto& [k, v] : grants_) v.erase("@" + u);
  } else {
    if (!users_.count(u)) throw SqlError("Operation ALTER USER failed for '" + u + "'@'%'", 1396);
    if (s.path != "\x01") users_[u] = hash;
  }
  return okResult(0);
}

// BACKUP DATABASE db TO {'dir' | S3 'bucket/prefix' ..}: a consistent snapshot of the database
// (<db>.backup) plus a manifest with the WAL position it corresponds to. RESTORE DATABASE db [AS new]
// FROM .. [UNTIL TIMESTAMP 'ts'] loads it and, with UNTIL, replays the archived WAL (--wal-archive)
// and the live WAL for that database up to the time given (point-in-time restore).
static std::string jsonEsc(const std::string& x) {
  std::string o;
  for (char ch : x) {
    if (ch == '"' || ch == '\\') o += '\\';
    o += ch;
  }
  return o;
}

ResultSet Engine::doBackup(Stmt& s) {
  const std::string db = lower(s.table);
  const bool s3 = s.setValue == "S3";
  const std::string stage = (dir_.empty() ? std::string("/tmp/warpstore") : dir_) + "/backup-stage-" + std::to_string(nowUs());
  struct Cleanup {
    std::vector<std::string> files;
    std::string dir;
    ~Cleanup() {
      for (auto& f : files) unlink(f.c_str());
      if (!dir.empty()) rmdir(dir.c_str());
    }
  } cleanup;
  std::string prefix = s.path;  // S3: bucket/prefix
  while (!prefix.empty() && prefix.back() == '/') prefix.pop_back();
  const std::string bucket = s3 ? s3Bucket(prefix) : "";
  const std::string keyBase = s3 ? (prefix.size() > bucket.size() ? prefix.substr(bucket.size() + 1) + "/" : "") + db + ".backup" : "";
  auto localFile = [&] {
    if (s3) {
      mkdirs(stage);
      cleanup.dir = stage;
      return stage + "/" + db + ".backup";
    }
    if (s.path.size() > 7 && s.path.compare(s.path.size() - 7, 7, ".backup") == 0) return s.path;
    struct stat st;
    if (s.setName == "BACKUP" && stat(s.path.c_str(), &st) != 0) mkdirs(s.path);
    return s.path + "/" + db + ".backup";
  };
  if (s.setName == "BACKUP") {
    {
      std::lock_guard<std::mutex> g(cat_.mu);
      if (!cat_.dbs.count(db)) throw SqlError("Unknown database '" + s.table + "'", 1049);
    }
    if (tTxn.active) commitTxn();
    const std::string file = localFile();
    std::pair<uint64_t, int64_t> pos{0, 0};
    auto t0 = Clock::now();
    writeSnapshot(file, db, nullptr, &pos);
    const int64_t now = nowUs();
    const std::string man = "{\"format\":\"warpstore-backup-1\",\"database\":\"" + jsonEsc(db) + "\",\"created_us\":" + std::to_string(now) +
                            ",\"created\":\"" + Value::Ts(now).toString() + "\",\"wal_gen\":" + std::to_string(pos.first) +
                            ",\"wal_offset\":" + std::to_string(pos.second) + ",\"data_dir\":\"" + jsonEsc(dir_) + "\"}\n";
    {
      FILE* f = fopen((file + ".manifest").c_str(), "wb");
      if (!f) throw SqlError("cannot write " + file + ".manifest", 1105);
      fwrite(man.data(), 1, man.size(), f);
      fflush(f);
      fdatasync(fileno(f));
      fclose(f);
    }
    if (s3) {
      cleanup.files = {file, file + ".manifest", file + ".users"};
      s3PutFile(bucket, keyBase, file, s.config, s.credentials);
      s3PutFile(bucket, keyBase + ".manifest", file + ".manifest", s.config, s.credentials);
    } else {
      unlink((file + ".users").c_str());  // users / grants are server-wide: not part of a database backup
    }
    struct stat st;
    stat(s3 ? (stage + "/" + db + ".backup").c_str() : file.c_str(), &st);
    return okResult(0, "backup of " + db + " written to " + (s3 ? "s3://" + bucket + "/" + keyBase : file) + " (" +
                           std::to_string(st.st_size) + " bytes, " + fmtMs(msSince(t0)) + " ms, WAL position " + std::to_string(pos.first) + ":" +
                           std::to_string(pos.second) + ")");
  }
  // RESTORE
  const std::string target = s.asName.empty() ? db : lower(s.asName);
  {
    std::lock_guard<std::mutex> g(cat_.mu);
    for (auto& [k, t] : cat_.tables)
      if (lower(t->db) == target) throw SqlError("Can't restore: database '" + target + "' has tables", 1007);
  }
  const std::string file = localFile();
  if (s3) {
    cleanup.files = {file, file + ".manifest"};
    s3GetFile(bucket, keyBase, file, s.config, s.credentials);
    try {
      s3GetFile(bucket, keyBase + ".manifest", file + ".manifest", s.config, s.credentials);
    } catch (std::exception&) {
    }
  }
  if (access(file.c_str(), R_OK) != 0) throw SqlError("backup file " + file + " not found", 1105);
  std::string man;
  if (FILE* f = fopen((file + ".manifest").c_str(), "rb")) {
    char buf[4096];
    man.assign(buf, fread(buf, 1, sizeof buf, f));
    fclose(f);
  }
  int64_t untilUs = 0;
  if (!s.until.empty()) {
    if (!parseDateTime(s.until, untilUs)) throw SqlError("bad UNTIL TIMESTAMP '" + s.until + "'", 1525);
    if (man.empty()) throw SqlError("UNTIL needs the backup's manifest (" + file + ".manifest)", 1105);
    if (std::atoll(jsonField(man, "created_us").c_str()) > untilUs)
      throw SqlError("UNTIL TIMESTAMP is before the backup was taken (" + jsonField(man, "created") + ")", 1105);
  }
  auto t0 = Clock::now();
  const bool saved = replaying_;
  replaying_ = true;  // the restore itself is not logged statement by statement ...
  int64_t replayed = 0;
  try {
    loadSnapshot(file, db, target, false);
    if (untilUs)
      replayed = replayFrom(db, target, std::strtoull(jsonField(man, "wal_gen").c_str(), nullptr, 10), std::atoll(jsonField(man, "wal_offset").c_str()), untilUs);
  } catch (...) {
    replaying_ = saved;
    throw;
  }
  replaying_ = saved;
  if (!dir_.empty()) doSnapshot();  // ... so make it durable with a checkpoint
  return okResult(0, "restored " + target + " from " + (s3 ? "s3://" + bucket + "/" + keyBase : file) +
                         (untilUs ? " + " + std::to_string(replayed) + " WAL records up to " + s.until : "") + " in " + fmtMs(msSince(t0)) + " ms");
}

// Replays the WAL from {gen, off} (archive files, then the live WAL) for database db (as asDb) up to
// time untilUs. Returns the number of statements applied.
int64_t Engine::replayFrom(const std::string& db, const std::string& asDb, uint64_t gen, int64_t off, int64_t untilUs) {
  std::map<uint64_t, std::string> files;
  if (!gWalArchive.empty())
    if (DIR* dp = opendir(gWalArchive.c_str())) {
      while (dirent* e = readdir(dp)) {
        unsigned long long g;
        if (sscanf(e->d_name, "wal-%llu.log", &g) == 1 && g >= gen) files[g] = gWalArchive + "/" + e->d_name;
      }
      closedir(dp);
    }
  uint64_t liveGen;
  {
    std::lock_guard<std::mutex> g(walMu_);
    liveGen = walGen_;
  }
  if (!dir_.empty() && liveGen >= gen && !files.count(liveGen)) files[liveGen] = dir_ + "/wal.log";
  if (files.empty() || files.begin()->first != gen)
    throw SqlError("point-in-time restore needs WAL generation " + std::to_string(gen) + " (is --wal-archive set to the directory with the archived WALs?)", 1105);
  for (auto it = files.begin(); std::next(it) != files.end(); ++it)
    if (std::next(it)->first != it->first + 1)
      throw SqlError("WAL generation " + std::to_string(it->first + 1) + " is missing from the archive", 1105);
  int64_t applied = 0;
  const std::string savedDb = currentDb();
  auto apply = [&](const std::string& rec) {
    size_t sep = rec.find('\x1f');
    const std::string recDb = sep == std::string::npos ? "" : rec.substr(0, sep);
    std::string sql = sep == std::string::npos ? rec : rec.substr(sep + 1);
    Stmt st;
    try {
      st = parseSql(sql);
    } catch (std::exception&) {
      return;
    }
    const size_t dot = st.table.find('.');
    const std::string tdb = st.type == Stmt::CREATE_DB || st.type == Stmt::DROP_DB ? lower(st.table)
                            : dot != std::string::npos                             ? lower(st.table.substr(0, dot))
                                                                                   : lower(recDb);
    if (tdb != db || st.type == Stmt::CREATE_DB) return;
    if (asDb != db && dot != std::string::npos) {  // db.table in the statement: point it at the new name
      const size_t at = lower(sql).find(db + ".");
      if (at != std::string::npos) sql.replace(at, db.size(), asDb);
    }
    if (st.type == Stmt::DROP_DB) sql = "DROP DATABASE " + asDb;
    currentDb() = lower(recDb) == db ? asDb : recDb;
    try {
      execute(sql);
      ++applied;
    } catch (std::exception& e) {
      fprintf(stderr, "[warpstore] point-in-time restore: %s: %s\n", e.what(), sql.substr(0, 200).c_str());
    }
  };
  bool done = false;
  for (auto& [g, path] : files) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) throw SqlError("cannot read " + path, 1105);
    char hdr[16];
    if (fread(hdr, 1, 16, f) != 16 || memcmp(hdr, "GDBWAL3\n", 8) != 0) {
      fclose(f);
      throw SqlError(path + " is not a format-3 WAL", 1105);
    }
    fseeko(f, g == gen ? std::max<int64_t>(off, 16) : 16, SEEK_SET);
    for (;;) {
      uint32_t len, crc;
      if (fread(&len, 4, 1, f) != 1 || fread(&crc, 4, 1, f) != 1 || len == 0) break;
      std::string rec(len, '\0');
      if (fread(rec.data(), 1, len, f) != len || (uint32_t)crc32(0, (const Bytef*)rec.data(), (uInt)len) != crc) break;
      if (rec.rfind("\x1eTS", 0) == 0 && rec.size() >= 11) {
        int64_t us;
        memcpy(&us, rec.data() + 3, 8);
        if (us > untilUs) {
          done = true;
          break;
        }
      } else if (rec.rfind("\x1eTXN", 0) == 0) {
        for (size_t i = 4; i + 4 <= rec.size();) {
          uint32_t l;
          memcpy(&l, rec.data() + i, 4);
          i += 4;
          if (i + l > rec.size()) break;
          apply(rec.substr(i, l));
          i += l;
        }
      } else {
        apply(rec);
      }
    }
    fclose(f);
    if (done) break;
  }
  currentDb() = savedDb;
  return applied;
}


void Engine::connectionClosed() {
  if (tTxn.active) rollbackTxn();
  for (auto& n : tTempTables) {
    try {
      execute("DROP TABLE IF EXISTS " + n);
    } catch (...) {
    }
  }
  tTempTables.clear();
}

void Engine::loadSnapshot(const std::string& path, const std::string& fromDb, const std::string& toDb, bool withUsers) {
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) return;
  auto mapDb = [&](const std::string& d) {
    if (fromDb.empty() && lower(d) == "gpudb") return std::string("warpstore");  // (data dirs from before the rename)
    return !fromDb.empty() && lower(d) == lower(fromDb) ? lower(toDb) : d;
  };
  auto t0 = Clock::now();
  const std::string savedDb = currentDb();
  char magic[8];
  if (fread(magic, 1, 8, f) != 8 || memcmp(magic, "GDBSNAP", 7) != 0 || magic[7] < '1' || magic[7] > '5') {
    fclose(f);
    throw SqlError("bad snapshot file " + path, 1105);
  }
  std::string whole;  // the decompressed snapshot, when it was checked first (see below)
  if (magic[7] >= '2') {
    // Decompress it all first, so the zstd checksum is verified before anything is parsed (a
    // damaged file must not be half-loaded, or send the parser off on a garbage count). Very large
    // ones (backups with their data inline) are read as a stream; the checksum is then checked at
    // the end.
    const long at = ftell(f);
    ZSTD_DCtx* dz = ZSTD_createDCtx();
    std::vector<char> in(1 << 20);
    std::string out(1 << 20, '\0');
    size_t used = 0, rc = 1;
    bool ok = true, big = false;
    for (size_t got; ok && !big && (got = fread(in.data(), 1, in.size(), f)) > 0;) {
      ZSTD_inBuffer ib{in.data(), got, 0};
      while (ib.pos < ib.size) {
        if (used == out.size()) {
          if (out.size() >= ((size_t)1 << 30)) { big = true; break; }
          out.resize(out.size() * 2);
        }
        ZSTD_outBuffer ob{out.data() + used, out.size() - used, 0};
        rc = ZSTD_decompressStream(dz, &ob, &ib);
        if (ZSTD_isError(rc)) { ok = false; break; }
        used += ob.pos;
      }
    }
    ZSTD_freeDCtx(dz);
    if (!ok || (!big && rc != 0)) {
      fclose(f);
      throw SqlError("corrupt snapshot " + path + (ok ? " (truncated)" : " (checksum mismatch or damaged data)"), 1105);
    }
    if (big) {
      fseek(f, at, SEEK_SET);
      f = zstdReader(f);
    } else {
      fclose(f);
      out.resize(used);
      whole = std::move(out);
      f = fmemopen(whole.data(), std::max<size_t>(whole.size(), 1), "rb");
      if (!f) throw SqlError("cannot read snapshot " + path, 1105);
      if (whole.empty()) fgetc(f);  // (an empty stream: EOF at once)
    }
  }
  const bool columnar = magic[7] >= '3';
  uint64_t n;
  getU64(f, n);
  for (uint64_t i = 0; i < n; ++i) {
    std::string d;
    getStr(f, d);
    if (d.size() > 5 && d.compare(d.size() - 5, 5, "\x1fSYNC") == 0) {
      d.resize(d.size() - 5);
      cat_.syncDbs.insert(mapDb(d));
    }
    cat_.dbs.insert(mapDb(d));
  }
  getU64(f, n);
  for (uint64_t i = 0; i < n; ++i) {
    std::string k, v;
    getStr(f, k);
    getStr(f, v);
    const size_t dot = k.find('.');
    if (dot != std::string::npos) k = mapDb(k.substr(0, dot)) + k.substr(dot);
    cat_.views[k] = v;
  }
  getU64(f, n);
  const uint64_t ntables = n;
  int64_t total = 0;
  for (uint64_t i = 0; i < n; ++i) {
    std::string db, ddl;
    uint64_t next, rows;
    getStr(f, db);
    getStr(f, ddl);
    getU64(f, next);
    getU64(f, rows);
    db = mapDb(db);
    currentDb() = db;
    Stmt cs = parseSql(ddl);
    doCreate(cs);
    auto t = cat_.get(db + "." + cs.table);
    std::unique_lock<FairSharedMutex> lk(t->mu);
    std::vector<Value> row(t->cols.size());
    const std::string dictDir = dir_ + "/dicts";
    const auto tt0 = Clock::now();
    if (columnar) readColumnar(f, *t, magic[7] >= '4', magic[7] >= '5' ? &dictDir : nullptr), rows = 0;  // (rows: counted below from the table)
    if (envSetting("TRACE")) fprintf(stderr, "[trace] snapshot: table %s in %.0f ms\n", t->name.c_str(), msSince(tt0));
    for (uint64_t r = 0; r < rows; ++r) {
      for (auto& v : row) getVal(f, v);
      t->appendRow(row);
    }
    t->autoNext = (int64_t)next;
    t->flush(true);
    total += columnar ? t->liveRows() : (int64_t)rows;
  }
  if (getU64(f, n))
    for (uint64_t i = 0; i < n; ++i) {
      auto p = std::make_shared<Pipeline>();
      for (auto* x : {&p->name, &p->db, &p->table, &p->glob, &p->delim, &p->cols}) getStr(f, *x);
      p->db = mapDb(p->db);
      uint64_t skip, want, nd;
      getU64(f, skip);
      getU64(f, want);
      getU64(f, nd);
      p->skip = (int64_t)skip;
      p->wantRun = want != 0;
      for (uint64_t k = 0; k < nd; ++k) {
        std::string d;
        getStr(f, d);
        p->done.insert(d);
      }
      pipes_[lower(p->name)] = p;
    }
  if (magic[7] >= '2' && whole.empty()) {  // (streamed: the end of the zstd stream, where its checksum is verified)
    char extra;
    const size_t k = fread(&extra, 1, 1, f);
    if (k != 0 || ferror(f)) {
      fclose(f);
      throw SqlError("corrupt snapshot " + path + " (checksum mismatch or trailing data)", 1105);
    }
  }
  fclose(f);
  if (FILE* uf = withUsers ? fopen((path + ".users").c_str(), "rb") : nullptr) {
    uint64_t nu;
    if (getU64(uf, nu))
      for (uint64_t i = 0; i < nu; ++i) {
        std::string u, h;
        if (!getStr(uf, u) || !getStr(uf, h)) break;
        std::lock_guard<std::mutex> g(usersMu_);
        users_[u] = h;
      }
    if (getU64(uf, nu))
      for (uint64_t i = 0; i < nu; ++i) {
        std::string d;
        if (!getStr(uf, d)) break;
        try {
          ResultSet rr;
          routineStatement(d, rr);
        } catch (std::exception& ex) {
          fprintf(stderr, "[warpstore] routine restore failed: %s\n", ex.what());
        }
      }
    if (getU64(uf, nu))
      for (uint64_t i = 0; i < nu; ++i) {
        std::string u, e;
        uint64_t ne;
        if (!getStr(uf, u) || !getU64(uf, ne)) break;
        std::lock_guard<std::mutex> g(usersMu_);
        for (uint64_t j = 0; j < ne && getStr(uf, e); ++j) grants_[u].insert(e);
      }
    fclose(uf);
  }
  currentDb() = savedDb;
  fprintf(stderr, "[warpstore] loaded snapshot: %llu tables, %lld rows in %.0f ms\n", (unsigned long long)ntables, (long long)total, msSince(t0));
}


}  // namespace gdb
