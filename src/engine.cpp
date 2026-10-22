#include "engine_core.h"

namespace gdb {
// ================================================================== Engine

// ---- Pipelines (SingleStore CREATE PIPELINE .. AS LOAD DATA FS '/dir/*.csv' INTO TABLE t): a
// background thread polls the glob and loads each new file once, through the normal (logged) LOAD.


// WAL fsync counters (/metrics).
static std::atomic<uint64_t> gWalSyncs{0}, gWalSyncUs{0}, gWalSyncMaxUs{0};
static const int64_t gStartUs = nowUs();
static std::atomic<bool> gDebugSyncFail{false};  // SET warpstore_debug_sync_fail = 1 (tests: fdatasync fails)
static bool timedSync(int fd) {
  const auto t0 = Clock::now();
  const bool ok = fdatasync(fd) == 0 && !gDebugSyncFail;
  const uint64_t us = (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - t0).count();
  ++gWalSyncs;
  gWalSyncUs += us;
  for (uint64_t m = gWalSyncMaxUs; us > m && !gWalSyncMaxUs.compare_exchange_weak(m, us);) {}
  return ok;
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
      if (behind && !walBroken_) {
        try {
          walSyncNow();
        } catch (std::exception&) {  // (fdatasync failed: walBroken_ is set, writes are refused)
        }
      }
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
      {
        RcuGuard rcu;  // (column versions replaced meanwhile stay readable)
        t->compactBuild(j);
      }
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
  RcuGuard rcu;
  const int64_t limit = tierMemLimit();
  const int64_t used = tierHostBytes() - tierPendingBytes();
  struct Ent {
    std::shared_ptr<Table> t;
    std::shared_ptr<Segment> seg;
    std::shared_ptr<SegData> d;  // the segment's data of column c
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
      if (!sg.full() || sg.n == 0 || sg.d->cold.size() != sg.d->dev.size()) continue;
      for (size_t c = 0; c < sg.d->dev.size(); ++c) {
        const std::shared_ptr<SegData>& dp = sg.cdp(c);
        SegData& D = *dp;
        const bool cold = segCold(D, (int)c);
        if (cold) {  // (present columns' heat is kept by vramBalance)
          const uint32_t u = __atomic_exchange_n(&D.uses[c], 0u, __ATOMIC_RELAXED);
          D.heat[c] = D.heat[c] * 0.9f + (float)u;
        }
        if (evict == cold || (evict && pinned[c])) continue;
        const auto& fc = cold ? D.file->cols[c] : SegFile::Col{};
        const int64_t b = cold ? (int64_t)fc.len + (fc.enc ? sg.n * (int64_t)fc.width : 0)
                               : (int64_t)D.host[c].size() + (D.enc[c] ? (int64_t)D.encHost[c].size() : 0);
        ents.push_back({t, sp, dp, (int)c, D.heat[c], b});
      }
    }
  }
  if (evict) {
    std::stable_sort(ents.begin(), ents.end(), [](const Ent& a, const Ent& b) { return a.heat < b.heat; });  // coldest first
    int64_t need = used - limit / 10 * 8, got = 0;
    size_t k = 0;
    while (k < ents.size() && got < need) got += ents[k++].bytes;
    ents.resize(k);
    for (auto& e : ents) segFileEnsure(*e.d, e.seg->n, e.t->sortKey);  // (outside the table lock: the data is immutable)
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
          if (segEvictBegin(*e->d, e->c, gen)) done.push_back({e->d, e->c, gen});
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
    segLoad(*e.d, e.c);
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
      if (!sg.full()) continue;  // GPU-built temp segments: not cacheable
      for (size_t c = 0; c < sg.d->dev.size(); ++c) {
        SegData& D = sg.cd(c);
        if (segCold(D, (int)c)) continue;  // (disk tier: tierBalance)
        const std::vector<uint8_t>& h = D.enc[c] ? D.encHost[c] : D.host[c];
        if (!D.dev[c] || h.empty()) continue;
        const uint32_t u = __atomic_exchange_n(&D.uses[c], 0u, __ATOMIC_RELAXED);
        D.heat[c] = D.heat[c] * 0.9f + (float)u;
        int hint = t->vramHint((int)c);
        for (auto& ix : t->indexes)  // a CAGRA graph refers to the column's VRAM: it must stay there
          if (ix.vec && ix.algo == 1 && ix.cols[0] == (int)c) hint = 1;
        anyMapped |= D.mapped[c] != 0;
        neverResident |= hint == 2 && !D.mapped[c];
        if (!D.mapped[c]) resident += (int64_t)h.size();
        ents.push_back({t, &sg, (int)c, h.size(), !D.mapped[c], hint == 1 ? 1e30f : hint == 2 ? -1.f : D.heat[c], hint});
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
  auto hostOf = [](Segment& sg, int c) -> std::vector<uint8_t>& {
    SegData& D = sg.cd(c);
    return D.enc[c] ? D.encHost[c] : D.host[c];
  };
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
          SegData& sg = m.e->seg->cd(m.e->c);
          std::vector<uint8_t>& h = hostOf(*m.e->seg, m.e->c);
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

// Room for `need` more bytes after the records written and the space reserved for statements in
// flight: the next chunk zeroed (and its size made durable) first. A full disk throws here.
std::atomic<bool> gDebugWalFull{false};
static thread_local int64_t tWalRoom = 0;  // this thread's statement's reservation (walReserve)
void Engine::walExtendLocked(int64_t need) {
  const int64_t others = walReserved_ - std::min(tWalRoom, walReserved_);
  if (walOff_ + others + need <= walCap_ && !gDebugWalFull) return;
  const int64_t chunk = std::max<int64_t>(64ll << 20, need + others);
  static const std::vector<char> zeros(1 << 20, 0);
  fseeko(wal_, walCap_, SEEK_SET);
  bool ok = !gDebugWalFull;  // (SET warpstore_debug_wal_full = 1, tests: the disk is full)
  if (!ok) errno = ENOSPC;
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

// Before an autocommit write statement changes anything: WAL room for its record, so a full disk
// fails the statement cleanly (nothing applied) instead of leaving a change that isn't logged.
int64_t Engine::walReserve(int64_t bytes) {
  std::lock_guard<std::mutex> g(walMu_);
  if (!wal_) return 0;
  walExtendLocked(bytes);
  walReserved_ += bytes;
  tWalRoom += bytes;
  return bytes;
}
void Engine::walUnreserve(int64_t bytes) {
  if (!bytes) return;
  std::lock_guard<std::mutex> g(walMu_);
  walReserved_ -= bytes;
  tWalRoom -= bytes;
}

void Engine::walWriteLocked(const std::string& rec) {
  const uint32_t len = (uint32_t)rec.size(), crc = (uint32_t)crc32(0, (const Bytef*)rec.data(), (uInt)rec.size());
  const int64_t need = 8 + (int64_t)rec.size();
  walExtendLocked(need);
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
    if (walBroken_) {  // (an earlier fdatasync failed: a retry would succeed without the lost writes)
      if (wait) throw SqlError("the WAL could not be made durable (fdatasync failed); the server is read-only until it is restarted", 1021);
      return;
    }
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
    // A failed fdatasync: what the page cache held may not reach the disk, and a retry can't tell
    // (the kernel drops the error once reported). Nothing written since is acknowledged as durable,
    // and writes are refused until a restart replays what the disk really has.
    if (fd >= 0 && !timedSync(fd)) {
      walBroken_ = true;
      fprintf(stderr, "[warpstore] WAL fdatasync failed: writes are refused until a restart\n");
    }
    lk.lock();
    if (walBroken_) {
      syncing_ = false;
      syncCv_.notify_all();
      if (wait) throw SqlError("the WAL could not be made durable (fdatasync failed); the server is read-only until it is restarted", 1021);
      return;
    }
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
              SegData& D = sg.cd(vc);
              if (segCold(D, vc)) segLoad(D, vc);  // (graph columns stay in memory)
              if (sg.n < 2048 || !D.dev[vc] || D.mapped[vc]) continue;
              std::lock_guard<std::mutex> g(ix.cagra->mu);
              if (ix.cagra->graphs.count(sg.id) || ix.cagra->failed.count(sg.id)) continue;
              std::string err;
              auto gr = cagraBuild((const float*)D.dev[vc], sg.n, ix.dim, ix.metric == 0, ix.graphDegree, err);
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
void waitTxnFree(Table& t, std::unique_lock<FairSharedMutex>& lk, const std::vector<Loc>* rows) {
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
bool keyLookup(const Table& t, const ExprP& w) {
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
uint32_t copyFile(const std::string& from, const std::string& to);
uint32_t fileCrc(const std::string& path);
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
void mkdirs(const std::string& d);

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
  // Strict SQL mode for this statement (values that don't fit their column are errors).
  struct StrictScope {
    bool prev = tStrictTypes;
    ~StrictScope() { tStrictTypes = prev; }
  } strictScope;
  tStrictTypes = sqlModeStrict() && !replaying_ && !tApplying && (s.type == Stmt::INSERT || s.type == Stmt::UPDATE || s.type == Stmt::ALTER);
  bool writes = false;  // (a statement that changes data or the schema: logged)
  switch (s.type) {
    case Stmt::SELECT: writes = !s.outfile.empty(); break;
    case Stmt::INSERT: case Stmt::UPDATE: case Stmt::DELETE: case Stmt::LOAD: case Stmt::CREATE: case Stmt::DROP:
    case Stmt::ALTER: case Stmt::TRUNCATE: case Stmt::RENAME: case Stmt::CREATE_DB: case Stmt::DROP_DB:
    case Stmt::CREATE_VIEW: case Stmt::DROP_VIEW: case Stmt::PIPELINE: case Stmt::USER: case Stmt::BACKUP:
      writes = !(s.type == Stmt::CREATE && s.temporary);
      break;
    default: break;
  }
  if (writes && ((isReplica_ && !tApplying) || (walBroken_ && !replaying_))) {
    if (isReplica_) throw SqlError("The server is a read-only replica (--replica-of); run writes on the primary or PROMOTE it", 1290);
    throw SqlError("The server is read-only: a WAL write failed (disk full?); free space and restart it", 1290);
  }
  // Autocommit writes: WAL room reserved first (a full disk fails the statement before it applies).
  struct WalRoom {
    Engine* e;
    int64_t bytes = 0;
    ~WalRoom() { e->walUnreserve(bytes); }
  } walRoom{this};
  if (writes && writeGate.owns_lock() && !tTxn.active && !replaying_) walRoom.bytes = walReserve((int64_t)sql.size() + 512);
  // In a transaction a failed statement is undone on its own (MySQL: the transaction stays open with
  // what came before), as a rollback to a savepoint taken when it started.
  struct StmtSavepoint {
    bool on;
    size_t undo, wal;
    int ex = std::uncaught_exceptions();
    ~StmtSavepoint() {
      if (!on || std::uncaught_exceptions() <= ex || !tTxn.active) return;
      try {
        undoLog() = nullptr;
        undoRecords(tTxn.undo, undo);
        tTxn.undo.resize(undo);
        tTxn.wal.resize(std::min(wal, tTxn.wal.size()));
      } catch (std::exception& e) {
        fprintf(stderr, "[warpstore] undoing a failed statement: %s\n", e.what());
      }
      undoLog() = &tTxn.undo;
    }
  } stmtSavepoint{writes && tTxn.active && !replaying_, tTxn.undo.size(), tTxn.wal.size()};
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
      else if (s.setName == "SQL_MODE") {
        std::string v = s.setValue;
        v.erase(std::remove(v.begin(), v.end(), '\''), v.end());
        tSqlMode = v == "DEFAULT" ? kDefaultSqlMode : v;
      }
      else if (s.setName == "WARPSTORE_DEBUG_WAL_FULL") gDebugWalFull = std::atoi(s.setValue.c_str()) != 0;
      else if (s.setName == "WARPSTORE_DEBUG_SYNC_FAIL") gDebugSyncFail = std::atoi(s.setValue.c_str()) != 0;
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

// Rebuilds a table with a new schema (ALTER TABLE): `from[i]` is the old column feeding new column i,
// or -1 for a new column (filled with its default).
std::shared_ptr<Table> Engine::rebuild(Table& t, const std::vector<ColumnMeta>& cols, const std::vector<int>& from,
                                      const std::vector<int>* newPk) {
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
  if (newPk) {  // ADD PRIMARY KEY (as CREATE TABLE: a single integer column gets the hash index)
    pkCols = *newPk;
    pk = pkCols.size() == 1 && cols[pkCols[0]].lt != LType::VARCHAR && cols[pkCols[0]].lt != LType::DOUBLE ? pkCols[0] : -1;
  }
  int sk = t.sortKey >= 0 ? mapCol(t.sortKey) : -1;
  auto nt = std::make_shared<Table>(t.name, cols, pk, sk, pkCols, shard, t.nparts, t.reference);
  if (newPk && pk < 0) {  // composite or string PRIMARY KEY: a unique index
    IndexMeta m;
    m.name = "PRIMARY";
    m.unique = true;
    m.cols = pkCols;
    nt->indexes.push_back(m);
  }
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
    std::vector<int> addPk;
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
        if (const IndexMeta& ix = cur->indexes.back(); ix.unique && !ix.vec) {  // the rows there must not repeat a key
          for (auto& [k, locs] : ix.map) {
            if (locs.size() < 2) continue;
            std::set<std::vector<int64_t>> seen;
            for (const Loc& l : locs) {
              std::vector<int64_t> tup(ix.cols.size());
              bool null = false;
              for (size_t q = 0; q < ix.cols.size() && !null; ++q) null = !cur->rawKey(l, ix.cols[q], tup[q]);
              if (!null && !seen.insert(tup).second) {
                const std::string name = ix.name;
                cur->indexes.pop_back();
                throw SqlError("Duplicate entry for key '" + name + "'", 1062);
              }
            }
          }
        }
        continue;
      }
      case Stmt::Alter::ADD_PK: {
        std::vector<int> pkc;
        for (auto& c : a.index.cols) {
          const int k = cur->colIndex(c);
          if (k < 0) throw SqlError("Key column '" + c + "' doesn't exist in table", 1072);
          pkc.push_back(k);
        }
        if (cur->pk >= 0 || !cur->pkCols.empty()) {  // the one it has: nothing to do; another: refused
          if (pkc != cur->pkCols) throw SqlError("Multiple primary key defined", 1068);
          continue;
        }
        if (pkc.size() == 1 && cols[pkc[0]].lt == LType::DOUBLE) throw SqlError("PRIMARY KEY on DOUBLE is not supported", 1235);
        for (int c : cur->shardKey)
          if (std::find(pkc.begin(), pkc.end(), c) == pkc.end())
            throw SqlError("The unique key named: 'PRIMARY' must contain all columns specified in the shard key", 1705);
        for (int c : pkc) cols[c].notNull = true;
        addPk = pkc;  // (the rebuild refuses NULLs and repeated keys, leaving the table as it was)
        break;
      }
      case Stmt::Alter::DROP_INDEX: {
        auto& ix = cur->indexes;
        ix.erase(std::remove_if(ix.begin(), ix.end(), [&](const IndexMeta& m) { return strcasecmp(m.name.c_str(), a.name.c_str()) == 0; }), ix.end());
        continue;
      }
    }
    auto nt = rebuild(*cur, cols, from, addPk.empty() ? nullptr : &addPk);
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
          {"max_allowed_packet", "1073741824"}, {"net_buffer_length", "16384"}, {"sql_mode", tSqlMode}, {"time_zone", "SYSTEM"},
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
          SegData& D = sp->cd(c);
          if (c >= D.dev.size() || !D.dev[c] || segCold(D, (int)c)) continue;
          const double mb = (double)(D.enc[c] ? D.encHost[c].size() : D.host[c].size()) / 1048576.0;
          (D.mapped[c] ? host : vram) += mb;
          if (c < D.heat.size()) heat += D.heat[c];
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
  if (tStrictTypes)
    for (size_t c = 0; c < nc; ++c)
      if (t.cols[c].pt != PType::F64 && (int)c != t.autoIncCol) t.checkFitsColumn((int)c, iv[c], 0, (size_t)n);
  t.appendColumns(iv, dv, n);
  return n;
}

static thread_local std::vector<std::vector<Value>>* tStmtDeleted = nullptr;  // rows the running INSERT deleted (doInsert)

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
  // A failed INSERT outside a transaction (a duplicate key, a NULL into NOT NULL, ...) leaves nothing
  // behind, as in MySQL: the rows it appended to the write buffers (not flushed meanwhile) go again.
  // (Rows REPLACE / ON DUPLICATE KEY UPDATE deleted are put back.)
  struct StmtUndo {
    Table& t;
    bool on, prevDefer;
    int ex = std::uncaught_exceptions();
    std::vector<int64_t> mark;
    std::vector<std::vector<Value>> deleted;
    StmtUndo(Table& tt, bool o) : t(tt), on(o), prevDefer(tt.deferFlush) {
      if (!on) return;
      t.deferFlush = true;
      for (auto& b : t.bufs) mark.push_back(b->n);
      tStmtDeleted = &deleted;
    }
    ~StmtUndo() {
      if (!on) return;
      tStmtDeleted = nullptr;
      t.deferFlush = prevDefer;
      try {
        if (std::uncaught_exceptions() > ex) {
          t.undoAppendsSince(mark);
          for (auto& row : deleted) t.appendRow(row);
        }
        if (!prevDefer)
          for (int p = 0; p < t.nparts; ++p)
            if (t.bufs[p]->n >= kSegRows) t.flushPart(p);
      } catch (std::exception& e) {
        fprintf(stderr, "[warpstore] after INSERT: %s\n", e.what());
      }
    }
  } stmtUndo{*t, !undoLog() && t->db != "__tmp"};
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
        if (tStmtDeleted) tStmtDeleted->push_back(t.rowValues(old));  // (put back if the statement fails)
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
      if (tStmtDeleted) tStmtDeleted->push_back(std::move(cur));  // (put back if the statement fails)
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
  // A batch is appended on another thread while the next one is converted (the load keeps the table
  // lock meanwhile); n counts the rows once their append is done.
  std::future<void> appending;
  int64_t pendingRows = 0;
  auto finishAppend = [&] {
    if (!appending.valid()) return;
    appending.get();
    n += pendingRows;
    pendingRows = 0;
  };
  struct AppendWait {  // (unwinding: the append in flight finishes first, and its rows count)
    std::future<void>& f;
    int64_t& n;
    int64_t& pending;
    ~AppendWait() {
      if (!f.valid()) return;
      try {
        f.get();
        n += pending;
      } catch (...) {
      }
    }
  } appendWait{appending, n, pendingRows};
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
            if (exactI && cm.pt == PType::I32) {  // (min / max: a loop that vectorizes)
              int64_t lo = INT64_MAX, hi = INT64_MIN;
              for (int64_t r = 0; r < rows; ++r) lo = std::min(lo, p.i[r]), hi = std::max(hi, p.i[r]);
              fits = rows == 0 || (lo > INT32_MIN && hi <= INT32_MAX);
            }
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
          if (cm.notNull || (int)c == t->pk) throw SqlError("Column '" + cm.name + "' cannot be null (row " + std::to_string(n + pendingRows + r + 1) + ")", 1048);
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
        if (p.kind == 1 && p.unit == 0 && cm.lt == LType::DATETIME) {
          // a plain integer (no timestamp type) into DATETIME / TIMESTAMP: Unix seconds (as a plain
          // integer into DATE is days since 1970-01-01)
          for (int64_t r = 0; r < rows; ++r) {
            if (!p.valid[r]) {
              null(r);
              continue;
            }
            int64_t us;
            if (__builtin_mul_overflow(p.i[r], (int64_t)1000000, &us))
              throw SqlError("Column '" + cm.name + "': " + std::to_string(p.i[r]) + " is out of range as Unix seconds", 1292);
            iv[c][r] = us;
          }
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
    finishAppend();  // (the previous batch's)
    checkKilled();
    lk.unlock();  // (readers get in between row groups)
    // Over --memory-limit: evict cold column data to the segment files now (the maintenance thread
    // can't get the table lock while a long load holds it).
    if (tierEnabled() && tierHostBytes() - tierPendingBytes() > tierMemLimit() / 10 * 9) tierBalance();
    std::this_thread::yield();
    lk.lock();
    pendingRows = rows;
    appending = std::async(std::launch::async, [tp = t.get(), iv = std::move(iv), dv = std::move(dv), rows, &appMs]() mutable {
      const auto ap0 = Clock::now();
      tp->appendColumns(iv, dv, rows, nullptr, true);  // (null() set hasNull while converting)
      appMs += msSince(ap0);
    });
  });
  finishAppend();
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
  for (auto& sg : t.segs) mark(segRawColumn(sg->cd(c), c, tmp), sg->n, *sg->del);
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
      bool host = false, disk = false;
      for (size_t c = 0; c < t->cols.size(); ++c) {
        const SegData& D = sg.cd(c);
        host = host || (c < D.mapped.size() && D.mapped[c]);
        disk = disk || segCold(D, (int)c);
      }
      r.rows.push_back({std::to_string(p), std::to_string(i), disk ? "disk-columnstore" : host ? "host-columnstore (PCIe)" : "gpu-columnstore",
                        std::to_string(sg.n), std::to_string(sg.ndel()),
                        std::to_string(sg.n * rowBytes + (sg.n + 31) / 32 * 4),
                        t->sortKey >= 0 ? fmtKey(sg.d->mn[t->sortKey]) : kNullCell,
                        t->sortKey >= 0 ? fmtKey(sg.d->mx[t->sortKey]) : kNullCell});
    }
    const WriteBuffer& b = *t->bufs[p];
    if (b.n > 0)
      r.rows.push_back({std::to_string(p), "-1", "cpu-write-buffer", std::to_string(b.n), std::to_string(b.ndel()), "0", kNullCell, kNullCell});
  }
  return r;
}

}  // namespace gdb
