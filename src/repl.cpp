// Replication: read replicas that bootstrap from a snapshot of the primary and then tail its WAL.
//
// The primary keeps the most recent WAL records (with a log sequence number) in memory and serves,
// on its HTTP port (Basic auth, a user with ALL ON *.*):
//   GET /replication/snapshot         -> {"lsn":N,"main":"<path>","users":"<path>"} (a fresh snapshot)
//   GET /replication/wal?from=N       -> records with lsn >= N, framed [u64 lsn][u32 len][db \x1f sql]
//                                        (long-polls up to 1 s; 410 when N fell out of the buffer)
//   GET /replication/file?path=P      -> a snapshot or LOAD DATA file the WAL refers to
// A replica (--replica-of http://user:pw@primary:httpport) is read-only for clients; PROMOTE turns
// it into a primary (failover).
#include <sys/stat.h>
#include <unistd.h>

#include <chrono>
#include <cstring>
#include <regex>
#include <thread>

#include "engine.h"
#include "sources.h"
#include "storage.h"

namespace gdb {

thread_local bool tApplying = false;  // this thread applies replicated records

namespace {
constexpr size_t kReplMaxBytes = 256 << 20;
constexpr size_t kReplMaxRecs = 1 << 20;
}  // namespace

void Engine::replAppend(const std::string& rec) {  // walMu_ held
  ++lsn_;
  replRing_.push_back({lsn_, rec});
  replBytes_ += rec.size();
  while (!replRing_.empty() && (replBytes_ > kReplMaxBytes || replRing_.size() > kReplMaxRecs)) {
    replBytes_ -= replRing_.front().second.size();
    replRing_.pop_front();
  }
  static const std::regex loadRe(R"(^[^\x1f]*\x1f\s*LOAD\s+DATA\s+INFILE\s+'([^']*)')", std::regex::icase);
  std::smatch m;
  if (rec.size() < 4096 && std::regex_search(rec, m, loadRe)) {
    std::lock_guard<std::mutex> g(replMu_);
    replFiles_.insert(m[1].str());
  }
  replCv_.notify_all();
}

std::string Engine::replSnapshot() {
  const std::string base = (dir_.empty() ? std::string("/tmp/gpudb") : dir_ + "/") + "repl-snapshot.bin";
  uint64_t lsn = 0;
  writeSnapshot(base, "", &lsn);
  {
    std::lock_guard<std::mutex> g(replMu_);
    replFiles_.insert(base);
    replFiles_.insert(base + ".users");
  }
  return "{\"lsn\":" + std::to_string(lsn) + ",\"epoch\":\"" + std::to_string(replEpoch_) + "\",\"main\":\"" + base + "\",\"users\":\"" + base + ".users\"}";
}

int gSyncReplicas = 0;
int gSyncReplicaTimeoutMs = 2000;
double gFailoverTimeoutS = 0;

void Engine::replWaitAck(uint64_t lsn) {
  if (gSyncReplicas <= 0 || tApplying) return;
  std::unique_lock<std::mutex> lk(replMu_);
  auto enough = [&] {
    int n = 0;
    for (auto& [id, a] : replAcks_) n += a.first >= lsn;
    return n >= gSyncReplicas;
  };
  if (enough()) return;
  if (!ackCv_.wait_for(lk, std::chrono::milliseconds(gSyncReplicaTimeoutMs), enough)) {
    static std::atomic<int64_t> lastWarn{0};
    const int64_t now = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    if (now - lastWarn.exchange(now) >= 10)
      fprintf(stderr, "[gpu-db] semi-sync replication: no acknowledgement from %d replica(s) within %d ms; committing anyway\n",
              gSyncReplicas, gSyncReplicaTimeoutMs);
  }
}

std::string Engine::replSince(uint64_t from, int waitMs, int& status, const std::string& replicaId, const std::string& epoch) {
  if (!epoch.empty() && epoch != std::to_string(replEpoch_)) {
    status = 410;
    return "positions from another primary incarnation; re-bootstrap";
  }
  std::unique_lock<std::mutex> lk(walMu_);
  if (!replicaId.empty() && from > 0 && from <= lsn_ + 1) {  // (a position from before a restart is no acknowledgement)
    std::lock_guard<std::mutex> g(replMu_);
    const int64_t now = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    replAcks_[replicaId] = {from - 1, now};
    for (auto it = replAcks_.begin(); it != replAcks_.end();)  // forget replicas gone for a minute
      it = now - it->second.second > 60000000 ? replAcks_.erase(it) : std::next(it);
    ackCv_.notify_all();
  }
  if (from > lsn_ + 1) {
    status = 410;
    return "replica is ahead of the primary";
  }
  if (from == lsn_ + 1) replCv_.wait_for(lk, std::chrono::milliseconds(waitMs), [&] { return lsn_ >= from; });
  if (from <= lsn_ && (replRing_.empty() || replRing_.front().first > from)) {
    status = 410;
    return "records before lsn " + std::to_string(replRing_.empty() ? lsn_ + 1 : replRing_.front().first) + " are gone; re-bootstrap";
  }
  status = 200;
  std::string out;
  size_t bytes = 0;
  // Records have consecutive lsns: the first one with lsn >= from is at from - front.
  size_t idx = replRing_.empty() ? 0 : (size_t)(from - replRing_.front().first);
  for (; idx < replRing_.size() && bytes < (32u << 20); ++idx) {
    const auto& [l, rec] = replRing_[idx];
    uint64_t L = l;
    uint32_t n = (uint32_t)rec.size();
    out.append((const char*)&L, 8);
    out.append((const char*)&n, 4);
    out += rec;
    bytes += rec.size();
  }
  return out;
}

bool Engine::replFileAllowed(const std::string& path) {
  std::lock_guard<std::mutex> g(replMu_);
  return replFiles_.count(path) > 0;
}

bool Engine::isReplica() const { return isReplica_; }

namespace {
int64_t nowUsR() {
  return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}
std::string urlEsc(const std::string& s) {
  static const char* hex = "0123456789ABCDEF";
  std::string o;
  for (unsigned char c : s) {
    if (isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~' || c == '/') o += (char)c;
    else o += '%', o += hex[c >> 4], o += hex[c & 15];
  }
  return o;
}
void writeFile(const std::string& path, const std::string& data) {
  FILE* f = fopen(path.c_str(), "wb");
  if (!f) throw SqlError("cannot write " + path, 1105);
  fwrite(data.data(), 1, data.size(), f);
  fclose(f);
}
}  // namespace

void Engine::startReplica(const std::string& url) {
  std::string base = url;
  while (!base.empty() && base.back() == '/') base.pop_back();
  replicaOf_ = base;
  isReplica_ = true;
  replRun_ = true;
  replThread_ = std::thread([this, base] {
    auto setState = [&](const std::string& s, const std::string& err = "") {
      std::lock_guard<std::mutex> g(replMu_);
      replState_ = s;
      if (!err.empty()) replError_ = err;
    };
    const std::string stage = (dir_.empty() ? std::string("/tmp/gpudb-replica-") : dir_ + "/replica-") + std::to_string(getpid());
    char host[256] = "";
    gethostname(host, sizeof host - 1);
    const std::string myId = std::string(host) + "-" + std::to_string(getpid()) + "-" + std::to_string(nowUsR());
    bool needBootstrap = true, bootstrapped = false;
    std::string epoch;  // the primary incarnation our positions belong to
    uint64_t next = 0;
    replLastContactUs_ = nowUsR();
    while (replRun_) {
      if (gFailoverTimeoutS > 0 && bootstrapped && nowUsR() - replLastContactUs_ > (int64_t)(gFailoverTimeoutS * 1e6)) {
        fprintf(stderr, "[gpu-db] primary unreachable for %.0f s: promoting this replica to primary\n", gFailoverTimeoutS);
        isReplica_ = false;
        replRun_ = false;
        setState("Promoted (primary unreachable)");
        break;
      }
      try {
        if (needBootstrap) {
          setState("Bootstrapping");
          std::string info = httpGet(base + "/replication/snapshot");
          uint64_t lsn = std::strtoull(jsonField(info, "lsn").c_str(), nullptr, 10);
          epoch = jsonField(info, "epoch");
          writeFile(stage + ".snap", httpGet(base + "/replication/file?path=" + urlEsc(jsonField(info, "main"))));
          writeFile(stage + ".snap.users", httpGet(base + "/replication/file?path=" + urlEsc(jsonField(info, "users"))));
          {
            // Start from an empty catalog, then load the primary's snapshot.
            std::vector<std::string> drop;
            {
              std::lock_guard<std::mutex> g(cat_.mu);
              for (auto& [k, t] : cat_.tables) drop.push_back(k);
              cat_.views.clear();
            }
            tApplying = true;
            for (auto& k : drop) {
              try {
                execute("DROP TABLE IF EXISTS " + k);
              } catch (...) {
              }
            }
            const bool saved = replaying_;
            replaying_ = true;
            try {
              loadSnapshot(stage + ".snap");
            } catch (...) {
              replaying_ = saved;
              tApplying = false;
              throw;
            }
            replaying_ = saved;
            tApplying = false;
          }
          unlink((stage + ".snap").c_str());
          unlink((stage + ".snap.users").c_str());
          next = lsn + 1;
          appliedLsn_ = lsn;
          needBootstrap = false;
          bootstrapped = true;
          setState("Streaming");
        }
        std::string body;
        try {
          body = httpGet(base + "/replication/wal?from=" + std::to_string(next) + "&id=" + urlEsc(myId) + "&epoch=" + epoch);
          replLastContactUs_ = nowUsR();
        } catch (SqlError& e) {
          if (std::string(e.what()).find("HTTP 410") != std::string::npos) {
            replLastContactUs_ = nowUsR();
            needBootstrap = true;
            continue;
          }
          throw;
        }
        if (body.empty()) replCaughtUpUs_ = nowUsR();  // nothing new: caught up
        size_t p = 0;
        tApplying = true;
        while (p + 12 <= body.size()) {
          uint64_t lsn;
          uint32_t n;
          memcpy(&lsn, body.data() + p, 8);
          memcpy(&n, body.data() + p + 8, 4);
          std::string rec = body.substr(p + 12, n);
          p += 12 + n;
          size_t sep = rec.find('\x1f');
          std::string sql = sep == std::string::npos ? rec : rec.substr(sep + 1);
          if (sep != std::string::npos) currentDb() = rec.substr(0, sep);
          static const std::regex loadRe(R"(^(\s*LOAD\s+DATA\s+INFILE\s+')([^']*)('[\s\S]*)$)", std::regex::icase);
          std::smatch m;
          std::string local;
          if (std::regex_match(sql, m, loadRe)) {  // fetch the file the primary loaded
            local = stage + "-load-" + std::to_string(lsn);
            writeFile(local, httpGet(base + "/replication/file?path=" + urlEsc(m[2].str())));
            sql = m[1].str() + local + m[3].str();
          }
          try {
            execute(sql);
          } catch (std::exception& e) {
            std::lock_guard<std::mutex> g(replMu_);
            replError_ = "lsn " + std::to_string(lsn) + ": " + e.what();
          }
          if (!local.empty() && dir_.empty()) unlink(local.c_str());
          appliedLsn_ = lsn;
          next = lsn + 1;
        }
        tApplying = false;
        setState("Streaming");
      } catch (std::exception& e) {
        tApplying = false;
        setState("Retrying", e.what());
        for (int i = 0; i < 20 && replRun_; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }
    }
  });
}

ResultSet Engine::replicationStatus() {
  ResultSet r;
  r.isResult = true;
  r.names = {"Role", "Primary", "State", "Applied_Lsn", "Primary_Lsn", "Last_Error", "Lag_Ms", "Replicas", "Sync_Replicas"};
  r.types = {LType::VARCHAR, LType::VARCHAR, LType::VARCHAR, LType::BIGINT, LType::BIGINT, LType::VARCHAR, LType::BIGINT, LType::VARCHAR, LType::BIGINT};
  uint64_t lsn;
  {
    std::lock_guard<std::mutex> g(walMu_);
    lsn = lsn_;
  }
  std::lock_guard<std::mutex> g(replMu_);
  const int64_t now = nowUsR();
  if (isReplica_) {
    const int64_t cu = replCaughtUpUs_;
    r.rows.push_back({"REPLICA", replicaOf_.substr(replicaOf_.find('@') == std::string::npos ? 0 : replicaOf_.find('@') + 1), replState_,
                      std::to_string((uint64_t)appliedLsn_), kNullCell, replError_.empty() ? kNullCell : replError_,
                      cu ? std::to_string(std::max<int64_t>(0, (now - cu) / 1000 - 1000)) : kNullCell, kNullCell, kNullCell});
  } else {
    // replicas seen in the last 10 s, with what they have applied
    std::string reps;
    for (auto& [id, a] : replAcks_)
      if (now - a.second < 10000000) reps += (reps.empty() ? "" : ", ") + id + "@" + std::to_string(a.first);
    r.rows.push_back({"PRIMARY", kNullCell, replicaOf_.empty() ? "Online" : "Online (promoted)", std::to_string(lsn), std::to_string(lsn), kNullCell,
                      kNullCell, reps.empty() ? kNullCell : reps, std::to_string(gSyncReplicas)});
  }
  return r;
}

void Engine::promote() {
  if (!isReplica_) throw SqlError("this server is not a replica", 1105);  // (also after an automatic failover)
  replRun_ = false;
  if (replThread_.joinable()) replThread_.join();
  isReplica_ = false;
}

}  // namespace gdb
