#include "monitor.h"

#include <sys/socket.h>
#include <time.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <map>
#include <unordered_map>

#include "common.h"

namespace gdb {

thread_local Session* tSession = nullptr;

void checkKilled() {
  if (tSession && tSession->killQuery.load(std::memory_order_relaxed)) {
    tSession->killQuery = false;
    throw SqlError("Query execution was interrupted", 1317);
  }
}

namespace monitor {

std::atomic<double> slowMs{-1};
std::string slowPath;
std::atomic<uint64_t> slowCount{0};

namespace {

int64_t nowUs() {
  return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

std::mutex sessMu;
std::map<uint32_t, std::shared_ptr<Session>> sessMap;

// Counters for /metrics.
std::atomic<uint64_t> connTotal{0}, stmtErrors{0};
const char* kKinds[] = {"select", "insert", "update", "delete", "load", "ddl", "txn", "other"};
std::atomic<uint64_t> stmtByKind[8];
const double kBuckets[] = {1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000};
constexpr int kNB = sizeof kBuckets / sizeof kBuckets[0];
std::atomic<uint64_t> hist[kNB + 1];
std::atomic<uint64_t> histSumUs{0};
std::atomic<uint64_t> rowsTotal{0};

int kindOf(const std::string& sql) {
  size_t i = 0;
  while (i < sql.size() && (isspace((unsigned char)sql[i]) || sql[i] == '(')) ++i;
  std::string w;
  while (i < sql.size() && isalpha((unsigned char)sql[i]) && w.size() < 12) w += (char)toupper((unsigned char)sql[i++]);
  if (w == "SELECT" || w == "WITH" || w == "SHOW" || w == "EXPLAIN" || w == "DESCRIBE" || w == "DESC") return 0;
  if (w == "INSERT" || w == "REPLACE") return 1;
  if (w == "UPDATE") return 2;
  if (w == "DELETE") return 3;
  if (w == "LOAD") return 4;
  if (w == "CREATE" || w == "DROP" || w == "ALTER" || w == "TRUNCATE" || w == "RENAME") return 5;
  if (w == "BEGIN" || w == "START" || w == "COMMIT" || w == "ROLLBACK" || w == "SAVEPOINT" || w == "RELEASE") return 6;
  return 7;
}

std::mutex statMu;
std::unordered_map<std::string, QueryStat> stats;
constexpr size_t kMaxDigests = 5000;

std::mutex slowMu;

std::string isoTime(int64_t us) {
  time_t s = us / 1000000;
  struct tm tm;
  gmtime_r(&s, &tm);
  char b[64];
  snprintf(b, sizeof b, "%04d-%02d-%02dT%02d:%02d:%02d.%06lldZ", tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec,
           (long long)(us % 1000000));
  return b;
}

}  // namespace

std::shared_ptr<Session> open(uint32_t id, const std::string& user, const std::string& host, int fd) {
  auto s = std::make_shared<Session>();
  s->id = id;
  s->user = user;
  s->host = host;
  s->fd = fd;
  s->connectedUs = s->sinceUs = nowUs();
  ++connTotal;
  std::lock_guard<std::mutex> g(sessMu);
  sessMap[id] = s;
  return s;
}

void close(uint32_t id) {
  std::lock_guard<std::mutex> g(sessMu);
  sessMap.erase(id);
}

std::vector<std::shared_ptr<Session>> sessions() {
  std::lock_guard<std::mutex> g(sessMu);
  std::vector<std::shared_ptr<Session>> out;
  for (auto& [id, s] : sessMap) out.push_back(s);
  return out;
}

bool kill(uint32_t id, bool queryOnly) {
  std::shared_ptr<Session> s;
  {
    std::lock_guard<std::mutex> g(sessMu);
    auto it = sessMap.find(id);
    if (it == sessMap.end()) return false;
    s = it->second;
  }
  {
    std::lock_guard<std::mutex> g(s->mu);
    if (s->command != "Sleep") s->killQuery = true;  // (only a running statement; an idle session has nothing to stop)
  }
  if (!queryOnly) {
    s->killConn = true;
    if (s->fd >= 0) shutdown(s->fd, SHUT_RDWR);  // the connection thread sees the socket close
  }
  return true;
}

void begin(Session* s, const std::string& sql, const std::string& db) {
  if (!s) return;
  std::lock_guard<std::mutex> g(s->mu);
  s->command = "Query";
  s->info = sql.size() > 4096 ? sql.substr(0, 4096) : sql;
  s->db = db;
  s->sinceUs = nowUs();
  s->killQuery = false;
}

std::string digest(const std::string& sql) {
  std::string o;
  o.reserve(std::min<size_t>(sql.size(), 2048));
  bool space = false;
  for (size_t i = 0; i < sql.size() && o.size() < 2048;) {
    const char c = sql[i];
    if (isspace((unsigned char)c)) {
      space = !o.empty();
      ++i;
      continue;
    }
    if (space) o += ' ', space = false;
    if (c == '\'' || c == '"') {  // string literal
      const char q = c;
      for (++i; i < sql.size(); ++i) {
        if (sql[i] == '\\') ++i;
        else if (sql[i] == q) {
          if (i + 1 < sql.size() && sql[i + 1] == q) ++i;
          else break;
        }
      }
      ++i;
      o += '?';
      continue;
    }
    const bool prevWord = !o.empty() && (isalnum((unsigned char)o.back()) || o.back() == '_' || o.back() == '`');
    if ((isdigit((unsigned char)c) || (c == '.' && i + 1 < sql.size() && isdigit((unsigned char)sql[i + 1]))) && !prevWord) {
      while (i < sql.size() && (isalnum((unsigned char)sql[i]) || sql[i] == '.')) ++i;  // number (incl. 1e5, 0x1f)
      o += '?';
      continue;
    }
    o += c;
    ++i;
  }
  // IN (?, ?, ?, ..) -> IN (...): the same statement with a different list length is one digest
  for (size_t p = 0; (p = o.find("?, ?", p)) != std::string::npos;) {
    size_t e = p + 1;
    while (o.compare(e, 3, ", ?") == 0) e += 3;
    o.replace(p, e - p, "...");
    p += 3;
  }
  for (size_t p = 0; (p = o.find("?,?", p)) != std::string::npos;) {  // (no spaces)
    size_t e = p + 1;
    while (o.compare(e, 2, ",?") == 0) e += 2;
    o.replace(p, e - p, "...");
    p += 3;
  }
  while (!o.empty() && (o.back() == ';' || o.back() == ' ')) o.pop_back();
  return o;
}

void end(Session* s, const std::string& sql, const std::string& db, double ms, int64_t rows, bool error, const std::string& plan) {
  const int64_t now = nowUs();
  if (s) {
    std::lock_guard<std::mutex> g(s->mu);
    s->command = "Sleep";
    s->info.clear();
    s->sinceUs = now;
  }
  ++stmtByKind[kindOf(sql)];
  if (error) ++stmtErrors;
  int b = 0;
  while (b < kNB && ms > kBuckets[b]) ++b;
  ++hist[b];
  histSumUs += (uint64_t)(ms * 1000);
  if (rows > 0) rowsTotal += (uint64_t)rows;
  {
    std::string d = digest(sql);
    std::lock_guard<std::mutex> g(statMu);
    auto it = stats.find(d);
    if (it == stats.end()) {
      if (stats.size() >= kMaxDigests) {  // drop the least recently seen tenth
        std::vector<std::pair<int64_t, std::string>> age;
        for (auto& [k, v] : stats) age.push_back({v.lastUs, k});
        std::nth_element(age.begin(), age.begin() + kMaxDigests / 10, age.end());
        for (size_t i = 0; i < kMaxDigests / 10; ++i) stats.erase(age[i].second);
      }
      it = stats.emplace(d, QueryStat{}).first;
      it->second.digest = d;
    }
    QueryStat& q = it->second;
    ++q.count;
    q.errors += error;
    q.totalMs += ms;
    q.maxMs = std::max(q.maxMs, ms);
    q.rows += std::max<int64_t>(rows, 0);
    q.lastUs = now;
  }
  const double th = slowMs.load();
  if (th >= 0 && ms >= th && !slowPath.empty()) {
    ++slowCount;
    std::lock_guard<std::mutex> g(slowMu);
    if (FILE* f = fopen(slowPath.c_str(), "a")) {
      fprintf(f, "# Time: %s\n# User@Host: %s[%s] @ %s []  Id: %u\n# Query_time: %.6f  Lock_time: 0.000000 Rows_sent: %lld  Rows_examined: 0%s%s\n",
              isoTime(now).c_str(), s ? s->user.c_str() : "", s ? s->user.c_str() : "", s ? s->host.c_str() : "", s ? s->id : 0, ms / 1000.0,
              (long long)std::max<int64_t>(rows, 0), plan.empty() ? "" : "  Plan: ", plan.c_str());
      if (!db.empty()) fprintf(f, "use %s;\n", db.c_str());
      fprintf(f, "SET timestamp=%lld;\n%s%s\n", (long long)(now / 1000000), sql.c_str(), !sql.empty() && sql.back() == ';' ? "" : ";");
      fclose(f);
    }
  }
}

std::vector<QueryStat> queryStats() {
  std::lock_guard<std::mutex> g(statMu);
  std::vector<QueryStat> out;
  out.reserve(stats.size());
  for (auto& [k, v] : stats) out.push_back(v);
  std::sort(out.begin(), out.end(), [](const QueryStat& a, const QueryStat& b) { return a.totalMs > b.totalMs; });
  return out;
}

void resetQueryStats() {
  std::lock_guard<std::mutex> g(statMu);
  stats.clear();
}

std::string prometheus() {
  std::string o;
  char b[256];
  auto line = [&](const char* fmt, auto... a) {
    snprintf(b, sizeof b, fmt, a...);
    o += b;
  };
  size_t active = 0, running = 0;
  for (auto& s : sessions()) {
    ++active;
    std::lock_guard<std::mutex> g(s->mu);
    running += s->command != "Sleep";
  }
  o += "# HELP warpstore_connections Open client connections.\n# TYPE warpstore_connections gauge\n";
  line("warpstore_connections %zu\n", active);
  o += "# HELP warpstore_running_statements Statements executing now.\n# TYPE warpstore_running_statements gauge\n";
  line("warpstore_running_statements %zu\n", running);
  o += "# HELP warpstore_connections_total Connections accepted since start.\n# TYPE warpstore_connections_total counter\n";
  line("warpstore_connections_total %llu\n", (unsigned long long)connTotal.load());
  o += "# HELP warpstore_statements_total Statements executed, by kind.\n# TYPE warpstore_statements_total counter\n";
  for (int k = 0; k < 8; ++k) line("warpstore_statements_total{kind=\"%s\"} %llu\n", kKinds[k], (unsigned long long)stmtByKind[k].load());
  o += "# HELP warpstore_statement_errors_total Statements that failed.\n# TYPE warpstore_statement_errors_total counter\n";
  line("warpstore_statement_errors_total %llu\n", (unsigned long long)stmtErrors.load());
  o += "# HELP warpstore_statement_duration_seconds Statement latency.\n# TYPE warpstore_statement_duration_seconds histogram\n";
  uint64_t cum = 0;
  for (int i = 0; i < kNB; ++i) {
    cum += hist[i].load();
    line("warpstore_statement_duration_seconds_bucket{le=\"%g\"} %llu\n", kBuckets[i] / 1000.0, (unsigned long long)cum);
  }
  cum += hist[kNB].load();
  line("warpstore_statement_duration_seconds_bucket{le=\"+Inf\"} %llu\n", (unsigned long long)cum);
  line("warpstore_statement_duration_seconds_sum %.6f\n", histSumUs.load() / 1e6);
  line("warpstore_statement_duration_seconds_count %llu\n", (unsigned long long)cum);
  o += "# HELP warpstore_rows_total Rows returned or affected.\n# TYPE warpstore_rows_total counter\n";
  line("warpstore_rows_total %llu\n", (unsigned long long)rowsTotal.load());
  o += "# HELP warpstore_slow_queries_total Statements over the slow query threshold.\n# TYPE warpstore_slow_queries_total counter\n";
  line("warpstore_slow_queries_total %llu\n", (unsigned long long)slowCount.load());
  return o;
}

}  // namespace monitor
}  // namespace gdb
