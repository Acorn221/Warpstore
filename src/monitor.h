// Observability: the session registry (SHOW PROCESSLIST, KILL), per-statement accounting (query
// digests for information_schema.QUERY_STATS, the slow query log, Prometheus counters).
#pragma once
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace gdb {

struct Session {
  uint32_t id = 0;
  std::string user, host;
  int fd = -1;
  int64_t connectedUs = 0;
  std::atomic<bool> killQuery{false}, killConn{false};
  mutable std::mutex mu;  // guards the fields below
  std::string db, command = "Sleep", info;
  int64_t sinceUs = 0;  // start of the current command (or of the idle time)
};

extern thread_local Session* tSession;  // the session this thread serves (null: internal work)
// Throws "Query execution was interrupted" (1317) when KILL QUERY / KILL targeted this session.
void checkKilled();

struct QueryStat {
  std::string digest;
  uint64_t count = 0, errors = 0;
  double totalMs = 0, maxMs = 0;
  int64_t rows = 0, lastUs = 0;
};

namespace monitor {
std::shared_ptr<Session> open(uint32_t id, const std::string& user, const std::string& host, int fd);
void close(uint32_t id);
std::vector<std::shared_ptr<Session>> sessions();
// KILL [QUERY] id. False when there is no such session.
bool kill(uint32_t id, bool queryOnly);
void begin(Session* s, const std::string& sql, const std::string& db);  // a statement starts
// ... and ends (rows: sent or affected; plan: the engine's plan summary, for the slow log).
void end(Session* s, const std::string& sql, const std::string& db, double ms, int64_t rows, bool error, const std::string& plan);
std::string digest(const std::string& sql);  // literals -> ?, whitespace collapsed
std::vector<QueryStat> queryStats();
void resetQueryStats();
std::string prometheus();  // the counters kept here, in the Prometheus text format

// Slow query log (MySQL's format: pt-query-digest reads it). Off when slowMs < 0 or no path.
extern std::atomic<double> slowMs;
extern std::string slowPath;
extern std::atomic<uint64_t> slowCount;
}  // namespace monitor

}  // namespace gdb
