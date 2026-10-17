// Query engine: binder, bytecode compiler, planner (CPU point-lookup vs GPU scan/join/aggregate
// routing, segment elimination) and DML, plus a logical write-ahead log for durability.
#pragma once
#include <cstdio>
#include <map>
#include <thread>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <vector>
#include "sql.h"
#include "storage.h"

namespace gdb {

extern const std::string kNullCell;

// Set by the cluster's columnar endpoint: SELECT results keep typed values (ResultSet::vals).
extern thread_local bool tTypedResult;
extern thread_local bool tRawNumber;  // a single numeric SELECT column comes back as ResultSet::numCol
extern thread_local bool tColumnar;   // an all-numeric GROUP BY result comes back as ResultSet::numCols
// >= 0: scans of partitioned (non-reference) tables read only this partition (a shuffle streams a
// leaf's rows one partition at a time; co-located joins stay correct partition by partition).
extern thread_local int tOnlyPart;
struct ResultSet;
// Columnar binary encoding of a result (typed or text rows), optionally only the given rows.
std::string encodeColumnar(const ResultSet& r, const std::vector<size_t>* rows = nullptr);
void devFree(void* p);  // gpu.h
struct ResultSet {
  bool isResult = false;
  std::vector<std::string> names;
  std::vector<LType> types;
  std::vector<uint8_t> jsonCols;  // per column: 1 = JSON (sent as MYSQL_TYPE_JSON, so drivers parse it)
  std::vector<std::vector<std::string>> rows;
  std::vector<std::vector<Value>> vals;  // typed rows instead of `rows` (nested queries)
  bool typed = false;
  std::shared_ptr<std::vector<double>> numCol;  // set instead of rows for single-number results when asked (tRawNumber)
  std::vector<std::vector<double>> numCols;     // set instead of rows for all-numeric results when asked (tColumnar)
  std::vector<std::shared_ptr<struct Dict>> colDicts;  // numCols: VARCHAR columns hold codes of these dictionaries
  // A GPU-built result (derived tables): one cudaMalloc'd column per item, taken over by makeTemp.
  struct DevColumn {
    void* p = nullptr;
    double mn = 0, mx = 0;
    ~DevColumn() { if (p) devFree(p); }
  };
  std::vector<std::shared_ptr<DevColumn>> devCols;
  int64_t devRows = 0;
  int64_t affected = 0;
  int64_t lastInsertId = 0;
  std::string info;
  std::vector<ResultSet> extra;  // further result sets (a CALL whose body produced several), sent in order
};

// Per-statement state of the query layer.
struct QueryCtx {
  std::vector<std::string> temps;           // catalog keys of temp tables to drop
  std::map<std::string, std::string> ctes;  // lower(CTE name) -> temp table name
};
struct Pipeline;  // CREATE PIPELINE: background file ingestion

// Set by the server while it runs a LOAD DATA LOCAL it staged (no FILE privilege needed for that).
extern thread_local bool tLocalInfile;
// Set while the server describes a prepared statement: SELECTs return their columns without rows.
extern thread_local bool tDescribeOnly;

class Engine {
 public:
  Engine(const std::string& dataDir, bool syncWal, int partitions);
  ~Engine();  // stops pipelines
  ResultSet execute(const std::string& sql);
  // Users (CREATE USER .. IDENTIFIED BY): mysql_native_password check of a handshake response.
  bool checkAuth(const std::string& user, const std::string& scramble, const std::string& token);
  void setUser(const std::string& user);
  void connectionClosed();  // rolls back an open transaction, drops TEMPORARY tables
  void recover();
  // Stored procedures / functions (proc.cpp).
  bool routineStatement(const std::string& sql, ResultSet& out);
  Value callFunction(const std::string& name, const std::vector<std::string>& args);
  std::vector<std::string> routineDefs();
  // Clustering (cluster.cpp): this server is an aggregator over these leaves (HTTP API base URLs).
  void setLeaves(const std::vector<std::string>& leaves);
  void startHealthChecks();
  void copyLeafTables(const std::string& src, const std::string& srcSfx, const std::string& dst, const std::string& dstSfx);
  void resyncLeaf(size_t i);
  void rebuildMirrors();
  ResultSet clusterStatus();
  // Leaf side of cluster shuffles (HTTP /api/v2/insert and /api/v2/reshuffle).
  int64_t insertBinary(const std::string& db, const std::string& table, const std::string& body);
  std::string reshuffle(const std::string& body);
  // Replication (repl.cpp).
  std::string replSnapshot();
  std::string replSince(uint64_t from, int waitMs, int& status);
  bool replFileAllowed(const std::string& path);
  void startReplica(const std::string& url);
  bool isReplica() const;
  ResultSet replicationStatus();
  void promote();
  // Passwords for HTTP Basic auth; SUPER = ALL ON *.*.
  bool checkPassword(const std::string& user, const std::string& pw);
  bool isSuper(const std::string& user);

 private:
  Catalog cat_;
  // VRAM cache: a background thread moves segment columns between VRAM and the host tier by heat.
  void vramBalance();
  std::atomic<bool> vramStop_{false};
  std::thread vramTh_;
  std::string dir_;
  FILE* wal_ = nullptr;
  std::mutex walMu_;
  bool sync_;
  // WAL (format 2: "GDBWAL2\n", then records of u32 length, u32 CRC32, payload). Group commit: a
  // committing writer waits until an fdatasync covers its record; one sync serves all waiters.
  void walAppendLocked(const std::string& rec);  // caller holds walMu_
  void walDurable(bool wait);                     // after releasing walMu_: wait for the disk (sync durability)
  void walSyncNow();                              // fdatasync everything written so far (group: one sync for all waiters)
  bool needsSync(const std::string& db);          // sync durability for writes to db (database option, session, server default)
  std::thread flushTh_;                           // async durability: syncs the WAL every 10 ms
  void walOpen(bool truncate);
  int64_t replayWal(const std::string& path, bool repairTail);
  std::mutex syncMu_;
  std::condition_variable syncCv_;
  uint64_t walWritten_ = 0, walSynced_ = 0;  // records written (walMu_) / known durable (syncMu_)
  int64_t walOff_ = 0, walCap_ = 0;  // logical end of the WAL / end of its pre-zeroed space (walMu_)
  bool syncing_ = false;
  bool replaying_ = false;
  std::map<std::string, std::shared_ptr<Pipeline>> pipes_;
  std::mutex pipesMu_;
  std::set<std::string> replayedLoads_;  // files loaded by replayed LOAD statements (pipelines skip them)
  std::mutex usersMu_;
  std::map<std::string, std::string> users_{{"root", ""}};  // user -> SHA1(SHA1(password)) (empty: no password)
  // Privileges: user or role -> "PRIV|scope" (scope *.*, db.*, db.table) and "@role" entries.
  std::map<std::string, std::set<std::string>> grants_;
  bool hasPriv(const std::string& user, const std::string& priv, const std::string& db, const std::string& table);
  void checkPrivileges(const Stmt& s);
  ResultSet doUser(Stmt& s);
  ResultSet doBackup(Stmt& s);
  void writeSnapshot(const std::string& path, const std::string& onlyDb, uint64_t* lsnOut = nullptr);
  // Replication state: recent WAL records by lsn (guarded by walMu_), replica thread.
  void replAppend(const std::string& rec);
  std::vector<std::string> leaves_;
  bool isDistributed(const std::string& table);
  bool clusterStatement(const std::string& sql, Stmt& s, ResultSet& out);
  ResultSet clusterSelect(Stmt& s, const std::string& sql, bool explain);
  ResultSet clusterAdmin(const std::string& sql);
  int64_t clusterInsertRows(Table& t, const std::vector<std::string>& cols, const std::vector<std::vector<std::string>>& rows,
                            const std::vector<std::vector<std::string>>& keyText, const std::string& verb, const std::string& tail);
  std::deque<std::pair<uint64_t, std::string>> replRing_;
  uint64_t lsn_ = 0;
  size_t replBytes_ = 0;
  std::condition_variable replCv_;
  std::mutex replMu_;
  std::set<std::string> replFiles_;
  std::string replicaOf_, replState_, replError_;
  std::atomic<bool> isReplica_{false}, replRun_{false};
  std::atomic<uint64_t> appliedLsn_{0};
  std::thread replThread_;
  ResultSet doPipeline(Stmt& s);
  // Remote sources (S3, Kafka, HTTP) and FORMAT JSON: fetched data is staged to a local file and
  // loaded with a (logged) LOAD DATA INFILE.
  ResultSet loadRemote(Stmt& s);
  int64_t ingest(const std::string& table, char delim, int64_t skip, const std::vector<std::string>& cols,
                 const std::vector<std::string>& jsonPaths, bool json, const std::string& data, const std::string& stagedName);
  void runRemotePipeline(const std::shared_ptr<Pipeline>& p);
  void startPipeline(const std::shared_ptr<Pipeline>& p);
  void stopPipeline(Pipeline& p);
  ResultSet showPipelines();
  int nparts_;  // partitions per sharded table (SingleStore: partitions per database)

  void logWrite(const std::string& sql, const std::string& db = "");
  ResultSet doTxn(const std::string& what);
  ResultSet doSnapshot();
  void loadSnapshot(const std::string& path);
  void commitTxn();
  void rollbackTxn();
  ResultSet doSelect(Stmt& s, bool explain);
  // Query layer: CTEs, derived tables, subqueries, UNION, COUNT(DISTINCT) on top of doSelect.
  ResultSet query(Stmt& s, QueryCtx& q, bool explain, bool typed);
  ResultSet queryImpl(Stmt& s, QueryCtx& q, bool explain, bool typed);
  ResultSet queryRetry(Stmt& s, QueryCtx& q, bool explain, bool typed);
  ResultSet runStatementQuery(Stmt& s, bool explain);
  std::string makeTemp(const ResultSet& r, QueryCtx& q, const std::string& hint);
  void resolveNaturalJoins(Stmt& s);
  void expandTableFunctions(Stmt& s);  // FROM fn(args) -> its SELECT  // NATURAL JOIN -> equalities on the shared column names
  void flattenDerived(Stmt& s);       // SELECT .. FROM (SELECT cols FROM t WHERE ..) x -> one query over t
  void resolveSubqueries(Stmt& s, QueryCtx& q);
  ResultSet windows(Stmt& s, QueryCtx& q, bool explain, bool typed);
  ResultSet hostAggregate(Stmt& s, QueryCtx& q, bool explain, bool typed);
  void expandStars(Stmt& s);
  void dropTemps(QueryCtx& q);
  int64_t insertRows(Table& t, const std::vector<int>& map, std::vector<std::vector<Value>>& rows, const Stmt& s);
  ResultSet doInsert(Stmt& s);
  ResultSet doCreate(Stmt& s);
  ResultSet doDrop(Stmt& s);
  ResultSet doModify(Stmt& s);
  ResultSet modifyViaSelect(Stmt& s);
  ResultSet groupingSetsQuery(Stmt& s, QueryCtx& q, bool explain, bool typed);
  ResultSet applyModify(Table& t, const std::vector<Loc>& matched, const std::vector<std::vector<Value>>& newRows);  // DELETE / UPDATE
  ResultSet doLoad(Stmt& s);
  ResultSet doOptimize(Stmt& s);
  ResultSet doShowTables(Stmt& s);
  ResultSet doAlter(Stmt& s);
  ResultSet doTruncate(Stmt& s);
  ResultSet doRename(Stmt& s);
  ResultSet doShowMisc(Stmt& s);
  ResultSet infoSchema(const std::string& table);
  std::string showCreate(const Table& t);
  std::shared_ptr<Table> rebuild(Table& t, const std::vector<ColumnMeta>& cols, const std::vector<int>& from);
  ResultSet doDescribe(Stmt& s);
  ResultSet doShowSegments(Stmt& s);
};

}  // namespace gdb
