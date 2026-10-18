// MySQL change data capture: warpstore as a live copy of a MySQL database. A consistent snapshot of the
// source's tables is loaded first, then the source's binary log (row format) is followed and every
// source transaction is applied as one warpstore transaction, together with the binlog position it
// reached (table _warpstore_cdc_position in the target database), so a restart resumes exactly once.
//
//   CREATE PIPELINE p AS LOAD DATA MYSQL 'mysql://user:pw@host:3306/db' INTO DATABASE copy; START PIPELINE p
//   LOAD DATA MYSQL 'mysql://user:pw@host:3306/db' INTO DATABASE copy      (one-off copy, no follow)
//
// URL options (?a=b&..): tables=t1,t2 (default: every base table), ssl=0 (no TLS), server_id=N.
// The source needs binlog_format=ROW and binlog_row_image=FULL (MySQL 8's defaults); the user needs
// SELECT, REPLICATION SLAVE and REPLICATION CLIENT.
#pragma once
#include <atomic>
#include <cstdint>
#include <functional>
#include <string>

namespace gdb {

class Engine;

struct CdcStatus {
  std::string state, error, position;
  int64_t rows = 0, txns = 0;
};

// Runs until `running` goes false (follow) or the snapshot is loaded (!follow). Throws SqlError for
// configuration errors before anything is copied; later errors are reported (and retried).
void mysqlCdcRun(Engine& e, const std::string& url, const std::string& target, const std::string& pipeName, const std::atomic<bool>& running,
                 bool follow, const std::function<void(const CdcStatus&)>& report);

}  // namespace gdb
