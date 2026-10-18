# Development

## Tests

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
scripts/test-containers.sh                 # optional: S3 / Kafka / MySQL containers for the ingest and CDC tests
.venv/bin/python tests/run.py              # everything that can run on this machine (~2-3 min)
.venv/bin/python tests/run.py fuzz tpch    # by name
WARPSTORE_FUZZ_N=5000 WARPSTORE_FUZZ_SEED=7 .venv/bin/python tests/run.py fuzz
```

The runner starts throwaway servers on free ports (one shared server for the SQL tests, own servers
with data dirs for the rest) and skips what the machine can't run:

| Test | What it checks |
|---|---|
| smoke, postagg, sharding | basic SQL, post-aggregation expressions, partition pruning / co-located joins against DuckDB |
| tpch | all 22 TPC-H queries against DuckDB on generated data (SF 0.1, `WARPSTORE_TPCH_SF`) |
| fuzz | random queries (expressions, CASE, COALESCE, NULLs, DECIMAL, joins, GROUP BY / HAVING, windows, subqueries, UNION, derived tables, CTEs, ORDER BY .. LIMIT) on the same random data in warpstore and DuckDB; it found five wrong-result bugs on its first run |
| orm-sqlalchemy, orm-drizzle, orm-prisma | the ORM suites (SQLAlchemy on pymysql and mysql-connector; the Node projects install themselves on first run) |
| mvcc | writers don't wait for a slow reader, consistent reads under concurrent transfers, flush / DROP / dictionary growth under a reader |
| crash | `kill -9` during transfers, inserts and snapshots: every acknowledged write survives, sums stay exact |
| backup | full restore, restore under another name, point-in-time restore over a WAL rotation, S3, deleted LOAD DATA files |
| replication | WAL-tailing replica, primary restart, semi-sync read-your-writes, automatic failover, replica of the promoted primary |
| observability | processlist, KILL, query stats, slow log, /metrics |
| ingest-s3, ingest-kafka, binlog | S3 / Kafka / MySQL CDC against local containers `warpstore-s3` (SeaweedFS), `warpstore-redpanda`, `warpstore-mysql` (MySQL 8.4) when they exist |

## Measuring query speed

Everything is SQL, from any MySQL client, per connection:

```sql
PROFILE SELECT ...;        -- runs the query (returns its rows) and records per-stage timings
SHOW PROFILE;              -- stage, time_ms, rows, detail: scans (segments, rows, join filters),
                           -- joins (co-located / broadcast), aggregation method, codegen plan used
EXPLAIN SELECT ...;        -- the plan: routing, segment elimination, join order and strategy
SHOW PLANCACHE;            -- compiled plans: origin (compiled / disk), compile time, executions, avg time
SHOW PLAN 3;               -- the generated CUDA source of plan 3
DROP ALL FROM PLANCACHE;
SET warpstore_codegen = ON | SYNC | OFF;      -- ON: interpret first, compile in background (default)
SET warpstore_dense_groupby = ON | OFF;       -- A/B switches for the optimizations
SET warpstore_join_filters = ON | OFF;
SET warpstore_join_filter_rows = 2000000;     -- tables at least this big are scanned with join filters
```

`python bench/qtime.py "SELECT ..." [--compare]` wraps these: min/median/p95 over N runs, the
profile, the plan, and with `--compare` the query timed with each optimization switched off.
`python bench/qtime.py --plancache` lists compiled plans.
