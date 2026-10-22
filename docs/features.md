# Features and SQL coverage

## SQL and feature coverage

Close to MySQL (and SingleStore's dialect) for everyday use (any MySQL driver; prepared statements, multi-statement
queries and `LOAD DATA LOCAL INFILE` are supported on the wire):

| Area | Supported |
|---|---|
| Types | INT family, BIGINT, DOUBLE, exact DECIMAL(p,s) (scaled int64), DATE, DATETIME/TIMESTAMP, VARCHAR/TEXT/JSON (dictionary encoded, case-insensitive collation by default; `SET collation_connection = utf8mb4_bin` for binary), `VECTOR(n)` (float32) |
| NULLs | Everywhere: three-valued logic, `IS [NOT] NULL`, NULL-skipping aggregates, outer-join NULLs, `\N` in LOAD DATA |
| DDL | CREATE/DROP TABLE (also `AS SELECT`, `LIKE`), REFERENCE tables, ALTER TABLE (add/drop/rename/modify column, rename, add/drop index), TRUNCATE, RENAME TABLE, CREATE/DROP INDEX, views, databases (`USE`, `db.table`) |
| Columns | DEFAULT (incl. CURRENT_TIMESTAMP), NOT NULL, AUTO_INCREMENT + `LAST_INSERT_ID()`, PRIMARY KEY, UNIQUE / secondary KEY (CPU hash indexes, lookups bypass the GPU), SHARD KEY, SORT KEY |
| DML | INSERT (column lists, multi-row, `SET`), INSERT .. SELECT, INSERT IGNORE, REPLACE, ON DUPLICATE KEY UPDATE with `VALUES()`, UPDATE / DELETE with any expression or subquery, LOAD DATA (CSV quoting, column lists, `LOCAL`, `FORMAT JSON` with `(col <- key::sub)` mappings, .gz/.zst/.lz4 files), `LOAD DATA S3` / `KAFKA` / `https://..`, SELECT .. INTO OUTFILE |
| Queries | Inner / LEFT / RIGHT / FULL OUTER / CROSS / NATURAL joins, `JOIN .. USING`, any ON condition (equi-joins hash-join on the GPU, also on expressions of each side such as `a.x = b.y + 1` or `a.d = DATE(b.ts)`, whose keys are computed on the GPU, and on DOUBLE columns by bit pattern; non-equi and mixed ON conditions run as a GPU join or cross join plus a GPU filter over the pairs, outer joins then add the preserved side's unmatched rows); joins also work on string columns (codes are remapped between the tables' dictionaries on the GPU). Subqueries (scalar, IN, EXISTS; equality-correlated ones are decorrelated into joins, other correlated scalar subqueries run once per distinct outer value and are joined back), CTEs incl. `WITH RECURSIVE`, derived tables (simple projections/filters of one table are merged into the outer query), UNION [ALL] / INTERSECT / EXCEPT, DISTINCT, GROUP BY / HAVING, window functions (ranking, LAG/LEAD, FIRST/LAST/NTH_VALUE, NTILE, framed aggregates), full-text `MATCH .. AGAINST`, `FOR UPDATE` |
| Aggregates | GPU: COUNT / SUM / AVG / MIN / MAX (+ DISTINCT). Host aggregation (GPU filters/joins, CPU groups): GROUP_CONCAT (ORDER BY, SEPARATOR), STRING_AGG, JSON_ARRAYAGG, JSON_OBJECTAGG, STDDEV / VARIANCE (POP / SAMP), MEDIAN, BIT_AND/OR/XOR, ANY_VALUE, DISTINCT aggregates over different columns, GROUP BY on multi-column string expressions |
| Functions | ~100: math, string (CONCAT, SUBSTRING, REPLACE, LPAD, REGEXP_LIKE, ...), date/time (DATE_ADD, DATEDIFF, DATE_FORMAT, EXTRACT, TIMESTAMPDIFF, ...), JSON (`JSON_EXTRACT_*`, `->`, `->>`), CAST/CONVERT (strings to numbers MySQL-style: the leading numeric prefix, else 0; string = number comparisons and arithmetic on strings convert the same way), CASE/IF/COALESCE/NULLIF. String functions run once per distinct dictionary entry, on all cores; regexes use RE2 (linear time), and REGEXP_REPLACE with capture groups uses PCRE2's JIT when the runtime library is present (5x faster, RE2 semantics for `$`; RE2 takes over on errors or the backtracking limit) |
| Vector search | `DOT_PRODUCT`, `EUCLIDEAN_DISTANCE`, `<*>`, `<->`, `JSON_ARRAY_PACK`; exact kNN on the GPU; `VECTOR INDEX (col) INDEX_OPTIONS '{"index_type":..,"metric_type":"DOT_PRODUCT"}'`: with NVIDIA cuVS installed (`scripts/install-cuvs.sh`, loaded at run time) `CAGRA` / `HNSW_*` / `AUTO` (the default) build a CAGRA GPU graph per segment, searched together in one call with deleted rows masked out and WHERE evaluated on the GPU as a per-segment filter bitset (filtered ANN); without cuVS, or for `IVF_*`, the IVF index (k-means lists). `SET warpstore_cagra_itopk`, `warpstore_nprobe`, `warpstore_vector_index = OFF` for exact. `EMBED(text)`: sentence embeddings (all-MiniLM-L6-v2, 384 dims, `scripts/fetch-embed-model.sh`) computed by a BERT encoder on the GPU (cuBLAS, TF32), batched over columns: `UPDATE t SET v = EMBED(body)`, `ORDER BY v <*> EMBED('question') DESC LIMIT 10` |
| Pipelines | `CREATE PIPELINE p AS LOAD DATA FS '/dir/*.csv' \| S3 'bucket/prefix*' CONFIG '{"region","endpoint_url"}' CREDENTIALS '{"aws_access_key_id",..}' \| KAFKA 'host:9092/topic' INTO TABLE t [FORMAT JSON (col <- a::b, ..)]`, or `.. AS LOAD DATA MYSQL 'mysql://user:pw@host:3306/db' INTO DATABASE copy` (a live copy of a MySQL database, see below), START / STOP / DROP / SHOW PIPELINES (passwords masked). Each file / object loads once and Kafka offsets (all partitions; gzip / snappy / lz4 / zstd batches) survive crashes exactly once (both are recovered from the logged loads) |
| Transactions | BEGIN / START TRANSACTION / COMMIT / ROLLBACK, `SET autocommit = 0`; a transaction is one WAL record written at COMMIT (replayed all or nothing), ROLLBACK replays an undo log. Durability works like SingleStore's: async by default (a commit is acknowledged once its WAL record is in the OS page cache, which survives a process crash; a background thread fdatasyncs every 10 ms, `WARPSTORE_FLUSH_MS`, so a power loss can lose the last few ms), sync where asked: `CREATE DATABASE d WITH SYNC DURABILITY`, `SET warpstore_sync_durability = ON` for a session, or `--sync-durability` for everything (group commit into a pre-zeroed WAL, so one fdatasync serves all waiting commits). WAL format 2: CRC32 per record; a torn or corrupt tail is cut off at recovery. SNAPSHOT (also run automatically once the WAL reaches `--snapshot-trigger-mb`, default 1024; 0 turns it off) writes the committed state only and rotates the WAL before installing itself, so a crash at any point recovers each change exactly once (checked with `kill -9` during transfers, inserts and snapshots). Read committed: other sessions don't see a transaction's inserts / updates / deletes until COMMIT (their scans get per-batch visibility bitmaps), and their writes to the same table wait for it (lock wait timeout 10 s). Long loads release the table lock every 256K rows. **MVCC**: a SELECT reads a per-statement snapshot of each table (copy-on-write deleted-row bitmaps, immutable segments, append-only write buffers, refcounted VRAM mirrors) and holds no table lock while it runs, so writers never wait for readers: with dashboards running, single-row inserts and updates went from ~1.2K/s (p99 15 ms) to ~8K/s (p99 2 ms). Key lookups (PRIMARY KEY / secondary index equality) still take a microsecond lock on the live table; `WARPSTORE_NO_MVCC=1` turns snapshots off. **Atomic commits**: a COMMIT is made durable (and acknowledged by semi-sync replicas) first, then stamped with one global sequence number in all of its tables at once, and a statement reading several tables takes all its snapshots at one moment, so readers see a transaction everywhere or nowhere. Write conflicts wait only for the transactions that hold the rows; deadlocks are detected through a wait-for graph (error 1213, the victim's transaction is rolled back, as in MySQL) instead of waiting out the 10 s lock timeout. A failed WAL write (disk full) fails the COMMIT and rolls it back; an autocommit write that can't be logged turns the server read-only until a restart |
| Procedural SQL | `CREATE [OR REPLACE] PROCEDURE p(a INT, ..) AS DECLARE .. BEGIN .. END` (MPSQL) or MySQL style: variables, `SET` / `:=`, IF / ELSEIF, WHILE, LOOP, REPEAT, `FOR i IN a .. b`, `FOR r IN SELECT ..` (cursor loops, `r.col`), EXIT / CONTINUE, `SELECT .. INTO`, `SCALAR(..)`, `ECHO SELECT`, RAISE / SIGNAL, `EXCEPTION WHEN OTHERS`, CALL, nested calls. `CREATE FUNCTION`: single-`RETURN` functions are inlined into queries (so they run on the GPU). Functions with procedural bodies also run over columns: the body runs once per distinct argument value, via a dictionary map for strings or a value-range map for INT / DATE, so it works in WHERE, GROUP BY and aggregates. CALL returns every `ECHO` / bare `SELECT` as a separate result set. Session `@variables` |
| Security | `CREATE USER .. IDENTIFIED BY` (mysql_native_password), roles (`CREATE ROLE`, `GRANT role TO u`), `GRANT / REVOKE` SELECT, INSERT, UPDATE, DELETE, CREATE, DROP, ALTER, ALL on `*.*`, `db.*`, `db.table` (enforced per statement, incl. subqueries), SHOW GRANTS. Row-level security through views filtered on `CURRENT_USER()`. HTTP Data API with Basic auth. Server-side `LOAD DATA INFILE` and `SELECT .. INTO OUTFILE` need the FILE (SUPER) privilege (`LOAD DATA LOCAL` doesn't); information_schema shows only what the user may read. Listens on 127.0.0.1 by default (`--bind`), and refuses a non-loopback address while root has no password (`--root-password` / `WARPSTORE_ROOT_PASSWORD`). TLS on the MySQL protocol with a self-signed certificate generated in the data dir (or `--tls-cert` / `--tls-key`), `--require-secure-transport`; random handshake challenges (OpenSSL), failed-login throttling (10 per minute per address), `--max-connections` |
| Clustering | `--leaves http://leaf1:httpport,http://leaf2:httpport|http://leaf2-replica:httpport` makes a server an aggregator over leaf servers (SingleStore's aggregator / leaf split). DDL runs everywhere; REFERENCE tables are replicated to every leaf. INSERT rows and LOAD DATA lines go to leaf hash(shard key); UPDATE / DELETE are broadcast. Co-located queries (one sharded table, or sharded tables joined on their shard keys, plus reference tables) run on the leaves: aggregations as partial aggregates combined on the aggregator's GPU, other queries with ORDER BY / LIMIT merged there. Joins of sharded tables on other columns are made co-located stage by stage: a table is reshuffled leaf to leaf by its join key, or a co-located group is joined on the leaves and its result reshuffled by the next join key (filters pushed into the shuffles; `EXPLAIN` lists the stages). Joins where neither side is on its shard key reshuffle both sides; joins on expressions (`a.x = b.y + 1000`, `a.v * 2 = b.w + 1`, `LOWER(a.s) = b.t`, `ABS(a.n - 250) = b.m`, DATE columns, string = number) reshuffle the tables with the expression values as computed shard keys (integers hashed as integers; other numbers normalized to DOUBLE and strings / dates hashed as text on both sides), also when a side is a co-located group joined on the leaves first. Other queries fetch each sharded table's needed columns (WHERE conjuncts pushed down) and finish on the aggregator. Results and shuffled rows travel in a columnar binary format (typed int64 / float64 arrays, NULL flags, strings), appended to the receiving tables without per-cell parsing, over a dedicated data plane: each server listens on `--data-port` (default HTTP port + 10000; `-1` disables it) for length-prefixed binary frames on persistent, authenticated TCP connections, and leaves without one are called over HTTP. A leaf runs its part of a shuffle one partition at a time (the scans of partitioned tables restricted to that partition; co-located joins stay correct partition by partition) and sends each partition's rows in 256K-row chunks, so it holds about 1/partitions of its shuffle output at once. 4M-row joins that reshuffle one or both sides by the join key run in 0.3 s on 2 local leaves. `ADD LEAF 'url'` / `REBALANCE PARTITIONS` move rows to the new hash layout with binary bulk inserts; a leaf with a replica fails over to it (`PROMOTE`) when the primary is down, found by queries or by health checks every 2 s. `SHOW LEAVES`; `EXPLAIN` shows the leaf SQL |
| Replication / HA | `--replica-of http://user:pw@primary:httpport`: a replica bootstraps from a snapshot of the primary, then tails its WAL over HTTP (incl. the files of LOAD DATA / pipelines); it is read-only, `SHOW REPLICATION STATUS` (lag, the primary's replicas), `PROMOTE` for failover. `--failover-timeout S` on a replica promotes it by itself when the primary has been unreachable for S seconds (2 s in the tests); `--sync-replicas N` on the primary makes a commit wait until N replicas have applied it (semi-sync; falls back to async after `--sync-replica-timeout-ms`, default 2000), so a failover loses no acknowledged commit. A primary restart is detected (incarnation epoch) and replicas re-bootstrap |
| Backups | `BACKUP DATABASE d TO '/dir'` or `TO S3 'bucket/prefix' CONFIG .. CREDENTIALS ..` (streamed, multipart above 64 MB): a consistent snapshot plus a manifest with its WAL position. `RESTORE DATABASE d [AS other] FROM '/dir' \| S3 .. [UNTIL TIMESTAMP '2026-10-17 12:00:00']`: with `--wal-archive DIR` rotated WALs are kept (they carry a time marker every second), so a restore can replay up to any point in time. Server-side `LOAD DATA INFILE` keeps a zstd-compressed copy of the file in the data dir (written while loading), so deleting the source file doesn't lose data on a crash; a load that fails part-way keeps the rows before the bad line and replays to the same state |
| Observability | `GET /metrics` on the HTTP port (Prometheus: statements by kind, latency histogram, errors, connections, rows, tables / segments / write buffers, VRAM, WAL size / fsyncs / fsync time, replication role / lag / replicas, plan cache, RSS). `SHOW [FULL] PROCESSLIST` / `information_schema.PROCESSLIST` (real sessions), `KILL [QUERY \| CONNECTION] id` (queries stop between execution stages and during loads), `information_schema.QUERY_STATS` (count / errors / total / avg / max ms / rows per statement shape, literals replaced; `FLUSH QUERY_STATS`), slow query log in MySQL's format (`<data dir>/slow.log`, over 1 s by default; `--slow-query-ms`, `SET GLOBAL long_query_time / slow_query_log`), `SELECT SLEEP(n)` |
| Metadata | information_schema with MySQL 8's column layouts (TABLES, COLUMNS, STATISTICS, KEY_COLUMN_USAGE, TABLE_CONSTRAINTS, REFERENTIAL_CONSTRAINTS, ROUTINES, ...; upper-case column labels like MySQL), system variables with MySQL defaults (`@@sql_mode`, `@@lower_case_table_names`, ..), `mysql` / `sys` / `performance_schema` schemas, SHOW CREATE TABLE / INDEX / FULL TABLES / VARIABLES / TABLE STATUS / PIPELINES, DESCRIBE |
| Storage | Compaction: a background pass (every 5 s) merges a partition's segments that are a quarter deleted, or small (snapshots and OPTIMIZE flush small ones), into new sorted segments; it builds them outside the table lock and swaps them in under it (rows deleted meanwhile, keys and indexes follow; `WARPSTORE_NO_COMPACT=1` turns it off). Tables larger than RAM: see [Disk tier](#disk-tier-tables-larger-than-ram). Tables larger than VRAM: segments beyond `--vram-limit` (or when VRAM is full) stay in pinned host RAM and are read over PCIe. `SNAPSHOT` checkpoints and `BACKUP DATABASE .. TO` / `RESTORE` files are columnar (dictionaries + physical column bytes per partition) and zstd-compressed: ClickBench 10M rows snapshot in 10 s to 0.99 GB and restart in 25 s (the earlier row format: 62 s, 1.9 GB, 99 s restart). Older row-format snapshots still load. After loads and restarts, large string dictionaries are sorted in the background (MIN/MAX/ORDER BY on strings), and dictionary-derived caches (sort order, LIKE bitmaps, string-function maps) are extended incrementally when inserts add strings instead of being rebuilt. VRAM cache: when columns spill to the host tier, scans count reads per segment column and a background thread (every 250 ms) keeps the hottest columns in VRAM, moving colder ones out only for a column 1.5x hotter (copies outside the table lock, pointer swaps under it; `WARPSTORE_NO_VRAM_CACHE=1` keeps the static first-come placement). GPU compression: integer segments are frame-of-reference packed to 8/16/32 bits, low-cardinality DOUBLE segments dictionary-coded, and decimal-valued DOUBLEs (prices like 406.1) stored as 16/32-bit integers / 10^s; on automatically once VRAM is 60% full (`--compress` always, `--no-compress` never). Generated kernels decode through out-of-line loads, so compile time stays flat |

String functions run "encoded": an expression over one string column is evaluated once per
dictionary entry, and the GPU applies the resulting code map. Expressions the GPU can't evaluate
(e.g. CONCAT of two columns) are computed on the host over the result rows.

### Still missing

- **Clustering**: `--redundancy 2` mirrors every leaf's shard on the next leaf. Writes go to both
  copies, and reads fail over to the mirror while a leaf is down. A leaf that comes back is resynced
  from its mirror by the health checks: its shard is copied back, and the mirror it keeps for the
  previous leaf is rebuilt. ADD LEAF and REBALANCE rebuild all the mirrors. Writes wait while a
  leaf resyncs. The other option is separate WAL-tailing replica processes (`--replica-of`,
  `primary|replica` leaf URLs).
- **Concurrency**: read committed (SingleStore's level) by default. `SET TRANSACTION ISOLATION LEVEL
  REPEATABLE READ` and `START TRANSACTION WITH CONSISTENT SNAPSHOT` give snapshot isolation (more than
  SingleStore offers): commits made after the snapshot stay invisible to the transaction (visibility
  bitmaps over change sets kept until no snapshot needs them; long snapshot transactions delay
  write-buffer flushes). Plain SELECTs read per-statement MVCC snapshots and never block writers. Write conflicts are row-level: UPDATE / DELETE wait only when a target row
  has uncommitted changes from another transaction, and INSERTs wait only on tables with a primary or
  unique key.
- **ACID**: no SERIALIZABLE (it maps to snapshot isolation, so write skew is possible), foreign keys are not enforced, DDL commits implicitly (as in MySQL), and cluster writes across leaves have no two-phase commit.
- **SQL**: triggers (SingleStore has none). Foreign keys are kept as metadata (SHOW CREATE TABLE, information_schema) but not enforced. LATERAL derived tables only for subqueries yielding at most one row (aggregates, LIMIT 1: what ORMs generate). (Queries whose WHERE / GROUP BY /
  aggregates use host-only functions over several columns, in joins too, are retried as a derived
  table that evaluates them on the host.)
- **Types**: BIGINT arithmetic (+ - *) is exact in projections, filters and aggregates; DIV and
  functions go through double. Aggregates of integer columns are exact (128-bit sums without GROUP
  BY, grouped BIGINT sums as two int64 sums of 32-bit halves, int64 MIN / MAX, NULLs skipped; a SUM
  beyond 64 bits comes back as DOUBLE). Overflow raises error 1690 (BIGINT value is out of range), on
  the GPU too. DECIMAL arithmetic is computed in double and rounded to the MySQL result scale
  (literals like `0.1` are DECIMAL, `/` adds 4 digits, AVG of exact numbers is DECIMAL).

### ORM / driver compatibility

Checked end to end (schema push / create_all, CRUD, relations, transactions, prepared statements):
**Drizzle** (drizzle-kit push is idempotent apart from a drizzle-kit composite-PK quirk that also
happens on MySQL; 29/29 checks incl. relational queries, which use LATERAL + JSON_ARRAYAGG), **Prisma
6** (`db push`, 20/20 client checks: nested writes, upsert, groupBy, `_count`, interactive
transactions, raw queries), **SQLAlchemy 2** over PyMySQL and mysql-connector (14/14 each: create_all,
reflection, selectin / joined loading, IntegrityError), mysql2 over TLS. Fixes this took: JOIN ..
USING, db.table.column references, DATABASE(), CONSTRAINT .. UNIQUE / FOREIGN KEY, DEFAULT (now()),
`VALUES (DEFAULT, ..)`, ON DUPLICATE KEY / IGNORE / REPLACE on UNIQUE keys, JSON_ARRAY / JSON_OBJECT
and nested JSON_ARRAYAGG, JSON result columns (MYSQL_TYPE_JSON), string comparisons across tables,
COALESCE over aggregates in correlated subqueries, result column names, `SET @@session.x`, BINARY,
prepared-statement result metadata at PREPARE.

### MySQL change data capture (warpstore as a live copy of MySQL)

```sql
CREATE PIPELINE shop_cdc AS LOAD DATA MYSQL 'mysql://cdc:secret@mysql-host:3306/shop' INTO DATABASE shop;
START PIPELINE shop_cdc;      -- snapshot copy, then follows the binlog
SHOW PIPELINES;               -- Streaming, transactions / rows applied, last error
LOAD DATA MYSQL 'mysql://cdc:secret@mysql-host:3306/shop' INTO DATABASE shop_copy;  -- one-off copy
```

Point dashboards, analytics and vector / full-text search at warpstore while MySQL stays the system of
record. The pipeline copies a consistent snapshot of the source's tables (types mapped: integers,
DECIMAL, dates and times, strings, ENUM / SET, JSON, BIT, ...), then follows the binary log
(row format: MySQL 8's default) and applies each source transaction as one warpstore transaction,
together with the binlog position it reached (`_warpstore_cdc_position`), so a crash resumes exactly
once. DDL on the source (CREATE / ALTER / DROP / TRUNCATE / RENAME TABLE) re-copies, drops or empties
the table. Speed on this machine: 200K-row snapshot 0.7 s; inserts follow in real time; a 100K-row
UPDATE arrives 1.2 s after it starts. The user needs `SELECT, REPLICATION SLAVE, REPLICATION CLIENT`;
connections use TLS when the server offers it (`?ssl=0` to turn it off) and support
caching_sha2_password (MySQL 8's default) and mysql_native_password. Options: `?tables=a,b`,
`&server_id=N`. Tables without a primary key are kept in sync too (deletes match all columns).

### Vector search and embeddings

300K 128-dim vectors with local structure (a 16-dim manifold, like real embeddings), top 10:

| | p50 | recall@10 |
|---|---:|---:|
| exact (GPU) | 41 ms | 1.000 |
| IVF index | 3.6 ms | 0.90 |
| CAGRA (cuVS), itopk 64 | 1.0 ms | 1.000 |
| CAGRA + `WHERE cat = 3` / `id < 20000` / `id < 3000` | 2.2-2.4 ms | 1.000 |

Graphs build in ~0.3 s per 37K-row segment (in the background after a restart). `EMBED()` matches
ONNX Runtime's output for the same model (cosine 1.000000) and embeds ~9K short texts/s on the RTX 3080;
`UPDATE docs SET v = EMBED(body)` over 20K rows takes 3.1 s, a semantic search ~10 ms.

Also: MATCH .. AGAINST uses an inverted index per column (built on first use, or in the background for
`FULLTEXT` columns, extended as rows arrive): a new search term costs ~1-9 ms instead of a scan of every
document. `caching_sha2_password` (MySQL 8 / 9 clients' default) is supported next to
mysql_native_password. The first query after a restart no longer waits for the background dictionary
sort (connection close / COMMIT took every table's lock).

### VRAM placement hints

`CREATE TABLE .. VRAM = PIN | AUTO | NEVER`, per column `col INT VRAM = NEVER`, `ALTER TABLE t VRAM =
PIN`, `ALTER TABLE t ALTER COLUMN c SET VRAM = NEVER`. PIN columns go to VRAM first, NEVER columns stay
in host RAM (read over PCIe), AUTO ones are placed by the VRAM cache by heat. `SHOW VRAM` lists every
column's VRAM / host MB, hint and heat. Hints are part of SHOW CREATE TABLE, so they persist.

### Disk tier (tables larger than RAM)

Column segments (1M rows each, immutable) get a file in `<data dir>/segments`. Under
`--memory-limit GB` (default: half the machine's RAM; `0`: no limit; `WARPSTORE_MEMORY_LIMIT`), a
background thread evicts the least-used column data from host memory once 90% of the limit is used,
down to 80% (writing the segment file first if it has none), and reads columns back, hottest first,
while under 75%. A query that touches a column that is on disk reads it into the cache if there is
room, otherwise for the scan only (just the rows of the batch, when the SORT KEY narrows it, read
ahead on I/O threads for the next segments): aggregations and filters run over windows of batches (1/8 of VRAM
by default, `WARPSTORE_SCAN_WINDOW_MB`) and free each window before reading the next, so one query's
memory stays bounded however large the table is. `SHOW SEGMENTS FROM t` shows `disk-columnstore` for
segments with columns on disk; `/metrics` has `warpstore_table_memory_bytes`.

`SNAPSHOT` refers to the segment files instead of copying the data (segments with more than half of
their rows deleted are still rewritten, which compacts them), so it only writes what changed, and a
restart takes seconds: tables come back on disk and the cache fills up in the background. Files no
snapshot or segment uses any more are deleted. `BACKUP` files stay self-contained.
String dictionaries live in append-only files too (`<data dir>/dicts`, zstd frames with CRC32s):
a snapshot appends the strings added since the last one before it pauses writes, and records how
many strings it covers (a torn tail from a snapshot that was never installed is cut off on load).
ClickBench 10M rows: writes pause about 50 ms per snapshot (3.8 s when the dictionaries were part of
the snapshot), and a restart takes 1.6 s.

Still in memory, whatever the limit: write buffers (up to 1M rows per partition), primary-key and
secondary indexes, string dictionaries, deleted-row bitmaps. `ALTER TABLE` rebuilds, `BACKUP` and
CPU-side DML on many rows read whole columns into memory. `SELECT .. ORDER BY .. LIMIT n` (n up to
10,000) scans only the filter and ORDER BY columns and reads the others for the n rows it returns. Segment
files store each column in zstd-compressed blocks of 16K rows (the SORT KEY column uncompressed) with a
block index, so reads of row ranges and single rows only decompress the blocks they need (files from
before, stored uncompressed, still load). With NVIDIA nvCOMP installed (`scripts/install-nvcomp.sh`,
loaded at run time; `WARPSTORE_NVCOMP_DIR`), scans copy the compressed blocks to the GPU and decompress
them there, batched per flush (before the kernels that read them); reads under 1 MB, and the cache's own
loads, are decompressed on the CPU. `WARPSTORE_GPU_DECOMPRESS=0` turns it off. GPUDirect Storage:
with the nvidia-fs driver loaded, the compressed blocks are read straight into VRAM through cuFile
(`WARPSTORE_GPUDIRECT=auto` (default) | `on` (also in cuFile's compatibility mode) | `off`). The server
log says which are in use at startup.

## Usage

```bash
./build/warpstore --port 3307 --data-dir ./warpstore-data     # or --no-wal; --partitions N; --codegen on|sync|off;
                                                      # --plancache-dir DIR (default <data-dir>/plancache); -v
```

```sql
CREATE TABLE events (id BIGINT, user_id INT, amount DOUBLE, ts DATE, channel VARCHAR(10),
                     PRIMARY KEY (id), SORT KEY (ts));      -- shard key defaults to id
CREATE TABLE clicks (event_id BIGINT, ts DATE, SHARD KEY (event_id));  -- co-located with events
CREATE REFERENCE TABLE channels (name VARCHAR(10), owner VARCHAR(20)); -- small, unsharded
LOAD DATA INFILE '/path/events.tbl' INTO TABLE events FIELDS TERMINATED BY '|';
INSERT INTO events VALUES (1, 42, 9.99, '2025-01-01', 'web');
SELECT channel, count(*), sum(amount) FROM events WHERE ts >= '2024-06-01' GROUP BY channel;
EXPLAIN SELECT ...;          -- shows route, segments scanned/eliminated, join order
SHOW SEGMENTS FROM events;   -- GPU segments + write buffer
OPTIMIZE TABLE events;       -- flush the write buffers into GPU segments
SHOW SEGMENTS FROM events;   -- per partition: segments + write buffer
```
