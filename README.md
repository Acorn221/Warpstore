# Warpstore

**A GPU-accelerated, MySQL-compatible HTAP database.** Speak the MySQL protocol to it from any
driver or ORM; transactional writes land in a CPU write buffer, analytics run as CUDA kernels over
column segments kept in GPU memory. Built in: vector search (GPU graph index), embeddings in SQL,
full-text search, S3 / Kafka / MySQL change-data-capture ingest, replication with failover, backups
with point-in-time restore, MVCC.

### 13x faster than DuckDB, 24x faster than ClickHouse, 87x faster than SingleStore

| ClickBench, all 43 queries (10M rows, hot) | Total time | Warpstore is faster by |
|---|---:|---:|
| **Warpstore** (one RTX 3080) | **0.23 s** | — |
| DuckDB 1.5.6 | 6.2 s | **13.5x** |
| ClickHouse 25.8 | 9.5 s | **24x** |
| SingleStore 9.1 | 30.1 s | **87x** |
| MySQL 8.4 (dashboard queries, 5M rows) | 1.2 to 16 s per query | **46x to 2,367x** |

<sub>Same machine (Ryzen 5 5500, 12 threads, + RTX 3080), geometric mean per query, Warpstore won 42
of 43 queries against DuckDB and all 43 against ClickHouse and SingleStore. At 10M rows the data fits
in VRAM; larger datasets spill to host memory and the gap narrows. Full tables, methodology and
caveats: [How it compares](#how-it-compares).</sub>

> **Status: experimental alpha.** Warpstore was developed as an experiment: how far can a GPU-first,
> MySQL-compatible database get, and how fast? It was built quickly, with heavy use of an AI coding
> assistant. It passes TPC-H, ClickBench, a differential fuzzer and ORM suites (see [Tests](#tests)),
> but it is young: NVIDIA GPUs only, one main developer, no production deployments. Don't make it
> the only home of data you care about; keep a system of record (for instance with warpstore as a live
> copy of MySQL, see [MySQL change data capture](#mysql-change-data-capture-warpstore-as-a-live-copy-of-mysql)).

## Quick start

**Docker** (needs an NVIDIA GPU and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)):

```bash
docker build -t warpstore .                      # ~10-20 min the first time (--build-arg JOBS=2 on small machines)
docker run --gpus all -p 3307:3307 -v warpstore-data:/data -e WARPSTORE_ROOT_PASSWORD=change-me warpstore
mysql -h 127.0.0.1 -P 3307 -u root -p        # or any MySQL driver / ORM
```

**From source** (Linux):

| Requirement | Version |
|---|---|
| NVIDIA GPU | Turing (RTX 20xx / T4) or newer; developed on an RTX 3080 |
| NVIDIA driver + CUDA toolkit | CUDA 12.x (nvcc, NVRTC, cuBLAS), tested with 12.6 |
| CMake, C++20 compiler | CMake 3.24+, GCC 11+ |
| Libraries | OpenSSL, libcurl, zstd, lz4, snappy, zlib, RE2 (PCRE2 optional): `sudo scripts/install-deps.sh` on Debian / Ubuntu |

```bash
sudo scripts/install-deps.sh                 # Debian / Ubuntu; CUDA from https://developer.nvidia.com/cuda-downloads
cmake -S . -B build && cmake --build build -j
./build/warpstore --port 3307 --data-dir ./warpstore-data
mysql -h 127.0.0.1 -P 3307 -u root
```

Optional extras: `scripts/install-cuvs.sh` (NVIDIA cuVS, ~3 GB: the GPU graph vector index; rerun
cmake afterwards) and `scripts/fetch-embed-model.sh` (the ~90 MB model behind `EMBED()`). Without
them vector indexes fall back to IVF and `EMBED()` reports that the model is missing.

By default the server listens on 127.0.0.1 only and root has no password; it refuses to listen on
other addresses until root has one (`--root-password` / `WARPSTORE_ROOT_PASSWORD`). TLS is on with a
generated certificate. `./build/warpstore --help` lists the options.

SingleStore is a trademark of SingleStore, Inc. warpstore is an independent project, not affiliated
with or endorsed by SingleStore; it supports much of SingleStore's SQL dialect (SHARD KEY, SORT KEY,
REFERENCE tables, pipelines, vector and JSON syntax) and is benchmarked against it below.

## How it compares

Measured 2026-10-18 unless noted. Same machine for the first two tables: Ryzen 5 5500 (6 cores / 12
threads), 54 GB RAM, RTX 3080 10 GB. All the data and scripts are in [bench/comparisons](bench/comparisons)
(`make_tables.py` prints these tables from the result files).

### ClickBench, measured on one machine

The 43 [ClickBench](https://github.com/ClickHouse/ClickBench) queries on the first 10M rows of its
`hits` table (the full 100M rows don't fit this GPU's 10 GB). Hot = best of 3 after one warm-up run;
first run = the first execution after a restart (Warpstore: compiled query plans cached on disk from
an earlier run; the cold-plan-cache figure is in brackets). Warpstore's results match DuckDB's on 35
queries; the other 8 differ only in the order of tied rows. SingleStore was measured on 2026-10-16
(`bench/clickbench/singlestore_local.py`; the dev image allows 8 threads and 2 partitions).

| System | Version / setup | All 43 queries, hot | All 43, first run | Warpstore faster, hot (geomean) | Warpstore faster, first run (geomean) | Queries Warpstore wins (hot) |
|---|---|---:|---:|---:|---:|---:|
| **Warpstore** | RTX 3080 10 GB | **0.23 s** | **1.69 s** (7.3 s with a cold plan cache) | — | — | — |
| DuckDB | 1.5.6, 12 threads | 6.20 s | 6.90 s | 13.5x | 6.3x | 42 / 43 |
| ClickHouse | 25.8, 12 threads, ClickBench schema, Docker | 9.54 s | 11.18 s | 24.0x | 10.4x | 43 / 43 |
| SingleStore | 9.1 free dev image, 8 threads, Docker | 30.07 s | 49.61 s | 86.9x | 84.6x | 43 / 43 |

<details><summary>Per query (milliseconds)</summary>

| Q | Warpstore hot | Warpstore first | DuckDB hot | DuckDB first | ClickHouse hot | ClickHouse first | SingleStore hot | SingleStore first |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.4 | 4.0 | 0.3 | 2.8 | 1.0 | 2.0 | 0.9 | 16.3 |
| 2 | 0.4 | 1.1 | 1.4 | 7.4 | 4.0 | 4.0 | 11.5 | 33.4 |
| 3 | 0.6 | 1.4 | 3.4 | 8.6 | 7.0 | 7.0 | 17.9 | 438.6 |
| 4 | 0.4 | 1.3 | 4.6 | 10.0 | 7.0 | 7.0 | 17.6 | 326.6 |
| 5 | 4.9 | 6.9 | 62.0 | 73.0 | 90.0 | 96.0 | 451.4 | 819.2 |
| 6 | 2.7 | 4.1 | 109.7 | 151.6 | 152.0 | 164.0 | 745.6 | 1181.8 |
| 7 | 0.4 | 1.4 | 0.5 | 1.5 | 4.0 | 4.0 | 1.0 | 22.8 |
| 8 | 1.3 | 2.4 | 2.6 | 3.2 | 6.0 | 7.0 | 6.1 | 50.9 |
| 9 | 6.0 | 7.6 | 87.7 | 83.6 | 102.0 | 107.0 | 227.4 | 689.9 |
| 10 | 8.3 | 11.6 | 127.3 | 137.9 | 108.0 | 114.0 | 586.2 | 1120.0 |
| 11 | 1.6 | 2.8 | 19.2 | 25.9 | 20.0 | 19.0 | 387.9 | 1363.3 |
| 12 | 1.7 | 3.0 | 21.0 | 25.5 | 28.0 | 27.0 | 404.2 | 1323.3 |
| 13 | 1.1 | 2.7 | 98.0 | 101.5 | 212.0 | 261.0 | 451.0 | 1152.2 |
| 14 | 4.8 | 6.7 | 204.0 | 208.8 | 252.0 | 289.0 | 955.6 | 1867.8 |
| 15 | 4.2 | 7.7 | 105.9 | 124.0 | 192.0 | 221.0 | 484.8 | 1208.1 |
| 16 | 6.4 | 7.0 | 77.6 | 81.5 | 88.0 | 111.0 | 243.1 | 648.5 |
| 17 | 8.0 | 9.1 | 191.3 | 194.5 | 404.0 | 582.0 | 686.4 | 1352.6 |
| 18 | 47.5 | 86.3 | 132.8 | 145.7 | 280.0 | 294.0 | 661.0 | 754.8 |
| 19 | 13.7 | 15.3 | 350.8 | 380.8 | 678.0 | 739.0 | 1692.8 | 2348.0 |
| 20 | 0.5 | 2.6 | 0.8 | 1.6 | 2.0 | 4.0 | 10.0 | 25.0 |
| 21 | 0.4 | 61.2 | 76.9 | 156.7 | 65.0 | 199.0 | 338.7 | 586.2 |
| 22 | 0.7 | 7.3 | 79.1 | 73.8 | 12.0 | 279.0 | 427.1 | 727.8 |
| 23 | 1.4 | 128.5 | 138.2 | 292.7 | 400.0 | 901.0 | 667.0 | 1088.7 |
| 24 | 5.2 | 6.7 | 72.8 | 114.8 | 149.0 | 139.0 | 882.6 | 1346.4 |
| 25 | 1.2 | 2.6 | 7.2 | 7.4 | 52.0 | 54.0 | 137.0 | 782.0 |
| 26 | 1.2 | 6.0 | 25.1 | 26.9 | 53.0 | 62.0 | 173.4 | 798.1 |
| 27 | 2.0 | 2.3 | 6.7 | 9.5 | 54.0 | 53.0 | 149.7 | 759.7 |
| 28 | 23.9 | 144.3 | 72.4 | 63.3 | 205.0 | 221.0 | 244.9 | 533.8 |
| 29 | 6.2 | 1001.3 | 1583.2 | 1790.9 | 2410.0 | 2468.0 | 8660.6 | 9688.5 |
| 30 | 1.4 | 2.7 | 9.7 | 11.8 | 13.0 | 13.0 | 394.1 | 1622.2 |
| 31 | 5.8 | 7.0 | 80.6 | 80.9 | 112.0 | 125.0 | 593.5 | 1264.6 |
| 32 | 6.2 | 7.3 | 98.6 | 111.8 | 162.0 | 197.0 | 1915.5 | 2636.5 |
| 33 | 23.6 | 25.0 | 674.9 | 703.7 | 1113.0 | 1237.0 | 1901.5 | 2582.7 |
| 34 | 3.1 | 4.2 | 690.4 | 674.1 | 828.0 | 846.0 | 1779.0 | 2189.8 |
| 35 | 10.4 | 11.6 | 707.8 | 709.6 | 836.0 | 855.0 | 1865.7 | 2413.1 |
| 36 | 6.3 | 7.7 | 100.5 | 109.2 | 61.0 | 73.0 | 451.8 | 907.8 |
| 37 | 1.0 | 2.8 | 39.6 | 42.1 | 89.0 | 96.0 | 319.5 | 577.3 |
| 38 | 1.0 | 1.4 | 9.0 | 9.2 | 35.0 | 38.0 | 276.2 | 373.5 |
| 39 | 0.9 | 2.6 | 8.5 | 12.4 | 26.0 | 30.0 | 140.6 | 285.1 |
| 40 | 3.7 | 24.8 | 100.6 | 102.1 | 195.0 | 203.0 | 508.6 | 1110.6 |
| 41 | 2.2 | 34.5 | 5.4 | 7.8 | 12.0 | 15.0 | 27.2 | 153.4 |
| 42 | 4.0 | 5.6 | 4.9 | 5.7 | 10.0 | 10.0 | 31.2 | 186.5 |
| 43 | 2.3 | 3.5 | 10.5 | 12.0 | 8.0 | 9.0 | 146.5 | 253.3 |

</details>

### Dashboards: MySQL vs Warpstore as a copy of it

Seven dashboard queries over the same 5M events (`bench/comparisons/dash_vs_mysql.py`), identical SQL,
identical results. MySQL 8.4 with a 5 GB buffer pool and indexes on the timestamp and user columns.
The one query MySQL answers from an index (one hour of data) is the closest; full-table aggregation
is where a copy in Warpstore pays off.

| Dashboard query | MySQL 8.4, hot | MySQL, first run | Warpstore, hot | Warpstore, first run | Warpstore faster (hot) |
|---|---:|---:|---:|---:|---:|
| channel totals since May | 1,420.0 ms | 1,453 ms | 0.60 ms | 5 ms | 2,367x |
| distinct users per country | 2,825.4 ms | 2,924 ms | 3.01 ms | 5 ms | 939x |
| daily series (ios) | 1,198.3 ms | 1,210 ms | 5.10 ms | 7 ms | 235x |
| top 10 users by spend | 7,620.2 ms | 7,648 ms | 7.26 ms | 8 ms | 1,050x |
| one hour, by country+channel | 3.4 ms | 4 ms | 0.61 ms | 1 ms | 6x |
| JSON field group-by | 4,222.7 ms | 4,223 ms | 4.76 ms | 8 ms | 887x |
| window rank per country | 16,461.4 ms | 16,774 ms | 354.72 ms | 662 ms | 46x |

### Against published ClickBench results (estimates)

Systems that couldn't be run here, from their latest results on an AWS c6a.4xlarge (16 vCPUs, 100M
rows). The estimate goes through DuckDB, which was measured both here and on that machine:
per query, (DuckDB here / Warpstore here) x (system / DuckDB on c6a.4xlarge), then the geometric
mean. Treat these as rough: for ClickHouse the estimate says ~6x while the direct measurement above
says 24x, so the method can be several times off. HeavyAI's and Kinetica's published results ran on
CPU-only machines (HeavyAI completed 20 of the queries), so they say nothing about their GPU modes.

| System | Published result | AWS machine (CPU) | Tuned | Queries compared | Its time vs DuckDB's (geomean) | Estimated Warpstore speedup |
|---|---|---|---|---:|---:|---:|
| Umbra | 2026-10-16 | c6a.4xlarge | no | 43 / 43 | 0.14x | ~2x |
| ClickHouse | 2026-10-17 | c6a.4xlarge | no | 43 / 43 | 0.42x | ~6x |
| HeavyAI | 2026-05-17 | c6a.4xlarge | no | 20 / 43 | 1.91x | ~15x |
| StarRocks | 2026-09-07 | c6a.4xlarge | no | 43 / 43 | 1.46x | ~20x |
| SingleStore | 2022-07-01 | c6a.4xlarge | yes | 38 / 43 | 1.84x | ~20x |
| Doris | 2026-05-10 | c6a.4xlarge | no | 43 / 43 | 1.57x | ~21x |
| Kinetica | 2026-05-17 | c6a.4xlarge | no | 42 / 43 | 1.60x | ~21x |
| TiDB (TiFlash only) | 2026-05-17 | c6a.4xlarge | no | 42 / 43 | 374.97x | ~4,972x |
| MySQL | 2025-07-11 | c6a.4xlarge | no | 43 / 43 | 902.61x | ~12,184x |
| PostgreSQL | 2026-09-21 | c6a.4xlarge | no | 43 / 43 | 1196.39x | ~16,150x |

### Against another GPU engine

[Sirius](https://github.com/sirius-db/sirius) (a GPU-native analytics engine, built as a DuckDB extension) published results on
an H100, which has about 4x the memory bandwidth of an RTX 3080. Scaled linearly from 10M to 100M rows,
Warpstore on the 3080 takes about twice Sirius's time per query on the H100, i.e. roughly level per
unit of memory bandwidth. A real 100M-row run on this GPU would spill to host RAM (see [When the data
doesn't fit in VRAM](#when-the-data-doesnt-fit-in-vram)) and be slower than the linear scaling.

| System | Hardware | Rows | All 43 queries, hot | All 43, cold run (ClickBench: includes reading from disk) / first run | Per query vs Warpstore (scaled to 100M rows) |
|---|---|---:|---:|---:|---:|
| Sirius (2026-03-09) | NVIDIA H100 (AWS p5.4xlarge) | 100M | 1.50 s | 977.92 s | 0.49x Warpstore's time |
| Warpstore | RTX 3080 10 GB | 10M | 0.23 s | 1.69 s | — (x10 rows: 2.3 s hot) |

**Caveats.** 10M rows favour engines with low per-query overhead; at 100M rows on this GPU Warpstore
would use its host-memory tier and lose some of its lead. DuckDB, ClickHouse and MySQL ran with their
default settings on a 6-core desktop. Warpstore is weakest on narrow index lookups (MySQL within 6x)
and on first runs before query plans are compiled.

## Architecture

```
 MySQL clients ──► MySQL protocol server (thread per connection)
                     │
                     ▼
               SQL parser → binder → planner ──────────────┐
                     │                                      │ routes each query:
      ┌──────────────┴───────────────┐                      │  • PK point lookup → CPU (µs, no GPU launch)
      ▼                              ▼                      │  • everything else → GPU
 CPU write buffer              GPU column segments           │
 (rowstore role: fast          (1M rows each, in VRAM;      │
  INSERT/UPDATE/DELETE,         host mirror = cold tier)     │
  PK hash index, WAL)          min/max metadata, deleted    │
      │   flush (sorted by       bitmasks, dict-encoded     │
      └──► SORT KEY on GPU) ──►  strings                    │
                                     │                      │
                                     ▼                      │
            CUDA kernels, compiled per query (NVRTC, ◄┘
            plan cache): fused scan+filter+aggregate (dense
            or hash), join-filtered scans, hash join, sort
```

## Compared with SingleStore's architecture

| SingleStore | warpstore |
|---|---|
| MySQL wire protocol | Same (v10 handshake, text protocol): any MySQL client/driver works |
| In-memory rowstore on top of the columnstore LSM | CPU write buffer; flushed to segments at 1M rows or `OPTIMIZE TABLE` |
| Columnstore segments + sort key + segment elimination | VRAM segments sorted by `SORT KEY`, min/max pruning (`EXPLAIN` shows it) |
| Deleted-row bitmasks; UPDATE = delete + insert | Same |
| Hash index for point lookups / unique keys | `PRIMARY KEY` hash index → CPU point-lookup route, duplicate-key errors |
| `SHARD KEY` hash partitioning, defaulting to the primary key | Same: rows hash to one of N partitions (`--partitions`, default 8), each with its own write buffer and segments. `SHARD KEY ()` is keyless (round-robin), `CREATE REFERENCE TABLE` is unsharded, and a primary key must contain the shard key (error 1705) |
| Single-partition queries | `shard_key = literal` scans one partition (`EXPLAIN`: "1/8 partitions (shard key routing)") |
| Co-located joins (no data movement) | Joins on both tables' shard keys run partition by partition with one small hash table per partition; join output keeps its partitioning, so multi-way joins stay co-located |
| Encoded execution (filter on dictionary codes) | `=`, `IN`, `LIKE`, ranges on strings are evaluated once on the dictionary, then a bitmap probe on the GPU |
| Encoded GROUP BY, "multiple counter arrays" | Dense GROUP BY: keys with small known ranges (dictionary codes, small ints, dates) index arrays directly instead of hashing, with up to 8 private copies per block in shared memory to spread hot keys |
| LLVM-compiled query plans + plan cache, `interpret_first` | Each query's expressions are translated from bytecode to typed CUDA C++, compiled with NVRTC and cached by shape (literals are parameters). New shapes run interpreted while compiling in the background. `SHOW PLANCACHE`, `SHOW PLAN <id>` |
| Join filters / runtime filters | Small tables are scanned first; big tables are scanned with exact bitmaps of the join keys already found, fused into the compiled scan filter |
| `PROFILE`, `SHOW PROFILE`, `EXPLAIN` | Same statements; per-stage time, rows and strategy (see "Measuring query speed") |
| Write-ahead log, recovery | Logical WAL, replayed on startup (`--sync` for fdatasync per commit) |

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

## Results (RTX 3080 10 GB vs DuckDB 1.5.6 on 12 CPU threads, same machine, same data)

`python bench/bench.py --sf 10`: TPC-H scale factor 10 (60M lineitem rows, about 5.2 GB in VRAM).
Medians of 5 warm runs, measured end to end through the MySQL protocol for warpstore (in-process for
DuckDB, on-disk database, 4 GB memory limit). **All results are checked against DuckDB and match.**

| query | warpstore | DuckDB | speedup |
|---|---:|---:|---:|
| Q1 pricing summary (scan + group by) | 14 ms | 287 ms | 20x |
| Q3 shipping priority (3-way join, 114K groups) | 8.4 ms | 173 ms | 21x |
| Q5 local supplier volume (6-way join) | 20 ms | 156 ms | 7.9x |
| Q6 forecast revenue (filter + sum) | 1.0 ms | 76 ms | 75x |
| Q12 shipping modes (join + CASE) | 8.2 ms | 112 ms | 14x |
| Q14 promotion effect (join + LIKE) | 5.5 ms | 143 ms | 26x |
| X1 high-cardinality GROUP BY (1M groups, top 10) | 10 ms | 375 ms | 38x |
| X2 filtered top-N projection | 3.6 ms | 30 ms | 8.3x |
| PK point lookup (incl. network round trip) | 0.07 ms | 0.55 ms | 8x |

### SingleStore SQL compatibility checklist

A 73-item probe of SingleStore SQL features (one representative statement each; script in the
session scratchpad, `features.py`) passes 69. Covered:
- **Table types:** ROWSTORE / COLUMNSTORE / REFERENCE / TEMPORARY tables, computed columns,
  UNIQUE and hash keys.
- **DDL and DML:** ALTER, TRUNCATE, INSERT .. ON DUPLICATE KEY / REPLACE / INSERT .. SELECT;
  multi-table UPDATE / DELETE, UPDATE / DELETE .. ORDER BY .. LIMIT, and correlated subqueries
  in DML.
- **Grouping and queries:** ROLLUP / CUBE / GROUPING SETS / GROUPING(), window functions,
  recursive CTEs, INTERSECT / EXCEPT.
- **JSON:** `::` / `::$` / `::%` and JSON_AGG.
- **Time series:** TIME_BUCKET and FIRST / LAST.
- **Statistics:** APPROX_COUNT_DISTINCT (exact), PERCENTILE_CONT / DISC .. WITHIN GROUP,
  APPROX_PERCENTILE and MEDIAN.
- **Geospatial:** GEOGRAPHY_POINT / DISTANCE / WITHIN_DISTANCE / CONTAINS / INTERSECTS /
  LONGITUDE / LATITUDE on WKT.
- **Search and vectors:** VECTOR functions and full-text MATCH.
- **Other functions:** SHA2, CRC32, TO/FROM_BASE64, REGEXP_SUBSTR / INSTR, CONVERT_TZ (offsets
  and IANA names), RAND, DIV.
- **Routines:** stored procedures, UDFs and table-valued functions (`FROM fn(args)`).
- **Transactions and security:** users, roles and grants.
- **Data loading and export:** pipelines, BACKUP and SELECT INTO OUTFILE.
- **Introspection:** information_schema, PROFILE, SHOW PARTITIONS / CREATE TABLE / PROCESSLIST,
  and views.

Also supported:
- SAVEPOINT / ROLLBACK TO [SAVEPOINT] / RELEASE SAVEPOINT: a partial undo inside a transaction.
- SQL-level `PREPARE s FROM '..?..'` / `EXECUTE s USING @a, ..` / `DEALLOCATE PREPARE s`, alongside
  protocol-level prepared statements.

**Parquet**: `LOAD DATA INFILE 'x.parquet' INTO TABLE t [FORMAT PARQUET]` works, from the local FS,
S3, HTTP and in pipelines; files are recognized by their magic bytes. A native reader handles flat
and nested schemas, PLAIN / dictionary / DELTA_BINARY_PACKED encodings, nullable columns, data pages v1 and
v2, and snappy / gzip / zstd compression. Logical types: DATE, TIMESTAMP (milli / micro / nano),
DECIMAL and legacy INT96 timestamps. Columns are matched to the table by name.

Nested columns load as JSON text. Structs become objects and maps become `{key: value}` objects.
Lists become arrays, in both the 3-level LIST and legacy repeated forms, lists of structs included,
keeping NULL vs empty lists and NULL elements distinct, nested to any depth (lists of lists, maps of
lists, lists of structs of lists). ClickBench's
`hits_0.parquet` (1M rows, 105 columns) loads in 16 s and matches DuckDB on every checked aggregate.

**ORC**: `LOAD DATA INFILE 'x.orc'` (or `FORMAT ORC`, or any file starting with ORC's magic bytes)
reads primitive columns, and nested LIST / MAP / STRUCT columns as JSON text (any depth). Compression: none, zlib, snappy, zstd. Encodings:
integer RLE v1 / v2 (all four v2 sub-encodings), byte / boolean RLE, and direct or dictionary
strings. Types: DATE, TIMESTAMP, DECIMAL, FLOAT / DOUBLE. Tested against pyarrow-written files.

**Avro**: `LOAD DATA INFILE 'x.avro'` (or `FORMAT AVRO`, or any file starting with Avro's magic
bytes) reads object container files. Codecs: null, deflate, snappy, zstandard. Supported types:
primitives, nullable unions, enums, fixed, and the date / timestamp / decimal logical types. Nested
records, arrays and maps are loaded as JSON text.

### All 22 TPC-H queries (SF1, unmodified SQL)

The 22 queries exactly as DuckDB's `tpch_queries()` ships them (standard SQL: correlated
subqueries, EXISTS / NOT EXISTS, derived tables with column lists, `SUBSTRING(x FROM a FOR b)`,
OR-factored join predicates, DECIMAL arithmetic), on the full TPC-H schema (8 tables, all columns,
shard keys, reference tables for nation / region): **22/22 results match DuckDB**, including on a
server limited to `--vram-limit 0.5`, where segments are compressed once VRAM is 60% full and the rest
spills to host memory. How subqueries run:
- IN subqueries return their values as one raw array, de-duplicated through a bitmap.
- EXISTS with a `<>` correlation becomes a single MIN / MAX grouping.
- Grouped derived tables come back as numeric columns written straight into the temporary table.

Queries whose outer joins all come at the end, such as decorrelated subqueries, have their
inner-join prefix ordered by cost; Q2 went from 57 to 23 ms. All 22 now run faster than DuckDB on the
server. A derived table built from a GROUP BY now stays in VRAM: its keys and aggregates are compacted
and written as the temp table's GPU segment, with no host round trip. That took Q21 from 200 to 23 ms,
Q20 from 47 to 9 ms and Q17 from 18 to 3 ms. IN / EXISTS subqueries over integer columns deduplicate in
a GPU bitmap (Q4: 100 to 22 ms). Q16's wall time (about 90 ms against DuckDB's 59) is mostly the Python
MySQL client parsing its 18K result rows; the server side takes 28 ms.

### ClickBench against SingleStore itself

The 43 ClickBench queries on the first 10M rows of `hits`, run on the same machine. warpstore runs on
the RTX 3080; SingleStore 9.1 runs in its free developer image on 8 of the Ryzen 5 5500's 12 hardware threads. warpstore is about
**95x faster hot and 94x on first runs** (geometric mean), and faster on all 43 queries. The sums
are 0.27 s vs 30.1 s hot and 2.6 s vs 49.6 s on first runs. The developer image is limited (2
partitions, 8 cores), so the conservative estimate, bridged through SingleStore's tuned published
result, is ~39x. Details are in `bench/clickbench/README.md`.

### When the data doesn't fit in VRAM

150M trades (4.2 GB raw; `bench/trades.py` data), 6 scan / group queries, hot totals; results identical in
every configuration. This machine's Ryzen 5 5500 is PCIe 3.0, so the host tier reads at ~12 GB/s
against ~760 GB/s from VRAM.

| VRAM budget | static placement (before) | VRAM cache + decimal-double encoding |
|---|---:|---:|
| unlimited (all in VRAM) | 28.7 ms | 28.7 ms |
| 1 GB | 455 ms | 99.5 ms |
| 0.5 GB | 557 ms | 235 ms |

ClickBench (10M rows, 2.3 GB compressed) with a 0.5 GB budget: hot total 0.83 s static, 0.72 s with
the cache, the same as with everything in VRAM (the hot columns fit). TPC-H SF1 still matches DuckDB
22/22 under `--vram-limit 0.5 --compress`.

### SingleStore's use cases, head to head (`bench/usecases.py`)

SingleStore markets itself for real-time dashboards, operational (OLTP) lookups, streaming ingest,
HTAP, vector / hybrid search, full-text search and JSON analytics. `bench/usecases.py` runs one small
workload per use case, with the same SQL, against warpstore and the local SingleStore 9.1 dev container
(8 threads, see above). Data: 5M events (id, user, ts, channel, country, DECIMAL amount, JSON) and 100K
docs (30-word text, VECTOR(128)). Clients are Python processes on the same machine; results match.

| use case | test | warpstore | SingleStore | |
|---|---|---:|---:|---:|
| dashboards | 6 aggregations, hot (filtered sum, COUNT DISTINCT, TIME_BUCKET series, top users, last hour, JSON group) | 0.4-7 ms | 4-181 ms | 9-86x |
| | window RANK per group over 353K groups | 249 ms | 250 ms | 1x |
| | 16 concurrent dashboard clients | 554 q/s | 17 q/s | 33x |
| OLTP | PK point reads, 16 clients | 40K/s | 3.2K/s | 12x |
| | PK updates, 16 clients | 25-30K/s | 2.5K/s | 10x |
| ingest | single-row INSERTs, 8 clients | 63K rows/s | 16K rows/s | 4x |
| | 1000-row INSERTs, 4 clients | 350K rows/s | 214K rows/s | 1.6x |
| | LOAD DATA 5M rows | 3.7 s | 33 s | 9x |
| HTAP | 4 inserters + 4 updaters + 4 point readers + 4 dashboards at once | 1.7K ins/s, 1.5K upd/s, 16K reads/s, 227 dash/s | 1.4K, 0.5K, 0.6K, 5 | 1.2-45x |
| vector | exact top-10 (DOT_PRODUCT, 100K x 128) | 13 ms | 610 ms | 45x |
| | filtered top-10 / full-text + vector | 0.9 ms / 0.8 ms | 77 ms / 114 ms | 90x / 140x |
| | IVF ANN top-10 (same recall) | 1.3 ms | 3.2 ms | 2.5x |
| full-text | MATCH .. AGAINST count / BM25 top-10 / group by, hot | 0.2-0.6 ms | 4-26 ms | 16-100x |
| | first query of a new term (no inverted index) | 230 ms | 5-220 ms | slower |

Write numbers with the default async durability on the final build: 48K single-row inserts/s (8 clients), about 950 inserts + 900 updates/s in the HTAP mix. With sync durability (fdatasync per group commit, this machine's consumer NVMe): 12.6K single-row inserts/s and about 470 + 470/s in the HTAP mix; a single client gets 3.3K commits/s. BM25 scores match SingleStore's (Lucene's formula). Getting here fixed three warpstore problems the
benchmark exposed: writers starved behind streams of analytic queries (the table lock now makes new
readers wait behind a waiting writer, PK point lookups excepted), every query after a write re-uploaded
the whole write buffer and rescanned it for key ranges (now incremental), and `std::regex_search` over
every statement made multi-MB INSERTs parse at 10 MB/s (7x faster now).

### Against SingleStore's published "trillion rows per second" query

SingleStore's headline benchmark ([blog](https://www.singlestore.com/blog/how-to-process-trillion-rows-per-second-ad-hoc-analytic-queries/))
ran `SELECT stock_symbol, count(*) as c FROM trade GROUP BY stock_symbol ORDER BY c desc LIMIT 10`
on 448 Xeon Skylake cores at 1.28 trillion rows/s, about **2.86 billion rows/s per core**. The same
query on 150M generated trades with skewed symbols (`python bench/gen_trades.py 150000000`,
`python bench/trades.py 150000000`) on one RTX 3080:

| query | time | rows/s | ≈ SingleStore cores |
|---|---:|---:|---:|
| top 10 symbols by trade count (their query) | 6.7 ms | 22 billion | 8 |
| same plus `sum(shares)` | 7.4 ms | 20 billion | 7 |
| `count(*) WHERE shares > 500` | 1.7 ms | 88 billion | 31 |

Their query took 46 ms on the interpreter-only engine; dense GROUP BY and codegen account for the
difference (`bench/qtime.py --compare`). It is still about 5x short of the card's memory bandwidth:
the remaining cost is shared-memory atomics on the hottest symbols.

HTAP mixed workload (`python bench/htap.py`, 10M-row table, 4 writers + point reader + analyst):
about 4,300 inserts/s + 1,800 updates/s + 8,300 point reads/s + 440 GPU aggregations/s (p50 1.3 ms)
at the same time. Every analytic query saw the row committed just before it (6664/6664 freshness
checks).

Caveats: DuckDB runs on DOUBLE columns here (not DECIMAL), and this machine had other heavy
processes running. Treat these as indicative numbers, not a rigorous benchmark.

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
| Transactions | BEGIN / START TRANSACTION / COMMIT / ROLLBACK, `SET autocommit = 0`; a transaction is one WAL record written at COMMIT (replayed all or nothing), ROLLBACK replays an undo log. Durability works like SingleStore's: async by default (a commit is acknowledged once its WAL record is in the OS page cache, which survives a process crash; a background thread fdatasyncs every 10 ms, `WARPSTORE_FLUSH_MS`, so a power loss can lose the last few ms), sync where asked: `CREATE DATABASE d WITH SYNC DURABILITY`, `SET warpstore_sync_durability = ON` for a session, or `--sync-durability` for everything (group commit into a pre-zeroed WAL, so one fdatasync serves all waiting commits). WAL format 2: CRC32 per record; a torn or corrupt tail is cut off at recovery. SNAPSHOT writes the committed state only and rotates the WAL before installing itself, so a crash at any point recovers each change exactly once (checked with `kill -9` during transfers, inserts and snapshots). Read committed: other sessions don't see a transaction's inserts / updates / deletes until COMMIT (their scans get per-batch visibility bitmaps), and their writes to the same table wait for it (lock wait timeout 10 s). Long loads release the table lock every 256K rows. **MVCC**: a SELECT reads a per-statement snapshot of each table (copy-on-write deleted-row bitmaps, immutable segments, append-only write buffers, refcounted VRAM mirrors) and holds no table lock while it runs, so writers never wait for readers: with dashboards running, single-row inserts and updates went from ~1.2K/s (p99 15 ms) to ~8K/s (p99 2 ms). Key lookups (PRIMARY KEY / secondary index equality) still take a microsecond lock on the live table; `WARPSTORE_NO_MVCC=1` turns snapshots off. **Atomic commits**: a COMMIT is made durable (and acknowledged by semi-sync replicas) first, then stamped with one global sequence number in all of its tables at once, and a statement reading several tables takes all its snapshots at one moment, so readers see a transaction everywhere or nowhere. Write conflicts wait only for the transactions that hold the rows; deadlocks are detected through a wait-for graph (error 1213, the victim's transaction is rolled back, as in MySQL) instead of waiting out the 10 s lock timeout. A failed WAL write (disk full) fails the COMMIT and rolls it back; an autocommit write that can't be logged turns the server read-only until a restart |
| Procedural SQL | `CREATE [OR REPLACE] PROCEDURE p(a INT, ..) AS DECLARE .. BEGIN .. END` (MPSQL) or MySQL style: variables, `SET` / `:=`, IF / ELSEIF, WHILE, LOOP, REPEAT, `FOR i IN a .. b`, `FOR r IN SELECT ..` (cursor loops, `r.col`), EXIT / CONTINUE, `SELECT .. INTO`, `SCALAR(..)`, `ECHO SELECT`, RAISE / SIGNAL, `EXCEPTION WHEN OTHERS`, CALL, nested calls. `CREATE FUNCTION`: single-`RETURN` functions are inlined into queries (so they run on the GPU). Functions with procedural bodies also run over columns: the body runs once per distinct argument value, via a dictionary map for strings or a value-range map for INT / DATE, so it works in WHERE, GROUP BY and aggregates. CALL returns every `ECHO` / bare `SELECT` as a separate result set. Session `@variables` |
| Security | `CREATE USER .. IDENTIFIED BY` (mysql_native_password), roles (`CREATE ROLE`, `GRANT role TO u`), `GRANT / REVOKE` SELECT, INSERT, UPDATE, DELETE, CREATE, DROP, ALTER, ALL on `*.*`, `db.*`, `db.table` (enforced per statement, incl. subqueries), SHOW GRANTS. Row-level security through views filtered on `CURRENT_USER()`. HTTP Data API with Basic auth. Server-side `LOAD DATA INFILE` and `SELECT .. INTO OUTFILE` need the FILE (SUPER) privilege (`LOAD DATA LOCAL` doesn't); information_schema shows only what the user may read. Listens on 127.0.0.1 by default (`--bind`), and refuses a non-loopback address while root has no password (`--root-password` / `WARPSTORE_ROOT_PASSWORD`). TLS on the MySQL protocol with a self-signed certificate generated in the data dir (or `--tls-cert` / `--tls-key`), `--require-secure-transport`; random handshake challenges (OpenSSL), failed-login throttling (10 per minute per address), `--max-connections` |
| Clustering | `--leaves http://leaf1:httpport,http://leaf2:httpport|http://leaf2-replica:httpport` makes a server an aggregator over leaf servers (SingleStore's aggregator / leaf split). DDL runs everywhere; REFERENCE tables are replicated to every leaf. INSERT rows and LOAD DATA lines go to leaf hash(shard key); UPDATE / DELETE are broadcast. Co-located queries (one sharded table, or sharded tables joined on their shard keys, plus reference tables) run on the leaves: aggregations as partial aggregates combined on the aggregator's GPU, other queries with ORDER BY / LIMIT merged there. Joins of sharded tables on other columns are made co-located stage by stage: a table is reshuffled leaf to leaf by its join key, or a co-located group is joined on the leaves and its result reshuffled by the next join key (filters pushed into the shuffles; `EXPLAIN` lists the stages). Joins where neither side is on its shard key reshuffle both sides; joins on expressions (`a.x = b.y + 1000`, `a.v * 2 = b.w + 1`, `LOWER(a.s) = b.t`, `ABS(a.n - 250) = b.m`, DATE columns, string = number) reshuffle the tables with the expression values as computed shard keys (integers hashed as integers; other numbers normalized to DOUBLE and strings / dates hashed as text on both sides), also when a side is a co-located group joined on the leaves first. Other queries fetch each sharded table's needed columns (WHERE conjuncts pushed down) and finish on the aggregator. Results and shuffled rows travel in a columnar binary format (typed int64 / float64 arrays, NULL flags, strings), appended to the receiving tables without per-cell parsing, over a dedicated data plane: each server listens on `--data-port` (default HTTP port + 10000; `-1` disables it) for length-prefixed binary frames on persistent, authenticated TCP connections, and leaves without one are called over HTTP. A leaf runs its part of a shuffle one partition at a time (the scans of partitioned tables restricted to that partition; co-located joins stay correct partition by partition) and sends each partition's rows in 256K-row chunks, so it holds about 1/partitions of its shuffle output at once. 4M-row joins that reshuffle one or both sides by the join key run in 0.3 s on 2 local leaves. `ADD LEAF 'url'` / `REBALANCE PARTITIONS` move rows to the new hash layout with binary bulk inserts; a leaf with a replica fails over to it (`PROMOTE`) when the primary is down, found by queries or by health checks every 2 s. `SHOW LEAVES`; `EXPLAIN` shows the leaf SQL |
| Replication / HA | `--replica-of http://user:pw@primary:httpport`: a replica bootstraps from a snapshot of the primary, then tails its WAL over HTTP (incl. the files of LOAD DATA / pipelines); it is read-only, `SHOW REPLICATION STATUS` (lag, the primary's replicas), `PROMOTE` for failover. `--failover-timeout S` on a replica promotes it by itself when the primary has been unreachable for S seconds (2 s in the tests); `--sync-replicas N` on the primary makes a commit wait until N replicas have applied it (semi-sync; falls back to async after `--sync-replica-timeout-ms`, default 2000), so a failover loses no acknowledged commit. A primary restart is detected (incarnation epoch) and replicas re-bootstrap |
| Backups | `BACKUP DATABASE d TO '/dir'` or `TO S3 'bucket/prefix' CONFIG .. CREDENTIALS ..` (streamed, multipart above 64 MB): a consistent snapshot plus a manifest with its WAL position. `RESTORE DATABASE d [AS other] FROM '/dir' \| S3 .. [UNTIL TIMESTAMP '2026-10-17 12:00:00']`: with `--wal-archive DIR` rotated WALs are kept (they carry a time marker every second), so a restore can replay up to any point in time. Server-side `LOAD DATA INFILE` loads a copy kept in the data dir, so deleting the source file doesn't lose data on a crash |
| Observability | `GET /metrics` on the HTTP port (Prometheus: statements by kind, latency histogram, errors, connections, rows, tables / segments / write buffers, VRAM, WAL size / fsyncs / fsync time, replication role / lag / replicas, plan cache, RSS). `SHOW [FULL] PROCESSLIST` / `information_schema.PROCESSLIST` (real sessions), `KILL [QUERY \| CONNECTION] id` (queries stop between execution stages and during loads), `information_schema.QUERY_STATS` (count / errors / total / avg / max ms / rows per statement shape, literals replaced; `FLUSH QUERY_STATS`), slow query log in MySQL's format (`<data dir>/slow.log`, over 1 s by default; `--slow-query-ms`, `SET GLOBAL long_query_time / slow_query_log`), `SELECT SLEEP(n)` |
| Metadata | information_schema with MySQL 8's column layouts (TABLES, COLUMNS, STATISTICS, KEY_COLUMN_USAGE, TABLE_CONSTRAINTS, REFERENTIAL_CONSTRAINTS, ROUTINES, ...; upper-case column labels like MySQL), system variables with MySQL defaults (`@@sql_mode`, `@@lower_case_table_names`, ..), `mysql` / `sys` / `performance_schema` schemas, SHOW CREATE TABLE / INDEX / FULL TABLES / VARIABLES / TABLE STATUS / PIPELINES, DESCRIBE |
| Storage | Tables larger than VRAM: segments beyond `--vram-limit` (or when VRAM is full) stay in pinned host RAM and are read over PCIe. `SNAPSHOT` checkpoints and `BACKUP DATABASE .. TO` / `RESTORE` files are columnar (dictionaries + physical column bytes per partition) and zstd-compressed: ClickBench 10M rows snapshot in 10 s to 0.99 GB and restart in 25 s (the earlier row format: 62 s, 1.9 GB, 99 s restart). Older row-format snapshots still load. After loads and restarts, large string dictionaries are sorted in the background (MIN/MAX/ORDER BY on strings), and dictionary-derived caches (sort order, LIKE bitmaps, string-function maps) are extended incrementally when inserts add strings instead of being rebuilt. VRAM cache: when columns spill to the host tier, scans count reads per segment column and a background thread (every 250 ms) keeps the hottest columns in VRAM, moving colder ones out only for a column 1.5x hotter (copies outside the table lock, pointer swaps under it; `WARPSTORE_NO_VRAM_CACHE=1` keeps the static first-come placement). GPU compression: integer segments are frame-of-reference packed to 8/16/32 bits, low-cardinality DOUBLE segments dictionary-coded, and decimal-valued DOUBLEs (prices like 406.1) stored as 16/32-bit integers / 10^s; on automatically once VRAM is 60% full (`--compress` always, `--no-compress` never). Generated kernels decode through out-of-line loads, so compile time stays flat |

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

## Design notes

- **FP64 is the trap on consumer GPUs.** The RTX 3080 runs FP64 at 1/64 of its FP32 rate. The first
  warp-reduction design spent about 2 ms per aggregate on FP64 adds. Replacing it with an
  integer-only warp bitonic sort plus a 5-step segmented scan made grouped aggregation about 3x
  faster. Typed codegen keeps integer columns in int64 arithmetic for the same reason, and
  fixed-point DECIMAL (int64) would be the next big win.
- **An interpreter is the bottleneck, not the GPU.** With the bytecode VM in every kernel the GPU
  ran at a few percent of its memory bandwidth (its stack lives in local memory). Compiling each
  query to typed CUDA with NVRTC made Q6 16x faster (16 → 1 ms) and is worth 1.5x on most queries.
  Compiling only the kernel a query needs and doing it in the background keeps the ~50-250 ms
  NVRTC cost off the first execution. That cost is real optimization work, not overhead (NVRTC's
  floor is ~22 ms, parsing the shared headers ~4 ms; lower ptxas levels save little and slow the
  kernels), so compiled plans persist on disk with a checksum: after the first time a query shape
  is seen, even a restarted server loads it in ~0.3 ms and uses it from the first execution.
- **Pick dense vs. hash GROUP BY by contention, not just cardinality.** Dense arrays made
  SingleStore's symbol query 2.8x faster, but Q1 (6 groups, FP64 sums) got 3x slower: every row
  hits the same few shared-memory addresses, and FP64 atomics are compare-and-swap loops on this
  GPU. Dense is used for 64+ groups or integer-only aggregates. Merging same-key lanes with
  `__match_any_sync` before the atomic was slower still.
- **Join filters beat faster joins.** Scanning lineitem with a bitmap of the order keys that
  survived the customer join keeps 302K of 60M rows (Q3 18 → 8.4 ms). A filter built from an
  unreduced table only adds work (Q12 got slower until filters required a 2x reduction).
- **Contention dominates low-cardinality GROUP BY.** Per-warp private shared-memory hash tables
  (no cross-warp atomics) beat a shared per-block table.
- **Don't let the host touch every group.** X1's GPU aggregation took 13 ms, but ranking its 1M
  groups on the CPU took 380 ms. HAVING, ORDER BY and LIMIT now compile to the same bytecode and
  run over the compacted groups in VRAM, so only the top-N rows cross PCIe (X1 417 → 15 ms,
  Q3 133 → 26 ms).
- **Benchmark against a fair baseline.** DuckDB is 2-3x faster on its own on-disk storage than on
  in-memory tables loaded with COPY; earlier numbers in this README used the slower setup.
- **Partitions help joins, up to a point.** Co-located, partition-wise joins on `o_orderkey =
  l_orderkey` took Q3 from 26 to 19 ms with 8 partitions. 32 partitions was slower overall: more
  kernel launches and smaller segments outweigh the smaller hash tables on one GPU.
- **Statistics matter for GPU hash tables.** Sizing them from dictionary sizes and segment min/max
  ranges, with a fast-abort on overflow, took a 200K-group query from 1.8 s to 64 ms.

## Layout

```
src/server.cpp     MySQL protocol, HTTP Data API, main
src/sql.*          lexer / parser / AST
src/engine.*       binder, bytecode compiler, planner, executor, DML, transactions, WAL, backups
src/storage.*      tables, write buffers, GPU segments, MVCC snapshots, flush, LOAD DATA
src/gpu.*          kernel launches, aggregation state, select, join, sort, gather
src/kernels.cuh    kernel bodies shared by the ahead-of-time build and runtime-compiled plans
src/jit.*          bytecode -> typed CUDA C++, NVRTC compilation, plan cache
src/devtypes.h     expression VM and plan structs shared by CPU, GPU and NVRTC
src/repl.cpp       WAL-tailing replicas, semi-sync, failover
src/cluster.cpp    aggregator / leaf clustering
src/monitor.*      sessions, KILL, query stats, slow log, Prometheus metrics
src/mysqlcdc.*     MySQL client + binlog decoder (change data capture)
src/sources.*      S3, HTTP, Kafka; parquet.cpp / orc.cpp / avro.cpp readers
src/cagra.*        cuVS GPU graph vector index (loaded at run time)
src/embed.*        BERT-style sentence embeddings on cuBLAS (EMBED())
src/proc.cpp       stored procedures, UDFs
tests/             run.py (the suite), lib.py, fuzz.py, ORM projects in tests/orm
bench/             TPC-H, ClickBench, use-case and HTAP benchmarks (DuckDB / SingleStore comparisons)
scripts/           install-deps.sh, test-containers.sh, install-cuvs.sh, fetch-embed-model.sh
```

## License

GNU Affero General Public License v3.0 (see [LICENSE](LICENSE)), with an additional permission for
linking the NVIDIA CUDA and cuVS libraries (see [NOTICE](NOTICE), which also lists third-party
components). Running a modified warpstore as a network service means offering its source to its users.
