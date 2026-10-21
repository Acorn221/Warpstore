# Benchmarks

All measurements on one machine unless noted: Ryzen 5 5500 (6 cores / 12 threads), 54 GB RAM, RTX 3080 10 GB.

## How it compares

Measured 2026-10-18 unless noted. Same machine for the first two tables: Ryzen 5 5500 (6 cores / 12
threads), 54 GB RAM, RTX 3080 10 GB. All the data and scripts are in [bench/comparisons](../bench/comparisons)
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

### When the data doesn't fit in RAM

ClickBench 10M rows with `--memory-limit` below the table's size (3.8 GB of column data, 0.78 GB of
compressed segment files): the
column data that doesn't fit is evicted to the segment files, read back when scanned and pushed through
the GPU in windows (`bench/clickbench/tier.py`, results in
[bench/comparisons](../bench/comparisons/clickbench_warpstore_tier_0.25gb.jsonl)). "Read from the SSD"
drops the files from the OS page cache before every query (`posix_fadvise(DONTNEED)`), so the reads
come from the NVMe drive. DuckDB and ClickHouse are compared with their hot, in-memory times.

| Warpstore setup | All 43 queries | Geomean | vs DuckDB hot (geomean) | vs ClickHouse hot (geomean) |
|---|---:|---:|---:|---:|
| everything in memory | 0.26 s | 2.5 ms | 13.4x faster | 23.9x faster |
| `--memory-limit 2`, files in the OS page cache | 0.29 s | 2.9 ms | 11.9x faster | 21.1x faster |
| `--memory-limit 2`, read from the SSD | 0.29 s | 2.8 ms | 12.1x faster | 21.4x faster |
| `--memory-limit 0.25`, files in the OS page cache | 0.64 s | 7.6 ms | 4.5x faster | 7.9x faster |
| `--memory-limit 0.25`, read from the SSD | 0.72 s | 7.4 ms | 4.6x faster | 8.1x faster |

How the disk tier keeps this fast:

- **Only the rows a scan needs.** Segments are sorted by the SORT KEY, so `WHERE CounterID = 62`
  narrows each segment to a small row range; only that range is read from the file (Q37-Q43: about
  110-200 ms each when whole columns were read, about 20 ms now).
- **Read-ahead on several threads.** While a scan works on one segment, the column it is reading is
  already being read for the next segments (I/O threads, several requests in flight on the SSD, into
  pooled pinned buffers that go back to the pool as soon as their copy to the GPU has run).
- **Compressed segment files.** Each column is stored in blocks of 16K rows, zstd-compressed (on top of
  the GPU encodings), with a block index, so a range read only reads and decompresses the blocks it
  covers, on the I/O threads: 3.8 GB of column data take 0.78 GB on disk, and the SSD run went from
  1.07 s to 0.77 s; it now takes as long as with the files in the page cache. The SORT KEY column is
  stored uncompressed (binary searches read single rows of it).
- **Late materialization for top-N.** `SELECT * .. ORDER BY .. LIMIT 10` scans only the filter and
  ORDER BY columns (with each row's location), sorts, and reads the other columns for the 10 rows left
  (Q24: 2.9 s reading all 105 columns, 45 ms now; the cells are read grouped by block, each block
  decompressed once).

- **Decompression on the GPU.** With nvCOMP installed (`scripts/install-nvcomp.sh`), the compressed
  blocks go to the GPU as they are (4-5x less over PCIe) and are decompressed there, all the columns of
  a batch in one nvCOMP call (each call costs ~0.4 ms however much it does, so small reads, like a
  narrow SORT KEY range, stay on the CPU). Here: 0.72 s from the SSD, 0.77 s with the CPU doing it
  (`WARPSTORE_GPU_DECOMPRESS=0`; it already decompresses on all cores, so the gain is modest).
  GPUDirect Storage (`WARPSTORE_GPUDIRECT`, via cuFile) reads the compressed blocks straight into VRAM
  when the nvidia-fs driver is loaded; this machine (a GeForce card) doesn't have it, so it is untested
  for speed here: in cuFile's compatibility mode the results are the same.

Before these, the 0.25 GB / SSD run took 5.2 s.

GPU compression formats on ClickBench columns (13 columns, 1M rows each, sorted like the table, in
chunks of 16K rows; RTX 3080, nvCOMP 5.3). zstd compresses the best and still decompresses at 10+ GB/s
on the GPU, and files stay readable on the CPU (and without nvCOMP), so segment files use it:

| Format | Ratio | GPU decompression, 806 chunks | One call |
|---|---:|---:|---:|
| zstd | 4.33 | 10-11.5 GB/s | 0.35-0.4 ms |
| Cascaded (RLE + delta + bit-packing) | 3.58 | (failed in this test) | |
| GDeflate | 3.34 | 13.3 GB/s | 3.2 ms |
| Bitcomp | 2.81 | (failed in this test) | 0.23 ms |
| LZ4 | 2.70 | 9.5 GB/s | 0.58 ms |
| frame of reference + bit-packing per chunk | 1.91 | | |
| ANS | 1.69 | 239 GB/s | 0.02 ms |

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
