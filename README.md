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
caveats: [docs/benchmarks.md](docs/benchmarks.md).</sub>

> **Status: experimental alpha.** Warpstore was developed as an experiment: how far can a GPU-first,
> MySQL-compatible database get, and how fast? It was built quickly, with heavy use of an AI coding
> assistant. It passes TPC-H, ClickBench, a differential fuzzer and ORM suites (see [Tests](docs/development.md#tests)),
> but it is young: NVIDIA GPUs only, one main developer, no production deployments. Don't make it
> the only home of data you care about; keep a system of record (for instance with warpstore as a live
> copy of MySQL, see [MySQL change data capture](docs/features.md#mysql-change-data-capture-warpstore-as-a-live-copy-of-mysql)).

## Trade-offs: should you use it?

Warpstore is fast because it keeps the data queries use in memory (hottest columns in GPU memory,
then host RAM, the rest on disk) and runs every analytic query as compiled GPU kernels. That shapes
what it is good and bad at.

**The hard limits**

- **It is fastest when the columns your queries touch fit in RAM.** Tables can be larger than RAM:
  column data beyond `--memory-limit` (default: half the RAM) lives in segment files and streams
  through the GPU in windows when scanned. With ClickBench's working set in memory nothing changes;
  with only 0.25 GB of RAM for a 3.8 GB table, reading from the SSD, the 43 queries take 1.1 s instead
  of 0.27 s (still 2.8x DuckDB with its data in memory, see [benchmarks](docs/benchmarks.md#when-the-data-doesnt-fit-in-ram)).
  Write buffers, primary-key and secondary indexes and string dictionaries still live in RAM, so
  billions of rows need a lot of it.
- **An NVIDIA GPU is required** (Turing or newer, CUDA 12). No AMD, Apple or CPU-only mode.
- **It is young.** One main developer, no production users yet, alpha quality. Clustering exists but
  is experimental; single-node is the tested setup.
- **Not every MySQL feature.** The wire protocol, common SQL, transactions and the popular ORMs work
  (see [features](docs/features.md)), but there are no triggers, foreign keys are not enforced, there is
  no SERIALIZABLE isolation, and stored procedures are a subset.

**Compared with the usual choices**

| | Pick Warpstore when | Pick the other when |
|---|---|---|
| **MySQL / MariaDB** | You need dashboards, reports or search over your data in milliseconds instead of seconds (46x to 2,000x faster here), or vectors / embeddings / full-text in SQL. Run it as a live copy of MySQL (change data capture) and keep MySQL as the system of record. | Data much larger than RAM with random access to it, decades of tooling and operational knowledge, strict durability by default, full MySQL compatibility, no GPU in the budget. Point lookups by key are fast in both. |
| **PostgreSQL** | The same analytics and search speed-ups, through the MySQL protocol and ORMs. | You rely on PostgreSQL's dialect or extensions (PostGIS, pgvector ecosystem, PL/pgSQL), SERIALIZABLE isolation, foreign keys, triggers. |
| **SingleStore** | Same model (row-store write buffer, columnar segments, MySQL protocol, SHARD / SORT keys, pipelines) but open source (AGPL), no licence fees, and 87x faster on one machine here. | You need a mature distributed system: high availability across nodes, managed cloud, bottomless storage on object stores for data larger than memory, enterprise support. |
| **ClickHouse** | You also need transactions, UPDATE / DELETE as ordinary statements, the MySQL protocol for applications and ORMs, and lower latency (24x faster here at 10M rows). | Very large data (billions of rows on disk), append-mostly logs and events, mature clustering, CPU-only servers. |
| **DuckDB** | A server that many clients and applications share, with transactions, replication and ingest pipelines (13x faster here). | An embedded, in-process engine for notebooks and data science, files (Parquet, CSV) larger than memory, no server and no GPU. |

**Durability**, for comparison: commits are acknowledged before the WAL reaches disk by default (as in
SingleStore; a power cut can lose the last ~10 ms, a crash of the process loses nothing). Use
`CREATE DATABASE .. WITH SYNC DURABILITY` or `--sync-durability` for MySQL-style durable commits.

## Features

- **MySQL compatible**: wire protocol, TLS, users / roles / GRANT, prepared statements, transactions
  (READ COMMITTED and snapshot isolation, MVCC: reads never block writes), information_schema;
  Drizzle, Prisma, SQLAlchemy and the usual drivers work.
- **Analytics on the GPU**: joins, GROUP BY, window functions, CTEs, subqueries, compiled per query
  (NVRTC) and cached; all 22 TPC-H queries; columnar compression; tables larger than VRAM spill to host RAM, larger than RAM to disk.
- **Search and AI in SQL**: vector search with a GPU graph index (cuVS CAGRA), `EMBED()` sentence
  embeddings computed on the GPU, BM25 full-text search, hybrid queries.
- **Getting data in**: `LOAD DATA` (CSV, JSON, Parquet, ORC, Avro), continuous pipelines from S3 and
  Kafka (exactly once), and a live copy of a MySQL database (binlog change data capture).
- **Operations**: WAL with crash recovery, backups to disk or S3 with point-in-time restore, replicas
  with semi-sync commits and automatic failover, Prometheus metrics, slow query log, `KILL`.

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

SingleStore is a trademark of SingleStore, Inc. Warpstore is an independent project, not affiliated
with or endorsed by SingleStore; it supports much of SingleStore's SQL dialect (SHARD KEY, SORT KEY,
REFERENCE tables, pipelines, vector and JSON syntax) and is benchmarked against it below.

## Documentation

| | |
|---|---|
| [docs/benchmarks.md](docs/benchmarks.md) | Every comparison, per query, with methods and caveats |
| [docs/features.md](docs/features.md) | SQL coverage, what's missing, ORMs, MySQL CDC, vector search, VRAM hints, server options |
| [docs/architecture.md](docs/architecture.md) | How it works, design notes, source layout |
| [docs/development.md](docs/development.md) | Running the tests, measuring query speed |
| [CONTRIBUTING.md](CONTRIBUTING.md) / [SECURITY.md](SECURITY.md) | Contributing, reporting vulnerabilities |

## License

GNU Affero General Public License v3.0 ([LICENSE](LICENSE)), with an additional permission for linking
NVIDIA's CUDA and cuVS libraries ([NOTICE](NOTICE)). Running a modified Warpstore as a network service
means offering its source to its users.
