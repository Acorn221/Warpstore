# Architecture

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
scripts/           install-deps.sh, test-containers.sh, install-cuvs.sh, install-nvcomp.sh, fetch-embed-model.sh
```
