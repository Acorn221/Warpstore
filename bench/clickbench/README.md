# ClickBench comparison (gpu-db vs DuckDB vs SingleStore's published result)

ClickBench (https://github.com/ClickHouse/ClickBench) runs 43 queries on a 100M-row web-analytics
table. SingleStore submitted a result in 2022 on an AWS c6a.4xlarge (16 vCPU, "tuned");
DuckDB has one from 2026 on the same machine type. Both are copied here
(`singlestore_c6a.4xlarge.json`, `duckdb_c6a.4xlarge.json`).

The full 100M rows don't fit this machine (10 GB VRAM, ~22 GB free RAM), so we run the first 10M
rows (10 of the 100 official parquet partitions) on gpu-db and on DuckDB, on the same machine, and
bridge to SingleStore through DuckDB:

    est. speedup vs SingleStore = (DuckDB_local / gpu-db_local) * (SingleStore_c6a / DuckDB_c6a)

`queries.sql` is SingleStore's query file, with one change: Q29 uses `'\\1'` (MySQL escaping) for
the regex backreference. `results_10m.jsonl` is the last run (hot = best of 3 after one warm-up
run; `gpudb_first` = the warm-up run, including plan compilation and dictionary caches).

Steps (`load.py`, then `run.py bench/clickbench [query numbers]`; `prof.py` shows per-stage profiles): download `hits_{0..9}.parquet` from
https://datasets.clickhouse.com/hits_compatible/athena_partitioned/, convert with DuckDB
(`binary_as_string=true`, epoch columns to timestamps), export TSV, `LOAD DATA` with the column
list in `cols.txt`, then run each query once + 3 times and compare results with DuckDB.

## Direct run against SingleStore on the same machine

`singlestore_local.py` runs the same 43 queries on the same 10M rows in SingleStore 9.1.1, using
the free `ghcr.io/singlestore-labs/singlestoredb-dev` image (no licence needed up to 8 cores; this host exposes 12 hardware threads).
Setup: `--cpus=8 --memory=16g`, utf8mb4_bin (1099 rows have 4-byte characters), `OPTIMIZE TABLE
hits FULL`, PCRE regexes for Q29. The dev image allows at most 2 partitions per database, but
flexible parallelism (`query_parallelism_per_leaf_core = 1`, 8 sub-partitions) kept 7-8 cores busy
during queries. Results are in `results_singlestore_local.jsonl`; gpu-db's are in `results_10m.jsonl`.

| (Ryzen 5 5500: 6 cores / 12 threads, RTX 3080) | sum of 43 hot runs | sum of first runs | gpu-db speedup (geomean) |
|---|---|---|---|
| gpu-db | 0.27 s | 2.6 s | — |
| DuckDB (12 threads) | 6.4 s | — | 15.3x hot |
| SingleStore 9.1 dev image (8 threads) | 30.1 s | 49.6 s | **95x hot, 94x first run** |

On these queries the dev image is 6.2x slower than DuckDB, while SingleStore's own tuned 2022
c6a.4xlarge submission is 2.6x slower. The container setup (8 of 12 hardware threads, 2 partitions, docker)
therefore probably understates SingleStore by about 2.4x. Bridging through the published result
gives a more conservative ~39x hot. gpu-db is faster than SingleStore on all 43 queries hot. The
closest are Q7 (MIN/MAX of a date, 2.9x) and Q1 (COUNT(*), 5.4x). On first runs it is slower only
on Q1 (22 ms vs 16 ms).
