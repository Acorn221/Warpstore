"""ClickBench (10M rows) with the table larger than --memory-limit: each query once in memory (hot,
best of 3), then with the column data evicted to the segment files and read back while scanning,
once with the files in the OS page cache and once with them dropped from it before every query
(posix_fadvise DONTNEED, so the reads come from the SSD).

usage: python bench/clickbench/tier.py DATA_DIR [LIMIT_GB] > bench/comparisons/clickbench_warpstore_tier.jsonl
DATA_DIR: a warpstore data directory with the hits table (it is snapshotted: the segment files are written).
"""
import json, os, subprocess, sys, time
import pymysql

S = "bench/clickbench"
data = sys.argv[1]
limit = sys.argv[2] if len(sys.argv) > 2 else "2"
queries = [q.strip() for q in open(f"{S}/queries.sql") if q.strip()]
BIN = os.environ.get("WARPSTORE_BIN", "build/warpstore")


def start(mem):
    log = open(os.path.join(data, "tier-bench.log"), "a")
    p = subprocess.Popen([BIN, "--data-dir", data, "--port", "3399", "--memory-limit", mem], stdout=log, stderr=log)
    for _ in range(600):
        try:
            c = pymysql.connect(host="127.0.0.1", port=3399, user="root", autocommit=True, read_timeout=600,
                                database=os.environ.get("WARPSTORE_TIER_DB"))  # (the database with hits, if not the default)
            return p, c.cursor()
        except pymysql.err.OperationalError:
            time.sleep(0.5)
    raise SystemExit("server did not start")


def stop(p, cur):
    p.terminate()
    p.wait(300)


def drop_cache():
    d = os.path.join(data, "segments")
    for f in os.listdir(d):
        fd = os.open(os.path.join(d, f), os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)


def run(cur, q, before=None):
    if before:
        before()
    t = time.perf_counter()
    cur.execute(q)
    cur.fetchall()
    return time.perf_counter() - t


res = [{"q": i + 1} for i in range(len(queries))]
# 1) everything in memory; SNAPSHOT writes the segment files (and refers to them).
# WARPSTORE_TIER_LIMITED_ONLY=1: skip this (the data dir already has its segment files).
if not os.environ.get("WARPSTORE_TIER_LIMITED_ONLY"):
    p, cur = start("0")
    for setup in ("SET collation_connection = utf8mb4_bin", "SET warpstore_codegen = SYNC"):
        cur.execute(setup)
    cur.execute("SNAPSHOT")
    for i, q in enumerate(queries):
        run(cur, q)  # compile + warm
        res[i]["memory"] = min(run(cur, q) for _ in range(3))
    stop(p, cur)
# 2) under the limit: the table starts on disk after the restart
p, cur = start(limit)
for setup in ("SET collation_connection = utf8mb4_bin", "SET warpstore_codegen = SYNC"):
    cur.execute(setup)
time.sleep(5)  # (the cache fills up to its limit)
for i, q in enumerate(queries):
    run(cur, q)  # compile
    res[i]["limited_pagecache"] = min(run(cur, q) for _ in range(3))
for i, q in enumerate(queries):
    res[i]["limited_ssd"] = min(run(cur, q, drop_cache) for _ in range(2))
stop(p, cur)
for r in res:
    print(json.dumps(r))
