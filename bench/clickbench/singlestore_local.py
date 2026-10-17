"""Runs ClickBench on a local SingleStore (the free singlestoredb-dev image) on this machine:
load the same 10M-row TSV gpu-db loads, then each query once (first) + best of 3 (hot).
usage: python bench/clickbench/singlestore_local.py bench/clickbench [load] > results_singlestore_local.jsonl
The container (8 cores is the license-free limit):
  docker run -d --name s2bench --cpus=8 --memory=16g -e ROOT_PASSWORD=bench -p 3316:3306 \
    -v $PWD/data/clickbench:/bench:ro ghcr.io/singlestore-labs/singlestoredb-dev:latest"""
import sys, time, json, pymysql

S = sys.argv[1]
qs = [q.strip() for q in open(f"{S}/queries.sql") if q.strip()]
c = pymysql.connect(host="127.0.0.1", port=3316, user="root", password="bench", autocommit=True,
                    read_timeout=900, local_infile=True).cursor()
if "load" in sys.argv[2:]:
    # utf8mb4: the data has 1099 rows with 4-byte characters, which create.sql's utf8_bin rejects.
    for v in ["character_set_server = 'utf8mb4'", "collation_server = 'utf8mb4_bin'", "collation_database = 'utf8mb4_bin'"]:
        c.execute("set global " + v)
    c.execute("drop database if exists bench")
    c.execute("create database bench")  # the free dev image allows at most 2 partitions
    c.execute("use bench")
    c.execute([x for x in open(f"{S}/create.sql").read().split(";") if "CREATE TABLE" in x][0])
    cols = open(f"{S}/cols.txt").read().strip()
    t = time.time()
    # SingleStore rejects 2 of the 10M rows ("Data invalid for column 'URL'"); they are skipped.
    c.execute(f"load data infile '/bench/hits.tsv' skip all errors into table hits fields terminated by '\\t' ({cols})")
    c.execute("optimize table hits full")
    print(json.dumps({"load_s": round(time.time() - t, 1)}), file=sys.stderr)
c.execute("use bench")
c.execute("set names utf8mb4 collate utf8mb4_bin")
c.execute("set global regexp_format = 'advanced'")  # PCRE: Q29 uses (?:..)
only = [int(x) for x in sys.argv[2:] if x.isdigit()] or range(1, len(qs) + 1)
for i in only:
    r = {"q": i}
    try:
        t = time.perf_counter(); c.execute(qs[i - 1]); c.fetchall(); r["s2_first"] = time.perf_counter() - t
        ts = []
        for _ in range(3):
            t = time.perf_counter(); c.execute(qs[i - 1]); c.fetchall(); ts.append(time.perf_counter() - t)
        r["s2"] = min(ts)
    except Exception as e:
        r["s2_err"] = str(e)[:160]
    print(json.dumps(r), flush=True)
