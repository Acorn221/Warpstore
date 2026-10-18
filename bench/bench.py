"""Load TPC-H into warpstore (over the MySQL protocol) and DuckDB, check results match, compare latency.

usage: python bench/bench.py --sf 1 [--port 3307] [--runs 5] [--skip-load]
"""
import argparse
import datetime
import decimal
import os
import statistics
import time

import duckdb
import pymysql

from tpch import QUERIES, TABLES, export


def norm(v):
    if isinstance(v, decimal.Decimal):
        return float(v)
    if isinstance(v, datetime.datetime):
        return v.date()
    return v


def same(a, b):
    if len(a) != len(b):
        return False, f"row count {len(a)} vs {len(b)}"
    for i, (ra, rb) in enumerate(zip(a, b)):
        for x, y in zip(ra, rb):
            x, y = norm(x), norm(y)
            if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                if abs(x - y) > 1e-6 * max(1.0, abs(x), abs(y)):
                    return False, f"row {i}: {ra} vs {rb}"
            elif x != y:
                return False, f"row {i}: {ra} vs {rb}"
    return True, ""


def timed(fn, runs):
    fn()  # warm-up
    ts = []
    for _ in range(runs):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1000)
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sf", type=float, default=1)
    ap.add_argument("--port", type=int, default=3307)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--skip-load", action="store_true")
    ap.add_argument("--data", default=None)
    args = ap.parse_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data = os.path.abspath(args.data or os.path.join(root, "data", f"tpch-sf{args.sf:g}"))
    if not os.path.exists(os.path.join(data, "lineitem.tbl")):
        if args.sf > 1:
            # In-memory dbgen needs ~2.3 GB of RAM per scale factor; generate big ones in chunks.
            raise SystemExit(f"{data} missing: generate it with bench/regen_lineitem.py-style chunked dbgen first")
        export(args.sf, data)

    g = pymysql.connect(host="127.0.0.1", port=args.port, user="root", password="", autocommit=True)
    gc = g.cursor()
    if not args.skip_load:
        for t, (ddl, _) in TABLES.items():
            gc.execute(f"DROP TABLE IF EXISTS {t}")
            gc.execute(f"CREATE TABLE {t} ({ddl})")
            t0 = time.perf_counter()
            n = gc.execute(f"LOAD DATA INFILE '{data}/{t}.tbl' INTO TABLE {t} FIELDS TERMINATED BY '|'")
            gc.execute(f"OPTIMIZE TABLE {t}")
            gc.fetchall()
            print(f"warpstore loaded {t}: {n:,} rows in {time.perf_counter() - t0:.1f}s")

    # On-disk DuckDB with a hard memory cap: an in-memory SF10 database grew past 20 GB of RSS and
    # triggered the OOM killer on a shared machine. The file is built once and reused.
    dbfile = os.path.join(data, "duckdb.db")
    if not os.path.exists(dbfile):
        build = dbfile + ".building"
        for f in (build, build + ".wal"):
            if os.path.exists(f):
                os.remove(f)
        b = duckdb.connect(build)
        b.execute("SET memory_limit='4GB'")
        b.execute(f"SET temp_directory='{data}/duckdb.tmp'")
        for t, (ddl, cols) in TABLES.items():
            # Same column types as warpstore (DOUBLE, not DECIMAL); drop warpstore-only key clauses.
            coldefs = [c.strip() for c in ddl.split(",") if not c.strip().upper().startswith(("PRIMARY", "SORT", "SHARD", "KEY"))]
            coldefs = [c for c in coldefs if not c.rstrip(")").strip().isidentifier() or " " in c]
            b.execute(f"CREATE TABLE {t} ({', '.join(coldefs)})")
            b.execute(f"COPY {t} FROM '{data}/{t}.tbl' (DELIMITER '|', HEADER false)")
            b.execute("CHECKPOINT")
        b.close()
        os.rename(build, dbfile)  # only a complete database is ever reused
    d = duckdb.connect(dbfile, read_only=True)
    d.execute("SET memory_limit='4GB'")
    print(f"DuckDB {duckdb.__version__} threads={d.execute('select current_setting(%s)' % repr('threads')).fetchone()[0]}")

    # Steady state: compile each query's plan on its first (warm-up) run instead of in the background,
    # as a server that has seen the query before would (compiled plans also persist on disk).
    gc.execute("SET warpstore_codegen = SYNC")
    print(f"\n{'query':42} {'warpstore ms':>10} {'duckdb ms':>10} {'speedup':>8}  result")
    rows = []
    for name, sql in QUERIES.items():
        def run_g():
            gc.execute(sql)
            return gc.fetchall()

        def run_d():
            return d.execute(sql).fetchall()

        ok, why = same(run_g(), run_d())
        tg = timed(run_g, args.runs)
        td = timed(run_d, args.runs)
        rows.append((name, tg, td))
        print(f"{name:42} {tg:10.2f} {td:10.2f} {td / tg:7.1f}x  {'MATCH' if ok else 'MISMATCH ' + why}")

    # OLTP-style point lookups through the same server.
    gc.execute("select max(o_orderkey) from orders")
    maxk = gc.fetchone()[0]
    import random
    keys = [random.randint(1, maxk) for _ in range(2000)]
    t = time.perf_counter()
    for k in keys:
        gc.execute(f"select o_custkey, o_totalprice, o_orderdate from orders where o_orderkey = {k}")
        gc.fetchall()
    tg = (time.perf_counter() - t) * 1000 / len(keys)
    t = time.perf_counter()
    for k in keys[:200]:
        d.execute(f"select o_custkey, o_totalprice, o_orderdate from orders where o_orderkey = {k}").fetchall()
    td = (time.perf_counter() - t) * 1000 / 200
    print(f"{'point lookup by PK (avg, incl. network)':42} {tg:10.3f} {td:10.3f} {td / tg:7.1f}x")


if __name__ == "__main__":
    main()
