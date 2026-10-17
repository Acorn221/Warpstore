"""Time a query against gpu-db and show where the time goes (PROFILE / SHOW PROFILE).

usage:
  python bench/qtime.py "SELECT ..."                 # timings + stage breakdown
  python bench/qtime.py -f query.sql --runs 20
  python bench/qtime.py "SELECT ..." --compare       # A/B the optimizations on the same query
  python bench/qtime.py --plancache                  # compiled plans, executions, average times

Every option is also plain SQL you can type in the mysql client:
  PROFILE SELECT ...;  SHOW PROFILE;  EXPLAIN SELECT ...;  SHOW PLANCACHE;  SHOW PLAN <id>;
  SET gpudb_codegen = ON|SYNC|OFF;  SET gpudb_dense_groupby = ON|OFF;  SET gpudb_join_filters = ON|OFF;
"""
import argparse
import statistics
import sys
import time

import pymysql

VARIANTS = [
    ("all optimizations", []),
    ("codegen off (interpreter)", ["SET gpudb_codegen = OFF"]),
    ("dense GROUP BY off", ["SET gpudb_dense_groupby = OFF"]),
    ("join filters off", ["SET gpudb_join_filters = OFF"]),
]
RESET = ["SET gpudb_codegen = SYNC", "SET gpudb_dense_groupby = ON", "SET gpudb_join_filters = ON"]


def timed(cur, sql, runs):
    cur.execute(sql)  # warm-up (and compile)
    cur.fetchall()
    ts = []
    for _ in range(runs):
        t = time.perf_counter()
        cur.execute(sql)
        cur.fetchall()
        ts.append((time.perf_counter() - t) * 1000)
    ts.sort()
    return ts


def fmt(ts):
    p95 = ts[min(len(ts) - 1, int(round(0.95 * (len(ts) - 1))))]
    return f"min {ts[0]:8.2f} ms   median {statistics.median(ts):8.2f} ms   p95 {p95:8.2f} ms"


def table(rows, headers):
    rows = [["" if v is None else str(v) for v in r] for r in rows]
    w = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h) for i, h in enumerate(headers)]
    print("  " + "  ".join(h.ljust(w[i]) for i, h in enumerate(headers)))
    for r in rows:
        print("  " + "  ".join(v.ljust(w[i]) for i, v in enumerate(r)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sql", nargs="?")
    ap.add_argument("-f", "--file")
    ap.add_argument("--port", type=int, default=3307)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--compare", action="store_true", help="time the query with each optimization switched off")
    ap.add_argument("--plancache", action="store_true", help="show the compiled plan cache and exit")
    a = ap.parse_args()
    cur = pymysql.connect(host=a.host, port=a.port, user="root", password="", autocommit=True).cursor()
    if a.plancache:
        cur.execute("SHOW PLANCACHE")
        table([r[:7] + (r[7][:60],) for r in cur.fetchall()],
              ["plan", "kind", "compile_ms", "execs", "avg_ms", "last_ms", "src_bytes", "query"])
        return
    sql = open(a.file).read().strip().rstrip(";") if a.file else a.sql
    if not sql:
        ap.error("give a query or -f FILE")
    for s in RESET:
        cur.execute(s)

    ts = timed(cur, sql, a.runs)
    print(f"{a.runs} runs: {fmt(ts)}  (end to end, including the MySQL protocol)")
    cur.execute("PROFILE " + sql)
    cur.fetchall()
    cur.execute("SHOW PROFILE")
    print("\nPROFILE (GPU work is synchronized per stage, so stages add up to slightly more than a normal run):")
    table(cur.fetchall(), ["stage", "time_ms", "rows", "detail"])
    cur.execute("EXPLAIN " + sql)
    print("\nEXPLAIN:")
    for (line,) in cur.fetchall():
        print("  " + line)

    if a.compare:
        print("\nA/B (median of %d runs):" % a.runs)
        base = None
        for name, sets in VARIANTS:
            for s in RESET + sets:
                cur.execute(s)
            med = statistics.median(timed(cur, sql, a.runs))
            base = base or med
            print(f"  {name:28} {med:8.2f} ms   {med / base:5.2f}x")
        for s in RESET:
            cur.execute(s)


if __name__ == "__main__":
    sys.exit(main())
