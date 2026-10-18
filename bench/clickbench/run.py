import pymysql, duckdb, time, json, sys, math, datetime, decimal
S = sys.argv[1]
gq = [q.strip() for q in open(f"{S}/queries.sql") if q.strip()]
dq = [q.strip() for q in open(f"{S}/duckdb_queries.sql") if q.strip()]
only = [int(x) for x in sys.argv[2:]] if len(sys.argv) > 2 else range(1, 44)
def norm(rows):
    out = []
    for r in rows:
        t = []
        for v in r:
            if isinstance(v, (float, decimal.Decimal)): v = float(v); t.append(round(v, 2) if abs(v) < 1e6 else float(f"{v:.6g}"))
            elif isinstance(v, (datetime.date, datetime.datetime)): t.append(str(v))
            elif isinstance(v, bytes): t.append(v.decode("utf-8", "replace"))
            else: t.append(v)
        out.append(tuple(t))
    return sorted(out, key=str)
res = {}
g = pymysql.connect(host="127.0.0.1", port=3307, user="root", autocommit=True, read_timeout=90)
gc = g.cursor()
gc.execute("SET collation_connection = utf8mb4_bin"); gc.execute("SET warpstore_codegen = SYNC")
d = duckdb.connect("data/clickbench/hits10m.duckdb", read_only=True)
d.execute("SET threads=12"); d.execute("SET memory_limit='4GB'")
for i in only:
    q, qd = gq[i-1], dq[i-1]
    r = {"q": i}
    try:
        t0 = time.perf_counter(); gc.execute(q); grow = gc.fetchall(); r["warpstore_first"] = time.perf_counter() - t0
        ts = []
        for _ in range(3):
            t = time.perf_counter(); gc.execute(q); gc.fetchall(); ts.append(time.perf_counter() - t)
        r["warpstore"] = min(ts)
    except Exception as e:
        r["warpstore_err"] = str(e)[:120]; grow = None
        if "Lost connection" in str(e): print(json.dumps(r)); sys.exit(2)
    try:
        drow = d.execute(qd).fetchall()
        ts = []
        for _ in range(3):
            t = time.perf_counter(); d.execute(qd).fetchall(); ts.append(time.perf_counter() - t)
        r["duckdb"] = min(ts)
    except Exception as e:
        r["duckdb_err"] = str(e)[:120]; drow = None
    if grow is not None and drow is not None:
        r["match"] = norm(grow) == norm(drow)
        if not r["match"]: r["g"] = str(norm(grow)[:2])[:160]; r["d"] = str(norm(drow)[:2])[:160]
    print(json.dumps(r), flush=True)
