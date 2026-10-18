# DuckDB first run (fresh process, after OS cache warm) and best of 3, per ClickBench query
import duckdb, time, json
qs=[q.strip() for q in open("bench/clickbench/duckdb_queries.sql") if q.strip()]
d=duckdb.connect("data/clickbench/hits10m.duckdb", read_only=True)
for i,q in enumerate(qs,1):
    t=time.perf_counter(); d.execute(q).fetchall(); first=time.perf_counter()-t
    best=min((lambda: (lambda t0: (d.execute(q).fetchall(), time.perf_counter()-t0)[1])(time.perf_counter()))() for _ in range(3))
    print(json.dumps({"q":i,"first":first,"hot":best}))
