# TPC-H: all 22 queries against DuckDB on generated data (SF 0.1 by default, GPUDB_TPCH_SF to change).
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import duckdb
from lib import TMP, Checker, norm_rows, rows_close, server_from_env

SF = float(os.environ.get("GPUDB_TPCH_SF", "0.1"))
d = duckdb.connect()
d.execute("INSTALL tpch; LOAD tpch")
d.execute(f"CALL dbgen(sf={SF})")
srv = server_from_env()
cur = srv.cursor()
cur.execute("DROP DATABASE IF EXISTS tpch")
cur.execute("CREATE DATABASE tpch")
cur.execute("USE tpch")
keys = {"lineitem": "l_orderkey", "orders": "o_orderkey", "customer": "c_custkey", "part": "p_partkey", "partsupp": "ps_partkey", "supplier": "s_suppkey"}
for (t,) in d.execute("SELECT table_name FROM information_schema.tables ORDER BY 1").fetchall():
    cols = d.execute(f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name='{t}' ORDER BY ordinal_position").fetchall()
    defs = [f"{n} {'VARCHAR(255)' if ty.upper().startswith('VARCHAR') else ty.upper().replace('INTEGER', 'INT')}" for n, ty in cols]
    k = keys.get(t)
    cur.execute(f"CREATE {'REFERENCE ' if not k else ''}TABLE {t} ({', '.join(defs)}{', SHARD KEY(' + k + ')' if k else ''})")
    path = os.path.join(TMP, f"tpch-{t}.tsv")
    d.execute(f"COPY {t} TO '{path}' (DELIMITER '\t', HEADER false)")
    cur.execute(f"LOAD DATA INFILE '{path}' INTO TABLE {t} FIELDS TERMINATED BY '\\t'")
    os.unlink(path)
c = Checker()
for n, sql in d.execute("SELECT query_nr, query FROM tpch_queries() ORDER BY 1").fetchall():
    sql = sql.strip().rstrip(";")
    try:
        cur.execute(sql)
        got = norm_rows(cur.fetchall(), places=2)
    except Exception as e:
        c.check(f"Q{n}", False, str(e)[:300])
        continue
    want = norm_rows(d.execute(sql).fetchall(), places=2)
    c.check(f"Q{n}", got == want or rows_close(got, want, 1e-2), f"\n  got  {got[:3]}\n  want {want[:3]}")
cur.execute("DROP DATABASE tpch")
c.done()
