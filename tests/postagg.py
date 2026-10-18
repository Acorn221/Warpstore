"""GROUP BY + HAVING / ORDER BY / LIMIT over many groups (GPU post-aggregation path), checked against DuckDB.

usage: python tests/postagg.py [port]
"""
import os
import random
import sys
import tempfile

import duckdb
import pymysql

port = int(sys.argv[1]) if len(sys.argv) > 1 else 3307
random.seed(7)
words = [f"w{i:04d}" for i in range(3000)]
rows = [(i, random.randint(1, 50000), random.choice(words), round(random.uniform(0, 100), 2), random.randint(0, 9))
        for i in range(300_000)]
path = os.path.join(tempfile.mkdtemp(), "pa.tbl")
with open(path, "w") as f:
    for r in rows:
        f.write("|".join(map(str, r)) + "\n")

ddl = "id BIGINT, k INT, name VARCHAR(10), v DOUBLE, c INT"
g = pymysql.connect(host="127.0.0.1", port=port, user="root", password="", autocommit=True).cursor()
g.execute("DROP TABLE IF EXISTS pa")
g.execute(f"CREATE TABLE pa ({ddl}, PRIMARY KEY (id))")
g.execute(f"LOAD DATA INFILE '{path}' INTO TABLE pa FIELDS TERMINATED BY '|'")
g.execute("OPTIMIZE TABLE pa")
g.fetchall()
d = duckdb.connect()
d.execute(f"CREATE TABLE pa ({ddl})")
d.execute(f"COPY pa FROM '{path}' (DELIMITER '|', HEADER false)")

QUERIES = [
    "select k, count(*) n, sum(v) s from pa group by k order by s desc, k limit 10",
    "select k, count(*) n from pa group by k order by n desc, k limit 5 offset 7",
    "select name, avg(v) a from pa group by name order by a, name limit 8",
    "select name, count(*) n from pa group by name order by name desc limit 6",
    "select k, c, sum(v) s from pa group by k, c having sum(v) > 150 and count(*) >= 2 order by s desc, k, c limit 10",
    "select name, max(v) m from pa group by name having max(v) < 99.5 order by m, name",
    "select name, count(*) n from pa group by name having name like 'w00%' order by n desc, name limit 4",
    "select k, sum(v) / count(*) r from pa group by k order by r desc, k limit 3",
    "select k * 2 + c kk, count(*) n from pa group by k, c order by n desc, kk limit 5",
]


def norm(rs):
    return [tuple(round(float(x), 6) if isinstance(x, (int, float)) or type(x).__name__ == "Decimal" else x for x in r) for r in rs]


ok = True
for q in QUERIES:
    try:
        g.execute(q)
        a = norm(g.fetchall())
    except Exception as e:  # noqa: BLE001
        a = f"ERROR {e}"
    b = norm(d.execute(q).fetchall())
    good = a == b
    ok &= good
    print(("PASS " if good else "FAIL ") + q)
    if not good:
        print("   warpstore:", a[:5] if isinstance(a, list) else a)
        print("   duckdb:", b[:5])
g.execute("DROP TABLE pa")
sys.exit(0 if ok else 1)
