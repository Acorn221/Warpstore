"""SingleStore's "trillion rows per second" query on one GPU.

usage: python bench/trades.py ROWS [--skip-load]
"""
import os
import statistics
import sys
import time

import pymysql

rows = int(sys.argv[1])
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path = os.path.join(root, "data", f"trades-{rows}.tbl")
c = pymysql.connect(host="127.0.0.1", port=3307, user="root", autocommit=True).cursor()
if "--skip-load" not in sys.argv:
    c.execute("DROP TABLE IF EXISTS trade")
    c.execute("CREATE TABLE trade (id BIGINT, stock_symbol VARCHAR(5), shares INT, share_price DOUBLE, "
              "trade_time DATE, SHARD KEY (id))")
    t = time.perf_counter()
    c.execute(f"LOAD DATA INFILE '{path}' INTO TABLE trade FIELDS TERMINATED BY '|'")
    c.execute("OPTIMIZE TABLE trade")
    c.fetchall()
    print(f"loaded {rows:,} rows in {time.perf_counter() - t:.1f}s")

QUERIES = {
    "SingleStore's query: top 10 symbols by trade count":
        "SELECT stock_symbol, count(*) as c FROM trade GROUP BY stock_symbol ORDER BY c desc LIMIT 10",
    "plus a sum (still grouped by symbol)":
        "SELECT stock_symbol, count(*) c, sum(shares) s FROM trade GROUP BY stock_symbol ORDER BY c desc LIMIT 10",
    "filtered count (no group by)":
        "SELECT count(*) FROM trade WHERE shares > 500",
}
for name, q in QUERIES.items():
    c.execute(q)
    c.fetchall()
    ts = []
    for _ in range(10):
        t = time.perf_counter()
        c.execute(q)
        res = c.fetchall()
        ts.append(time.perf_counter() - t)
    med = statistics.median(ts)
    print(f"{name:55} {med * 1000:7.2f} ms  {rows / med / 1e9:6.2f} billion rows/s")
