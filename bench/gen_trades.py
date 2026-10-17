"""Generate a SingleStore "trillion rows per second"-style trade table (skewed stock symbols) with DuckDB.

usage: python bench/gen_trades.py ROWS     -> data/trades-ROWS.tbl  (id|stock_symbol|shares|share_price|trade_time)
"""
import os
import sys

import duckdb

rows = int(sys.argv[1])
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out = os.path.join(root, "data", f"trades-{rows}.tbl")
d = duckdb.connect()
d.execute("SET memory_limit='2GB'")
d.execute("SET preserve_insertion_order=false")
# Log-uniform symbol index => P(symbol k) ~ 1/k: a few large caps get most of the trades.
d.execute(f"""COPY (
  SELECT range AS id,
         'S' || lpad(CAST(CAST(floor(exp(random() * ln(3000))) AS INT) AS VARCHAR), 4, '0') AS stock_symbol,
         CAST(1 + floor(random() * 1000) AS INT) AS shares,
         round(1 + random() * 500, 2) AS share_price,
         DATE '2010-01-01' + CAST(floor(random() * 5000) AS INT) AS trade_time
  FROM range({rows})
) TO '{out}' (DELIMITER '|', HEADER false)""")
print(out, os.path.getsize(out) // 2**20, "MiB")
