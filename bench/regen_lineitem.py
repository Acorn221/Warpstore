"""Memory-safe, resumable lineitem generation: dbgen in chunks (children/step) with a DuckDB memory cap.

usage: python bench/regen_lineitem.py SF OUT [first_step last_step]   (then run again with no steps to concatenate)
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import duckdb
from tpch import TABLES
sf, out, chunks = float(sys.argv[1]), sys.argv[2], 10
steps = range(int(sys.argv[3]), int(sys.argv[4]) + 1) if len(sys.argv) > 4 else range(chunks)
for step in steps:
    part = f"{out}.part{step}"
    if os.path.exists(part):
        continue
    c = duckdb.connect()
    c.execute("SET memory_limit='6GB'; SET preserve_insertion_order=false; SET threads=4; INSTALL tpch; LOAD tpch")
    c.execute(f"CALL dbgen(sf={sf}, children={chunks}, step={step})")
    c.execute(f"COPY (SELECT {TABLES['lineitem'][1]} FROM lineitem) TO '{part}.tmp' (DELIMITER '|', HEADER false, FORMAT csv)")
    c.close()
    os.rename(part + ".tmp", part)
    print("chunk", step, "done", flush=True)
if len(sys.argv) <= 4 and all(os.path.exists(f"{out}.part{i}") for i in range(chunks)):
    with open(out + ".tmp", "wb") as f:
        for i in range(chunks):
            with open(f"{out}.part{i}", "rb") as p:
                while b := p.read(64 << 20):
                    f.write(b)
    os.rename(out + ".tmp", out)
    for i in range(chunks):
        os.remove(f"{out}.part{i}")
    print("done", out)
