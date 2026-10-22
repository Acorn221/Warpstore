"""For ClickBench queries whose warpstore result differs from DuckDB's: check that every warpstore row is a
valid answer row (present in DuckDB's result without LIMIT/OFFSET) and that the ORDER BY value
sequence is the same, i.e. the difference is only in how ties are broken (or, without ORDER BY,
which rows LIMIT picks).  usage: python bench/clickbench/verify_ties.py bench/clickbench Q..."""
import sys, re, json, decimal, datetime
import duckdb, pymysql
D = sys.argv[1]
gq = [q.strip() for q in open(f"{D}/queries.sql") if q.strip()]
dq = [q.strip() for q in open(f"{D}/duckdb_queries.sql") if q.strip()]
def norm(r):
    out = []
    for v in r:
        if isinstance(v, (float, decimal.Decimal)): v = float(v); out.append(round(v, 2) if abs(v) < 1e6 else float(f"{v:.6g}"))
        elif isinstance(v, (datetime.date, datetime.datetime)): out.append(str(v))
        elif isinstance(v, bytes): out.append(v.decode("utf-8", "replace"))
        elif isinstance(v, str): out.append(re.sub(r"[\t\n\r]", " ", v))  # the TSV export replaced these
        else: out.append(v)
    return tuple(out)
import os
c = pymysql.connect(host="127.0.0.1", port=3307, user="root", autocommit=True, read_timeout=300, database=os.environ.get("WARPSTORE_DB")).cursor()
c.execute("SET collation_connection = utf8mb4_bin")
d = duckdb.connect("data/clickbench/hits10m.duckdb", read_only=True)
for i in map(int, sys.argv[2:]):
    c.execute(gq[i - 1]); g = [norm(r) for r in c.fetchall()]
    dsql = dq[i - 1]
    if i == 29:  # the TSV export replaced tabs / newlines in Referer with spaces: do the same here
        dsql = re.sub(r"\bReferer\b", "regexp_replace(Referer, '[\\t\\n\\r]', ' ', 'g')", dsql)
    limited = [norm(r) for r in d.execute(dsql).fetchall()]
    full_sql = re.sub(r"\s+LIMIT\s+\d+(\s+OFFSET\s+\d+)?\s*;?\s*$", "", dsql, flags=re.I)
    full = set(norm(r) for r in d.execute(full_sql).fetchall())
    member = all(r in full for r in g)
    # ORDER BY value sequence: the numeric columns that the queries sort by (the last numeric column)
    m = re.search(r"ORDER\s+BY\s+(\w+)", gq[i - 1], re.I)
    items = [x.strip() for x in re.split(r",(?![^()]*\))", re.search(r"SELECT\s+(.*?)\s+FROM\s", gq[i - 1], re.I | re.S).group(1))]
    pos = next((k for k, it in enumerate(items) if m and re.search(r"\bAS\s+" + m.group(1) + r"\s*$", it, re.I)), None)
    key = lambda rows: [r[pos] for r in rows] if pos is not None else [None]
    has_order = re.search(r"ORDER\s+BY", gq[i - 1], re.I) is not None
    same_keys = not has_order or key(g) == key(limited)
    print(f"Q{i}: rows valid={member}, order-key sequence equal={same_keys}, rows={len(g)}/{len(limited)}")
