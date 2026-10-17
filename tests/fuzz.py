# Differential fuzzing: random queries over the same random data in gpu-db and DuckDB; any result
# difference or unexpected error fails. GPUDB_FUZZ_N queries (default 400), GPUDB_FUZZ_SEED.
# Avoided on purpose (MySQL and DuckDB legitimately differ): case-insensitive string collation,
# division by zero, ROUND half cases, integer overflow, float formatting.
import os, random, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import duckdb, pymysql
from lib import Checker, norm_rows, rows_close, server_from_env

N = int(os.environ.get("GPUDB_FUZZ_N", "400"))
SEED = int(os.environ.get("GPUDB_FUZZ_SEED", "12345"))
R = random.Random(SEED)
WORDS = ["ant", "bee", "cat", "dog", "eel", "fox", "gnu", "hen", "ibis", "jay", "kiwi", "lark"]

srv = server_from_env()
g = srv.cursor()
g.execute("DROP DATABASE IF EXISTS fuzz")
g.execute("CREATE DATABASE fuzz")
g.execute("USE fuzz")
KEEP = os.environ.get("GPUDB_FUZZ_KEEP")  # path: keep the data (DuckDB file + gpu-db database) for debugging
if KEEP and os.path.exists(KEEP):
    os.unlink(KEEP)
d = duckdb.connect(KEEP or ":memory:")
d.execute("SET default_null_order = 'nulls_first_on_asc_last_on_desc'")  # MySQL: NULLs sort first ascending

T1 = "id INT, a INT, b BIGINT, c DOUBLE, s VARCHAR(20), d DATE, m DECIMAL(10,2)"
T2 = "id INT, t1_id INT, x INT, s2 VARCHAR(20)"
g.execute(f"CREATE TABLE t1 ({T1}, PRIMARY KEY(id))")
g.execute(f"CREATE TABLE t2 ({T2}, PRIMARY KEY(id), SHARD KEY(id))")
d.execute(f"CREATE TABLE t1 ({T1.replace('VARCHAR(20)', 'VARCHAR')})")
d.execute(f"CREATE TABLE t2 ({T2.replace('VARCHAR(20)', 'VARCHAR')})")


def lit(v):
    if v is None:
        return "NULL"
    if isinstance(v, str):
        return "'" + v + "'"
    return str(v)


def maybe(v, p=0.12):
    return None if R.random() < p else v


rows1 = []
for i in range(1, 3001):
    rows1.append((i, maybe(R.randint(-50, 50)), R.choice([R.randint(-10**6, 10**6), R.randint(0, 9)]), maybe(round(R.uniform(-1000, 1000), 3)),
                  maybe(R.choice(WORDS) + ("" if R.random() < 0.7 else str(R.randint(0, 9)))),
                  maybe(f"20{R.randint(20, 25)}-{R.randint(1, 12):02d}-{R.randint(1, 28):02d}"), maybe(round(R.uniform(-500, 500), 2))))
rows2 = [(i, maybe(R.randint(1, 3200), 0.05), maybe(R.randint(0, 20)), maybe(R.choice(WORDS))) for i in range(1, 1201)]
for name, rows in (("t1", rows1), ("t2", rows2)):
    for k in range(0, len(rows), 500):
        vals = ",".join("(" + ",".join(lit(v) for v in r) + ")" for r in rows[k:k + 500])
        g.execute(f"INSERT INTO {name} VALUES {vals}")
        d.execute(f"INSERT INTO {name} VALUES {vals}")
# a second segment plus deletes and updates, so the queries cover mixed storage
g.execute("OPTIMIZE TABLE t1")
for sql in ("DELETE FROM t1 WHERE id % 17 = 3", "UPDATE t1 SET a = a + 1 WHERE id % 13 = 5", "UPDATE t2 SET s2 = 'zebra' WHERE id % 11 = 0"):
    g.execute(sql)
    d.execute(sql)

INT_COLS = {"t1": ["a", "b", "id"], "t2": ["x", "t1_id", "id"]}
NUM_COLS = {"t1": ["c", "m"], "t2": []}
STR_COLS = {"t1": ["s"], "t2": ["s2"]}


class Q:
    """One generated expression, rendered for both dialects."""

    def __init__(self, my, dk=None, kind="num"):
        self.my, self.dk, self.kind = my, dk if dk is not None else my, kind


def col(tabs, kind):
    opts = []
    for alias, t in tabs:
        src = {"int": INT_COLS[t], "num": INT_COLS[t] + NUM_COLS[t], "str": STR_COLS[t], "date": ["d"] if t == "t1" else []}[kind]
        opts += [f"{alias}.{c}" for c in src]
    if not opts:
        return None
    c = R.choice(opts)
    return Q(c, c, kind)


def num_expr(tabs, depth=0):
    r = R.random()
    if depth > 2 or r < 0.35:
        c = col(tabs, "num")
        return c if R.random() < 0.85 else Q(str(R.randint(-20, 20)))
    if r < 0.6:
        a, b = num_expr(tabs, depth + 1), num_expr(tabs, depth + 1)
        op = R.choice(["+", "-", "*"])
        return Q(f"({a.my} {op} {b.my})", f"({a.dk} {op} {b.dk})")
    if r < 0.68:
        a, b = num_expr(tabs, depth + 1), num_expr(tabs, depth + 1)
        return Q(f"({a.my} / NULLIF({b.my}, 0))", f"({a.dk} / NULLIF({b.dk}, 0))")
    if r < 0.75:
        a = num_expr(tabs, depth + 1)
        return Q(f"ABS({a.my})", f"ABS({a.dk})")
    if r < 0.82:
        a, b = num_expr(tabs, depth + 1), num_expr(tabs, depth + 1)
        return Q(f"COALESCE({a.my}, {b.my})", f"COALESCE({a.dk}, {b.dk})")
    if r < 0.9:
        p = pred(tabs, depth + 1)
        a, b = num_expr(tabs, depth + 1), num_expr(tabs, depth + 1)
        return Q(f"CASE WHEN {p.my} THEN {a.my} ELSE {b.my} END", f"CASE WHEN {p.dk} THEN {a.dk} ELSE {b.dk} END")
    if r < 0.95:
        s = str_expr(tabs, depth + 1)
        return Q(f"LENGTH({s.my})", f"LENGTH({s.dk})")
    c = col(tabs, "int")
    m = R.randint(2, 7)
    return Q(f"({c.my} % {m})", f"({c.dk} % {m})")


def str_expr(tabs, depth=0):
    c = col(tabs, "str") or Q("'x'")
    r = R.random()
    if depth > 1 or r < 0.5:
        return c
    if r < 0.65:
        return Q(f"UPPER({c.my})", f"UPPER({c.dk})")
    if r < 0.8:
        a, b = R.randint(1, 3), R.randint(1, 3)
        return Q(f"SUBSTRING({c.my}, {a}, {b})", f"SUBSTRING({c.dk}, {a}, {b})")
    if r < 0.9:
        o = str_expr(tabs, depth + 1)
        return Q(f"CONCAT({c.my}, '-', {o.my})", f"({c.dk} || '-' || {o.dk})")  # DuckDB's CONCAT skips NULLs
    return Q(f"COALESCE({c.my}, 'none')", f"COALESCE({c.dk}, 'none')")


def pred(tabs, depth=0):
    r = R.random()
    if depth < 2 and r < 0.2:
        a, b = pred(tabs, depth + 1), pred(tabs, depth + 1)
        op = R.choice(["AND", "OR"])
        return Q(f"({a.my} {op} {b.my})", f"({a.dk} {op} {b.dk})")
    if depth < 2 and r < 0.26:
        a = pred(tabs, depth + 1)
        return Q(f"NOT ({a.my})", f"NOT ({a.dk})")
    if r < 0.55:
        a = num_expr(tabs, depth + 1)
        op = R.choice(["=", "<>", "<", "<=", ">", ">="])
        return _cmp(a, op)
    if r < 0.63:
        a = col(tabs, "num")
        lo = R.randint(-40, 20)
        return _between(a, lo)
    if r < 0.72:
        a = col(tabs, "int")
        vals = ", ".join(str(R.randint(-10, 10)) for _ in range(R.randint(1, 5)))
        neg = "NOT " if R.random() < 0.3 else ""
        return Q(f"{a.my} {neg}IN ({vals})", f"{a.dk} {neg}IN ({vals})")
    if r < 0.8:
        a = R.choice([col(tabs, "num"), col(tabs, "str") or col(tabs, "num")])
        neg = "NOT " if R.random() < 0.4 else ""
        return Q(f"{a.my} IS {neg}NULL", f"{a.dk} IS {neg}NULL")
    if r < 0.88:
        s = col(tabs, "str")
        if s:
            pat = R.choice(["a%", "%a%", "_o%", "%e", "b__", "%"])
            return Q(f"{s.my} LIKE '{pat}'", f"{s.dk} LIKE '{pat}'")
    if r < 0.94:
        s = col(tabs, "str")
        if s:
            w = R.choice(WORDS)
            op = R.choice(["=", "<>", "<", ">="])
            return Q(f"{s.my} {op} '{w}'", f"{s.dk} {op} '{w}'")
    dc = col(tabs, "date")
    if dc:
        day = f"20{R.randint(20, 25)}-{R.randint(1, 12):02d}-01"
        op = R.choice(["<", ">=", "="])
        return Q(f"{dc.my} {op} DATE '{day}'", f"{dc.dk} {op} DATE '{day}'")
    return _cmp(num_expr(tabs, depth + 1), ">")


def _cmp(a, op):
    k = R.randint(-30, 30)
    return Q(f"{a.my} {op} {k}", f"{a.dk} {op} {k}")


def _between(a, lo):
    hi = lo + R.randint(0, 40)
    return Q(f"{a.my} BETWEEN {lo} AND {hi}", f"{a.dk} BETWEEN {lo} AND {hi}")


def agg(tabs):
    r = R.random()
    if r < 0.2:
        return Q("COUNT(*)")
    a = num_expr(tabs, 1)
    f = R.choice(["SUM", "MIN", "MAX", "AVG", "COUNT", "COUNT_DISTINCT"])
    if f == "COUNT_DISTINCT":
        c = R.choice([col(tabs, "int"), col(tabs, "str") or col(tabs, "int")])
        return Q(f"COUNT(DISTINCT {c.my})", f"COUNT(DISTINCT {c.dk})")
    return Q(f"{f}({a.my})", f"{f}({a.dk})")


def from_clause():
    r = R.random()
    if r < 0.55:
        return [("t1", "t1")], "t1", "t1"
    if r < 0.7:
        return [("t2", "t2")], "t2", "t2"
    jt = R.choice(["JOIN", "LEFT JOIN"])
    return [("t1", "t1"), ("t2", "t2")], f"t1 {jt} t2 ON t1.id = t2.t1_id", f"t1 {jt} t2 ON t1.id = t2.t1_id"


def gen():
    tabs, fm, fd = from_clause()
    where = pred(tabs) if R.random() < 0.75 else None
    shape = R.random()
    order = R.random() < 0.4
    if shape < 0.45:  # grouped
        # no division in keys: MySQL rounds DECIMAL quotients to 4 places, so its groups merge where DuckDB's don't
        keys = [R.choice([col(tabs, "int"), col(tabs, "str") or col(tabs, "int"), num_expr(tabs, 2)]) for _ in range(R.randint(0, 2))]
        keys = [k if "/" not in k.my else col(tabs, "int") for k in keys]
        aggs = [agg(tabs) for _ in range(R.randint(1, 3))]
        items = keys + aggs
        my = "SELECT " + ", ".join(f"{e.my} AS c{i}" for i, e in enumerate(items)) + f" FROM {fm}"
        dk = "SELECT " + ", ".join(f"{e.dk} AS c{i}" for i, e in enumerate(items)) + f" FROM {fd}"
        if where:
            my += f" WHERE {where.my}"
            dk += f" WHERE {where.dk}"
        if keys:
            my += " GROUP BY " + ", ".join(e.my for e in keys)
            dk += " GROUP BY " + ", ".join(e.dk for e in keys)
            if R.random() < 0.3:
                k = R.randint(0, 5)
                my += f" HAVING COUNT(*) > {k}"
                dk += f" HAVING COUNT(*) > {k}"
    elif shape < 0.55:  # window
        part = col(tabs, "int")
        o = col(tabs, "int")
        f = R.choice(["ROW_NUMBER()", "RANK()", "DENSE_RANK()", "SUM({x}) OVER", "COUNT(*) OVER"])
        idc = "t1.id" if tabs[0][0] == "t1" else "t2.id"
        if tabs[-1][0] == "t2" and len(tabs) == 2:
            idc = "t1.id, t2.id"
        w = f"PARTITION BY {part.my} ORDER BY {o.my}, {idc}"
        if f.startswith("SUM"):
            x = col(tabs, "int")
            fx = f"SUM({x.my}) OVER ({w})"
        elif f.startswith("COUNT"):
            fx = f"COUNT(*) OVER (PARTITION BY {part.my})"
        else:
            fx = f"{f} OVER ({w})"
        my = f"SELECT {idc.split(',')[0]} AS c0, {fx} AS c1 FROM {fm}"
        dk = my.replace(fm, fd)
        if where:
            my += f" WHERE {where.my}"
            dk += f" WHERE {where.dk}"
        if "," in idc:
            my = my.replace("AS c0,", f"AS c0, {idc.split(',')[1]} AS c2,")
            dk = dk.replace("AS c0,", f"AS c0, {idc.split(',')[1]} AS c2,")
    elif shape < 0.65:  # subquery
        inner = pred([("t2", "t2")])
        sub = R.choice(["IN", "NOT IN", "EXISTS", "SCALAR"])
        e = num_expr([("t1", "t1")], 1)
        if sub in ("IN", "NOT IN"):
            w = f"t1.a {sub} (SELECT t2.x FROM t2 WHERE {inner.my} AND t2.x IS NOT NULL)"
        elif sub == "EXISTS":
            w = f"EXISTS (SELECT 1 FROM t2 WHERE t2.t1_id = t1.id AND {inner.my})"
        else:
            w = f"t1.a > (SELECT AVG(t2.x) FROM t2 WHERE {inner.my})"
        my = dk = f"SELECT t1.id AS c0, {e.my} AS c1 FROM t1 WHERE {w}"
        dk = f"SELECT t1.id AS c0, {e.dk} AS c1 FROM t1 WHERE {w.replace(inner.my, inner.dk)}"
    elif shape < 0.72:  # union / distinct
        p1, p2 = pred([("t1", "t1")]), pred([("t1", "t1")])
        u = R.choice(["UNION", "UNION ALL"])
        my = f"SELECT t1.a AS c0, t1.s AS c1 FROM t1 WHERE {p1.my} {u} SELECT t1.a, t1.s FROM t1 WHERE {p2.my}"
        dk = f"SELECT t1.a AS c0, t1.s AS c1 FROM t1 WHERE {p1.dk} {u} SELECT t1.a, t1.s FROM t1 WHERE {p2.dk}"
    else:  # plain projection
        items = [R.choice([num_expr(tabs), str_expr(tabs)]) for _ in range(R.randint(1, 3))]
        dist = "DISTINCT " if R.random() < 0.15 and not any("/" in e.my for e in items) else ""
        my = f"SELECT {dist}" + ", ".join(f"{e.my} AS c{i}" for i, e in enumerate(items)) + f" FROM {fm}"
        dk = f"SELECT {dist}" + ", ".join(f"{e.dk} AS c{i}" for i, e in enumerate(items)) + f" FROM {fd}"
        if where:
            my += f" WHERE {where.my}"
            dk += f" WHERE {where.dk}"
    ncols = my.split(" FROM ")[0].count(" AS c")
    cols = ", ".join(f"c{i}" for i in range(ncols)) if ncols else None
    w = R.random()
    if cols and w < 0.12:  # derived table
        my, dk = f"SELECT * FROM ({my}) dt WHERE c0 IS NOT NULL", f"SELECT * FROM ({dk}) dt WHERE c0 IS NOT NULL"
    elif cols and w < 0.22:  # CTE + aggregate over it
        my = f"WITH q AS ({my}) SELECT COUNT(*), COUNT(c0), MAX(c0) FROM q"
        dk = f"WITH q AS ({dk}) SELECT COUNT(*), COUNT(c0), MAX(c0) FROM q"
    elif cols and w < 0.4 and "/" not in my and " UNION" not in my:  # total order + LIMIT (ties are identical rows)
        dirs = ", ".join(f"c{i}{R.choice(['', ' DESC'])}" for i in range(ncols))
        k = R.choice([1, 5, 20, 100])
        my, dk = f"{my} ORDER BY {dirs} LIMIT {k}", f"{dk} ORDER BY {dirs} LIMIT {k}"
    return my, dk


c = Checker()
errors = {}
t0 = time.time()
for i in range(N):
    my, dk = gen()
    try:
        want = d.execute(dk).fetchall()
    except duckdb.Error:
        continue  # generator produced something DuckDB rejects; not interesting
    try:
        g.execute(my)
        got = g.fetchall()
    except pymysql.MySQLError as e:
        key = str(e.args[1])[:80]
        errors[key] = errors.get(key, 0) + 1
        c.check(f"q{i}", False, f"ERROR {e.args}\n  {my}")
        continue
    a, b = norm_rows(got, ordered=False, places=4), norm_rows(want, ordered=False, places=4)
    if a != b and len(a) == len(b):  # last-digit rounding can reorder rows: sort on coarser numbers
        coarse = lambda r: tuple((x is None, str(type(x)), round(x, 2) if isinstance(x, float) else x if x is not None else 0) for x in r)
        a, b = sorted(a, key=coarse), sorted(b, key=coarse)
    # 1.5e-4: MySQL rounds DECIMAL division to 4 places, DuckDB keeps more digits
    c.check(f"q{i}", a == b or rows_close(a, b, 1e-6, 1.5e-4), f"\n  {my}\n  gpu-db {len(a)} rows {a[:4]}\n  duckdb {len(b)} rows {b[:4]}")
print(f"{N} queries in {time.time() - t0:.1f}s, seed {SEED}")
for k, v in sorted(errors.items(), key=lambda kv: -kv[1]):
    print(f"  {v:4d} x {k}")
if not KEEP:
    g.execute("DROP DATABASE fuzz")
c.done()
