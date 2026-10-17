"""SHARD KEY / partitioning: placement, routing, co-located joins, DML across partitions. Checked against DuckDB.

usage: python tests/sharding.py [port]      (server started with the default 8 partitions)
"""
import os
import random
import sys
import tempfile

import duckdb
import pymysql

port = int(sys.argv[1]) if len(sys.argv) > 1 else 3307
g = pymysql.connect(host="127.0.0.1", port=port, user="root", password="", autocommit=True).cursor()
d = duckdb.connect()
fails = 0


def check(name, cond, detail=""):
    global fails
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"  {detail}"))
    fails += 0 if cond else 1


def gq(sql):
    g.execute(sql)
    return g.fetchall()


def explain(sql):
    return "\n".join(r[0] for r in gq("EXPLAIN " + sql))


def norm(rs):
    num = lambda x: isinstance(x, (int, float)) or type(x).__name__ == "Decimal"
    return sorted(tuple(float(x) if num(x) else str(x) for x in r) for r in rs)


def close(a, b):
    if len(a) != len(b):
        return False
    for ra, rb in zip(a, b):
        for x, y in zip(ra, rb):
            if isinstance(x, float) and isinstance(y, float):
                if abs(x - y) > 1e-9 * max(1.0, abs(x), abs(y)):  # FP64 sums differ in order
                    return False
            elif x != y:
                return False
    return True


def same(sql):
    a, b = norm(gq(sql)), norm(d.execute(sql).fetchall())
    check("same result: " + sql, close(a, b), f"gpu-db {a[:4]} vs duckdb {b[:4]}")


def both(sql, duck_sql=None):
    g.execute(sql)
    d.execute(duck_sql or sql)


for t in ("orders_s", "items_s", "region_r", "keyless_s", "strkey_s", "bad_s"):
    g.execute(f"DROP TABLE IF EXISTS {t}")

# ---- DDL rules
both("CREATE TABLE orders_s (o_id BIGINT, o_cust INT, o_region INT, o_total DOUBLE, o_day DATE, PRIMARY KEY (o_id))",
     "CREATE TABLE orders_s (o_id BIGINT, o_cust INT, o_region INT, o_total DOUBLE, o_day DATE)")
both("CREATE TABLE items_s (i_order BIGINT, i_line INT, i_qty INT, i_price DOUBLE, SHARD KEY (i_order))",
     "CREATE TABLE items_s (i_order BIGINT, i_line INT, i_qty INT, i_price DOUBLE)")
both("CREATE REFERENCE TABLE region_r (r_id INT, r_name VARCHAR(20), PRIMARY KEY (r_id))",
     "CREATE TABLE region_r (r_id INT, r_name VARCHAR(20))")
both("CREATE TABLE keyless_s (k INT, v DOUBLE, SHARD KEY ())", "CREATE TABLE keyless_s (k INT, v DOUBLE)")
both("CREATE TABLE strkey_s (name VARCHAR(20), v INT, PRIMARY KEY (name))", "CREATE TABLE strkey_s (name VARCHAR(20), v INT)")
try:
    g.execute("CREATE TABLE bad_s (a INT, b INT, PRIMARY KEY (a), SHARD KEY (b))")
    check("shard key outside the primary key is rejected", False, "no error")
except pymysql.err.MySQLError as e:
    check("shard key outside the primary key is rejected", e.args[0] == 1705, str(e))

# ---- data: bulk loaded (segments) plus trickled inserts (write buffers)
random.seed(11)
tmp = tempfile.mkdtemp()
N = 400_000
orders = [(i, random.randint(1, 20000), random.randint(0, 4), round(random.uniform(1, 500), 2),
           f"2024-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}") for i in range(1, N + 1)]
items = [(o[0], ln, random.randint(1, 50), round(random.uniform(1, 100), 2))
         for o in orders for ln in range(random.randint(1, 4))]
for name, rows in (("orders_s", orders[:-2000]), ("items_s", items[:-5000])):
    path = os.path.join(tmp, name + ".tbl")
    with open(path, "w") as f:
        f.write("".join("|".join(map(str, r)) + "\n" for r in rows))
    g.execute(f"LOAD DATA INFILE '{path}' INTO TABLE {name} FIELDS TERMINATED BY '|'")
    d.execute(f"COPY {name} FROM '{path}' (DELIMITER '|', HEADER false)")
    g.execute(f"OPTIMIZE TABLE {name}")
    g.fetchall()
for o in orders[-2000:]:
    both("INSERT INTO orders_s VALUES (%d, %d, %d, %s, '%s')" % o)
for i in range(0, 5000, 500):
    vals = ", ".join("(%d, %d, %d, %s)" % r for r in items[-5000 + i:-5000 + i + 500 or None])
    both("INSERT INTO items_s VALUES " + vals)
both("INSERT INTO region_r VALUES (0, 'north'), (1, 'south'), (2, 'east'), (3, 'west'), (4, 'central')")
both("INSERT INTO keyless_s VALUES " + ", ".join(f"({k}, {k * 1.5})" for k in range(1, 801)))
both("INSERT INTO strkey_s VALUES " + ", ".join(f"('name{k}', {k})" for k in range(1, 501)))
g.execute("OPTIMIZE TABLE keyless_s")
g.fetchall()

# ---- placement
segs = gq("SHOW SEGMENTS FROM orders_s")
parts = {int(r[0]) for r in segs}
check("orders_s rows are spread over 8 partitions", parts == set(range(8)), str(sorted(parts)))
per = {}
for r in segs:
    per[int(r[0])] = per.get(int(r[0]), 0) + int(r[3])
check("partitions are balanced (hash of the primary key)", max(per.values()) < 1.1 * min(per.values()), str(per))
ref = gq("SHOW SEGMENTS FROM region_r")
check("reference table is a single partition", {int(r[0]) for r in ref} == {0}, str(ref))
kl = {int(r[0]): int(r[3]) for r in gq("SHOW SEGMENTS FROM keyless_s")}
check("keyless table is spread round-robin", len(kl) == 8 and set(kl.values()) == {100}, str(kl))

# ---- routing
e = explain("select count(*), sum(i_qty) from items_s where i_order = 12345")
check("shard-key equality scans one partition", "1/8 partitions (shard key routing)" in e, e)
e = explain("select count(*) from items_s where i_qty = 3")
check("non-shard-key filter scans every partition", "8/8 partitions" in e, e)
same("select count(*), sum(i_qty), sum(i_price) from items_s where i_order = 12345")
same(f"select count(*), sum(i_qty) from items_s where i_order = {orders[-1][0]}")  # lives in a write buffer
same("select count(*), sum(i_qty) from items_s where i_order = 999999999")
same("select v from strkey_s where name = 'name77'")
same("select count(*) from strkey_s where name = 'nope'")
same("select o_id, o_total from orders_s where o_id = 777")

# ---- joins
q = ("select o_region, count(*), sum(i_qty * i_price) from orders_s, items_s "
     "where o_id = i_order and o_total > 250 group by o_region")
e = explain(q)
check("join on both shard keys is co-located", "co-located, 8 partitions" in e, e)
same(q)
q2 = ("select o_id, i_line, i_qty from orders_s join items_s on o_id = i_order "
      "where o_cust = 4242 order by o_id, i_line")
same(q2)
q3 = ("select r_name, count(*), sum(o_total) from orders_s, region_r where o_region = r_id group by r_name")
same(q3)
q4 = ("select r_name, sum(i_qty) from region_r, orders_s, items_s "
      "where r_id = o_region and o_id = i_order and o_day >= date '2024-06-01' group by r_name")
same(q4)
q5 = ("select o_cust, count(*) n from orders_s, items_s where o_id = i_order group by o_cust order by n desc, o_cust limit 5")
same(q5)
# A join that is not on both shard keys must not be treated as co-located.
nq = "select count(*) from orders_s, items_s where o_cust = i_order and o_region = 2"
e = explain(nq)
check("join off the shard key uses the broadcast path", "broadcast" in e and "co-located" not in e, e)
same(nq)
e = explain(q4)
check("orders->items stays co-located after joining a reference table", "co-located" in e, e)

# ---- the same joins with join filters forced on every table (normally only tables >= 2M rows)
g.execute("SET gpudb_join_filter_rows = 0")
qsel = q.replace("o_total > 250", "o_total > 400")  # keeps ~20% of orders: selective enough to filter items
e = explain(qsel)
check("join filters are applied when enabled", "join filter" in e, e)
for jq in (qsel, q, q2, q3, q4, q5, nq):
    same(jq)
g.execute("SET gpudb_join_filters = OFF")
for jq in (q, q4):
    same(jq)
g.execute("SET gpudb_join_filters = ON")
g.execute("SET gpudb_join_filter_rows = 2000000")

# ---- DML across partitions: UPDATE of the shard key moves the row to its new partition
both("UPDATE items_s SET i_order = 424242 WHERE i_order = 100 AND i_line = 0")
both("DELETE FROM orders_s WHERE o_id = 101")
both("UPDATE orders_s SET o_total = o_total + 1 WHERE o_id = 102")
same("select count(*), sum(i_qty) from items_s where i_order = 424242")
same("select count(*), sum(i_qty) from items_s where i_order = 100")
same("select count(*), sum(o_total) from orders_s where o_id between 100 and 105")
same(q)
same("select count(*), sum(o_total) from orders_s")
same("select count(*), sum(i_qty) from items_s")

for t in ("orders_s", "items_s", "region_r", "keyless_s", "strkey_s"):
    g.execute(f"DROP TABLE {t}")
print("ALL PASS" if fails == 0 else f"{fails} FAILED")
sys.exit(1 if fails else 0)
