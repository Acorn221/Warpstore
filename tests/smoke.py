# Basic DDL/DML/query round trip.
import datetime, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import Checker, server_from_env

srv = server_from_env()
cur = srv.cursor()
c = Checker()
def q(sql):
    cur.execute(sql)
    return [tuple(r) for r in cur.fetchall()]

cur.execute("CREATE DATABASE IF NOT EXISTS smoke")
cur.execute("USE smoke")
cur.execute("DROP TABLE IF EXISTS t")
cur.execute("CREATE TABLE t (id INT PRIMARY KEY, name VARCHAR(20), price DOUBLE, d DATE, SORT KEY(id))")
cur.execute("INSERT INTO t VALUES (1,'apple',1.5,'2024-01-01'),(2,'banana',0.25,'2024-02-01'),(3,'cherry',3.0,'2024-03-01'),(4,'apple',2.5,'2024-04-01')")
c.eq("select all", len(q("SELECT * FROM t")), 4)
c.eq("point", q("SELECT name FROM t WHERE id = 3"), [("cherry",)])
c.eq("group by", q("SELECT name, COUNT(*), SUM(price), MIN(d) FROM t GROUP BY name ORDER BY name"),
     [("apple", 2, 4.0, datetime.date(2024, 1, 1)), ("banana", 1, 0.25, datetime.date(2024, 2, 1)), ("cherry", 1, 3.0, datetime.date(2024, 3, 1))])
c.eq("or filter", q("SELECT COUNT(*), SUM(price) FROM t WHERE name LIKE 'b%' OR price > 2"), [(3, 5.75)])
c.eq("order limit", q("SELECT id, price*2 AS p2 FROM t WHERE d >= DATE '2024-02-01' ORDER BY p2 DESC LIMIT 2"), [(3, 6.0), (4, 5.0)])
cur.execute("UPDATE t SET price = price + 10 WHERE name = 'apple'")
c.eq("update rowcount", cur.rowcount, 2)
cur.execute("DELETE FROM t WHERE id = 2")
c.eq("delete rowcount", cur.rowcount, 1)
q("OPTIMIZE TABLE t")
cur.execute("INSERT INTO t VALUES (5,'date',9.0,'2024-05-01')")
c.eq("after optimize+insert", q("SELECT id, price FROM t ORDER BY id"), [(1, 11.5), (3, 3.0), (4, 12.5), (5, 9.0)])
c.eq("having", q("SELECT name, SUM(price) s FROM t GROUP BY name HAVING s > 5 ORDER BY s DESC"), [("apple", 24.0), ("date", 9.0)])
c.raises("duplicate key", lambda: cur.execute("INSERT INTO t VALUES (5,'dup',1,'2024-01-01')"), 1062)
c.eq("case group", q("SELECT CASE WHEN price > 10 THEN 1 ELSE 0 END AS big, COUNT(*) FROM t GROUP BY big ORDER BY big"), [(0, 2), (1, 2)])
c.eq("year()", q("SELECT YEAR(d) y, COUNT(*) FROM t GROUP BY y ORDER BY y"), [(2024, 4)])
c.eq("in list", q("SELECT id FROM t WHERE name IN ('apple','date') ORDER BY id"), [(1,), (4,), (5,)])
c.check("explain", len(q("EXPLAIN SELECT name, SUM(price) FROM t WHERE id > 2 GROUP BY name")) > 0)
cur.execute("DROP DATABASE smoke")
# String functions count characters, not bytes (UTF-8), and case-map beyond ASCII.
c.eq("utf-8 substring / left / right", q("SELECT SUBSTRING('привет', 1, 3), SUBSTRING('привет', -2), LEFT('привет', 2), RIGHT('héllo', 3), MID('日本語テキスト', 3, 2)"),
     [("при", "ет", "пр", "llo", "語テ")])
c.eq("utf-8 case mapping / reverse", q("SELECT UPPER('привет мир'), LOWER('ПРИВЕТ Ünïcödé'), REVERSE('añb')"), [("ПРИВЕТ МИР", "привет ünïcödé", "bña")])
c.eq("utf-8 pad / locate / lengths", q("SELECT LPAD('ж', 4, 'аб'), LOCATE('в', 'привет'), INSTR('日本語', '語'), CHAR_LENGTH('日本語'), LENGTH('日本語')"),
     [("абаж", 4, 3, 3, 9)])
# GROUP BY DATE_FORMAT(datetime, ..): evaluated per time bucket (here: minutes) on the GPU.
cur.execute("CREATE DATABASE IF NOT EXISTS smoke2")
cur.execute("CREATE TABLE smoke2.ts (id INT, t DATETIME)")
base = datetime.datetime(2024, 3, 9, 22, 58, 0)
tvals = [base + datetime.timedelta(seconds=37 * i) for i in range(400)]
cur.execute("INSERT INTO smoke2.ts VALUES " + ",".join(f"({i}, '{v}')" for i, v in enumerate(tvals)) + ", (999, NULL)")
want = {}
for v in tvals:
    k = v.strftime("%Y/%m/%d %H:%M")
    want[k] = want.get(k, 0) + 1
want = sorted(want.items()) + [(None, 1)]
got = q("SELECT DATE_FORMAT(t, '%Y/%m/%d %H:%i') k, COUNT(*) FROM smoke2.ts GROUP BY k ORDER BY k IS NULL, k")
c.eq("GROUP BY DATE_FORMAT", got, want)
cur.execute("DROP DATABASE smoke2")
c.done()
