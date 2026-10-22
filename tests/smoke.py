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
c.eq("NULL-safe equality", q("SELECT 1 <=> 1, NULL <=> NULL, 1 <=> NULL, 1 IS NOT DISTINCT FROM 1, NULL IS DISTINCT FROM NULL, 2 IS DISTINCT FROM NULL"),
     [(1, 1, 0, 1, 0, 1)])
c.eq("REGEXP / RLIKE operators", q("SELECT 'video one' REGEXP 'video|music', 'news' NOT REGEXP '^n', 'abc' RLIKE 'b'"), [(1, 0, 1)])
c.eq("UNION drops duplicates", q("SELECT COUNT(*) FROM (SELECT 1 x UNION SELECT 1 UNION SELECT 2 UNION ALL SELECT 2) t"), [(3,)])
c.eq("bitwise operators", q("SELECT 5 & 3, 5 | 3, 5 ^ 3, 1 << 4, 256 >> 2, 1 + 2 << 1, 6 & 3 | 8"), [(1, 7, 6, 16, 64, 6, 10)])
c.eq("GREATEST / LEAST on strings", q("SELECT GREATEST('apple', 'pear', 'banana'), LEAST('b', 'a', 'c'), GREATEST('a', NULL)"), [("pear", "a", None)])
c.eq("STR_TO_DATE", q("SELECT STR_TO_DATE('05/03/2024', '%d/%m/%Y'), STR_TO_DATE('Mar 5 2024 2:07 PM', '%b %e %Y %l:%i %p')"),
     [(datetime.date(2024, 3, 5), datetime.datetime(2024, 3, 5, 14, 7))])
c.eq("FIND_IN_SET / UNHEX / INSERT", q("SELECT FIND_IN_SET('b', 'a,b,c'), UNHEX('4D7953514C'), INSERT('Quadratic', 3, 4, 'What')"), [(2, "MySQL", "QuWhattic")])
c.eq("EXTRACT compound units", q("SELECT EXTRACT(YEAR_MONTH FROM '2024-03-05 10:20:30'), EXTRACT(DAY_SECOND FROM TIMESTAMP '2024-03-05 10:20:30')"),
     [(202403, 5102030)])
c.eq("time functions", q("SELECT TIMEDIFF('2024-03-05 10:20:30', '2024-03-05 09:00:00'), SEC_TO_TIME(3661), TIME_TO_SEC('01:01:01'), MAKEDATE(2024, 60)"),
     [("01:20:30", "01:01:01", 3661, datetime.date(2024, 2, 29))])
c.eq("JSON edits and inspection", q("""SELECT JSON_SET('{"a":1}', '$.b', 2, '$.a', JSON_ARRAY(1)), JSON_REMOVE('{"a":1,"b":{"c":2,"d":3}}', '$.b.c'),
     JSON_TYPE('[1]'), JSON_CONTAINS_PATH('{"o":{"z":1}}', 'one', '$.o.z'), JSON_KEYS('{"k":1,"o":2}')"""),
     [('{"a":[1], "b": 2}', '{"a":1,"b":{"d":3}}', "ARRAY", 1, '["k", "o"]')])
c.eq("BIN / CONV / CHAR / INET_ATON", q("SELECT BIN(10), CONV('ff', 16, 10), CHAR(72, 105), INET_ATON('10.0.0.1'), INET_NTOA(167772161)"),
     [("1010", "255", "Hi", 167772161, "10.0.0.1")])
c.eq("row values, LIKE ESCAPE, POSITION", q("SELECT (1, 'a') IN ((2, 'b'), (1, 'a')), 'a_c' LIKE 'a!_c' ESCAPE '!', 'abc' LIKE 'a!_c' ESCAPE '!', POSITION('c' IN 'abc')"),
     [(1, 1, 0, 3)])
c.eq("named window", q("SELECT x, ROW_NUMBER() OVER w, SUM(x) OVER w FROM (SELECT 1 x UNION ALL SELECT 2) t WINDOW w AS (ORDER BY x)"), [(1, 1, 1), (2, 2, 3)])
q("SELECT SQL_CALC_FOUND_ROWS x FROM (SELECT 1 x UNION ALL SELECT 2 UNION ALL SELECT 3) t LIMIT 1")
c.eq("SQL_CALC_FOUND_ROWS / FOUND_ROWS()", q("SELECT FOUND_ROWS()"), [(3,)])
c.eq("ROLLUP keeps the key type", q("SELECT x, COUNT(*) FROM (SELECT 1 x UNION ALL SELECT 2) t GROUP BY x WITH ROLLUP"), [(1, 1), (2, 1), (None, 2)])
cur.execute("CREATE TABLE longs (id INT, s LONGTEXT)")
cur.execute("INSERT INTO longs VALUES (1, %s), (2, %s), (3, %s)", ("x" * 70000 + "end", "y" * 65535, "z" * 65534))
c.eq("strings of 64 KB and more", q("SELECT id, LENGTH(s), RIGHT(s, 3) FROM longs ORDER BY id"), [(1, 70003, "end"), (2, 65535, "yyy"), (3, 65534, "zzz")])
c.done()
