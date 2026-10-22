# Strict SQL mode (the default sql_mode, as MySQL 8): a value that doesn't fit its column type is an
# error and the statement changes nothing - integer ranges (TINYINT .. BIGINT, UNSIGNED), DECIMAL
# precision, VARCHAR / CHAR length in characters, TEXT bytes, ENUM members, non-numeric strings into
# integers - for INSERT (rows and columnar INSERT .. SELECT), UPDATE (row by row, vectorized and
# in place) and ALTER TABLE. SET sql_mode = '' keeps values as given; LOAD DATA is not strict (as
# MySQL's LOAD DATA LOCAL); what a non-strict session wrote replays after kill -9.
import os, sys
from decimal import Decimal
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import TMP, Checker, Server

c = Checker()
srv = Server(name="strict", http=False).start()
cur = srv.cursor()


def q(sql):
    cur.execute(sql)
    return [tuple(r) for r in cur.fetchall()]


cur.execute("CREATE DATABASE s")
cur.execute("USE s")
c.eq("default sql_mode is strict", q("SELECT @@sql_mode")[0][0].count("STRICT_TRANS_TABLES"), 1)
cur.execute("CREATE TABLE t (id INT PRIMARY KEY, ti TINYINT, u INT UNSIGNED, s VARCHAR(3), d DECIMAL(4,1), e ENUM('a','b'), tx TINYTEXT)")
cur.execute("INSERT INTO t VALUES (1, 127, 4000000000, 'abc', 999.9, 'a', 'x')")
cur.execute("INSERT INTO t VALUES (2, -128, 0, 'äöü', -999.9, 'B', NULL)")
c.eq("values at the limits; ENUM keeps the member's spelling; INT UNSIGNED above 2^31",
     q("SELECT id, ti, u, s, d, e FROM t ORDER BY id"), [(1, 127, 4000000000, "abc", Decimal("999.9"), "a"), (2, -128, 0, "äöü", Decimal("-999.9"), "b")])
before = q("SELECT * FROM t ORDER BY id")
for sql, code in [("INSERT INTO t VALUES (3, 128, 1, 'a', 1, 'a', '')", 1264), ("INSERT INTO t VALUES (3, 1, -1, 'a', 1, 'a', '')", 1264),
                  ("INSERT INTO t VALUES (3, 1, 1, 'abcd', 1, 'a', '')", 1406), ("INSERT INTO t VALUES (3, 1, 1, 'a', 1000, 'a', '')", 1264),
                  ("INSERT INTO t VALUES (3, 1, 1, 'a', 1, 'c', '')", 1265), ("INSERT INTO t VALUES (3, 'x', 1, 'a', 1, 'a', '')", 1366),
                  ("INSERT INTO t VALUES (3, 1, 1, 'a', 1, 'a', REPEAT('x', 256))", 1406),
                  ("INSERT INTO t VALUES (3, 1, 1, 'a', 1, 'a', ''), (4, 300, 1, 'a', 1, 'a', '')", 1264),
                  ("UPDATE t SET ti = ti + 1", 1264), ("UPDATE t SET s = 'toolong'", 1406), ("UPDATE t SET d = d * 2", 1264),
                  ("UPDATE t SET e = 'z'", 1265), ("UPDATE t SET ti = id * 100", 1264)]:
    c.raises(f"refused: {sql}", lambda: cur.execute(sql), code)
c.eq("nothing changed by the refused statements", q("SELECT * FROM t ORDER BY id"), before)

# columnar INSERT .. SELECT and the in-place (column version) UPDATE path, on segments
cur.execute("CREATE TABLE big (k INT, v SMALLINT, d DECIMAL(5,2), SORT KEY (k))")
cur.execute("CREATE TABLE src (k INT, v INT)")
cur.execute("INSERT INTO src VALUES " + ",".join(f"({i}, {i % 30000})" for i in range(50000)))
cur.execute("INSERT INTO big SELECT k, v, 1.5 FROM src")
cur.execute("OPTIMIZE TABLE big")
cur.fetchall()
cur.execute("INSERT INTO src VALUES (99999, 40000)")
c.raises("INSERT .. SELECT with a value out of range", lambda: cur.execute("INSERT INTO big SELECT k, v, 0 FROM src"), 1264)
c.raises("in-place UPDATE out of range (SMALLINT)", lambda: cur.execute("UPDATE big SET v = v + 10000"), 1264)
c.raises("in-place UPDATE out of range (DECIMAL precision)", lambda: cur.execute("UPDATE big SET d = d + 999"), 1264)
c.eq("the table is unchanged", q("SELECT COUNT(*), SUM(v), SUM(d) FROM big"),
     [(50000, Decimal(sum(i % 30000 for i in range(50000))), Decimal("75000.00"))])
cur.execute("UPDATE big SET v = v + 100, d = d + 1")
c.eq("an in-place UPDATE that fits", q("SELECT COUNT(*), SUM(v), SUM(d) FROM big"),
     [(50000, Decimal(sum(i % 30000 for i in range(50000)) + 5000000), Decimal("125000.00"))])

# ALTER TABLE to a narrower type over values that don't fit: refused, table unchanged
cur.execute("CREATE TABLE al (id INT PRIMARY KEY, v INT, s VARCHAR(10))")
cur.execute("INSERT INTO al VALUES (1, 300, 'dddddddd'), (2, 5, 'ab')")
c.raises("MODIFY to TINYINT over 300", lambda: cur.execute("ALTER TABLE al MODIFY v TINYINT"), 1264)
c.raises("MODIFY to VARCHAR(2) over longer strings", lambda: cur.execute("ALTER TABLE al MODIFY s VARCHAR(2)"), 1406)
c.eq("table unchanged", q("SELECT * FROM al ORDER BY id"), [(1, 300, "dddddddd"), (2, 5, "ab")])
cur.execute("ALTER TABLE al MODIFY v SMALLINT")
c.eq("MODIFY that fits", q("SELECT SUM(v) FROM al"), [(Decimal(305),)])

# LOAD DATA: not strict (kept as given)
path = os.path.join(TMP, f"strict-{os.getpid()}.tsv")
with open(path, "w") as f:
    f.write("10\t500\tlonger than three\n")
cur.execute("CREATE TABLE ld (id INT, ti TINYINT, s VARCHAR(3))")
cur.execute(f"LOAD DATA INFILE '{path}' INTO TABLE ld")
c.eq("LOAD DATA keeps values as given", q("SELECT id, s FROM ld"), [(10, "longer than three")])

# a non-strict session; what it wrote replays after kill -9
cur.execute("SET sql_mode = ''")
c.eq("sql_mode set", q("SELECT @@sql_mode"), [("",)])
cur.execute("INSERT INTO t VALUES (5, 1, 1, 'abcdef', 1, 'a', '')")
cur.execute("SET sql_mode = DEFAULT")
c.raises("strict again after SET sql_mode = DEFAULT", lambda: cur.execute("INSERT INTO t VALUES (6, 1, 1, 'abcdef', 1, 'a', '')"), 1406)
srv.kill()
srv.start()
cur = srv.cursor("s")
c.eq("after kill -9: the non-strict row replayed", q("SELECT s FROM t WHERE id = 5"), [("abcdef",)])
srv.stop()
c.done()
