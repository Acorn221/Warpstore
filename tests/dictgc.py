# Dictionary garbage: dictionaries only grow, so strings of deleted / updated rows stay. OPTIMIZE
# TABLE rebuilds a table whose dictionaries are mostly unused strings; contents, keys, indexes and
# AUTO_INCREMENT are unchanged, and the rebuild survives a kill -9.
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import Checker, Server

c = Checker()
srv = Server(name="dictgc", http=False).start()
cur = srv.cursor()


def q(sql):
    cur.execute(sql)
    return [tuple(r) for r in cur.fetchall()]


cur.execute("CREATE DATABASE g")
cur.execute("USE g")
cur.execute("CREATE TABLE t (id INT AUTO_INCREMENT PRIMARY KEY, k INT, s VARCHAR(40), tag VARCHAR(10), KEY (k))")
cur.execute("CREATE TABLE h (s VARCHAR(40), n INT)")  # no keys
for b in range(10):
    cur.execute("INSERT INTO t (k, s, tag) VALUES " + ",".join(f"({i % 97}, 'unique string {i}', 't{i % 5}')" for i in range(b * 5000, (b + 1) * 5000)))
    cur.execute("INSERT INTO h VALUES " + ",".join(f"('h {i}', {i})" for i in range(b * 5000, (b + 1) * 5000)))
cur.execute("DELETE FROM t WHERE id % 5 <> 0")                         # 80% of the strings unused
cur.execute("UPDATE t SET s = CONCAT(s, ' v2') WHERE id % 10 = 0")   # old versions unused too
cur.execute("DELETE FROM h WHERE n % 4 <> 0")

checks = ["SELECT COUNT(*), SUM(id), SUM(k), COUNT(DISTINCT s), MIN(s), MAX(s), COUNT(DISTINCT tag) FROM t",
          "SELECT id, s FROM t WHERE id = 1005", "SELECT COUNT(*), SUM(id) FROM t WHERE k = 42",
          "SELECT s, COUNT(*) FROM t GROUP BY s ORDER BY s LIMIT 3", "SELECT COUNT(*), SUM(n), COUNT(DISTINCT s) FROM h"]
before = [q(x) for x in checks]
r = q("OPTIMIZE TABLE t")
c.check("OPTIMIZE rebuilds a table full of unused strings", "rebuilt" in r[0][3], r)
c.check("OPTIMIZE rebuilds the keyless table too", "rebuilt" in q("OPTIMIZE TABLE h")[0][3])
c.eq("contents after the rebuild", [q(x) for x in checks], before)
cur.execute("INSERT INTO t (k, s, tag) VALUES (1, 'after', 'x')")
c.eq("AUTO_INCREMENT continues where it was", q("SELECT MAX(id) FROM t"), [(50001,)])
c.raises("primary key still enforced", lambda: cur.execute("INSERT INTO t VALUES (5, 1, 'dup', 'x')"), 1062)
after = [q(x) for x in checks]
c.check("a table with few unused strings is not rebuilt", "rebuilt" not in q("OPTIMIZE TABLE t")[0][3])
c.eq("no leftover tables", sorted(r[0] for r in q("SHOW TABLES")), ["h", "t"])
srv.kill()
srv.start()
cur = srv.cursor()
cur.execute("USE g")
c.eq("contents after kill -9 (the rebuild replays)", [q(x) for x in checks], after)
c.eq("tables after kill -9", sorted(r[0] for r in q("SHOW TABLES")), ["h", "t"])
srv.stop()
c.done()
