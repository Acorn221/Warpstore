# Column-level UPDATEs when VRAM is short (--vram-limit below the table's size): columns on the host
# tier (no GPU-only versions), new versions that don't fit (the host path instead), GPU-only versions
# copied to the host when needed (a SNAPSHOT, a CPU read); values checked, then kill -9 + replay.
import os, sys
from decimal import Decimal
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import Checker, Server

N = 2_000_000
c = Checker()
srv = Server("--vram-limit", "0.03", "--partitions", "2", name="updvram", http=False).start()
cur = srv.cursor()


def q(sql):
    cur.execute(sql)
    return [tuple(r) for r in cur.fetchall()]


cur.execute("CREATE DATABASE v")
cur.execute("USE v")
cur.execute("CREATE TABLE t (k BIGINT, a BIGINT, b BIGINT, SORT KEY (k))")
cur.execute("CREATE TABLE src (k BIGINT)")
cur.execute("INSERT INTO src VALUES " + ",".join(f"({i})" for i in range(20000)))
cur.execute("INSERT INTO t SELECT x.k * 100 + y.k, (x.k * 100 + y.k) % 1000, x.k * 100 + y.k FROM src x JOIN src y ON y.k < 100")
cur.execute("OPTIMIZE TABLE t")
cur.fetchall()
want_a, want_b = sum(i % 1000 for i in range(N)), sum(range(N))
c.eq("loaded", q("SELECT COUNT(*), SUM(a), SUM(b) FROM t"), [(N, Decimal(want_a), Decimal(want_b))])
for r in range(3):
    cur.execute("UPDATE t SET b = b + 1, a = a + 2")
    want_a += 2 * N
    want_b += N
    c.eq(f"after UPDATE {r + 1}", q("SELECT COUNT(*), SUM(a), SUM(b) FROM t"), [(N, Decimal(want_a), Decimal(want_b))])
cur.execute("UPDATE t SET a = 7 WHERE k % 2 = 0")
want_a = sum(7 if i % 2 == 0 else i % 1000 + 6 for i in range(N))
c.eq("CPU reads of updated columns (point and filtered)", q("SELECT a, b FROM t WHERE k = 12345"), [(12345 % 1000 + 6, 12345 + 3)])
c.eq("after a partial UPDATE", q("SELECT SUM(a) FROM t"), [(Decimal(want_a),)])
cur.execute("SNAPSHOT")
cur.execute("UPDATE t SET b = b - 3")
want_b -= 3 * N
srv.kill()
srv.start()
cur = srv.cursor("v")
c.eq("after kill -9 (snapshot + replay)", q("SELECT COUNT(*), SUM(a), SUM(b) FROM t"), [(N, Decimal(want_a), Decimal(want_b))])
c.eq("a row", q("SELECT a, b FROM t WHERE k = 12345"), [(12345 % 1000 + 6, 12345)])
srv.stop()
srv.start()
cur = srv.cursor("v")
c.eq("after a clean restart", q("SELECT COUNT(*), SUM(a), SUM(b) FROM t"), [(N, Decimal(want_a), Decimal(want_b))])
srv.stop()
c.done()
