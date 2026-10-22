# A full disk and a failed fdatasync. Full disk (SET warpstore_debug_wal_full = 1: every WAL
# extension fails with ENOSPC): an autocommit write fails before it changes anything and a COMMIT
# rolls its transaction back, the server stays writable once there is room again, and a restart
# has exactly the acknowledged rows. A failed fdatasync (SET warpstore_debug_sync_fail = 1) under sync
# durability: the commit is not acknowledged and writes are refused until a restart.
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pymysql
from lib import Checker, Server

c = Checker()
srv = Server(name="diskfull", http=False).start()
cur = srv.cursor()


def q(sql):
    cur.execute(sql)
    return [tuple(r) for r in cur.fetchall()]


cur.execute("CREATE DATABASE f")
cur.execute("USE f")
cur.execute("CREATE TABLE t (id INT PRIMARY KEY, v INT)")
cur.execute("INSERT INTO t VALUES " + ",".join(f"({i}, {i})" for i in range(100)))

cur.execute("SET warpstore_debug_wal_full = 1")
for sql in ["INSERT INTO t VALUES (1000, 1)", "UPDATE t SET v = v + 1", "DELETE FROM t WHERE id < 50"]:
    try:
        cur.execute(sql)
        c.check(f"disk full: {sql.split()[0]} refused", False, "accepted")
    except pymysql.err.MySQLError as e:
        c.check(f"disk full: {sql.split()[0]} refused", e.args[0] == 1021, e)
c.eq("disk full: nothing was applied", q("SELECT COUNT(*), SUM(v) FROM t"), [(100, 4950)])
cur.execute("BEGIN")
cur.execute("INSERT INTO t VALUES (2000, 1)")
cur.execute("UPDATE t SET v = 0 WHERE id = 5")
try:
    cur.execute("COMMIT")
    c.check("disk full: COMMIT refused", False, "committed")
except pymysql.err.MySQLError as e:
    c.check("disk full: COMMIT refused", e.args[0] == 1021, e)
c.eq("disk full: the transaction rolled back", q("SELECT COUNT(*), SUM(v) FROM t"), [(100, 4950)])
cur.execute("SET warpstore_debug_wal_full = 0")
cur.execute("INSERT INTO t VALUES (3000, 7)")
c.eq("room again: writes work (not read-only)", q("SELECT COUNT(*), SUM(v) FROM t"), [(101, 4957)])
srv.kill()
srv.start()
cur = srv.cursor("f")
c.eq("after kill -9: exactly the acknowledged rows", q("SELECT COUNT(*), SUM(v) FROM t"), [(101, 4957)])

cur.execute("SET warpstore_sync_durability = ON")
cur.execute("SET warpstore_debug_sync_fail = 1")
try:
    cur.execute("INSERT INTO t VALUES (4000, 1)")
    c.check("fdatasync failed: the commit is not acknowledged", False, "acknowledged")
except pymysql.err.MySQLError as e:
    c.check("fdatasync failed: the commit is not acknowledged", e.args[0] == 1021, e)
cur.execute("SET warpstore_debug_sync_fail = 0")
try:
    cur.execute("INSERT INTO t VALUES (4001, 1)")
    c.check("after a failed fdatasync writes are refused", False, "accepted")
except pymysql.err.MySQLError as e:
    c.check("after a failed fdatasync writes are refused", e.args[0] == 1290, e)
c.eq("reads still work", q("SELECT COUNT(*) FROM t WHERE id < 4000"), [(101,)])
srv.kill()
srv.start()
cur = srv.cursor("f")
n = q("SELECT COUNT(*) FROM t")[0][0]
c.check("restart: the acknowledged rows, the unacknowledged one maybe", n in (101, 102), n)
cur.execute("INSERT INTO t VALUES (5000, 1)")
c.eq("restart: writable again", q("SELECT COUNT(*) FROM t WHERE id = 5000"), [(1,)])
srv.stop()
c.done()
