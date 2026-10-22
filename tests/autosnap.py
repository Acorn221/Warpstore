# Automatic snapshots (--snapshot-trigger-mb): the WAL is checkpointed while writes go on, it stays
# small, and a kill -9 afterwards recovers every committed row.
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import Checker, Server

c = Checker()
srv = Server("--snapshot-trigger-mb", "2", name="autosnap", http=False).start()
cur = srv.cursor()
cur.execute("CREATE DATABASE a")
cur.execute("USE a")
cur.execute("CREATE TABLE t (id INT PRIMARY KEY, s VARCHAR(40), v BIGINT)")
cur.execute("CREATE TABLE u (k INT, s VARCHAR(40))")
n = 0
t0 = time.time()
while time.time() - t0 < 8:
    cur.execute("INSERT INTO t VALUES " + ",".join(f"({n + i}, 'row {n + i} {'x' * (i % 30)}', {(n + i) * 3})" for i in range(500)))
    cur.execute("INSERT INTO u VALUES " + ",".join(f"({i}, 'u{(n + i) % 1000}')" for i in range(500)))
    if n % 5000 == 0:
        cur.execute(f"UPDATE t SET v = v + 1 WHERE id % 7 = 0 AND id >= {n - 5000}")
        cur.execute(f"DELETE FROM u WHERE k % 9 = 0 AND k < 100")
    n += 500
log = open(srv.log).read()
snaps = log.count("automatic snapshot (")
c.check("automatic snapshots ran while writing", snaps >= 2, f"{snaps} snapshots, {n} rows")
cur.execute("SELECT COUNT(*), SUM(v), COUNT(DISTINCT s) FROM t")
want = [tuple(r) for r in cur.fetchall()]
cur.execute("SELECT COUNT(*), COUNT(DISTINCT s) FROM u")
want_u = [tuple(r) for r in cur.fetchall()]
srv.kill()
srv.start()
cur = srv.cursor()
replayed = [l for l in open(srv.log) if "replayed" in l]
c.check("restart replays only the WAL since the last snapshot", replayed and int(replayed[-1].split()[2]) < n // 500, replayed)
cur.execute("USE a")
cur.execute("SELECT COUNT(*), SUM(v), COUNT(DISTINCT s) FROM t")
c.eq("rows after kill -9", [tuple(r) for r in cur.fetchall()], want)
cur.execute("SELECT COUNT(*), COUNT(DISTINCT s) FROM u")
c.eq("second table after kill -9", [tuple(r) for r in cur.fetchall()], want_u)

# Saved collation orders: string order after a restart (read from the order files) is the same.
def orders():
    out = []
    for coll in ("utf8mb4_bin", "utf8mb4_general_ci"):
        cur.execute(f"SET collation_connection = {coll}")
        cur.execute("SELECT MIN(s), MAX(s) FROM t")
        out.append(tuple(cur.fetchall()[0]))
        cur.execute("SELECT s FROM t ORDER BY s DESC, id LIMIT 3")
        out.append(tuple(r[0] for r in cur.fetchall()))
    return out


cur.execute("INSERT INTO t VALUES (-1, 'Zebra', 0), (-2, 'apple', 0), (-3, 'zz top', 0)")
want_orders = orders()
cur.execute("SNAPSHOT")
srv.stop()
srv.start()
cur = srv.cursor()
cur.execute("USE a")
c.eq("string order after a restart (saved orders)", orders(), want_orders)
c.check("saved order files", any(f.endswith(".o0") for f in os.listdir(os.path.join(srv.data_dir, "dicts"))))
cur.execute("SELECT COUNT(*), SUM(v), COUNT(DISTINCT s) FROM t WHERE id >= 0")
c.eq("rows after a restart", [tuple(r) for r in cur.fetchall()], want)

# Dictionary files: a tail left by a snapshot that was never installed is cut off at load, and the
# strings the WAL replays after it are appended in the right place.
cur.execute("SNAPSHOT")
dd = os.path.join(srv.data_dir, "dicts")
for f in os.listdir(dd):
    with open(os.path.join(dd, f), "ab") as fh:
        fh.write(b"WDFR" + os.urandom(200))
cur.execute("INSERT INTO u VALUES " + ",".join(f"({i}, 'tail{i}')" for i in range(300)))
srv.kill()
srv.start()
cur = srv.cursor()
cur.execute("USE a")
cur.execute("SNAPSHOT")
srv.stop()
srv.start()
cur = srv.cursor()
cur.execute("USE a")
cur.execute("SELECT COUNT(*), COUNT(DISTINCT s) FROM u WHERE s LIKE 'tail%'")
c.eq("strings after a torn dictionary tail", [tuple(r) for r in cur.fetchall()], [(300, 300)])
cur.execute("SELECT COUNT(*), SUM(v), COUNT(DISTINCT s) FROM t WHERE id >= 0")
c.eq("rows after a torn dictionary tail", [tuple(r) for r in cur.fetchall()], want)
srv.stop()
c.done()
