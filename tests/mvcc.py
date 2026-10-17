# MVCC: readers work on per-statement snapshots, so
#   * writers don't wait for a slow reader (and its result is the state when it started),
#   * a reader sees each statement's state consistently while transfers run (sum invariant),
#   * flushes (OPTIMIZE), DROP TABLE and string dictionary growth under a running reader are safe.
# A slow reader is simulated with SET gpudb_debug_scan_delay_ms (it sleeps before each table scan).
import os, random, sys, threading, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pymysql
from lib import Checker, Server

c = Checker()
srv = Server(name="mvcc", http=False).start()
cur = srv.cursor()
cur.execute("CREATE DATABASE m")
cur.execute("USE m")
cur.execute("CREATE TABLE acct (id INT PRIMARY KEY, v BIGINT, tag VARCHAR(20))")
cur.execute("INSERT INTO acct VALUES " + ",".join(f"({i}, 1000, 't{i % 10}')" for i in range(1000)))
cur.execute("OPTIMIZE TABLE acct")  # rows in a segment and in the write buffer
cur.execute("INSERT INTO acct VALUES " + ",".join(f"({i}, 1000, 't{i % 10}')" for i in range(1000, 2000)))
TOTAL = 2000 * 1000


def slow(sql, delay, out, key):
    r = srv.cursor("m")
    r.execute(f"SET gpudb_debug_scan_delay_ms = {delay}")
    t = time.time()
    r.execute(sql)
    out[key] = (r.fetchall(), time.time() - t)


# 1. writers don't wait for a slow reader, which sees the state when it started
out = {}
th = threading.Thread(target=slow, args=("SELECT COUNT(*), SUM(v) FROM acct", 1500, out, "r"))
th.start()
time.sleep(0.3)
w = srv.cursor("m")
t = time.time()
for i in range(200):
    w.execute(f"INSERT INTO acct VALUES ({5000 + i}, 1, 'new')")
w.execute("UPDATE acct SET v = v + 1 WHERE id < 100")
w.execute("DELETE FROM acct WHERE id BETWEEN 100 AND 199")
wt = time.time() - t
th.join()
print(f"200 inserts + update + delete during a 1.5 s read: {wt * 1000:.0f} ms; reader {out['r'][1]:.2f}s")
c.check("writers don't wait for the reader", wt < 1.0, f"{wt:.2f}s")
c.eq("the reader sees the state when it started", out["r"][0], ((2000, TOTAL),))
c.eq("new statements see the writes", cur.execute("SELECT COUNT(*), SUM(v) FROM acct") and cur.fetchall(), ((2000 + 200 - 100, TOTAL + 200 + 100 - 100 * 1000),))

# 2. sum invariant under concurrent transfers (fast and slow readers)
cur.execute("DELETE FROM acct WHERE id >= 5000")
cur.execute("UPDATE acct SET v = 1000")
cur.execute("INSERT INTO acct SELECT id, 1000, 't' FROM acct WHERE id < 0")
cur.execute("SELECT COUNT(*), SUM(v) FROM acct")
n0, s0 = cur.fetchone()
stop = threading.Event()
errors = []


def transfers():
    k = srv.connect("m", autocommit=False)
    kc = k.cursor()
    rnd = random.Random(1)
    ids = [r for r in range(2000) if not 100 <= r < 200]  # (100..199 were deleted above)
    while not stop.is_set():
        a, b = rnd.sample(ids, 2)
        try:
            kc.execute("BEGIN")
            kc.execute(f"UPDATE acct SET v = v - 7 WHERE id = {a}")
            kc.execute(f"UPDATE acct SET v = v + 7 WHERE id = {b}")
            k.commit()
        except pymysql.MySQLError as e:
            k.rollback()


def reader(delay, sums):
    r = srv.cursor("m")
    if delay:
        r.execute(f"SET gpudb_debug_scan_delay_ms = {delay}")
    while not stop.is_set():
        r.execute("SELECT SUM(v), COUNT(*) FROM acct")
        sums.append(tuple(r.fetchone()))
        r.execute("SELECT COUNT(*) FROM acct a JOIN acct b ON a.id = b.id")  # self-join: one snapshot for both sides
        sums.append(("self", r.fetchone()[0]))


ths = [threading.Thread(target=transfers) for _ in range(3)]
fast, slowS = [], []
ths += [threading.Thread(target=reader, args=(0, fast)), threading.Thread(target=reader, args=(50, slowS))]
for x in ths:
    x.start()
time.sleep(4)
cur.execute("OPTIMIZE TABLE acct")  # a flush while readers hold snapshots
time.sleep(2)
stop.set()
for x in ths:
    x.join()
bad = [x for x in fast + slowS if x != (s0, n0) and x != ("self", n0)]
print(f"{len(fast) + len(slowS)} reads during transfers, {len(bad)} inconsistent")
c.check("every read sees a consistent state", not bad and len(fast) > 20, str(bad[:5]))

# 3. DROP TABLE and dictionary growth under a running reader
cur.execute("CREATE TABLE tmp (id INT, s VARCHAR(30))")
cur.execute("INSERT INTO tmp VALUES " + ",".join(f"({i}, 'v{i}')" for i in range(5000)))
out = {}
th = threading.Thread(target=slow, args=("SELECT COUNT(DISTINCT s), MAX(s) FROM tmp", 800, out, "r"))
th.start()
time.sleep(0.2)
for k in range(20):  # thousands of new strings while the reader decodes the dictionary
    w.execute("INSERT INTO tmp VALUES " + ",".join(f"({i}, 'new{i}x{k}')" for i in range(500)))
w.execute("DROP TABLE tmp")
th.join()
c.eq("reader of a dropped table finishes on its snapshot", out["r"][0], ((5000, "v999"),))
srv.stop()
c.done()
