# Background compaction: small segments and segments full of deleted rows are merged; primary-key
# and secondary-index lookups, sort order, deletes made while it runs, and a kill -9 all stay right.
import os, random, sys, threading, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import Checker, Server

os.environ["WARPSTORE_TRACE"] = "1"
c = Checker()
srv = Server("--partitions", "2", name="compact", http=False).start()
cur = srv.cursor()


def q(sql):
    cur.execute(sql)
    return [tuple(r) for r in cur.fetchall()]


def segments(t):
    return [r for r in q(f"SHOW SEGMENTS FROM {t}") if r[1] != "-1"]


cur.execute("CREATE DATABASE k")
cur.execute("USE k")
cur.execute("CREATE TABLE t (id INT PRIMARY KEY, g INT, s VARCHAR(20), v BIGINT, KEY (g), SORT KEY (v))")
cur.execute("CREATE TABLE h (a INT, s VARCHAR(20))")  # no keys: bulk paths
rows = {}
R = random.Random(5)
for b in range(12):  # 12 flushes of 20K rows: small segments
    vals = []
    for i in range(b * 20000, (b + 1) * 20000):
        rows[i] = (i % 97, f"s{i % 501}", R.randint(-10**9, 10**9))
        vals.append(f"({i}, {rows[i][0]}, '{rows[i][1]}', {rows[i][2]})")
    cur.execute("INSERT INTO t VALUES " + ",".join(vals))
    cur.execute("INSERT INTO h VALUES " + ",".join(f"({i}, 'h{i % 13}')" for i in range(b * 20000, (b + 1) * 20000)))
    q("OPTIMIZE TABLE t")
    q("OPTIMIZE TABLE h")
cur.execute("DELETE FROM t WHERE id % 3 = 0")
cur.execute("DELETE FROM h WHERE a % 2 = 0")
for i in list(rows):
    if i % 3 == 0:
        del rows[i]
cur.execute("UPDATE t SET v = v + 1 WHERE id % 10 = 1")
for i in rows:
    if i % 10 == 1:
        rows[i] = (rows[i][0], rows[i][1], rows[i][2] + 1)
before_t, before_h = len(segments("t")), len(segments("h"))

# deletes while compaction runs
stop = False


def churn():
    c2 = srv.cursor("k")
    for i in range(1, 240000, 1000):
        if stop:
            break
        c2.execute(f"DELETE FROM t WHERE id = {i + 1}")
        time.sleep(0.02)


th = threading.Thread(target=churn)
th.start()
deadline = time.time() + 60  # (background compaction; slow on a loaded machine)
while time.time() < deadline and (len(segments("t")) >= before_t or len(segments("h")) >= before_h):
    time.sleep(0.5)
th.join()
for i in range(1, 240000, 1000):
    rows.pop(i + 1, None)
after_t, after_h = len(segments("t")), len(segments("h"))
c.check("segments were merged", after_t < before_t and after_h < before_h, (before_t, after_t, before_h, after_h))
log = open(srv.log).read()
c.check("compaction ran", "compacted" in log)


def check_all(label):
    c.eq(f"{label}: count / sums", q("SELECT COUNT(*), SUM(g), SUM(v), COUNT(DISTINCT s) FROM t"),
         [(len(rows), sum(r[0] for r in rows.values()), sum(r[2] for r in rows.values()), len({r[1] for r in rows.values()}))])
    ids = R.sample(sorted(rows), 200) + [3, 6, 2, 1002, 1001]
    got = {}
    for i in ids:
        r = q(f"SELECT g, s, v FROM t WHERE id = {i}")
        got[i] = r[0] if r else None
    c.eq(f"{label}: primary key lookups", got, {i: rows.get(i) for i in ids})
    c.eq(f"{label}: secondary index lookup", q("SELECT COUNT(*), SUM(v) FROM t WHERE g = 42"),
         [(sum(1 for r in rows.values() if r[0] == 42), sum(r[2] for r in rows.values() if r[0] == 42))])
    c.eq(f"{label}: ORDER BY sort key", q("SELECT id FROM t ORDER BY v, id LIMIT 5"),
         [(i,) for i in sorted(rows, key=lambda i: (rows[i][2], i))[:5]])
    c.eq(f"{label}: keyless table", q("SELECT COUNT(*), SUM(a) FROM h"), [(120000, sum(range(1, 240000, 2)))])


check_all("after compaction")
cur.execute("INSERT INTO t VALUES (3, 1, 'back', 7)")  # a deleted key can be inserted again
rows[3] = (1, "back", 7)
c.raises("duplicate key after compaction", lambda: cur.execute("INSERT INTO t VALUES (4, 1, 'x', 1)"), 1062)
srv.kill()
srv.start()
cur = srv.cursor()
cur.execute("USE k")
check_all("after kill -9")
srv.stop()
c.done()
