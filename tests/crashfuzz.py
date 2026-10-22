# Randomized crash recovery: concurrent writers (inserts, point and range updates, deletes, bulk
# inserts) with automatic snapshots every ~0.2 MB of WAL, killed with SIGKILL at random moments. After
# each restart every writer's rows must equal its model of the acknowledged statements, with or
# without the one statement it had in flight. WARPSTORE_CRASH_ROUNDS (default 6), WARPSTORE_CRASH_SEED.
import os, random, sys, threading, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pymysql
from lib import Checker, Server

ROUNDS = int(os.environ.get("WARPSTORE_CRASH_ROUNDS", "6"))
SEED = int(os.environ.get("WARPSTORE_CRASH_SEED", "7"))
WRITERS = 4
SPAN = 1_000_000  # id range per writer

c = Checker()
srv = Server("--snapshot-trigger-mb", "0.2", name="crashfuzz", http=False).start()
cur = srv.cursor()
cur.execute("CREATE DATABASE cf")
cur.execute("CREATE TABLE cf.t (id BIGINT PRIMARY KEY, w INT, v BIGINT, s VARCHAR(30))")
cur.execute("CREATE TABLE cf.h (w INT, k BIGINT, v BIGINT)")  # no keys: duplicates would show


class Writer:
    def __init__(self, w):
        self.w, self.rows, self.hist, self.next = w, {}, {}, w * SPAN
        self.inflight = None  # (apply function) of the statement in progress
        self.R = random.Random(SEED * 100 + w)

    def ops(self):
        """One random statement: (sql, apply(model))."""
        R, w = self.R, self.w
        ids = list(self.rows)
        r = R.random()
        if r < 0.3 or not ids:
            n = R.choice([1, 1, 5, 50])
            new = list(range(self.next, self.next + n))
            vals = {i: (R.randint(-1000, 1000), f"s{R.randint(0, 99)}") for i in new}
            sql = "INSERT INTO cf.t VALUES " + ",".join(f"({i}, {w}, {v}, '{s}')" for i, (v, s) in vals.items())

            def ap(m, h):
                for i, vs in vals.items():
                    m[i] = vs
            self.next += n
            return sql, ap
        if r < 0.5:
            i, k = R.choice(ids), R.randint(1, 9)
            return f"UPDATE cf.t SET v = v + {k} WHERE id = {i}", lambda m, h: m.__setitem__(i, (m[i][0] + k, m[i][1])) if i in m else None
        if r < 0.62:
            lo = R.choice(ids)
            hi = lo + R.randint(0, 200)
            s = f"u{R.randint(0, 9)}"

            def ap(m, h):
                for i in list(m):
                    if lo <= i <= hi:
                        m[i] = (m[i][0] * 2, s)
            return f"UPDATE cf.t SET v = v * 2, s = '{s}' WHERE id BETWEEN {lo} AND {hi} AND w = {w}", ap
        if r < 0.75:
            i = R.choice(ids)
            return f"DELETE FROM cf.t WHERE id = {i}", lambda m, h: m.pop(i, None)
        if r < 0.82:
            lo = R.choice(ids)
            hi = lo + R.randint(0, 50)

            def ap(m, h):
                for i in list(m):
                    if lo <= i <= hi:
                        del m[i]
            return f"DELETE FROM cf.t WHERE id BETWEEN {lo} AND {hi} AND w = {w}", ap
        k = R.randint(0, 999)
        n = R.choice([1, 20, 300])

        def ap(m, h):
            h[k] = h.get(k, 0) + n
        return f"INSERT INTO cf.h VALUES " + ",".join(f"({w}, {k}, {j})" for j in range(n)), ap

    def run(self, stop):
        try:
            conn = srv.connect()
            cc = conn.cursor()
        except Exception:
            return
        while not stop.is_set():
            sql, ap = self.ops()
            self.inflight = ap
            try:
                cc.execute(sql)
            except pymysql.err.IntegrityError:
                self.inflight = None  # (rejected: nothing changed)
                continue
            except Exception:
                return  # killed: the statement may or may not have been applied
            ap(self.rows, self.hist)
            self.inflight = None


def state(w):
    cur.execute(f"SELECT id, v, s FROM cf.t WHERE w = {w}")
    rows = {r[0]: (r[1], r[2]) for r in cur.fetchall()}
    cur.execute(f"SELECT k, COUNT(*) FROM cf.h WHERE w = {w} GROUP BY k")
    hist = {r[0]: r[1] for r in cur.fetchall()}
    return rows, hist


writers = [Writer(w) for w in range(WRITERS)]
R = random.Random(SEED)
for rnd in range(ROUNDS):
    stop = threading.Event()
    th = [threading.Thread(target=x.run, args=(stop,)) for x in writers]
    for t in th:
        t.start()
    time.sleep(R.uniform(2.0, 6.0))
    srv.kill()
    stop.set()
    for t in th:
        t.join()
    srv.start()
    cur = srv.cursor()
    for x in writers:
        got = state(x.w)
        ok = got == (x.rows, x.hist)
        if not ok and x.inflight:  # the in-flight statement made it
            m, h = dict(x.rows), dict(x.hist)
            x.inflight(m, h)
            if got == (m, h):
                x.rows, x.hist, ok = m, h, True
        x.inflight = None
        c.check(f"round {rnd} writer {x.w}: recovered rows match the acknowledged statements", ok,
                f"{len(got[0])} rows / {sum(got[1].values())} history rows; model {len(x.rows)} / {sum(x.hist.values())}")
        if not ok:
            x.rows, x.hist = got  # (continue from what is there)
import glob
snaps = sum(open(f).read().count("automatic snapshot (") for f in glob.glob(srv.log.rsplit("-", 1)[0] + "-*.log"))
c.check("automatic snapshots ran between the crashes", snaps > 0, snaps)
srv.stop()
c.done()
