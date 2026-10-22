# Column-level UPDATE: rows in segments keep their places and only the SET columns get new versions
# (Segment::ov). Random UPDATEs / INSERTs / DELETEs against a Python model, with rows both in segments
# and in write buffers; keys, indexes and the sort key (which go through delete + re-insert) mixed in.
# Then: a slow reader keeps its snapshot, segment elimination after values leave the old range, NULLs,
# SNAPSHOT (format 6) + kill -9 + replay, clean restart, OPTIMIZE, and cold columns (disk tier).
import datetime, os, random, sys, threading, time
from decimal import Decimal
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import Checker, Server

N = 200_000
R = random.Random(11)
c = Checker()
srv = Server("--partitions", "4", name="colupdate", http=False).start()
cur = srv.cursor()


def q(sql):
    cur.execute(sql)
    return [tuple(r) for r in cur.fetchall()]


D0 = datetime.date(2020, 1, 1)
cur.execute("CREATE DATABASE u")
cur.execute("USE u")
cur.execute("CREATE TABLE t (id BIGINT PRIMARY KEY, k INT, v BIGINT, d DECIMAL(12,2), s VARCHAR(20), f DOUBLE, ts DATE, KEY (k), SORT KEY (ts))")
cur.execute("CREATE TABLE h (a INT, b BIGINT, s VARCHAR(20))")  # no keys
model = {}
hm = []
for b in range(0, N, 20_000):
    rows = []
    for i in range(b, b + 20_000):
        r = (i % 997, i * 3, Decimal(i % 5000) / 100, f"s{i % 300}", (i % 64) / 2, D0 + datetime.timedelta(days=i % 900))
        model[i] = r
        rows.append(f"({i}, {r[0]}, {r[1]}, {r[2]}, '{r[3]}', {r[4]}, '{r[5]}')")
    cur.execute("INSERT INTO t VALUES " + ",".join(rows))
    cur.execute("INSERT INTO h VALUES " + ",".join(f"({i % 101}, {i}, 'h{i % 7}')" for i in range(b, b + 20_000)))
    hm += [[i % 101, i, f"h{i % 7}"] for i in range(b, b + 20_000)]
cur.execute("OPTIMIZE TABLE t")
cur.fetchall()
cur.execute("OPTIMIZE TABLE h")
cur.fetchall()
nextId = N


def dump():
    return q("SELECT id, k, v, d, s, f, ts FROM t ORDER BY id")


def want():
    return [(i,) + tuple(r) for i, r in sorted(model.items())]


def hdump():
    return sorted(q("SELECT a, b, s FROM h"))


def setm(i, **kw):
    k, v, d, s, f, ts = model[i]
    loc = dict(k=k, v=v, d=d, s=s, f=f, ts=ts)
    loc.update(kw)
    model[i] = (loc["k"], loc["v"], loc["d"], loc["s"], loc["f"], loc["ts"])


def op():
    global nextId
    x = R.random()
    ids = list(model)
    if x < 0.15:
        lim = R.randint(0, 997)
        add = R.randint(1, 9)
        cur.execute(f"UPDATE t SET v = v + {add} WHERE k < {lim}")
        for i, r in model.items():
            if r[0] < lim:
                setm(i, v=r[1] + add)
    elif x < 0.27:
        m, rr = R.randint(2, 9), R.randint(0, 1)
        sv = f"x{R.randint(0, 50)}"
        cur.execute(f"UPDATE t SET s = '{sv}' WHERE id % {m} = {rr}")
        for i in model:
            if i % m == rr:
                setm(i, s=sv)
    elif x < 0.35:
        a = R.choice(ids)
        b = a + R.randint(0, 30000)
        cur.execute(f"UPDATE t SET d = NULL WHERE id BETWEEN {a} AND {b}")
        for i in model:
            if a <= i <= b:
                setm(i, d=None)
    elif x < 0.45:
        sv = f"s{R.randint(0, 299)}"
        cur.execute(f"UPDATE t SET f = f * 2, v = v - 1 WHERE s = '{sv}'")
        for i, r in model.items():
            if r[3] == sv:
                setm(i, f=r[4] * 2, v=r[1] - 1)
    elif x < 0.52:  # chained SETs: f reads the new v
        lo = R.choice(ids)
        cur.execute(f"UPDATE t SET v = v + 1, f = v WHERE id >= {lo} AND id < {lo + 5000}")
        for i, r in model.items():
            if lo <= i < lo + 5000:
                setm(i, v=r[1] + 1, f=float(r[1] + 1))
    elif x < 0.58:  # an indexed column: delete + re-insert
        lo = R.choice(ids)
        cur.execute(f"UPDATE t SET k = k + 1, v = 7 WHERE id >= {lo} AND id < {lo + 3000}")
        for i, r in model.items():
            if lo <= i < lo + 3000:
                setm(i, k=r[0] + 1, v=7)
    elif x < 0.62:  # the sort key: delete + re-insert
        lo = R.choice(ids)
        cur.execute(f"UPDATE t SET ts = '2030-01-01', d = 1.25 WHERE id >= {lo} AND id < {lo + 2000}")
        for i in model:
            if lo <= i < lo + 2000:
                setm(i, ts=datetime.date(2030, 1, 1), d=Decimal("1.25"))
    elif x < 0.72:  # rows into the write buffers
        n = R.choice([1, 50, 3000])
        rows = []
        for i in range(nextId, nextId + n):
            r = (i % 997, i, Decimal(i % 100), f"n{i % 13}", 0.5, D0)
            model[i] = r
            rows.append(f"({i}, {r[0]}, {r[1]}, {r[2]}, '{r[3]}', {r[4]}, '{r[5]}')")
        nextId += n
        cur.execute("INSERT INTO t VALUES " + ",".join(rows))
    elif x < 0.8:
        lo = R.choice(ids)
        cur.execute(f"DELETE FROM t WHERE id >= {lo} AND id < {lo + 4000}")
        for i in [i for i in model if lo <= i < lo + 4000]:
            del model[i]
    elif x < 0.9:  # keyless table: columns of segment rows replaced, buffer rows copied
        a = R.randint(0, 100)
        cur.execute(f"UPDATE h SET b = b + 1, s = 'z' WHERE a = {a}")
        for r in hm:
            if r[0] == a:
                r[1] += 1
                r[2] = "z"
        if R.random() < 0.5:
            cur.execute(f"INSERT INTO h VALUES ({a}, -1, 'new')")
            hm.append([a, -1, "new"])
    else:
        i = R.choice(ids)
        cur.execute(f"UPDATE t SET v = 99999999999, s = 'big' WHERE id = {i}")
        setm(i, v=99999999999, s="big")


def hwant():
    return sorted(tuple(r) for r in hm)


for step in range(60):
    op()
    if step % 15 == 14:
        c.eq(f"after {step + 1} random statements", dump(), want())
        c.eq(f"keyless table after {step + 1} statements", hdump(), hwant())
c.check("UPDATEs replaced segment columns in place (no dead rows from them)",
        sum(int(r[4]) for r in q("SHOW SEGMENTS FROM h") if r[1] != "-1") < 50_000, q("SHOW SEGMENTS FROM h"))

# keys, indexes and segment elimination still right
i = R.choice(list(model))
c.eq("primary key lookup", q(f"SELECT id, v, s FROM t WHERE id = {i}"), [(i, model[i][1], model[i][3])])
kk = model[i][0]
c.eq("index lookup", q(f"SELECT COUNT(*), SUM(v) FROM t WHERE k = {kk}"),
     [(sum(1 for r in model.values() if r[0] == kk), Decimal(sum(r[1] for r in model.values() if r[0] == kk)))])
cur.execute("UPDATE t SET v = 123456789012 WHERE id = 17")
setm(17, v=123456789012)
c.eq("a value outside the segment's old range is found", q("SELECT id FROM t WHERE v > 100000000000 AND id = 17"), [(17,)])
cur.execute("UPDATE t SET f = NULL WHERE id % 1000 = 3")
for j in model:
    if j % 1000 == 3:
        setm(j, f=None)
c.eq("NULLs written into a column without them", q("SELECT COUNT(*) FROM t WHERE f IS NULL"), [(sum(1 for r in model.values() if r[4] is None),)])

# the fused path's way out: an overflow and a NULL into NOT NULL are reported, nothing changes
before = dump()
c.raises("v + k out of range: refused", lambda: cur.execute("UPDATE t SET v = v + 9223372036854775000"), 1690)
cur.execute("CREATE TABLE nn (a BIGINT NOT NULL, b BIGINT, SORT KEY (a))")
cur.execute("INSERT INTO nn VALUES " + ",".join(f"({i}, {'NULL' if i % 3 == 0 else i})" for i in range(60000)))
cur.execute("OPTIMIZE TABLE nn")
cur.fetchall()
c.raises("NULL copied into NOT NULL: refused", lambda: cur.execute("UPDATE nn SET a = b"), 1048)
c.eq("nothing changed by the refused UPDATEs", (dump(), q("SELECT COUNT(*), SUM(a), COUNT(b) FROM nn")), (before, [(60000, Decimal(sum(range(60000))), 40000)]))
cur.execute("UPDATE nn SET b = a, a = a + 1")
c.eq("fused copy + add", q("SELECT COUNT(*), SUM(a), SUM(b) FROM nn"), [(60000, Decimal(sum(range(60000)) + 60000), Decimal(sum(range(60000))))])

# a refused UPDATE (duplicate key) changes nothing: checked before any row is deleted
cur.execute("CREATE TABLE uk (id INT PRIMARY KEY, v INT, UNIQUE KEY (v))")
cur.execute("INSERT INTO uk VALUES " + ",".join(f"({i}, {i})" for i in range(10)))
for sql in ["UPDATE uk SET id = 5 WHERE id IN (1, 2)", "UPDATE uk SET v = 3 WHERE id = 4", "UPDATE uk SET v = v + 1 WHERE id < 5"]:
    c.raises(f"duplicate key refused: {sql}", lambda: cur.execute(sql), 1062)
    c.eq(f"nothing changed: {sql}", q("SELECT COUNT(*), SUM(id), SUM(v) FROM uk"), [(10, Decimal(45), Decimal(45))])
cur.execute("UPDATE uk SET id = id + 100")
cur.execute("UPDATE uk SET v = NULL WHERE id < 103")
c.eq("key shift and NULLs in a UNIQUE column", q("SELECT COUNT(*), SUM(id), SUM(v), COUNT(v) FROM uk"), [(10, Decimal(1045), Decimal(42), 7)])

# a slow reader keeps the version it started with
before = q("SELECT SUM(v) FROM t")
out = {}


def slow():
    r = srv.cursor("u")
    r.execute("SET warpstore_debug_scan_delay_ms = 1500")
    r.execute("SELECT SUM(v) FROM t")
    out["r"] = [tuple(x) for x in r.fetchall()]


th = threading.Thread(target=slow)
th.start()
time.sleep(0.4)
t0 = time.time()
cur.execute("UPDATE t SET v = v + 1000")
for j in model:
    setm(j, v=model[j][1] + 1000)
c.check("the UPDATE does not wait for the slow reader", time.time() - t0 < 1.0, time.time() - t0)
th.join()
c.eq("the slow reader sees the values from before the UPDATE", out["r"], before)
c.eq("after the UPDATE of every row", dump(), want())

# snapshot (format 6) + more updates + kill -9: replayed on top
cur.execute("SNAPSHOT")
for _ in range(8):
    op()
w = want()
srv.kill()
srv.start()
cur = srv.cursor("u")
c.eq("after kill -9 (snapshot + WAL replay)", dump(), w)
c.eq("keyless table after kill -9", hdump(), hwant())
for _ in range(5):
    op()
cur.execute("SNAPSHOT")
w = want()
srv.stop()
srv.start()
cur = srv.cursor("u")
c.eq("after a clean restart", dump(), w)
for _ in range(5):
    op()
cur.execute("OPTIMIZE TABLE t")
cur.fetchall()
c.eq("after OPTIMIZE", dump(), want())
srv.stop()

# disk tier: UPDATE columns that are only in the segment files
srv2 = Server("--memory-limit", "0.03", "--partitions", "1", name="colupdate-tier").start()
cur = srv2.cursor()
cur.execute("CREATE DATABASE u")
cur.execute("USE u")
cur.execute("CREATE TABLE t (id BIGINT, a BIGINT, b BIGINT, c BIGINT, SORT KEY (id))")
M = 3_000_000
for b in range(0, M, 500_000):
    cur.execute("INSERT INTO t VALUES " + ",".join(f"({i}, {i % 100}, {i}, {i * 2})" for i in range(b, b + 500_000)))
cur.execute("OPTIMIZE TABLE t")
cur.fetchall()
t0 = time.time()
while time.time() - t0 < 30 and "disk-columnstore" not in {r[2] for r in q("SHOW SEGMENTS FROM t")}:
    time.sleep(0.5)
c.check("columns evicted to the segment files", "disk-columnstore" in {r[2] for r in q("SHOW SEGMENTS FROM t")})
cur.execute("UPDATE t SET b = b + a, c = 5 WHERE a < 50")
exp = [(M, sum(i % 100 for i in range(M)), sum(i + (i % 100 if i % 100 < 50 else 0) for i in range(M)),
        sum(5 if i % 100 < 50 else i * 2 for i in range(M)))]
exp = [tuple(Decimal(x) if k else x for k, x in enumerate(exp[0]))]
c.eq("UPDATE of cold columns", q("SELECT COUNT(*), SUM(a), SUM(b), SUM(c) FROM t"), exp)
cur.execute("SNAPSHOT")
cur.execute("UPDATE t SET c = c + 1 WHERE a = 0")
exp = [(exp[0][0], exp[0][1], exp[0][2], exp[0][3] + M // 100)]
srv2.kill()
srv2.start()
cur = srv2.cursor("u")
c.eq("cold columns after kill -9", q("SELECT COUNT(*), SUM(a), SUM(b), SUM(c) FROM t"), exp)
srv2.stop()
c.done()
