# Column-level UPDATE: rows in segments keep their places and only the SET columns get new versions
# (Segment::ov); refused UPDATEs / INSERTs change nothing. Random UPDATEs / INSERTs / DELETEs against
# a Python model, with rows both in segments and in write buffers; keys, indexes and the sort key
# (which go through delete + re-insert) mixed in. Then: a slow reader keeps its snapshot, segment
# elimination after values leave the old range, NULLs, SNAPSHOT (format 6) + kill -9 + replay, clean
# restart, OPTIMIZE, and cold columns (disk tier).
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

# a refused INSERT leaves nothing behind either (MySQL: the statement is atomic)
cur.execute("CREATE TABLE ins (id INT PRIMARY KEY, v INT NOT NULL)")
cur.execute("CREATE TABLE src (id INT, v INT)")
cur.execute("INSERT INTO src VALUES " + ",".join(f"({i}, {'NULL' if i == 50000 else i})" for i in range(60000)))
for sql, code in [("INSERT INTO ins VALUES (1, 1), (2, 2), (1, 3)", 1062), ("INSERT INTO ins VALUES (10, 1), (11, NULL)", 1048),
                  ("INSERT INTO ins SELECT * FROM src", 1048), ("INSERT INTO ins SELECT id, 1 FROM src UNION ALL SELECT 5, 5", 1062)]:
    c.raises(f"refused: {sql}", lambda: cur.execute(sql), code)
    c.eq(f"nothing inserted: {sql}", q("SELECT COUNT(*) FROM ins"), [(0,)])
cur.execute("INSERT INTO ins VALUES (1, 1), (2, 2)")
c.eq("the keys of a refused INSERT are free again", q("SELECT COUNT(*), SUM(id) FROM ins"), [(2, Decimal(3))])

# in a transaction a refused statement is undone on its own (the transaction goes on); a table
# with a VECTOR column (rows rewritten from values) loses nothing either
cur.execute("CREATE TABLE tx (id INT PRIMARY KEY, v INT NOT NULL)")
cur.execute("CREATE TABLE vx (id INT PRIMARY KEY, v INT NOT NULL, e VECTOR(2))")
cur.execute("INSERT INTO tx VALUES (1, 1), (2, 2), (3, 3)")
cur.execute("INSERT INTO vx VALUES (1, 1, '[1,2]'), (2, 2, '[1,2]'), (3, 3, '[1,2]')")
cur.execute("BEGIN")
c.raises("in a transaction: INSERT refused", lambda: cur.execute("INSERT INTO tx VALUES (10, 1), (11, 1), (1, 1)"), 1062)
c.raises("in a transaction: UPDATE refused", lambda: cur.execute("UPDATE tx SET id = 3 WHERE id IN (1, 2)"), 1062)
cur.execute("INSERT INTO tx VALUES (7, 7)")
cur.execute("COMMIT")
c.eq("the transaction kept only its successful statement", q("SELECT COUNT(*), SUM(id), SUM(v) FROM tx"), [(4, Decimal(13), Decimal(13))])
c.raises("VECTOR table: duplicate key refused", lambda: cur.execute("UPDATE vx SET id = 3 WHERE id IN (1, 2)"), 1062)
c.raises("VECTOR table: NULL into NOT NULL refused", lambda: cur.execute("UPDATE vx SET v = NULL WHERE id > 1"), 1048)
c.eq("VECTOR table: nothing changed", q("SELECT COUNT(*), SUM(id), SUM(v) FROM vx"), [(3, Decimal(6), Decimal(6))])

# REPLACE / ON DUPLICATE KEY UPDATE that fail part way put back the rows they deleted
cur.execute("CREATE TABLE rp (id INT PRIMARY KEY, v INT NOT NULL, w INT, UNIQUE KEY (w))")
cur.execute("INSERT INTO rp VALUES (1, 1, 1), (2, 2, 2), (3, 3, 3)")
for sql in ["REPLACE INTO rp VALUES (1, 10, 1), (4, NULL, 4)", "INSERT INTO rp VALUES (2, 0, 0), (5, NULL, 5) ON DUPLICATE KEY UPDATE v = 20"]:
    c.raises(f"refused: {sql}", lambda: cur.execute(sql), 1048)
    c.eq(f"nothing changed: {sql}", q("SELECT COUNT(*), SUM(id), SUM(v), SUM(w) FROM rp"), [(3, Decimal(6), Decimal(6), Decimal(6))])
cur.execute("REPLACE INTO rp VALUES (1, 10, 1), (4, 4, 4)")
c.eq("REPLACE that succeeds", q("SELECT COUNT(*), SUM(id), SUM(v), SUM(w) FROM rp"), [(4, Decimal(10), Decimal(19), Decimal(10))])

# ALTER TABLE .. ADD UNIQUE KEY over repeated values (NULLs may repeat), a second PRIMARY KEY
cur.execute("CREATE TABLE ak (id INT PRIMARY KEY, v INT)")
cur.execute("INSERT INTO ak VALUES (1, 1), (2, NULL), (3, 1), (4, NULL)")
c.raises("ADD UNIQUE KEY over duplicates refused", lambda: cur.execute("ALTER TABLE ak ADD UNIQUE KEY (v)"), 1062)
c.raises("a second PRIMARY KEY refused", lambda: cur.execute("ALTER TABLE ak ADD PRIMARY KEY (v)"), 1068)
cur.execute("UPDATE ak SET v = 2 WHERE id = 3")
cur.execute("ALTER TABLE ak ADD UNIQUE KEY (v)")
c.raises("the UNIQUE key added is enforced", lambda: cur.execute("INSERT INTO ak VALUES (5, 2)"), 1062)

# ADD PRIMARY KEY on a table without one: refused over NULLs / repeated keys (table unchanged),
# then enforced (an integer column: the hash index; several columns: a unique index)
cur.execute("CREATE TABLE apk (id INT, s VARCHAR(10), v INT)")
cur.execute("INSERT INTO apk VALUES (1, 'a', 1), (2, 'b', 2), (NULL, 'd', 4)")
c.raises("ADD PRIMARY KEY over a NULL refused", lambda: cur.execute("ALTER TABLE apk ADD PRIMARY KEY (id)"), 1048)
cur.execute("DELETE FROM apk WHERE id IS NULL")
cur.execute("INSERT INTO apk VALUES (2, 'c', 3)")
c.raises("ADD PRIMARY KEY over repeated keys refused", lambda: cur.execute("ALTER TABLE apk ADD PRIMARY KEY (id)"), 1062)
c.eq("the table is unchanged", q("SELECT COUNT(*), SUM(v) FROM apk"), [(3, Decimal(6))])
cur.execute("UPDATE apk SET id = 3 WHERE s = 'c'")
cur.execute("ALTER TABLE apk ADD PRIMARY KEY (id)")
c.raises("the PRIMARY KEY added is enforced", lambda: cur.execute("INSERT INTO apk VALUES (3, 'x', 9)"), 1062)
c.eq("point lookup by the new key", q("SELECT s FROM apk WHERE id = 3"), [("c",)])
cur.execute("CREATE TABLE apk2 (a VARCHAR(5), b INT)")
cur.execute("INSERT INTO apk2 VALUES ('x', 1), ('x', 2)")
cur.execute("ALTER TABLE apk2 ADD PRIMARY KEY (a, b)")
c.raises("a composite PRIMARY KEY added is enforced", lambda: cur.execute("INSERT INTO apk2 VALUES ('x', 1)"), 1062)

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
