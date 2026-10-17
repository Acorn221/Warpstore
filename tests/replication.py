# Replication: a replica follows the primary (DML, transactions, LOAD DATA, DDL, primary restarts),
# is read-only, semi-sync commits are visible on the replica as soon as they return, and a replica
# with --failover-timeout promotes itself when the primary dies.
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pymysql
from lib import TMP, Checker, Server

c = Checker()


def digest(srv):
    cur = srv.cursor()
    try:
        out = []
        for sql in ("SELECT COUNT(*), SUM(qty), SUM(price) FROM app.orders", "SELECT COUNT(*), SUM(v) FROM app.ev",
                    "SELECT name FROM app.cust ORDER BY id", "SELECT COUNT(*) FROM app.big_orders"):
            cur.execute(sql)
            out.append(tuple(cur.fetchall()))
        return out
    except pymysql.MySQLError as e:
        return str(e)


def converge(a, b, name, timeout=30):
    t = time.time()
    while time.time() - t < timeout:
        da, db = digest(a), digest(b)
        if da == db and not isinstance(da, str):
            return c.check(name, True)
        time.sleep(0.2)
    return c.eq(name, digest(b), digest(a))


def status(srv):
    cur = srv.cursor()
    cur.execute("SHOW REPLICATION STATUS")
    cols = [d[0] for d in cur.description]
    return dict(zip(cols, cur.fetchone()))


P = Server(name="primary").start()
R = Server("--replica-of", f"http://root:@127.0.0.1:{P.http_port}", name="replica").start()
p = P.cursor()
p.execute("CREATE DATABASE app")
p.execute("USE app")
p.execute("CREATE TABLE cust (id INT PRIMARY KEY, name VARCHAR(30))")
p.execute("CREATE TABLE orders (id BIGINT PRIMARY KEY, cust INT, qty INT, price DECIMAL(10,2), SHARD KEY(id))")
p.execute("CREATE TABLE ev (id BIGINT PRIMARY KEY, v INT)")
p.execute("CREATE VIEW big_orders AS SELECT * FROM orders WHERE qty > 5")
p.execute("INSERT INTO cust VALUES " + ",".join(f"({i}, 'c{i}')" for i in range(50)))
p.execute("INSERT INTO orders VALUES " + ",".join(f"({i}, {i % 50}, {i % 10}, {i % 7}.50)" for i in range(5000)))
p.execute("INSERT INTO ev VALUES (1, 1)")
converge(P, R, "initial copy")

c.raises("replica is read-only", lambda: R.cursor().execute("INSERT INTO app.ev VALUES (2, 2)"), 1290)

p.execute("BEGIN")
p.execute("UPDATE orders SET qty = qty + 1 WHERE cust = 3")
p.execute("DELETE FROM orders WHERE id % 97 = 0")
p.execute("COMMIT")
csv = os.path.join(TMP, f"repl-{os.getpid()}.csv")
with open(csv, "w") as f:
    f.writelines(f"{100000 + i},{i % 13}\n" for i in range(20000))
p.execute(f"LOAD DATA INFILE '{csv}' INTO TABLE ev FIELDS TERMINATED BY ','")
os.unlink(csv)
p.execute("ALTER TABLE cust ADD COLUMN tier INT DEFAULT 1")
p.execute("UPDATE cust SET name = CONCAT(name, '!') WHERE id < 5")
converge(P, R, "transactions, LOAD DATA, ALTER")

P.restart()  # the primary's log positions start over: the replica re-bootstraps
p = P.cursor("app")
p.execute("INSERT INTO ev VALUES (3, 3)")
converge(P, R, "after a primary restart")

# semi-sync: a commit returns only once the replica has applied it
P.stop()
P.args = ["--sync-replicas", "1", "--sync-replica-timeout-ms", "3000"]
P.start()
converge(P, R, "replica back after the primary restarts with --sync-replicas")
p = P.cursor("app")
r = R.cursor("app")
misses = 0
t0 = time.time()
for i in range(100):
    p.execute(f"INSERT INTO ev VALUES ({200000 + i}, 1)")
    r.execute(f"SELECT COUNT(*) FROM ev WHERE id = {200000 + i}")
    misses += r.fetchone()[0] != 1
print(f"semi-sync: 100 commits in {time.time() - t0:.2f}s, {misses} not yet on the replica")
c.eq("semi-sync commits are on the replica when they return", misses, 0)
st = status(P)
c.check("primary lists its replica", st.get("Replicas") is not None, str(st))

# automatic failover
R.stop()
P.args = []
P.restart()  # back to async commits (the replica below fails over after 2 s: start it after restarts)
R2 = Server("--replica-of", f"http://root:@127.0.0.1:{P.http_port}", "--failover-timeout", "2", name="replica2").start()
converge(P, R2, "second replica")
p = P.cursor("app")
p.execute("INSERT INTO ev VALUES (4, 4)")
converge(P, R2, "last write before the crash")
before = digest(P)
P.kill()
t = time.time()
while time.time() - t < 20 and status(R2)["Role"] != "PRIMARY":
    time.sleep(0.2)
st = status(R2)
print(f"failover after {time.time() - t:.1f}s: {st['Role']} {st['State']}")
c.eq("replica promoted itself", st["Role"], "PRIMARY")
c.eq("promoted replica has every write", digest(R2), before)
R2.cursor("app").execute("INSERT INTO ev VALUES (5, 5)")
c.check("promoted replica accepts writes", True)
R3 = Server("--replica-of", f"http://root:@127.0.0.1:{R2.http_port}", name="replica3").start()
converge(R2, R3, "new replica of the promoted primary")
for s in (R2, R3):
    s.stop()
c.done()
