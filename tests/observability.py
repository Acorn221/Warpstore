# Observability: SHOW PROCESSLIST / information_schema.PROCESSLIST, KILL [QUERY], CONNECTION_ID(),
# the slow query log, information_schema.QUERY_STATS and the Prometheus /metrics endpoint.
import os, sys, threading, time, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pymysql
from lib import Checker, Server

c = Checker()
srv = Server("--slow-query-ms", "100", name="observe").start()
admin = srv.cursor()


def q(cur, sql):
    cur.execute(sql)
    return [tuple(r) for r in cur.fetchall()]


a = srv.connect()
acur = a.cursor()
aid = q(acur, "SELECT CONNECTION_ID()")[0][0]
c.check("CONNECTION_ID() is a real id", aid > 1, str(aid))

out = {}
th = threading.Thread(target=lambda: out.update(r=q(acur, "SELECT SLEEP(5)"), t=time.time()))
t0 = time.time()
th.start()
time.sleep(0.3)
pl = q(admin, "SHOW FULL PROCESSLIST")
mine = [r for r in pl if r[0] == aid]
c.check("processlist shows the running statement", mine and mine[0][4] == "Query" and "SLEEP" in (mine[0][7] or ""), str(pl))
c.check("information_schema.PROCESSLIST", any(r[0] == aid for r in q(admin, "SELECT ID, COMMAND, INFO FROM information_schema.PROCESSLIST")))
q(admin, f"KILL QUERY {aid}")
th.join(5)
c.check("KILL QUERY interrupts", out.get("r") == [(1,)] and out["t"] - t0 < 2, str(out))
c.eq("connection survives KILL QUERY", q(acur, "SELECT 1"), [(1,)])

b = srv.connect()
bcur = b.cursor()
bid = q(bcur, "SELECT CONNECTION_ID()")[0][0]
q(admin, "CREATE USER IF NOT EXISTS mon IDENTIFIED BY 'pw'")
m = srv.connect(user="mon", password="pw")
c.raises("KILL of someone else's session needs SUPER", lambda: m.cursor().execute(f"KILL {bid}"), 1095)
c.check("non-SUPER users see only their own sessions", all(r[1] == "mon" for r in q(m.cursor(), "SHOW PROCESSLIST")))
q(admin, f"KILL {bid}")
try:
    bcur.execute("SELECT 1")
    bcur.fetchall()
    dropped = False
except pymysql.err.OperationalError:
    dropped = True
c.check("KILL closes the connection", dropped)
c.raises("KILL of an unknown id", lambda: admin.execute("KILL 999999"), 1094)

# query stats: one digest for the same statement with different literals
admin.execute("CREATE DATABASE obs")
admin.execute("CREATE TABLE obs.t (id INT PRIMARY KEY, s VARCHAR(10))")
admin.execute("INSERT INTO obs.t VALUES (1,'a'),(2,'b'),(3,'c')")
q(admin, "FLUSH QUERY_STATS")
for i in (1, 2, 3):
    q(acur, f"SELECT s FROM obs.t WHERE id = {i}")
q(acur, "SELECT s FROM obs.t WHERE id IN (1, 2, 3)")
q(acur, "SELECT s FROM obs.t WHERE id IN (1, 2)")
st = q(admin, "SELECT DIGEST_TEXT, COUNT_STAR, ROWS FROM information_schema.QUERY_STATS WHERE DIGEST_TEXT LIKE 'SELECT s FROM obs.t%' ORDER BY DIGEST_TEXT")
c.eq("query digests", st, [("SELECT s FROM obs.t WHERE id = ?", 3, 3), ("SELECT s FROM obs.t WHERE id IN (...)", 2, 5)])

# slow query log
q(acur, "SELECT SLEEP(0.2)")
log = open(os.path.join(srv.data_dir, "slow.log")).read()
c.check("slow log entry", "# Query_time: 0.2" in log and "SELECT SLEEP(0.2);" in log, log[-500:])
c.check("fast statements stay out of the slow log", "WHERE id = 1" not in log)
q(admin, "SET GLOBAL slow_query_log = OFF")
q(acur, "SELECT SLEEP(0.15)")
c.check("SET GLOBAL slow_query_log = OFF", "SLEEP(0.15)" not in open(os.path.join(srv.data_dir, "slow.log")).read())

# Prometheus
text = urllib.request.urlopen(f"http://127.0.0.1:{srv.http_port}/metrics", timeout=10).read().decode()
vals = {}
for line in text.splitlines():
    if line and not line.startswith("#"):
        k, v = line.rsplit(" ", 1)
        vals[k] = float(v)
c.check("metrics: select count", vals.get('warpstore_statements_total{kind="select"}', 0) >= 8, str(sorted(vals)[:10]))
c.check("metrics: rows", vals.get("warpstore_rows", -1) == 3)
c.check("metrics: connections", vals.get("warpstore_connections", 0) >= 2)
c.check("metrics: histogram", vals.get('warpstore_statement_duration_seconds_bucket{le="+Inf"}', 0) == vals.get("warpstore_statement_duration_seconds_count", -1))
c.check("metrics: WAL", "warpstore_wal_syncs_total" in vals and "warpstore_wal_bytes" in vals)
c.check("metrics: slow queries", vals.get("warpstore_slow_queries_total", 0) >= 2)
srv.stop()
c.done()
