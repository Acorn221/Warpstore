# Kafka ingest against the local Redpanda container (warpstore-redpanda, 127.0.0.1:19092): LOAD DATA
# KAFKA (JSON and delimited messages, several partitions) and a pipeline that keeps consuming, with
# exactly-once offsets across a crash.
import json, os, subprocess, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import Checker, Server

BROKER = "127.0.0.1:19092"
RUN = str(os.getpid())


def rpk(*args, stdin=None):
    r = subprocess.run(["docker", "exec", "-i", "warpstore-redpanda", "rpk", *args], input=stdin, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r.stdout


def produce(topic, lines):
    rpk("topic", "produce", topic, "-f", "%v\n", stdin="".join(l + "\n" for l in lines))


c = Checker()
srv = Server(name="kafka", http=False).start()
cur = srv.cursor()
cur.execute("CREATE DATABASE k")
cur.execute("USE k")


def q(sql):
    cur.execute(sql)
    return [tuple(r) for r in cur.fetchall()]


# one-shot loads
tj, tc, tp = f"j{RUN}", f"c{RUN}", f"p{RUN}"
for t in (tj, tc, tp):
    rpk("topic", "create", t, "-p", "3")
produce(tj, [json.dumps({"id": i, "user": {"name": f"u{i % 5}"}, "amount": i * 0.5}) for i in range(1000)])
cur.execute("CREATE TABLE ev (id BIGINT, name VARCHAR(10), amount DOUBLE)")
q(f"LOAD DATA KAFKA '{BROKER}/{tj}' INTO TABLE ev FORMAT JSON (id <- id, name <- user::name, amount <- amount)")
c.eq("JSON messages from 3 partitions", q("SELECT COUNT(*), COUNT(DISTINCT id), SUM(amount), COUNT(DISTINCT name) FROM ev"), [(1000, 1000, 249750.0, 5)])
produce(tc, [f"{i},x{i % 3}" for i in range(500)])
cur.execute("CREATE TABLE cv (id INT, tag VARCHAR(10))")
q(f"LOAD DATA KAFKA '{BROKER}/{tc}' INTO TABLE cv FIELDS TERMINATED BY ','")
c.eq("delimited messages", q("SELECT COUNT(*), COUNT(DISTINCT tag) FROM cv"), [(500, 3)])

# pipeline: keeps consuming, exactly once across a crash
cur.execute("CREATE TABLE stream (id BIGINT, v INT)")
q(f"CREATE PIPELINE ps AS LOAD DATA KAFKA '{BROKER}/{tp}' BATCH_INTERVAL 200 INTO TABLE stream FORMAT JSON (id <- id, v <- v)")
q("START PIPELINE ps")


def wait_count(n, timeout=30):
    t = time.time()
    while time.time() - t < timeout and q("SELECT COUNT(*) FROM stream")[0][0] < n:
        time.sleep(0.2)
    return q("SELECT COUNT(*) FROM stream")[0][0]


produce(tp, [json.dumps({"id": i, "v": 1}) for i in range(0, 500)])
c.eq("pipeline consumes", wait_count(500), 500)
produce(tp, [json.dumps({"id": i, "v": 1}) for i in range(500, 800)])
c.eq("pipeline keeps consuming", wait_count(800), 800)
srv.kill()
produce(tp, [json.dumps({"id": i, "v": 1}) for i in range(800, 1000)])  # while the server is down
srv.start()
cur = srv.cursor("k")
c.eq("after a crash: resumes where it was", wait_count(1000), 1000)
time.sleep(1.5)
c.eq("exactly once", q("SELECT COUNT(*), COUNT(DISTINCT id) FROM stream"), [(1000, 1000)])
q("STOP PIPELINE ps")
for t in (tj, tc, tp):
    rpk("topic", "delete", t)
srv.stop()
c.done()
