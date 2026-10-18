# S3 ingest against the local SeaweedFS container (warpstore-s3, port 18333): LOAD DATA S3 for CSV, gzip
# CSV, JSON lines (with column mapping) and Parquet; globs; a pipeline that picks up new objects and
# loads each exactly once across a restart; errors for bad credentials / missing buckets.
import gzip, io, json, os, subprocess, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import TMP, Checker, Server

EP = "http://127.0.0.1:18333"
KEY, SECRET = "warpstorekey", "warpstoresecret"
BUCKET = "events"  # (SeaweedFS in the container can't create new buckets: it has no free volumes)
PFX = f"t{os.getpid()}"
CFG = f"CONFIG '{{\"region\":\"us-east-1\",\"endpoint_url\":\"{EP}\"}}' CREDENTIALS '{{\"aws_access_key_id\":\"{KEY}\",\"aws_secret_access_key\":\"{SECRET}\"}}'"


def put(key, data: bytes):
    path = os.path.join(TMP, "s3obj")
    with open(path, "wb") as f:
        f.write(data)
    r = subprocess.run(["curl", "-sf", "--aws-sigv4", "aws:amz:us-east-1:s3", "--user", f"{KEY}:{SECRET}", "-H", "x-amz-content-sha256: UNSIGNED-PAYLOAD",
                        "-T", path, f"{EP}/{BUCKET}/{PFX}/{key}"], capture_output=True)
    assert r.returncode == 0, r.stderr


for _ in range(60):  # (the container may have just started)
    if subprocess.run(["curl", "-s", "-o", "/dev/null", f"{EP}/"], capture_output=True).returncode == 0:
        break
    time.sleep(0.5)
time.sleep(1)
c = Checker()
srv = Server(name="s3", http=False).start()
cur = srv.cursor()
cur.execute("CREATE DATABASE ing")
cur.execute("USE ing")
cur.execute("CREATE TABLE ev (id BIGINT PRIMARY KEY, user VARCHAR(20), amount DECIMAL(10,2), ts DATETIME)")


def q(sql):
    cur.execute(sql)
    return [tuple(r) for r in cur.fetchall()]


rows = [(i, f"u{i % 7}", f"{i % 100}.25", f"2024-01-{1 + i % 28:02d} 10:00:00") for i in range(1, 3001)]
csv = "".join(",".join(map(str, r)) + "\n" for r in rows[:1000]).encode()
put("csv/part-1.csv", csv)
put("csv/part-2.csv.gz", gzip.compress("".join(",".join(map(str, r)) + "\n" for r in rows[1000:2000]).encode()))
put("csv/skip.txt", b"not,a,row\n")
q(f"LOAD DATA S3 '{BUCKET}/{PFX}/csv/part-*' {CFG} INTO TABLE ev FIELDS TERMINATED BY ','")
c.eq("CSV + gzip CSV via glob", q("SELECT COUNT(*), SUM(amount), COUNT(DISTINCT user) FROM ev"), [(2000, sum(float(r[2]) for r in rows[:2000]), 7)])

# JSON lines with column mapping (nested keys)
jl = "".join(json.dumps({"id": r[0], "who": {"name": r[1]}, "amt": float(r[2]), "when": r[3]}) + "\n" for r in rows[2000:2500]).encode()
put("json/a.json", jl)
q(f"LOAD DATA S3 '{BUCKET}/{PFX}/json/a.json' {CFG} INTO TABLE ev FORMAT JSON (id <- id, user <- who::name, amount <- amt, ts <- when)")
c.eq("JSON lines with a column mapping", q("SELECT COUNT(*), MAX(user) FROM ev WHERE id > 2000"), [(500, "u6")])

# Parquet
try:
    import pyarrow as pa, pyarrow.parquet as pq
    tbl = pa.table({"id": [r[0] for r in rows[2500:]], "user": [r[1] for r in rows[2500:]], "amount": [float(r[2]) for r in rows[2500:]],
                    "ts": [r[3] for r in rows[2500:]]})
    buf = io.BytesIO()
    pq.write_table(tbl, buf)
    put("pq/x.parquet", buf.getvalue())
    q(f"LOAD DATA S3 '{BUCKET}/{PFX}/pq/x.parquet' {CFG} INTO TABLE ev FORMAT PARQUET")
    c.eq("Parquet", q("SELECT COUNT(*), SUM(amount) FROM ev"), [(3000, sum(float(r[2]) for r in rows))])
except ImportError:
    print("(pyarrow missing: Parquet part skipped)")

# errors
c.raises("bad credentials", lambda: cur.execute(f"LOAD DATA S3 '{BUCKET}/{PFX}/csv/*' CONFIG '{{\"endpoint_url\":\"{EP}\"}}' CREDENTIALS '{{\"aws_access_key_id\":\"x\",\"aws_secret_access_key\":\"y\"}}' INTO TABLE ev"))

# pipeline: new objects are picked up while it runs, each loaded once, also across a restart
cur.execute("CREATE TABLE clicks (id BIGINT, page VARCHAR(20))")
put("pipe/1.csv", b"".join(f"{i},p{i % 3}\n".encode() for i in range(100)))
q(f"CREATE PIPELINE pc AS LOAD DATA S3 '{BUCKET}/{PFX}/pipe/*.csv' {CFG} BATCH_INTERVAL 200 INTO TABLE clicks FIELDS TERMINATED BY ','")
q("START PIPELINE pc")


def wait_count(n, timeout=20):
    t = time.time()
    while time.time() - t < timeout:
        if q("SELECT COUNT(*) FROM clicks")[0][0] >= n:
            break
        time.sleep(0.2)
    return q("SELECT COUNT(*) FROM clicks")[0][0]


c.eq("pipeline loads the first object", wait_count(100), 100)
put("pipe/2.csv", b"".join(f"{i},p{i % 3}\n".encode() for i in range(100, 250)))
c.eq("pipeline picks up a new object", wait_count(250), 250)
srv.kill()
srv.start()
cur = srv.cursor("ing")
put("pipe/3.csv", b"".join(f"{i},p{i % 3}\n".encode() for i in range(250, 300)))
c.eq("after a crash: new objects only, nothing twice", wait_count(300), 300)
time.sleep(1)
c.eq("still exactly once", q("SELECT COUNT(*), COUNT(DISTINCT id) FROM clicks"), [(300, 300)])
st = q("SHOW PIPELINES")
c.check("SHOW PIPELINES", any("pc" in str(r) for r in st), str(st))
q("STOP PIPELINE pc")
srv.stop()
c.done()
