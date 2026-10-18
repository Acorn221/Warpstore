# Backup / restore: full restore, restore under another name, point-in-time restore over a WAL
# rotation (--wal-archive), S3 (when the warpstore-s3 container is running), and LOAD DATA files that are
# deleted before a crash.
import os, subprocess, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import TMP, Checker, Server

c = Checker()
arch = os.path.join(TMP, f"archive-{os.getpid()}")
bdir = os.path.join(TMP, f"backups-{os.getpid()}")
srv = Server("--wal-archive", arch, name="backup", http=False).start()
cur = srv.cursor()


def q(sql):
    cur.execute(sql)
    return [tuple(r) for r in cur.fetchall()]


def state(db):
    return (q(f"SELECT id, name, price, qty FROM {db}.items ORDER BY id"), q(f"SELECT COUNT(*), SUM(v) FROM {db}.ev"),
            q(f"SELECT * FROM {db}.cheap ORDER BY id"))


cur.execute("CREATE DATABASE shop")
cur.execute("USE shop")
cur.execute("CREATE TABLE items (id INT PRIMARY KEY, name VARCHAR(40), price DECIMAL(10,2), qty INT)")
cur.execute("CREATE TABLE ev (id BIGINT PRIMARY KEY, v INT)")
cur.execute("CREATE VIEW cheap AS SELECT id, name FROM items WHERE price < 5")
cur.execute("INSERT INTO items VALUES " + ",".join(f"({i}, 'item{i % 37}', {i % 13}.25, {i})" for i in range(1, 2001)))
cur.execute("INSERT INTO ev VALUES " + ",".join(f"({i}, {i % 7})" for i in range(1, 5001)))
s0 = state("shop")
r = q(f"BACKUP DATABASE shop TO '{bdir}'")
print(r)
c.check("backup file", os.path.exists(f"{bdir}/shop.backup") and os.path.exists(f"{bdir}/shop.backup.manifest"))

# phase 1 (before T1), then T1, then phase 2 across a WAL rotation
cur.execute("UPDATE items SET price = price + 1 WHERE id % 3 = 0")
cur.execute("DELETE FROM ev WHERE id % 10 = 0")
cur.execute("BEGIN")
cur.execute("INSERT INTO ev VALUES (100001, 5), (100002, 6)")
cur.execute("COMMIT")
time.sleep(1.2)
t1 = q("SELECT NOW()")[0][0].strftime("%Y-%m-%d %H:%M:%S")
s1 = state("shop")
time.sleep(1.2)
cur.execute("INSERT INTO items VALUES (5000, 'late', 1.00, 1)")
q("SNAPSHOT")  # rotates the WAL: phase 1 is now only in the archive
cur.execute("DELETE FROM items WHERE id < 100")
s2 = state("shop")
c.check("archive has a WAL", any(f.startswith("wal-") for f in os.listdir(arch)) if os.path.isdir(arch) else False)

q(f"RESTORE DATABASE shop AS shop_full FROM '{bdir}'")
c.eq("restore AS: state at backup time", state("shop_full"), s0)
q(f"RESTORE DATABASE shop AS shop_t1 FROM '{bdir}' UNTIL TIMESTAMP '{t1}'")
c.eq("point-in-time restore to T1", state("shop_t1"), s1)
q(f"RESTORE DATABASE shop AS shop_now FROM '{bdir}' UNTIL TIMESTAMP '2999-01-01 00:00:00'")
c.eq("point-in-time restore to now", state("shop_now"), s2)
c.raises("restore over a database with tables", lambda: cur.execute(f"RESTORE DATABASE shop FROM '{bdir}'"), 1007)
cur.execute("DROP DATABASE shop")
q(f"RESTORE DATABASE shop FROM '{bdir}'")
c.eq("restore after DROP", state("shop"), s0)

# LOAD DATA of a file that is deleted before a crash: the WAL replays a copy in the data directory
csv = os.path.join(TMP, f"load-{os.getpid()}.csv")
with open(csv, "w") as f:
    f.writelines(f"{i},{i * 2}\n" for i in range(1, 30001))
cur.execute("CREATE TABLE shop.loaded (a INT, b INT)")
cur.execute(f"LOAD DATA INFILE '{csv}' INTO TABLE shop.loaded FIELDS TERMINATED BY ','")
os.unlink(csv)
srv.kill()
srv.start()
cur = srv.cursor()
c.eq("restored databases survive a crash", state("shop_t1"), s1)
c.eq("LOAD DATA survives deleting its file + crash", q("SELECT COUNT(*), SUM(b) FROM shop.loaded"), [(30000, 900030000)])

# S3 (SeaweedFS container from the ingest tests)
up = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}", "warpstore-s3"], capture_output=True, text=True).stdout.strip() == "true"
if up:
    cfg = "CONFIG '{\"region\":\"us-east-1\",\"endpoint_url\":\"http://127.0.0.1:18333\"}' CREDENTIALS '{\"aws_access_key_id\":\"warpstorekey\",\"aws_secret_access_key\":\"warpstoresecret\"}'"
    try:
        print(q(f"BACKUP DATABASE shop TO S3 'events/backup-test-{os.getpid()}' {cfg}"))
        q(f"RESTORE DATABASE shop AS shop_s3 FROM S3 'events/backup-test-{os.getpid()}' {cfg}")
        c.eq("S3 backup + restore", state("shop_s3"), state("shop"))
    except Exception as e:
        c.check("S3 backup + restore", False, str(e))
else:
    print("(S3 part skipped: warpstore-s3 container not running)")
srv.stop()
c.done()
