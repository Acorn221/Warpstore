# Disk tier: a table about 3x larger than --memory-limit. Column data is evicted to the segment files
# and read back for queries (into the cache, or for one batch when it is full); point lookups, DML,
# SNAPSHOT (segments by reference), restart, kill -9 and DROP TABLE (files removed) on top of it.
# Runs with and without compressed segments.
import os, re, sys, time, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import TMP, Checker, Server

N = 4_000_000
LIMIT_GB = 0.04  # ~43 MB; the table is ~100 MB of column data
os.environ.setdefault("WARPSTORE_SCAN_WINDOW_MB", "16")  # several scan windows per query

c = Checker()
path = os.path.join(TMP, f"tier-{os.getpid()}.tsv")
with open(path, "w") as f:
    for i in range(N):
        f.write(f"{i}\t{i % 1000}\t{i * 0.5}\ts{i % 50}\n")
want_total = [(N, sum(i % 1000 for i in range(N)), N * (N - 1) / 4)]


def metric(srv, name):
    text = urllib.request.urlopen(f"http://127.0.0.1:{srv.http_port}/metrics", timeout=10).read().decode()
    m = re.search(rf"^{name} (\S+)$", text, re.M)
    return float(m.group(1)) if m else None


def q(cur, sql):
    cur.execute(sql)
    return [tuple(r) for r in cur.fetchall()]


def tiers(cur):
    return {r[2] for r in q(cur, "SHOW SEGMENTS FROM t")}


def wait_evicted(srv, cur, label):
    t0 = time.time()
    while time.time() - t0 < 30:
        used = metric(srv, "warpstore_table_memory_bytes")
        if used is not None and used <= LIMIT_GB * (1 << 30) and "disk-columnstore" in tiers(cur):
            break
        time.sleep(0.3)
    c.check(f"{label}: evicted under the limit", used <= LIMIT_GB * (1 << 30) and "disk-columnstore" in tiers(cur), f"{used} bytes, {tiers(cur)}")


def queries(cur, label, deleted=False):
    lo = 1000 if deleted else 0
    n = N - lo
    tot = q(cur, "SELECT COUNT(*), SUM(g), SUM(v) FROM t")
    want_g = sum(i % 1000 for i in range(lo, N)) + (-1 - 42 % 1000 if deleted and 42 >= lo else 0)
    c.eq(f"{label}: totals", [(tot[0][0], int(tot[0][1]), tot[0][2])], [(n, want_g, (N * (N - 1) - lo * (lo - 1)) / 4)])
    c.eq(f"{label}: GROUP BY string", q(cur, "SELECT s, COUNT(*) FROM t GROUP BY s ORDER BY s LIMIT 2"),
         [("s0", len([i for i in range(lo, N) if i % 50 == 0])), ("s1", len([i for i in range(lo, N) if i % 50 == 1]))])
    c.eq(f"{label}: sort-key range", q(cur, "SELECT COUNT(*), MIN(id), MAX(v) FROM t WHERE id BETWEEN 2500000 AND 2500999"), [(1000, 2500000, 2500999 * 0.5)])
    c.eq(f"{label}: point lookup", q(cur, "SELECT g, v, s FROM t WHERE id = 3333333"), [(333, 3333333 * 0.5, "s33")])
    top = [i for i in range(N - 1, lo - 1, -1) if i % 1000 == 7][:3]
    c.eq(f"{label}: top-N, other columns read for the top rows", q(cur, "SELECT * FROM t WHERE g = 7 ORDER BY v DESC LIMIT 3"),
         [(i, 7, i * 0.5, f"s{i % 50}") for i in top])
    c.eq(f"{label}: top-N with OFFSET", q(cur, "SELECT s, id FROM t ORDER BY id DESC LIMIT 2 OFFSET 1"), [(f"s{(N - 2) % 50}", N - 2), (f"s{(N - 3) % 50}", N - 3)])
    c.eq(f"{label}: filter", q(cur, "SELECT COUNT(*) FROM t WHERE g = 7 AND v > 1000000"), [(len([i for i in range(2000001, N) if i % 1000 == 7]),)])


for compress in ("--no-compress", "--compress"):
    tag = compress[2:]
    srv = Server("--memory-limit", str(LIMIT_GB), "--partitions", "1", compress, name=f"tier-{tag}").start()
    cur = srv.cursor()
    q(cur, "CREATE DATABASE d")
    cur = srv.cursor("d")
    q(cur, "CREATE TABLE t (id BIGINT PRIMARY KEY, g INT, v DOUBLE, s VARCHAR(16), SORT KEY(id))")
    q(cur, f"LOAD DATA INFILE '{path}' INTO TABLE t FIELDS TERMINATED BY '\\t'")
    wait_evicted(srv, cur, tag)
    queries(cur, f"{tag} cold")
    queries(cur, f"{tag} again")
    segdir = os.path.join(srv.data_dir, "segments")
    c.check(f"{tag}: segment files written", len(os.listdir(segdir)) > 0)
    disk = sum(os.path.getsize(os.path.join(segdir, f)) for f in os.listdir(segdir))
    c.check(f"{tag}: segment files compressed", 0 < disk < 48 << 20, f"{disk} bytes for ~96 MB of column data")
    # DML on cold segments
    q(cur, "UPDATE t SET g = -1 WHERE id = 42")
    q(cur, "DELETE FROM t WHERE id < 1000")
    srv.connect().commit()
    queries(cur, f"{tag} after DML", deleted=True)
    q(cur, "SNAPSHOT")
    files = set(os.listdir(segdir))
    srv.restart()
    cur = srv.cursor("d")
    c.check(f"{tag}: restart: segments start on disk", "disk-columnstore" in tiers(cur), str(tiers(cur)))
    queries(cur, f"{tag} after restart", deleted=True)
    c.eq(f"{tag}: restart keeps the snapshot's files", set(os.listdir(segdir)) >= files, True)
    # more rows, kill -9 (WAL replay on top of segments from files)
    q(cur, "INSERT INTO t VALUES (-5, 5, 5.5, 'neg')")
    srv.kill()
    srv.start()
    cur = srv.cursor("d")
    c.eq(f"{tag}: after kill -9", q(cur, "SELECT COUNT(*) FROM t"), [(N - 1000 + 1,)])
    c.eq(f"{tag}: row from the WAL", q(cur, "SELECT s FROM t WHERE id = -5"), [("neg",)])
    # DROP TABLE: its files go once a snapshot no longer refers to them
    q(cur, "DROP TABLE t")
    q(cur, "SNAPSHOT")
    time.sleep(0.5)
    c.eq(f"{tag}: DROP TABLE + SNAPSHOT removes the files", os.listdir(segdir), [])
    srv.stop()
os.unlink(path)
c.done()
