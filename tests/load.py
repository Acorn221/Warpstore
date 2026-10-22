# LOAD DATA INFILE: the parallel loader (several 64 MB chunks, a segment flush in the middle of a
# chunk), the row-at-a-time path (primary key, AUTO_INCREMENT), shard keys, CSV quoting, ESCAPED BY,
# NULLs, CRLF, IGNORE n LINES, error rows, zstd input, and replay of the WAL's compressed copy after
# the source file is deleted and the server killed.
import os, random, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decimal import Decimal
from lib import TMP, Checker, Server

c = Checker()
srv = Server("--partitions", "1", name="load", http=False).start()
cur = srv.cursor()


def q(sql):
    cur.execute(sql)
    return [tuple(r) for r in cur.fetchall()]


def path(name):
    return os.path.join(TMP, f"{name}-{os.getpid()}")


cur.execute("CREATE DATABASE l")
cur.execute("USE l")

# 1) Plain table, > 64 MB (several chunks), 1.2M rows (the 1M-row segment flush lands mid-chunk).
N = 1_200_000
rnd = random.Random(7)
words = [f"w{i}" * (1 + i % 5) for i in range(5000)]
big = path("big.tsv")
exp = {"n": 0, "a": 0, "b": 0, "cents": 0, "nulls": 0, "d": 0.0, "words": set(), "maxdt": ""}
with open(big, "w") as f:
    for i in range(N):
        a = rnd.randint(-1000, 1000)
        b = rnd.randint(0, 1 << 40)
        cents = rnd.randint(0, 10**7)
        w = words[rnd.randrange(len(words))] if i % 11 else None
        dt = f"2024-{1 + i % 12:02d}-{1 + i % 28:02d} {i % 24:02d}:{i % 60:02d}:{(i * 7) % 60:02d}"
        d = (i % 1000) / 8
        f.write(f"{i}\t{a}\t{b}\t{cents // 100}.{cents % 100:02d}\t{w if w else chr(92) + 'N'}\t2024-03-{1 + i % 28:02d}\t{dt}\t{d}\n")
        exp["n"] += 1; exp["a"] += a; exp["b"] += b; exp["cents"] += cents; exp["d"] += d
        if w: exp["words"].add(w)
        else: exp["nulls"] += 1
        exp["maxdt"] = max(exp["maxdt"], dt)
c.check("big file is over 64 MB", os.path.getsize(big) > 64 << 20, os.path.getsize(big))
cur.execute("CREATE TABLE big (id INT, a INT, b BIGINT, m DECIMAL(12,2), w VARCHAR(40), dd DATE, dt DATETIME, x DOUBLE)")
cur.execute(f"LOAD DATA INFILE '{big}' INTO TABLE big FIELDS TERMINATED BY '\\t'")


def check_big(label):
    r = q("SELECT COUNT(*), SUM(a), SUM(b), SUM(m), COUNT(DISTINCT w), SUM(w IS NULL), MAX(dt), SUM(x) FROM big")[0]
    c.eq(f"{label}: count, sums, distinct, NULLs, max datetime",
         (r[0], int(r[1]), int(r[2]), Decimal(r[3]), r[4], int(r[5]), str(r[6])),
         (exp["n"], exp["a"], exp["b"], Decimal(exp["cents"]) / 100, len(exp["words"]), exp["nulls"], exp["maxdt"]))
    c.check(f"{label}: double sum", abs(r[7] - exp["d"]) < 1e-6 * exp["d"], (r[7], exp["d"]))
    got = q("SELECT id, w, dd, dt FROM big WHERE id IN (0, 11, 999999, 1000000, 1199999) ORDER BY id")
    c.eq(f"{label}: sample rows", [(g[0], g[1] is None, str(g[2]), str(g[3])) for g in got],
         [(i, i % 11 == 0, f"2024-03-{1 + i % 28:02d}", f"2024-{1 + i % 12:02d}-{1 + i % 28:02d} {i % 24:02d}:{i % 60:02d}:{(i * 7) % 60:02d}")
          for i in (0, 11, 999999, 1000000, 1199999)])


check_big("parallel load")
c.check("a segment was flushed", any(r[1] != "-1" for r in q("SHOW SEGMENTS FROM big")))

# 2) Primary key + AUTO_INCREMENT (row-at-a-time path), column list, IGNORE 1 LINES, CRLF.
pk = path("pk.csv")
with open(pk, "w", newline="") as f:
    f.write("name,score\r\n")
    f.writelines(f"n{i},{i * 3}\r\n" for i in range(50000))
cur.execute("CREATE TABLE pk (id INT AUTO_INCREMENT PRIMARY KEY, name VARCHAR(20), score INT)")
cur.execute(f"LOAD DATA INFILE '{pk}' INTO TABLE pk FIELDS TERMINATED BY ',' IGNORE 1 LINES (name, score)")
c.eq("auto increment ids in file order", q("SELECT COUNT(*), MIN(id), MAX(id), SUM(score), SUM(id * score) FROM pk"),
     [(50000, 1, 50000, 3 * 49999 * 50000 // 2, sum((i + 1) * i * 3 for i in range(50000)))])
c.eq("name of id 777", q("SELECT name FROM pk WHERE id = 777"), [("n776",)])
dup = path("dup.csv")
with open(dup, "w") as f:
    f.writelines(f"{i},x{i}\n" for i in [1, 2, 3, 2, 4])
cur.execute("CREATE TABLE pk2 (id INT PRIMARY KEY, v VARCHAR(10))")
c.raises("duplicate primary key", lambda: cur.execute(f"LOAD DATA INFILE '{dup}' INTO TABLE pk2 FIELDS TERMINATED BY ','"), 1062)

# 3) Shard key (rows hashed to partitions).
sk = path("sk.tsv")
with open(sk, "w") as f:
    f.writelines(f"k{i % 997}\t{i}\n" for i in range(200000))
cur.execute("CREATE TABLE sk (k VARCHAR(10), v INT, SHARD KEY (k))")
cur.execute(f"LOAD DATA INFILE '{sk}' INTO TABLE sk FIELDS TERMINATED BY '\\t'")
c.eq("shard key table", q("SELECT COUNT(*), COUNT(DISTINCT k), SUM(v) FROM sk"), [(200000, 997, 199999 * 200000 // 2)])
c.eq("shard key group", q("SELECT COUNT(*), SUM(v) FROM sk WHERE k = 'k5'"),
     [(len(range(5, 200000, 997)), sum(range(5, 200000, 997)))])

# 4) CSV quoting, NULLs, ESCAPED BY, no newline at the end.
csv = path("q.csv")
with open(csv, "w") as f:
    f.write('1,"a, b",\\N\n2,"say ""hi""",5\n3,plain,7')
cur.execute("CREATE TABLE qt (id INT, s VARCHAR(20), n INT)")
cur.execute(f"LOAD DATA INFILE '{csv}' INTO TABLE qt FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '\"'")
c.eq("quoted fields, NULL, last line without newline", q("SELECT id, s, n FROM qt ORDER BY id"),
     [(1, "a, b", None), (2, 'say "hi"', 5), (3, "plain", 7)])
esc = path("e.tsv")
with open(esc, "w") as f:
    f.write("1\ttab\\there\n2\tback\\\\slash\n")
cur.execute("CREATE TABLE et (id INT, s VARCHAR(20))")
cur.execute(f"LOAD DATA INFILE '{esc}' INTO TABLE et FIELDS TERMINATED BY '\\t' ESCAPED BY '\\\\'")
c.eq("ESCAPED BY", q("SELECT s FROM et ORDER BY id"), [("tab\there",), ("back\\slash",)])

# 5) A bad value: the error names its line (counting the skipped header); earlier rows stay.
bad = path("bad.csv")
with open(bad, "w") as f:
    f.write("h\n")
    f.writelines(f"{i}\n" for i in range(300000))
    f.write("abc\n")
cur.execute("CREATE TABLE bad (m DECIMAL(5,1))")
try:
    cur.execute(f"LOAD DATA INFILE '{bad}' INTO TABLE bad IGNORE 1 LINES")
    c.check("bad decimal raises", False)
except Exception as e:
    c.check("bad decimal: error names line 300002", "300002" in str(e), str(e))
c.eq("rows before the bad line are kept", q("SELECT COUNT(*) FROM bad"), [(300000,)])

# 6) zstd-compressed input; an empty file.
zin = path("z.tsv")
with open(zin, "w") as f:
    f.writelines(f"{i}\tz{i % 10}\n" for i in range(100000))
subprocess.run(["zstd", "-q", "-f", "--rm", zin, "-o", zin + ".zst"], check=True)
cur.execute("CREATE TABLE zt (id INT, s VARCHAR(10))")
cur.execute(f"LOAD DATA INFILE '{zin}.zst' INTO TABLE zt FIELDS TERMINATED BY '\\t'")
c.eq("zstd input", q("SELECT COUNT(*), SUM(id), COUNT(DISTINCT s) FROM zt"), [(100000, 99999 * 100000 // 2, 10)])
empty = path("empty.tsv")
open(empty, "w").close()
cur.execute(f"LOAD DATA INFILE '{empty}' INTO TABLE zt")
c.eq("empty file", q("SELECT COUNT(*) FROM zt"), [(100000,)])

# 7) The WAL's copies are compressed; they replay after the sources are gone and the server is killed.
loads = os.path.join(srv.data_dir, "loads")
copies = sum(os.path.getsize(os.path.join(loads, f)) for f in os.listdir(loads))
c.check("compressed copies are much smaller than the big file", copies < os.path.getsize(big) / 2, (copies, os.path.getsize(big)))
for p in (big, pk, sk, csv, esc, bad, zin + ".zst", empty):
    os.unlink(p)
before = [q(f"SELECT COUNT(*) FROM {t}") for t in ("big", "pk", "sk", "qt", "et", "bad", "zt")]
srv.kill()
srv.start()
cur = srv.cursor()
cur.execute("USE l")
check_big("after crash replay")
c.eq("all tables after replay", [q(f"SELECT COUNT(*) FROM {t}") for t in ("big", "pk", "sk", "qt", "et", "bad", "zt")], before)
c.eq("pk table after replay", q("SELECT COUNT(*), MAX(id), SUM(score) FROM pk"), [(50000, 50000, 3 * 49999 * 50000 // 2)])
srv.stop()
c.done()
