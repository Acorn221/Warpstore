# LOAD DATA INFILE: the parallel loader (several 64 MB chunks, a segment flush in the middle of a
# chunk), the row-at-a-time path (primary key, AUTO_INCREMENT), shard keys, CSV quoting, ESCAPED BY,
# NULLs, CRLF, IGNORE n LINES, error rows, zstd input, and replay of the WAL's compressed copy after
# the source file is deleted and the server killed.
import pymysql
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
# INSERT .. SELECT into AUTO_INCREMENT / PRIMARY KEY / NOT NULL columns.
cur.execute("CREATE TABLE ai (id INT AUTO_INCREMENT PRIMARY KEY, v INT NOT NULL)")
cur.execute("INSERT INTO ai (v) SELECT score FROM pk WHERE id <= 1000")
cur.execute("INSERT INTO ai SELECT NULL, score FROM pk WHERE id <= 10")
c.eq("INSERT .. SELECT numbers AUTO_INCREMENT", q("SELECT COUNT(*), MIN(id), MAX(id), SUM(v) FROM ai"),
     [(1010, 1, 1010, 3 * 999 * 1000 // 2 + 3 * 9 * 10 // 2)])
c.raises("INSERT .. SELECT duplicate key", lambda: cur.execute("INSERT INTO ai SELECT id, score FROM pk WHERE id <= 5"), 1062)
c.raises("INSERT .. SELECT NULL into NOT NULL", lambda: cur.execute("INSERT INTO ai (v) SELECT NULL FROM pk WHERE id <= 5"), 1048)
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

# 7) Parquet, decoded straight into columns: every physical type, dates, timestamps (ms / us / ns),
# decimals, NULLs, small dictionaries, several row groups and codecs, absent columns, a column list;
# nested columns go through text (JSON).
import datetime as dt
import pyarrow as pa, pyarrow.parquet as pq
PN = 30000
prng = random.Random(11)
pdata = {
    "id": pa.array(list(range(PN)), pa.int64()),
    "i32": pa.array([prng.randint(-2**31, 2**31 - 1) if i % 13 else None for i in range(PN)], pa.int32()),
    "f": pa.array([i / 4 for i in range(PN)], pa.float32()),
    "d": pa.array([prng.uniform(-1e6, 1e6) if i % 17 else None for i in range(PN)], pa.float64()),
    "b": pa.array([i % 3 == 0 for i in range(PN)], pa.bool_()),
    "tiny": pa.array([["5", "g", "6"][i % 3] for i in range(PN)], pa.string()),  # (a dictionary page of a few bytes)
    "s": pa.array([f"str{prng.randint(0, 5000)}" if i % 7 else None for i in range(PN)], pa.string()),
    "day": pa.array([dt.date(2020, 1, 1) + dt.timedelta(days=i % 1500) for i in range(PN)], pa.date32()),
    "tms": pa.array([dt.datetime(2021, 3, 4, 5, 6, 7) + dt.timedelta(milliseconds=i * 1000) for i in range(PN)], pa.timestamp("ms")),
    "tus": pa.array([dt.datetime(2022, 1, 1) + dt.timedelta(microseconds=i * 1000001) for i in range(PN)], pa.timestamp("us")),
    "tns": pa.array([dt.datetime(2023, 6, 1) + dt.timedelta(seconds=i) for i in range(PN)], pa.timestamp("ns")),
    "dec": pa.array([Decimal(f"{prng.randint(-10**8, 10**8)}.{prng.randint(0, 99):02d}") for i in range(PN)], pa.decimal128(12, 2)),
    "dec4": pa.array([Decimal(f"{i % 1000}.{i % 10000:04d}") for i in range(PN)], pa.decimal128(8, 4)),
}
ptab = pa.table(pdata)
cur.execute("CREATE TABLE pq (id BIGINT, i32 INT, f DOUBLE, d DOUBLE, b INT, tiny VARCHAR(4), s VARCHAR(20), day DATE, "
            "tms DATETIME, tus DATETIME, tns DATETIME, `dec` DECIMAL(12,2), dec4 DECIMAL(10,2), extra INT)")
pqfiles = []
for codec in ("snappy", "zstd", "gzip", "none"):
    pf = path(f"p-{codec}.parquet")
    pq.write_table(ptab, pf, compression=codec, row_group_size=7000, use_dictionary=codec != "none")
    pqfiles.append(pf)
    cur.execute(f"LOAD DATA INFILE '{pf}' INTO TABLE pq")


def pq_expect(k):
    rows = ptab.to_pylist()
    i32s = [r["i32"] for r in rows if r["i32"] is not None]
    ds = [r["d"] for r in rows if r["d"] is not None]
    return (k * PN, k * len(i32s), k * sum(i32s), k * sum(1 for r in rows if r["b"]), k * len(ds),
            len({r["s"] for r in rows if r["s"] is not None}), sorted({r["tiny"] for r in rows}))


def pq_got():
    r = q("SELECT COUNT(*), COUNT(i32), SUM(i32), SUM(b), COUNT(d), COUNT(DISTINCT s) FROM pq")[0]
    tiny = sorted(x[0] for x in q("SELECT DISTINCT tiny FROM pq"))
    return (r[0], r[1], int(r[2]), int(r[3]), r[4], r[5], tiny)


c.eq("Parquet: counts, NULLs, sums, small dictionaries (4 codecs)", pq_got(), pq_expect(4))
c.check("Parquet: double sum", abs(q("SELECT SUM(d) FROM pq")[0][0] - 4 * sum(x for x in pdata["d"].to_pylist() if x is not None)) < 1e-3)
sample = [0, 1, 13, 777, 29999]
got = [a + b[1:] for a, b in zip(q("SELECT DISTINCT id, f, day, tms, tus FROM pq WHERE id IN (0, 1, 13, 777, 29999) ORDER BY id"),
                                  q("SELECT DISTINCT id, tns, `dec`, dec4, extra, s FROM pq WHERE id IN (0, 1, 13, 777, 29999) ORDER BY id"))]
want = []
for i in sample:
    r = ptab.slice(i, 1).to_pylist()[0]
    want.append((i, float(r["f"]), r["day"], r["tms"], r["tus"].replace(microsecond=r["tus"].microsecond), r["tns"].to_pydatetime() if hasattr(r["tns"], "to_pydatetime") else r["tns"],
                 r["dec"], r["dec4"].quantize(Decimal("0.01")), None, r["s"]))
c.eq("Parquet: values (float, date, timestamps, decimals, absent column, NULL string)", [tuple(g) for g in got], want)
# a column list, by name
cur.execute("CREATE TABLE pq2 (a BIGINT, z VARCHAR(10), s VARCHAR(20))")
cur.execute(f"LOAD DATA INFILE '{pqfiles[0]}' INTO TABLE pq2 (a, s)")
c.eq("Parquet: column list", q("SELECT COUNT(*), COUNT(a), COUNT(z), COUNT(s) FROM pq2"), [(PN, 0, 0, sum(1 for x in pdata['s'].to_pylist() if x is not None))])
# NOT NULL violated by a NULL in the file
cur.execute("CREATE TABLE pq3 (i32 INT NOT NULL)")
c.raises("Parquet: NULL into NOT NULL", lambda: cur.execute(f"LOAD DATA INFILE '{pqfiles[1]}' INTO TABLE pq3"), 1048)
# plain integers (no date / timestamp type, as in ClickBench's hits files): Unix seconds into
# DATETIME, days since 1970-01-01 into DATE
epochs = path("epochs.parquet")
pq.write_table(pa.table({"t": pa.array([1372796926, 0, None], pa.int64()), "d": pa.array([15888, 0, 1], pa.uint16())}), epochs)
cur.execute("CREATE TABLE pqe (t DATETIME, d DATE)")
cur.execute(f"LOAD DATA INFILE '{epochs}' INTO TABLE pqe")
c.eq("Parquet: integer Unix seconds / days into DATETIME / DATE", q("SELECT t, d FROM pqe ORDER BY d"),
     [(dt.datetime(1970, 1, 1), dt.date(1970, 1, 1)), (None, dt.date(1970, 1, 2)), (dt.datetime(2013, 7, 2, 20, 28, 46), dt.date(2013, 7, 2))])
# nested: a list column loads as JSON text
nested = path("nested.parquet")
pq.write_table(pa.table({"k": pa.array([1, 2, 3]), "tags": pa.array([["a", "b"], [], None], pa.list_(pa.string()))}), nested)
cur.execute("CREATE TABLE pqn (k INT, tags VARCHAR(40))")
cur.execute(f"LOAD DATA INFILE '{nested}' INTO TABLE pqn")
c.eq("Parquet: nested column as JSON", q("SELECT k, tags FROM pqn ORDER BY k"), [(1, '["a","b"]'), (2, "[]"), (3, None)])
# a damaged page (page checksums on) in the last row group: an error naming it after the earlier row
# groups loaded, which stay and replay (below)
dmg = path("damaged.parquet")
pq.write_table(pa.table({"a": pa.array(range(30000)), "s": pa.array([f"v{i}" for i in range(30000)])}), dmg,
               row_group_size=10000, compression="snappy", write_page_checksum=True, use_dictionary=False)
meta = pq.ParquetFile(dmg).metadata
off = meta.row_group(2).column(0).data_page_offset + meta.row_group(2).column(0).total_compressed_size // 2
with open(dmg, "r+b") as fh:
    fh.seek(off)
    b = fh.read(1)
    fh.seek(off)
    fh.write(bytes([b[0] ^ 0x5A]))
cur.execute("CREATE TABLE dmg (a INT, s VARCHAR(10))")
try:
    cur.execute(f"LOAD DATA INFILE '{dmg}' INTO TABLE dmg")
    c.check("Parquet: damaged page detected", False, "loaded")
except pymysql.MySQLError as e:
    c.check("Parquet: damaged page detected", "damaged" in str(e) or "bad snappy" in str(e), str(e))
c.eq("Parquet: rows before the damaged row group", q("SELECT COUNT(*), SUM(a) FROM dmg"), [(20000, sum(range(20000)))])

# 8) The WAL's copies are compressed; they replay after the sources are gone and the server is killed.
loads = os.path.join(srv.data_dir, "loads")
copies = sum(os.path.getsize(os.path.join(loads, f)) for f in os.listdir(loads))
c.check("compressed copies are much smaller than the big file", copies < os.path.getsize(big) / 2, (copies, os.path.getsize(big)))
for p in [big, pk, sk, csv, esc, bad, zin + ".zst", empty, nested, dmg] + pqfiles:
    os.unlink(p)
pq_before = pq_got()
before = [q(f"SELECT COUNT(*) FROM {t}") for t in ("big", "pk", "sk", "qt", "et", "bad", "zt", "ai", "pq", "pq2", "pqn", "dmg")]
srv.kill()
srv.start()
cur = srv.cursor()
cur.execute("USE l")
check_big("after crash replay")
c.eq("all tables after replay", [q(f"SELECT COUNT(*) FROM {t}") for t in ("big", "pk", "sk", "qt", "et", "bad", "zt", "ai", "pq", "pq2", "pqn", "dmg")], before)
c.eq("Parquet tables after replay", pq_got(), pq_before)
c.eq("pk table after replay", q("SELECT COUNT(*), MAX(id), SUM(score) FROM pk"), [(50000, 50000, 3 * 49999 * 50000 // 2)])
srv.stop()

# 9) A damaged load copy (one flipped byte) is refused at replay, not loaded as other rows; the
# escape hatch starts without it.
copies = sorted(f for f in os.listdir(loads) if f.endswith(".parquet") and "-partial" not in f)
c.check("load copies carry their CRC", copies and all("-c" in f for f in copies), copies)
victim = os.path.join(loads, copies[0])
with open(victim, "r+b") as fh:
    fh.seek(os.path.getsize(victim) // 2)
    b = fh.read(1)
    fh.seek(-1, 1)
    fh.write(bytes([b[0] ^ 0x20]))
try:
    srv.start(timeout=120)
    c.check("damaged load copy refused", False, "server started")
    srv.stop()
except RuntimeError as e:
    c.check("damaged load copy refused", "CRC mismatch" in str(e) or "replay failed" in str(e), str(e)[-300:])
os.environ["WARPSTORE_REPLAY_IGNORE_ERRORS"] = "1"
srv.start()
del os.environ["WARPSTORE_REPLAY_IGNORE_ERRORS"]
c.check("starts with WARPSTORE_REPLAY_IGNORE_ERRORS=1", True)
srv.stop()
c.done()
