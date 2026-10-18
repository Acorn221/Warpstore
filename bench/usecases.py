"""SingleStore's main use cases as small workloads, run the same way against warpstore or SingleStore.

Use cases: real-time dashboards (latency + concurrency), operational point reads / updates,
streaming ingest, HTAP (dashboards while writes run), vector search (exact, filtered, ANN), full-text
search, JSON analytics.

usage: .venv/bin/python bench/usecases.py --port 3307 --name warpstore [--load]
       .venv/bin/python bench/usecases.py --port 3316 --password bench --name s2 --docker s2bench [--load]
Prints one JSON line per measurement.
"""
import argparse
import json
import multiprocessing as mp
import os
import random
import subprocess
import time

import numpy as np
import pymysql

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "usecases")
EVENTS, DOCS, DIM = 5_000_000, 100_000, 128
CHANNELS = ["web", "ios", "android", "api"]
COUNTRIES = ["US", "GB", "DE", "FR", "IN", "BR", "JP", "CA", "AU", "ES", "IT", "NL", "SE", "MX", "KR"]
A = None  # parsed args (set in main, inherited by forked workers)


def conn():
    c = pymysql.connect(host="127.0.0.1", port=A.port, user="root", password=A.password, autocommit=True,
                        read_timeout=600, local_infile=True)
    cur = c.cursor()
    cur.execute("USE uc")
    return cur


def out(**kw):
    kw["sys"] = A.name
    print(json.dumps(kw), flush=True)


def gen():
    os.makedirs(DATA, exist_ok=True)
    ev = os.path.join(DATA, "events.tsv")
    if not os.path.exists(ev):
        rng = np.random.default_rng(1)
        n = EVENTS
        user = rng.zipf(1.3, n) % 200_000
        ts = rng.integers(0, 180 * 86400, n)
        ch = rng.integers(0, 4, n)
        co = rng.integers(0, len(COUNTRIES), n)
        amt = rng.integers(1, 100_000, n)
        dev = rng.integers(0, 6, n)
        base = np.datetime64("2024-01-01T00:00:00")
        with open(ev, "w") as f:
            for i in range(n):
                t = str(base + np.timedelta64(int(ts[i]), "s")).replace("T", " ")
                f.write(f"{i}\t{user[i]}\t{t}\t{CHANNELS[ch[i]]}\t{COUNTRIES[co[i]]}\t{amt[i] // 100}.{amt[i] % 100:02d}"
                        f'\t{{"device":"d{dev[i]}","ver":{i % 7}}}\n')
    rng = np.random.default_rng(2)
    vocab = [f"w{i}" for i in range(5000)]
    words = rng.zipf(1.2, (DOCS, 30)) % 5000
    vecs = rng.standard_normal((DOCS, DIM)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    docs = [(i, i % 10, " ".join(vocab[w] for w in words[i])) for i in range(DOCS)]
    return ev, docs, vecs


def load(ev, docs, vecs):
    c = pymysql.connect(host="127.0.0.1", port=A.port, user="root", password=A.password, autocommit=True,
                        read_timeout=3600, local_infile=True).cursor()
    c.execute("DROP DATABASE IF EXISTS uc")
    c.execute("CREATE DATABASE uc")
    c.execute("USE uc")
    c.execute("""CREATE TABLE events (id BIGINT NOT NULL, user_id INT, ts DATETIME, channel VARCHAR(16),
                 country VARCHAR(4), amount DECIMAL(12,2), payload JSON, PRIMARY KEY (id), SHARD KEY (id),
                 SORT KEY (ts))""")
    c.execute(f"""CREATE TABLE docs (id BIGINT NOT NULL, cat INT, body TEXT, v VECTOR({DIM}), PRIMARY KEY (id),
                 SHARD KEY (id), FULLTEXT USING VERSION 2 ft (body))""")
    path = ev
    if A.docker:
        subprocess.run(["docker", "cp", ev, f"{A.docker}:/tmp/events.tsv"], check=True)
        path = "/tmp/events.tsv"
    t = time.perf_counter()
    c.execute(f"LOAD DATA INFILE '{path}' INTO TABLE events FIELDS TERMINATED BY '\\t'")
    out(test="load_events_5M", s=round(time.perf_counter() - t, 2))
    t = time.perf_counter()
    for i in range(0, DOCS, 1000):
        vals = ",".join(f"({d[0]},{d[1]},'{d[2]}','[{','.join(f'{x:.6f}' for x in vecs[d[0]])}]')"
                        for d in docs[i:i + 1000])
        c.execute("INSERT INTO docs VALUES " + vals)
    out(test="load_docs_100K_insert", s=round(time.perf_counter() - t, 2))
    t = time.perf_counter()
    c.execute("OPTIMIZE TABLE events FULL" if A.docker else "OPTIMIZE TABLE events")
    c.execute("OPTIMIZE TABLE docs FULL" if A.docker else "OPTIMIZE TABLE docs")
    out(test="optimize", s=round(time.perf_counter() - t, 2))


DASH = {
    "dash_channel_sum": "SELECT channel, COUNT(*), SUM(amount) FROM events WHERE ts >= '2024-05-01' GROUP BY channel ORDER BY channel",
    "dash_distinct_users": "SELECT country, COUNT(DISTINCT user_id) c FROM events GROUP BY country ORDER BY c DESC, country LIMIT 10",
    "dash_daily_series": "SELECT TIME_BUCKET('1d', ts) b, COUNT(*), AVG(amount) FROM events WHERE channel = 'ios' GROUP BY b ORDER BY b",
    "dash_top_users": "SELECT user_id, SUM(amount) s FROM events GROUP BY user_id ORDER BY s DESC, user_id LIMIT 10",
    "dash_last_hour": "SELECT country, channel, COUNT(*) FROM events WHERE ts BETWEEN '2024-06-15 10:00:00' AND '2024-06-15 11:00:00' GROUP BY country, channel ORDER BY 3 DESC, 1, 2 LIMIT 5",
    "json_device": "SELECT JSON_EXTRACT_STRING(payload, 'device') d, COUNT(*), SUM(amount) FROM events GROUP BY d ORDER BY d",
    "window_rank": "SELECT country, user_id, s FROM (SELECT country, user_id, SUM(amount) s, RANK() OVER (PARTITION BY country ORDER BY SUM(amount) DESC) r FROM events GROUP BY country, user_id) t WHERE r = 1 ORDER BY country",
}


def timed(c, q, reps=5):
    t = time.perf_counter(); c.execute(q); rows = c.fetchall(); first = time.perf_counter() - t
    best = first
    for _ in range(reps - 1):
        t = time.perf_counter(); c.execute(q); c.fetchall(); best = min(best, time.perf_counter() - t)
    return first, best, rows


def single_latency():
    c = conn()
    for name, q in DASH.items():
        try:
            first, best, rows = timed(c, q)
            out(test=name, first_ms=round(first * 1000, 2), ms=round(best * 1000, 2), rows=len(rows),
                head=str(rows[:2])[:120])
        except Exception as e:
            out(test=name, err=str(e)[:200])


def worker(kind, secs, seed, q, base):
    random.seed(seed)
    c = conn()
    lat, n, end = [], 0, time.time() + secs
    nid = base + seed * 10_000_000
    qs = list(DASH.values())[:5]
    err = 0
    while time.time() < end:
        t = time.perf_counter()
        try:
            step(c, kind, qs, nid)
        except pymysql.MySQLError:
            err += 1
        nid += 1000 if kind == "batch" else 1
        lat.append(time.perf_counter() - t); n += 1
    q.put((kind, n, lat))


def step(c, kind, qs, nid):
    if True:
        if kind == "point":
            c.execute(f"SELECT * FROM events WHERE id = {random.randrange(EVENTS)}"); c.fetchall()
        elif kind == "update":
            c.execute(f"UPDATE events SET amount = amount + 1 WHERE id = {random.randrange(EVENTS)}")
        elif kind == "insert1":
            c.execute(f"INSERT INTO events VALUES ({nid}, 7, '2024-06-30 12:00:00', 'web', 'US', 1.50, '{{\"device\":\"d1\",\"ver\":1}}')")
        elif kind == "batch":
            c.execute("INSERT INTO events VALUES " + ",".join(
                f"({nid + k}, {k}, '2024-06-30 12:00:00', 'api', 'DE', 2.25, '{{\"device\":\"d2\",\"ver\":2}}')" for k in range(1000)))
        elif kind == "dash":
            c.execute(random.choice(qs)); c.fetchall()


RUN = [int(time.time()) % 1000]


def run_mix(test, mix, secs=10):
    RUN[0] += 1
    q = mp.Queue()
    ps = [mp.Process(target=worker, args=(k, secs, i, q, RUN[0] * 1_000_000_000)) for i, k in enumerate(mix)]
    for p in ps: p.start()
    res = [q.get() for _ in ps]
    for p in ps: p.join()
    by = {}
    for k, n, lat in res:
        by.setdefault(k, [0, []]); by[k][0] += n; by[k][1] += lat
    for k, (n, lat) in by.items():
        lat.sort()
        mult = 1000 if k == "batch" else 1
        out(test=test, kind=k, clients=mix.count(k), per_s=round(n * mult / secs), p50_ms=round(lat[len(lat) // 2] * 1000, 2),
            p99_ms=round(lat[int(len(lat) * 0.99)] * 1000, 2))


def vector_tests(vecs):
    c = conn()
    rng = np.random.default_rng(3)
    qv = rng.standard_normal((20, DIM)).astype(np.float32)
    qv /= np.linalg.norm(qv, axis=1, keepdims=True)
    exact = [set(np.argsort(-(vecs @ v))[:10].tolist()) for v in qv]
    lit = lambda v: "'[" + ",".join(f"{x:.6f}" for x in v) + "]'"
    cast = (lambda v: f"{lit(v)}:>VECTOR({DIM})") if A.docker else lit

    def bench(test, sql_of, check=True):
        ts, rec = [], []
        for i, v in enumerate(qv):
            t = time.perf_counter(); c.execute(sql_of(v)); rows = c.fetchall(); ts.append(time.perf_counter() - t)
            if check: rec.append(len(exact[i] & {r[0] for r in rows}) / 10)
        ts.sort()
        out(test=test, p50_ms=round(ts[len(ts) // 2] * 1000, 2), recall=round(sum(rec) / len(rec), 3) if rec else None)

    if not A.docker: c.execute("SET warpstore_vector_index = OFF")
    try:
        bench("vector_exact_top10", lambda v: f"SELECT id FROM docs ORDER BY v <*> {cast(v)} DESC LIMIT 10")
        bench("vector_filtered_top10", lambda v: f"SELECT id FROM docs WHERE cat = 3 ORDER BY v <*> {cast(v)} DESC LIMIT 10", False)
        bench("hybrid_text_vector", lambda v: s2fts("SELECT id FROM docs WHERE MATCH(body) AGAINST('w17')") + f" ORDER BY v <*> {cast(v)} DESC LIMIT 10", False)
    except Exception as e:
        out(test="vector_exact", err=str(e)[:200])
    try:
        t = time.perf_counter()
        c.execute("ALTER TABLE docs ADD VECTOR INDEX vi (v) INDEX_OPTIONS '{\"index_type\":\"IVF_FLAT\",\"metric_type\":\"DOT_PRODUCT\",\"nlist\":316}'")
        if not A.docker:
            c.execute("SET warpstore_vector_index = ON"); c.execute("OPTIMIZE TABLE docs")
        out(test="vector_index_build", s=round(time.perf_counter() - t, 2))
        bench("vector_ann_top10", lambda v: f"SELECT id FROM docs ORDER BY v <*> {cast(v)} DESC LIMIT 10")
    except Exception as e:
        out(test="vector_ann", err=str(e)[:200])


def s2fts(q):
    # SingleStore's FULLTEXT VERSION 2 index is queried as MATCH (TABLE t) AGAINST ('col:term ..')
    import re
    if not A.docker: return q
    return re.sub(r"MATCH\(body\) AGAINST\('([^']*)'\)",
                  lambda m: "MATCH(TABLE docs) AGAINST('" + " ".join("body:" + w for w in m.group(1).split()) + "')", q)


def fulltext_tests():
    c = conn()
    for name, q in {
        "fts_count_common": "SELECT COUNT(*) FROM docs WHERE MATCH(body) AGAINST('w3')",
        "fts_count_rare": "SELECT COUNT(*) FROM docs WHERE MATCH(body) AGAINST('w4321')",
        "fts_top10_scored": "SELECT id, MATCH(body) AGAINST('w42 w99') s FROM docs WHERE MATCH(body) AGAINST('w42 w99') ORDER BY s DESC, id LIMIT 10",
        "fts_group_by_cat": "SELECT cat, COUNT(*) FROM docs WHERE MATCH(body) AGAINST('w250') GROUP BY cat ORDER BY cat",
    }.items():
        try:
            first, best, rows = timed(c, s2fts(q))
            out(test=name, first_ms=round(first * 1000, 2), ms=round(best * 1000, 2), head=str(rows[:2])[:100])
        except Exception as e:
            out(test=name, err=str(e)[:200])


def main():
    global A
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=3307)
    ap.add_argument("--password", default="")
    ap.add_argument("--name", default="warpstore")
    ap.add_argument("--docker", default="")
    ap.add_argument("--load", action="store_true")
    ap.add_argument("--only", default="")
    A = ap.parse_args()
    ev, docs, vecs = gen()
    if A.load: load(ev, docs, vecs)
    only = A.only.split(",") if A.only else ["dash", "oltp", "vector", "fts"]
    if "dash" in only:
        single_latency()
        run_mix("dash_concurrency", ["dash"] * 16)
    if "oltp" in only:
        run_mix("point_reads", ["point"] * 16)
        run_mix("point_updates", ["update"] * 16)
        run_mix("ingest_single_rows", ["insert1"] * 8)
        run_mix("ingest_batches", ["batch"] * 4)
        run_mix("htap", ["insert1"] * 4 + ["update"] * 4 + ["point"] * 4 + ["dash"] * 4)
    if "htap" in only and "oltp" not in only:
        run_mix("htap", ["insert1"] * 4 + ["update"] * 4 + ["point"] * 4 + ["dash"] * 4)
    if "vector" in only: vector_tests(vecs)
    if "fts" in only: fulltext_tests()


if __name__ == "__main__":
    mp.set_start_method("fork")
    main()
