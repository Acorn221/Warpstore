"""HTAP mixed workload: OLTP writers + point reads + GPU analytics on the same table, concurrently.

Measures write throughput, query latency percentiles, and freshness (an analytic query must see a
row immediately after the INSERT that wrote it returned).

usage: python bench/htap.py [--port 3307] [--rows 10000000] [--seconds 15] [--writers 4]
"""
import argparse
import os
import random
import statistics
import threading
import time

import duckdb
import pymysql


def conn(port):
    return pymysql.connect(host="127.0.0.1", port=port, user="root", autocommit=True).cursor()


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * p))] if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=3307)
    ap.add_argument("--rows", type=int, default=10_000_000)
    ap.add_argument("--seconds", type=float, default=15)
    ap.add_argument("--writers", type=int, default=4)
    a = ap.parse_args()

    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", f"events-{a.rows}.tbl")
    if not os.path.exists(path):
        duckdb.sql(f"""COPY (SELECT i, (i * 7919) % 100000 AS user_id, round(random() * 100, 2) AS amount,
                       DATE '2024-01-01' + (i % 365)::INT AS ts, ['web','ios','android','api'][1 + i % 4] AS channel
                       FROM range({a.rows}) t(i)) TO '{path}' (DELIMITER '|', HEADER false)""")
    c = conn(a.port)
    c.execute("DROP TABLE IF EXISTS events")
    c.execute("CREATE TABLE events (id BIGINT, user_id INT, amount DOUBLE, ts DATE, channel VARCHAR(10), "
              "PRIMARY KEY (id), SORT KEY (ts))")
    t = time.perf_counter()
    c.execute(f"LOAD DATA INFILE '{path}' INTO TABLE events FIELDS TERMINATED BY '|'")
    c.execute("OPTIMIZE TABLE events")
    c.fetchall()
    print(f"loaded {a.rows:,} rows in {time.perf_counter() - t:.1f}s")

    stop = time.perf_counter() + a.seconds
    next_id = [a.rows + 1]
    lock = threading.Lock()
    stats = {"insert": [], "update": [], "point": [], "agg": [], "stale": 0, "fresh_checks": 0}

    def writer(seed):
        w = conn(a.port)
        rnd = random.Random(seed)
        while time.perf_counter() < stop:
            if rnd.random() < 0.7:
                with lock:
                    i = next_id[0]
                    next_id[0] += 1
                t0 = time.perf_counter()
                w.execute(f"INSERT INTO events VALUES ({i}, {rnd.randrange(100000)}, {rnd.random() * 100:.2f}, "
                          f"'2025-01-{rnd.randrange(1, 29):02d}', 'web')")
                stats["insert"].append(time.perf_counter() - t0)
            else:
                t0 = time.perf_counter()
                w.execute(f"UPDATE events SET amount = amount + 1 WHERE id = {rnd.randrange(a.rows)}")
                stats["update"].append(time.perf_counter() - t0)

    def point_reader():
        r = conn(a.port)
        rnd = random.Random(7)
        while time.perf_counter() < stop:
            t0 = time.perf_counter()
            r.execute(f"SELECT user_id, amount, ts FROM events WHERE id = {rnd.randrange(a.rows)}")
            r.fetchall()
            stats["point"].append(time.perf_counter() - t0)

    def analyst():
        r = conn(a.port)
        w = conn(a.port)
        q = ("SELECT channel, count(*), sum(amount), avg(amount) FROM events "
             "WHERE ts >= DATE '2024-03-01' GROUP BY channel ORDER BY channel")
        marker = 10**12
        while time.perf_counter() < stop:
            t0 = time.perf_counter()
            r.execute(q)
            r.fetchall()
            stats["agg"].append(time.perf_counter() - t0)
            # Freshness: a row is visible to GPU analytics as soon as its INSERT returns.
            marker += 1
            w.execute(f"INSERT INTO events VALUES ({marker}, 1, 1000000, '2026-06-01', 'fresh')")
            r.execute("SELECT count(*) FROM events WHERE channel = 'fresh' AND amount = 1000000")
            stats["fresh_checks"] += 1
            if r.fetchone()[0] != marker - 10**12:
                stats["stale"] += 1

    threads = [threading.Thread(target=writer, args=(s,)) for s in range(a.writers)]
    threads += [threading.Thread(target=point_reader), threading.Thread(target=analyst)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    print(f"\n{a.seconds:.0f}s mixed workload, {a.writers} writer threads + 1 point reader + 1 analyst")
    for k in ("insert", "update", "point", "agg"):
        xs = stats[k]
        print(f"  {k:7} {len(xs) / a.seconds:9.0f} ops/s   p50 {pct(xs, .5) * 1000:7.2f} ms   p99 {pct(xs, .99) * 1000:7.2f} ms")
    print(f"  freshness: {stats['fresh_checks'] - stats['stale']}/{stats['fresh_checks']} analytic reads saw the "
          f"just-committed row")
    c.execute("SHOW SEGMENTS FROM events")
    segs = c.fetchall()
    # columns: partition, segment, tier, rows, ...
    nseg = sum(1 for r in segs if r[2] == "gpu-columnstore")
    nbuf = sum(int(r[3]) for r in segs if r[2] == "cpu-write-buffer")
    nparts = len({r[0] for r in segs})
    print(f"  storage: {nseg} GPU segments across {nparts} partitions, {nbuf} rows in write buffers")


if __name__ == "__main__":
    main()
