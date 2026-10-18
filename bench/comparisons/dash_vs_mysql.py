import os, sys, time, pymysql
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tests"))
from lib import Server
Q = {
 "channel totals since May": "SELECT channel, COUNT(*), SUM(amount) FROM events WHERE ts >= '2024-05-01' GROUP BY channel ORDER BY channel",
 "distinct users per country": "SELECT country, COUNT(DISTINCT user_id) c FROM events GROUP BY country ORDER BY c DESC, country LIMIT 10",
 "daily series (ios)": "SELECT DATE(ts) b, COUNT(*), AVG(amount) FROM events WHERE channel = 'ios' GROUP BY b ORDER BY b",
 "top 10 users by spend": "SELECT user_id, SUM(amount) s FROM events GROUP BY user_id ORDER BY s DESC, user_id LIMIT 10",
 "one hour, by country+channel": "SELECT country, channel, COUNT(*) FROM events WHERE ts BETWEEN '2024-06-15 10:00:00' AND '2024-06-15 11:00:00' GROUP BY country, channel ORDER BY 3 DESC, 1, 2 LIMIT 5",
 "JSON field group-by": "SELECT payload->>'$.device' d, COUNT(*), SUM(amount) FROM events GROUP BY d ORDER BY d",
 "window rank per country": "SELECT country, user_id, s FROM (SELECT country, user_id, SUM(amount) s, RANK() OVER (PARTITION BY country ORDER BY SUM(amount) DESC) r FROM events GROUP BY country, user_id) t WHERE r = 1 ORDER BY country",
}
def bench(cur, reps):
    out = {}
    for name, q in Q.items():
        t = time.perf_counter(); cur.execute(q); cur.fetchall(); first = time.perf_counter() - t
        best = first
        for _ in range(reps):
            t = time.perf_counter(); cur.execute(q); rows = cur.fetchall(); best = min(best, time.perf_counter() - t)
        out[name] = (first, best, rows)
    return out
my = pymysql.connect(host="127.0.0.1", port=13307, user="root", password="bench", database="uc", read_timeout=3600).cursor()
m = bench(my, 2)
with Server(data_dir=None, http=False) as s:
    c = s.cursor(); c.execute("CREATE DATABASE uc"); c.execute("USE uc")
    c.execute("CREATE TABLE events (id BIGINT NOT NULL, user_id INT, ts DATETIME, channel VARCHAR(16), country VARCHAR(4), amount DECIMAL(12,2), payload JSON, PRIMARY KEY (id), SORT KEY (ts))")
    c.execute("LOAD DATA INFILE 'data/usecases/events.tsv' INTO TABLE events FIELDS TERMINATED BY '\\t'")
    c.execute("OPTIMIZE TABLE events")
    w = bench(c, 5)
print(f"{'query':32s} {'MySQL 8.4':>11s} {'Warpstore':>11s} {'speedup':>8s}  (hot; first run in brackets)")
for name in Q:
    mf, mb, mr = m[name]; wf, wb, wr = w[name]
    same = [tuple(str(round(float(x), 2)) if isinstance(x, (float,)) or type(x).__name__ == "Decimal" else str(x) for x in r) for r in mr] == \
           [tuple(str(round(float(x), 2)) if isinstance(x, (float,)) or type(x).__name__ == "Decimal" else str(x) for x in r) for r in wr]
    print(f"{name:32s} {mb*1000:9.1f}ms {wb*1000:9.2f}ms {mb/wb:7.0f}x  ({mf*1000:.0f} / {wf*1000:.0f} ms){'' if same else '  [results differ]'}")
