"""Prints the README's comparison tables (markdown) from the result files in this directory."""
import json, math, os

H = os.path.dirname(os.path.abspath(__file__))
cl = lambda x: max(x, 1e-4)
gm = lambda xs: math.exp(sum(math.log(x) for x in xs) / len(xs))
ms = lambda s: f"{s * 1000:.1f}" if s < 10 else f"{s * 1000:.0f}"


def jl(p):
    return [json.loads(l) for l in open(os.path.join(H, p)) if l.startswith("{")]


W = {r["q"]: cl(r["warpstore"]) for r in jl("clickbench_warpstore_hot.jsonl")}
WF = {r["q"]: cl(r["warpstore_first"]) for r in jl("clickbench_warpstore_first.jsonl")}
WC = {r["q"]: cl(r["warpstore_first"]) for r in jl("clickbench_warpstore_hot.jsonl")}  # first run, cold plan cache
D = {r["q"]: cl(r["hot"]) for r in jl("clickbench_duckdb.jsonl")}
DF = {r["q"]: cl(r["first"]) for r in jl("clickbench_duckdb.jsonl")}
CH, CHF = {}, {}
for l in open(os.path.join(H, "clickbench_clickhouse.txt")):
    p = l.split()
    CHF[int(p[0])], CH[int(p[0])] = cl(float(p[1])), cl(min(map(float, p[2:5])))
S2, S2F = {}, {}
for r in jl("../clickbench/results_singlestore_local.jsonl"):
    S2[r["q"]], S2F[r["q"]] = cl(r["s2"]), cl(r["s2_first"])
Q = sorted(W)

print("| System | Version / setup | All 43 queries, hot | All 43, first run | Warpstore faster, hot (geomean) | Warpstore faster, first run (geomean) | Queries Warpstore wins (hot) |")
print("|---|---|---:|---:|---:|---:|---:|")
print(f"| **Warpstore** | RTX 3080 10 GB | **{sum(W.values()):.2f} s** | **{sum(WF.values()):.2f} s** ({sum(WC.values()):.1f} s with a cold plan cache) | — | — | — |")
for name, setup, h, f in [("DuckDB", "1.5.6, 12 threads", D, DF), ("ClickHouse", "25.8, 12 threads, ClickBench schema, Docker", CH, CHF),
                          ("SingleStore", "9.1 free dev image, 8 threads, Docker", S2, S2F)]:
    print(f"| {name} | {setup} | {sum(h.values()):.2f} s | {sum(f.values()):.2f} s | {gm([h[q] / W[q] for q in Q]):.1f}x | "
          f"{gm([f[q] / WF[q] for q in Q]):.1f}x | {sum(1 for q in Q if W[q] < h[q])} / 43 |")

print("\n<details><summary>Per query (milliseconds)</summary>\n")
print("| Q | Warpstore hot | Warpstore first | DuckDB hot | DuckDB first | ClickHouse hot | ClickHouse first | SingleStore hot | SingleStore first |")
print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for q in Q:
    print(f"| {q} | {ms(W[q])} | {ms(WF[q])} | {ms(D[q])} | {ms(DF[q])} | {ms(CH[q])} | {ms(CHF[q])} | {ms(S2[q])} | {ms(S2F[q])} |")
print("\n</details>\n")

m = json.load(open(os.path.join(H, "mysql_dashboards.json")))["rows"]
print("| Dashboard query | MySQL 8.4, hot | MySQL, first run | Warpstore, hot | Warpstore, first run | Warpstore faster (hot) |")
print("|---|---:|---:|---:|---:|---:|")
for name, mh, mf, wh, wf in m:
    print(f"| {name} | {mh:,.1f} ms | {mf:,.0f} ms | {wh:.2f} ms | {wf:.0f} ms | {mh / wh:,.0f}x |")

def pub(s):
    d = json.load(open(os.path.join(H, "published", s + ".json")))
    hot = {}
    for i, r in enumerate(d["result"], 1):
        v = [x for x in r[1:] if x is not None]
        if v:
            hot[i] = cl(min(v))
    return d, hot

_, DP = pub("duckdb")
rows = []
for s in ["umbra", "clickhouse", "heavyai", "starrocks", "singlestore", "doris", "kinetica", "tidb", "mysql", "postgresql"]:
    d, X = pub(s)
    qs = [q for q in Q if q in X and q in DP]
    rows.append((gm([(D[q] / W[q]) * (X[q] / DP[q]) for q in qs]), d, len(qs), gm([X[q] / DP[q] for q in qs])))
print("\n| System | Published result | AWS machine (CPU) | Tuned | Queries compared | Its time vs DuckDB's (geomean) | Estimated Warpstore speedup |")
print("|---|---|---|---|---:|---:|---:|")
for est, d, n, vs in sorted(rows, key=lambda r: r[0]):
    print(f"| {d['system']} | {d['date']} | {d['machine']} | {d.get('tuned', 'no')} | {n} / 43 | {vs:.2f}x | ~{est:,.0f}x |")

d, S = pub("sirius")
sf = sum(r[0] for r in d["result"] if r[0])
print("\n| System | Hardware | Rows | All 43 queries, hot | All 43, first run | Per query vs Warpstore (scaled to 100M rows) |")
print("|---|---|---:|---:|---:|---:|")
print(f"| Sirius ({d['date']}) | NVIDIA H100 (AWS {d['machine']}) | 100M | {sum(S.values()):.2f} s | {sf:.2f} s | {gm([S[q] / (10 * W[q]) for q in Q]):.2f}x Warpstore's time |")
print(f"| Warpstore | RTX 3080 10 GB | 10M | {sum(W.values()):.2f} s | {sum(WF.values()):.2f} s | — (x10 rows: {10 * sum(W.values()):.1f} s hot) |")

# Disk tier: the table (3.8 GB of segment files) larger than --memory-limit
print()
print("| Warpstore setup | All 43 queries | Geomean | vs DuckDB hot (geomean) | vs ClickHouse hot (geomean) |")
print("|---|---:|---:|---:|---:|")
for label, f, k in [("everything in memory", "clickbench_warpstore_tier_2gb.jsonl", "memory"),
                    ("`--memory-limit 2`, files in the OS page cache", "clickbench_warpstore_tier_2gb.jsonl", "limited_pagecache"),
                    ("`--memory-limit 2`, read from the SSD", "clickbench_warpstore_tier_2gb.jsonl", "limited_ssd"),
                    ("`--memory-limit 0.25`, files in the OS page cache", "clickbench_warpstore_tier_0.25gb.jsonl", "limited_pagecache"),
                    ("`--memory-limit 0.25`, read from the SSD", "clickbench_warpstore_tier_0.25gb.jsonl", "limited_ssd")]:
    T = {r["q"]: cl(r[k]) for r in jl(f)}
    print(f"| {label} | {sum(T.values()):.2f} s | {gm(list(T.values())) * 1000:.1f} ms | {gm([D[q] / T[q] for q in Q]):.1f}x faster | {gm([CH[q] / T[q] for q in Q]):.1f}x faster |")
