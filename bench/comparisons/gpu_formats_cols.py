"""Writes 13 ClickBench columns (1M rows, sorted like warpstore's segments; strings as dictionary codes)
as raw binary files for gpu_formats.cu:
  python bench/comparisons/gpu_formats_cols.py OUT_DIR
  nvcc -O2 -std=c++17 -I$NVCOMP/include bench/comparisons/gpu_formats.cu -L$NVCOMP/lib64 -l:libnvcomp.so.5 -lzstd -o gpu_formats
  LD_LIBRARY_PATH=$NVCOMP/lib64 ./gpu_formats OUT_DIR
(NVCOMP=~/.local/share/warpstore-nvcomp/nvidia/libnvcomp)"""
import sys, duckdb, numpy as np
out = sys.argv[1]
d = duckdb.connect()
src = "(SELECT * FROM read_parquet('data/clickbench/hits_0.parquet') LIMIT 1048576)"
order = "CounterID, EventDate, UserID, EventTime, WatchID"
nums = {"WatchID": np.int64, "UserID": np.int64, "EventTime": np.int64, "ClientIP": np.int32, "CounterID": np.int32, "RegionID": np.int32,
        "ResolutionWidth": np.int32, "IsRefresh": np.int32, "EventDate": np.int32}
t = d.execute(f"SELECT {', '.join(nums)} FROM {src} ORDER BY {order}").fetchnumpy()
for c, ty in nums.items():
    np.asarray(t[c]).astype(ty).tofile(f"{out}/{c}.bin")
strs = ["URL", "SearchPhrase", "Title", "Referer"]
codes = ", ".join(f"(dense_rank() OVER (ORDER BY {c}) - 1)::INT AS {c}" for c in strs)
t = d.execute(f"SELECT {', '.join(strs)} FROM (SELECT {codes}, {order} FROM {src}) ORDER BY {order}").fetchnumpy()
for c in strs:
    np.asarray(t[c]).astype(np.int32).tofile(f"{out}/{c}.bin")
