# Contributing

Thanks for looking! gpu-db is alpha software; bug reports with a reproducing SQL script are the most
useful contribution.

## Build and test

```bash
sudo scripts/install-deps.sh          # Debian / Ubuntu build dependencies (CUDA 12.x separately)
cmake -S . -B build && cmake --build build -j
python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
.venv/bin/python tests/run.py         # ~2-3 min; tests that need missing tools are skipped
scripts/test-containers.sh            # optional: S3 / Kafka / MySQL containers for the ingest and CDC tests
```

`tests/run.py --list` shows the tests; pass names to run a subset. Before sending a change to the
engine, run the whole suite and a longer fuzz run, e.g. `GPUDB_FUZZ_N=5000 .venv/bin/python tests/run.py fuzz`.
The ORM tests need Node.js (they install their packages on first run).

## Where things are

See the "Layout" section of the README. Most query work happens in `src/engine.cu` (binding,
planning, GPU execution), storage in `src/storage.*`, SQL parsing in `src/sql.cpp`, the MySQL
protocol in `src/server.cpp`.

## Style

Match the surrounding code: C++20, two-space indent, comments that say why rather than what. Keep
changes focused; include a test (a check in an existing test file is fine) for fixes and features.
