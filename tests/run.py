#!/usr/bin/env python3
"""warpstore test suite.

  .venv/bin/python tests/run.py                 # everything that can run here
  .venv/bin/python tests/run.py smoke tpch     # by name (substring match)
  .venv/bin/python tests/run.py --list

Tests that need a shared server get WARPSTORE_PORT (one server, --no-wal, started here). Tests marked
"own" start their own servers through tests/lib.py (crash, backup, replication, ...). A test is
skipped when what it needs is missing (node, Docker containers, cuVS, the embedding model).
"""
import os, shutil, subprocess, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def docker_up(name):
    if not shutil.which("docker"):
        return False
    r = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}", name], capture_output=True, text=True)
    if r.returncode != 0:
        return False
    if r.stdout.strip() != "true":
        subprocess.run(["docker", "start", name], capture_output=True)
        time.sleep(3)
    return True


def npm_ready(d):
    if not shutil.which("npx"):
        return False
    if not os.path.isdir(os.path.join(d, "node_modules")):
        if subprocess.run(["npm", "ci", "--silent"], cwd=d).returncode != 0:
            return False
    return True


def drizzle(port):
    d = os.path.join(HERE, "orm", "drizzle")
    c = lib.Server(data_dir=None, http=False)
    c.port = port
    c.cursor().execute("CREATE DATABASE IF NOT EXISTS app")
    env = dict(os.environ, WARPSTORE_PORT=str(port))
    r = subprocess.run(["npx", "drizzle-kit", "push", "--force"], cwd=d, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-2000:])
        return 1
    return subprocess.run(["npx", "tsx", "test.ts"], cwd=d, env=env).returncode


def prisma(port):
    d = os.path.join(HERE, "orm", "prisma")
    env = dict(os.environ, DATABASE_URL=f"mysql://root@127.0.0.1:{port}/papp")
    for cmd in (["npx", "prisma", "db", "push", "--skip-generate", "--accept-data-loss"], ["npx", "prisma", "generate"]):
        r = subprocess.run(cmd, cwd=d, env=env, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-2000:], r.stderr[-2000:])
            return 1
    return subprocess.run(["node", "test.js"], cwd=d, env=env).returncode


def sqlalchemy(port):
    rc = 0
    for drv in ("pymysql", "mysqlconnector"):
        try:
            __import__({"pymysql": "pymysql", "mysqlconnector": "mysql.connector"}[drv])
        except ImportError:
            continue
        print(f"-- mysql+{drv}")
        rc |= subprocess.run([PY, os.path.join(HERE, "orm", "sqlalchemy_test.py"), f"mysql+{drv}://root@127.0.0.1:{port}/mysql"]).returncode
    return rc


def has_module(m):
    try:
        __import__(m)
        return True
    except ImportError:
        return False


# name, how, requirement (None = always). how: a script name (shared server unless own=True) or a function(port).
TESTS = [
    ("smoke", "smoke.py", None, False),
    ("sql", "sql.py", None, False),
    ("postagg", "postagg.py", None, False),
    ("sharding", "sharding.py", None, False),
    ("tpch", "tpch.py", None, False),
    ("fuzz", "fuzz.py", None, False),
    ("orm-sqlalchemy", sqlalchemy, lambda: has_module("sqlalchemy"), False),
    ("orm-drizzle", drizzle, lambda: npm_ready(os.path.join(HERE, "orm", "drizzle")), False),
    ("orm-prisma", prisma, lambda: npm_ready(os.path.join(HERE, "orm", "prisma")), False),
    ("vector", "vector.py", None, False),
    ("mvcc", "mvcc.py", None, True),
    ("observability", "observability.py", None, True),
    ("crash", "crash.py", None, True),
    ("load", "load.py", None, True),
    ("autosnap", "autosnap.py", None, True),
    ("tier", "tier.py", None, True),
    ("backup", "backup.py", None, True),
    ("replication", "replication.py", None, True),
    ("ingest-s3", "ingest_s3.py", lambda: docker_up("warpstore-s3"), True),
    ("ingest-kafka", "ingest_kafka.py", lambda: docker_up("warpstore-redpanda"), True),
    ("binlog", "binlog.py", lambda: docker_up("warpstore-mysql"), True),
]


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if "--list" in sys.argv:
        for t in TESTS:
            print(t[0])
        return
    sel = [t for t in TESTS if not args or any(a in t[0] for a in args)]
    sel = [t for t in sel if isinstance(t[1], str) is False or os.path.exists(os.path.join(HERE, t[1]))]
    shared = None
    results = []
    try:
        for name, how, req, own in sel:
            if req and not req():
                results.append((name, "SKIP", 0))
                print(f"=== {name}: SKIP (requirement missing)")
                continue
            if not own and shared is None:
                shared = lib.Server(data_dir=None, http=False, name="shared").start()
            print(f"=== {name}", flush=True)
            t0 = time.time()
            env = dict(os.environ, WARPSTORE_TEST_TMP=lib.TMP)
            if not own:
                env["WARPSTORE_PORT"] = str(shared.port)
            if callable(how):
                rc = how(shared.port)
            else:
                rc = subprocess.run([PY, os.path.join(HERE, how)] + ([str(shared.port)] if not own else []), env=env).returncode
            if shared and shared.proc.poll() is not None:
                print(f"shared server died; log {shared.log}")
                shared = None
                rc = rc or 1
            results.append((name, "PASS" if rc == 0 else "FAIL", time.time() - t0))
            print(f"=== {name}: {results[-1][1]} ({results[-1][2]:.1f}s)", flush=True)
    finally:
        if shared:
            shared.stop()
    print("\nsummary")
    for n, r, t in results:
        print(f"  {r:4s}  {n:16s} {t:6.1f}s")
    failed = [n for n, r, _ in results if r == "FAIL"]
    print(f"{len(results) - len(failed)}/{len(results)} passed or skipped; data and logs in {lib.TMP}")
    if not failed:
        shutil.rmtree(lib.TMP, ignore_errors=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
