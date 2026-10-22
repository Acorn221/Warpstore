#!/usr/bin/env python3
"""warpstore test suite.

  .venv/bin/python tests/run.py                 # everything that can run here
  .venv/bin/python tests/run.py smoke tpch     # by name (substring match)
  .venv/bin/python tests/run.py --list

Tests that need a shared server get WARPSTORE_PORT (one server, --no-wal, started here). Tests marked
"own" start their own servers through tests/lib.py (crash, backup, replication, ...). A test is
skipped when what it needs is missing (node, Docker containers, cuVS, the embedding model).
"""
import concurrent.futures, threading
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
    ("crashfuzz", "crashfuzz.py", None, True),
    ("dictgc", "dictgc.py", None, True),
    ("colupdate", "colupdate.py", None, True),
    ("diskfull", "diskfull.py", None, True),
    ("compact", "compact.py", None, True),
    ("tier", "tier.py", None, True),
    ("backup", "backup.py", None, True),
    ("replication", "replication.py", None, True),
    ("ingest-s3", "ingest_s3.py", lambda: docker_up("warpstore-s3"), True),
    ("ingest-kafka", "ingest_kafka.py", lambda: docker_up("warpstore-redpanda"), True),
    ("binlog", "binlog.py", lambda: docker_up("warpstore-mysql"), True),
]


def main():
    """run.py [names..] [-j N] [--quick] [--list]: the test groups (all by default; a name that matches a
    group exactly selects only it, else every group containing it). Groups that start their own server
    run N at a time (default 4), next to the groups that share one server; each group's output is
    printed when it finishes. --quick: smoke, sql and a short fuzz run (for iterating)."""
    argv = sys.argv[1:]
    jobs = 4
    if "-j" in argv:
        jobs = max(1, int(argv[argv.index("-j") + 1]))
        del argv[argv.index("-j"):argv.index("-j") + 2]
    args = [a for a in argv if not a.startswith("-")]
    if "--list" in argv:
        for t in TESTS:
            print(t[0])
        return
    if "--quick" in argv:
        args = ["smoke", "sql", "fuzz"]
        os.environ.setdefault("WARPSTORE_FUZZ_N", "150")
    names = {t[0] for t in TESTS}
    sel = [t for t in TESTS if not args or any(a == t[0] if a in names else a in t[0] for a in args)]
    sel = [t for t in sel if isinstance(t[1], str) is False or os.path.exists(os.path.join(HERE, t[1]))]
    results = {}
    lock = threading.Lock()
    shared_box = {"srv": None}

    def report(name, status, secs, out):
        with lock:
            results[name] = (status, secs)
            if out:
                sys.stdout.write(out if out.endswith("\n") else out + "\n")
            print(f"=== {name}: {status} ({secs:.1f}s)", flush=True)

    def run_one(name, how, own):
        t0 = time.time()
        env = dict(os.environ, WARPSTORE_TEST_TMP=lib.TMP)
        if callable(how):
            try:
                rc = how(shared_box["srv"].port)
            except Exception as e:
                print(f"{name}: {e}")
                rc = 1
            report(name, "PASS" if rc == 0 else "FAIL", time.time() - t0, "")
            return
        if not own:
            env["WARPSTORE_PORT"] = str(shared_box["srv"].port)
        p = subprocess.run([PY, os.path.join(HERE, how)] + ([str(shared_box["srv"].port)] if not own else []), env=env,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        rc = p.returncode
        if not own and shared_box["srv"] and shared_box["srv"].proc.poll() is not None:
            p.stdout += f"shared server died; log {shared_box['srv'].log}\n"
            shared_box["srv"] = None
            rc = rc or 1
        report(name, "PASS" if rc == 0 else "FAIL", time.time() - t0, f"=== {name}\n" + p.stdout)

    runnable = []
    for name, how, req, own in sel:
        if req and not req():
            report(name, "SKIP", 0, "")
            continue
        runnable.append((name, how, own))
    shared_tests = [t for t in runnable if not t[2]]
    own_tests = [t for t in runnable if t[2]]

    def shared_lane():
        for name, how, own in shared_tests:
            if shared_box["srv"] is None:
                shared_box["srv"] = lib.Server(data_dir=None, http=False, name="shared").start()
            run_one(name, how, own)

    try:
        lane = threading.Thread(target=shared_lane)
        lane.start()
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, jobs - (1 if shared_tests else 0))) as ex:
            list(ex.map(lambda t: run_one(*t), own_tests))
        lane.join()
    finally:
        if shared_box["srv"]:
            shared_box["srv"].stop()
    print("\nsummary")
    order = [t[0] for t in sel]
    for n in order:
        if n in results:
            r, t = results[n]
            print(f"  {r:4s}  {n:16s} {t:6.1f}s")
    failed = [n for n, (r, _) in results.items() if r == "FAIL"]
    print(f"{len(results) - len(failed)}/{len(results)} passed or skipped; data and logs in {lib.TMP}")
    if not failed:
        shutil.rmtree(lib.TMP, ignore_errors=True)
    lib._failed = bool(failed)  # (lib's exit cleanup keeps the data of a failed run)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
