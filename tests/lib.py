# Shared helpers for the test suite: start/stop throwaway servers, assertions, DuckDB comparison.
import datetime, decimal, os, shutil, signal, socket, subprocess, sys, tempfile, time

import pymysql

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.environ.get("GPUDB_BIN", os.path.join(ROOT, "build", "gpudb"))
TMP = os.environ.get("GPUDB_TEST_TMP") or tempfile.mkdtemp(prefix="gpudb-test-")
os.makedirs(TMP, exist_ok=True)


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class Server:
    """A gpu-db process on free ports with its own data dir (None: --no-wal)."""

    def __init__(self, *args, data_dir="auto", mem=os.environ.get("GPUDB_TEST_MEM", "6G"), name="srv", http=True):
        self.args = list(args)
        self.name = name
        self.data_dir = os.path.join(TMP, name + "-" + str(os.getpid()) + "-" + str(free_port())) if data_dir == "auto" else data_dir
        self.mem = mem
        self.port = free_port()
        self.http_port = free_port() if http else 0
        self.proc = None
        self.starts = 0

    def start(self, timeout=120):
        self.starts += 1
        self.log = os.path.join(TMP, f"{self.name}-{self.port}-{self.starts}.log")
        cmd = [BIN, "--port", str(self.port)]
        cmd += ["--data-dir", self.data_dir] if self.data_dir else ["--no-wal"]
        if self.http_port:
            cmd += ["--http-port", str(self.http_port), "--data-port", str(free_port())]
        cmd += self.args
        if self.mem and shutil.which("systemd-run"):
            cmd = ["systemd-run", "--user", "--scope", "--quiet", "-p", f"MemoryMax={self.mem}"] + cmd
        self.proc = subprocess.Popen(cmd, stdout=open(self.log, "w"), stderr=subprocess.STDOUT)
        t = time.time()
        while time.time() - t < timeout:
            if self.proc.poll() is not None:
                raise RuntimeError(f"server exited ({self.proc.returncode}); log {self.log}:\n" + open(self.log).read()[-2000:])
            try:
                self.connect().close()
                return self
            except Exception:
                time.sleep(0.1)
        raise RuntimeError("server did not start; log " + self.log)

    def connect(self, db=None, **kw):
        kw.setdefault("autocommit", True)
        kw.setdefault("read_timeout", 600)
        return pymysql.connect(host="127.0.0.1", port=self.port, user=kw.pop("user", "root"), password=kw.pop("password", ""), database=db, **kw)

    def cursor(self, db=None, **kw):
        return self.connect(db, **kw).cursor()

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()

    def kill(self):
        if self.proc and self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait()

    def restart(self):
        self.stop()
        return self.start()

    def __enter__(self):
        return self.start()

    def __exit__(self, *a):
        self.stop()


def server_from_env():
    """The runner's shared server (GPUDB_PORT), or a fresh one when a test is run by hand."""
    if os.environ.get("GPUDB_PORT"):
        s = Server(data_dir=None, http=False)
        s.port = int(os.environ["GPUDB_PORT"])
        return s
    return Server().start()


def _diff(got, want, path=""):
    """First difference between nested sequences (big results stay readable)."""
    if isinstance(got, (list, tuple)) and isinstance(want, (list, tuple)):
        for i, (a, b) in enumerate(zip(got, want)):
            if a != b:
                return _diff(a, b, f"{path}[{i}]")
        if len(got) != len(want):
            return f"\n  {path}: length {len(got)} vs {len(want)}; extra: {(got[len(want):] or want[len(got):])[:3]!r}"
    return f"\n  {path}: got {got!r}\n  {' ' * len(path)}  want {want!r}"


class Checker:
    def __init__(self):
        self.fails = 0
        self.n = 0

    def check(self, name, cond, detail=""):
        self.n += 1
        if not cond:
            self.fails += 1
            print(f"FAIL {name} {detail}"[:2000])
        elif os.environ.get("GPUDB_TEST_VERBOSE"):
            print(f"ok   {name}")
        return cond

    def eq(self, name, got, want):
        detail = f"\n  got:  {got!r}\n  want: {want!r}"
        if got != want and isinstance(got, (list, tuple)) and isinstance(want, (list, tuple)):
            detail = _diff(got, want)
        return self.check(name, got == want, detail)

    def raises(self, name, fn, code=None):
        try:
            fn()
        except pymysql.MySQLError as e:
            return self.check(name, code is None or e.args[0] == code, f"error {e.args}")
        return self.check(name, False, "no error")

    def done(self):
        print(f"{self.n - self.fails}/{self.n} checks passed")
        sys.exit(1 if self.fails else 0)


def norm_value(v, places=6):
    if isinstance(v, decimal.Decimal):
        v = float(v)
    if isinstance(v, float):
        if v != v:
            return "NaN"
        return round(v, places) if abs(v) < 1e15 else float(f"{v:.12g}")
    if isinstance(v, datetime.datetime) and v.time() == datetime.time():
        return v.date()
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace")
    if isinstance(v, bool):
        return int(v)
    return v


def norm_rows(rows, ordered=True, places=6):
    out = [tuple(norm_value(v, places) for v in r) for r in rows]
    return out if ordered else sorted(out, key=lambda r: tuple((x is None, str(type(x)), x if x is not None else 0) for x in r))


def rows_close(a, b, rel=1e-6, abs_tol=0.0):
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if len(x) != len(y):
            return False
        for u, v in zip(x, y):
            if isinstance(u, (int, float)) and isinstance(v, (int, float)) and not isinstance(u, bool):
                if abs(u - v) > max(abs_tol, rel * max(1.0, abs(u), abs(v))):
                    return False
            elif u != v:
                return False
    return True
