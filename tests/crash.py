# Crash safety: concurrent transfers (sum invariant) and acknowledged inserts while SNAPSHOTs run,
# kill -9 at random moments (sometimes with garbage appended to the WAL), restart, verify.
# GPUDB_CRASH_CYCLES (default 5). A process kill keeps the OS page cache, so every acknowledged
# write must survive even with the default async durability.
import multiprocessing as mp, os, random, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pymysql
from lib import TMP, Checker, Server

CYCLES = int(os.environ.get("GPUDB_CRASH_CYCLES", "5"))
ACK = os.path.join(TMP, f"crash-acks-{os.getpid()}")


def worker(port, wid, stop):
    random.seed(wid * 7919 + int(time.time()))
    acks = open(f"{ACK}.{wid}", "a")
    try:
        c = pymysql.connect(host="127.0.0.1", port=port, user="root", database="bank", autocommit=False, read_timeout=30)
        cur = c.cursor()
        n = 0
        while not stop.is_set():
            n += 1
            try:
                if n % 3 == 0:
                    eid = wid * 10**12 + int(time.time() * 1e6) % 10**12
                    cur.execute(f"INSERT INTO ev VALUES ({eid}, {wid})")
                    c.commit()
                    acks.write(f"{eid}\n")
                    acks.flush()
                else:
                    a, b = random.sample(range(100), 2)
                    x = random.randint(1, 50)
                    cur.execute("BEGIN")
                    cur.execute(f"UPDATE acct SET v = v - {x} WHERE id = {a}")
                    cur.execute(f"UPDATE acct SET v = v + {x} WHERE id = {b}")
                    c.commit()
            except pymysql.err.OperationalError:
                return
            except pymysql.MySQLError:
                try:
                    c.rollback()
                except Exception:
                    return
    except Exception:
        return


def snapper(port, stop):
    try:
        c = pymysql.connect(host="127.0.0.1", port=port, user="root", autocommit=True)
        while not stop.is_set():
            time.sleep(random.uniform(0.3, 1.5))
            c.cursor().execute("SNAPSHOT")
    except Exception:
        return


def main():
    srv = Server(name="crash", http=False).start()
    cur = srv.cursor()
    cur.execute("CREATE DATABASE bank")
    cur.execute("USE bank")
    cur.execute("CREATE TABLE acct (id BIGINT PRIMARY KEY, v BIGINT)")
    cur.execute("INSERT INTO acct VALUES " + ",".join(f"({i},1000)" for i in range(100)))
    cur.execute("CREATE TABLE ev (id BIGINT PRIMARY KEY, w INT)")
    c = Checker()
    for cycle in range(1, CYCLES + 1):
        stop = mp.Event()
        ps = [mp.Process(target=worker, args=(srv.port, w, stop)) for w in range(4)] + [mp.Process(target=snapper, args=(srv.port, stop))]
        for p in ps:
            p.start()
        time.sleep(random.uniform(1.0, 3.0))
        srv.kill()
        stop.set()
        for p in ps:
            p.join(10)
        if cycle % 3 == 0:  # torn / garbage WAL tail
            with open(os.path.join(srv.data_dir, "wal.log"), "ab") as f:
                f.write(os.urandom(random.randint(1, 40)))
        srv.start()
        cur = srv.cursor("bank")
        cur.execute("SELECT COUNT(*), SUM(v) FROM acct")
        n, total = cur.fetchone()
        cur.execute("SELECT id FROM ev")
        have = {r[0] for r in cur.fetchall()}
        acked = set()
        for w in range(4):
            if os.path.exists(f"{ACK}.{w}"):
                acked |= {int(x) for x in open(f"{ACK}.{w}") if x.strip()}
        missing = acked - have
        print(f"cycle {cycle}: sum {total}, acked {len(acked)}, present {len(have)}, missing {len(missing)}")
        c.eq(f"cycle {cycle} accounts", n, 100)
        c.eq(f"cycle {cycle} transfer sum", int(total), 100000)
        c.eq(f"cycle {cycle} acknowledged inserts present", len(missing), 0)
    srv.stop()
    c.done()


if __name__ == "__main__":
    mp.set_start_method("fork")
    main()
