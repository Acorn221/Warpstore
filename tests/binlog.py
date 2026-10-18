# MySQL CDC against the local MySQL 8.4 container (warpstore-mysql, 127.0.0.1:13306, user cdc / cdcpw):
# snapshot + binlog following of many column types, inserts / updates (also of the primary key) /
# deletes / multi-table and rolled back transactions, DDL, a table without a primary key, a crash of
# warpstore while the source keeps changing, and a one-off LOAD DATA MYSQL copy.
import datetime, decimal, json, os, random, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pymysql
from lib import Checker, Server

SRC = dict(host="127.0.0.1", port=13306, user="root", password="warpstoreroot", autocommit=True, ssl={"check_hostname": False})
DB = f"shop{os.getpid()}"
URL = f"mysql://cdc:cdcpw@127.0.0.1:13306/{DB}"
c = Checker()
my = pymysql.connect(**SRC).cursor()
my.execute(f"DROP DATABASE IF EXISTS {DB}")
my.execute(f"CREATE DATABASE {DB}")
my.execute(f"USE {DB}")
my.execute("""CREATE TABLE orders (id INT AUTO_INCREMENT PRIMARY KEY, customer VARCHAR(50), amount DECIMAL(10,2), qty SMALLINT UNSIGNED,
  status ENUM('new','paid','shipped'), tags SET('a','b','c'), created DATETIME(3), d DATE, t TIME, flag TINYINT(1), big BIGINT,
  f FLOAT, dbl DOUBLE, meta JSON, note TEXT, bits BIT(8), y YEAR, neg MEDIUMINT, u INT UNSIGNED, ts TIMESTAMP(6) NULL)""")
my.execute("CREATE TABLE items (order_id INT, sku VARCHAR(20), price DECIMAL(12,4), PRIMARY KEY (order_id, sku))")
my.execute("CREATE TABLE nopk (a INT, b VARCHAR(10))")
R = random.Random(7)


def order_row(i):
    return (f"cust'{i % 50}\\x", f"{R.randint(1, 99999) / 100:.2f}", R.randint(0, 65535), R.choice(["new", "paid", "shipped"]),
            R.choice(["", "a", "a,c", "b,c"]), f"2024-0{1 + i % 9}-1{i % 10} 12:34:56.{i % 1000:03d}", f"2024-02-{1 + i % 28:02d}",
            f"{i % 24:02d}:{i % 60:02d}:07", i % 2, R.randint(-2**62, 2**62), R.random() * 100, R.random() * 1e6 - 5e5,
            json.dumps({"k": i, "arr": [1, "two", None, True], "nested": {"x": i * 0.5}}), f"note {i}\twith tab\nand newline" if i % 7 == 0 else None,
            i % 256, 2000 + i % 50, -i * 3, 4000000000 - i, f"2024-03-0{1 + i % 9} 01:02:03.{i:06d}" if i % 3 else None)


def insert_orders(n):
    my.executemany("INSERT INTO orders (customer, amount, qty, status, tags, created, d, t, flag, big, f, dbl, meta, note, bits, y, neg, u, ts) "
                   "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,b'00000000' | %s,%s,%s,%s,%s)", [order_row(R.randint(0, 10**6)) for _ in range(n)])


insert_orders(1000)
my.executemany("INSERT INTO items VALUES (%s,%s,%s)", [(R.randint(1, 1000), f"sku{k}", R.randint(1, 10**7) / 10**4) for k in range(500)])
my.executemany("INSERT INTO nopk VALUES (%s,%s)", [(k % 10, f"v{k % 3}") for k in range(100)])

srv = Server(name="binlog", http=False).start()
g = srv.cursor()


def norm(v):
    if isinstance(v, decimal.Decimal):
        return round(float(v), 4)
    if isinstance(v, float):
        return float(f"{v:.6g}")
    if isinstance(v, datetime.timedelta):
        s = int(v.total_seconds())
        return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"
    if isinstance(v, (datetime.datetime, datetime.date)):
        return str(v)
    if isinstance(v, bytes):
        return int.from_bytes(v, "big")
    if isinstance(v, str) and v[:1] in "{[":
        try:
            return json.dumps(json.loads(v), sort_keys=True)
        except ValueError:
            return v
    return v


def table_rows(cur, db, t, order):
    cur.execute(f"SELECT * FROM {db}.{t} ORDER BY {order}")
    return [tuple(norm(x) for x in r) for r in cur.fetchall()]


def same(t, order, target, timeout=30):
    end = time.time() + timeout
    while True:
        a = table_rows(my, DB, t, order)
        try:
            b = table_rows(g, target, t, order)
        except pymysql.MySQLError as e:
            b = str(e)
        if a == b or time.time() > end:
            return a, b
        time.sleep(0.3)


def check_all(label, target="shop_copy"):
    for t, order in (("orders", "id"), ("items", "order_id, sku"), ("nopk", "a, b")):
        a, b = same(t, order, target)
        c.eq(f"{label}: {t}", b, a)


g.execute(f"CREATE PIPELINE p AS LOAD DATA MYSQL '{URL}' INTO DATABASE shop_copy")
g.execute("START PIPELINE p")
check_all("snapshot")

# live changes
insert_orders(200)
my.execute("UPDATE orders SET amount = amount + 1, status = 'paid', meta = JSON_SET(meta, '$.k', 'changed') WHERE id % 5 = 0")
my.execute("UPDATE orders SET id = id + 100000 WHERE id % 50 = 1")  # primary key change
my.execute("DELETE FROM orders WHERE id % 7 = 3")
my.execute("BEGIN")
my.execute("INSERT INTO items VALUES (1, 'txn-sku', 1.5)")
my.execute("UPDATE orders SET qty = 1 WHERE id < 20")
my.execute("COMMIT")
my.execute("BEGIN")
my.execute("INSERT INTO items VALUES (2, 'rolled-back', 9)")
my.execute("ROLLBACK")
my.execute("DELETE FROM nopk WHERE a = 3 LIMIT 2")
my.execute("UPDATE nopk SET b = 'zz' WHERE a = 4")
check_all("binlog changes")

# DDL
my.execute("ALTER TABLE orders ADD COLUMN extra INT DEFAULT 7")
my.execute("INSERT INTO orders (customer, extra) VALUES ('after-ddl', 42)")
my.execute("CREATE TABLE newt (k INT PRIMARY KEY, v VARCHAR(10))")
my.execute("INSERT INTO newt VALUES (1, 'x'), (2, 'y')")
a, b = same("newt", "k", "shop_copy")
c.eq("CREATE TABLE on the source", b, a)
check_all("after ALTER TABLE")

# warpstore crashes while the source keeps changing; it resumes from its saved position exactly once
srv.kill()
insert_orders(100)
my.executemany("INSERT INTO nopk VALUES (%s,%s)", [(k, "aftercrash") for k in range(50)])
my.execute("DELETE FROM orders WHERE id % 11 = 0")
srv.start()
g = srv.cursor()
check_all("after a crash")

# one-off copy
g.execute(f"LOAD DATA MYSQL '{URL}' INTO DATABASE shop_once")
check_all("LOAD DATA MYSQL", "shop_once")
g.execute("SHOW PIPELINES")
print(g.fetchall())
g.execute("STOP PIPELINE p")
srv.stop()
my.execute(f"DROP DATABASE {DB}")
c.done()
