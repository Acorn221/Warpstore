"""TPC-H schema, data export and queries shared by the loader and the benchmark.

Comment columns and a few other high-cardinality free-text columns are omitted: they are
irrelevant to the queries below and would only bloat the string dictionaries.
"""

TABLES = {
    "region": ("r_regionkey INT, r_name VARCHAR(25), PRIMARY KEY (r_regionkey)",
               "r_regionkey, r_name"),
    "nation": ("n_nationkey INT, n_name VARCHAR(25), n_regionkey INT, PRIMARY KEY (n_nationkey)",
               "n_nationkey, n_name, n_regionkey"),
    "supplier": ("s_suppkey INT, s_name VARCHAR(25), s_nationkey INT, s_acctbal DOUBLE, PRIMARY KEY (s_suppkey)",
                 "s_suppkey, s_name, s_nationkey, s_acctbal"),
    "customer": ("c_custkey INT, c_nationkey INT, c_acctbal DOUBLE, c_mktsegment VARCHAR(10), PRIMARY KEY (c_custkey)",
                 "c_custkey, c_nationkey, c_acctbal, c_mktsegment"),
    "part": ("p_partkey INT, p_brand VARCHAR(10), p_type VARCHAR(25), p_size INT, p_container VARCHAR(10), "
             "p_retailprice DOUBLE, PRIMARY KEY (p_partkey)",
             "p_partkey, p_brand, p_type, p_size, p_container, p_retailprice"),
    "orders": ("o_orderkey INT, o_custkey INT, o_orderstatus VARCHAR(1), o_totalprice DOUBLE, o_orderdate DATE, "
               "o_orderpriority VARCHAR(15), o_shippriority INT, PRIMARY KEY (o_orderkey), SORT KEY (o_orderdate)",
               "o_orderkey, o_custkey, o_orderstatus, o_totalprice, o_orderdate, o_orderpriority, o_shippriority"),
    "lineitem": ("l_orderkey INT, l_partkey INT, l_suppkey INT, l_linenumber INT, l_quantity DOUBLE, "
                 "l_extendedprice DOUBLE, l_discount DOUBLE, l_tax DOUBLE, l_returnflag VARCHAR(1), "
                 "l_linestatus VARCHAR(1), l_shipdate DATE, l_commitdate DATE, l_receiptdate DATE, "
                 "l_shipinstruct VARCHAR(25), l_shipmode VARCHAR(10), SHARD KEY (l_orderkey), SORT KEY (l_shipdate)",
                 "l_orderkey, l_partkey, l_suppkey, l_linenumber, l_quantity, l_extendedprice, l_discount, l_tax, "
                 "l_returnflag, l_linestatus, l_shipdate, l_commitdate, l_receiptdate, l_shipinstruct, l_shipmode"),
}

QUERIES = {
    "Q1 pricing summary (scan+groupby)": """
select l_returnflag, l_linestatus, sum(l_quantity) as sum_qty, sum(l_extendedprice) as sum_base_price,
  sum(l_extendedprice * (1 - l_discount)) as sum_disc_price,
  sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) as sum_charge,
  avg(l_quantity) as avg_qty, avg(l_extendedprice) as avg_price, avg(l_discount) as avg_disc, count(*) as count_order
from lineitem
where l_shipdate <= date '1998-12-01' - interval '90' day
group by l_returnflag, l_linestatus
order by l_returnflag, l_linestatus""",
    "Q3 shipping priority (3-way join)": """
select l_orderkey, sum(l_extendedprice * (1 - l_discount)) as revenue, o_orderdate, o_shippriority
from customer, orders, lineitem
where c_mktsegment = 'BUILDING' and c_custkey = o_custkey and l_orderkey = o_orderkey
  and o_orderdate < date '1995-03-15' and l_shipdate > date '1995-03-15'
group by l_orderkey, o_orderdate, o_shippriority
order by revenue desc, o_orderdate
limit 10""",
    "Q5 local supplier volume (6-way join)": """
select n_name, sum(l_extendedprice * (1 - l_discount)) as revenue
from customer, orders, lineitem, supplier, nation, region
where c_custkey = o_custkey and l_orderkey = o_orderkey and l_suppkey = s_suppkey
  and c_nationkey = s_nationkey and s_nationkey = n_nationkey and n_regionkey = r_regionkey
  and r_name = 'ASIA' and o_orderdate >= date '1994-01-01' and o_orderdate < date '1994-01-01' + interval '1' year
group by n_name
order by revenue desc""",
    "Q6 forecast revenue (filter+sum)": """
select sum(l_extendedprice * l_discount) as revenue
from lineitem
where l_shipdate >= date '1994-01-01' and l_shipdate < date '1994-01-01' + interval '1' year
  and l_discount between 0.05 and 0.07 and l_quantity < 24""",
    "Q12 shipping modes (join+case)": """
select l_shipmode,
  sum(case when o_orderpriority = '1-URGENT' or o_orderpriority = '2-HIGH' then 1 else 0 end) as high_line_count,
  sum(case when o_orderpriority <> '1-URGENT' and o_orderpriority <> '2-HIGH' then 1 else 0 end) as low_line_count
from orders, lineitem
where o_orderkey = l_orderkey and l_shipmode in ('MAIL', 'SHIP') and l_commitdate < l_receiptdate
  and l_shipdate < l_commitdate and l_receiptdate >= date '1994-01-01'
  and l_receiptdate < date '1994-01-01' + interval '1' year
group by l_shipmode
order by l_shipmode""",
    "Q14 promotion effect (join+like)": """
select 100.00 * sum(case when p_type like 'PROMO%' then l_extendedprice * (1 - l_discount) else 0 end)
  / sum(l_extendedprice * (1 - l_discount)) as promo_revenue
from lineitem, part
where l_partkey = p_partkey and l_shipdate >= date '1995-09-01'
  and l_shipdate < date '1995-09-01' + interval '1' month""",
    "X1 high-cardinality group by (orders)": """
select o_custkey, count(*) as n, sum(o_totalprice) as total
from orders
group by o_custkey
order by total desc, o_custkey
limit 10""",
    "X2 filtered top-N projection": """
select l_orderkey, l_linenumber, l_extendedprice
from lineitem
where l_shipdate = date '1996-06-01' and l_shipmode = 'AIR'
order by l_extendedprice desc, l_orderkey
limit 5""",
}


def export(sf: float, out_dir: str) -> None:
    """Generate TPC-H with DuckDB and write pipe-delimited files for LOAD DATA."""
    import os
    import duckdb

    os.makedirs(out_dir, exist_ok=True)
    con = duckdb.connect()
    con.execute("INSTALL tpch; LOAD tpch")
    con.execute(f"CALL dbgen(sf={sf})")
    for t, (_, cols) in TABLES.items():
        path = os.path.join(out_dir, f"{t}.tbl")
        con.execute(f"COPY (SELECT {cols} FROM {t}) TO '{path}' (DELIMITER '|', HEADER false)")
        print("wrote", path)
