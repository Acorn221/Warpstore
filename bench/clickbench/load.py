import pymysql, time
c=pymysql.connect(host="127.0.0.1",port=3307,user="root",autocommit=True,read_timeout=3600).cursor()
ddl=[x for x in open("bench/clickbench/create.sql").read().split(";") if "CREATE TABLE" in x][0]
c.execute(ddl)
cols=open("bench/clickbench/cols.txt").read()
t=time.time(); n=c.execute(f"load data infile 'data/clickbench/hits.tsv' into table hits fields terminated by '\\t' ({cols})")
c.execute("optimize table hits"); c.fetchall()
print("loaded", n, f"{time.time()-t:.1f}s")
