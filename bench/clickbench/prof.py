import pymysql, sys, time
S=sys.argv[1]
gq=[q.strip() for q in open(f"{S}/queries.sql") if q.strip()]
c=pymysql.connect(host="127.0.0.1",port=3307,user="root",autocommit=True,read_timeout=300).cursor()
c.execute("SET collation_connection = utf8mb4_bin"); c.execute("set gpudb_codegen = sync")
for i in map(int, sys.argv[2:]):
    q=gq[i-1]
    c.execute(q); c.fetchall()
    t=time.time(); c.execute("profile "+q); c.fetchall(); el=time.time()-t
    c.execute("show profile"); print(f"Q{i} {el*1000:.1f} ms")
    for r in c.fetchall(): print(f"    {r[0][:40]:40} {r[1]:>9} {str(r[2] or ''):>9} {(r[3] or '')[:100]}")
