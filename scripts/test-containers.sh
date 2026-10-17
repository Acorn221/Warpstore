#!/bin/sh
# Creates the local containers the ingest / CDC tests use (tests/ingest_s3.py, ingest_kafka.py,
# binlog.py); the test runner skips those tests when the containers don't exist. The credentials are
# local throwaway ones, bound to 127.0.0.1 only. Remove with: docker rm -f gpudb-s3 gpudb-redpanda gpudb-mysql
set -e
CONF=$(mktemp -d)
cat > "$CONF/s3.json" <<'JSON'
{"identities":[{"name":"admin","credentials":[{"accessKey":"gpudbkey","secretKey":"gpudbsecret"}],"actions":["Admin","Read","Write","List","Tagging"]}]}
JSON
docker run -d --name gpudb-s3 -p 127.0.0.1:18333:8333 -v "$CONF/s3.json:/etc/s3.json:ro" \
  chrislusf/seaweedfs:3.80 server -s3 -s3.config=/etc/s3.json -dir=/data -volume.max=5
docker run -d --name gpudb-redpanda -p 127.0.0.1:19092:19092 redpandadata/redpanda:v24.2.7 \
  redpanda start --smp 1 --memory 700M --reserve-memory 0M --overprovisioned --node-id 0 --check=false \
  --kafka-addr PLAINTEXT://0.0.0.0:19092 --advertise-kafka-addr PLAINTEXT://127.0.0.1:19092
docker run -d --name gpudb-mysql -p 127.0.0.1:13306:3306 --memory 1g -e MYSQL_ROOT_PASSWORD=gpudbroot \
  mysql:8.4 --server-id=1 --innodb-buffer-pool-size=64M --performance-schema=OFF
echo "waiting for the containers ..."
for i in $(seq 90); do
  docker exec gpudb-mysql mysqladmin -uroot -pgpudbroot ping >/dev/null 2>&1 && break
  sleep 2
done
docker exec gpudb-mysql mysql -uroot -pgpudbroot -e \
  "CREATE USER IF NOT EXISTS 'cdc'@'%' IDENTIFIED BY 'cdcpw'; GRANT SELECT, RELOAD, REPLICATION SLAVE, REPLICATION CLIENT ON *.* TO 'cdc'@'%';"
sleep 3
curl -sf --aws-sigv4 aws:amz:us-east-1:s3 --user gpudbkey:gpudbsecret -X PUT http://127.0.0.1:18333/events >/dev/null || true
echo "ready: run .venv/bin/python tests/run.py"
