#!/bin/sh
# prints: query# first best_of_3_after_first (seconds)
n=0
while IFS= read -r q; do
  n=$((n+1)); t=""
  for i in 1 2 3 4; do
    s=$(clickhouse-client --time --format=Null --max_threads=12 -q "$q" 2>&1 >/dev/null | tail -1)
    t="$t $s"
  done
  echo "$n$t"
done < /ch/queries.sql
