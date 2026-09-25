#!/usr/bin/env bash
# E1 第二轮：0.8B / 2B / 4B 依次起在 GPU4 上（一次只起一个，跑完即停），每个模型跑
# A / B / B0（不带修正）+ A / B（带 HINTS），最后 35B 原生跑 C（带 / 不带 HINTS）。
# 在 ycs2 上：cd /tmp/e1 && nohup ./run-r2.sh > run-r2.log 2>&1 &
set -u
cd "$(dirname "$0")"
BUN=~/.bun/bin/bun
OUT=${OUT:-/tmp/e1/results-r2.jsonl}
export SET=r2 OUT

wait_ready() {
  for _ in $(seq 1 60); do
    curl -s -m 3 localhost:30001/v1/models | grep -q '"object"' && return 0
    [ "$(docker inspect -f '{{.State.Status}}' e1-small 2>/dev/null)" = "exited" ] && return 1
    sleep 8
  done
  return 1
}

for M in 0.8B 2B 4B; do
  MODEL=Qwen3.5-$M ~/jev/launch-e1-small.sh >/dev/null
  if ! wait_ready; then echo "!! $M failed to start"; docker logs e1-small 2>&1 | tail -5; continue; fi
  export SMALL_MODEL=Qwen/Qwen3.5-$M TAG=$M
  HINTS=0 $BUN bench.ts A,B,B0
  HINTS=1 $BUN bench.ts A,B
  docker stop e1-small >/dev/null
done

export TAG=35B
HINTS=0 $BUN bench.ts C
HINTS=1 $BUN bench.ts C
echo "== all done"
