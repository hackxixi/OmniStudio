#!/usr/bin/env bash
# E2：Qwen3.5-0.8B / 2B / 4B / 9B(fp8) 依次起在 GPU4 上（SGLang :30001 + OpenJev :18120，一次一个，跑完即停），
# 最后对现役 35B OpenJev（:18110）跑同一套考试。
# 在 ycs2 上：cd /tmp/exp/e2-jev-size && nohup ./run-e2.sh > run-e2.log 2>&1 &
set -u
cd "$(dirname "$0")"
BUN=~/.bun/bin/bun
KEY=$(cat ~/jev/openjev.key)
export OUT=${OUT:-$PWD/results-e2.jsonl}

wait_url() { # url, tries
  for _ in $(seq 1 "$2"); do
    curl -s -m 3 -H "authorization: Bearer $KEY" "$1" | grep -q '"object"' && return 0
    sleep 10
  done
  return 1
}

for M in 0.8B 2B 4B 9B; do
  Q=""; [ "$M" = 9B ] && Q=fp8
  MODEL=Qwen3.5-$M QUANT=$Q ~/jev/launch-e1-small.sh >/dev/null
  if ! wait_url http://127.0.0.1:30001/v1/models 60; then echo "!! $M sglang failed"; docker logs e1-small 2>&1 | tail -5; docker stop e1-small >/dev/null; continue; fi
  MODEL=Qwen3.5-$M ~/jev/launch-e2-openjev.sh >/dev/null
  if ! wait_url http://127.0.0.1:18120/v1/models 60; then echo "!! $M openjev failed"; docker logs e2-openjev 2>&1 | tail -8; docker stop e2-openjev e1-small >/dev/null; continue; fi
  JEV_URL=http://127.0.0.1:18120 TAG=$M $BUN sweep.ts select,route,verify
  docker stop e2-openjev e1-small >/dev/null
done

JEV_URL=http://127.0.0.1:18110 TAG=35B $BUN sweep.ts select,route,verify
echo "== all done"
