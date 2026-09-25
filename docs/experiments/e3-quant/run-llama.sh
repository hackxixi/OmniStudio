#!/usr/bin/env bash
# E3 的 llama.cpp 版本：每个 GGUF 起 llama-server（:18131，8K，单槽，全部层上 GPU）+ llama_jev_server（:18130），
# 跑与 run-mac.sh 相同的 select + verify + A 组，内存 = 权重文件（mmap）+ 进程 footprint（运行时 buffer）。
# 用法：LLAMA_SERVER=… TOKENIZER=… OPENJEV_SRC=… PY=… VERIFY_N=100 ./run-llama.sh 标签=路径.gguf ...
set -u
cd "$(dirname "$0")"
HERE=$PWD
OUT=${OUT:-$HERE/results-mac.jsonl}
export JEV_KEY="" OUT

for spec in "$@"; do
  name=${spec%%=*}; gguf=${spec#*=}
  "$LLAMA_SERVER" -m "$gguf" --host 127.0.0.1 --port 18131 -c 8192 -np 1 -ngl 99 > "$HERE/llama-$name.log" 2>&1 &
  lpid=$!
  for _ in $(seq 1 90); do curl -s -m 2 localhost:18131/health | grep -q ok && break; sleep 2; done
  PYTHONPATH=$OPENJEV_SRC "$PY" llama_jev_server.py --tokenizer "$TOKENIZER" --llama-url http://127.0.0.1:18131 --port 18130 --name "$name" > "$HERE/server-$name.log" 2>&1 &
  spid=$!
  for _ in $(seq 1 60); do curl -s -m 2 localhost:18130/health/live >/dev/null && break; sleep 2; done
  (cd ../e2-jev-size && JEV_URL=http://127.0.0.1:18130 TAG=$name bun sweep.ts select,verify)
  (cd ../e1-jev-orchestration && SET=r2 HINTS=1 TAG=$name SMALL_URL=http://127.0.0.1:18130 SMALL_MODEL=$name bun bench.ts A)
  # 这版 llama.cpp 默认不打印 buffer 大小；footprint 不含 mmap 的权重文件，所以内存 = 权重文件 + 进程 footprint
  mem="weights $(du -h "$gguf" | cut -f1) + footprint $(footprint -p $lpid 2>/dev/null | grep -oE 'Footprint: [0-9.]+ [KMG]B' | head -1)"
  s=$(curl -s localhost:18130/stats)
  echo "[$name] stats: $s | $mem"
  echo "{\"tag\":\"$name\",\"kind\":\"stats\",\"stats\":$s,\"llama_buffers\":\"$mem\"}" >> "$OUT"
  kill $spid $lpid; wait $spid $lpid 2>/dev/null
  rm -f "$HERE/llama-$name.log" "$HERE/server-$name.log"
done
echo "== all done"
