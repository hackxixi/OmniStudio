#!/usr/bin/env bash
# E3（本机 16GB Apple Silicon）：每个 MLX 量化版本起一个 mlx_jev_server（一份权重，同时当 JEV 与解码器），
# 依次跑 E2 的 select + verify（VERIFY_N 条抽样）与 E1 第二轮的 A 组（原生工具调用，HINTS=1），
# 最后把 /stats（权重 / 峰值内存、预填充 / 生成速度、JEV 延迟）写进结果。
# 用法：MODELS_DIR=... OPENJEV_SRC=... PY=... ./run-mac.sh <模型目录名>...
set -u
cd "$(dirname "$0")"
HERE=$PWD
OUT=${OUT:-$HERE/results-mac.jsonl}
PORT=18130
export JEV_KEY="" OUT

for name in "$@"; do
  dir=$MODELS_DIR/$name
  [ -d "$dir" ] || { echo "!! $name missing"; continue; }
  PYTHONPATH=$OPENJEV_SRC "$PY" mlx_jev_server.py --model "$dir" --port $PORT > "$HERE/server-$name.log" 2>&1 &
  pid=$!
  for _ in $(seq 1 120); do curl -s -m 2 localhost:$PORT/health/live >/dev/null && break; kill -0 $pid 2>/dev/null || break; sleep 2; done
  if ! curl -s -m 2 localhost:$PORT/health/live >/dev/null; then echo "!! $name failed to start"; tail -5 "$HERE/server-$name.log"; kill $pid 2>/dev/null; continue; fi
  echo "[$name] up: $(curl -s localhost:$PORT/stats)"
  (cd ../e2-jev-size && JEV_URL=http://127.0.0.1:$PORT TAG=$name bun sweep.ts select,verify)
  (cd ../e1-jev-orchestration && SET=r2 HINTS=1 TAG=$name SMALL_URL=http://127.0.0.1:$PORT SMALL_MODEL=$name bun bench.ts A)
  s=$(curl -s localhost:$PORT/stats)
  echo "[$name] stats: $s"
  echo "{\"tag\":\"$name\",\"kind\":\"stats\",\"stats\":$s}" >> "$OUT"
  kill $pid; wait $pid 2>/dev/null
done
echo "== all done"
