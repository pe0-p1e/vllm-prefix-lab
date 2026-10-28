#!/usr/bin/env bash
# Must be invoked explicitly; no installation or unrelated process cleanup.
set -euo pipefail
ROOT="/state/partition/fyzhou"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${LAB_VENV:-$ROOT/venvs/vllm}"
MODEL="${LAB_MODEL:-/state/partition/model/Qwen3-0.6B}"
LENGTHS="${LAB_LENGTHS:-2048 4096 8192}"
RATIOS="${LAB_RATIOS:-0 0.25 0.5 0.75 1}"
REPEATS="${LAB_REPEATS:-3}"
MAXLEN="${LAB_MAX_MODEL_LEN:-16384}"
GPU_UTIL="${LAB_GPU_UTIL:-0.4}"
BLOCK="${LAB_BLOCK_SIZE:-16}"
PORT="${LAB_PORT:-18083}"
OUT="$ROOT/results/prefix-lab/$(date -u +%Y%m%dT%H%M%SZ)"
[[ -d "$MODEL" && -x "$VENV/bin/vllm" && -x "$VENV/bin/python" ]] || {
  echo "Missing local model or vLLM environment" >&2; exit 2;
}
# Don't take a busy GPU automatically or override someone else's service.
[[ "${CUDA_VISIBLE_DEVICES:-}" =~ ^[0-9]+$ ]] || {
  echo "Set CUDA_VISIBLE_DEVICES to a single physical GPU index" >&2; exit 2;
}
GPU="$CUDA_VISIBLE_DEVICES"
USED="$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
TOTAL="$(nvidia-smi -i "$GPU" --query-gpu=memory.total --format=csv,noheader,nounits | tr -d ' ')"
(( USED <= 512 && TOTAL > 0 )) || { echo "GPU not idle; aborting" >&2; exit 2; }
export HF_HOME="$ROOT/cache/huggingface" XDG_CACHE_HOME="$ROOT/cache/xdg"
export VLLM_CACHE_ROOT="$ROOT/cache/vllm" TRITON_CACHE_DIR="$ROOT/cache/triton"
export CUDA_CACHE_PATH="$ROOT/cache/cuda" TMPDIR="$ROOT/tmp"
mkdir -p "$OUT" "$HF_HOME" "$XDG_CACHE_HOME" "$VLLM_CACHE_ROOT" \
  "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$TMPDIR"
printf '%s\n' "$OUT" > "$ROOT/results/prefix-lab/LATEST"
"$VENV/bin/vllm" --version > "$OUT/version.txt"
nvidia-smi -i "$GPU" > "$OUT/gpu.txt"
printf 'model=%s\nlengths=%s\nratios=%s\nrepeats=%s\nmaxlen=%s\n' \
 "$MODEL" "$LENGTHS" "$RATIOS" "$REPEATS" "$MAXLEN" > "$OUT/config.txt"
PID=""
cleanup() {
  if [[ -n "$PID" ]] && kill -0 "$PID" 2>/dev/null; then
    kill -TERM "$PID" 2>/dev/null || true
    wait "$PID" 2>/dev/null || true
  fi
  PID=""
}
trap cleanup EXIT
for L in $LENGTHS; do
  if (( L + 1 > MAXLEN )); then
    echo "SKIP $L (max model len=$MAXLEN)" | tee -a "$OUT/skipped.txt"
    continue
  fi
  for MODE in off on; do
    SET="$RATIOS"; [[ "$MODE" == off ]] && SET="0"
    for R in $SET; do
      "$VENV/bin/python" - "$PORT" <<'PY'
import socket,sys
with socket.socket() as s:
    if s.connect_ex(("127.0.0.1", int(sys.argv[1]))) == 0:
        raise SystemExit("Port occupied; abort")
PY
      CASE="$OUT/L${L}_${MODE}_R${R}"
      mkdir -p "$CASE"
      ARGS=("$VENV/bin/vllm" serve "$MODEL" --served-model-name prefix-lab
        --max-model-len "$MAXLEN" --gpu-memory-utilization "$GPU_UTIL"
        --host 127.0.0.1 --port "$PORT")
      if [[ "$MODE" == on ]]; then
        ARGS+=(--enable-prefix-caching)
      else
        ARGS+=(--no-enable-prefix-caching)
      fi
      printf '%q ' "${ARGS[@]}" > "$CASE/command.txt"
      printf '\n' >> "$CASE/command.txt"
      "${ARGS[@]}" > "$CASE/server.log" 2>&1 &
      PID=$!
      READY=0
      for ((i=0;i<240;i++)); do
        kill -0 "$PID" 2>/dev/null || break
        if curl --noproxy '*' -fsS --max-time 2 \
          "http://127.0.0.1:$PORT/v1/models" > "$CASE/models.json" 2>/dev/null; then
          READY=1; break
        fi
        sleep 1
      done
      (( READY == 1 )) || { echo "Startup failed: $CASE/server.log" >&2; exit 1; }
      PYTHONPATH="$REPO" "$VENV/bin/python" -m src.prefix_lab \
        --model-path "$MODEL" --base-url "http://127.0.0.1:$PORT" \
        --mode "$MODE" --length "$L" --ratio "$R" --repeats "$REPEATS" \
        --block-size "$BLOCK" --out "$CASE/result.json" \
        > "$CASE/probe.log" 2>&1 || {
          cat "$CASE/probe.log" >&2; exit 3;
        }
      curl --noproxy '*' -fsS "http://127.0.0.1:$PORT/metrics" \
        > "$CASE/metrics.txt" || true
      cleanup
      echo "PASS $CASE"
    done
  done
done
"$VENV/bin/python" "$REPO/scripts/summary.py" "$OUT"
echo "Results: $OUT"
