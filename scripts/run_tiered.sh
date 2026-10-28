#!/usr/bin/env bash
# HBM-only APC versus native pinned-DRAM KV offload on one idle GPU.
set -euo pipefail
source "$(dirname "$0")/env.sh"
# vLLM gpu-memory-utilization is a fraction of total VRAM, not free VRAM.
export LAB_REQUIRED_GPU_FRACTION="$(printenv TIER_GPU_UTIL || echo 0.28)"
source "$(dirname "$0")/select_gpu.sh"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

MODEL_PATH="$(printenv MODEL_OVERRIDE || echo /state/partition/model/Qwen3-0.6B)"
PREFIX_LEN="$(printenv TIER_PREFIX_LEN || echo 8192)"
SUFFIX_LEN="$(printenv TIER_SUFFIX_LEN || echo 64)"
OUTPUT_LEN="$(printenv TIER_OUTPUT_LEN || echo 256)"
MAX_MODEL_LEN="$(printenv TIER_MAX_MODEL_LEN || echo 16384)"
GPU_UTIL="$(printenv TIER_GPU_UTIL || echo 0.28)"
CPU_GIB="$(printenv TIER_CPU_GIB || echo 12)"
PORT="$(printenv TIER_PORT || echo 18081)"
MODE_CHOICE="$(printenv TIER_MODE || echo both)"
VLLM="$LAB_VENV/bin/vllm"
PYTHON="$LAB_VENV/bin/python"

[[ -x "$VLLM" && -x "$PYTHON" ]] || { echo 'Install vLLM first' >&2; exit 2; }
[[ -d "$MODEL_PATH" ]] || { echo "Model not found: $MODEL_PATH" >&2; exit 2; }
[[ "$PREFIX_LEN" =~ ^[0-9]+$ && "$SUFFIX_LEN" =~ ^[0-9]+$ && "$OUTPUT_LEN" =~ ^[0-9]+$ ]] || exit 2
(( PREFIX_LEN + SUFFIX_LEN + OUTPUT_LEN < MAX_MODEL_LEN )) || { echo 'Context too short' >&2; exit 2; }
[[ "$MODE_CHOICE" == both || "$MODE_CHOICE" == cpu || "$MODE_CHOICE" == gpu ]] || { echo 'TIER_MODE must be both, gpu, or cpu' >&2; exit 2; }

if [[ "$MODE_CHOICE" != gpu ]]; then
  python3 - "$CPU_GIB" <<'PY'
import sys
with open("/proc/meminfo") as stream:
    available_kib = int(next(s.split()[1] for s in stream if s.startswith("MemAvailable:")))
required = (float(sys.argv[1]) + 6) * (1024**2)
print(f"Available host RAM: {available_kib/1024/1024:.1f} GiB; CPU KV budget: {sys.argv[1]} GiB")
if available_kib < required:
    sys.exit("Insufficient available RAM for CPU cache + 6 GiB headroom on shared node.")
PY
fi

RUN_ROOT="$LAB_REPO_DIR/results/tiered-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$RUN_ROOT"
printf '%s\n' "$RUN_ROOT" > "$LAB_REPO_DIR/results/TIERED_LATEST"
{
  echo "model=$MODEL_PATH gpu=$CUDA_VISIBLE_DEVICES prefix=$PREFIX_LEN suffix=$SUFFIX_LEN output=$OUTPUT_LEN"
  echo "gpu_memory_utilization=$GPU_UTIL cpu_gib=$CPU_GIB max_model_len=$MAX_MODEL_LEN port=$PORT"
  echo "shared_gpu_opt_in=${LAB_ALLOW_SHARED_GPU:-0} gpu_reserve_mib=${LAB_GPU_RESERVE_MIB:-2048}"
  echo "tier_order=${TIER_ORDER:-gpu-cpu}"
  "$VLLM" --version
  nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader
} > "$RUN_ROOT/setup.txt"

python3 - "$PORT" <<'PY'
import socket,sys
with socket.socket() as sock:
    sock.settimeout(.5)
    if sock.connect_ex(("127.0.0.1",int(sys.argv[1]))) == 0:
        sys.exit("Port already in use. Set TIER_PORT; will not kill other services.")
PY

SERVER_PID=""
stop_server() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  SERVER_PID=""
}
trap stop_server EXIT INT TERM
MODES="gpu cpu"
TIER_ORDER="${TIER_ORDER:-gpu-cpu}"
if [[ "$TIER_ORDER" == cpu-gpu ]]; then
  MODES="cpu gpu"
elif [[ "$TIER_ORDER" != gpu-cpu ]]; then
  echo "TIER_ORDER must be gpu-cpu or cpu-gpu" >&2
  exit 2
fi
if [[ "$MODE_CHOICE" != both ]]; then MODES="$MODE_CHOICE"; fi
export VLLM_USE_FLASHINFER_SAMPLER=0
# Lock the filler count across GPU and CPU modes so they use the same trace.
FILLERS_LOCKED="$(printenv TIER_FILLERS || true)"

for MODE in $MODES; do
  # Other jobs can change memory use between the two server restarts.
  source "$(dirname "$0")/select_gpu.sh"
  OUT="$RUN_ROOT/$MODE"
  mkdir -p "$OUT"
  echo "=== $MODE: prefix=$PREFIX_LEN output=$OUTPUT_LEN GPU=$CUDA_VISIBLE_DEVICES ==="
  nvidia-smi --query-gpu=index,memory.used,memory.free,memory.total,utilization.gpu --format=csv,noheader >> "$OUT/machine_before.txt"
  CMD=("$VLLM" serve "$MODEL_PATH"
    --served-model-name qwen-tiered-lab
    --max-model-len "$MAX_MODEL_LEN"
    --gpu-memory-utilization "$GPU_UTIL"
    --enable-prefix-caching
    --host 127.0.0.1 --port "$PORT")
  if [[ "$MODE" == cpu ]]; then
    CMD+=(--kv-offloading-backend native --kv-offloading-size "$CPU_GIB")
  fi
  printf '%q ' "${CMD[@]}" > "$OUT/command.txt"
  printf '\n' >> "$OUT/command.txt"
  "${CMD[@]}" > "$OUT/server.log" 2>&1 &
  SERVER_PID=$!

  READY=0
  for ((i=0; i<240; i++)); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "Server exited: $OUT/server.log" >&2
      bash "$LAB_REPO_DIR/scripts/diagnose_startup.sh" "$OUT/server.log" >&2 || true
      exit 1
    fi
    if curl --noproxy '*' -fsS --max-time 1 "http://127.0.0.1:$PORT/v1/models" > "$OUT/models.json" 2>/dev/null; then
      READY=1; break
    fi
    sleep 1
  done
  (( READY == 1 )) || { echo "Startup timeout: $OUT/server.log" >&2; exit 1; }

  GPU_TOKENS="$(grep -oE 'GPU KV cache size: [0-9,]+ tokens' "$OUT/server.log" | tail -n1 | awk '{print $5}' | tr -d ',' || true)"
  [[ "$GPU_TOKENS" =~ ^[0-9]+$ ]] || { echo 'Missing GPU KV cache capacity; aborting unverified eviction experiment.' >&2; exit 1; }
  AUTO_FILLERS=$(( (GPU_TOKENS + PREFIX_LEN - 1) / PREFIX_LEN + 1 ))
  if [[ -z "$FILLERS_LOCKED" ]]; then FILLERS_LOCKED="$AUTO_FILLERS"; fi
  FILLERS="$FILLERS_LOCKED"
  (( FILLERS >= 2 && FILLERS <= 32 )) || { echo "Unreasonable filler count: $FILLERS" >&2; exit 1; }
  echo "GPU KV cache size: $GPU_TOKENS tokens; distinct filler prefixes: $FILLERS"

  if [[ "$MODE" == cpu ]]; then
    GPU_CACHE_GIB="$(grep -oE 'Available KV cache memory: [0-9.]+ GiB' "$OUT/server.log" | tail -n1 | awk '{print $5}' || true)"
    [[ -n "$GPU_CACHE_GIB" ]] || { echo 'Missing GPU KV memory estimate; aborting CPU capacity check' >&2; exit 1; }
    python3 - "$GPU_CACHE_GIB" "$GPU_TOKENS" "$FILLERS" "$PREFIX_LEN" "$SUFFIX_LEN" "$OUTPUT_LEN" "$CPU_GIB" <<'PY'
import sys
gpu_gib,tokens,fillers,prefix,suffix,out,cpu_gib = map(float,sys.argv[1:])
estimated = gpu_gib/tokens * (fillers+1)*(prefix+suffix+out)
print(f"Estimated host KV required: {estimated:.2f} GiB / {cpu_gib:.2f} GiB")
if estimated > .85*cpu_gib:
    sys.exit("Host cache too small for controlled A-B-A test. Raise TIER_CPU_GIB or lower TIER_GPU_UTIL.")
PY
  fi

  "$PYTHON" "$LAB_REPO_DIR/scripts/tiered_probe.py" \
    --mode "$MODE" --model-path "$MODEL_PATH" \
    --base-url "http://127.0.0.1:$PORT" \
    --prefix-len "$PREFIX_LEN" --suffix-len "$SUFFIX_LEN" \
    --output-len "$OUTPUT_LEN" --fillers "$FILLERS" \
    --gpu-kv-tokens "$GPU_TOKENS" \
    --out "$OUT/probe.json" | tee "$OUT/probe.log"
  curl --noproxy '*' -fsS "http://127.0.0.1:$PORT/metrics" > "$OUT/metrics_after.txt" || true
  stop_server
  sleep 4
done
"$PYTHON" "$LAB_REPO_DIR/scripts/summarize_tiered.py" "$RUN_ROOT"
echo "Results: $RUN_ROOT"
