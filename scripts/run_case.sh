#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
source "$(dirname "$0")/select_gpu.sh"
[[ $# -eq 4 ]] || { echo 'Usage: run_case.sh on|off prefix_len concurrency out_dir' >&2; exit 2; }
APC="$1"; PREFIX_LEN="$2"; CONCURRENCY="$3"; OUT="$4"
[[ "$APC" == on || "$APC" == off ]] || exit 2
[[ "$PREFIX_LEN" =~ ^[0-9]+$ && "$CONCURRENCY" =~ ^[0-9]+$ ]] || exit 2
VLLM="$LAB_VENV/bin/vllm"
[[ -x "$VLLM" ]] || { echo 'First run: bash scripts/install_vllm.sh' >&2; exit 2; }
MODEL_PATH=""
for m in /state/partition/model/Qwen3-0.6B /state/partition/model/Qwen3.5-2B; do
  if [[ -d "$m" ]]; then MODEL_PATH="$m"; break; fi
done
MODEL_OVERRIDE="$(printenv MODEL_OVERRIDE || true)"
if [[ -n "$MODEL_OVERRIDE" ]]; then MODEL_PATH="$MODEL_OVERRIDE"; fi
[[ -d "$MODEL_PATH" ]] || { echo 'Set MODEL_OVERRIDE to an existing local model directory' >&2; exit 2; }
MODEL_LEN="$(printenv MAX_MODEL_LEN || echo 8192)"
SUFFIX_LEN="$(printenv SUFFIX_LEN || echo 128)"
OUTPUT_LEN="$(printenv OUTPUT_LEN || echo 32)"
GPU_MEMORY_UTILIZATION="$(printenv GPU_MEMORY_UTILIZATION || echo 0.45)"
# FlashInfer sampling JIT is not required for the prefix-cache experiment.
# The PyTorch/Triton sampler avoids a missing/broken nvcc/ninja toolchain.
# Override with VLLM_USE_FLASHINFER_SAMPLER=1 only after toolchain verification.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
REQUEST_RATE="$(printenv REQUEST_RATE || echo 2)"
NUM_PROMPTS="$(printenv NUM_PROMPTS || echo 48)"
NUM_PREFIXES="$(printenv NUM_PREFIXES || echo 4)"
LAB_PORT="$(printenv LAB_PORT || echo 18080)"
(( PREFIX_LEN + SUFFIX_LEN + OUTPUT_LEN < MODEL_LEN )) || { echo 'Context budget exceeded' >&2; exit 2; }
mkdir -p "$OUT"
python3 - "$LAB_PORT" <<'PY'
import socket,sys
s=socket.socket()
try:
    s.settimeout(0.5)
    if s.connect_ex(('127.0.0.1',int(sys.argv[1]))) == 0:
        raise SystemExit('Port already in use. Will not terminate another service.')
finally:
    s.close()
PY
HELP="$("$VLLM" serve --help=all 2>&1 || true)"
if [[ "$APC" == on ]]; then
  if grep -q -- '--enable-prefix-caching' <<< "$HELP"; then FLAG='--enable-prefix-caching'
  else echo 'APC enable flag not found: inspect vllm serve --help=all' >&2; exit 2; fi
else
  if grep -q -- '--no-enable-prefix-caching' <<< "$HELP"; then FLAG='--no-enable-prefix-caching'
  elif grep -q -- '--disable-prefix-caching' <<< "$HELP"; then FLAG='--disable-prefix-caching'
  else echo 'APC disable flag not found: inspect vllm serve --help=all' >&2; exit 2; fi
fi
"$VLLM" --version > "$OUT/vllm_version.txt" 2>&1 || true
{ date -Is; uname -a; nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader || true; } > "$OUT/machine.txt"
printf 'apc=%s\nprefix_len=%s\nconcurrency=%s\nmodel_path=%s\nmax_model_len=%s\ngpu_memory_utilization=%s\nrequest_rate=%s\nnum_prompts=%s\nnum_prefixes=%s\nsuffix_len=%s\noutput_len=%s\n' "$APC" "$PREFIX_LEN" "$CONCURRENCY" "$MODEL_PATH" "$MODEL_LEN" "$GPU_MEMORY_UTILIZATION" "$REQUEST_RATE" "$NUM_PROMPTS" "$NUM_PREFIXES" "$SUFFIX_LEN" "$OUTPUT_LEN" > "$OUT/config.txt"
printf 'CUDA_VISIBLE_DEVICES=%s\n' "$CUDA_VISIBLE_DEVICES" >> "$OUT/config.txt"
printf 'VLLM_USE_FLASHINFER_SAMPLER=%s\nFLASHINFER_WORKSPACE_BASE=%s\n' "$VLLM_USE_FLASHINFER_SAMPLER" "$FLASHINFER_WORKSPACE_BASE" >> "$OUT/config.txt"
"$VLLM" serve "$MODEL_PATH" \
 --served-model-name qwen-lab \
 --max-model-len "$MODEL_LEN" \
 --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
 "$FLAG" --host 127.0.0.1 --port "$LAB_PORT" > "$OUT/server.log" 2>&1 &
SERVER_PID=$!
stop_server() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap stop_server EXIT INT TERM
READY=0
for ((i=0;i<240;i++)); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "Server exited during startup. See $OUT/server.log" >&2
    bash "$LAB_REPO_DIR/scripts/diagnose_startup.sh" "$OUT/server.log" 2>&1 | tee "$OUT/startup_diagnostics.txt" >&2
    tail -n 20 "$OUT/server.log" >&2
    exit 1
  fi
  if curl --noproxy '*' -fsS --max-time 1 "http://127.0.0.1:$LAB_PORT/v1/models" > "$OUT/models.json" 2>/dev/null; then READY=1; break; fi
  sleep 1
done
[[ "$READY" -eq 1 ]] || { echo "Server readiness timeout: $OUT/server.log" >&2; exit 1; }
curl --noproxy '*' -fsS "http://127.0.0.1:$LAB_PORT/metrics" > "$OUT/metrics_before.txt" || true
set +e
"$VLLM" bench serve \
 --backend openai \
 --host 127.0.0.1 --port "$LAB_PORT" \
 --model qwen-lab --tokenizer "$MODEL_PATH" \
 --dataset-name prefix_repetition \
 --prefix-repetition-prefix-len "$PREFIX_LEN" \
 --prefix-repetition-suffix-len "$SUFFIX_LEN" \
 --prefix-repetition-num-prefixes "$NUM_PREFIXES" \
 --prefix-repetition-output-len "$OUTPUT_LEN" \
 --num-prompts "$NUM_PROMPTS" --request-rate "$REQUEST_RATE" \
 --max-concurrency "$CONCURRENCY" \
 --seed 42 --percentile-metrics ttft,tpot,e2el --metric-percentiles 50,95,99 \
 --save-result --result-dir "$OUT" --result-filename result.json \
 >"$OUT/bench.log" 2>&1
RC=$?
set -e
curl --noproxy '*' -fsS "http://127.0.0.1:$LAB_PORT/metrics" > "$OUT/metrics_after.txt" || true
if (( RC != 0 )); then echo "Benchmark exit=$RC; see $OUT/bench.log" >&2; tail -n 50 "$OUT/bench.log" >&2; exit "$RC"; fi
[[ -s "$OUT/result.json" ]] || { echo "No benchmark JSON in $OUT" >&2; exit 1; }
echo "Saved $OUT/result.json"
