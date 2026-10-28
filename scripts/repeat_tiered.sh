#!/usr/bin/env bash
# Paired vLLM GPU-only vs CPU-KV-recovery trials with alternating order.
# All GPU/RAM guards and direct-only networking remain in run_tiered.sh.
set -euo pipefail
source "$(dirname "$0")/env.sh"
REPEATS="$(printenv TIER_REPEATS || echo 4)"
[[ "$REPEATS" =~ ^[0-9]+$ ]] || { echo 'TIER_REPEATS must be an integer' >&2; exit 2; }
(( REPEATS >= 2 && REPEATS <= 20 )) || { echo 'TIER_REPEATS must be 2..20' >&2; exit 2; }
[[ "$(printenv TIER_MODE || echo both)" == both ]] || {
  echo 'Paired comparison requires TIER_MODE=both (or unset)' >&2
  exit 2
}
ROOT="$LAB_REPO_DIR/results/tiered-repeat-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$ROOT"
printf 'repetition,order,run_root\n' > "$ROOT/manifest.csv"
printf '%s\n' "$ROOT" > "$LAB_REPO_DIR/results/TIERED_REPEAT_LATEST"
for (( i=1; i<=REPEATS; i++ )); do
  if (( i % 2 )); then ORDER="gpu-cpu"; else ORDER="cpu-gpu"; fi
  printf '=== trial %d / %d (%s) ===\n' "$i" "$REPEATS" "$ORDER"
  if ! TIER_ORDER="$ORDER" bash "$LAB_REPO_DIR/scripts/run_tiered.sh" > "$ROOT/trial-$i.log" 2>&1; then
    echo "Trial $i failed. Inspect $ROOT/trial-$i.log" >&2
    tail -n 80 "$ROOT/trial-$i.log" >&2
    exit 1
  fi
  CHILD="$(cat "$LAB_REPO_DIR/results/TIERED_LATEST")"
  [[ -s "$CHILD/gpu/probe.json" && -s "$CHILD/cpu/probe.json" ]] || {
    echo 'Paired trial finished without both probe results' >&2
    exit 1
  }
  printf '%d,%s,%s\n' "$i" "$ORDER" "$CHILD" >> "$ROOT/manifest.csv"
  tail -n 6 "$ROOT/trial-$i.log"
done
"$LAB_VENV/bin/python" "$LAB_REPO_DIR/scripts/aggregate_tiered.py" "$ROOT"
echo "Repeated results: $ROOT"
