#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
MODE="$1"
case "$MODE" in
  --quick) PREFIX_SET="1024"; CONC_SET="2" ;;
  --full) PREFIX_SET="1024 4096"; CONC_SET="1 4" ;;
  --list|--dry-run)
    echo 'Quick: APC off/on, prefix 1024, concurrency 2'
    echo 'Full: APC off/on, prefix 1024/4096, concurrency 1/4'
    exit 0 ;;
  *) echo 'Usage: bash scripts/run_matrix.sh --quick|--full|--list' >&2; exit 2 ;;
esac
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_ROOT="$LAB_REPO_DIR/results/$RUN_ID"
mkdir -p "$RUN_ROOT"
printf '%s\n' "$RUN_ROOT" > "$LAB_REPO_DIR/results/LATEST"
for prefix in $PREFIX_SET; do
  for conc in $CONC_SET; do
    for apc in off on; do
      out="$RUN_ROOT/$apc-$prefix-$conc"
      echo "===== APC=$apc prefix=$prefix concurrency=$conc ====="
      bash "$LAB_REPO_DIR/scripts/run_case.sh" "$apc" "$prefix" "$conc" "$out"
      sleep 4
    done
  done
done
python3 "$LAB_REPO_DIR/scripts/summarize.py" "$RUN_ROOT"
echo "Results: $RUN_ROOT"
