#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
printf '\nPaths: repo=%s root=%s venv=%s\n' "$LAB_REPO_DIR" "$LAB_ROOT" "$LAB_VENV"
uname -srm
python3 --version || true
for cmd in uv codex mihomo gh vllm curl git nvidia-smi; do
  printf '%-12s ' "$cmd"; command -v "$cmd" || true
done
printf '\nAvailable models:\n'
for p in /state/partition/model/Qwen3-0.6B /state/partition/model/Qwen3.5-2B; do
  [[ ! -d "$p" ]] || echo "$p"
done
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader || true
git -C "$LAB_REPO_DIR" status -sb || true
