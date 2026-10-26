#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
command -v uv >/dev/null || { echo 'Install uv first' >&2; exit 2; }
if [[ ! -x "$LAB_VENV/bin/python" ]]; then uv venv --python 3.12 --seed "$LAB_VENV"; fi
uv pip install --python "$LAB_VENV/bin/python" vllm --torch-backend=auto
"$LAB_VENV/bin/vllm" --version
