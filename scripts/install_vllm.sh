#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
# Install dependencies without optional Mihomo. Confined to this child script.
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
command -v uv >/dev/null || { echo 'Install uv first: bash scripts/install_uv.sh' >&2; exit 2; }
INDEX_URL="$(printenv LAB_PYPI_INDEX_URL || echo https://pypi.tuna.tsinghua.edu.cn/simple)"
echo "Installing vLLM without a proxy; PyPI index: $INDEX_URL"
if [[ ! -x "$LAB_VENV/bin/python" ]]; then
  uv venv --python 3.12 --seed "$LAB_VENV"
fi
uv pip install --python "$LAB_VENV/bin/python" \
  --index-url "$INDEX_URL" vllm --torch-backend=auto
"$LAB_VENV/bin/vllm" --version
