#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"

# Only this installer child shell: no Mihomo proxy, no global environment edits.
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
command -v uv >/dev/null || { echo 'uv missing: run scripts/install_uv.sh' >&2; exit 2; }

INDEX_URL="$(printenv LAB_PYPI_INDEX_URL || echo https://pypi.tuna.tsinghua.edu.cn/simple)"
VLLM_SPEC="$(printenv LAB_VLLM_SPEC || echo 'vllm==0.32.0')"

# Default 30s uv HTTP stall timeout is too short for large CUDA wheels on
# some campus networks. Avoid saturating the network or storage with downloads.
export UV_HTTP_TIMEOUT="$(printenv LAB_UV_HTTP_TIMEOUT || echo 300)"
export UV_HTTP_CONNECT_TIMEOUT="$(printenv LAB_UV_HTTP_CONNECT_TIMEOUT || echo 30)"
export UV_HTTP_RETRIES="$(printenv LAB_UV_HTTP_RETRIES || echo 6)"
export UV_CONCURRENT_DOWNLOADS="$(printenv LAB_UV_CONCURRENT_DOWNLOADS || echo 2)"
export UV_CONCURRENT_INSTALLS="$(printenv LAB_UV_CONCURRENT_INSTALLS || echo 2)"

echo "Install: $VLLM_SPEC"
echo "PyPI: $INDEX_URL"
echo "Proxy: disabled in installer subprocess"
echo "uv read timeout: $UV_HTTP_TIMEOUT seconds; retries: $UV_HTTP_RETRIES"
echo "Concurrent downloads: $UV_CONCURRENT_DOWNLOADS; installs: $UV_CONCURRENT_INSTALLS"
echo 'Available disk space and inodes:'
df -hP "$LAB_ROOT" "$UV_CACHE_DIR" || true
df -Pi "$LAB_ROOT" "$UV_CACHE_DIR" || true

if [[ ! -x "$LAB_VENV/bin/python" ]]; then
  uv venv --python 3.12 --seed "$LAB_VENV"
fi

echo 'Installing into existing virtual environment; previously cached wheels will be reused.'
uv pip install --python "$LAB_VENV/bin/python" \
  --index-url "$INDEX_URL" \
  --torch-backend=auto \
  "$VLLM_SPEC"

"$LAB_VENV/bin/vllm" --version
echo "Installed in: $LAB_VENV"
