#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
# Installation is direct-only even if caller left Mihomo proxy variables active.
# No global or persistent shell changes.
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

if [[ -x "$LAB_TOOLS/uv" ]]; then "$LAB_TOOLS/uv" --version; exit 0; fi
if [[ -x "$LAB_ROOT/uv-bootstrap/bin/uv" ]]; then "$LAB_ROOT/uv-bootstrap/bin/uv" --version; exit 0; fi
if command -v uv >/dev/null; then uv --version; exit 0; fi

if [[ $# -eq 0 ]] || [[ $# -eq 1 && "$1" == "--pypi-mirror" ]]; then
  # Install only into lab-managed virtual environment using domestic PyPI.
  python3 -m venv "$LAB_ROOT/uv-bootstrap"
  "$LAB_ROOT/uv-bootstrap/bin/python" -m pip install \
    --index-url https://pypi.tuna.tsinghua.edu.cn/simple uv
  "$LAB_ROOT/uv-bootstrap/bin/uv" --version
else
  echo 'Usage: bash scripts/install_uv.sh [--pypi-mirror]' >&2
  exit 2
fi
