#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
if [[ -x "$LAB_TOOLS/uv" ]]; then "$LAB_TOOLS/uv" --version; exit 0; fi
if [[ -x "$LAB_ROOT/uv-bootstrap/bin/uv" ]]; then "$LAB_ROOT/uv-bootstrap/bin/uv" --version; exit 0; fi
if command -v uv >/dev/null; then uv --version; exit 0; fi
if [[ "$#" -gt 0 && "$1" == "--pypi-mirror" ]]; then
  python3 -m venv "$LAB_ROOT/uv-bootstrap"
  "$LAB_ROOT/uv-bootstrap/bin/python" -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple uv
  "$LAB_ROOT/uv-bootstrap/bin/uv" --version
else
  curl -fsSL https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$LAB_TOOLS" UV_NO_MODIFY_PATH=1 sh
  "$LAB_TOOLS/uv" --version
fi
