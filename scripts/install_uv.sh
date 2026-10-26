#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
if [[ -x "$LAB_TOOLS/uv" ]]; then "$LAB_TOOLS/uv" --version; exit 0; fi
if command -v uv >/dev/null; then echo "Using existing $(command -v uv)"; uv --version; exit 0; fi
curl -fsSL https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$LAB_TOOLS" sh
"$LAB_TOOLS/uv" --version
