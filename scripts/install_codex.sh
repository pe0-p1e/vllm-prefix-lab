#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
echo "Binary: $CODEX_INSTALL_DIR, Auth/config: $CODEX_HOME"
# Official standalone installer, isolated installation path.
if [[ "$(printenv CODEX_FROM_GITHUB || echo 0)" == 1 ]]; then
  curl -fsSL https://chatgpt.com/codex/install.sh | env CODEX_NON_INTERACTIVE=1 CODEX_INSTALLER_USE_RELEASES_OPENAI_COM=false CODEX_INSTALL_DIR="$CODEX_INSTALL_DIR" CODEX_HOME="$CODEX_HOME" sh
else
  curl -fsSL https://chatgpt.com/codex/install.sh | env CODEX_NON_INTERACTIVE=1 CODEX_INSTALL_DIR="$CODEX_INSTALL_DIR" CODEX_HOME="$CODEX_HOME" sh
fi
"$LAB_TOOLS/codex" --version
echo 'Run: codex login --device-auth'
