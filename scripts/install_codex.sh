#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
if [[ "$#" -gt 0 && "$1" == "--mirror" ]]; then
  VERSION=rust-v0.164.0
  case "$(uname -m)" in
    x86_64) ASSET=codex-x86_64-unknown-linux-musl.tar.gz; BIN_NAME=codex-x86_64-unknown-linux-musl; SHA256=90dd0b03b28afcb7fa733a4f73b413db2916a47347efd43fb669cc1c774ee8e4 ;;
    aarch64) ASSET=codex-aarch64-unknown-linux-musl.tar.gz; BIN_NAME=codex-aarch64-unknown-linux-musl; SHA256=29a507a2a3fd8f6cd94b96b762d281d299d15d7902dee6132d314b78cf9108eb ;;
    *) echo 'Unsupported architecture' >&2; exit 2 ;;
  esac
  UPSTREAM="https://github.com/openai/codex/releases/download/$VERSION/$ASSET"
  MIRROR="$(printenv LAB_GITHUB_MIRROR || echo https://ghfast.top/)"
  ARCHIVE="$TMPDIR/$ASSET-$$"
  STAGE="$TMPDIR/codex-stage-$$"
  mkdir -p "$STAGE"
  trap 'rm -f "$ARCHIVE"; rm -rf "$STAGE"' EXIT
  curl -fL --retry 2 --connect-timeout 15 --max-time 180 -o "$ARCHIVE" "$MIRROR$UPSTREAM"
  echo "$SHA256  $ARCHIVE" | sha256sum -c -
  tar -xzf "$ARCHIVE" -C "$STAGE" "$BIN_NAME"
  chmod 700 "$STAGE/$BIN_NAME"
  "$STAGE/$BIN_NAME" --version
  mv "$STAGE/$BIN_NAME" "$CODEX_INSTALL_DIR/codex"
else
  curl -fsSL https://chatgpt.com/codex/install.sh | env CODEX_NON_INTERACTIVE=1 CODEX_INSTALL_DIR="$CODEX_INSTALL_DIR" CODEX_HOME="$CODEX_HOME" sh
fi
"$CODEX_INSTALL_DIR/codex" --version
echo "Codex login/config: $CODEX_HOME"
echo 'Run codex login --device-auth'
