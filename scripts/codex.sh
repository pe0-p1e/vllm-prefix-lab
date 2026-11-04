#!/usr/bin/env bash
# Start Codex with a per-process Clash/mihomo proxy.
#
# Usage:
#   bash scripts/codex.sh
#   bash scripts/codex.sh resume
#   bash scripts/codex.sh resume --last
#   CLASH_PORT=7890 bash scripts/codex.sh resume
#
# This script does not modify the parent shell, other terminals, or system
# proxy settings. It does not install software or start/stop mihomo.
set -euo pipefail

PARTITION="${FY_PARTITION:-/state/partition/fyzhou}"
CODEX_BIN="${CODEX_BIN:-$PARTITION/tools/bin/codex}"

if [[ ! -x "$CODEX_BIN" ]]; then
    echo "Codex not executable: $CODEX_BIN" >&2
    echo "Repair the partition-local Codex install.sh installation first." >&2
    exit 1
fi

# Reuse an already selected CODEX_HOME or existing login/session state.
# Only create a partition-local directory when no existing state is present.
if [[ -z "${CODEX_HOME:-}" ]]; then
    if [[ -d "$HOME/.codex" ]]; then
        export CODEX_HOME="$HOME/.codex"
    else
        export CODEX_HOME="$PARTITION/tools/codex-home"
        mkdir -p "$CODEX_HOME"
    fi
fi

# Resolve the optional code-mode helper only when a real executable exists.
# Never create a placeholder or a dangling symlink.
if [[ -z "${CODEX_CODE_MODE_HOST_PATH:-}" ]]; then
    for helper in \
        "$PARTITION/tools/bin/codex-code-mode-host" \
        "$CODEX_HOME/packages/standalone/current/bin/codex-code-mode-host" \
        "$PARTITION/tools/codex-home/packages/standalone/current/bin/codex-code-mode-host"
    do
        if [[ -x "$helper" ]]; then
            export CODEX_CODE_MODE_HOST_PATH="$helper"
            break
        fi
    done
fi

# Select a listening loopback proxy. Override with CLASH_PORT or CLASH_URL.
proxy_port_open() {
    (echo >"/dev/tcp/127.0.0.1/$1") >/dev/null 2>&1
}

if [[ -n "${CLASH_URL:-}" ]]; then
    proxy="$CLASH_URL"
elif [[ -n "${CLASH_PORT:-}" ]]; then
    if ! proxy_port_open "$CLASH_PORT"; then
        echo "No proxy listening on 127.0.0.1:$CLASH_PORT" >&2
        exit 1
    fi
    proxy="http://127.0.0.1:$CLASH_PORT"
else
    proxy=""
    for port in 17890 7890 7897; do
        if proxy_port_open "$port"; then
            proxy="http://127.0.0.1:$port"
            break
        fi
    done
    if [[ -z "$proxy" ]]; then
        echo "No local Clash/mihomo proxy found on ports 17890, 7890, 7897." >&2
        echo "Start your existing proxy first, or set CLASH_PORT=<port>." >&2
        exit 1
    fi
fi

export HTTP_PROXY="$proxy"
export HTTPS_PROXY="$proxy"
export ALL_PROXY="$proxy"
export http_proxy="$proxy"
export https_proxy="$proxy"
export all_proxy="$proxy"
export NO_PROXY="localhost,127.0.0.1,::1,${NO_PROXY:-}"
export no_proxy="$NO_PROXY"

echo "[codex] binary: $CODEX_BIN" >&2
echo "[codex] proxy:  $proxy (Codex process only)" >&2
echo "[codex] home:   $CODEX_HOME" >&2
if [[ -n "${CODEX_CODE_MODE_HOST_PATH:-}" ]]; then
    if [[ ! -x "$CODEX_CODE_MODE_HOST_PATH" ]]; then
        echo "CODEX_CODE_MODE_HOST_PATH is not executable: $CODEX_CODE_MODE_HOST_PATH" >&2
        exit 1
    fi
    echo "[codex] helper: $CODEX_CODE_MODE_HOST_PATH" >&2
fi

# Preserve every Codex CLI argument, including 'resume' and '--last'.
exec "$CODEX_BIN" "$@"
