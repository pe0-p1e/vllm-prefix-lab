#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
D="$LAB_ROOT/private/mihomo"
BIN="$LAB_TOOLS/mihomo"
CFG="$D/config.yaml"
PID="$D/mihomo.pid"
LOG="$D/mihomo.log"
mkdir -p "$D"
case "$1" in
 init)
    if [[ -e "$CFG" ]]; then echo "Config exists: $CFG"; exit 0; fi
    cp "$LAB_REPO_DIR/configs/mihomo.example.yaml" "$CFG"
    chmod 600 "$CFG"
    echo "DIRECT-only sample created: $CFG; replace with trusted private profile"
    ;;
 start)
    [[ -x "$BIN" && -f "$CFG" ]] || { echo 'Install mihomo and supply a private config first' >&2; exit 2; }
    if [[ -e "$PID" ]] && kill -0 "$(cat "$PID")" 2>/dev/null; then echo 'Already running'; exit 0; fi
    "$BIN" -t -d "$D" -f "$CFG"
    nohup "$BIN" -d "$D" -f "$CFG" > "$LOG" 2>&1 </dev/null &
    echo "$!" > "$PID"
    sleep 1
    kill -0 "$(cat "$PID")" 2>/dev/null || { tail -n 40 "$LOG"; exit 1; }
    echo "Started owned Mihomo process: $(cat "$PID")"
    ;;
 stop)
    [[ -f "$PID" ]] || { echo 'Not running'; exit 0; }
    p="$(cat "$PID")"
    if kill -0 "$p" 2>/dev/null && ps -p "$p" -o args= | grep -F "$BIN" >/dev/null; then kill "$p"; fi
    rm -f "$PID"
    ;;
 status)
    if [[ -f "$PID" ]] && kill -0 "$(cat "$PID")" 2>/dev/null; then echo running; else echo stopped; fi
    ;;
 logs) tail -n 40 "$LOG" ;;
 *) echo 'Usage: bash scripts/mihomo.sh init|start|stop|status|logs' >&2; exit 2 ;;
esac
