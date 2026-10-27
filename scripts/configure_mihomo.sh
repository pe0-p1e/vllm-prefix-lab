#!/usr/bin/env bash
# Securely import a Clash/Mihomo YAML subscription for the current user only.
set -euo pipefail
source "$(dirname "$0")/env.sh"

D="$LAB_ROOT/private/mihomo"
CFG="$D/config.yaml"
BIN="$LAB_TOOLS/mihomo"
[[ -x "$BIN" ]] || { echo "Run: bash scripts/install_mihomo.sh" >&2; exit 2; }

mkdir -p "$D"
chmod 700 "$D"
umask 077
RAW="$(mktemp "$D/.download.XXXXXX")"
STAGED="$(mktemp "$D/.staged.XXXXXX.yaml")"
cleanup() { rm -f "$RAW" "$STAGED"; }
trap cleanup EXIT INT TERM

if [[ $# -eq 2 && "$1" == "--file" ]]; then
  cp -- "$2" "$RAW"
elif [[ $# -eq 0 ]]; then
  read -r -s -p 'Paste your Mihomo/Clash YAML subscription URL: ' SUB_URL
  printf '\n'
  [[ "$SUB_URL" == https://* ]] || { echo 'Expected an HTTPS URL' >&2; exit 2; }
  # Ensure the URL cannot escape the curl --config quoted string.
  case "$SUB_URL" in
    *$'\r'*|*$'\n'*|*'"'*|*'\'*)
      echo 'Invalid URL characters' >&2
      exit 2
      ;;
  esac
  # Passing config via stdin prevents exposing the subscription token in argv.
  printf 'url = "%s"\n' "$SUB_URL" | curl --config - \
    --fail --location --silent --show-error --retry 2 \
    --connect-timeout 15 --max-time 90 \
    --user-agent 'clash.meta' --output "$RAW"
  unset SUB_URL
else
  echo 'Usage: bash scripts/configure_mihomo.sh [--file /path/to/local.yaml]' >&2
  exit 2
fi

# Normalize local listeners without third-party YAML dependencies.
# Mihomo validates the generated YAML before we replace the existing config.
python3 - "$RAW" "$STAGED" <<'PY'
from pathlib import Path
import re
import sys

src, dst = map(Path, sys.argv[1:])
raw = src.read_bytes()
if not 50 <= len(raw) <= 12_000_000:
    raise SystemExit("Unexpected subscription size; refusing to import")
try:
    contents = raw.decode("utf-8-sig")
except UnicodeDecodeError:
    raise SystemExit("Subscription is not UTF-8 Clash/Mihomo YAML")
if re.search(r"(?im)^\s*<(?:html|!doctype html)", contents):
    raise SystemExit("Subscription returned HTML instead of YAML")
if not re.search(r"(?m)^(?:proxies|proxy-providers|proxy-groups|rules):\s*(?:\S|$)", contents):
    raise SystemExit("Missing Clash/Mihomo YAML sections; perhaps this is a base64 subscription")
if re.search(r"(?m)^listeners\s*:", contents):
    raise SystemExit("Custom listeners require manual review on a shared server")

fixed = {
    "mixed-port": "17890",
    "port": "0",
    "socks-port": "0",
    "redir-port": "0",
    "tproxy-port": "0",
    "allow-lan": "false",
    "bind-address": "'127.0.0.1'",
    "external-controller": "'127.0.0.1:19090'",
}
seen = set()
out = []
root = ""
for line in contents.splitlines(keepends=True):
    m = re.match(r"^([a-zA-Z][a-zA-Z0-9_-]*):(?:\s|$)", line)
    if m:
        root = m.group(1)
        if root in fixed:
            if root not in seen:
                out.append(root + ": " + fixed[root] + "\n")
                seen.add(root)
            continue
    if root == "tun" and re.match(r"^\s+enable\s*:", line):
        line = re.match(r"^\s*", line).group() + "enable: false\n"
    if root == "dns" and re.match(r"^\s+listen\s*:", line):
        line = re.match(r"^\s*", line).group() + "listen: '127.0.0.1:1053'\n"
    out.append(line)
for key, value in fixed.items():
    if key not in seen:
        out.append("\n" + key + ": " + value + "\n")
dst.write_text("".join(out), encoding="utf-8")
PY

chmod 600 "$STAGED"
if ! "$BIN" -t -d "$D" -f "$STAGED"; then
  echo 'Mihomo config validation failed; old config remains untouched.' >&2
  echo 'If subscription format is incompatible, export a Clash Meta YAML profile and use --file.' >&2
  exit 1
fi
if [[ -f "$CFG" ]]; then
  cp -- "$CFG" "$D/config.backup.$(date -u +%Y%m%dT%H%M%SZ).yaml"
fi
mv -f -- "$STAGED" "$CFG"
chmod 600 "$CFG"
echo "Configuration saved: $CFG"
echo 'Run: bash scripts/mihomo.sh start'
echo 'Then: source scripts/proxy_env.sh on'
