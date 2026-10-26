#!/usr/bin/env bash
# Import a Clash/Mihomo YAML subscription into a private, loopback-only config.
# The subscription URL is read silently and never written to shell history or Git.
set -euo pipefail
source "$(dirname "$0")/env.sh"
D="$LAB_ROOT/private/mihomo"
CFG="$D/config.yaml"
BIN="$LAB_TOOLS/mihomo"
[[ -x "$BIN" ]] || { echo "Install Mihomo first: bash scripts/install_mihomo.sh" >&2; exit 2; }
mkdir -p "$D"
chmod 700 "$D"
umask 077
RAW="$(mktemp "$D/.download.XXXXXX")"
STAGED="$(mktemp "$D/.staged.XXXXXX.yaml")"
cleanup() { rm -f "$RAW" "$STAGED"; }
trap cleanup EXIT INT TERM
if [[ $# -eq 2 && "$1" == "--file" ]]; then
  cp -- "$2" "$RAW"
else
  if [[ $# -ne 0 ]]; then echo 'Usage: bash scripts/configure_mihomo.sh [--file /path/to/local.yaml]' >&2; exit 2; fi
  read -r -s -p 'Paste your Mihomo/Clash YAML subscription URL: ' SUB_URL
  printf '\n'
  [[ "$SUB_URL" == https://* ]] || { echo 'Expected an HTTPS subscription URL' >&2; exit 2; }
  # Some subscription servers choose the Clash YAML format based on User-Agent.
  curl --fail --location --silent --show-error --retry 2 \
    --connect-timeout 15 --max-time 90 \
    --user-agent 'clash.meta' -o "$RAW" "$SUB_URL"
  unset SUB_URL
fi
# Pure Python stdlib edit of top-level keys. Validity is checked with Mihomo below.
python3 - "$RAW" "$STAGED" <<'PY'
from pathlib import Path
import re
import sys
src, dst = map(Path, sys.argv[1:])
raw = src.read_bytes()
if len(raw) < 50 or len(raw) > 12_000_000:
    raise SystemExit("Unusual subscription size; refusing to import")
try:
    text = raw.decode("utf-8-sig")
except UnicodeDecodeError:
    raise SystemExit("Subscription is not UTF-8 Clash/Mihomo YAML; could be a base64/v2ray format")
if re.search(r"(?im)^\s*<(?:html|!doctype html)", text):
    raise SystemExit("Subscription returned HTML instead of YAML")
if not re.search(r"(?m)^(?:proxies|proxy-providers|proxy-groups|rules):\s*(?:\S|$)", text):
    raise SystemExit("No Clash/Mihomo YAML sections found; check subscription type")
if re.search(r"(?m)^listeners:\s*", text):
    raise SystemExit("Custom listeners: detected; manual review required on shared server")
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
for line in text.splitlines(keepends=True):
    m = re.match(r"^([a-zA-Z][a-zA-Z0-9_-]*):(?:\s|$)", line)
    if m:
        root = m.group(1)
        if root in fixed:
            if root not in seen:
                out.append(root + ": " + fixed[root] + "\n")
                seen.add(root)
            continue
    if root == "tun" and re.match(r"^\s+enable\s*:", line):
        indent = re.match(r"^\s*", line).group()
        line = indent + "enable: false\n"
    if root == "dns" and re.match(r"^\s+listen\s*:", line):
        indent = re.match(r"^\s*", line).group()
        line = indent + "listen: '127.0.0.1:1053'\n"
    out.append(line)
for k, v in fixed.items():
    if k not in seen:
        out.append("\n" + k + ": " + v + "\n")
dst.write_text("".join(out), encoding="utf-8")
PY
chmod 600 "$STAGED"
# -t may download upstream GEO/rule files according to the user's provider config.
# Never replace a working configuration if validation fails.
if ! "$BIN" -t -d "$D" -f "$STAGED"; then
  echo 'Mihomo rejected this config. Existing config remains unchanged.' >&2
  echo 'If the subscription is not in Clash/Mihomo YAML format, export Clash Meta YAML from your provider or use --file.' >&2
  exit 1
fi
if [[ -e "$CFG" ]]; then
  cp -- "$CFG" "$D/config.backup.$(date -u +%Y%m%dT%H%M%SZ).yaml"
  chmod 600 "$D"/config.backup.*.yaml
fi
mv -f -- "$STAGED" "$CFG"
chmod 600 "$CFG"
echo "Validated Mihomo configuration: $CFG"
echo 'Local HTTP/SOCKS proxy: 127.0.0.1:17890'
echo 'Next: bash scripts/mihomo.sh start && source scripts/proxy_env.sh on'
