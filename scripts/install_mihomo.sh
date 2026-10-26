#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
case "$(uname -m)" in
  x86_64) WANT='linux-amd64' ;;
  aarch64) WANT='linux-arm64' ;;
  *) echo 'Unsupported architecture' >&2; exit 2 ;;
esac
URL="$(WANT="$WANT" python3 - <<'PY'
import json,os,urllib.request
r=urllib.request.Request('https://api.github.com/repos/MetaCubeX/mihomo/releases/latest',headers={'User-Agent':'vllm-prefix-lab'})
d=json.load(urllib.request.urlopen(r,timeout=30)); w=os.environ['WANT']
a=[x for x in d['assets'] if w in x['name'] and x['name'].endswith('.gz') and 'compatible' not in x['name']]
a.sort(key=lambda x:('v3' not in x['name'],'v2' not in x['name'],len(x['name'])))
if not a: raise SystemExit('Official Mihomo binary not found')
print(a[0]['browser_download_url'])
PY
)"
ARCHIVE="$TMPDIR/mihomo-download-$$.gz"
trap 'rm -f "$ARCHIVE"' EXIT
curl -fL --retry 3 -o "$ARCHIVE" "$URL"
gzip -t "$ARCHIVE"
gzip -dc "$ARCHIVE" > "$LAB_TOOLS/mihomo"
chmod 700 "$LAB_TOOLS/mihomo"
"$LAB_TOOLS/mihomo" -v
