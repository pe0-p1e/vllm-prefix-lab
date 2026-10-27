#!/usr/bin/env bash
# Obtain upstream GeoIP/GeoSite files via project-documented CDN endpoints.
set -euo pipefail
source "$(dirname "$0")/env.sh"
D="$LAB_ROOT/private/mihomo"
mkdir -p "$D"
chmod 700 "$D"
umask 077

[[ $# -gt 0 ]] || { echo 'Usage: bash scripts/fetch_geodata.sh filename ...' >&2; exit 2; }
for NAME in "$@"; do
  case "$NAME" in
    geoip.metadb|geoip.dat|geosite.dat|country.mmdb|GeoLite2-ASN.mmdb) ;;
    *) echo "Unsupported GEO filename: $NAME" >&2; exit 2 ;;
  esac
  DEST="$D/$NAME"
  if [[ -s "$DEST" ]] && [[ $(stat -c %s "$DEST") -gt 200000 ]]; then
    echo "Already present: $DEST"
    continue
  fi
  TMP="$(mktemp "$D/.geo-download.XXXXXX")"
  trap 'rm -f "$TMP"' EXIT
  OK=0
  for URL in \
    "https://cdn.jsdelivr.net/gh/MetaCubeX/meta-rules-dat@release/$NAME" \
    "https://testingcf.jsdelivr.net/gh/MetaCubeX/meta-rules-dat@release/$NAME" \
    "https://ghproxy.net/https://github.com/MetaCubeX/meta-rules-dat/releases/download/latest/$NAME"; do
    : > "$TMP"
    HOST="$(printf '%s' "$URL" | cut -d/ -f3)"
    echo "Downloading $NAME via $HOST"
    if curl -fsSL --retry 1 --connect-timeout 10 --max-time 90 -o "$TMP" "$URL"; then
      SIZE="$(stat -c %s "$TMP")"
      if (( SIZE >= 200000 && SIZE <= 80000000 )); then
        mv "$TMP" "$DEST"
        chmod 600 "$DEST"
        trap - EXIT
        OK=1
        echo "Saved $DEST ($SIZE bytes)"
        break
      fi
      echo "Rejected unexpected file size ($SIZE bytes)" >&2
    fi
  done
  if (( OK == 0 )); then
    echo "All upstream GEO CDN/mirror downloads failed for $NAME" >&2
    echo "Download $NAME on another machine and copy it to $DEST, then retry." >&2
    exit 1
  fi
done
