#!/usr/bin/env bash
# Fetch official GEO assets through the same GitHub mirror used for cloning.
# Usage: bash scripts/fetch_geodata.sh geoip.metadb [geosite.dat] [geoip.dat]
set -euo pipefail
source "$(dirname "$0")/env.sh"
D="$LAB_ROOT/private/mihomo"
mkdir -p "$D"
chmod 700 "$D"
umask 077
MIRROR="$(printenv LAB_GITHUB_MIRROR || echo https://ghfast.top/)"
case "$MIRROR" in
  https://*/) ;;
  *) echo "LAB_GITHUB_MIRROR must be an HTTPS URL ending in /" >&2; exit 2 ;;
esac
[[ $# -ge 1 ]] || { echo "Provide at least one filename" >&2; exit 2; }
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
  cleanup() { rm -f "$TMP"; }
  trap cleanup EXIT
  URL="https://github.com/MetaCubeX/meta-rules-dat/releases/download/latest/$NAME"
  echo "Downloading GEO asset through mirror: $NAME"
  curl -fL --retry 3 --connect-timeout 15 --max-time 180 -o "$TMP" "$MIRROR$URL"
  SIZE="$(stat -c %s "$TMP")"
  if (( SIZE < 200000 || SIZE > 80000000 )); then
    echo "Unexpected GEO asset size ($SIZE bytes); refusing to install: $NAME" >&2
    exit 1
  fi
  mv "$TMP" "$DEST"
  chmod 600 "$DEST"
  trap - EXIT
  echo "Saved $DEST ($SIZE bytes)"
done
