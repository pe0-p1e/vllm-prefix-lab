#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
# Official GitHub release assets, pinned and SHA-256 checked.
VERSION=v1.19.33
case "$(uname -m)" in
  x86_64) ASSET="mihomo-linux-amd64-$VERSION.gz"; SHA256="132f6f3edf9a65e62f04853aed3358472e0b2b7666daa8419edfa2ddf5bb6323" ;;
  aarch64) ASSET="mihomo-linux-arm64-$VERSION.gz"; SHA256="ba408b317080ee265c46cb923a50018c1b82cbd708551cdd46a1e3e307396187" ;;
  *) echo 'Unsupported architecture' >&2; exit 2 ;;
esac
UPSTREAM="https://github.com/MetaCubeX/mihomo/releases/download/$VERSION/$ASSET"
MIRROR="$(printenv LAB_GITHUB_MIRROR || echo https://ghfast.top/)"
ARCHIVE="$TMPDIR/mihomo-$ASSET-$$"
NEW_BIN="$LAB_TOOLS/.mihomo-new-$$"
trap 'rm -f "$ARCHIVE" "$NEW_BIN"' EXIT
curl -fL --retry 2 --connect-timeout 15 --max-time 180 -o "$ARCHIVE" "$MIRROR$UPSTREAM"
echo "$SHA256  $ARCHIVE" | sha256sum -c -
gzip -t "$ARCHIVE"
gzip -dc "$ARCHIVE" > "$NEW_BIN"
chmod 700 "$NEW_BIN"
"$NEW_BIN" -v
mv "$NEW_BIN" "$LAB_TOOLS/mihomo"
echo "Installed $LAB_TOOLS/mihomo"
