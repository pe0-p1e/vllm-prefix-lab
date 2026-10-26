#!/usr/bin/env bash
# Source as: source scripts/proxy_env.sh on|off|status
if [[ "$0" == "$BASH_SOURCE" ]]; then echo 'This file must be sourced' >&2; exit 2; fi
case "$1" in
  on)
    export HTTP_PROXY="$(printenv LAB_PROXY_URL || echo http://127.0.0.1:17890)"
    export HTTPS_PROXY="$HTTP_PROXY" ALL_PROXY="$HTTP_PROXY"
    export http_proxy="$HTTP_PROXY" https_proxy="$HTTP_PROXY" all_proxy="$HTTP_PROXY"
    export NO_PROXY="127.0.0.1,localhost,::1" no_proxy="127.0.0.1,localhost,::1"
    echo "Proxy enabled ONLY in this shell: $HTTP_PROXY"
    ;;
  off)
    unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
    echo 'Proxy disabled in this shell'
    ;;
  status)
    printf 'HTTP_PROXY=%s\n' "$(printenv HTTP_PROXY || echo unset)"
    ;;
  *) echo 'Use on|off|status' >&2; return 2 ;;
esac
