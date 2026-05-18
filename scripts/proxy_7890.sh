#!/usr/bin/env bash
#启动方式：source /workspace/hyh/yajiang-aef/scripts/proxy_7890.sh
#bash启动也可以，但会新开一个进程，导致当前终端无法走新端口
PROXY_PORT="${PROXY_PORT:-7980}"

export http_proxy=http://127.0.0.1:${PROXY_PORT}
export https_proxy=http://127.0.0.1:${PROXY_PORT}
export HTTP_PROXY=http://127.0.0.1:${PROXY_PORT}
export HTTPS_PROXY=http://127.0.0.1:${PROXY_PORT}
export all_proxy=socks5://127.0.0.1:${PROXY_PORT}
export ALL_PROXY=socks5://127.0.0.1:${PROXY_PORT}
export no_proxy=localhost,127.0.0.1
export NO_PROXY=localhost,127.0.0.1

echo "Proxy set to 127.0.0.1:${PROXY_PORT} for current shell."
