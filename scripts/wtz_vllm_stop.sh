#!/usr/bin/env bash
set -euo pipefail

VLLM_UNIT="${VLLM_UNIT:-vllm-qwen14b.service}"
VLLM_PORT="${VLLM_PORT:-8000}"

if ! command -v fuser >/dev/null 2>&1; then
    echo "[ERROR] 未找到 fuser，请安装 psmisc" >&2
    exit 1
fi

echo "[vLLM] 停止 systemd service: $VLLM_UNIT"

if systemctl is-active --quiet "$VLLM_UNIT"; then
    sudo systemctl stop "$VLLM_UNIT"
fi

sudo systemctl reset-failed "$VLLM_UNIT" 2>/dev/null || true

echo "[vLLM] 检查端口 $VLLM_PORT 上的遗留进程"

if sudo fuser "${VLLM_PORT}/tcp" >/dev/null 2>&1; then
    echo "[vLLM] 发现遗留进程，发送 SIGTERM"
    sudo fuser -TERM -k "${VLLM_PORT}/tcp" 2>/dev/null || true

    for _ in $(seq 1 15); do
        if ! sudo fuser "${VLLM_PORT}/tcp" >/dev/null 2>&1; then
            break
        fi
        sleep 1
    done
fi

if sudo fuser "${VLLM_PORT}/tcp" >/dev/null 2>&1; then
    echo "[vLLM] 遗留进程未退出，发送 SIGKILL"
    sudo fuser -KILL -k "${VLLM_PORT}/tcp" 2>/dev/null || true
fi

if sudo fuser "${VLLM_PORT}/tcp" >/dev/null 2>&1; then
    echo "[ERROR] 端口 $VLLM_PORT 仍被占用" >&2

    if command -v ss >/dev/null 2>&1; then
        sudo ss -tlnp |
            grep -E ":${VLLM_PORT}([^0-9]|$)" || true
    fi

    exit 1
fi

echo "[vLLM] 端口 $VLLM_PORT 已释放"