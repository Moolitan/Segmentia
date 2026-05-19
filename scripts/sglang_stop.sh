#!/usr/bin/env bash
# 停止由 sglang_start.sh 启动的本地 SGLang 服务，并确保端口释放。
# 用法: ./sglang_stop.sh  或  SGLANG_PORT=8001 ./sglang_stop.sh
#
# 可选环境变量:
#   SGLANG_PORT  与 sglang_start.sh 一致的端口，默认 8000

set -u

SGLANG_PORT="${SGLANG_PORT:-8000}"

# 匹配监听行中的端口（避免 8000 误匹配 80000）
_ss_port_pat() {
    echo ":${SGLANG_PORT}([^0-9]|$)"
}

port_in_use() {
    if command -v ss &>/dev/null; then
        ss -tlnp 2>/dev/null | grep -qE "$(_ss_port_pat)"
        return $?
    fi
    if command -v lsof &>/dev/null; then
        lsof -nP -iTCP:"${SGLANG_PORT}" -sTCP:LISTEN >/dev/null 2>&1
        return $?
    fi
    return 1
}

# 返回当前在 PORT 上 LISTEN 的 PID 列表（lsof + ss 合并，避免漏掉子进程/不同视图）
get_listen_pids() {
    local pids=""
    if command -v lsof &>/dev/null; then
        pids=$(lsof -nP -iTCP:"${SGLANG_PORT}" -sTCP:LISTEN -t 2>/dev/null || true)
    fi
    if command -v ss &>/dev/null; then
        while IFS= read -r line; do
            for x in $(echo "$line" | grep -oE 'pid=[0-9]+' | cut -d= -f2); do
                pids="${pids}${x}"$'\n'
            done
        done < <(ss -tlnp 2>/dev/null | grep -E "$(_ss_port_pat)" || true)
    fi
    echo "$pids" | tr -s ' \n' '\n' | grep -E '^[0-9]+$' | sort -u
}

# Linux：杀掉所有占用该 TCP 端口的进程（SGLang 也可能存在父子/多 worker，单靠一个 PID 不够）
kill_by_fuser() {
    command -v fuser &>/dev/null || return 1
    fuser "${SGLANG_PORT}/tcp" &>/dev/null || return 1
    echo "  fuser: SIGTERM ${SGLANG_PORT}/tcp"
    fuser -TERM -k "${SGLANG_PORT}/tcp" 2>/dev/null || true
    return 0
}

echo "Stopping SGLang (port $SGLANG_PORT)..."

if ! port_in_use; then
    echo "No process found listening on port $SGLANG_PORT (SGLang may already be stopped)."
    exit 0
fi

kill_by_fuser || true

# 按 PID 再补一轮（无 fuser 或非 Linux 时主要依赖这段）
mapfile -t pids < <(get_listen_pids)
if [[ ${#pids[@]} -gt 0 ]]; then
    for pid in "${pids[@]}"; do
        [[ -z "$pid" ]] && continue
        if kill -0 "$pid" 2>/dev/null; then
            echo "  SIGTERM PID $pid"
            kill -TERM "$pid" 2>/dev/null || true
        fi
    done
fi

wait_count=0
while [[ $wait_count -lt 15 ]]; do
    if ! port_in_use; then
        echo "Port $SGLANG_PORT released."
        echo "Done."
        exit 0
    fi
    sleep 1
    wait_count=$((wait_count + 1))
done

echo "  Port still busy, fuser SIGKILL ${SGLANG_PORT}/tcp"
if command -v fuser &>/dev/null; then
    fuser -KILL -k "${SGLANG_PORT}/tcp" 2>/dev/null || true
fi

mapfile -t pids < <(get_listen_pids)
for pid in "${pids[@]}"; do
    [[ -z "$pid" ]] && continue
    if kill -0 "$pid" 2>/dev/null; then
        echo "  SIGKILL PID $pid"
        kill -KILL "$pid" 2>/dev/null || true
    fi
done

sleep 1
if port_in_use; then
    echo "WARNING: port $SGLANG_PORT may still be in use:" >&2
    ss -tlnp 2>/dev/null | grep -E ":${SGLANG_PORT}\s" || true
    lsof -nP -iTCP:"${SGLANG_PORT}" -sTCP:LISTEN 2>/dev/null || true
    exit 1
fi

echo "Done."

# 仅当脚本被 source 到当前 shell 时，才尝试退出 conda 环境；
# 若以 ./sglang_stop.sh 执行，这是一个子进程，conda deactivate 不会影响你的当前终端。
if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
    if command -v conda &>/dev/null; then
        if [[ "${CONDA_DEFAULT_ENV:-}" == "sglang" ]]; then
            conda deactivate || true
        fi
    fi
fi

exit 0
