#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

unset LD_LIBRARY_PATH
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
unset ALL_PROXY all_proxy ftp_proxy FTP_PROXY

# =============================================================================
# 脚本路径
# =============================================================================

VLLM_START_SCRIPT="$SCRIPT_DIR/wtz_vllm_start.sh"
VLLM_STOP_SCRIPT="$SCRIPT_DIR/wtz_vllm_stop.sh"
CGROUP_MONITOR_SCRIPT="$SCRIPT_DIR/monitor_cgroup_v2.sh"
RUN_MULTURN="$SCRIPT_DIR/03_14B_anthropic/run_multurn3.py"

for required_file in \
    "$VLLM_START_SCRIPT" \
    "$VLLM_STOP_SCRIPT" \
    "$CGROUP_MONITOR_SCRIPT" \
    "$RUN_MULTURN"
do
    if [[ ! -f "$required_file" ]]; then
        echo "[ERROR] 文件不存在: $required_file" >&2
        exit 1
    fi
done

# =============================================================================
# vLLM 配置
# =============================================================================

VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.9}"
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-0}"

VLLM_UNIT="${VLLM_UNIT:-vllm-qwen14b.service}"
VLLM_UNIT_NAME="${VLLM_UNIT%.service}"

export VLLM_PORT
export VLLM_API_KEY
export VLLM_MAX_MODEL_LEN
export VLLM_GPU_UTIL
export VLLM_ENFORCE_EAGER
export VLLM_UNIT

# =============================================================================
# Benchmark 配置
# =============================================================================

BENCH_ROOT="${BENCH_ROOT:-$ROOT/anthropic_skill_benchmark}"
WRL="${WRL:-03_14B_anthropic_3}"

WS_PARENT="${MULTRUN_BENCH_WORKSPACE:-$ROOT/workspace/$WRL}"
SK_PARENT="${SK_PARENT:-$ROOT/skills}"
RES_PARENT="${MULTRUN_BENCH_RESULTS:-$ROOT/results/$WRL}"
LOG_PARENT="${MULTRUN_BENCH_LOG:-$ROOT/log/$WRL}"

SEQ_ARR=(
    doc_coauthoring_design_doc
)

# =============================================================================
# 时间和采样配置
# =============================================================================

VLLM_READY_MAX_ATTEMPTS="${VLLM_READY_MAX_ATTEMPTS:-180}"
VLLM_READY_INTERVAL="${VLLM_READY_INTERVAL:-2}"

# vLLM 就绪后的空闲观察时间。
STEADY_WAIT_SECONDS="${STEADY_WAIT_SECONDS:-120}"

# 其中有多少秒用于 perf 空闲基线采样。
PERF_IDLE_SECONDS="${PERF_IDLE_SECONDS:-60}"

# benchmark 结束后的继续观察时间。
POST_WAIT_SECONDS="${POST_WAIT_SECONDS:-60}"

# cgroup 内存监控周期，单位为秒。
MONITOR_INTERVAL="${MONITOR_INTERVAL:-1}"

# perf 输出周期，单位为毫秒。
PERF_INTERVAL_MS="${PERF_INTERVAL_MS:-1000}"

# 1：启用 perf；0：禁用 perf。
PERF_ENABLE="${PERF_ENABLE:-1}"

# =============================================================================
# 输出目录
# =============================================================================

RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
MEMORY_DIR="${MEMORY_DIR:-$ROOT/results/memory/$RUN_ID}"

PHASE_FILE="$MEMORY_DIR/phase"
PHASE_EVENTS="$MEMORY_DIR/phase_events.csv"
VLLM_LOG="$MEMORY_DIR/vllm.log"
VLLM_START_GATE="$MEMORY_DIR/vllm.start.gate"

PERF_IDLE_OUTPUT="$MEMORY_DIR/perf_idle.csv"
PERF_INFERENCE_OUTPUT="$MEMORY_DIR/perf_inference.csv"
PERF_TEST_OUTPUT="$MEMORY_DIR/perf_test.log"

mkdir -p \
    "$MEMORY_DIR" \
    "$WS_PARENT" \
    "$RES_PARENT" \
    "$LOG_PARENT"

# =============================================================================
# 全局状态
# =============================================================================

MONITOR_PID=""
PERF_IDLE_PID=""
PERF_INFERENCE_PID=""
SUDO_KEEPALIVE_PID=""

CLEANUP_RUNNING=0

failed_seqs=()

# =============================================================================
# 阶段记录
# =============================================================================

set_phase()
{
    local phase="$1"

    printf '%s\n' "$phase" > "$PHASE_FILE"

    printf '%s,%s\n' \
        "$(date --iso-8601=ns)" \
        "$phase" \
        >> "$PHASE_EVENTS"

    echo "[phase] $phase"
}

# =============================================================================
# sudo 凭据保持
# =============================================================================

start_sudo_keepalive()
{
    sudo -v

    (
        while true; do
            sudo -n -v 2>/dev/null || exit 0
            sleep 30
        done
    ) &

    SUDO_KEEPALIVE_PID=$!
}

stop_sudo_keepalive()
{
    if [[ -n "${SUDO_KEEPALIVE_PID:-}" ]] &&
       kill -0 "$SUDO_KEEPALIVE_PID" 2>/dev/null; then

        kill "$SUDO_KEEPALIVE_PID" 2>/dev/null || true
        wait "$SUDO_KEEPALIVE_PID" 2>/dev/null || true
    fi

    SUDO_KEEPALIVE_PID=""
}

# =============================================================================
# vLLM 健康检查
# =============================================================================

vllm_probe_ok()
{
    local code_models
    local code_health

    code_models="$(
        curl \
            -sS \
            -o /dev/null \
            -w '%{http_code}' \
            --connect-timeout 5 \
            --max-time 15 \
            -H "Authorization: Bearer ${VLLM_API_KEY}" \
            "http://127.0.0.1:${VLLM_PORT}/v1/models" \
            2>/dev/null || true
    )"

    [[ -n "$code_models" ]] || code_models="000"

    if [[ "$code_models" == "200" ||
          "$code_models" == "401" ||
          "$code_models" == "403" ]]; then
        return 0
    fi

    code_health="$(
        curl \
            -sS \
            -o /dev/null \
            -w '%{http_code}' \
            --connect-timeout 5 \
            --max-time 15 \
            "http://127.0.0.1:${VLLM_PORT}/health" \
            2>/dev/null || true
    )"

    [[ -n "$code_health" ]] || code_health="000"

    [[ "$code_health" == "200" ]]
}

wait_vllm_ready()
{
    local attempt=0

    if ! command -v curl >/dev/null 2>&1; then
        echo "[ERROR] 当前 PATH 中找不到 curl" >&2
        return 1
    fi

    echo "[vLLM] 等待服务就绪，端口 $VLLM_PORT"

    while (( attempt < VLLM_READY_MAX_ATTEMPTS )); do
        if vllm_probe_ok; then
            echo "[vLLM] 服务已就绪"
            return 0
        fi

        attempt=$((attempt + 1))

        if (( attempt % 15 == 0 )); then
            echo "[vLLM] 等待中: $attempt/$VLLM_READY_MAX_ATTEMPTS"
        fi

        if ! systemctl is-active --quiet "$VLLM_UNIT"; then
            echo "[ERROR] $VLLM_UNIT 已退出" >&2
            tail -n 100 "$VLLM_LOG" 2>/dev/null || true
            return 1
        fi

        sleep "$VLLM_READY_INTERVAL"
    done

    echo "[ERROR] vLLM 就绪超时，日志: $VLLM_LOG" >&2
    return 1
}

# =============================================================================
# cgroup 查询
# =============================================================================

get_cgroup_path()
{
    systemctl show \
        --property=ControlGroup \
        --value \
        "$VLLM_UNIT" \
        2>/dev/null || true
}

get_perf_cgroup_name()
{
    local cgroup_path

    cgroup_path="$(get_cgroup_path)"
    cgroup_path="${cgroup_path#/}"

    printf '%s\n' "$cgroup_path"
}

wait_cgroup_created()
{
    local attempt
    local cgroup_path

    for attempt in $(seq 1 100); do
        cgroup_path="$(get_cgroup_path)"

        if [[ -n "$cgroup_path" &&
              -d "/sys/fs/cgroup${cgroup_path}" ]]; then

            printf '%s\n' "$cgroup_path"
            return 0
        fi

        sleep 0.1
    done

    return 1
}

# =============================================================================
# perf 访问事件监控
# =============================================================================

refresh_sudo()
{
    # 优先使用已有凭据；无有效凭据时，在当前前台终端提示输入密码。
    if ! sudo -n true 2>/dev/null; then
        sudo -v
    fi
}

check_perf_support()
{
    local perf_cgroup

    if [[ "$PERF_ENABLE" != "1" ]]; then
        echo "[perf] 已禁用"
        return 0
    fi

    if ! command -v perf >/dev/null 2>&1; then
        echo "[ERROR] 找不到 perf，请安装对应内核的 linux-tools" >&2
        return 1
    fi

    perf_cgroup="$(get_perf_cgroup_name)"

    if [[ -z "$perf_cgroup" ]]; then
        echo "[ERROR] 无法取得 perf cgroup 名称" >&2
        return 1
    fi

    refresh_sudo

    echo "[perf] 测试 cgroup 事件采集"
    echo "[perf] cgroup: $perf_cgroup"

    : > "$PERF_TEST_OUTPUT"

    if ! sudo -n perf stat \
        -a \
        --no-big-num \
        -e '{cycles,instructions,cache-references,cache-misses}' \
        -G "$perf_cgroup" \
        -- sleep 2 \
        > /dev/null \
        2> "$PERF_TEST_OUTPUT"
    then
        echo "[ERROR] perf cgroup 测试命令失败：" >&2
        cat "$PERF_TEST_OUTPUT" >&2 || true
        return 1
    fi

    if grep -qiE \
        'not supported|not counted|permission denied|no permission|failed|error:' \
        "$PERF_TEST_OUTPUT"
    then
        echo "[ERROR] perf 事件不可用：" >&2
        cat "$PERF_TEST_OUTPUT" >&2 || true
        return 1
    fi

    for event in \
        cycles \
        instructions \
        cache-references \
        cache-misses
    do
        if ! grep -q "$event" "$PERF_TEST_OUTPUT"; then
            echo "[ERROR] perf 测试结果缺少事件: $event" >&2
            cat "$PERF_TEST_OUTPUT" >&2 || true
            return 1
        fi
    done

    echo "[perf] cgroup 事件采集测试成功"
}

start_perf_recording()
{
    local pid_variable="$1"
    local output_file="$2"
    local duration="$3"

    local perf_cgroup
    local pid_file
    local perf_pid=""
    local attempt

    perf_cgroup="$(get_perf_cgroup_name)"
    pid_file="${output_file}.pid"

    if [[ -z "$perf_cgroup" ]]; then
        echo "[ERROR] 无法取得 perf cgroup 名称" >&2
        return 1
    fi

    refresh_sudo

    rm -f "$pid_file"
    : > "$output_file"

    echo "[perf] 开始采样"
    echo "[perf] cgroup:  $perf_cgroup"
    echo "[perf] duration: ${duration}s"
    echo "[perf] output:   $output_file"

    # 关键设计：
    # 1. sudo 命令在前台完成，不把 sudo 自身放到后台；
    # 2. root shell 在内部创建后台 perf；
    # 3. setsid shell 先把自己的 PID 写入文件；
    # 4. exec 后该 PID 直接成为 perf PID；
    # 5. perf 成为独立进程组组长，停止时可以使用负 PID 终止整组。
    sudo -n bash -c '
        set -euo pipefail

        pid_file="$1"
        output_file="$2"
        interval_ms="$3"
        perf_cgroup="$4"
        duration="$5"

        setsid bash -c '\''
            set -euo pipefail

            pid_file="$1"
            output_file="$2"
            interval_ms="$3"
            perf_cgroup="$4"
            duration="$5"

            printf "%s\n" "$$" > "$pid_file"

            exec perf stat \
                -a \
                --no-big-num \
                -I "$interval_ms" \
                -x, \
                -e "{cycles,instructions,cache-references,cache-misses}" \
                -G "$perf_cgroup" \
                -- sleep "$duration" \
                > /dev/null \
                2> "$output_file"
        '\'' perf-worker \
            "$pid_file" \
            "$output_file" \
            "$interval_ms" \
            "$perf_cgroup" \
            "$duration" \
            < /dev/null &

        # 等到子进程写出 PID，避免父 shell 提前退出。
        for _ in $(seq 1 50); do
            if [[ -s "$pid_file" ]]; then
                exit 0
            fi
            sleep 0.1
        done

        echo "perf PID file was not created" >&2
        exit 1
    ' perf-launcher \
        "$pid_file" \
        "$output_file" \
        "$PERF_INTERVAL_MS" \
        "$perf_cgroup" \
        "$duration"

    for attempt in $(seq 1 50); do
        if [[ -s "$pid_file" ]]; then
            perf_pid="$(cat "$pid_file" 2>/dev/null || true)"
            break
        fi
        sleep 0.1
    done

    if [[ ! "$perf_pid" =~ ^[0-9]+$ ]]; then
        echo "[ERROR] 未获得有效的 perf PID" >&2
        cat "$output_file" >&2 || true
        return 1
    fi

    printf -v "$pid_variable" '%s' "$perf_pid"

    # 等待第一轮间隔数据产生。
    sleep 2

    if ! sudo -n kill -0 "$perf_pid" 2>/dev/null; then
        echo "[ERROR] perf 进程提前退出，PID=$perf_pid" >&2
        cat "$output_file" >&2 || true
        rm -f "$pid_file"
        return 1
    fi

    if grep -qiE \
        'sudo:|not supported|not counted|permission denied|no permission|failed|error:' \
        "$output_file"
    then
        echo "[ERROR] perf 输出包含错误：" >&2
        cat "$output_file" >&2 || true
        stop_perf_process "$perf_pid" || true
        rm -f "$pid_file"
        return 1
    fi

    if ! grep -qE \
        'cycles|instructions|cache-references|cache-misses' \
        "$output_file"
    then
        echo "[ERROR] perf 未产生有效事件数据：" >&2
        cat "$output_file" >&2 || true
        stop_perf_process "$perf_pid" || true
        rm -f "$pid_file"
        return 1
    fi

    echo "[perf] PID: $perf_pid"
}

perf_process_alive()
{
    local pid="$1"

    [[ "$pid" =~ ^[0-9]+$ ]] || return 1

    sudo -n kill -0 "$pid" 2>/dev/null
}

stop_perf_process()
{
    local pid="$1"
    local attempt

    [[ -n "$pid" ]] || return 0
    [[ "$pid" =~ ^[0-9]+$ ]] || return 0

    refresh_sudo

    if ! perf_process_alive "$pid"; then
        return 0
    fi

    echo "[perf] 停止采样 PID $pid"

    # perf 是独立进程组的组长。
    # SIGINT 让 perf 正常刷新最后一批统计数据并退出。
    sudo -n kill -INT -- "-$pid" 2>/dev/null || true

    for attempt in $(seq 1 15); do
        if ! perf_process_alive "$pid"; then
            return 0
        fi
        sleep 1
    done

    echo "[perf] SIGINT 后仍未退出，发送 SIGTERM"

    sudo -n kill -TERM -- "-$pid" 2>/dev/null || true

    for attempt in $(seq 1 5); do
        if ! perf_process_alive "$pid"; then
            return 0
        fi
        sleep 1
    done

    echo "[perf] SIGTERM 后仍未退出，发送 SIGKILL"

    sudo -n kill -KILL -- "-$pid" 2>/dev/null || true
}

wait_perf_process()
{
    local pid="$1"

    [[ "$pid" =~ ^[0-9]+$ ]] || return 1

    while perf_process_alive "$pid"; do
        sleep 1
    done
}

run_idle_perf()
{
    local duration
    local remaining

    if [[ "$PERF_ENABLE" != "1" ]]; then
        sleep "$STEADY_WAIT_SECONDS"
        return 0
    fi

    duration="$PERF_IDLE_SECONDS"

    if (( duration > STEADY_WAIT_SECONDS )); then
        duration="$STEADY_WAIT_SECONDS"
    fi

    if (( duration <= 0 )); then
        sleep "$STEADY_WAIT_SECONDS"
        return 0
    fi

    start_perf_recording \
        PERF_IDLE_PID \
        "$PERF_IDLE_OUTPUT" \
        "$duration"

    echo "[perf] 等待空闲阶段采样完成，共 ${duration}s"

    wait_perf_process "$PERF_IDLE_PID"

    rm -f "${PERF_IDLE_OUTPUT}.pid"
    PERF_IDLE_PID=""

    if ! grep -q 'cycles' "$PERF_IDLE_OUTPUT"; then
        echo "[ERROR] 空闲阶段 perf 文件没有有效数据" >&2
        cat "$PERF_IDLE_OUTPUT" >&2 || true
        return 1
    fi

    echo "[perf] 空闲阶段采样完成"

    remaining=$((STEADY_WAIT_SECONDS - duration))

    if (( remaining > 0 )); then
        sleep "$remaining"
    fi
}

start_inference_perf()
{
    if [[ "$PERF_ENABLE" != "1" ]]; then
        return 0
    fi

    # 设置较长上限，benchmark 完成后主动发送 SIGINT。
    start_perf_recording \
        PERF_INFERENCE_PID \
        "$PERF_INFERENCE_OUTPUT" \
        86400
}

stop_inference_perf()
{
    if [[ "$PERF_ENABLE" != "1" ]]; then
        return 0
    fi

    if [[ -n "$PERF_INFERENCE_PID" ]]; then
        stop_perf_process "$PERF_INFERENCE_PID"

        rm -f "${PERF_INFERENCE_OUTPUT}.pid"

        if ! grep -q 'cycles' "$PERF_INFERENCE_OUTPUT"; then
            echo "[ERROR] 推理阶段 perf 文件没有有效数据" >&2
            cat "$PERF_INFERENCE_OUTPUT" >&2 || true
            return 1
        fi
    fi

    PERF_INFERENCE_PID=""

    echo "[perf] 推理阶段采样完成"
}



# =============================================================================
# vLLM cgroup 启动
# =============================================================================

start_vllm_cgroup()
{
    local cgroup_path

    if [[ -z "${CONDA_PREFIX:-}" ]]; then
        echo "[ERROR] CONDA_PREFIX 未设置，请先激活 vLLM 环境" >&2
        exit 1
    fi

    VLLM_UNIT="$VLLM_UNIT" \
    VLLM_PORT="$VLLM_PORT" \
        bash "$VLLM_STOP_SCRIPT" || true

    rm -f "$VLLM_START_GATE"

    set_phase "CGROUP_CREATING"

    sudo systemd-run \
        --unit="$VLLM_UNIT_NAME" \
        --description="Qwen3-14B vLLM inference engine" \
        --service-type=exec \
        --property=MemoryAccounting=yes \
        --property=CPUAccounting=yes \
        --property=TasksAccounting=yes \
        --property=KillMode=control-group \
        --property=TimeoutStopSec=90s \
        --property="StandardOutput=append:$VLLM_LOG" \
        --property="StandardError=append:$VLLM_LOG" \
        --setenv="CONDA_PREFIX=$CONDA_PREFIX" \
        --setenv="PATH=$PATH" \
        --setenv="VLLM_PORT=$VLLM_PORT" \
        --setenv="VLLM_API_KEY=$VLLM_API_KEY" \
        --setenv="VLLM_MAX_MODEL_LEN=$VLLM_MAX_MODEL_LEN" \
        --setenv="VLLM_GPU_UTIL=$VLLM_GPU_UTIL" \
        --setenv="VLLM_ENFORCE_EAGER=$VLLM_ENFORCE_EAGER" \
        --setenv="VLLM_START_GATE=$VLLM_START_GATE" \
        /usr/bin/bash "$VLLM_START_SCRIPT"

    if ! cgroup_path="$(wait_cgroup_created)"; then
        echo "[ERROR] vLLM cgroup 创建失败" >&2
        exit 1
    fi

    echo "[vLLM] cgroup: /sys/fs/cgroup${cgroup_path}"

    set_phase "WAITING_START_GATE"

    INTERVAL="$MONITOR_INTERVAL" \
        bash "$CGROUP_MONITOR_SCRIPT" \
        "$VLLM_UNIT" \
        "$PHASE_FILE" \
        "$MEMORY_DIR" &

    MONITOR_PID=$!

    printf '%s\n' "$MONITOR_PID" > "$MEMORY_DIR/monitor.pid"

    sleep 2

    set_phase "MODEL_LOADING"

    touch "$VLLM_START_GATE"
}

# =============================================================================
# 监控停止与最终状态保存
# =============================================================================

stop_monitor()
{
    if [[ -n "${MONITOR_PID:-}" ]] &&
       kill -0 "$MONITOR_PID" 2>/dev/null; then

        kill -TERM "$MONITOR_PID" 2>/dev/null || true
        wait "$MONITOR_PID" 2>/dev/null || true
    fi

    MONITOR_PID=""

    rm -f "$MEMORY_DIR/monitor.pid"
}

save_final_cgroup_state()
{
    local cgroup_path
    local cgroup_dir

    cgroup_path="$(get_cgroup_path)"

    [[ -n "$cgroup_path" ]] || return 0

    cgroup_dir="/sys/fs/cgroup${cgroup_path}"

    [[ -d "$cgroup_dir" ]] || return 0

    cat "$cgroup_dir/memory.current" \
        > "$MEMORY_DIR/final_memory_current.txt" \
        2>/dev/null || true

    cat "$cgroup_dir/memory.peak" \
        > "$MEMORY_DIR/final_memory_peak.txt" \
        2>/dev/null || true

    cat "$cgroup_dir/memory.swap.current" \
        > "$MEMORY_DIR/final_memory_swap_current.txt" \
        2>/dev/null || true

    cat "$cgroup_dir/memory.stat" \
        > "$MEMORY_DIR/final_memory_stat.txt" \
        2>/dev/null || true

    cat "$cgroup_dir/memory.events" \
        > "$MEMORY_DIR/final_memory_events.txt" \
        2>/dev/null || true

    cat "$cgroup_dir/memory.pressure" \
        > "$MEMORY_DIR/final_memory_pressure.txt" \
        2>/dev/null || true

    cat "$cgroup_dir/memory.numa_stat" \
        > "$MEMORY_DIR/final_memory_numa_stat.txt" \
        2>/dev/null || true

    cat "$cgroup_dir/cgroup.procs" \
        > "$MEMORY_DIR/final_cgroup_procs.txt" \
        2>/dev/null || true
}

stop_vllm_cgroup()
{
    stop_inference_perf || true

    if systemctl is-active --quiet "$VLLM_UNIT"; then
        set_phase "STOPPING"

        save_final_cgroup_state

        # 必须先停止监控，再删除 vLLM cgroup。
        stop_monitor

        VLLM_UNIT="$VLLM_UNIT" \
        VLLM_PORT="$VLLM_PORT" \
            bash "$VLLM_STOP_SCRIPT" || true
    else
        stop_monitor
    fi

    sudo systemctl reset-failed "$VLLM_UNIT" 2>/dev/null || true

    set_phase "STOPPED"
}

# =============================================================================
# Benchmark
# =============================================================================

run_all()
{
    local n_seqs="${#SEQ_ARR[@]}"
    local seq_idx=0

    local repo
    local seq_ws
    local seq_res
    local seq_log
    local bench_dir
    local seq_skills
    local exit_code

    for repo in "${SEQ_ARR[@]}"; do
        seq_idx=$((seq_idx + 1))

        seq_ws="$WS_PARENT/$repo"
        seq_res="$RES_PARENT/$repo"
        seq_log="$LOG_PARENT/$repo"
        bench_dir="$BENCH_ROOT/$repo"
        seq_skills="$seq_ws/.agents/skills"

        mkdir -p \
            "$seq_ws" \
            "$seq_res" \
            "$seq_res/figures" \
            "$seq_log"

        if [[ -d "$bench_dir/seed_files" ]]; then
            rm -rf "$seq_ws/seed_files"
            cp -a "$bench_dir/seed_files" "$seq_ws/"

            echo "[bench] 已同步 seed_files: $repo"
        fi

        mkdir -p "$seq_ws/.agents"

        rm -rf "$seq_skills"
        cp -a "$SK_PARENT"/. "$seq_skills"/

        echo
        echo "========== run_multurn: $repo ($seq_idx/$n_seqs) =========="
        echo "workspace: $seq_ws"
        echo "结果目录:  $seq_res"
        echo "日志目录:  $seq_log"

        set_phase "INFERENCE_${repo}"

        if python "$RUN_MULTURN" \
            --benchmark-repo "$repo" \
            --bench-root "$BENCH_ROOT" \
            --vllm-port "$VLLM_PORT" \
            --workspace "$seq_ws" \
            --output "$seq_res/multiturn_sequence_traces.json" \
            --log-dir "$seq_log" \
            "$@"
        then
            echo "========== $repo 完成 =========="
        else
            exit_code=$?

            echo \
"========== [ERROR] $repo 失败(exit=$exit_code)，继续下一个 =========="

            failed_seqs+=("$repo")
        fi
    done
}

# =============================================================================
# 清理
# =============================================================================

cleanup()
{
    local exit_code=$?

    if (( CLEANUP_RUNNING == 1 )); then
        exit "$exit_code"
    fi

    CLEANUP_RUNNING=1

    trap - EXIT INT TERM

    stop_perf_process "$PERF_IDLE_PID" || true
    stop_inference_perf || true
    stop_vllm_cgroup || true
    stop_sudo_keepalive || true

    exit "$exit_code"
}

# =============================================================================
# 实验元数据
# =============================================================================

echo "timestamp,phase" > "$PHASE_EVENTS"

cat > "$MEMORY_DIR/experiment.env" <<EOF
RUN_ID=$RUN_ID

VLLM_UNIT=$VLLM_UNIT
VLLM_PORT=$VLLM_PORT
VLLM_MAX_MODEL_LEN=$VLLM_MAX_MODEL_LEN
VLLM_GPU_UTIL=$VLLM_GPU_UTIL
VLLM_ENFORCE_EAGER=$VLLM_ENFORCE_EAGER

STEADY_WAIT_SECONDS=$STEADY_WAIT_SECONDS
POST_WAIT_SECONDS=$POST_WAIT_SECONDS
MONITOR_INTERVAL=$MONITOR_INTERVAL

PERF_ENABLE=$PERF_ENABLE
PERF_IDLE_SECONDS=$PERF_IDLE_SECONDS
PERF_INTERVAL_MS=$PERF_INTERVAL_MS

BENCH_ROOT=$BENCH_ROOT
WS_PARENT=$WS_PARENT
SK_PARENT=$SK_PARENT
RES_PARENT=$RES_PARENT
LOG_PARENT=$LOG_PARENT

CONDA_PREFIX=${CONDA_PREFIX:-}
EOF

# =============================================================================
# 主流程
# =============================================================================

trap cleanup EXIT INT TERM

start_sudo_keepalive

set_phase "PRE_START"

echo "[vLLM] 创建独立 cgroup"
start_vllm_cgroup

wait_vllm_ready

set_phase "READY"

check_perf_support

echo "[vLLM] 就绪后继续采样 ${STEADY_WAIT_SECONDS}s"
set_phase "IDLE_STEADY_WAIT"

run_idle_perf

set_phase "IDLE_STEADY"

echo "[perf] 开始记录多轮推理期间的访问事件"
start_inference_perf

run_all "$@"

stop_inference_perf

set_phase "POST_INFERENCE"

echo "[bench] 推理完成后继续采样 ${POST_WAIT_SECONDS}s"
sleep "$POST_WAIT_SECONDS"

stop_vllm_cgroup
stop_sudo_keepalive

trap - EXIT INT TERM

# =============================================================================
# 结果输出
# =============================================================================

echo
echo "========== benchmark 全部完成 =========="

total_seqs="${#SEQ_ARR[@]}"

if (( ${#failed_seqs[@]} > 0 )); then
    echo \
"失败序列 (${#failed_seqs[@]}/$total_seqs): ${failed_seqs[*]}"
else
    echo "全部 $total_seqs 个序列成功"
fi

echo
echo "内存统计目录: $MEMORY_DIR"
echo "cgroup 内存:   $MEMORY_DIR/vllm_cgroup_memory.csv"
echo "进程统计:      $MEMORY_DIR/vllm_cgroup_processes.csv"
echo "GPU 统计:      $MEMORY_DIR/vllm_gpu_memory.csv"

if [[ "$PERF_ENABLE" == "1" ]]; then
    echo "空闲 perf:     $PERF_IDLE_OUTPUT"
    echo "推理 perf:     $PERF_INFERENCE_OUTPUT"
fi

echo "vLLM 日志:     $VLLM_LOG"
echo "benchmark 结果: $RES_PARENT"
echo "benchmark 日志: $LOG_PARENT"