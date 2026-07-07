#!/usr/bin/env bash
set -euo pipefail

if (( $# != 3 )); then
    echo "Usage: $0 <systemd-unit> <phase-file> <output-directory>" >&2
    exit 1
fi

UNIT="$1"
PHASE_FILE="$2"
OUTPUT_DIR="$3"

INTERVAL="${INTERVAL:-1}"

mkdir -p "$OUTPUT_DIR"

MEMORY_CSV="$OUTPUT_DIR/vllm_cgroup_memory.csv"
PROCESS_CSV="$OUTPUT_DIR/vllm_cgroup_processes.csv"
GPU_CSV="$OUTPUT_DIR/vllm_gpu_memory.csv"

CGROUP_PATH=$(
    systemctl show \
        --property=ControlGroup \
        --value \
        "$UNIT"
)

if [[ -z "$CGROUP_PATH" ]]; then
    echo "[ERROR] 无法取得 $UNIT 的 cgroup 路径" >&2
    exit 1
fi

CGROUP_DIR="/sys/fs/cgroup${CGROUP_PATH}"

if [[ ! -d "$CGROUP_DIR" ]]; then
    echo "[ERROR] cgroup 目录不存在: $CGROUP_DIR" >&2
    exit 1
fi

read_value()
{
    local file="$1"
    local default="${2:-0}"

    if [[ -r "$file" ]]; then
        cat "$file"
    else
        echo "$default"
    fi
}

read_named_value()
{
    local key="$1"
    local file="$2"

    awk -v key="$key" '
        $1 == key {
            print $2 + 0
            found = 1
            exit
        }
        END {
            if (!found)
                print 0
        }
    ' "$file" 2>/dev/null
}

read_psi_avg10()
{
    local type="$1"
    local file="$2"

    awk -v type="$type" '
        $1 == type {
            for (i = 2; i <= NF; i++) {
                split($i, pair, "=")
                if (pair[1] == "avg10") {
                    print pair[2]
                    found = 1
                    exit
                }
            }
        }
        END {
            if (!found)
                print 0
        }
    ' "$file" 2>/dev/null
}

read_meminfo()
{
    local key="$1"

    awk -v key="$key" '
        $1 == key ":" {
            print $2 + 0
            exit
        }
    ' /proc/meminfo
}

read_vmstat()
{
    local key="$1"

    awk -v key="$key" '
        $1 == key {
            print $2 + 0
            exit
        }
    ' /proc/vmstat
}

echo \
"timestamp_epoch,timestamp_iso,elapsed_seconds,phase,\
memory_current_bytes,memory_peak_bytes,memory_swap_current_bytes,\
anon_bytes,file_bytes,kernel_bytes,kernel_stack_bytes,pagetables_bytes,\
percpu_bytes,sock_bytes,shmem_bytes,file_mapped_bytes,file_dirty_bytes,\
file_writeback_bytes,slab_bytes,slab_reclaimable_bytes,slab_unreclaimable_bytes,\
pgfault_total,pgmajfault_total,pgfault_delta,pgmajfault_delta,\
workingset_refault_anon,workingset_refault_file,\
memory_events_low,memory_events_high,memory_events_max,\
memory_events_oom,memory_events_oom_kill,\
psi_some_avg10,psi_full_avg10,process_count,\
mem_available_kb,swap_used_kb,\
system_pswpin_total,system_pswpout_total,\
system_pswpin_delta,system_pswpout_delta" \
> "$MEMORY_CSV"

echo \
"timestamp_epoch,timestamp_iso,elapsed_seconds,phase,\
pid,ppid,state,rss_kb,vmlck_kb,command" \
> "$PROCESS_CSV"

echo \
"timestamp_epoch,timestamp_iso,elapsed_seconds,phase,\
gpu_index,gpu_memory_used_mib,gpu_memory_total_mib,\
gpu_utilization_percent,pid,process_name,process_gpu_memory_mib" \
> "$GPU_CSV"

START_NS=$(date +%s%N)

previous_pgfault=0
previous_pgmajfault=0
previous_pswpin=0
previous_pswpout=0
first_sample=1

trap 'exit 0' INT TERM

while [[ -d "$CGROUP_DIR" ]]; do
    NOW_NS=$(date +%s%N)
    TIMESTAMP_EPOCH=$(date +%s.%N)
    TIMESTAMP_ISO=$(date --iso-8601=ns)

    ELAPSED_SECONDS=$(
        awk -v now="$NOW_NS" -v start="$START_NS" \
            'BEGIN {printf "%.6f", (now - start) / 1000000000}'
    )

    if [[ -r "$PHASE_FILE" ]]; then
        PHASE=$(tr ',\n' '__' < "$PHASE_FILE")
    else
        PHASE="UNKNOWN"
    fi

    MEMORY_CURRENT=$(read_value "$CGROUP_DIR/memory.current")
    MEMORY_PEAK=$(read_value "$CGROUP_DIR/memory.peak" 0)
    MEMORY_SWAP_CURRENT=$(read_value "$CGROUP_DIR/memory.swap.current" 0)

    MEMORY_STAT="$CGROUP_DIR/memory.stat"
    MEMORY_EVENTS="$CGROUP_DIR/memory.events"
    MEMORY_PRESSURE="$CGROUP_DIR/memory.pressure"

    ANON=$(read_named_value anon "$MEMORY_STAT")
    FILE=$(read_named_value file "$MEMORY_STAT")
    KERNEL=$(read_named_value kernel "$MEMORY_STAT")
    KERNEL_STACK=$(read_named_value kernel_stack "$MEMORY_STAT")
    PAGETABLES=$(read_named_value pagetables "$MEMORY_STAT")
    PERCPU=$(read_named_value percpu "$MEMORY_STAT")
    SOCK=$(read_named_value sock "$MEMORY_STAT")
    SHMEM=$(read_named_value shmem "$MEMORY_STAT")
    FILE_MAPPED=$(read_named_value file_mapped "$MEMORY_STAT")
    FILE_DIRTY=$(read_named_value file_dirty "$MEMORY_STAT")
    FILE_WRITEBACK=$(read_named_value file_writeback "$MEMORY_STAT")
    SLAB=$(read_named_value slab "$MEMORY_STAT")
    SLAB_RECLAIMABLE=$(read_named_value slab_reclaimable "$MEMORY_STAT")
    SLAB_UNRECLAIMABLE=$(read_named_value slab_unreclaimable "$MEMORY_STAT")

    PGFAULT=$(read_named_value pgfault "$MEMORY_STAT")
    PGMAJFAULT=$(read_named_value pgmajfault "$MEMORY_STAT")

    WORKINGSET_REFAULT_ANON=$(
        read_named_value workingset_refault_anon "$MEMORY_STAT"
    )

    WORKINGSET_REFAULT_FILE=$(
        read_named_value workingset_refault_file "$MEMORY_STAT"
    )

    EVENT_LOW=$(read_named_value low "$MEMORY_EVENTS")
    EVENT_HIGH=$(read_named_value high "$MEMORY_EVENTS")
    EVENT_MAX=$(read_named_value max "$MEMORY_EVENTS")
    EVENT_OOM=$(read_named_value oom "$MEMORY_EVENTS")
    EVENT_OOM_KILL=$(read_named_value oom_kill "$MEMORY_EVENTS")

    PSI_SOME=$(read_psi_avg10 some "$MEMORY_PRESSURE")
    PSI_FULL=$(read_psi_avg10 full "$MEMORY_PRESSURE")

    PROCESS_COUNT=$(
        wc -l < "$CGROUP_DIR/cgroup.procs" 2>/dev/null || echo 0
    )

    MEM_AVAILABLE=$(read_meminfo MemAvailable)
    SWAP_TOTAL=$(read_meminfo SwapTotal)
    SWAP_FREE=$(read_meminfo SwapFree)
    SWAP_USED=$((SWAP_TOTAL - SWAP_FREE))

    PSWPIN=$(read_vmstat pswpin)
    PSWPOUT=$(read_vmstat pswpout)

    if (( first_sample == 1 )); then
        PGFAULT_DELTA=0
        PGMAJFAULT_DELTA=0
        PSWPIN_DELTA=0
        PSWPOUT_DELTA=0
        first_sample=0
    else
        PGFAULT_DELTA=$((PGFAULT - previous_pgfault))
        PGMAJFAULT_DELTA=$((PGMAJFAULT - previous_pgmajfault))
        PSWPIN_DELTA=$((PSWPIN - previous_pswpin))
        PSWPOUT_DELTA=$((PSWPOUT - previous_pswpout))
    fi

    previous_pgfault=$PGFAULT
    previous_pgmajfault=$PGMAJFAULT
    previous_pswpin=$PSWPIN
    previous_pswpout=$PSWPOUT

    echo \
"$TIMESTAMP_EPOCH,$TIMESTAMP_ISO,$ELAPSED_SECONDS,$PHASE,\
$MEMORY_CURRENT,$MEMORY_PEAK,$MEMORY_SWAP_CURRENT,\
$ANON,$FILE,$KERNEL,$KERNEL_STACK,$PAGETABLES,\
$PERCPU,$SOCK,$SHMEM,$FILE_MAPPED,$FILE_DIRTY,\
$FILE_WRITEBACK,$SLAB,$SLAB_RECLAIMABLE,$SLAB_UNRECLAIMABLE,\
$PGFAULT,$PGMAJFAULT,$PGFAULT_DELTA,$PGMAJFAULT_DELTA,\
$WORKINGSET_REFAULT_ANON,$WORKINGSET_REFAULT_FILE,\
$EVENT_LOW,$EVENT_HIGH,$EVENT_MAX,\
$EVENT_OOM,$EVENT_OOM_KILL,\
$PSI_SOME,$PSI_FULL,$PROCESS_COUNT,\
$MEM_AVAILABLE,$SWAP_USED,\
$PSWPIN,$PSWPOUT,$PSWPIN_DELTA,$PSWPOUT_DELTA" \
    >> "$MEMORY_CSV"

    if [[ -r "$CGROUP_DIR/cgroup.procs" ]]; then
        while IFS= read -r PID; do
            [[ -r "/proc/$PID/status" ]] || continue

            PPID_VALUE=$(
                awk '$1 == "PPid:" {print $2}' "/proc/$PID/status"
            )

            STATE=$(
                awk '$1 == "State:" {print $2}' "/proc/$PID/status"
            )

            RSS_KB=$(
                awk '$1 == "VmRSS:" {print $2}' "/proc/$PID/status"
            )

            VMLCK_KB=$(
                awk '$1 == "VmLck:" {print $2}' "/proc/$PID/status"
            )

            COMMAND=$(
                tr '\0' ' ' < "/proc/$PID/cmdline" 2>/dev/null |
                    sed 's/,/;/g'
            )

            echo \
"$TIMESTAMP_EPOCH,$TIMESTAMP_ISO,$ELAPSED_SECONDS,$PHASE,\
$PID,${PPID_VALUE:-0},${STATE:-NA},${RSS_KB:-0},${VMLCK_KB:-0},\
\"$COMMAND\"" \
            >> "$PROCESS_CSV"

        done < "$CGROUP_DIR/cgroup.procs"
    fi

    if command -v nvidia-smi >/dev/null 2>&1; then
        declare -A GPU_PROCESS_MEMORY=()
        declare -A GPU_PROCESS_NAME=()

        while IFS=',' read -r PID PROCESS_NAME PROCESS_MEMORY; do
            PID=$(xargs <<< "$PID")
            PROCESS_NAME=$(xargs <<< "$PROCESS_NAME")
            PROCESS_MEMORY=$(xargs <<< "$PROCESS_MEMORY")

            [[ "$PID" =~ ^[0-9]+$ ]] || continue

            GPU_PROCESS_MEMORY["$PID"]="$PROCESS_MEMORY"
            GPU_PROCESS_NAME["$PID"]="$PROCESS_NAME"

        done < <(
            nvidia-smi \
                --query-compute-apps=pid,process_name,used_memory \
                --format=csv,noheader,nounits 2>/dev/null || true
        )

        while IFS=',' read -r GPU_INDEX GPU_USED GPU_TOTAL GPU_UTIL; do
            GPU_INDEX=$(xargs <<< "$GPU_INDEX")
            GPU_USED=$(xargs <<< "$GPU_USED")
            GPU_TOTAL=$(xargs <<< "$GPU_TOTAL")
            GPU_UTIL=$(xargs <<< "$GPU_UTIL")

            MATCHED=0

            while IFS= read -r PID; do
                if [[ -n "${GPU_PROCESS_MEMORY[$PID]+x}" ]]; then
                    echo \
"$TIMESTAMP_EPOCH,$TIMESTAMP_ISO,$ELAPSED_SECONDS,$PHASE,\
$GPU_INDEX,$GPU_USED,$GPU_TOTAL,$GPU_UTIL,\
$PID,\"${GPU_PROCESS_NAME[$PID]}\",${GPU_PROCESS_MEMORY[$PID]}" \
                    >> "$GPU_CSV"

                    MATCHED=1
                fi
            done < "$CGROUP_DIR/cgroup.procs"

            if (( MATCHED == 0 )); then
                echo \
"$TIMESTAMP_EPOCH,$TIMESTAMP_ISO,$ELAPSED_SECONDS,$PHASE,\
$GPU_INDEX,$GPU_USED,$GPU_TOTAL,$GPU_UTIL,NA,NA,0" \
                >> "$GPU_CSV"
            fi

        done < <(
            nvidia-smi \
                --query-gpu=index,memory.used,memory.total,utilization.gpu \
                --format=csv,noheader,nounits 2>/dev/null || true
        )

        unset GPU_PROCESS_MEMORY
        unset GPU_PROCESS_NAME
    fi

    sleep "$INTERVAL"
done
