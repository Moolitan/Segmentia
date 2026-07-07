#!/bin/bash
#  监控推理进程的 Locked/Pinned 内存及 Swap 行为
# 用法: ./monitor_v5.sh <PID>

if [ $# -lt 1 ]; then
    echo "Usage: $0 <PID>"
    exit 1
fi

PID=$1
OUTPUT="memory_pin_analysis.csv"
INTERVAL=1

# CSV 表头（重点关注 locked_kb, vmlck_kb, swap_pss_kb）
echo "timestamp,mem_available_kb,swap_used_kb,pswpin,pswpout,pgpgin,pgpgout,pid_majflt,swap_pss_kb,swap_kb,locked_kb,vmlck_kb,anon_kb,pss_anon_kb,pss_file_kb,private_dirty_kb" > "$OUTPUT"

trap "echo '监控停止, 数据写入 $OUTPUT'; exit 0" INT TERM

while true; do
    TS=$(date +%s.%N)

    # 系统指标
    MemAvail=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
    SwapTotal=$(awk '/SwapTotal/ {print $2}' /proc/meminfo)
    SwapFree=$(awk '/SwapFree/ {print $2}' /proc/meminfo)
    SwapUsed=$((SwapTotal - SwapFree))

    pswpin=$(awk '/pswpin/ {print $2}' /proc/vmstat)
    pswpout=$(awk '/pswpout/ {print $2}' /proc/vmstat)
    pgpgin=$(awk '/pgpgin/ {print $2}' /proc/vmstat)
    pgpgout=$(awk '/pgpgout/ {print $2}' /proc/vmstat)

    # 进程指标
    if [ -f "/proc/$PID/stat" ]; then
        maj_flt=$(awk '{print $12}' /proc/$PID/stat)
    else
        maj_flt="NA"
    fi

    if [ -f "/proc/$PID/status" ]; then
        VmLck=$(awk '/VmLck:/ {print $2}' /proc/$PID/status)
    else
        VmLck="NA"
    fi

    if [ -f "/proc/$PID/smaps_rollup" ]; then
        rollup=$(cat /proc/$PID/smaps_rollup)
        swap_pss=$(echo "$rollup" | awk '/SwapPss:/ {print $2}')
        swap=$(echo "$rollup" | awk '/^Swap:/ {print $2}')
        locked=$(echo "$rollup" | awk '/Locked:/ {print $2}')
        anon=$(echo "$rollup" | awk '/Anonymous:/ {print $2}')
        pss_anon=$(echo "$rollup" | awk '/Pss_Anon:/ {print $2}')
        pss_file=$(echo "$rollup" | awk '/Pss_File:/ {print $2}')
        private_dirty=$(echo "$rollup" | awk '/Private_Dirty:/ {print $2}')
    else
        swap_pss="NA"; swap="NA"; locked="NA"; anon="NA"
        pss_anon="NA"; pss_file="NA"; private_dirty="NA"
    fi

    echo "$TS,$MemAvail,$SwapUsed,$pswpin,$pswpout,$pgpgin,$pgpgout,$maj_flt,$swap_pss,$swap,$locked,$VmLck,$anon,$pss_anon,$pss_file,$private_dirty" >> "$OUTPUT"

    sleep "$INTERVAL"
done