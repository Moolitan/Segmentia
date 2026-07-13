#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CSKCACHE_ROOT="$ROOT/CSKCache"
VLLM_ROOT="${VLLM_ROOT:-/home/wsh/vllm}"
MODEL_PATH="${VLLM_MODEL_PATH:-/mnt/Large_Language_Model_Lab_1/llm_models/Qwen3-14B/Qwen/Qwen3-14B}"
SERVED_MODEL="${VLLM_SERVED_NAME:-Qwen3}"
PORT="${VLLM_PORT:-8013}"
API_KEY="${VLLM_API_KEY:-EMPTY}"
GPU_UTIL="${VLLM_GPU_UTIL:-0.9}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
KV_DIR="${CSKCACHE_KV_DIR:-/mnt/Large_Language_Model_Lab_1/wsh/Segmentia/output/07_cskcache/offline_skill_kv}"
OUTPUT_DIR="${CSKCACHE_AGENT_OUTPUT_DIR:-/mnt/Large_Language_Model_Lab_1/wsh/Segmentia/output/07_cskcache/agent_trace_replay}"
LOG_DIR="${VLLM_LOG_DIR:-$ROOT/log}"
TASKS="${TASKS:-internal_comms_incident_update,doc_coauthoring_design_doc,mcp_server_and_spec,web_artifact_with_theme,launch_poster_page_pack,slack_launch_pack}"

OVERWRITE=0
if [[ "${1:-}" == "--overwrite" ]]; then
  OVERWRITE=1
  shift
fi
if [[ "$#" != "0" ]]; then
  echo "usage: $0 [--overwrite]" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy || true
export PYTHONPATH="$CSKCACHE_ROOT:$VLLM_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CSKCACHE_DISK_DIR="$KV_DIR"
# Avoid materializing the complete disk catalog at every task restart. Explicit
# reuse signals still load the requested entry lazily from the disk tier.
export CSKCACHE_CAPTURE_ONLY=1
unset CSKCACHE_CPU_MAX_BYTES || true
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

KV_TRANSFER_CONFIG='{"kv_connector":"CSKCacheConnectorV1","kv_connector_module_path":"cskcache.integration.vllm.v1_connector","kv_role":"kv_both"}'
SERVER_PID=""

cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  SERVER_PID=""
  VLLM_PORT="$PORT" bash "$ROOT/scripts/vllm_stop.sh" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

IFS=',' read -ra TASK_LIST <<< "$TASKS"
failures=0
for task in "${TASK_LIST[@]}"; do
  task="${task// /}"
  [[ -z "$task" ]] && continue
  output="$OUTPUT_DIR/$task.jsonl"
  if [[ -f "$output" && "$OVERWRITE" == "0" ]]; then
    echo "[skipped_completed] task=$task output=$output"
    continue
  fi

  cleanup
  log_file="$LOG_DIR/07_cskcache_trace_agent_${task}.log"
  echo "[vLLM] restart boundary=(mode=cskcache, task=$task) port=$PORT log=$log_file"
  cd "$VLLM_ROOT"
  python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL" \
    --api-key "$API_KEY" \
    --port "$PORT" \
    --dtype auto \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --enable-prefix-caching \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --reasoning-parser qwen3 \
    --no-enable-log-requests \
    --kv-transfer-config "$KV_TRANSFER_CONFIG" \
    >"$log_file" 2>&1 &
  SERVER_PID=$!

  cd "$ROOT"
  args=(
    --task "$task"
    --model-path "$MODEL_PATH"
    --model "$SERVED_MODEL"
    --base-url "http://127.0.0.1:$PORT"
    --api-key "$API_KEY"
    --kv-dir "$KV_DIR"
    --output "$output"
    --max-tokens "$MAX_TOKENS"
    --max-model-len "$MAX_MODEL_LEN"
  )
  if [[ "$OVERWRITE" == "1" ]]; then
    args+=(--overwrite)
  fi

  set +e
  python scripts/07_cskcache/replay_trace_agent.py "${args[@]}"
  status=$?
  set -e
  cleanup
  if [[ "$status" != "0" ]]; then
    failures=$((failures + 1))
    echo "[failed] task=$task status=$status"
  else
    echo "[completed] task=$task output=$output"
  fi
done

if [[ "$failures" != "0" ]]; then
  echo "$failures task(s) failed" >&2
  exit 1
fi
echo "[done] output_dir=$OUTPUT_DIR"
