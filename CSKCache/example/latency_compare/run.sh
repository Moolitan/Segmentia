#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
VLLM_ROOT="$ROOT/vllm"
LMCACHE_ROOT="$ROOT/LMCache"
CSKCACHE_ROOT="$ROOT/CSKCache"
PYTHON_BIN="${PYTHON_BIN:-python}"
VLLM_BIN="${VLLM_BIN:-vllm}"
PYTHONPATH_VALUE="$SCRIPT_DIR:$VLLM_ROOT:$LMCACHE_ROOT:$CSKCACHE_ROOT${PYTHONPATH:+:$PYTHONPATH}"

if (($#)); then
  echo "[error] this launcher accepts no arguments; edit config.py instead" >&2
  exit 2
fi
if [[ "${CONDA_DEFAULT_ENV:-}" != "opencode" ]]; then
  echo "[error] activate conda environment opencode first" >&2
  exit 2
fi

eval "$(
  CSKCACHE_LATENCY_EXPORT_CONFIG=1 PYTHONPATH="$PYTHONPATH_VALUE" \
    "$PYTHON_BIN" "$SCRIPT_DIR/run.py"
)"

RUNTIME_DIR="$(mktemp -d /tmp/cskcache-latency.XXXXXXXX)"
SERVER_LOG="$RUNTIME_DIR/vllm.log"
export CSKCACHE_PROFILE_TRACE_PATH="$RUNTIME_DIR/cskcache_profile.jsonl"
: > "$CSKCACHE_PROFILE_TRACE_PATH"
SERVER_PID=""

cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  if [[ "$RUNTIME_DIR" == /tmp/cskcache-latency.* ]]; then
    rm -rf -- "$RUNTIME_DIR"
  fi
}
trap cleanup EXIT INT TERM

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy || true
export PYTHONHASHSEED=0
export VLLM_SERVER_DEV_MODE=1
export VLLM_CSK_T0_PREFETCH=1
export LMCACHE_CHUNK_SIZE=256
export LMCACHE_USE_LAYERWISE=True
export LMCACHE_FORCE_SKIP_SAVE=1
export LMCACHE_LOCAL_CPU=True
export LMCACHE_MAX_LOCAL_CPU_SIZE="$LATENCY_LOCAL_CPU_GB"
export LMCACHE_MAX_LOCAL_DISK_SIZE="$LATENCY_LOCAL_DISK_GB"
export LMCACHE_EXTRA_CONFIG="$LATENCY_LMCACHE_EXTRA_CONFIG"
export CSKCACHE_PROFILE=1
export CSKCACHE_FINE_TIMELINE=0
export CSKCACHE_DISABLE_VISUALIZER=1

if [[ "$LATENCY_STORAGE_BACKEND" == "raw_block" ]]; then
  export LMCACHE_STORAGE_PLUGINS=raw_block
  unset LMCACHE_LOCAL_DISK || true
else
  export LMCACHE_STORAGE_PLUGINS=""
  export LMCACHE_LOCAL_DISK="$LATENCY_BACKEND_ROOT"
fi

PYTHONPATH="$PYTHONPATH_VALUE" "$VLLM_BIN" serve "$LATENCY_MODEL_PATH" \
  --served-model-name "$LATENCY_SERVED_MODEL" \
  --api-key "$LATENCY_API_KEY" \
  --port "$LATENCY_PORT" \
  --dtype auto \
  --max-model-len "$LATENCY_MAX_MODEL_LEN" \
  --gpu-memory-utilization "$LATENCY_GPU_UTIL" \
  --kv-transfer-config "$LATENCY_KV_TRANSFER_CONFIG" \
  --enforce-eager \
  --no-enable-log-requests \
  --enable-prefix-caching \
  --no-async-scheduling \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --reasoning-parser qwen3 \
  > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

ready=0
for _ in $(seq 1 450); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[error] vLLM exited before readiness" >&2
    tail -n 80 "$SERVER_LOG" >&2 || true
    exit 1
  fi
  code="$(curl -sS -o /dev/null -w '%{http_code}' \
    --connect-timeout 3 --max-time 10 \
    -H "Authorization: Bearer $LATENCY_API_KEY" \
    "http://127.0.0.1:$LATENCY_PORT/v1/models" 2>/dev/null || true)"
  if [[ "$code" == "200" ]]; then
    ready=1
    break
  fi
  sleep 2
done
if [[ "$ready" != "1" ]]; then
  echo "[error] vLLM readiness timeout" >&2
  tail -n 80 "$SERVER_LOG" >&2 || true
  exit 1
fi

if ! PYTHONPATH="$PYTHONPATH_VALUE" "$PYTHON_BIN" "$SCRIPT_DIR/run.py"; then
  tail -n 80 "$SERVER_LOG" >&2 || true
  exit 1
fi
