#!/usr/bin/env bash
# Starts vLLM with LMCache's CacheBlend enabled (LMCacheConnectorV1,
# enable_blending=True), then drives run_agent.py for each benchmark task.
# Independent of scripts/07_cskcache/run_real_agent.sh -- no shared code,
# just the same overall shape (start server, wait ready, run task, restart).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="${VLLM_MODEL_PATH:-/mnt/Large_Language_Model_Lab_1/llm_models/Qwen3-14B/Qwen/Qwen3-14B}"
SERVED_MODEL="${VLLM_SERVED_NAME:-Qwen3}"
PORT="${VLLM_PORT:-8100}"
API_KEY="${VLLM_API_KEY:-EMPTY}"
GPU_UTIL="${VLLM_GPU_UTIL:-0.85}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
# Raw run artifacts (workspace, per-request JSON, agent/vLLM logs, answers.txt)
# go to the big mnt disk, same convention as 07_cskcache -- not into the git
# repo's results/ dir. results/08_lmcache_mp/ is reserved for final plots and
# CSV summaries only, produced later by separate analysis scripts.
RUN_DIR="${LMCACHE_AGENT_RUN_DIR:-/mnt/Large_Language_Model_Lab_1/wsh/Segmentia/output/08_lmcache_mp/agent_skill_reuse_runs/$RUN_ID}"
TASKS="${TASKS:-doc_coauthoring_design_doc}"
READY_ATTEMPTS="${VLLM_READY_MAX_ATTEMPTS:-450}"
READY_INTERVAL="${VLLM_READY_INTERVAL:-2}"

OVERWRITE=0
DRY_RUN=0
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --overwrite) OVERWRITE=1 ;;
    --dry-run) DRY_RUN=1 ;;
    *) echo "usage: $0 [--overwrite] [--dry-run]" >&2; exit 2 ;;
  esac
  shift
done

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy || true
export VLLM_API_KEY="$API_KEY"
export VLLM_SERVED_NAME="$SERVED_MODEL"
export OPENHANDS_SUPPRESS_BANNER=1
export LITELLM_LOCAL_MODEL_COST_MAP=True
# Fixes the seed vLLM/LMCache use for content hashing (NONE_HASH). Without
# this, each vLLM restart gets a random seed and the same skill text hashes
# to a different chunk_hash every time -- silently breaking reuse across
# restarts (the disk cache never gets hit). Official vLLM/LMCache production
# guidance (e.g. LMCache's k8s operator sets this by default).
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"

# LMCache CacheBlend config (same recipe validated in
# scripts/08_lmcache_mp's blend_skill_reuse.py smoke test).
export LMCACHE_CHUNK_SIZE="${LMCACHE_CHUNK_SIZE:-256}"
export LMCACHE_ENABLE_BLENDING="${LMCACHE_ENABLE_BLENDING:-True}"
# A plain-text separator like " # # " isn't safe: BPE retokenizes the whole
# rendered prompt server-side, so the same literal substring can tokenize
# differently depending on what surrounds it -- verified empirically to
# sometimes vanish entirely, corrupting LMCache's chunk boundaries (crashes
# with a divide-by-zero on a zero-length chunk). Two *different* real special
# tokens back-to-back tokenize atomically regardless of context (special
# tokens are matched before BPE merging), so this is stable no matter what
# text surrounds it. (Deliberately two different tokens, not one repeated --
# LMCache's own `tokenizer.encode(sep)[1:]` drops the first encoded token to
# strip a BOS that Qwen's tokenizer doesn't actually add for a bare special
# token, so a single special token alone would be sliced down to nothing.)
export LMCACHE_BLEND_SPECIAL_STR="${LMCACHE_BLEND_SPECIAL_STR:-<|fim_pad|><|repo_name|>}"
export LMCACHE_USE_LAYERWISE="${LMCACHE_USE_LAYERWISE:-True}"
export LMCACHE_BLEND_CHECK_LAYERS="${LMCACHE_BLEND_CHECK_LAYERS:-1}"
export LMCACHE_BLEND_RECOMPUTE_RATIOS="${LMCACHE_BLEND_RECOMPUTE_RATIOS:-0.15}"

# L1 = CPU memory (fast, small). L2 = local disk (slower, bigger, and what
# actually persists data instead of just deleting it once L1's LRU evicts).
# Every stored chunk is written to both; CPU eviction only drops the L1
# copy, the L2 (disk) copy still answers later lookups.
export LMCACHE_LOCAL_CPU="${LMCACHE_LOCAL_CPU:-True}"
export LMCACHE_MAX_LOCAL_CPU_SIZE="${LMCACHE_MAX_LOCAL_CPU_SIZE:-5}"
LMCACHE_DISK_DIR="${LMCACHE_DISK_DIR:-/mnt/Large_Language_Model_Lab_1/wsh/lmcache_disk/agent_skill_reuse}"
mkdir -p "$LMCACHE_DISK_DIR"
export LMCACHE_LOCAL_DISK="file://${LMCACHE_DISK_DIR}/"
export LMCACHE_MAX_LOCAL_DISK_SIZE="${LMCACHE_MAX_LOCAL_DISK_SIZE:-50}"

KV_TRANSFER_CONFIG='{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
SERVER_PID=""

cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  SERVER_PID=""
}
trap cleanup EXIT INT TERM

wait_vllm_ready() {
  local attempt=0
  local code="000"
  while ((attempt < READY_ATTEMPTS)); do
    if [[ -n "$SERVER_PID" ]] && ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "[error] vLLM exited before readiness" >&2
      return 1
    fi
    code="$(curl -sS -o /dev/null -w '%{http_code}' \
      --connect-timeout 3 --max-time 10 \
      -H "Authorization: Bearer $API_KEY" \
      "http://127.0.0.1:$PORT/v1/models" 2>/dev/null || true)"
    if [[ "$code" == "200" ]]; then
      echo "[vLLM] ready port=$PORT"
      return 0
    fi
    attempt=$((attempt + 1))
    sleep "$READY_INTERVAL"
  done
  echo "[error] vLLM readiness timeout port=$PORT" >&2
  return 1
}

run_driver() {
  local task="$1"
  local args=(
    --benchmark-repo "$task"
    --bench-root "$ROOT/anthropic_skill_benchmark"
    --skills-dir "$ROOT/skills"
    --run-dir "$RUN_DIR"
    --model "$SERVED_MODEL"
    --vllm-port "$PORT"
  )
  [[ "$OVERWRITE" == "1" ]] && args+=(--overwrite)
  [[ "$DRY_RUN" == "1" ]] && args+=(--dry-run)
  # This python process is a pure HTTP client to vLLM (via OpenHands SDK) and
  # spawns tmux for TerminalTool -- it never touches CUDA/torch/lmcache, so it
  # doesn't need LD_LIBRARY_PATH pointing at the conda env's lib dir. Leaving
  # that set breaks /usr/bin/tmux: conda's older libtinfo.so.6 shadows the
  # system one and tmux fails to create a session.
  env -u LD_LIBRARY_PATH python "$SCRIPT_DIR/run_agent.py" "${args[@]}"
}

IFS=',' read -ra TASK_LIST <<< "$TASKS"
if [[ "$DRY_RUN" == "1" ]]; then
  for task in "${TASK_LIST[@]}"; do
    task="${task// /}"
    [[ -z "$task" ]] || run_driver "$task"
  done
  exit 0
fi

mkdir -p "$RUN_DIR"
failures=0
for task in "${TASK_LIST[@]}"; do
  task="${task// /}"
  [[ -z "$task" ]] && continue
  if [[ -f "$RUN_DIR/$task/_summary.json" && "$OVERWRITE" == "0" ]] &&
     grep -q '"status": "completed"' "$RUN_DIR/$task/_summary.json"; then
    echo "[skipped_existing] task=$task"
    continue
  fi

  cleanup
  echo "[vLLM] restart boundary=(mode=lmcache-blend, task=$task)"
  vllm serve "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL" \
    --api-key "$API_KEY" \
    --port "$PORT" \
    --dtype auto \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --no-enable-prefix-caching \
    --enforce-eager \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --reasoning-parser qwen3 \
    --no-enable-log-requests \
    --kv-transfer-config "$KV_TRANSFER_CONFIG" \
    >"$RUN_DIR/vllm_${task}.log" 2>&1 &
  SERVER_PID=$!

  if ! wait_vllm_ready; then
    failures=$((failures + 1))
    cleanup
    continue
  fi
  set +e
  run_driver "$task"
  status=$?
  set -e
  cleanup
  if [[ "$status" != "0" ]]; then
    failures=$((failures + 1))
    echo "[failed] task=$task status=$status"
  else
    echo "[completed] task=$task"
  fi
done

if [[ "$failures" != "0" ]]; then
  echo "$failures task(s) failed" >&2
  exit 1
fi
echo "[done] run_dir=$RUN_DIR"
