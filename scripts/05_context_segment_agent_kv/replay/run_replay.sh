#!/usr/bin/env bash
# Replay saved agent traces (src/traces/) against a modified vLLM, driving
# skill KV reuse via ContextSegmentKV. No agent framework involved.
#
# Usage:
#   bash run_replay.sh                       # recompute + reuse, all tasks
#   REUSE_SCOPE=cross-task bash run_replay.sh # cross-task skill reuse
#   TASKS=internal_comms_incident_update,slack_launch_pack bash run_replay.sh
#   MODE=reuse bash run_replay.sh
#   MODE=recompute bash run_replay.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$ROOT"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy || true

export VLLM_PORT="${VLLM_PORT:-8000}"
export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
export VLLM_SERVED_NAME="${VLLM_SERVED_NAME:-Qwen3}"

TASKS="${TASKS:-all}"
MODE="${MODE:-all}"
REUSE_SCOPE="${REUSE_SCOPE:-per-task}"
OUTPUT="${OUTPUT:-$ROOT/results/05_context_segment_agent_kv/replay/replay_${REUSE_SCOPE}_${MODE}.json}"

# Start a fresh vLLM instance, clearing both the ContextSegmentKV registry and
# vLLM's own prefix (radix) cache.
start_vllm() {
  echo "[vLLM] restart (empty prefix cache + ContextSegmentKV registry)"
  unset VLLM_CONTEXT_SEGMENT_KV_SAVE_DIR
  unset VLLM_CONTEXT_SEGMENT_KV_DIR
  bash "$ROOT/scripts/vllm_stop.sh" || true
  bash "$ROOT/scripts/vllm_start.sh"
  for i in $(seq 1 180); do
    code="$(curl -sS -o /dev/null -w '%{http_code}' --connect-timeout 3 --max-time 10 \
      -H "Authorization: Bearer ${VLLM_API_KEY}" \
      "http://127.0.0.1:${VLLM_PORT}/v1/models" 2>/dev/null || true)"
    [[ "$code" == "200" ]] && { echo "[vLLM] ready"; break; }
    sleep 2
  done
}

run_mode() {
  local mode="$1"
  local output="$2"
  python "$SCRIPT_DIR/replay_trace_context_segment.py" \
    --tasks "$TASKS" \
    --mode "$mode" \
    --reuse-scope "$REUSE_SCOPE" \
    --vllm-port "$VLLM_PORT" \
    --model "$VLLM_SERVED_NAME" \
    --output "$output"
  echo "[done] result: $output"
}

if [[ "$MODE" == "all" ]]; then
  # Run recompute and reuse in separate vLLM instances so vLLM's prefix cache
  # does not carry over from recompute into reuse, which would mask injection.
  OUT_RECOMPUTE="$ROOT/results/05_context_segment_agent_kv/replay/replay_${REUSE_SCOPE}_recompute.json"
  OUT_REUSE="$ROOT/results/05_context_segment_agent_kv/replay/replay_${REUSE_SCOPE}_reuse.json"

  if [[ "${RESTART_VLLM:-1}" == "1" ]]; then start_vllm; fi
  run_mode recompute "$OUT_RECOMPUTE"

  start_vllm   # always restart before reuse to clear prefix cache
  run_mode reuse "$OUT_REUSE"

  # Merge the two single-mode result files into the combined output and
  # recompute the summary that requires both recompute and reuse runs.
  export _MERGE_F_R="$OUT_RECOMPUTE"
  export _MERGE_F_U="$OUT_REUSE"
  export _MERGE_OUT="$OUTPUT"
  python - <<'PYEOF'
import json, os

d_r = json.loads(open(os.environ["_MERGE_F_R"]).read())
d_u = json.loads(open(os.environ["_MERGE_F_U"]).read())
runs_r = [r for r in d_r.get("runs", []) if r["mode"] == "recompute"]
runs_u = [r for r in d_u.get("runs", []) if r["mode"] == "reuse"]
all_runs = runs_r + runs_u

by_task_mode = {(r["task"], r["mode"]): r for r in all_runs}
tasks = [r["task"] for r in runs_r]
summary = []
for task in tasks:
    rc = by_task_mode.get((task, "recompute"))
    ru = by_task_mode.get((task, "reuse"))
    if not rc or not ru:
        continue
    delta = ru["total_latency_s"] - rc["total_latency_s"]
    speedup = rc["total_latency_s"] / ru["total_latency_s"] if ru["total_latency_s"] > 0 else None
    summary.append({
        "task": task,
        "recompute_latency_s": rc["total_latency_s"],
        "recompute_prompt_tokens": rc["total_prompt_tokens"],
        "reuse_latency_s": ru["total_latency_s"],
        "reuse_prompt_tokens": ru["total_prompt_tokens"],
        "segkv_source_tokens": ru["segkv_source_tokens"],
        "segkv_target_tokens": ru["segkv_target_tokens"],
        "latency_delta_s": round(delta, 4),
        "latency_speedup": round(speedup, 3) if speedup else None,
    })

merged = {**d_r, "mode": "all", "runs": all_runs, "summary": summary}
out = os.environ["_MERGE_OUT"]
open(out, "w").write(json.dumps(merged, indent=2, ensure_ascii=False))
print(f"[done] merged: {out}")
PYEOF
else
  if [[ "${RESTART_VLLM:-1}" == "1" ]]; then start_vllm; fi
  run_mode "$MODE" "$OUTPUT"
fi
