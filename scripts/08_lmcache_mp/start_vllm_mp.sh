#!/usr/bin/env bash
# 第 2 步：起一个单卡 vLLM 实例，KV connector 指向 LMCache MP server
# （kv_connector: LMCacheMPConnector），而不是进程内嵌 LMCache。
# 要先跑 start_lmcache_server.sh 把 server 启起来，这个脚本才连得上
# （默认连 tcp://localhost:5555，两边都没手动指定端口，用的是同一个默认值）。
# 用前先 `conda activate opencode`。
set -euo pipefail

MODEL_PATH="${VLLM_MODEL_PATH:-/mnt/Large_Language_Model_Lab_1/llm_models/Qwen3-14B/Qwen/Qwen3-14B}"
SERVED_NAME="${VLLM_SERVED_NAME:-Qwen3}"
PORT="${VLLM_PORT:-8100}"

vllm serve "$MODEL_PATH" \
    --served-model-name "$SERVED_NAME" \
    --kv-transfer-config '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both"}' \
    --disable-hybrid-kv-cache-manager \
    --gpu-memory-utilization 0.85 \
    --no-enable-prefix-caching \
    --enforce-eager \
    --port "$PORT"
