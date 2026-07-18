#!/usr/bin/env bash
# 第 3 步：冒烟测试 —— 同一个 prompt 发两遍。
# 第一遍是 miss，LMCache MP server 会把这段 KV 存起来；
# 第二遍应该命中缓存，总耗时应该明显变短。
# vllm serve 那边要开 --no-enable-prefix-caching，不然快是 vLLM 自己的
# prefix cache 在起作用，看不出到底是不是 LMCache 生效。
set -euo pipefail

HOST="${VLLM_HOST:-localhost}"
PORT="${VLLM_PORT:-8100}"
MODEL="${VLLM_SERVED_NAME:-Qwen3}"
URL="http://${HOST}:${PORT}/v1/chat/completions"

# 凑一段够长（覆盖默认 chunk-size=256 token）的重复上下文，保证真的会被
# LMCache 分块缓存，而不是随手几个词就被当成太短不值得缓存。
CONTEXT="You are reviewing the onboarding manual for a small robotics lab. The manual covers lab safety rules, equipment checkout procedures, how to reserve the GPU workstations, and the weekly reporting format that every intern must follow before Friday's stand-up meeting. "
PROMPT="$(printf '%s' "$CONTEXT""$CONTEXT""$CONTEXT""$CONTEXT""$CONTEXT""$CONTEXT")Question: based on the manual above, list the four topics it covers, in one short sentence each."

PAYLOAD=$(python3 -c '
import json, sys
print(json.dumps({
    "model": sys.argv[1],
    "messages": [{"role": "user", "content": sys.argv[2]}],
    "max_tokens": 100,
    "temperature": 0,
}))
' "$MODEL" "$PROMPT")

send_once() {
    echo "=== $1 ==="
    curl -s -X POST "$URL" -H "Content-Type: application/json" -d "$PAYLOAD" -w "\n[耗时] %{time_total}s\n"
    echo
}

send_once "第 1 次（预期 miss）"
send_once "第 2 次（预期命中 LMCache，应该明显更快）"
