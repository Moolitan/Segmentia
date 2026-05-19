# ContextSegmentKV Real Agent Demo

This directory runs `anthropic_skill_benchmark/slack_launch_pack` through the
same OpenHands agent path used by:

```text
scripts/run_multurn_bench_qwen_14B_anthropic.sh
scripts/03_14B_anthropic/run_multurn3.py
```

Main entrypoint:

```bash
bash scripts/05_context_segment_agent_kv/run_real_multurn_context_segment.sh
```

Flow:

1. Start vLLM from `/home/wsh/vllm` with:
   - `VLLM_CONTEXT_SEGMENT_KV_SAVE_DIR`
   - `VLLM_CONTEXT_SEGMENT_KV_DIR`
2. Copy `anthropic_skill_benchmark/slack_launch_pack/seed_files` into a fresh
   workspace.
3. Copy repository `skills/` into `workspace/.agents/skills`.
4. Offline prefill the expected context segments:
   - `internal-comms/SKILL.md`
   - `slack-gif-creator/SKILL.md`
   - `brand-guidelines/SKILL.md`
5. Create the real OpenHands Agent and Conversation via `run_multurn3.py`.
6. Run all `slack_launch_pack` turns.
7. Wrap the real `llm._transport_call`. When a real request contains one of the
   cached segment texts outside the system message, compute the exact token span
   with vLLM `/tokenize` and attach:

```json
{
  "vllm_xargs": {
    "context_segment_cache": "{\"targets\": [...]}"
  }
}
```

This means the online calls are produced by the actual agent loop, not by a
hand-written prompt.

Run:

```bash
bash scripts/05_context_segment_agent_kv/run_real_multurn_context_segment.sh
```

Output:

```text
results/05_context_segment_agent_kv/slack_launch_pack/multiturn_sequence_traces.json
results/05_context_segment_agent_kv/slack_launch_pack/kv_cache/<cache_id>.pt
log/05_context_segment_agent_kv/slack_launch_pack.log
```

The output JSON includes:

```text
offline_context_segments
context_segment_kv_events
llm_calls
```

`context_segment_kv_events` records which real agent requests received KV
injection and the `[target_start, target_end)` spans.

There is also a smaller low-level sanity-check script:

```text
run_context_segment_agent_demo.sh
```

That script is not the benchmark path; it exists only to debug the raw
offline/online request mechanism.
