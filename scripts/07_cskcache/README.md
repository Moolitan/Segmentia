# CSKCache offline skill prefill

This directory builds context-free CSKCache entries from `skills/*/SKILL.md`.
Each Markdown file is tokenized locally with `add_special_tokens=False`, and the
resulting token ID list is sent directly to `/v1/completions`. No chat template,
role marker, tool wrapper, BOS, or EOS is intentionally added. The saved source
span is always `[0, len(skill_token_ids))`; the one generated completion token is
outside that span and is not saved.

Large KV payloads default to:

```text
/mnt/Large_Language_Model_Lab_1/wsh/Segmentia/output/07_cskcache/offline_skill_kv/
```

The wrapper processes skills in sorted order. It starts a fresh CSKCache-enabled
vLLM server for one skill, saves that skill, fully stops the server, and only
then starts the next one. Prefix caching is also disabled. This makes the server
lifecycle and all runtime KV state isolated at the individual-skill boundary:

```bash
cd /home/wsh/openhands_code_research
conda activate opencode
bash scripts/07_cskcache/run_offline_prefill_skills.sh
```

Existing cache IDs are skipped. To replace them:

```bash
bash scripts/07_cskcache/run_offline_prefill_skills.sh --overwrite
```

A tokenizer-only check does not start vLLM or write KV tensors:

```bash
PYTHONPATH=/home/wsh/openhands_code_research/CSKCache \
python scripts/07_cskcache/offline_prefill_skills.py --dry-run
```

`manifest.json` is refreshed after every skill. A failed skill is recorded and
processing continues after its server is stopped; the wrapper exits nonzero
after the batch if any failed. Capture-only mode scans disk sidecars for existing
cache IDs but does not reload earlier skills' large tensors into each new server.

## Trace agent replay

`replay_trace_agent.py` sends every
`src/traces/<task>/turn_<N>_inv_<M>.json` as one independent chat-completions
request. A trace JSON already contains the complete history for that invocation,
so generated output is recorded but is not appended to the next JSON.

The wrapper restarts vLLM at the `(mode=cskcache, task)` boundary. Within one
task it keeps the server alive and sends every JSON in `(turn, invocation)`
order, allowing vLLM prefix caching to carry historical prompt KV forward. Only
the skill body newly introduced by the current JSON receives an explicit
CSKCache reuse signal; skill bodies already in the common prefix are not loaded
again.

Before each reuse request, the driver renders the full Qwen3 chat token sequence
locally and finds the exact `skills/<name>/SKILL.md` token subsequence. It does
not add the legacy 06 `<context_segment>` wrapper. For a Markdown file ending in
one newline, the converter removes that newline from the tool message because
Qwen's tool-response template supplies it; the final rendered prompt still
contains the original Markdown exactly once. The driver also requires the trace
body to equal the local Markdown and the offline sidecar token count to match.
The target span therefore excludes chat-template and `tool_response` wrappers.

Run all tasks:

```bash
cd /home/wsh/openhands_code_research
conda activate opencode
bash scripts/07_cskcache/run_trace_agent.sh
```

Completed task outputs are skipped. Replaying a task always starts from its
first JSON; replace completed or partial task output with:

```bash
bash scripts/07_cskcache/run_trace_agent.sh --overwrite
```

Default outputs are written to:

```text
/mnt/Large_Language_Model_Lab_1/wsh/Segmentia/output/07_cskcache/agent_trace_replay/<task>.jsonl
```

A local tokenizer-only plan validates all trace ordering, skill text, cache
sidecars, and token spans without starting vLLM:

```bash
PYTHONPATH=/home/wsh/openhands_code_research/CSKCache \
python scripts/07_cskcache/replay_trace_agent.py \
  --task doc_coauthoring_design_doc --plan-only
```
