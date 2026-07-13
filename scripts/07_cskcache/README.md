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

## Real OpenHands agent run

`run_real_agent.py` is a self-contained OpenHands runner. It directly constructs
the SDK `LLM`, `Agent`, tools, and `Conversation`, and locally loads
`anthropic_skill_benchmark/<task>/turns/turn_N.txt` in numeric turn order. The
agent configuration follows the real benchmark setup, but there is no runtime
import or dynamic loading from `scripts/03_14B_anthropic`, `05`, or `06`. It
does not replay `src/traces`: the model chooses tools, OpenHands executes them in
an isolated workspace, and each real tool result enters the next model request.

The CSKCache injector wraps the OpenHands LLM `_transport_call`. It normalizes
Qwen tool messages to string content, renders the exact final prompt with the
local tokenizer, and searches for complete offline `SKILL.md` token sequences.
A reuse signal is emitted only when an exact skill span lies beyond the stable
token prefix shared with the preceding successful request. Mentioning a skill
by name does not trigger reuse; the full SkillTool result must actually enter
the prompt. Historical spans inside the stable prefix are inherited through
vLLM prefix caching and are not explicitly loaded again.

If one model response invokes multiple SkillTools, their complete results enter
the next request together. The injector emits one ordered `entries` list, and
CSKCache loads each non-overlapping span in target-token order while normally
prefilling any gaps. This preserves the real agent's batched tool-call behavior.

For skills with `scripts/`, `references/`, or `assets/`, OpenHands normally
appends a resource listing after the guide. The HTTP prompt copy moves that
listing before the guide so the complete canonical Markdown remains the final
contiguous tool-content segment; no resource information is removed and the
Conversation event itself is unchanged.

Every actual transport attempt is written immediately as
`requests/turn_N_inv_M.json`, including messages, tools, response/error,
latency, prompt length, stable-prefix length, and the optional CSKCache signal.
The agent's generated tool call and the real tool result therefore determine
the next JSON instead of a prerecorded trace.

OpenHands intermediate state is also saved incrementally. `agent.log` contains
the SDK console output, while `openhands_events.jsonl` contains one immediately
flushed record per Conversation event, including the current benchmark turn,
event type, assistant action, tool observation, message, or state event. Events
already written remain available when a later tool or model request fails.

The wrapper restarts vLLM at the `(mode=cskcache, task)` boundary. It enables
prefix caching, runs all benchmark turns for that task in one Conversation, and
stops the server before the next task. Workspaces and raw request artifacts are
isolated below a run ID on external storage.

The wrapper also keeps dynamic-library environments process-local. Its parent
shell unsets `LD_LIBRARY_PATH` so OpenHands can use the system `tmux` without
loading Conda's incompatible `libtinfo`. Only the vLLM server command receives
`${CONDA_PREFIX}/lib`, which supplies the newer `libstdc++.so.6` required by
`vllm._C`. Do not export the Conda library path globally in this wrapper.

Run all tasks:

```bash
cd /home/wsh/openhands_code_research
conda activate opencode
bash scripts/07_cskcache/run_real_agent.sh
```

Completed task outputs are skipped. A failed task must restart from its first
turn because both Conversation and prefix-cache state are task-local. Replace a
task directory with:

```bash
bash scripts/07_cskcache/run_real_agent.sh --overwrite
```

Default outputs are written to:

```text
/mnt/Large_Language_Model_Lab_1/wsh/Segmentia/output/07_cskcache/real_agent_runs/<run_id>/
  <task>/workspace/
  <task>/requests/turn_N_inv_M.json
  <task>/_summary.json
  <task>/agent.log
  <task>/openhands_events.jsonl
  vllm_<task>.log
```

A dry run validates benchmark turns, all local skill text hashes, tokenizer
lengths, and offline cache sidecars without creating workspaces or starting
vLLM:

```bash
bash scripts/07_cskcache/run_real_agent.sh --dry-run
```
