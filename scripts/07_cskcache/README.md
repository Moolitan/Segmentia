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
