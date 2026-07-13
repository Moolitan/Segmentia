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

The wrapper starts one CSKCache-enabled vLLM server, processes skills serially,
and stops the server on exit:

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
processing continues; the driver exits nonzero after the batch if any failed.
