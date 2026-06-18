# 06 Context-Free Segment Cache

This directory is a standalone harness for comparing trace decode outputs under
three conditions:

- `recompute`: no segment KV reuse.
- `direct`: load offline skill KV and inject it without RoPE correction.
- `rope`: load offline skill KV and inject it with vLLM's RoPE key correction.

It intentionally does not import `scripts/05_context_segment_agent_kv`.

## Important Prefix-Cache Note

The modified vLLM scheduler caps prefix-cache hits at the first loaded target
span and then stops prefill exactly at `target_start`, so a request with a valid
`context_segment_cache.targets[]` should still inject even when normal prefix
caching is enabled. Do not judge a reuse run valid unless vLLM was started with
`VLLM_CONTEXT_SEGMENT_KV_DIR` pointing at the offline `.pt` files.

## 1. Offline Skill KV Prefill

Large KV `.pt` files live outside the repo by default:

```bash
export SEGMENTIA_OUTPUT_DIR=/mnt/Large_Language_Model_Lab_1/wsh/Segmentia/output/06_context_free_segment_cache
```

Start vLLM with a save dir:

```bash
conda activate opencode
export VLLM_CONTEXT_SEGMENT_KV_SAVE_DIR=$SEGMENTIA_OUTPUT_DIR/offline_skill_kv
bash scripts/vllm_stop.sh || true
bash scripts/vllm_start.sh
```

Then collect skill KV:

```bash
python scripts/06_context_free_segment_cache/prefill_skill_kv.py \
  --tasks all \
  --kv-dir $SEGMENTIA_OUTPUT_DIR/offline_skill_kv
```

The script verifies whether the no-context wrapped skill tokens match every
configured target span. If token identity fails, vLLM will reject reuse; inspect
`manifest.json` before continuing.

## 2. Decode Comparison

Restart vLLM with the offline KV loaded and a CKSim dump dir:

```bash
export VLLM_CONTEXT_SEGMENT_KV_DIR=$SEGMENTIA_OUTPUT_DIR/offline_skill_kv
export VLLM_CONTEXT_SEGMENT_KV_SAVE_DIR=$SEGMENTIA_OUTPUT_DIR/cksim_kv
bash scripts/vllm_stop.sh || true
bash scripts/vllm_start.sh
```

Run decode at skill occurrence 2/3 only:

```bash
python scripts/06_context_free_segment_cache/run_decode_compare.py \
  --tasks all \
  --occurrences 2,3 \
  --modes recompute,direct,rope \
  --max-tokens 4096 \
  --output results/problem_exploration/headline_semantic_action_gap/data/decode_outputs.jsonl
```

Each JSONL row contains the full response for one task/skill/occurrence/mode/sample:
`text` (visible deliverable), `content` (raw visible content), `reasoning`
(the hidden `reasoning_content` chain-of-thought emitted by Qwen3),
`tool_calls`, and `finish_reason`. Saving `reasoning` and the structured
`tool_calls` is what lets the scorer evaluate the full generated sequence and
the action-level trajectory rather than visible wording alone.

Rows are flushed after each request, so a later vLLM failure does not lose
completed generations. Use `--append --skip-existing` to resume a partial run.

### Repeated sampling (stable effect vs sampling noise)

`--repeats N` decodes each (case, mode) N times and tags rows with
`sample_index`. Use it with a non-zero temperature so the samples actually
vary; `--seed-base S` makes sample `i` use seed `S+i` for reproducible runs:

```bash
python scripts/06_context_free_segment_cache/run_decode_compare.py \
  --tasks all --occurrences 2,3 --modes recompute,direct,rope \
  --repeats 5 --temperature 0.7 --seed-base 1000 \
  --max-tokens 4096 \
  --output results/problem_exploration/stability_systematic_vs_noise/data/decode_outputs_stability.jsonl
```

`run_decode_compare.sh` forwards `REPEATS`, `TEMPERATURE`, and `SEED_BASE`
environment variables to the same flags.

`--dump-kv-for-cksim` is intentionally off by default. It registers large
recompute/direct/rope spans for every case and the current vLLM registry keeps
those tensors on GPU after saving, which can OOM on long prompts. Prefer running
decode first, then run a smaller CKSim pass by task or occurrence.

## 3. Metrics

```bash
python scripts/06_context_free_segment_cache/evaluate_outputs.py \
  --input results/problem_exploration/headline_semantic_action_gap/data/decode_outputs.jsonl \
  --cksim-kv-dir $SEGMENTIA_OUTPUT_DIR/cksim_kv
```

Use `--skip-embedding` for a quick BLEU/ROUGE/CKSim-only pass.

The scorer aggregates over samples and emits three first-class metric families
per reuse mode, all measured against the recompute reference:

- **Text similarity, split by stream.** `full_*` scores the full sequence
  (reasoning + visible deliverable); `deliverable_*` scores only the visible
  output; `reasoning_*` scores only the hidden chain-of-thought. A reuse mode
  can no longer pass by matching the visible answer while diverging in its plan.
- **Action / trajectory consistency.** From the structured `tool_calls` we
  compute `modality_match_rate` (tool vs text), `tool_set_match_rate`, and
  `trajectory_match_rate` (exact ordered tool-name sequence). These are the
  primary behavior-fidelity numbers.
- **CKSim** of the reused key/value against recompute (unchanged).

`metrics_rows.csv` is one aggregated row per (case, reuse mode);
`metrics_summary.json` has per-mode means including a `stability` section.

### Stability: is a divergence systematic or just noise?

`stability_rows.csv` reports, per (case, mode) **including recompute**, the
`action_self_consistency` = the fraction of that mode's own samples that repeat
its majority action. The recompute self-consistency is the sampling-noise
floor. A reuse mode's divergence from recompute counts as a *stable systematic*
effect only when the mode is itself self-consistent (it keeps making the same
different choice) while differing from the recompute majority — not when both
recompute and the reuse mode are flipping around at the noise floor.

## 4. Value-side repair experiment (key vs value)

The plain `rope` arm fixes only the reused *key* position; the *value* is left
as the no-context skill value. To test whether the residual behavior gap is a
key problem or a value problem, run the 2x2 oracle ablation. It builds mixed
per-case KV files offline (no vLLM changes) and injects them through the
existing `direct`/`rope` paths:

| arm | key | value | inject mode |
|---|---|---|---|
| `rope` | skill (RoPE-corrected) | skill | rope |
| `vrep` | skill (RoPE-corrected) | recompute oracle | rope |
| `krep` | recompute oracle | skill | direct |
| `oracle` | recompute oracle | recompute oracle | direct |

Reading: `rope`→`vrep` isolates the marginal effect of repairing the **value**;
`rope`→`krep` isolates repairing the **key**; `oracle` vs `recompute` checks the
splice path itself is faithful (it should reproduce recompute).

```bash
conda activate opencode
bash scripts/06_context_free_segment_cache/run_value_repair_compare.sh
```

The script runs three phases with per-arm vLLM restarts: (A) recompute while
dumping the in-context oracle KV into `cksim_kv/`, (B)
`build_repair_arms_kv.py` splices the mixed KV into `repair_arms_kv/`, (C)
decode `rope`/`vrep`/`krep`/`oracle`. It honors `TASKS`, `OCCURRENCES`,
`REPEATS`, `TEMPERATURE`, and `SEED_BASE`, and prints the matching
`evaluate_outputs.py` command (writing `value_repair_*` metric files).

Note: phase A dumps KV for every selected case; per the prefix-cache note above
this keeps tensors resident and can OOM on the longest skills. Narrow with
`TASKS=...` / `OCCURRENCES=...` if needed.
