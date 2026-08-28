# Section 6.3: fixed Skill-length TTFT scaling

This experiment measures server-side TTFT for one frozen task--Skill pair in
each of six Qwen3-14B token buckets: `<1K`, `1K-3K`, `3K-5K`, `5K-8K`,
`8K-10K`, and `>10K`. Boundaries are left-closed and right-open. The first
five pairs come from the frozen SkillsBench checkout. SkillsBench has no
`>10K` Skill, so the final pair is the explicitly labelled
`curated_repository_task` named `proof-gradient-descent-audit`, using the
repository's 13,314-token `proof-checker` Skill.

Each bucket compares exactly three systems:

- `Full`: normal full prefill with no external Skill KV connector.
- `CacheBlend-15%`: the existing deviation-top-k path, selecting 15% of the
  reusable interval for recomputation.
- `CSKCache-5%`: ratio-prefix correction with a 5% calibration interval.

Every complete arm has five measurement repetitions per bucket, for
`6 × 3 × 5 = 90` samples on one model. Before every excluded request A, the
runner clears ordinary vLLM prefix cache. It preserves the prefix from A to the
immediately following measured request B, while external Skill KV remains
resident. The service restarts once per `(platform, system)`. Invalid attempts
are marked failed and are retried into a new attempt directory on resume.

## 1. Build the dedicated Qwen3-14B offline pool

The plan command performs source/hash/tokenizer checks but starts no GPU
service and writes nothing:

```bash
cd /home/wsh/openhands_code_research
conda activate opencode
bash CSKCache/example/paper_evaluation/section_6_3_latency_scaling/offline_cache/run.sh --plan
```

Build and verify the six-object raw-block pool:

```bash
bash CSKCache/example/paper_evaluation/section_6_3_latency_scaling/offline_cache/run.sh \
  2>&1 | tee /tmp/fixed_length_offline_qwen3_14b.log
```

The build is resumable. Success writes:

```text
/mnt/Large_Language_Model_Lab_1/wsh/CSKCache/cache_pools/
  Qwen3-14B-fixed-length-v1/fixed_length_manifest.json
```

The measurement runner refuses to start unless this manifest authenticates
the Catalog, all six source texts, token identities, and complete layer lists.

## 2. Run the longest-workload smoke gate

The smoke run uses only `>10K`, one repetition, and all three systems. It keeps
the run active so the full command can reuse those three valid cases:

```bash
LATENCY_SCALING_LIMIT=smoke \
  bash CSKCache/example/paper_evaluation/section_6_3_latency_scaling/run.sh
```

## 3. Run the complete Qwen3-14B matrix

```bash
LATENCY_SCALING_LIMIT=all \
  bash CSKCache/example/paper_evaluation/section_6_3_latency_scaling/run.sh
```

The completed run contains `selected_workloads.csv`, `samples.csv`,
`summary.csv`, `analysis.md`, and the real single-model interim plot
`ttft_a6000_qwen3_14b.{pdf,png}`. The x-axis is the six length buckets; each
bucket has three grouped bars; the y-axis is server-side TTFT.

## 4. Produce the final 1×3 model figure

After the medium and large A100 model runs have real complete summaries, merge
the three explicit run directories:

```bash
PYTHONPATH=CSKCache:CSKCache/example \
  python -m paper_evaluation.section_6_3_latency_scaling.merge_model_panels \
  --run-dir /path/to/a6000_qwen3_14b/run \
  --run-dir /path/to/a100_medium/run \
  --run-dir /path/to/a100_large/run \
  --output-dir /path/to/merged
```

This emits `ttft_model_comparison.{pdf,png}` as a 1×3 figure, one model per
column. It fails closed if any model lacks all 18 bucket/system summaries and
never fills missing panels with placeholders.
