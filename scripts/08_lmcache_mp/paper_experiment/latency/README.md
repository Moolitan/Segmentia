# 8K Skill latency experiment

This experiment measures one-token end-to-end latency for the same prompt under:

1. `full`: complete prompt Prefill without LMCache;
2. `direct`: direct reuse of the offline Skill KV;
3. `correction`: recompute the first 256 Skill tokens and apply the Section 3.2 K-only correction (`[132,256)`, `alpha=0.6`).

It uses the existing offline object `Auto-claude-code-research-in-sleep/paper-write` (8021 tokens). It does not run OpenHands and does not create another fake Skill. Every prompt has 1024 dynamic-prefix tokens, the exact cached Skill tokens, and 32 dynamic-suffix tokens. A sample is byte-identical across modes, while different samples have different leading tokens so vLLM automatic prefix caching cannot turn repeated measurements into prefix hits.

## Lightweight validation

```bash
conda activate opencode
bash scripts/08_lmcache_mp/paper_experiment/latency/run.sh --dry-run
```

This rebuilds the cached object with the local tokenizer, checks all manifest hashes, and verifies cross-mode prompt identity. It does not start vLLM.

## GPU run

```bash
conda activate opencode
RUN_ID=paper-write-8k-v1 \
  bash scripts/08_lmcache_mp/paper_experiment/latency/run.sh
```

Each `(replica, mode, task)` starts a fresh vLLM server. Defaults are three replicas, one cold request, two warmups, and ten measured requests per leaf. The mode order rotates across replicas. Set `RESUME=1` (default) to validate and skip completed leaves when resuming the same `RUN_ID`.

Raw outputs are written under:

```text
/mnt/Large_Language_Model_Lab_1/wsh/Segmentia/output/08_lmcache_mp/paper_experiment_latency/<RUN_ID>/
```

The lightweight summary is written to:

```text
results/problem_exploration/skill_latency_8k/
```

The reported latency is non-streaming HTTP wall time through one generated token. It includes scheduling, Prefill/cache load/correction, and one decode token; it is not labeled as pure Prefill time or strict streaming TTFT.
