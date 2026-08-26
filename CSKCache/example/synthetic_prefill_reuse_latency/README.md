# Synthetic TTFT comparison

This microbenchmark runs Qwen3-14B without an Agent, real Skill text, tool-call
generation, or SSD access. It measures three execution methods:

- `normal_prefill`: compute the complete prompt normally;
- `direct_reuse`: install authenticated cached KV without auxiliary correction;
- `deviation_topk`: follow CacheBlend's check-once policy—compute through check
  layer 1, rank tokens by squared-L2 key deviation, select the largest 15%, and
  recompute that same selected set through all later layers.

The fixed token sizes are 512, 1024, 2048, 4096, and 8192. Each method runs in
its own fresh engine process because correction strategy is deployment-wide.
Each point receives one warm-up and five measured requests. Length order is
reversed on odd repetitions, and every request clears vLLM's prefix cache.

For reuse methods, synthetic KV is ready in Pinned CPU before the timer starts.
The validation gate requires scheduler activation, 40 loaded/corrected layers,
the expected execution method, and clean ticket release. `deviation_topk` also
requires exactly one selection event at layer 1 and the expected candidate and
selected-token counts on all 40 layers.

Run from the repository environment:

```bash
conda activate opencode
bash CSKCache/example/synthetic_prefill_reuse_latency/run.sh
```

Raw artifacts are written below
`/mnt/Large_Language_Model_Lab_1/wsh/CSKCache/output/synthetic_prefill_reuse_latency/`.
The lightweight CSV, Markdown, PNG, and PDF outputs are published below
`results/problem_exploration/cskcache_synthetic_prefill_reuse_latency/`.
