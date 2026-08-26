# CSKCache paper evaluation

This directory mirrors Sections 6.1--6.6 of the paper. Every subsection owns
one zero-argument `run.sh`; there is intentionally no launcher that runs the
whole evaluation suite.

Machine paths and enabled platforms live in `config.py`. Each subsection keeps
its experiment matrix in its local `config.py`. Raw artifacts are written to
the external `OUTPUT_ROOT`, and every measured case is saved to both
`samples.jsonl` and the stable, cross-platform `samples.csv` schema. Each
analyzer additionally writes `summary.csv` and publication figures.

Latency is measured inside the instrumented workspace copy at `vllm/`, from
`api_request_received` to `first_token_ready`. Request A, offline cache build,
and (except for the explicit Blocking-SSD ablation) SSD prefetch are outside
this interval. Before GPU experiments, run Section 6.1 and build every listed
Skill into each active model's raw-block Catalog with
`CSKCache/example/offline_skill_kv`; the suite fails before server startup when
a required cache object is absent.

After copying result directories from other machines, add their roots to
`MERGE_INPUT_ROOTS` and run:

```bash
cd CSKCache/example/paper_evaluation
python merge_results.py
python plot_merged.py
```

Both utilities accept no command-line paths and never modify source runs.
They produce `combined_samples.csv`, `combined_summary.csv`, and any available
cross-platform quality, TTFT, and concurrency figures under
`MERGED_OUTPUT_DIR`.
