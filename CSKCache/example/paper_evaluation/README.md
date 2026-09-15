# CSKCache paper evaluation

This directory mirrors the evaluation sections of the paper. Every subsection
owns one zero-argument `run.sh`; there is intentionally no launcher that runs
the whole evaluation suite.

`section_6_8_mechanism_ablation` separates contiguous leading recomputation
from residual compensation and sweeps the compensation coefficient.
`section_6_9_baselines` measures native prefix caching across turns and the
front-loaded candidate-Skill layout. Both read the frozen tasks of
`section_6_2_correction_quality` so their numbers are comparable to it.

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

## Profitability profile

`config.py` exposes `PROFITABILITY_ENABLED` and `SYSTEM_PROFILE_PATH` for the
Section 4.4 decision. With the gate enabled and no matching deployment profile,
rank 0 prints one first-use line, skips the gate for that request, and persists
the request's measured prefill, SSD, H2D, and composition costs. Later requests
use that table to evaluate
`G = T_pf - (T_ready + T_comp + epsilon)` at the authenticated Skill boundary.

When `SYSTEM_PROFILE_PATH` is `None`, the table is stored in
`<catalog directory parent>/cskcache_metadata/system_profiles/<fingerprint>.json`;
setting it selects an explicit file. An independently measured calibration can
be imported without starting the server again:

```bash
python -m cskcache.tools.calibrate_system calibration.json
```

The JSON object supplies `metadata_path`, optional `system_profile_path`, the
deployment `environment`, `strategy`, `ratio`, and four measurements named
`prefill`, `ssd_read`, `h2d`, and `composition`. Each measurement contains a
positive integer `amount` (tokens for compute, bytes for transfers) and its
positive `duration_ms`.
