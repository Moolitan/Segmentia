# Pinned KV pipeline example

This example starts the real vLLM EngineCore, `CSKCacheConnectorV1`, and Qwen3
model, but sends only one model request. Before that request, it calls vLLM's
generic connector-control API; the CSKCache connector creates a Skill ticket
and binds its Tool observation. CSKCache then uses an example-only LMCache
storage plugin to fill the configured 40-layer object directly into LMCache's
pinned CPU buffers; no SSD read is performed. `config.py` selects Skill
chunking independently from the persistent and pinned-memory layouts. The
default `256-token chunks + packed_chunks_single_layer`
configuration retains fixed-size logical chunks in one pinned region per model
layer; `chunk_single_layer` keeps one pinned region per `(chunk, layer)`.
Both layouts are assembled into the same contiguous GPU staging tensor before
correction and PagedKV installation.

Edit `config.py`, activate the `opencode` environment, and run:

```bash
bash CSKCache/example/pinned_kv_pipeline/run.sh
```

The external output directory contains `cskcache_profile.jsonl`,
`request_result.json`, `summary.json`, and `pipeline.png`. The timeline has
three lanes: H2D, staged-key RoPE/layout work, and the layerwise calibration
forward plus KV correction/installation.
Each `calibration_correct_install` layer record also partitions that final lane
into `calibration_forward_ms`, `calibration_commit_ms`,
`residual_correction_ms`, and `suffix_commit_ms`. These are adjacent CUDA-event
intervals on the existing compute stream; profiling does not add a per-phase
synchronization. `summary.json` reports both the 40-layer totals for those four
phases and their sum alongside the original `compute_gpu_ms`.
The synthetic object and online prompt contain the same complete Skill token
sequence. With the default aligned prefix, the first 32 Skill tokens are fully
recomputed, the next 32 are used by the layerwise calibration forward, and the
remaining 7936 tokens are corrected and installed from the pinned KV.
After the layer-0 bootstrap, case JSON selects the pipeline order independently
of the host-buffer layout. H2D-first submits `H2D(l+1)`, enqueues the layer-`l`
calibration and KV installation work, then synchronizes. Compute-first reverses
those two submissions before the same synchronization. Both paths use the same
device-wide synchronization and RoPE/stage implementation. The standalone
default is H2D-first; a case may set `execution_order` to either `h2d_first` or
`compute_first`.

## Packed-layout stage-crossover sweep

Edit `sweep_config.py` and run:

```bash
bash CSKCache/example/pinned_kv_pipeline/run_sweep.sh
```

Every case starts a fresh process, runs one unmeasured request with the same
Skill length, calibration length, packed Host layout, and execution order as
its measured request, resets vLLM's prefix cache without resetting the
connector, moves the warm-up profile aside, and executes the measured request
with a new ticket and request ID.

The current short-Skill matrix varies:

- Skill length: 512, 1024, and 3000 tokens.
- Calibration ratio: 1%, 2%, and 5%. Each calibration interval is computed
  from the current Skill length and rounded to the nearest token, with exact
  half-token cases rounded upward. The resulting calibration lengths are
  5/10/26, 10/20/51, and 30/60/150 tokens, respectively.
  The minimum 32-token full-recompute prefix and its alignment padding are not
  included in this ratio.
- Execution order: Compute-first.
- Three independently warmed measured processes per configuration.

The logical chunk size remains 256 tokens. Storage and pinned Host memory both
use `packed_chunks_single_layer`, so all logical chunks belonging to one model
layer share one contiguous object. Consequently, each layer contributes one
Host source object to LMCache's layerwise H2D consumer. Fixing this final data
path isolates the effect of Skill length and calibration work from per-object
Host submission fragmentation. The matrix contains 3 x 3 x 3 = 27 measured
cases.

Each case has its own specification, output directory, and log. A failed case
is retried once. Three consecutive final failures stop the sweep; isolated
failures are recorded and later cases continue. Re-running the same command
resumes the fixed `RUN_NAME`: complete cases are skipped and incomplete case
directories are preserved with a `.failed-*` suffix.

Results are written below
`pinned_kv_pipeline/sweeps/pipeline_short_skill_stage_crossover_ratio_v1/`.
`per_run.csv` contains all measurements; `aggregate.csv` preserves the same
grouping schema and reports median, mean, and sample standard deviation across
the three repetitions. Each case also preserves
`warmup_profile.jsonl`, making the removal of the layer-0 first-use cost directly
auditable. The CSV files include four 40-layer compute-phase totals and four
stable-layer medians, allowing the calibration forward, calibration install,
residual correction, and suffix install costs to be compared without changing
the pipeline schedule. `balance_points.csv` linearly interpolates the first
sign change between stable per-layer calibration-compute and next-layer H2D
latencies. `stage_crossover_ratio_sweep.{png,pdf}` contains one panel per Skill
length. Actual calibration ratio is on the horizontal axis; red shows
stable per-layer calibration compute, blue shows pinned CPU-to-GPU H2D, and
the light bands show one sample standard deviation. A dashed vertical line
marks the interpolated balance point. This remains a pinned CPU-to-GPU
experiment, not an SSD-read test and not an accuracy evaluation.

For an unattended run:

```bash
nohup bash CSKCache/example/pinned_kv_pipeline/run_sweep.sh \
  > /mnt/Large_Language_Model_Lab_1/wsh/CSKCache/output/pinned_kv_pipeline/pipeline_short_skill_stage_crossover_ratio_v1.nohup.log \
  2>&1 &
echo $!
```

Monitor it with:

```bash
tail -f /mnt/Large_Language_Model_Lab_1/wsh/CSKCache/output/pinned_kv_pipeline/pipeline_short_skill_stage_crossover_ratio_v1.nohup.log
```

## Calibration-forward microbenchmark

Edit `forward_microbench_config.py` and run:

```bash
bash CSKCache/example/pinned_kv_pipeline/run_forward_microbench.sh
```

This is a performance diagnostic, not a correctness or accuracy experiment. It
uses a 512-token synthetic Skill and calibration lengths 16/32/128. All three
lengths keep `calibration_start=288`, so the CSKCache auxiliary forward and the
native vLLM comparison see the same prefix context. Each arm uses eager
Qwen3-14B execution on one worker.

The CSKCache arm reuses `run.py`. Its normal per-layer forward time is computed
from uninstrumented stable layers. Only layers 10, 20, and 30 receive detailed
CUDA events for input normalization, QKV projection, Q/K normalization, RoPE,
PagedKV prefix extraction, KV concatenation, attention, output projection,
post-attention normalization, and MLP. This keeps event overhead out of the
median used for the CSKCache-versus-native comparison.

The native arm first sends a 288-token prefix to populate vLLM's prefix cache,
then profiles the model forward for the same prefix plus `P` new tokens. Its
worker hook surrounds only `Qwen3ForCausalLM.forward`; logits, sampling,
scheduler time, and request wall time are excluded. The output records both
`num_cached_tokens` and the model's actual input-token count so the exact
prefix hit is auditable.

Three independent repetitions are written under
`pinned_kv_pipeline/microbenchmarks/calibration_forward_microbenchmark_v1/`.
`raw.csv` preserves each process, while `summary.csv` and `summary.json` report
cross-repetition medians. Re-running resumes complete cases; an incomplete
CSKCache directory is renamed with an `.incomplete-*` suffix before retrying.
