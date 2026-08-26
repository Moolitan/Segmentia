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

## Host-layout, compute, and H2D latency sweep

Edit `sweep_config.py` and run:

```bash
bash CSKCache/example/pinned_kv_pipeline/run_sweep.sh
```

Every case starts a fresh process, runs one same-shape warm-up request, resets
vLLM's prefix cache without resetting the connector, moves the warm-up profile
aside, and executes the measured request with a new ticket and request ID.

The current matrix uses six Skill lengths (512, 1024, 3000, 5000, 8192, and
10000 tokens), a fixed 32-token calibration interval, 256-token logical chunks,
and Compute-first execution. Each Skill length is measured with two pinned Host
layouts: `chunk_single_layer`, which exposes one source object per logical
chunk and layer, and `packed_chunks_single_layer`, which exposes one source
object for the complete layer. There is one measured process per configuration,
so the six paired points require 12 measured requests.

The experiment compares three stable-layer medians over layers 5--34:

- H2D latency with `chunk_single_layer`;
- calibration forward, correction, and KV installation latency from the packed
  run;
- H2D latency with `packed_chunks_single_layer`.

This isolates the effect of growing per-layer source-object count while holding
the logical chunk size and calibration token count fixed.

Each case has its own specification, output directory, and log. A failed case
is retried once. Three consecutive final failures stop the sweep; isolated
failures are recorded and later cases continue. Re-running the same command
resumes the fixed `RUN_NAME`: complete cases are skipped and incomplete case
directories are preserved with a `.failed-*` suffix.

Results are written below
`pinned_kv_pipeline/sweeps/pipeline_layout_compute_latency_v1/`.
`per_run.csv` contains all measurements and `aggregate.csv` preserves the
per-layout grouping. Each case also preserves
`warmup_profile.jsonl`, making the removal of the layer-0 first-use cost directly
auditable. The CSV files include four 40-layer compute-phase totals and four
stable-layer medians, allowing the calibration forward, calibration install,
residual correction, and suffix install costs to be compared without changing
the pipeline schedule. `latency_comparison.csv` contains only the three plotted
metrics. `layout_compute_latency.{png,pdf}` plots them against Skill length.
This remains a pinned CPU-to-GPU experiment, not an SSD-read test and not an
accuracy evaluation. The persistent synthetic Catalog remains packed; the
`chunk_single_layer` arm changes the Host source-object organization after
acquisition rather than measuring a fine-grained SSD layout.

For an unattended run:

```bash
nohup bash CSKCache/example/pinned_kv_pipeline/run_sweep.sh \
  > /mnt/Large_Language_Model_Lab_1/wsh/CSKCache/output/pinned_kv_pipeline/pipeline_layout_compute_latency_v1.nohup.log \
  2>&1 &
echo $!
```

Monitor it with:

```bash
tail -f /mnt/Large_Language_Model_Lab_1/wsh/CSKCache/output/pinned_kv_pipeline/pipeline_layout_compute_latency_v1.nohup.log
```

## Chunk-single-layer completion-latency sweep

Run the dedicated Skill-length by chunk-size matrix with:

```bash
bash CSKCache/example/pinned_kv_pipeline/run_chunk_single_layer_completion_sweep.sh
```

Pinned Host memory uses `chunk_single_layer`, which is the layout under test.
The synthetic persistent Catalog remains `packed_chunks_single_layer` because
the current Catalog contract stores one complete Skill extent per model layer;
this experiment does not read SSD, and only the Host layout determines how many
source objects are submitted by the measured H2D path. The matrix crosses Skill
lengths 1K/2K/4K/8K with chunk sizes
64/128/256/512 tokens. Calibration stays at 32 tokens and execution stays
Compute-first. Every configuration runs in three fresh processes; each process
executes one same-shape warm-up, clears vLLM's prefix cache without resetting
the connector, and then executes one measured request. Odd repetitions reverse
the 16-arm order to reduce fixed run-position bias.

The real request still restores all 40 Qwen3-14B layers. The primary metric,
`completion_through_layer_ms`, is recomputed from the H2D, staged-key transform,
and calibration/correction/install events belonging only to layers 0--38.
Layer 39 remains in every raw profile for audit, but it is excluded from the
metric and figure because its known cold-tail latency would otherwise determine
the endpoint. This metric is a profiled restoration-pipeline span, not request
wall time.

Results are resumable under
`pinned_kv_pipeline/sweeps/chunk_single_layer_completion_without_final_layer_v2/`.
`per_run.csv` preserves all 48 processes, `aggregate.csv` reports median, mean,
sample standard deviation, minimum, and maximum for each of the 16
configurations, and `completion_latency_without_final_layer.{png,pdf}` plots
the median latency against Skill length with one line per chunk size. A failed
case is retried once; three consecutive final failures stop later cases while
preserving completed outputs and logs.

## Packed-chunks-single-layer completion-latency control

Run the matched packed-layout control with:

```bash
bash CSKCache/example/pinned_kv_pipeline/run_packed_chunks_single_layer_completion_sweep.sh
```

This entry reuses the same completion runner and changes only the selected
configuration module. Persistent storage and pinned Host memory both use
`packed_chunks_single_layer`; Skill lengths remain 1K/2K/4K/8K, logical chunk
sizes remain 64/128/256/512 tokens, and every configuration still has three
fresh-process repetitions, for 48 measured cases. Calibration, Compute-first
ordering, same-shape warm-up, prefix-cache reset, retry/resume behavior, and the
layer-0--38 primary metric are identical to the chunk-single-layer sweep.

Because all logical chunks belonging to one layer share one physical Host
object, changing logical chunk size should not change the measured per-layer
source-object count. The matrix is therefore a control for distinguishing
logical chunk metadata from physical Host submission granularity. Results are
written to
`pinned_kv_pipeline/sweeps/packed_chunks_single_layer_completion_without_final_layer_v1/`;
they do not overwrite the completed chunk-single-layer run.

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

## Copy-engine/SM contention diagnosis

Edit `contention_config.py` and run:

```bash
bash CSKCache/example/pinned_kv_pipeline/run_contention.sh
```

This is the first go/no-go test for reuse-induced bandwidth inversion. It fixes
the synthetic Skill at 3072 tokens, the calibration interval at 32 tokens, and
the Host layout at `packed_chunks_single_layer`. Four independently warmed arms
measure H2D-only, calibration-only, concurrent H2D and calibration, and the
normal full restoration path. Each arm is repeated three times.

The first three arms use an example-only connector. H2D copies the same packed
K/V range as the normal restoration path from pinned CPU memory into private
double-buffered GPU tensors. Calibration uses the real CSKCache auxiliary
Qwen3 layer forward. The concurrent arm submits the next layer's copy on the
LMCache load stream while the current calibration layer runs on the default
stream; it deliberately excludes residual correction and PagedKV installation.
After the diagnostic interval, the connector runs the normal restoration path
once so the serving request consumes valid KV. No diagnostic behavior is added
to production CSKCache, vLLM, or LMCache.

`raw.csv` preserves every repetition and `summary.json` reports the median
stable-layer H2D and calibration slowdowns. A 15% slowdown in either stage opens
the Nsight counter gate; it does not by itself establish an HBM/L2 root cause.
Results are written below
`pinned_kv_pipeline/contention/reuse_induced_bandwidth_inversion_smoke_v1/`.
