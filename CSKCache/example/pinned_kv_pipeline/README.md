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

## Calibration-ratio and chunk-size sweep

Edit `sweep_config.py` and run:

```bash
bash CSKCache/example/pinned_kv_pipeline/run_sweep.sh
```

Every process runs one unmeasured request with the same Skill length,
calibration length, host layout, chunk size, and execution order as its measured
request. It then resets vLLM's prefix cache without resetting the connector,
moves the warm-up profile aside, and executes the measured request with a new
ticket and request ID.

The matrix fixes the Skill at 8192 tokens and varies:

- Calibration ratio: 5%, 10%, and 15%. Calibration token counts are rounded to
  the nearest 16-token boundary, producing 416, 816, and 1232 tokens. The
  minimum full-recompute prefix and its alignment padding are not included in
  this ratio.
- Logical chunk size: 64, 128, 256, 512, 1024, and 8192 tokens. The last point
  naturally produces one chunk; it is not a separate chunking mode.
- Host layout: `chunk_single_layer` and `packed_chunks_single_layer`.
- Execution order: H2D-first and Compute-first.
- One warmed measured process per configuration.

This gives 3 x 6 x 2 x 2 = 72 measured cases. Both host layouts use the same
logical chunk plan and the same persistent per-layer source. With
`chunk_single_layer`, one model layer exposes one MemoryObj per chunk and the
LMCache consumer enqueues separate K/V copies for every source object. With
`packed_chunks_single_layer`, all logical chunks of a layer share one contiguous
MemoryObj, so the consumer enqueues one large K copy and one large V copy. The
`batched_to_gpu` call groups the source objects at its API boundary but does not
fuse the per-object `copy_` operations.

Each case has its own specification, output directory, and log. A failed case
is retried once. Three consecutive final failures stop the sweep; isolated
failures are recorded and later cases continue. Re-running the same command
resumes the fixed `RUN_NAME`: complete cases are skipped and incomplete case
directories are preserved with a `.failed-*` suffix.

Results are written below
`pinned_kv_pipeline/sweeps/pipeline_calibration_ratio_chunk_sweep_v1/`.
`per_run.csv` contains all measurements; `aggregate.csv` preserves the same
grouping schema so repetitions can be increased later. Each case also preserves
`warmup_profile.jsonl`, making the removal of the layer-0 first-use cost directly
auditable. `calibration_ratio_chunk_sweep.{png,pdf}` contains three panels for
the calibration ratios. Blue/red encode H2D-first/Compute-first, while
solid/dashed lines encode chunk-single-layer/packed-chunks-single-layer. The
vertical axis is the warmed pipeline span from the first layer H2D start to the
last layer correction/install completion. This remains a pinned CPU-to-GPU
experiment, not an SSD-read test.

For an unattended run:

```bash
nohup bash CSKCache/example/pinned_kv_pipeline/run_sweep.sh \
  > /mnt/Large_Language_Model_Lab_1/wsh/CSKCache/output/pinned_kv_pipeline/pipeline_calibration_ratio_chunk_sweep_v1.nohup.log \
  2>&1 &
echo $!
```

Monitor it with:

```bash
tail -f /mnt/Large_Language_Model_Lab_1/wsh/CSKCache/output/pinned_kv_pipeline/pipeline_calibration_ratio_chunk_sweep_v1.nohup.log
```
