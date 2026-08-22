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

## Warmed critical sweep

Edit `sweep_config.py` and run:

```bash
bash CSKCache/example/pinned_kv_pipeline/run_sweep.sh
```

This follow-up sweep removes the calibration-path first-use cost observed in the
exploratory matrix. Every process runs one unmeasured CSKCache request with the
same Skill length, calibration length, layout, and execution order. It then
resets vLLM's prefix cache without resetting the connector, moves the warm-up
profile aside, and runs one measured request. The warm-up and measured requests
use distinct tickets and request IDs.

The matrix keeps only the decisive configurations:

- Skill tokens: 1K, 3K, 5K, and 8K.
- Calibration tokens: 16, 16, 32, and 64 respectively, carried over from the
  exploratory sweep and held constant within each Skill length.
- Transfer organization: per-chunk layer objects, packed 256-token chunks in
  one layer object, and a one-chunk layer reference whose chunk size equals the
  Skill length.
- Execution order: H2D-first and Compute-first.
- Three independent warmed processes per configuration.

This gives 4 x 3 x 2 x 3 = 72 measured cases. The 256-token chunkwise and
packed-layer variants share the same logical chunks and persistent per-layer
region; only the pinned-memory H2D source organization changes. The one-chunk
case is a reference produced by setting `chunk_size_tokens=skill_tokens`; it is
not a separate chunking mode.

Each case has its own specification, output directory, and log. A failed case
is retried once. Three consecutive final failures stop the sweep; isolated
failures are recorded and later cases continue. Re-running the same command
resumes the fixed `RUN_NAME`: complete cases are skipped and incomplete case
directories are preserved with a `.failed-*` suffix.

Results are written below
`pinned_kv_pipeline/sweeps/pipeline_warmed_critical_v2/`. `per_run.csv` contains
all independent measurements; `aggregate.csv` reports median, mean, and sample
standard deviation for every metric. Each case also preserves
`warmup_profile.jsonl`, making the removal of the layer-0 first-use cost directly
auditable. This remains a pinned CPU-to-GPU experiment, not an SSD-read test.

For an unattended run:

```bash
nohup bash CSKCache/example/pinned_kv_pipeline/run_sweep.sh \
  > /mnt/Large_Language_Model_Lab_1/wsh/CSKCache/output/pinned_kv_pipeline/pipeline_warmed_critical_v2.nohup.log \
  2>&1 &
echo $!
```

Monitor it with:

```bash
tail -f /mnt/Large_Language_Model_Lab_1/wsh/CSKCache/output/pinned_kv_pipeline/pipeline_warmed_critical_v2.nohup.log
```
