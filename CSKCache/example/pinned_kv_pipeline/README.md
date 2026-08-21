# Pinned KV pipeline example

This example starts the real vLLM EngineCore, LMCache connector, and Qwen3
model, but sends only one model request. Before that request, it calls vLLM's
generic connector-control API; `CSKCacheConnectorV1` creates a Skill ticket and
binds its Tool observation. An example-only LMCache storage plugin fills the configured
40-layer object directly into LMCache's pinned CPU buffers; no SSD read is
performed. Case JSON may select either one complete pinned object per layer or
a 256-token chunk-major pinned layout. Both layouts are assembled into the same
contiguous GPU staging tensor before correction and PagedKV installation.

Edit `config.py`, activate the `opencode` environment, and run:

```bash
bash CSKCache/example/pinned_kv_pipeline/run.sh
```

The external output directory contains `cskcache_profile.jsonl`,
`request_result.json`, `summary.json`, and `pipeline.png`. The timeline has
three lanes: H2D, staged-key RoPE/layout work, and calibration/residual/install.
The synthetic object and online prompt contain the same complete Skill token
sequence. With the default aligned prefix, the first 32 Skill tokens are fully
recomputed, the next 32 are used by the layerwise calibration forward, and the
remaining 7936 tokens are corrected and installed from the pinned KV.
After the layer-0 bootstrap, case JSON selects the pipeline order independently
of the host-buffer layout. H2D-first submits `H2D(l+1)`, enqueues `C/R/I(l)`,
then synchronizes. Compute-first enqueues `C/R/I(l)`, submits `H2D(l+1)`, then
synchronizes. Both paths use the same device-wide synchronization and
RoPE/stage implementation. The standalone default is H2D-first; a case may set
`execution_order` to either `h2d_first` or `compute_first`.

To sweep Skill length and calibration length, edit `sweep_config.py` and run:

```bash
bash CSKCache/example/pinned_kv_pipeline/run_sweep.sh
```

Each point runs in an isolated vLLM process. The sweep takes the median of
middle-layer `C/R/I(l)` and concurrent `H2D(l+1)` intervals, then writes
`per_run.csv`, `aggregate.csv`, `balance_points.csv`, `summary.json`, and the
`balance_curves.png`/`.pdf` figures under the external sweep output directory.

The two-dimensional comparison is orchestrated from
`scripts/08_lmcache_mp/pinned_kv_layout_diagnosis/run_pipeline.sh`. It sweeps
full-layer and 256/512/1024/2048-token chunk-major objects under both execution
orders, and restarts vLLM for every `(granularity, order, repetition)` case.
