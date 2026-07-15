# CSKCache H2D Microbenchmark

This benchmark isolates the current 40-layer `VLLMPagedGPUConnector.to_gpu()`
path from agent generation and vLLM scheduling. It is diagnostic code: it does
not add a pinned-memory tier or change the production connector.

## Matrix

The runner executes eight conditions, one fresh Python process per condition:

```text
profiling:      off, on
CPU allocation: pageable, pinned
position shift: 0, 17000
```

All conditions use the same `doc-coauthoring` entry (3300 tokens, 40 layers,
about 540 MB), five warmups, and thirty measured repetitions by default.
Pinned memory means that the already-loaded CPU tensors are copied once into
page-locked CPU allocations before warmup; this setup cost is recorded but is
excluded from iteration latency. It is an experiment variable, not a new
storage tier.

Position shift zero skips RoPE correction entirely. Shift 17000 instantiates
Qwen3's weight-free vLLM rotary embedding from the real model geometry and
executes the same CSKCache relative-key correction used in production.

## Timing semantics

Every iteration records:

- `operation_wall_ms`: connector call plus the outer completion synchronize;
- `end_to_end_wall_ms`: operation time plus profiling aggregation when enabled;
- `outer_cuda_ms`: one outer CUDA-event pair around the complete connector path;
- `path_gbps`: entry bytes divided by `outer_cuda_ms`;
- profiling-on internal CUDA/host stages for key H2D, value H2D, optional RoPE,
  and scatter.

The profiling on/off overhead comparison uses `end_to_end_wall_ms`. Both modes
retain exactly one outer event pair because otherwise the profiling-off CUDA
completion time would not be measurable. Internal per-layer events exist only
in profiling-on cases.

After warmup, correctness validation gathers the first and last token from the
first and last layers and compares them against the connector's prepared K/V.
Every measured iteration also requires complete layer accounting and zero
skipped layers.

## Run

The agent only prepares and statically validates the benchmark. The user runs:

```bash
bash scripts/07_cskcache/h2d_microbenchmark/run.sh
```

The runner activates the configured conda environment and prepends its `lib/`
directory to `LD_LIBRARY_PATH`. This is required because importing the local
vLLM CUDA extension directly needs the newer conda `libstdc++` ABI; unlike the
vLLM server launcher, this standalone process must not clear that library path.
The shell may still print a `libtinfo.so.6` version warning before the runner
starts; that warning is unrelated to the Python CUDA-extension import.

Useful overrides:

```bash
RUN_ID=my_run \
CSKCACHE_H2D_WARMUP=5 \
CSKCACHE_H2D_REPETITIONS=30 \
CSKCACHE_H2D_DEVICE=cuda:0 \
bash scripts/07_cskcache/h2d_microbenchmark/run.sh
```

Use `--dry-run` to print all eight commands. A completed case JSONL is the
resume boundary: rerunning skips it. `--overwrite` reruns all conditions.
Failure leaves no completed case file, so retry reruns that case and summary is
not produced until all eight cases exist.

## Output

Large/raw outputs stay on external storage:

```text
/mnt/Large_Language_Model_Lab_1/wsh/Segmentia/output/07_cskcache/
  h2d_microbenchmark/<run_id>/
    cases/*.jsonl
    config.json
    raw_iterations.jsonl
    summary.csv
    comparisons.csv
    run.log
```

`summary.csv` contains p10/p50/p90/p95, mean, standard deviation, coefficient
of variation, and min/max for wall time, CUDA time, and effective bandwidth.
`comparisons.csv` directly reports profiling-on versus off, pinned versus
pageable, and shifted versus zero-shift median deltas.
