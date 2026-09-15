# Section 6.6: steady-state concurrent throughput

Run `bash run.sh` with no arguments. Edit `config.py` to select platforms or
change the fixed matrix.

The fixed matrix compares `Full Prefill` with `SkillCache-5%` on the verified
6.4K- and 13.3K-token Skills at concurrency 1, 2, 4, and 8. Each
`(model, system, Skill, concurrency, replica)` starts one fresh vLLM server.
There is no server or prefix-cache reuse across points.

SkillCache points use one fixed 40-GiB LMCache Host pool at every concurrency.
The pool size is deliberately constant so capacity does not become a changing
resource across the throughput sweep. During warmup and measurement, the runner
samples LMCache allocated Host bytes and active objects from `/metrics`, plus
the vLLM process-group RSS and system available memory, every 0.2 seconds.

Each concurrent slot repeatedly performs one complete Skill invocation:
request A selects the Skill and creates the prefetch ticket, then request B
prefills the selected Skill and emits one token. As soon as B completes, that
slot starts its next invocation. Request A and SSD prefetch therefore remain
inside the throughput wall-clock interval. A unique nonce precedes every Skill,
so ordinary vLLM prefix caching cannot reuse the Skill span across invocations.

After a 20-second warmup, the runner counts request-B completions inside a
120-second measurement window and drains requests already in flight without
counting late completions. The primary metric is completed request B/s. Per
request server TTFT is retained as a validity diagnostic. `samples.csv` and
`samples.jsonl` contain counted requests; `replica_summary.csv` contains one row
per server lifecycle; `summary.csv` reports the median and range across three
replicas. Each attempt additionally writes `memory_samples.jsonl` and
`memory_summary.json`; run-level `memory_replica_summary.csv` and
`memory_summary.csv` aggregate them. The final single-column PDF and PNG contain
one subplot per Skill length, overlaying peak Host KV allocation on sustained
throughput.

If a request fails, SkillCache falls back, or a SkillCache request reports zero
reused tokens, the point is marked failed and the run exits. Re-running the same
command resumes completed points and creates a new attempt directory for the
failed point.
