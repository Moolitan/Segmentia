# Section 6.6: concurrency

Run `bash run.sh` with no arguments. Edit `config.py` to select platforms or
change the fixed matrix.

For each `(model, system, concurrency, replica)`, one fresh vLLM server is
started. Request A and any CSKCache SSD prefetch are excluded. After all
request-B inputs are prepared, the prefix cache is cleared once and a thread
barrier releases all request-B calls together. The two reported metrics are
server-side TTFT and completed requests per second. Every request is retained
in `samples.csv` and `samples.jsonl`; batch aggregates are in `summary.csv`.
