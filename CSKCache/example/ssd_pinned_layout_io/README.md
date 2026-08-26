# SSD → Pinned layout I/O benchmark

This example measures exactly 16 serial cases:

```text
4 CSKCache layouts × {posix, io_uring} × {buffered, O_DIRECT}
```

The SSD and pinned-CPU layouts always match. Every case transfers the same
logical KV byte count into the same reusable aligned pinned arena, and every
layout is derived from `cskcache.layouts.build_layout_plan`. There is no layout
conversion, H2D copy, GPU work, vLLM request, or concurrent SSD traffic.

For each complete Skill read, all extents belonging to the selected layout are
passed to one `RawBlockCore.read_extents_into` call. Thus `posix` performs its
synchronous extent loop and `io_uring` performs its batch inside the same
backend implementation. Allocation, raw-device open, buffered-page eviction,
and SHA-256 verification are outside the timed interval. All io_uring cases use
ordinary non-fixed buffers because a 12,518-token packed-all-layers region is a
single 1.91-GiB pinned buffer, which this kernel rejects during fixed-buffer
registration. This policy is applied uniformly to all four layouts.

Buffered cases request `POSIX_FADV_DONTNEED` before every warmup and measured
read. This is a cold-cache hint, not a kernel guarantee. `O_DIRECT` bypasses the
page cache. The io_uring queue depth is 32; queue depth is not applicable to the
synchronous POSIX cases.

Edit `config.py`, then run the benchmark yourself:

```bash
cd /home/wsh/openhands_code_research/CSKCache/example/ssd_pinned_layout_io
bash run.sh
```

The result directory name is generated automatically from the UTC start time.

Preparation creates four deterministic raw files and manifests below
`/mnt/990_pro/cskcache_layout_io/12518_tokens`. It fails if the configured path is not on the
expected writable NVMe mount or if an incompatible artifact already exists.
The measured artifacts are written below
`/mnt/Large_Language_Model_Lab_1/wsh/CSKCache/output/ssd_pinned_layout_io`:

- `samples.json`: every repetition and the randomized serial case order;
- `summary.csv` / `summary.json`: p50/p95 latency and throughput;
- `latency_bar_chart.png`: grouped p50-latency bars for four layouts and four transfer policies.

The default dataset has 12,518 tokens, 40 layers, two KV tensors, hidden width
1,024, and BF16-sized elements. Its 2,050,949,120-byte logical payload is aligned
for every region in all four layouts. Changing dimensions to produce an
unaligned region fails early so O_DIRECT and buffered arms never transfer
different physical byte counts.
