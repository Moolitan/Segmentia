# SSD → Pinned scatter/gather benchmark

This example compares independent reads with true Linux vectored reads:

```text
4 pinned layouts × 2 I/O engines × 2 open modes × 2 submission modes
= 32 cases
```

The source is the already prepared 12,518-token
`packed_chunks_all_layers.raw`: one contiguous 1.91-GiB payload. The destination
is one of the four CSKCache pinned layouts. The canonical packed byte stream is
mapped directly into aligned subviews of the destination arena; no temporary
payload buffer or CPU repack is used.

Submission modes:

- `multi_read`: one independent read request for every non-contiguous target
  segment. io_uring batches those SQEs; POSIX executes synchronous `pread` calls.
- `readv`: POSIX uses one `preadv` syscall per iovec group and io_uring uses one
  `Readv` SQE per group. Linux `IOV_MAX` bounds a group to 1,024 iovecs.

The packed-all and packed-single-layer destinations need 1 and 40 iovecs. The
two chunked destinations need 3,920 iovecs, split into four vector groups. Every
segment offset, address, and length is 4-KiB aligned, so buffered and O_DIRECT
cases transfer exactly the same logical bytes. Buffered cases request
`POSIX_FADV_DONTNEED` outside the timed interval. SHA-256 verification is also
outside timing and compares each destination region with its prepared layout
manifest.

The benchmark requires a rebuilt local `lmcache_rust_raw_block_io` containing
`preadv_into` and `readv_uring`. Build it once in the `opencode` environment:

```bash
cd /home/wsh/openhands_code_research/LMCache/rust/raw_block
maturin develop --release
```

Then run without arguments:

```bash
cd /home/wsh/openhands_code_research
bash CSKCache/example/ssd_pinned_scatter_gather/run.sh
```

The script reuses `/mnt/990_pro/cskcache_layout_io/12518_tokens`; it does not
prepare another copy. Results are written to a timestamped directory below
`/mnt/Large_Language_Model_Lab_1/wsh/CSKCache/output/ssd_pinned_scatter_gather`.
