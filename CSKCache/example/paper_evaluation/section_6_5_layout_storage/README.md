# Section 6.5: physical layout and storage hierarchy

Run `bash run.sh` with no arguments.

The first experiment imports the already completed 12518-token SSD-to-pinned
matrix for four physical layouts and four POSIX/io_uring x buffered/O_DIRECT
strategies. The imported rows remain identifiable by their source manifest and
are copied into this subsection's stable CSV.

The second experiment measures one real CSKCache request path in two modes.
`Blocking SSD` sends request B as soon as request A returns, so any unfinished
SSD load is inside server-side TTFT. `Prefetched SSD` waits for `csk_host_ready`
before request B, so SSD loading is excluded. These two bars isolate the value
of A-to-B prefetch; they are deliberately not mixed with the lower-level
SSD-to-pinned latency figure.
