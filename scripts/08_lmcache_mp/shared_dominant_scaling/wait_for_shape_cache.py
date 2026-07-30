#!/usr/bin/env python3
"""Wait for every layer of one controlled Skill object to reach SSD."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from validate_capture import inspect_layer_files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--layers", type=int, default=40)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype-bytes", type=int, default=2)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    args = parser.parse_args()
    request = json.loads(args.request.read_text(encoding="utf-8"))
    if request.get("status") != "completed":
        raise ValueError(f"request is not completed: {args.request}")
    segment_tokens = int(request["cache_end"]) - int(request["segment_start"])
    expected_bytes = (
        2 * segment_tokens * args.kv_heads * args.head_dim * args.dtype_bytes
    )
    deadline = time.monotonic() + args.timeout_s
    last_error = "cache files not inspected"
    while time.monotonic() < deadline:
        try:
            storage = inspect_layer_files(
                cache_dir=args.cache_dir,
                expected_layers=args.layers,
                expected_bytes=expected_bytes,
                expected_start=int(request["segment_start"]),
                expected_tokens=segment_tokens,
            )
        except ValueError as exc:
            last_error = str(exc)
            time.sleep(0.5)
            continue
        print(
            f"[ssd-ready] cache={args.cache_dir} layers={storage['layer_count']} "
            f"bytes_per_layer={storage['bytes_per_layer']}"
        )
        return
    raise TimeoutError(f"SSD cache incomplete after {args.timeout_s}s: {last_error}")


if __name__ == "__main__":
    main()
