#!/usr/bin/env python3
"""Release prepared follower requests through one barrier without retries."""
from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from capture_common import atomic_write_json
from shared_bank_gpu_closure.send_request import post_json


def load_specs(spec_dir: Path, followers: int) -> list[dict[str, Any]]:
    records = []
    for index in range(followers):
        path = spec_dir / f"follower-{index:03d}.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("status") != "prepared":
            raise ValueError(f"request spec is not prepared: {path}")
        if record.get("role") != f"follower-{index:03d}":
            raise ValueError(f"unexpected follower role in {path}")
        records.append(record)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--followers", type=int, default=4)
    parser.add_argument("--base-url", default="http://127.0.0.1:8100")
    parser.add_argument("--api-key", default="EMPTY")
    args = parser.parse_args()
    if args.followers < 2:
        raise ValueError("concurrent sender requires at least two followers")
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")

    records = load_specs(args.spec_dir, args.followers)
    args.output_dir.mkdir(parents=True)
    outputs = []
    for index, record in enumerate(records):
        output = args.output_dir / f"follower-{index:03d}.json"
        record["status"] = "sending"
        atomic_write_json(output, record)
        outputs.append(output)

    barrier = threading.Barrier(args.followers + 1)

    def send_one(index: int) -> dict[str, Any]:
        record = records[index]
        output = outputs[index]
        barrier.wait()
        started = time.perf_counter()
        try:
            response = post_json(
                f"{args.base_url.rstrip('/')}/v1/completions",
                args.api_key,
                record["request"],
            )
            if "error" in response:
                raise RuntimeError(
                    f"vLLM returned an error object: {response['error']}"
                )
            record.update(
                status="completed",
                elapsed_s=round(time.perf_counter() - started, 6),
                response=response,
                response_id=response.get("id"),
            )
        except Exception as exc:
            record.update(
                status="failed",
                elapsed_s=round(time.perf_counter() - started, 6),
                error=f"{type(exc).__name__}: {exc}",
            )
        atomic_write_json(output, record)
        return record

    wall_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.followers) as pool:
        futures = [pool.submit(send_one, index) for index in range(args.followers)]
        barrier.wait()
        completed = [future.result() for future in futures]
    wall_s = round(time.perf_counter() - wall_start, 6)
    failures = [row for row in completed if row["status"] != "completed"]
    atomic_write_json(
        args.output_dir / "manifest.json",
        {
            "schema_version": 1,
            "followers": args.followers,
            "wall_s": wall_s,
            "completed": args.followers - len(failures),
            "failed": len(failures),
            "request_ids": [row["request_id"] for row in completed],
            "outputs": [str(path.resolve()) for path in outputs],
        },
    )
    if failures:
        failed_ids = ", ".join(row["request_id"] for row in failures)
        raise RuntimeError(f"concurrent requests failed without retry: {failed_ids}")
    print(f"[completed] followers={args.followers} wall_s={wall_s}")


if __name__ == "__main__":
    main()
