#!/usr/bin/env python3
"""Build the shared SSD or measure one fixed-prefix latency arm."""
from __future__ import annotations

import argparse
import json
import random
import socket
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from benchmark_common import (
    ARMS,
    DEFAULT_LENGTHS,
    atomic_write_json,
    build_prompt,
    parse_lengths,
    request_payload,
    token_sha256,
)


def post_json(url: str, api_key: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=900) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError, OSError, socket.timeout) as exc:
        raise RuntimeError(f"request failed: {exc}") from exc
    if not isinstance(body, dict):
        raise TypeError("completion endpoint did not return a JSON object")
    return body


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()


def run_one(
    *, args: argparse.Namespace, arm: str, skill_length: int,
    replica: int, nonce: int, kind: str, ordinal: int
) -> dict[str, Any]:
    prompt, segment_start, segment_end, cache_end = build_prompt(
        skill_length=skill_length,
        phase=args.phase,
        arm=arm,
        replica=replica,
        nonce=nonce,
    )
    request_id = (
        f"segmentia-fixed-prefix-{args.phase}-{arm}-r{replica}-"
        f"l{skill_length}-{kind}{ordinal}"
    )
    payload = request_payload(
        model=args.model,
        arm=arm,
        prompt=prompt,
        segment_start=segment_start,
        segment_end=segment_end,
        cache_end=cache_end,
        request_id=request_id,
        skip_save=args.phase == "measure" and arm != "full",
    )
    row: dict[str, Any] = {
        "schema_version": 1,
        "phase": args.phase,
        "arm": arm,
        "replica": replica,
        "kind": kind,
        "ordinal": ordinal,
        "skill_tokens": skill_length,
        "prompt_tokens": len(prompt),
        "segment_start": segment_start,
        "segment_end": segment_end,
        "cache_end": cache_end,
        "prompt_sha256": token_sha256(prompt),
        "skill_sha256": token_sha256(prompt[segment_start:cache_end]),
        "request_id": request_id,
    }
    if args.prepare_only:
        row["status"] = "prepared"
        return row
    started = time.perf_counter()
    response = post_json(
        f"{args.base_url.rstrip('/')}/v1/completions", args.api_key, payload
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    choices = response.get("choices") or []
    if len(choices) != 1:
        raise ValueError("completion response must contain exactly one choice")
    usage = response.get("usage") or {}
    row.update(
        {
            "status": "completed",
            "elapsed_ms": elapsed_ms,
            "response_id": response.get("id"),
            "finish_reason": choices[0].get("finish_reason"),
            "completion_tokens": usage.get("completion_tokens"),
        }
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("source", "measure"), required=True)
    parser.add_argument("--arm", choices=ARMS, default="direct")
    parser.add_argument("--lengths", default=",".join(map(str, DEFAULT_LENGTHS)))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8100")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", default="Qwen3")
    parser.add_argument("--replica", type=int, default=0)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--measurements", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--cpu-prefetch", action="store_true")
    args = parser.parse_args()
    if args.replica < 0 or args.warmups < 0 or args.measurements <= 0:
        raise ValueError("replica/warmups/measurements are out of range")
    lengths = parse_lengths(args.lengths)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timings = args.output_dir / "timings.jsonl"
    if timings.exists():
        raise FileExistsError(f"refusing to overwrite {timings}")

    rows: list[dict[str, Any]] = []
    if args.phase == "source":
        for ordinal, length in enumerate(lengths):
            rows.append(
                run_one(
                    args=args,
                    arm="direct",
                    skill_length=length,
                    replica=0,
                    nonce=ordinal,
                    kind="source",
                    ordinal=ordinal,
                )
            )
    else:
        rng = random.Random(args.seed + args.replica * 1009 + ARMS.index(args.arm))
        for kind, count in (("warmup", args.warmups), ("measure", args.measurements)):
            for ordinal in range(count):
                ordered = lengths.copy()
                rng.shuffle(ordered)
                for length in ordered:
                    # Keep the request corpus identical across arms while the
                    # arm-specific RNG may vary presentation order.
                    nonce = 1_000_000 + ordinal * 10_000 + length + (
                        0 if kind == "warmup" else 500_000
                    )
                    rows.append(
                        run_one(
                            args=args,
                            arm=args.arm,
                            skill_length=length,
                            replica=args.replica,
                            nonce=nonce,
                            kind=kind,
                            ordinal=ordinal,
                        )
                    )

    for row in rows:
        append_jsonl(timings, row)
        print(
            f"[{row['status']}] phase={row['phase']} arm={row['arm']} "
            f"length={row['skill_tokens']} kind={row['kind']}"
            + (f" elapsed_ms={row['elapsed_ms']:.3f}" if "elapsed_ms" in row else "")
        )
    manifest = {
        "schema_version": 1,
        "status": "prepared" if args.prepare_only else "completed",
        "phase": args.phase,
        "arm": "direct" if args.phase == "source" else args.arm,
        "replica": args.replica,
        "lengths": lengths,
        "warmups": 0 if args.phase == "source" else args.warmups,
        "measurements": 1 if args.phase == "source" else args.measurements,
        "seed": args.seed,
        "cpu_prefetch": bool(args.cpu_prefetch),
        "timings": str(timings.resolve()),
    }
    atomic_write_json(args.output_dir / "manifest.json", manifest)


if __name__ == "__main__":
    main()
