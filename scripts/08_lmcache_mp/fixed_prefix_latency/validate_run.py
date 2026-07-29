#!/usr/bin/env python3
"""Validate SSD completeness and per-request Segmentia control flow."""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from benchmark_common import PREFIX_ARMS, REUSE_ARMS, parse_lengths


EVENT_MARKER = "SEGMENTIA_EVENT "
PROFILE_MARKER = "SEGMENTIA_PROFILE_EVENT "
LAYER_RE = re.compile(r"^(?P<base>.+)@(?P<layer>\d+)\.pt$")


def jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"no records in {path}")
    return rows


def events(path: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    decoder = json.JSONDecoder()
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if EVENT_MARKER not in line:
            continue
        try:
            event, _ = decoder.raw_decode(line.split(EVENT_MARKER, 1)[1].lstrip())
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            result.append(event)
    return result


def profile_events(path: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    decoder = json.JSONDecoder()
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if PROFILE_MARKER not in line:
            continue
        try:
            event, _ = decoder.raw_decode(line.split(PROFILE_MARKER, 1)[1].lstrip())
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            result.append(event)
    return result


def request_events(all_events: list[dict[str, Any]], response_id: str) -> list[dict[str, Any]]:
    return [
        event
        for event in all_events
        if isinstance(event.get("request_id"), str)
        and (
            event["request_id"] == response_id
            or event["request_id"].startswith(f"{response_id}-")
            or event["request_id"].startswith(f"{response_id}:")
        )
    ]


def inspect_ssd(cache_dir: Path, lengths: list[int], layers: int) -> dict[str, Any]:
    groups: dict[str, dict[int, tuple[Path, dict[str, Any]]]] = defaultdict(dict)
    sidecars = sorted(cache_dir.glob("*.pt.meta.json"))
    for sidecar in sidecars:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        data_name = payload.get("data_file")
        if not isinstance(data_name, str):
            raise ValueError(f"invalid data_file in {sidecar}")
        match = LAYER_RE.fullmatch(data_name)
        if match is None:
            raise ValueError(f"unrecognized layer data file: {data_name}")
        data = cache_dir / data_name
        if not data.is_file() or data.stat().st_size != int(payload.get("size", -1)):
            raise ValueError(f"missing or size-mismatched data for {sidecar}")
        layer = int(match.group("layer"))
        base = match.group("base")
        if layer in groups[base]:
            raise ValueError(f"duplicate layer {layer} in SSD group {base}")
        groups[base][layer] = (data, payload)
    if len(groups) != len(lengths):
        raise ValueError(f"expected {len(lengths)} SSD groups, found {len(groups)}")
    observed_lengths: list[int] = []
    for base, entries in groups.items():
        if set(entries) != set(range(layers)):
            raise ValueError(f"incomplete layer group {base}: layers={sorted(entries)}")
        group_lengths: set[int] = set()
        for _, payload in entries.values():
            shape = payload.get("shape")
            if not isinstance(shape, list) or len(shape) < 2:
                raise ValueError(f"invalid shape in SSD group {base}")
            group_lengths.add(int(shape[1]))
            cached_positions = payload.get("cached_positions")
            if not isinstance(cached_positions, dict):
                raise ValueError(f"missing cached positions in SSD group {base}")
            if cached_positions.get("kind") == "range":
                if int(cached_positions.get("length", -1)) != int(shape[1]):
                    raise ValueError(f"cached-position length mismatch in {base}")
            elif cached_positions.get("kind") == "list":
                if len(cached_positions.get("values", [])) != int(shape[1]):
                    raise ValueError(f"cached-position list mismatch in {base}")
            else:
                raise ValueError(f"unsupported cached-position encoding in {base}")
        if len(group_lengths) != 1:
            raise ValueError(f"layers disagree on token length in {base}")
        observed_lengths.append(group_lengths.pop())
    if sorted(observed_lengths) != sorted(lengths):
        raise ValueError(
            f"SSD token lengths mismatch: expected={sorted(lengths)} observed={sorted(observed_lengths)}"
        )
    return {
        "groups": len(groups),
        "layers": sum(len(entries) for entries in groups.values()),
        "token_lengths": sorted(observed_lengths),
    }


def validate_ssd(args: argparse.Namespace) -> None:
    lengths = parse_lengths(args.lengths)
    deadline = time.monotonic() + args.timeout_s
    error = "not inspected"
    while True:
        try:
            result = inspect_ssd(args.cache_dir, lengths, args.layers)
            print(f"[ssd-valid] groups={result['groups']} layers={result['layers']} lengths={result['token_lengths']}")
            return
        except ValueError as exc:
            error = str(exc)
        if time.monotonic() >= deadline:
            raise TimeoutError(f"SSD validation timed out: {error}")
        time.sleep(args.poll_s)


def one(events_for_request: list[dict[str, Any]], name: str) -> dict[str, Any]:
    matches = [event for event in events_for_request if event.get("event") == name]
    if len(matches) != 1:
        raise ValueError(f"expected one {name}, got {len(matches)}")
    return matches[0]


def validate_leaf(args: argparse.Namespace) -> None:
    manifest = json.loads((args.leaf / "manifest.json").read_text(encoding="utf-8"))
    arm = manifest["arm"]
    if arm != args.arm or manifest.get("status") != "completed":
        raise ValueError(f"invalid leaf manifest for arm={args.arm}")
    rows = jsonl(args.leaf / "timings.jsonl")
    measured = [row for row in rows if row["kind"] == "measure"]
    lengths = [int(value) for value in manifest["lengths"]]
    expected = len(lengths) * int(manifest["measurements"])
    if len(measured) != expected:
        raise ValueError(f"expected {expected} measured rows, got {len(measured)}")
    counts: dict[int, int] = defaultdict(int)
    for row in measured:
        if row.get("status") != "completed" or not isinstance(row.get("elapsed_ms"), (int, float)):
            raise ValueError("leaf contains an incomplete timing row")
        counts[int(row["skill_tokens"])] += 1
    if counts != {length: int(manifest["measurements"]) for length in lengths}:
        raise ValueError(f"per-length measurement counts mismatch: {dict(counts)}")

    all_events = events(args.log)
    all_profile_events = profile_events(args.log)
    cpu_prefetch = bool(manifest.get("cpu_prefetch", False))
    if cpu_prefetch and not all_profile_events:
        raise ValueError("CPU-prefetch leaf has no SEGMENTIA_PROFILE_EVENT records")
    fallback_rows = 0
    external_rows = 0
    for row in rows:
        response_id = row.get("response_id")
        if not isinstance(response_id, str) or not response_id:
            raise ValueError("completed timing row is missing response_id")
        current = request_events(all_events, response_id)
        current_profile = request_events(all_profile_events, response_id)
        if arm == "full":
            if current:
                raise ValueError(f"Full request unexpectedly emitted Segmentia events: {current}")
            continue
        length = int(row["skill_tokens"])
        if arm in PREFIX_ARMS:
            fallback = [
                event
                for event in current
                if event.get("event") == "segmentia_prefix_length_fallback"
            ]
            if fallback:
                if len(fallback) != 1:
                    raise ValueError(f"multiple length fallbacks for length={length}")
                event = fallback[0]
                cursor = int(event["lookup_cursor"])
                expected_reusable = max(int(row["cache_end"]) - cursor, 0)
                aligned_prefix = cursor - int(row["segment_start"])
                if (
                    event.get("phase") != "full_local"
                    or event.get("reusable_tokens") != expected_reusable
                    or expected_reusable >= 256
                    or aligned_prefix < 256
                ):
                    raise ValueError(f"invalid length fallback for length={length}: {event}")
                fallback_rows += 1
                if cpu_prefetch and any(
                    event.get("event", "").startswith("segmentia_cpu_")
                    for event in current_profile
                ):
                    raise ValueError(
                        f"length fallback unexpectedly prefetched CPU KV for length={length}"
                    )
                continue
        apply = one(current, "segmentia_lookup_external_apply")
        cursor = int(apply["lookup_cursor"])
        expected_external = int(row["cache_end"]) - cursor
        if arm in PREFIX_ARMS and expected_external < 256:
            raise ValueError(f"prefix arm loaded an under-length suffix for length={length}")
        if apply.get("matched_end") != int(row["cache_end"]):
            raise ValueError(f"external match did not reach cache_end for length={length}")
        if apply.get("external_tokens_applied") != expected_external:
            raise ValueError(
                f"external token count mismatch for length={length}: event={apply} expected={expected_external}"
            )
        external_rows += 1
        if cpu_prefetch:
            is_first_for_length = row["kind"] == "warmup" and row["ordinal"] == 0
            expected_probe_event = (
                "segmentia_cpu_prefetch_complete"
                if is_first_for_length
                else "segmentia_cpu_cache_hit"
            )
            probe = one(current_profile, expected_probe_event)
            activate = one(current_profile, "segmentia_cpu_activate")
            expected_source = "ssd" if is_first_for_length else "cpu"
            if activate.get("source_tier") != expected_source:
                raise ValueError(
                    f"CPU activation source mismatch for length={length}: {activate}"
                )
            cpu_reads = [
                event
                for event in current_profile
                if event.get("event") == "segmentia_storage_read"
                and event.get("storage_tier") == "cpu"
            ]
            ssd_reads = [
                event
                for event in current_profile
                if event.get("event") == "segmentia_storage_read"
                and event.get("storage_tier") == "ssd"
            ]
            if len(cpu_reads) != args.layers or ssd_reads:
                raise ValueError(
                    f"worker retrieval tier mismatch for length={length}: "
                    f"cpu_reads={len(cpu_reads)} ssd_reads={len(ssd_reads)}"
                )
            if probe.get("matched_tokens") != length:
                raise ValueError(
                    f"CPU probe token count mismatch for length={length}: {probe}"
                )
    print(
        f"[leaf-valid] arm={arm} rows={len(rows)} measured={len(measured)} "
        f"fallback={fallback_rows} external={external_rows}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    ssd = subparsers.add_parser("ssd")
    ssd.add_argument("--cache-dir", type=Path, required=True)
    ssd.add_argument("--lengths", required=True)
    ssd.add_argument("--layers", type=int, default=40)
    ssd.add_argument("--timeout-s", type=float, default=120.0)
    ssd.add_argument("--poll-s", type=float, default=0.5)
    leaf = subparsers.add_parser("leaf")
    leaf.add_argument("--leaf", type=Path, required=True)
    leaf.add_argument("--log", type=Path, required=True)
    leaf.add_argument("--arm", choices=("full", *sorted(REUSE_ARMS)), required=True)
    leaf.add_argument("--layers", type=int, default=40)
    args = parser.parse_args()
    if args.command == "ssd":
        validate_ssd(args)
    else:
        validate_leaf(args)


if __name__ == "__main__":
    main()
