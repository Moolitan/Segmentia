#!/usr/bin/env python3
"""Validate one source -> target-reuse -> target-full capture triplet."""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from capture_common import atomic_write_json, sha256_text


EVENT_MARKER = "SEGMENTIA_EVENT "
PROFILE_MARKER = "SEGMENTIA_PROFILE_EVENT "
LAYER_FILE_RE = re.compile(r"^(?P<key>.+)@(?P<layer>\d+)\.pt$")


def load_completed(path: Path, phase: str) -> dict[str, Any]:
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("phase") != phase or record.get("status") != "completed":
        raise ValueError(
            f"Invalid request record {path}: phase={record.get('phase')!r} "
            f"status={record.get('status')!r}"
        )
    return record


def load_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    decoder = json.JSONDecoder()
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if EVENT_MARKER not in line:
            continue
        payload = line.split(EVENT_MARKER, 1)[1].lstrip()
        try:
            event, _ = decoder.raw_decode(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def load_profile_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    decoder = json.JSONDecoder()
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if PROFILE_MARKER not in line:
            continue
        payload = line.split(PROFILE_MARKER, 1)[1].lstrip()
        try:
            event, _ = decoder.raw_decode(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def external_apply_event(log_path: Path, record: dict[str, Any]) -> dict[str, Any]:
    response_id = record.get("response_id")
    if not isinstance(response_id, str) or not response_id:
        raise ValueError(f"Request record has no response_id: {record.get('phase')}")
    matches = [
        event
        for event in load_events(log_path)
        if event.get("event") == "segmentia_lookup_external_apply"
        and isinstance(event.get("request_id"), str)
        and (
            event["request_id"] == response_id
            or event["request_id"].startswith(f"{response_id}-")
        )
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one external-apply event for response_id={response_id!r} "
            f"in {log_path}, got {len(matches)}"
        )
    return matches[0]


def require_cold_miss(record: dict[str, Any], event: dict[str, Any]) -> None:
    if not (
        event.get("lookup_start") == record["segment_start"]
        and event.get("matched_end") == record["segment_start"]
        and event.get("external_tokens_applied") == 0
    ):
        raise ValueError(
            f"Phase {record['phase']} is not a verified cold miss: event={event}"
        )


def require_external_hit(record: dict[str, Any], event: dict[str, Any]) -> None:
    if not (
        event.get("lookup_start") == record["segment_start"]
        and isinstance(event.get("lookup_cursor"), int)
        and isinstance(event.get("matched_end"), int)
        and event["matched_end"] > event["lookup_cursor"]
        and isinstance(event.get("external_tokens_applied"), int)
        and event["external_tokens_applied"] > 0
    ):
        raise ValueError(
            f"Phase {record['phase']} is not a verified external hit: event={event}"
        )


def inspect_layer_files(
    *,
    cache_dir: Path,
    expected_layers: int,
    expected_bytes: int,
) -> dict[str, Any]:
    groups: dict[str, dict[int, Path]] = defaultdict(dict)
    unmatched: list[str] = []
    for path in sorted(cache_dir.glob("*.pt")):
        match = LAYER_FILE_RE.fullmatch(path.name)
        if match is None:
            unmatched.append(path.name)
            continue
        layer = int(match.group("layer"))
        key = match.group("key")
        if layer in groups[key]:
            raise ValueError(f"Duplicate layer={layer} for cache key={key!r}")
        groups[key][layer] = path
    if unmatched:
        raise ValueError(f"Unrecognized .pt files in {cache_dir}: {unmatched}")
    if len(groups) != 1:
        raise ValueError(
            f"Expected exactly one Skill cache key in {cache_dir}, got {len(groups)}"
        )
    key, layers = next(iter(groups.items()))
    expected_layer_ids = set(range(expected_layers))
    if set(layers) != expected_layer_ids:
        raise ValueError(
            f"Layer set mismatch in {cache_dir}: "
            f"missing={sorted(expected_layer_ids - set(layers))} "
            f"extra={sorted(set(layers) - expected_layer_ids)}"
        )
    sizes = {layer: path.stat().st_size for layer, path in layers.items()}
    bad_sizes = {layer: size for layer, size in sizes.items() if size != expected_bytes}
    if bad_sizes:
        raise ValueError(
            f"Unexpected KV file sizes in {cache_dir}; expected={expected_bytes}, "
            f"bad={bad_sizes}"
        )
    return {
        "cache_dir": str(cache_dir.resolve()),
        "cache_key_prefix": key,
        "layer_count": len(layers),
        "layer_ids": sorted(layers),
        "bytes_per_layer": expected_bytes,
        "files": [str(layers[layer].resolve()) for layer in sorted(layers)],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--layers", type=int, default=40)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype-bytes", type=int, default=2)
    args = parser.parse_args()

    case_dir = args.case_dir
    source = load_completed(case_dir / "source" / "request.json", "source")
    target_reuse = load_completed(
        case_dir / "target_reuse" / "request.json", "target_reuse"
    )
    target_full = load_completed(
        case_dir / "target_full" / "request.json", "target_full"
    )
    records = (source, target_reuse, target_full)
    case_ids = {record.get("case_id") for record in records}
    skills = {record.get("skill") for record in records}
    if len(case_ids) != 1 or len(skills) != 1:
        raise ValueError(f"Request identity mismatch: cases={case_ids} skills={skills}")
    if target_reuse["prompt_token_ids"] != target_full["prompt_token_ids"]:
        raise ValueError("target_reuse and target_full prompts are not identical")
    if not (
        source["segment_token_ids"]
        == target_reuse["segment_token_ids"]
        == target_full["segment_token_ids"]
    ):
        raise ValueError("Source and target Skill segment token IDs differ")
    reset_record = json.loads(
        (case_dir / "prefix_cache_reset.json").read_text(encoding="utf-8")
    )
    if reset_record != {"success": True}:
        raise ValueError(
            f"Local vLLM prefix cache was not cleanly reset: {reset_record}"
        )

    shared_log = case_dir / "shared_vllm.log"
    source_event = external_apply_event(shared_log, source)
    target_reuse_event = external_apply_event(
        shared_log, target_reuse
    )
    target_full_event = external_apply_event(
        case_dir / "target_full" / "vllm.log", target_full
    )
    require_cold_miss(source, source_event)
    require_external_hit(target_reuse, target_reuse_event)
    require_cold_miss(target_full, target_full_event)

    segment_tokens = int(source["segment_token_count"])
    if any(int(record["segment_token_count"]) != segment_tokens for record in records):
        raise ValueError("Source and target Skill segment lengths differ")
    expected_bytes = (
        2 * segment_tokens * args.kv_heads * args.head_dim * args.dtype_bytes
    )
    response_id = target_reuse["response_id"]
    ssd_read_events = [
        event
        for event in load_profile_events(shared_log)
        if event.get("event") == "segmentia_storage_read"
        and event.get("storage_tier") == "ssd"
        and isinstance(event.get("request_id"), str)
        and (
            event["request_id"] == response_id
            or event["request_id"].startswith(f"{response_id}-")
        )
    ]
    total_ssd_bytes = sum(
        int(event.get("bytes", 0)) for event in ssd_read_events
    )
    expected_total_bytes = expected_bytes * args.layers
    if len(ssd_read_events) != args.layers or total_ssd_bytes != expected_total_bytes:
        raise ValueError(
            "Target reuse was not fully served layer-wise from SSD: "
            f"events={len(ssd_read_events)}/{args.layers} "
            f"bytes={total_ssd_bytes}/{expected_total_bytes}"
        )
    shared_storage = inspect_layer_files(
        cache_dir=case_dir / "shared_ssd",
        expected_layers=args.layers,
        expected_bytes=expected_bytes,
    )
    target_full_storage = inspect_layer_files(
        cache_dir=case_dir / "target_full_ssd",
        expected_layers=args.layers,
        expected_bytes=expected_bytes,
    )
    if Path(shared_storage["cache_key_prefix"]).name != Path(
        target_full_storage["cache_key_prefix"]
    ).name:
        raise ValueError("Source and target-full LMCache key prefixes differ")

    manifest = {
        "schema_version": 1,
        "status": "completed",
        "case_id": source["case_id"],
        "skill": source["skill"],
        "skill_sha256": source["skill_sha256"],
        "source": {
            "task": source["task"],
            "turn": source["turn"],
            "invocation": source["invocation"],
            "request_path": str((case_dir / "source" / "request.json").resolve()),
            "event": source_event,
        },
        "target": {
            "task": target_reuse["task"],
            "turn": target_reuse["turn"],
            "invocation": target_reuse["invocation"],
            "reuse_request_path": str(
                (case_dir / "target_reuse" / "request.json").resolve()
            ),
            "full_request_path": str(
                (case_dir / "target_full" / "request.json").resolve()
            ),
            "reuse_event": target_reuse_event,
            "full_event": target_full_event,
        },
        "segment_token_count": segment_tokens,
        "segment_token_sha256": sha256_text(
            json.dumps(source["segment_token_ids"], separators=(",", ":"))
        ),
        "kv_shape_per_layer": [2, segment_tokens, args.kv_heads, args.head_dim],
        "dtype_bytes": args.dtype_bytes,
        "local_prefix_cache_reset": reset_record,
        "target_reuse_ssd_read": {
            "event_count": len(ssd_read_events),
            "total_bytes": total_ssd_bytes,
            "events": ssd_read_events,
        },
        "shared_storage": shared_storage,
        "target_full_storage": target_full_storage,
    }
    atomic_write_json(case_dir / "manifest.json", manifest)
    print(
        f"[validated] case={source['case_id']} skill={source['skill']} "
        f"tokens={segment_tokens} layers={args.layers}"
    )


if __name__ == "__main__":
    main()
