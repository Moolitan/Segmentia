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
REHYDRATION_RE = re.compile(
    r"Local disk rehydration complete: "
    r"recovered_groups=(?P<groups>\d+) "
    r"recovered_layers=(?P<layers>\d+) "
    r"recovered_bytes=(?P<bytes>\d+) "
    r"invalid_sidecars=(?P<invalid>\d+) "
    r"incomplete_groups=(?P<incomplete>\d+) "
    r"skipped_capacity_groups=(?P<capacity>\d+)"
)


def file_has_nonzero_bytes(path: Path, chunk_bytes: int = 1024 * 1024) -> bool:
    """Reject allocated-but-never-populated raw KV files without loading them."""

    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            if any(chunk):
                return True
    return False


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


def request_event(
    log_path: Path, record: dict[str, Any], event_name: str
) -> dict[str, Any]:
    response_id = record.get("response_id")
    if not isinstance(response_id, str) or not response_id:
        raise ValueError(f"Request record has no response_id: {record.get('phase')}")
    matches = [
        event
        for event in load_events(log_path)
        if event.get("event") == event_name
        and isinstance(event.get("request_id"), str)
        and (
            event["request_id"] == response_id
            or event["request_id"].startswith(f"{response_id}-")
        )
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one {event_name!r} event for response_id={response_id!r} "
            f"in {log_path}, got {len(matches)}"
        )
    return matches[0]


def require_cold_miss(record: dict[str, Any], event: dict[str, Any]) -> None:
    cursor = event.get("lookup_cursor")
    if not (
        event.get("external_tokens") == 0
        and event.get("phase") == "local_fallback"
        and isinstance(cursor, int)
        and record["segment_start"] <= cursor < record["segment_start"] + 16
        and event.get("retained_local_tokens") == cursor
    ):
        raise ValueError(
            f"Phase {record['phase']} is not a verified cold miss: event={event}"
        )


def require_external_hit(record: dict[str, Any], event: dict[str, Any]) -> None:
    cached_end = record["segment_end"] - len(record["effective_separator_tokens"])
    if not (
        event.get("lookup_start") == record["segment_start"]
        and isinstance(event.get("lookup_cursor"), int)
        and event.get("matched_end") == cached_end
        and event.get("external_tokens_applied")
        == cached_end - event["lookup_cursor"]
    ):
        raise ValueError(
            f"Phase {record['phase']} is not a verified external hit: event={event}"
        )


def inspect_layer_files(
    *,
    cache_dir: Path,
    expected_layers: int,
    expected_bytes: int,
    expected_start: int,
    expected_tokens: int,
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
    zero_layers = [
        layer for layer, path in layers.items() if not file_has_nonzero_bytes(path)
    ]
    if zero_layers:
        raise ValueError(
            f"Raw KV files contain no populated values in {cache_dir}: "
            f"layers={zero_layers}"
        )
    sidecars = {path.name for path in cache_dir.glob("*.pt.meta.json")}
    expected_sidecars = {f"{path.name}.meta.json" for path in layers.values()}
    if sidecars != expected_sidecars:
        raise ValueError(
            f"Sidecar set mismatch in {cache_dir}: "
            f"missing={sorted(expected_sidecars - sidecars)} "
            f"extra={sorted(sidecars - expected_sidecars)}"
        )
    for layer, path in layers.items():
        sidecar_path = cache_dir / f"{path.name}.meta.json"
        metadata = json.loads(sidecar_path.read_text(encoding="utf-8"))
        positions = metadata.get("cached_positions")
        if not (
            metadata.get("data_file") == path.name
            and metadata.get("size") == expected_bytes
            and metadata.get("shape") == [2, expected_tokens, 1024]
            and positions
            == {"kind": "range", "start": expected_start, "length": expected_tokens}
        ):
            raise ValueError(
                f"Layer {layer} sidecar does not describe the expected Skill KV: "
                f"{metadata}"
            )
    return {
        "cache_dir": str(cache_dir.resolve()),
        "cache_key_prefix": key,
        "layer_count": len(layers),
        "layer_ids": sorted(layers),
        "bytes_per_layer": expected_bytes,
        "files": [str(layers[layer].resolve()) for layer in sorted(layers)],
        "sidecars": [
            str((cache_dir / f"{layers[layer].name}.meta.json").resolve())
            for layer in sorted(layers)
        ],
    }


def require_rehydration_summary(
    log_path: Path,
    *,
    expected_layers: int,
    expect_recovered: bool,
) -> dict[str, int]:
    summaries = [
        {name: int(value) for name, value in match.groupdict().items()}
        for match in REHYDRATION_RE.finditer(
            log_path.read_text(encoding="utf-8", errors="replace")
        )
    ]
    if len(summaries) != 1:
        raise ValueError(
            f"Expected one rehydration summary in {log_path}, got {len(summaries)}"
        )
    summary = summaries[0]
    if summary["invalid"] or summary["incomplete"] or summary["capacity"]:
        raise ValueError(f"Rehydration rejected SSD metadata: {summary}")
    if expect_recovered:
        if summary["groups"] < 1 or summary["layers"] != expected_layers:
            raise ValueError(
                f"Expected a complete {expected_layers}-layer recovery: {summary}"
            )
    elif summary["groups"] != 0 or summary["layers"] != 0 or summary["bytes"] != 0:
        raise ValueError(f"Fresh SSD namespace unexpectedly recovered data: {summary}")
    return summary


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
    source_log = case_dir / "source" / "vllm.log"
    target_reuse_log = case_dir / "target_reuse" / "vllm.log"
    target_full_log = case_dir / "target_full" / "vllm.log"
    source_event = request_event(
        source_log, source, "segmentia_lookup_complete"
    )
    target_reuse_event = request_event(
        target_reuse_log, target_reuse, "segmentia_lookup_external_apply"
    )
    target_full_event = request_event(
        target_full_log, target_full, "segmentia_lookup_complete"
    )
    require_cold_miss(source, source_event)
    require_external_hit(target_reuse, target_reuse_event)
    require_cold_miss(target_full, target_full_event)

    segment_tokens = int(source["segment_token_count"])
    if any(int(record["segment_token_count"]) != segment_tokens for record in records):
        raise ValueError("Source and target Skill segment lengths differ")
    separator_tokens = source.get("effective_separator_tokens")
    if not isinstance(separator_tokens, list) or not separator_tokens:
        raise ValueError("Request record has no effective separator token sequence")
    if any(
        record.get("effective_separator_tokens") != separator_tokens
        for record in records
    ):
        raise ValueError("Source and target effective separators differ")
    cached_tokens = segment_tokens - len(separator_tokens)
    if cached_tokens <= 0:
        raise ValueError("Skill content is empty after removing the closing separator")
    expected_bytes = (
        2 * cached_tokens * args.kv_heads * args.head_dim * args.dtype_bytes
    )
    response_id = target_reuse["response_id"]
    ssd_read_events = [
        event
        for event in load_profile_events(target_reuse_log)
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
    rehydration = {
        "source": require_rehydration_summary(
            source_log, expected_layers=args.layers, expect_recovered=False
        ),
        "target_reuse": require_rehydration_summary(
            target_reuse_log, expected_layers=args.layers, expect_recovered=True
        ),
        "target_full": require_rehydration_summary(
            target_full_log, expected_layers=args.layers, expect_recovered=False
        ),
    }
    shared_storage = inspect_layer_files(
        cache_dir=case_dir / "shared_ssd",
        expected_layers=args.layers,
        expected_bytes=expected_bytes,
        expected_start=source["segment_start"],
        expected_tokens=cached_tokens,
    )
    target_full_storage = inspect_layer_files(
        cache_dir=case_dir / "target_full_ssd",
        expected_layers=args.layers,
        expected_bytes=expected_bytes,
        expected_start=target_full["segment_start"],
        expected_tokens=cached_tokens,
    )
    if Path(shared_storage["cache_key_prefix"]).name != Path(
        target_full_storage["cache_key_prefix"]
    ).name:
        raise ValueError("Source and target-full LMCache key prefixes differ")

    manifest = {
        "schema_version": 2,
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
        "cached_skill_token_count": cached_tokens,
        "segment_token_sha256": sha256_text(
            json.dumps(source["segment_token_ids"], separators=(",", ":"))
        ),
        "kv_shape_per_layer": [2, cached_tokens, args.kv_heads, args.head_dim],
        "dtype_bytes": args.dtype_bytes,
        "service_lifecycle": "source_stop__target_reuse_restart__target_full_fresh",
        "rehydration": rehydration,
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
