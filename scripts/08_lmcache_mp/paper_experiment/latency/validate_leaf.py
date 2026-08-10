#!/usr/bin/env python3
"""Reject incomplete latency leaves or reuse requests without external KV apply."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from workload import MODES, atomic_write_json


EVENT_MARKER = "SEGMENTIA_EVENT "
FORBIDDEN_EVENTS = {
    "segmentia_prefix_length_fallback",
    "segmentia_prefix_probe_unavailable",
    "segmentia_lookup_allocation_fallback",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"no timing rows in {path}")
    return rows


def read_events(path: Path) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if EVENT_MARKER not in line:
            continue
        try:
            event, _ = decoder.raw_decode(line.split(EVENT_MARKER, 1)[1].lstrip())
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def request_events(
    events: list[dict[str, Any]], row: dict[str, Any]
) -> list[dict[str, Any]]:
    response_id = row.get("response_id")
    submitted_id = row.get("request_id")
    candidates = [value for value in (response_id, submitted_id) if isinstance(value, str)]
    matched: list[dict[str, Any]] = []
    for event in events:
        event_id = event.get("request_id")
        if not isinstance(event_id, str):
            continue
        if any(
            event_id == candidate
            or event_id.startswith(f"{candidate}-")
            or event_id.startswith(f"{candidate}:")
            or candidate in event_id
            for candidate in candidates
        ):
            matched.append(event)
    return matched


def exactly_one(events: list[dict[str, Any]], name: str) -> dict[str, Any]:
    matches = [event for event in events if event.get("event") == name]
    if len(matches) != 1:
        raise ValueError(f"expected one {name}, found {len(matches)}")
    return matches[0]


def validate(args: argparse.Namespace) -> None:
    manifest_path = args.leaf / "manifest.json"
    timings_path = args.leaf / "timings.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed" or manifest.get("mode") != args.mode:
        raise ValueError(f"invalid completed manifest for mode={args.mode}")
    rows = read_jsonl(timings_path)
    expected = 1 + int(manifest["warmups"]) + int(manifest["measurements"])
    if len(rows) != expected:
        raise ValueError(f"expected {expected} rows, found {len(rows)}")

    observed_samples: set[str] = set()
    observed_first_hashes: set[str] = set()
    for row in rows:
        if row.get("mode") != args.mode or row.get("status") != "completed":
            raise ValueError("timing leaf contains a wrong-mode or incomplete row")
        if not isinstance(row.get("elapsed_ms"), (int, float)) or row["elapsed_ms"] <= 0:
            raise ValueError("timing leaf contains an invalid latency")
        sample_id = str(row["sample_id"])
        if sample_id in observed_samples:
            raise ValueError(f"duplicate sample: {sample_id}")
        observed_samples.add(sample_id)
        observed_first_hashes.add(str(row["prefix_sha256"]))
        if row["skill_sha256"] != manifest["skill_token_ids_sha256"]:
            raise ValueError("request Skill tokens disagree with offline manifest")
    if len(observed_first_hashes) != expected:
        raise ValueError("dynamic requests reused an identical prefix")

    all_events = read_events(args.log)
    external_counts: list[int] = []
    lookup_cursors: list[int] = []
    for row in rows:
        current = request_events(all_events, row)
        forbidden = [
            event for event in current if event.get("event") in FORBIDDEN_EVENTS
        ]
        if forbidden:
            raise ValueError(f"request entered a forbidden fallback path: {forbidden}")
        if args.mode == "full":
            if current:
                raise ValueError(f"full request unexpectedly emitted Segmentia events: {current}")
            continue
        apply = exactly_one(current, "segmentia_lookup_external_apply")
        matched_end = int(apply.get("matched_end", -1))
        cursor = int(apply.get("lookup_cursor", -1))
        external = int(apply.get("external_tokens_applied", -1))
        if matched_end != int(row["segment_end"]):
            raise ValueError(f"external lookup did not reach Skill end: {apply}")
        if external <= 0 or external != matched_end - cursor:
            raise ValueError(f"external token accounting is invalid: {apply}")
        if cursor < int(row["segment_start"]) or cursor >= matched_end:
            raise ValueError(f"external lookup cursor is outside the Skill: {apply}")
        expected_cursor = int(row["segment_start"]) + (
            256 if args.mode == "correction" else 0
        )
        if cursor != expected_cursor:
            raise ValueError(
                f"unexpected aligned lookup cursor: expected={expected_cursor} event={apply}"
            )
        external_counts.append(external)
        lookup_cursors.append(cursor)

    validation = {
        "schema_version": 1,
        "status": "valid",
        "mode": args.mode,
        "rows": len(rows),
        "measured_rows": sum(row["kind"] == "measure" for row in rows),
        "external_apply_rows": len(external_counts),
        "external_tokens_min": min(external_counts) if external_counts else 0,
        "external_tokens_max": max(external_counts) if external_counts else 0,
        "lookup_cursor_min": min(lookup_cursors) if lookup_cursors else None,
        "lookup_cursor_max": max(lookup_cursors) if lookup_cursors else None,
    }
    atomic_write_json(args.leaf / "validation.json", validation)
    print(
        f"[leaf-valid] mode={args.mode} rows={len(rows)} "
        f"external={len(external_counts)} token_range="
        f"[{validation['external_tokens_min']},{validation['external_tokens_max']}]"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leaf", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    args = parser.parse_args()
    validate(args)


if __name__ == "__main__":
    main()
