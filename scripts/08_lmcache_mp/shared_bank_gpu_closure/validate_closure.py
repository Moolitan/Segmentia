#!/usr/bin/env python3
"""Validate the real owner-to-READY-follower GPU closure."""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

from capture_common import atomic_write_json


def load_completed(path: Path, role: str) -> dict[str, Any]:
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("status") != "completed" or record.get("role") != role:
        raise ValueError(f"invalid completed {role} record: {path}")
    response = record.get("response")
    if not isinstance(response, dict) or not isinstance(response.get("choices"), list):
        raise ValueError(f"{role} has no completion choices")
    if not isinstance(record.get("response_id"), str):
        raise ValueError(f"{role} has no response ID")
    return record


def structured_events(log_text: str, marker: str) -> list[dict[str, Any]]:
    events = []
    needle = f"{marker} "
    decoder = json.JSONDecoder()
    for line in log_text.splitlines():
        if needle not in line:
            continue
        try:
            # LMCache's console formatter can append ANSI-styled source
            # information after the structured JSON object. Decode exactly
            # the first JSON value instead of requiring the rest of the log
            # line to be valid JSON too.
            payload, _ = decoder.raw_decode(line.split(needle, 1)[1].lstrip())
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            events.append(payload)
    return events


def request_matches(event_request_id: object, response_id: str) -> bool:
    """Match vLLM's per-prompt ID derived from an OpenAI response ID."""
    return isinstance(event_request_id, str) and (
        event_request_id == response_id
        or event_request_id.startswith(f"{response_id}-")
        or event_request_id.startswith(f"{response_id}:")
    )


def exactly_one(
    events: list[dict[str, Any]], event: str, response_id: str
) -> dict[str, Any]:
    matches = [
        row
        for row in events
        if row.get("event") == event
        and request_matches(row.get("request_id"), response_id)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one event={event} response_id_prefix={response_id}, "
            f"got {len(matches)}"
        )
    return matches[0]


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Shared Skill GPU closure",
        "",
        f"- Gate: **{summary['gate']}**",
        f"- Case: `{summary['case_id']}`",
        f"- Skill: `{summary['skill']}`",
        f"- Shared tokens: {summary['shared_tokens']}",
        f"- Shared blocks: {summary['shared_blocks']}",
        f"- Owner H2D tokens/layer: {summary['owner_h2d_tokens']}",
        f"- Follower calibration H2D tokens/layer: {summary['follower_h2d_tokens']}",
        f"- Follower finite μ layers: {summary['follower_finite_layers']}",
        f"- Follower commit layers: {summary['follower_commit_layers']}",
        "",
        "The follower uses the same READY Bank key and physical block IDs as the "
        "owner. Its prefix is a deterministic perturbation of the frozen target "
        "trace; the real Skill tokens and their absolute target span are unchanged.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layers", type=int, default=40)
    args = parser.parse_args()

    source = load_completed(args.run_dir / "source" / "request.json", "source")
    owner = load_completed(args.run_dir / "target" / "owner.json", "owner")
    follower = load_completed(
        args.run_dir / "target" / "follower.json", "follower"
    )
    if not (
        source["skill_sha256"]
        == owner["skill_sha256"]
        == follower["skill_sha256"]
    ):
        raise ValueError("source/owner/follower Skill hashes differ")
    if owner["segment_token_ids"] != follower["segment_token_ids"]:
        raise ValueError("owner/follower Skill segments differ")
    if (owner["segment_start"], owner["segment_end"]) != (
        follower["segment_start"], follower["segment_end"]
    ):
        raise ValueError("owner/follower Skill spans differ")
    if owner["prefix_sha256"] == follower["prefix_sha256"]:
        raise ValueError("owner/follower prefixes are identical")

    target_log_path = args.run_dir / "target" / "vllm.log"
    target_log = target_log_path.read_text(encoding="utf-8", errors="replace")
    if (
        "EngineCore encountered an issue" in target_log
        or "Traceback (most recent call last)" in target_log
    ):
        raise ValueError("target log contains an EngineCore error or traceback")
    match = re.search(
        r"Local disk rehydration complete: recovered_groups=(\d+) "
        r"recovered_layers=(\d+) recovered_bytes=(\d+)",
        target_log,
    )
    if match is None or int(match.group(1)) < 1 or int(match.group(2)) < args.layers:
        raise ValueError("target did not rehydrate a complete SSD Skill group")

    scheduler_events = structured_events(target_log, "SEGMENTIA_EVENT")
    profile_events = structured_events(target_log, "SEGMENTIA_PROFILE_EVENT")
    owner_id = owner["response_id"]
    follower_id = follower["response_id"]
    owner_activate = exactly_one(
        scheduler_events, "segmentia_shared_bank_activate", owner_id
    )
    publish = exactly_one(
        scheduler_events, "segmentia_shared_bank_publish", owner_id
    )
    follower_activate = exactly_one(
        scheduler_events, "segmentia_shared_bank_activate", follower_id
    )
    if owner_activate.get("activation_mode") != "owner_load":
        raise ValueError("first target request was not the Bank owner")
    if not publish.get("success") or publish.get("bank_state") != "ready":
        raise ValueError("owner did not publish the Bank as READY")
    if follower_activate.get("activation_mode") != "follower_correction_only":
        raise ValueError("second target request was not a correction-only follower")
    if follower_activate.get("bank_state") != "ready":
        raise ValueError("follower did not read a READY Bank")
    for field in ("bank_token_hash", "shared_block_ids"):
        if owner_activate.get(field) != follower_activate.get(field):
            raise ValueError(f"owner/follower differ in {field}")

    correction = exactly_one(
        profile_events, "segmentia_correction_only_complete", follower_id
    )
    if correction.get("processed_layers") != args.layers:
        raise ValueError("follower did not process every correction layer")
    if correction.get("finite_layers") != args.layers:
        raise ValueError("follower produced a non-finite μ layer")
    if correction.get("commit_layers") != 0:
        raise ValueError("follower correction-only path committed Bank KV")
    owner_commits = [
        row
        for row in profile_events
        if row.get("event") == "segmentia_commit_layer"
        and request_matches(row.get("request_id"), owner_id)
    ]
    follower_commits = [
        row
        for row in profile_events
        if row.get("event") == "segmentia_commit_layer"
        and request_matches(row.get("request_id"), follower_id)
    ]
    if len(owner_commits) != args.layers or follower_commits:
        raise ValueError(
            f"unexpected commit counts owner={len(owner_commits)} "
            f"follower={len(follower_commits)}"
        )
    owner_h2d = exactly_one(profile_events, "segmentia_h2d_breakdown", owner_id)
    follower_h2d = exactly_one(
        profile_events, "segmentia_h2d_breakdown", follower_id
    )
    calibration_tokens = int(correction["calibration_tokens"])
    if follower_h2d.get("tokens") != calibration_tokens:
        raise ValueError("follower H2D range is not calibration-only")
    if int(owner_h2d.get("tokens", 0)) <= int(follower_h2d["tokens"]):
        raise ValueError("owner H2D range is not larger than follower calibration")
    if owner_h2d.get("canonical_shared_suffix") is not True:
        raise ValueError("owner did not materialize canonical Skill-relative B1 K")
    if owner_h2d.get("correction_only") is not False:
        raise ValueError("owner H2D event was marked correction-only")
    if follower_h2d.get("canonical_shared_suffix") is not False:
        raise ValueError("follower attempted to rematerialize the canonical B1")
    if follower_h2d.get("correction_only") is not True:
        raise ValueError("follower H2D event was not marked correction-only")

    summary = {
        "schema_version": 1,
        "gate": "go",
        "case_id": owner["case_id"],
        "skill": owner["skill"],
        "bank_token_hash": owner_activate["bank_token_hash"],
        "shared_tokens": int(owner_activate["shared_end"])
        - int(owner_activate["shared_start"]),
        "shared_blocks": len(owner_activate["shared_block_ids"]),
        "owner_h2d_tokens": int(owner_h2d["tokens"]),
        "follower_h2d_tokens": int(follower_h2d["tokens"]),
        "follower_processed_layers": correction["processed_layers"],
        "follower_finite_layers": correction["finite_layers"],
        "follower_nonzero_layers": correction["nonzero_layers"],
        "follower_commit_layers": len(follower_commits),
        "rehydrated_groups": int(match.group(1)),
        "rehydrated_layers": int(match.group(2)),
        "source_request": str((args.run_dir / "source" / "request.json").resolve()),
        "owner_request": str((args.run_dir / "target" / "owner.json").resolve()),
        "follower_request": str((args.run_dir / "target" / "follower.json").resolve()),
        "target_log": str(target_log_path.resolve()),
    }
    atomic_write_json(args.run_dir / "manifest.json", summary)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "figures").mkdir(exist_ok=True)
    (args.output_dir / "tables").mkdir(exist_ok=True)
    (args.output_dir / "data").mkdir(exist_ok=True)
    atomic_write_json(args.output_dir / "data" / "summary.json", summary)
    write_summary(args.output_dir / "summary.md", summary)
    with (args.output_dir / "source_manifest.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(("artifact", "source"))
        writer.writerow(("data/summary.json", args.run_dir / "manifest.json"))
        writer.writerow(("summary.md", target_log_path.resolve()))
    print(
        f"[validated] gate=go shared_tokens={summary['shared_tokens']} "
        f"owner_h2d={summary['owner_h2d_tokens']} "
        f"follower_h2d={summary['follower_h2d_tokens']}"
    )


if __name__ == "__main__":
    main()
