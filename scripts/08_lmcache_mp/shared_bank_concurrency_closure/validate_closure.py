#!/usr/bin/env python3
"""Validate one owner followed by concurrent READY-Bank followers."""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

from capture_common import atomic_write_json
from shared_bank_gpu_closure.validate_closure import (
    exactly_one,
    load_completed,
    request_matches,
    structured_events,
)


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Shared Skill 并发 GPU 闭环",
        "",
        f"- Gate: **{summary['gate']}**",
        f"- Case: `{summary['case_id']}`",
        f"- Skill: `{summary['skill']}`",
        f"- Concurrent followers: {summary['followers']}",
        f"- Shared tokens/blocks: {summary['shared_tokens']}/{summary['shared_blocks']}",
        f"- Maximum overlapping leases: {summary['max_lease_count']}",
        f"- Follower H2D tokens/layer: {summary['follower_h2d_tokens']}",
        f"- Final Bank leases/state: {summary['final_lease_count']}/{summary['final_bank_state']}",
        "",
        "所有 follower 使用同一 READY Bank key 和物理 block IDs；每个 follower "
        "仅搬运 calibration anchor，完成 40 层请求私有纠错且不提交 Bank KV。",
        "",
        "该 gate 只验证 N=4 并发正确性与租约重叠，不提供吞吐或 3× 性能结论。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--followers", type=int, default=4)
    parser.add_argument("--layers", type=int, default=40)
    args = parser.parse_args()
    if args.followers < 2:
        raise ValueError("concurrency validation requires at least two followers")

    source = load_completed(args.run_dir / "source" / "request.json", "source")
    owner = load_completed(args.run_dir / "target" / "owner.json", "owner")
    followers = [
        load_completed(
            args.run_dir / "target" / "followers" / f"follower-{index:03d}.json",
            f"follower-{index:03d}",
        )
        for index in range(args.followers)
    ]
    if any(row["skill_sha256"] != owner["skill_sha256"] for row in followers):
        raise ValueError("owner/follower Skill hashes differ")
    if source["skill_sha256"] != owner["skill_sha256"]:
        raise ValueError("source/target Skill hashes differ")
    if any(row["segment_token_ids"] != owner["segment_token_ids"] for row in followers):
        raise ValueError("owner/follower Skill segments differ")
    prefix_hashes = {owner["prefix_sha256"], *(row["prefix_sha256"] for row in followers)}
    if len(prefix_hashes) != args.followers + 1:
        raise ValueError("owner/follower prefixes are not pairwise distinct")

    target_log_path = args.run_dir / "target" / "vllm.log"
    target_log = target_log_path.read_text(encoding="utf-8", errors="replace")
    if (
        "EngineCore encountered an issue" in target_log
        or "Traceback (most recent call last)" in target_log
    ):
        raise ValueError("target log contains an EngineCore error or traceback")
    rehydration = re.search(
        r"Local disk rehydration complete: recovered_groups=(\d+) "
        r"recovered_layers=(\d+) recovered_bytes=(\d+)",
        target_log,
    )
    if (
        rehydration is None
        or int(rehydration.group(1)) < 1
        or int(rehydration.group(2)) < args.layers
    ):
        raise ValueError("target did not rehydrate a complete SSD Skill group")

    scheduler_events = structured_events(target_log, "SEGMENTIA_EVENT")
    profile_events = structured_events(target_log, "SEGMENTIA_PROFILE_EVENT")
    owner_id = owner["response_id"]
    owner_activate = exactly_one(
        scheduler_events, "segmentia_shared_bank_activate", owner_id
    )
    publish = exactly_one(
        scheduler_events, "segmentia_shared_bank_publish", owner_id
    )
    if owner_activate.get("activation_mode") != "owner_load":
        raise ValueError("target owner did not load the Bank")
    if not publish.get("success") or publish.get("bank_state") != "ready":
        raise ValueError("owner did not publish READY")
    owner_commits = [
        row
        for row in profile_events
        if row.get("event") == "segmentia_commit_layer"
        and request_matches(row.get("request_id"), owner_id)
    ]
    if len(owner_commits) != args.layers:
        raise ValueError(f"owner commit layers={len(owner_commits)}")

    reference_hash = owner_activate["bank_token_hash"]
    reference_blocks = owner_activate["shared_block_ids"]
    shared_tokens = int(owner_activate["shared_end"]) - int(
        owner_activate["shared_start"]
    )
    follower_rows = []
    lease_counts = []
    follower_h2d_tokens: set[int] = set()
    for follower in followers:
        response_id = follower["response_id"]
        activation = exactly_one(
            scheduler_events, "segmentia_shared_bank_activate", response_id
        )
        if activation.get("activation_mode") != "follower_correction_only":
            raise ValueError(f"{follower['role']} was not correction-only")
        if activation.get("bank_state") != "ready":
            raise ValueError(f"{follower['role']} did not use READY")
        if activation.get("bank_token_hash") != reference_hash:
            raise ValueError(f"{follower['role']} used a different Bank key")
        if activation.get("shared_block_ids") != reference_blocks:
            raise ValueError(f"{follower['role']} used different physical blocks")
        lease_count = activation.get("lease_count")
        if not isinstance(lease_count, int) or lease_count < 1:
            raise ValueError(f"{follower['role']} has invalid activation lease_count")
        lease_counts.append(lease_count)

        correction = exactly_one(
            profile_events, "segmentia_correction_only_complete", response_id
        )
        if correction.get("processed_layers") != args.layers:
            raise ValueError(f"{follower['role']} did not process every layer")
        if correction.get("finite_layers") != args.layers:
            raise ValueError(f"{follower['role']} produced non-finite mu")
        if correction.get("nonzero_layers") != args.layers:
            raise ValueError(f"{follower['role']} produced zero mu")
        if correction.get("commit_layers") != 0:
            raise ValueError(f"{follower['role']} committed Bank KV")
        commits = [
            row
            for row in profile_events
            if row.get("event") == "segmentia_commit_layer"
            and request_matches(row.get("request_id"), response_id)
        ]
        if commits:
            raise ValueError(f"{follower['role']} has commit events")
        h2d = exactly_one(profile_events, "segmentia_h2d_breakdown", response_id)
        calibration_tokens = int(correction["calibration_tokens"])
        if h2d.get("tokens") != calibration_tokens:
            raise ValueError(f"{follower['role']} H2D is not calibration-only")
        if h2d.get("correction_only") is not True:
            raise ValueError(f"{follower['role']} H2D lacks correction-only flag")
        if h2d.get("canonical_shared_suffix") is not False:
            raise ValueError(f"{follower['role']} rematerialized shared suffix")
        follower_h2d_tokens.add(int(h2d["tokens"]))
        release = exactly_one(
            scheduler_events, "segmentia_shared_bank_release", response_id
        )
        if release.get("bank_token_hash") != reference_hash:
            raise ValueError(f"{follower['role']} released another Bank")
        follower_rows.append(
            {
                "role": follower["role"],
                "response_id": response_id,
                "elapsed_s": follower["elapsed_s"],
                "activation_lease_count": lease_count,
                "release_lease_count": release.get("lease_count"),
            }
        )

    if len(follower_h2d_tokens) != 1:
        raise ValueError("followers used different calibration H2D lengths")
    max_lease_count = max(lease_counts)
    if max_lease_count < 2:
        raise ValueError("follower leases never overlapped")
    follower_release_events = [
        exactly_one(
            scheduler_events,
            "segmentia_shared_bank_release",
            follower["response_id"],
        )
        for follower in followers
    ]
    final_release = max(
        follower_release_events, key=lambda row: int(row["monotonic_ns"])
    )
    if final_release.get("lease_count") != 0:
        raise ValueError("Bank retained follower leases after all completions")
    if final_release.get("bank_state") != "ready":
        raise ValueError("Bank was not READY after final follower release")
    publishes = [
        row
        for row in scheduler_events
        if row.get("event") == "segmentia_shared_bank_publish"
        and row.get("bank_token_hash") == reference_hash
    ]
    if len(publishes) != 1:
        raise ValueError(f"expected one Bank publish, got {len(publishes)}")

    concurrent_manifest = json.loads(
        (args.run_dir / "target" / "followers" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    if concurrent_manifest.get("completed") != args.followers:
        raise ValueError("concurrent sender did not complete every follower")
    summary = {
        "schema_version": 1,
        "gate": "go",
        "case_id": owner["case_id"],
        "skill": owner["skill"],
        "followers": args.followers,
        "bank_token_hash": reference_hash,
        "shared_tokens": shared_tokens,
        "shared_blocks": len(reference_blocks),
        "max_lease_count": max_lease_count,
        "follower_h2d_tokens": next(iter(follower_h2d_tokens)),
        "final_lease_count": final_release["lease_count"],
        "final_bank_state": final_release["bank_state"],
        "concurrent_wall_s": concurrent_manifest["wall_s"],
        "rehydrated_groups": int(rehydration.group(1)),
        "rehydrated_layers": int(rehydration.group(2)),
        "follower_rows": follower_rows,
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
        f"[validated] gate=go followers={args.followers} "
        f"max_leases={max_lease_count} shared_blocks={len(reference_blocks)}"
    )


if __name__ == "__main__":
    main()
