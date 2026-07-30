#!/usr/bin/env python3
"""Build source, owner, and distinct same-Skill follower request specs."""
from __future__ import annotations

import argparse
from pathlib import Path

from capture_common import atomic_write_json
from shared_bank_gpu_closure.prepare_requests import (
    load_prepared,
    perturb_prefix,
    request_spec,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-record", type=Path, required=True)
    parser.add_argument("--target-record", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--followers", type=int, default=4)
    parser.add_argument("--model", default="Qwen3")
    args = parser.parse_args()
    if args.followers < 2:
        raise ValueError("the concurrency closure requires at least two followers")
    if args.output_dir.exists():
        raise FileExistsError(f"request output already exists: {args.output_dir}")

    source = load_prepared(args.source_record, "source")
    target = load_prepared(args.target_record, "target")
    if (source["case_id"], source["skill_sha256"]) != (
        target["case_id"], target["skill_sha256"]
    ):
        raise ValueError("source and target do not describe the same Skill case")
    source_separator = list(source["effective_separator_tokens"])
    target_separator = list(target["effective_separator_tokens"])
    if (
        source["segment_token_ids"][: -len(source_separator)]
        != target["segment_token_ids"][: -len(target_separator)]
    ):
        raise ValueError("source and target canonical Skill tokens differ")

    source_spec = request_spec(
        record=source,
        prompt=list(source["prompt_token_ids"]),
        role="source",
        model=args.model,
        context_variant="original_source_trace",
    )
    owner_spec = request_spec(
        record=target,
        prompt=list(target["prompt_token_ids"]),
        role="owner",
        model=args.model,
        context_variant="original_target_trace",
    )
    followers = []
    for follower_index in range(args.followers):
        role = f"follower-{follower_index:03d}"
        spec = request_spec(
            record=target,
            prompt=perturb_prefix(target, follower_index),
            role=role,
            model=args.model,
            context_variant=f"deterministic_prefix_perturbation_{follower_index:03d}",
        )
        if spec["segment_token_ids"] != owner_spec["segment_token_ids"]:
            raise RuntimeError(f"{role} perturbation modified the Skill segment")
        followers.append(spec)

    prefix_hashes = {owner_spec["prefix_sha256"]}
    for spec in followers:
        if spec["prefix_sha256"] in prefix_hashes:
            raise RuntimeError("owner/follower prefixes are not pairwise distinct")
        prefix_hashes.add(spec["prefix_sha256"])

    args.output_dir.mkdir(parents=True)
    atomic_write_json(args.output_dir / "source.json", source_spec)
    atomic_write_json(args.output_dir / "owner.json", owner_spec)
    for follower_index, spec in enumerate(followers):
        atomic_write_json(
            args.output_dir / f"follower-{follower_index:03d}.json", spec
        )
    atomic_write_json(
        args.output_dir / "manifest.json",
        {
            "schema_version": 1,
            "case_id": source["case_id"],
            "skill": source["skill"],
            "skill_sha256": source["skill_sha256"],
            "followers": args.followers,
            "pairwise_distinct_prefixes": True,
            "same_skill_segment": True,
            "segment_start": owner_spec["segment_start"],
            "segment_end": owner_spec["segment_end"],
            "segment_tokens": owner_spec["segment_token_count"],
            "follower_request_ids": [row["request_id"] for row in followers],
        },
    )
    print(
        f"[prepared] case={source['case_id']} followers={args.followers} "
        f"span=[{owner_spec['segment_start']},{owner_spec['segment_end']})"
    )


if __name__ == "__main__":
    main()
