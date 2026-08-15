#!/usr/bin/env python3
"""Build one source, one Bank owner, and one controlled follower request."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from capture_common import atomic_write_json


def token_hash(tokens: list[int]) -> str:
    digest = hashlib.sha256()
    for token in tokens:
        digest.update(int(token).to_bytes(4, "little", signed=False))
    return digest.hexdigest()


def load_prepared(path: Path, endpoint: str) -> dict[str, Any]:
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("status") != "prepared" or record.get("endpoint") != endpoint:
        raise ValueError(f"invalid prepared {endpoint} record: {path}")
    return record


def lookup(record: dict[str, Any]) -> dict[str, Any]:
    separator_tokens = record["effective_separator_tokens"]
    return {
        "segment_start": int(record["segment_start"]),
        "segment_end": int(record["segment_end"]),
        "correction_mode": "prefix_k_headwise",
        "cache_end": int(record["segment_end"]) - len(separator_tokens),
        "prefix_tokens": 256,
        "calibration_start": 132,
        "calibration_end": 256,
        "minimum_reuse_tokens": 256,
        "correction_alpha": 0.6,
    }


def perturb_prefix(record: dict[str, Any], variant_index: int = 0) -> list[int]:
    if variant_index < 0:
        raise ValueError("variant_index must be non-negative")
    prompt = list(record["prompt_token_ids"])
    separator_tokens = list(record["effective_separator_tokens"])
    mutable_end = int(record["segment_start"]) - len(separator_tokens)
    if mutable_end <= 0:
        raise ValueError("target prompt has no context before the Skill separator")
    forbidden = set(separator_tokens)
    for index in range(mutable_end):
        candidate = 1000 + (
            (index * 1543 + 104729 + variant_index * 7919) % 140000
        )
        if candidate in forbidden:
            candidate += 1
        prompt[index] = candidate
    if prompt == record["prompt_token_ids"]:
        raise RuntimeError("follower prefix perturbation changed no tokens")
    return prompt


def request_spec(
    *, record: dict[str, Any], prompt: list[int], role: str, model: str,
    context_variant: str
) -> dict[str, Any]:
    segment_start = int(record["segment_start"])
    segment_end = int(record["segment_end"])
    request_id = f"segmentia-shared-gpu-{role}"
    return {
        "schema_version": 1,
        "role": role,
        "request_id": request_id,
        "case_id": record["case_id"],
        "skill": record["skill"],
        "skill_sha256": record["skill_sha256"],
        "origin_endpoint": record["endpoint"],
        "context_variant": context_variant,
        "effective_separator_tokens": record["effective_separator_tokens"],
        "prompt_token_count": len(prompt),
        "prompt_token_ids": prompt,
        "prompt_sha256": token_hash(prompt),
        "prefix_sha256": token_hash(prompt[:segment_start]),
        "segment_start": segment_start,
        "segment_end": segment_end,
        "segment_token_count": segment_end - segment_start,
        "segment_token_ids": prompt[segment_start:segment_end],
        "segment_sha256": token_hash(prompt[segment_start:segment_end]),
        "request": {
            "model": model,
            "prompt": prompt,
            "max_tokens": 1,
            "temperature": 0,
            "request_id": request_id,
            "kv_transfer_params": {
                "lmcache_segmentia_lookup": lookup(record),
            },
        },
        "status": "prepared",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-record", type=Path, required=True)
    parser.add_argument("--target-record", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="Qwen3")
    args = parser.parse_args()

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
    source_skill = source["segment_token_ids"][: -len(source_separator)]
    target_skill = target["segment_token_ids"][: -len(target_separator)]
    if source_skill != target_skill:
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
    follower_spec = request_spec(
        record=target,
        prompt=perturb_prefix(target),
        role="follower",
        model=args.model,
        context_variant="deterministic_prefix_perturbation",
    )
    if owner_spec["segment_token_ids"] != follower_spec["segment_token_ids"]:
        raise RuntimeError("follower perturbation modified the Skill segment")
    if owner_spec["prefix_sha256"] == follower_spec["prefix_sha256"]:
        raise RuntimeError("owner and follower prefixes are identical")

    args.output_dir.mkdir(parents=True)
    for role, spec in (
        ("source", source_spec),
        ("owner", owner_spec),
        ("follower", follower_spec),
    ):
        atomic_write_json(args.output_dir / f"{role}.json", spec)
    atomic_write_json(
        args.output_dir / "manifest.json",
        {
            "schema_version": 1,
            "case_id": source["case_id"],
            "skill": source["skill"],
            "skill_sha256": source["skill_sha256"],
            "owner_follower_same_segment": True,
            "owner_follower_different_prefix": True,
            "owner_segment_start": owner_spec["segment_start"],
            "owner_segment_end": owner_spec["segment_end"],
            "segment_tokens": owner_spec["segment_token_count"],
            "follower_context_variant": follower_spec["context_variant"],
        },
    )
    print(
        f"[prepared] case={source['case_id']} skill={source['skill']} "
        f"owner_span=[{owner_spec['segment_start']},{owner_spec['segment_end']}) "
        f"tokens={owner_spec['segment_token_count']}"
    )


if __name__ == "__main__":
    main()
