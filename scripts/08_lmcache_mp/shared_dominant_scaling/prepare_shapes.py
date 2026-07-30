#!/usr/bin/env python3
"""Build controlled long-Skill request shapes from real captured Skill tokens."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from capture_common import atomic_write_json


BLOCK_SIZE = 16
PREFIX_TOKENS = 256
CALIBRATION_TOKENS = 124
FOLLOWERS = 4
GENERATOR_VERSION = 3


@dataclass(frozen=True)
class Shape:
    name: str
    private_tokens: int
    shared_tokens: int

    @property
    def segment_start(self) -> int:
        return self.private_tokens - PREFIX_TOKENS

    @property
    def segment_end(self) -> int:
        return self.private_tokens + self.shared_tokens

    @property
    def ratio(self) -> float:
        return self.shared_tokens / self.private_tokens


SHAPES = (
    Shape("long-6k", private_tokens=2048, shared_tokens=6144),
    Shape("long-8k", private_tokens=1024, shared_tokens=8192),
)


def cycle_tokens(tokens: list[int], length: int, offset: int = 0) -> list[int]:
    if not tokens:
        raise ValueError("cannot cycle an empty token sequence")
    return [tokens[(offset + index) % len(tokens)] for index in range(length)]


def token_hash(tokens: list[int]) -> str:
    payload = b"".join(
        int(token).to_bytes(4, "little", signed=False) for token in tokens
    )
    return hashlib.sha256(payload).hexdigest()


def theoretical_kv_gain(followers: int, ratio: float) -> float:
    return followers * (1.0 + ratio) / (followers + ratio)


def lookup_config(
    shape: Shape, segment_start: int, separator_length: int
) -> dict[str, Any]:
    cache_end = segment_start + PREFIX_TOKENS + shape.shared_tokens
    return {
        "segment_start": segment_start,
        "segment_end": cache_end + separator_length,
        "cache_end": cache_end,
        "correction_mode": "prefix_k_headwise",
        "prefix_tokens": PREFIX_TOKENS,
        "calibration_start": PREFIX_TOKENS - CALIBRATION_TOKENS,
        "calibration_end": PREFIX_TOKENS,
        "minimum_reuse_tokens": 256,
    }


def request_record(
    *,
    shape: Shape,
    role: str,
    prefix: list[int],
    skill: list[int],
    suffix: list[int],
    separator_length: int,
    with_lookup: bool,
    max_tokens: int,
) -> dict[str, Any]:
    prompt = [*prefix, *skill, *suffix]
    request_id = f"segmentia-dominant-{shape.name}-{role}"
    if not with_lookup:
        request_id += "-full"
    request: dict[str, Any] = {
        "model": "Qwen3",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "request_id": request_id,
    }
    if with_lookup:
        config = lookup_config(shape, len(prefix), separator_length)
        request["kv_transfer_params"] = {
            "lmcache_segmentia_lookup": config
        }
    else:
        config = lookup_config(shape, len(prefix), separator_length)
    return {
        "schema_version": 1,
        "status": "prepared",
        "shape": shape.name,
        "role": role,
        "request_id": request_id,
        "prompt_tokens": len(prompt),
        "private_tokens": len(prefix) + PREFIX_TOKENS,
        "shared_tokens": shape.shared_tokens,
        "segment_start": len(prefix),
        "segment_end": len(prefix) + len(skill),
        "cache_end": config["cache_end"],
        "segment_token_hash": token_hash(skill),
        "request": request,
    }


def contains_subsequence(tokens: list[int], subsequence: list[int]) -> bool:
    width = len(subsequence)
    return any(
        tokens[index : index + width] == subsequence
        for index in range(len(tokens) - width + 1)
    )


def load_seed(path: Path) -> tuple[list[int], list[int], list[int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    segment = payload.get("segment_token_ids")
    prompt = payload.get("request", {}).get("prompt")
    separator = payload.get("effective_separator_tokens")
    if not isinstance(segment, list) or not segment:
        raise ValueError(f"seed has no segment_token_ids: {path}")
    if not isinstance(prompt, list) or not prompt:
        raise ValueError(f"seed has no token-id request.prompt: {path}")
    if not isinstance(separator, list) or not separator:
        raise ValueError(f"seed has no effective_separator_tokens: {path}")
    if not all(
        isinstance(token, int) and token >= 0
        for token in segment + prompt + separator
    ):
        raise ValueError("seed tokens must be non-negative integers")
    if segment[-len(separator) :] != separator:
        raise ValueError("seed segment does not end with its separator")
    content = segment[: -len(separator)]
    if not content:
        raise ValueError("seed has no Skill content before the separator")
    if contains_subsequence(content, separator):
        raise ValueError("seed Skill content contains an internal separator")
    return content, prompt, separator


def write_shape(
    output_dir: Path,
    shape: Shape,
    seed_skill: list[int],
    seed_prompt: list[int],
    separator: list[int],
    max_tokens: int,
) -> dict[str, Any]:
    shape_dir = output_dir / shape.name
    reuse_dir = shape_dir / "reuse"
    full_dir = shape_dir / "full"
    reuse_dir.mkdir(parents=True)
    full_dir.mkdir(parents=True)

    cacheable_skill_length = PREFIX_TOKENS + shape.shared_tokens
    skill_content = cycle_tokens(seed_skill, cacheable_skill_length)
    skill = [*skill_content, *separator]
    suffix = cycle_tokens(seed_prompt, 16, offset=211)

    # The source position differs by 256 tokens so the SSD object is genuinely
    # reused across positions after restart, rather than becoming an APC hit.
    source_start = shape.segment_start - 256
    if source_start < len(separator):
        raise ValueError(f"source prefix would be negative for {shape.name}")
    source_prefix = [
        *cycle_tokens(
            seed_prompt, source_start - len(separator), offset=17
        ),
        *separator,
    ]
    target_prefixes = [
        [
            *cycle_tokens(
                seed_prompt,
                shape.segment_start - len(separator),
                offset=101 + 97 * index,
            ),
            *separator,
        ]
        for index in range(FOLLOWERS + 1)
    ]
    prefix_hashes = {token_hash(prefix) for prefix in target_prefixes}
    if len(prefix_hashes) != len(target_prefixes):
        raise ValueError("controlled private prefixes are not pairwise distinct")

    source = request_record(
        shape=shape,
        role="source",
        prefix=source_prefix,
        skill=skill,
        suffix=suffix,
        separator_length=len(separator),
        with_lookup=True,
        max_tokens=max_tokens,
    )
    owner_reuse = request_record(
        shape=shape,
        role="owner",
        prefix=target_prefixes[0],
        skill=skill,
        suffix=suffix,
        separator_length=len(separator),
        with_lookup=True,
        max_tokens=max_tokens,
    )
    owner_full = request_record(
        shape=shape,
        role="owner",
        prefix=target_prefixes[0],
        skill=skill,
        suffix=suffix,
        separator_length=len(separator),
        with_lookup=False,
        max_tokens=max_tokens,
    )
    atomic_write_json(shape_dir / "source.json", source)
    atomic_write_json(reuse_dir / "owner.json", owner_reuse)
    atomic_write_json(full_dir / "owner.json", owner_full)

    for index in range(FOLLOWERS):
        role = f"follower-{index:03d}"
        reuse = request_record(
            shape=shape,
            role=role,
            prefix=target_prefixes[index + 1],
            skill=skill,
            suffix=suffix,
            separator_length=len(separator),
            with_lookup=True,
            max_tokens=max_tokens,
        )
        full = request_record(
            shape=shape,
            role=role,
            prefix=target_prefixes[index + 1],
            skill=skill,
            suffix=suffix,
            separator_length=len(separator),
            with_lookup=False,
            max_tokens=max_tokens,
        )
        atomic_write_json(reuse_dir / f"{role}.json", reuse)
        atomic_write_json(full_dir / f"{role}.json", full)

    geometry = {
        "shape": shape.name,
        "block_size": BLOCK_SIZE,
        "prefix_tokens": PREFIX_TOKENS,
        "calibration_tokens": CALIBRATION_TOKENS,
        "private_0_p_tokens": shape.private_tokens,
        "shared_b1_tokens": shape.shared_tokens,
        "segment_tokens": len(skill),
        "cacheable_skill_tokens": cacheable_skill_length,
        "separator_tokens": separator,
        "target_segment_start": shape.segment_start,
        "target_p": shape.private_tokens,
        "target_cache_end": shape.segment_end,
        "target_segment_end": shape.segment_end + len(separator),
        "source_segment_start": source_start,
        "rho_shared_over_private": shape.ratio,
        "theoretical_kv_gain_n4": theoretical_kv_gain(4, shape.ratio),
        "skill_construction": "deterministic_cycle_of_real_captured_skill_tokens",
        "natural_skill_case": False,
        "segment_token_hash": token_hash(skill),
        "cacheable_skill_token_hash": token_hash(skill_content),
    }
    atomic_write_json(shape_dir / "manifest.json", geometry)
    return geometry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=1)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")
    if args.max_tokens < 1:
        raise ValueError("max-tokens must be positive")

    seed_skill, seed_prompt, separator = load_seed(args.seed_spec)
    args.output_dir.mkdir(parents=True)
    geometries = [
        write_shape(
            args.output_dir,
            shape,
            seed_skill,
            seed_prompt,
            separator,
            args.max_tokens,
        )
        for shape in SHAPES
    ]
    manifest = {
        "schema_version": 1,
        "generator_version": GENERATOR_VERSION,
        "purpose": "controlled_shared_dominant_token_geometry_stress",
        "seed_spec": str(args.seed_spec.resolve()),
        "seed_skill_content_tokens": len(seed_skill),
        "separator_tokens": separator,
        "followers_prepared": FOLLOWERS,
        "shapes": geometries,
    }
    atomic_write_json(args.output_dir / "manifest.json", manifest)
    for geometry in geometries:
        print(
            f"[prepared] shape={geometry['shape']} "
            f"P={geometry['private_0_p_tokens']} "
            f"B1={geometry['shared_b1_tokens']} "
            f"rho={geometry['rho_shared_over_private']:.3f} "
            f"G4={geometry['theoretical_kv_gain_n4']:.3f}"
        )


if __name__ == "__main__":
    main()
