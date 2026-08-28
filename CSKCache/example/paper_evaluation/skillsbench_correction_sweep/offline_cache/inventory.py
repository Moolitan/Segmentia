#!/usr/bin/env python3
"""Build and validate the frozen SkillsBench offline-cache inventory."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

import torch
from transformers import AutoConfig, AutoTokenizer

import config as cfg


REPOSITORY_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "CSKCache/cskcache").is_dir()
)
sys.path.insert(0, str(REPOSITORY_ROOT / "CSKCache"))

from cskcache import build_skill_token_identity  # noqa: E402


@dataclass(frozen=True)
class SkillVariant:
    skill_name: str
    skill_version: str
    token_ids_sha256: str
    token_count: int
    layer_bytes: int
    representative_path: str
    cache_id: str
    task_ids: tuple[str, ...]
    source_paths: tuple[str, ...]


def checkout_commit() -> str:
    return subprocess.check_output(
        ["git", "-C", str(cfg.SKILLSBENCH_ROOT), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def _model_dtype(model_config) -> torch.dtype:
    configured = getattr(model_config, "torch_dtype", None)
    if isinstance(configured, torch.dtype):
        return configured
    dtype = getattr(torch, str(configured), None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"unsupported model dtype: {configured!r}")
    return dtype


def build_inventory() -> tuple[dict[str, object], tuple[SkillVariant, ...]]:
    if not cfg.TASKS_ROOT.is_dir():
        raise FileNotFoundError(f"SkillsBench tasks are missing: {cfg.TASKS_ROOT}")
    if not cfg.MODEL_PATH.is_dir():
        raise FileNotFoundError(f"model is missing: {cfg.MODEL_PATH}")
    commit = checkout_commit()
    if commit != cfg.SKILLSBENCH_COMMIT:
        raise RuntimeError(
            f"SkillsBench commit is {commit}, expected {cfg.SKILLSBENCH_COMMIT}"
        )

    paths = sorted(cfg.TASKS_ROOT.glob("*/environment/skills/*/SKILL.md"))
    grouped: dict[tuple[str, str], list[Path]] = {}
    for path in paths:
        text = path.read_text(encoding="utf-8")
        key = (path.parent.name, hashlib.sha256(text.encode("utf-8")).hexdigest())
        grouped.setdefault(key, []).append(path)

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.MODEL_PATH,
        local_files_only=True,
        trust_remote_code=True,
    )
    model_config = AutoConfig.from_pretrained(
        cfg.MODEL_PATH,
        local_files_only=True,
        trust_remote_code=True,
    )
    layers = int(model_config.num_hidden_layers)
    head_dim = int(
        getattr(
            model_config,
            "head_dim",
            int(model_config.hidden_size) // int(model_config.num_attention_heads),
        )
    )
    kv_hidden_size = int(model_config.num_key_value_heads) * head_dim
    dtype = _model_dtype(model_config)
    bytes_per_token_per_layer = 2 * kv_hidden_size * dtype.itemsize

    variants: list[SkillVariant] = []
    for (skill_name, _), occurrences in sorted(grouped.items()):
        representative = occurrences[0]
        text = representative.read_text(encoding="utf-8")
        identity = build_skill_token_identity(tokenizer, skill_name, text)
        skill_version = hashlib.sha256(
            identity.cache_text.encode("utf-8")
        ).hexdigest()
        relative_paths = tuple(
            str(path.relative_to(cfg.TASKS_ROOT)) for path in occurrences
        )
        variants.append(
            SkillVariant(
                skill_name=skill_name,
                skill_version=skill_version,
                token_ids_sha256=identity.token_ids_sha256,
                token_count=len(identity.token_ids),
                layer_bytes=len(identity.token_ids) * bytes_per_token_per_layer,
                representative_path=str(representative.resolve()),
                cache_id=str(representative.parent.relative_to(cfg.TASKS_ROOT)),
                task_ids=tuple(path.parts[0] for path in map(Path, relative_paths)),
                source_paths=relative_paths,
            )
        )

    token_counts = [variant.token_count for variant in variants]
    layer_slot_count = len(variants) * layers
    raw_metadata_reserve = cfg.RAW_METADATA_BYTES
    max_slots = (cfg.RAW_CAPACITY_BYTES - raw_metadata_reserve) // cfg.RAW_SLOT_BYTES
    metadata_payload_capacity = cfg.RAW_METADATA_BYTES // 2 - 4096
    conservative_metadata_payload = layer_slot_count * 8192 + 1024**2
    max_layer_bytes = max(variant.layer_bytes for variant in variants)
    if max_layer_bytes + 4096 > cfg.RAW_SLOT_BYTES:
        raise RuntimeError(
            "raw slot is too small: "
            f"need {max_layer_bytes + 4096}, configured {cfg.RAW_SLOT_BYTES}"
        )
    if layer_slot_count > max_slots:
        raise RuntimeError(
            f"raw container has about {max_slots} slots, needs {layer_slot_count}"
        )
    if conservative_metadata_payload > metadata_payload_capacity:
        raise RuntimeError(
            "raw metadata checkpoint is too small: "
            f"conservative estimate {conservative_metadata_payload}, "
            f"capacity {metadata_payload_capacity}"
        )

    summary: dict[str, object] = {
        "artifact_type": "skillsbench_cskcache_inventory",
        "skillsbench_root": str(cfg.SKILLSBENCH_ROOT.resolve()),
        "skillsbench_commit": commit,
        "task_count": len({path.relative_to(cfg.TASKS_ROOT).parts[0] for path in paths}),
        "skill_occurrence_count": len(paths),
        "skill_name_count": len({path.parent.name for path in paths}),
        "skill_body_version_count": len(variants),
        "total_tokens": sum(token_counts),
        "min_tokens": min(token_counts),
        "median_tokens": statistics.median(token_counts),
        "max_tokens": max(token_counts),
        "model_layers": layers,
        "kv_hidden_size": kv_hidden_size,
        "dtype": str(dtype).removeprefix("torch."),
        "exact_kv_bytes": sum(variant.layer_bytes for variant in variants) * layers,
        "raw_slot_bytes": cfg.RAW_SLOT_BYTES,
        "raw_metadata_bytes": cfg.RAW_METADATA_BYTES,
        "metadata_payload_capacity_bytes": metadata_payload_capacity,
        "conservative_metadata_payload_bytes": conservative_metadata_payload,
        "required_layer_slots": layer_slot_count,
        "available_layer_slots": max_slots,
        "raw_capacity_bytes": cfg.RAW_CAPACITY_BYTES,
        "pool_root": str((cfg.POOL_ROOT / cfg.POOL_MODEL_DIR).resolve()),
    }
    frozen_counts = {
        "task_count": cfg.EXPECTED_TASKS,
        "skill_occurrence_count": cfg.EXPECTED_OCCURRENCES,
        "skill_name_count": cfg.EXPECTED_SKILL_NAMES,
        "skill_body_version_count": cfg.EXPECTED_BODY_VERSIONS,
    }
    mismatches = {
        key: (summary[key], expected)
        for key, expected in frozen_counts.items()
        if summary[key] != expected
    }
    if mismatches:
        raise RuntimeError(f"frozen SkillsBench inventory changed: {mismatches}")
    return summary, tuple(variants)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    summary, _ = build_inventory()
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return
    print(
        "[inventory] "
        f"tasks={summary['task_count']} "
        f"occurrences={summary['skill_occurrence_count']} "
        f"names={summary['skill_name_count']} "
        f"body_versions={summary['skill_body_version_count']} "
        f"tokens={summary['total_tokens']} "
        f"kv_gib={int(summary['exact_kv_bytes']) / 1024**3:.3f} "
        f"slots={summary['required_layer_slots']}/{summary['available_layer_slots']}"
    )


if __name__ == "__main__":
    main()
