#!/usr/bin/env python3
"""Upgrade completed schema-v3 context-segment manifests without recomputing KV."""
from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from prefill_skill_pool import atomic_json, inspect_layer_group, token_hash
from skill_cache_tokens import (
    CACHE_OBJECT_TYPE,
    CACHE_SCHEMA_VERSION,
    LOCATOR_KIND,
    context_segment_start_marker_text,
    qwen_context_segment_start_marker_token_ids,
    qwen_context_segment_token_ids,
)


OLD_SCHEMA_VERSION = 3
DEFAULT_POOL_DIR = Path(
    "/mnt/Large_Language_Model_Lab_1/wsh/skill_save_pool/Qwen3-14B"
)
DEFAULT_MODEL_PATH = Path(
    "/mnt/Large_Language_Model_Lab_1/llm_models/Qwen3-14B/Qwen/Qwen3-14B"
)


@dataclass(frozen=True)
class ManifestUpgrade:
    path: Path
    record: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-dir", type=Path, default=DEFAULT_POOL_DIR)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--expected-layers", type=int, default=40)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Atomically replace validated manifests; default is validation only.",
    )
    return parser.parse_args()


def read_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"manifest is not a JSON object: {path}")
    return value


def prepare_upgrade(
    manifest_path: Path,
    record: dict[str, Any],
    tokenizer: Any,
    expected_layers: int,
) -> ManifestUpgrade | None:
    """Validate one v3 cache and construct its v4 metadata in memory."""
    schema = record.get("schema_version")
    if schema == CACHE_SCHEMA_VERSION:
        return None
    if schema != OLD_SCHEMA_VERSION or record.get("cache_object") != CACHE_OBJECT_TYPE:
        return None
    if record.get("status") != "completed":
        raise RuntimeError(f"v3 manifest is not completed: {manifest_path}")
    if not manifest_path.with_name("COMPLETED").is_file():
        raise RuntimeError(f"missing COMPLETED marker: {manifest_path}")

    skill_name = record.get("skill_name")
    skill_path_value = record.get("skill_path")
    if not isinstance(skill_name, str) or not skill_name:
        raise RuntimeError(f"missing skill_name: {manifest_path}")
    if not isinstance(skill_path_value, str):
        raise RuntimeError(f"missing skill_path: {manifest_path}")
    skill_path = Path(skill_path_value)
    if not skill_path.is_file():
        raise RuntimeError(f"Skill source does not exist: {skill_path}")
    if skill_path.parent.name != skill_name:
        raise RuntimeError(f"Skill name/path mismatch: {manifest_path}")

    skill_text = skill_path.read_text(encoding="utf-8")
    full_token_ids = qwen_context_segment_token_ids(
        tokenizer, skill_name, skill_text
    )
    if record.get("token_count") != len(full_token_ids):
        raise RuntimeError(f"stale token_count: {manifest_path}")
    if record.get("token_ids_sha256") != token_hash(full_token_ids):
        raise RuntimeError(f"stale token_ids_sha256: {manifest_path}")

    kv_dir = manifest_path.parent / "kv"
    sidecars = sorted(kv_dir.glob("*.pt.meta.json"))
    data_files = inspect_layer_group(
        kv_dir, sidecars, len(full_token_ids), expected_layers
    )
    recorded_files = record.get("data_files")
    if recorded_files is not None and sorted(recorded_files) != sorted(data_files):
        raise RuntimeError(f"manifest/KV file list mismatch: {manifest_path}")

    marker_token_ids = qwen_context_segment_start_marker_token_ids(
        tokenizer, skill_name
    )
    if full_token_ids[: len(marker_token_ids)] != marker_token_ids:
        raise RuntimeError(f"start marker is not a token prefix: {manifest_path}")

    upgraded = copy.deepcopy(record)
    upgraded["schema_version"] = CACHE_SCHEMA_VERSION
    upgraded["locator"] = {
        "kind": LOCATOR_KIND,
        "start_marker_text": context_segment_start_marker_text(skill_name),
        "start_marker_token_ids": marker_token_ids,
        "start_marker_token_count": len(marker_token_ids),
        "start_marker_token_ids_sha256": token_hash(marker_token_ids),
    }
    return ManifestUpgrade(manifest_path, upgraded)


def collect_upgrades(
    pool_dir: Path,
    tokenizer: Any,
    expected_layers: int,
) -> tuple[list[ManifestUpgrade], dict[int | str, int]]:
    upgrades: list[ManifestUpgrade] = []
    schemas: dict[int | str, int] = {}
    for manifest_path in sorted(pool_dir.rglob("manifest.json")):
        record = read_manifest(manifest_path)
        schema = record.get("schema_version", "missing")
        schemas[schema] = schemas.get(schema, 0) + 1
        upgrade = prepare_upgrade(
            manifest_path, record, tokenizer, expected_layers
        )
        if upgrade is not None:
            upgrades.append(upgrade)
    return upgrades, schemas


def main() -> None:
    args = parse_args()
    pool_dir = args.pool_dir.resolve()
    if not pool_dir.is_dir():
        raise FileNotFoundError(f"Skill pool does not exist: {pool_dir}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        local_files_only=True,
        trust_remote_code=True,
    )

    # Complete every read/hash/KV validation before replacing any manifest.
    upgrades, schemas = collect_upgrades(
        pool_dir, tokenizer, args.expected_layers
    )
    print(
        f"[validated] pool={pool_dir} schemas={schemas} "
        f"schema3_context_segments={len(upgrades)}"
    )
    if not args.apply:
        print("[dry-run] no manifest was changed; pass --apply to commit upgrades")
        return
    for upgrade in upgrades:
        atomic_json(upgrade.path, upgrade.record)
    print(
        f"[completed] upgraded={len(upgrades)} target_schema={CACHE_SCHEMA_VERSION}; "
        "KV files were not modified"
    )


if __name__ == "__main__":
    main()
