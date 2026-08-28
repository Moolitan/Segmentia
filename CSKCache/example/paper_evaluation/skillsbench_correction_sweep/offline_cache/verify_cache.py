#!/usr/bin/env python3
"""Verify all frozen SkillsBench versions in the dedicated CSKCache Catalog."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import time
import uuid

from transformers import AutoConfig

import config as cfg
from inventory import REPOSITORY_ROOT, build_inventory


sys.path.insert(0, str(REPOSITORY_ROOT / "CSKCache"))

from cskcache import (  # noqa: E402
    MetadataManager,
    fingerprint_model,
    fingerprint_tokenizer,
)


def atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    summary, variants = build_inventory()
    pool_root = (cfg.POOL_ROOT / cfg.POOL_MODEL_DIR).resolve()
    catalog_path = pool_root / "raw" / "catalog.json"
    if not catalog_path.is_file():
        raise FileNotFoundError(f"Catalog is missing: {catalog_path}")

    model_config = AutoConfig.from_pretrained(
        cfg.MODEL_PATH,
        local_files_only=True,
        trust_remote_code=True,
    )
    expected_layers = int(model_config.num_hidden_layers)
    manager = MetadataManager(catalog_path, expected_layers=expected_layers)
    objects = manager.list_objects()
    model_fingerprint = fingerprint_model(cfg.MODEL_PATH.resolve())
    tokenizer_fingerprint = fingerprint_tokenizer(cfg.MODEL_PATH.resolve())
    actual = {(item.skill_name, item.skill_version): item for item in objects}
    expected = {(item.skill_name, item.skill_version): item for item in variants}
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    if missing or unexpected:
        raise RuntimeError(
            f"Catalog identity mismatch: missing={missing[:8]} "
            f"unexpected={unexpected[:8]}"
        )

    manifest_objects: list[dict[str, object]] = []
    for key in sorted(expected):
        planned = expected[key]
        cached = actual[key]
        expected_object_id = (
            f"{planned.skill_name}:{planned.token_ids_sha256[:16]}:"
            f"{model_fingerprint[:16]}"
        )
        checks = {
            "object_id": (cached.object_id, expected_object_id),
            "token_ids_sha256": (
                cached.token_ids_sha256,
                planned.token_ids_sha256,
            ),
            "token_count": (cached.token_count, planned.token_count),
            "model_fingerprint": (cached.model_fingerprint, model_fingerprint),
            "tokenizer_fingerprint": (
                cached.tokenizer_fingerprint,
                tokenizer_fingerprint,
            ),
            "layer_count": (len(cached.layers), expected_layers),
        }
        failed = {
            name: values for name, values in checks.items() if values[0] != values[1]
        }
        wrong_lengths = [
            layer.layer_id
            for layer in cached.layers
            if layer.length_bytes != planned.layer_bytes
        ]
        if failed or wrong_lengths:
            raise RuntimeError(
                f"invalid object {cached.object_id}: checks={failed} "
                f"wrong_layer_lengths={wrong_lengths}"
            )
        manifest_objects.append(
            {
                "object_id": cached.object_id,
                "skill_name": cached.skill_name,
                "skill_version": cached.skill_version,
                "token_ids_sha256": cached.token_ids_sha256,
                "token_count": cached.token_count,
                "layer_bytes": planned.layer_bytes,
                "cache_id": planned.cache_id,
                "representative_path": planned.representative_path,
                "task_ids": list(planned.task_ids),
                "source_paths": list(planned.source_paths),
            }
        )

    containers = manager.list_containers()
    if len(containers) != 1:
        raise RuntimeError(f"expected one raw container, found {len(containers)}")
    container = containers[0]
    raw_path = Path(container.raw_file_path)
    if not raw_path.is_file() or raw_path.stat().st_size != container.capacity_bytes:
        raise RuntimeError(f"raw container is missing or truncated: {raw_path}")
    raw_stat = raw_path.stat()
    payload = {
        **summary,
        "artifact_type": "skillsbench_cskcache_verified_manifest",
        "status": "verified",
        "verified_at_unix_ns": time.time_ns(),
        "catalog": str(catalog_path),
        "catalog_sha256": hashlib.sha256(catalog_path.read_bytes()).hexdigest(),
        "raw_file": str(raw_path),
        "raw_logical_bytes": raw_stat.st_size,
        "raw_allocated_bytes": raw_stat.st_blocks * 512,
        "container_id": container.container_id,
        "storage_generation": container.storage_generation,
        "model_fingerprint": model_fingerprint,
        "tokenizer_fingerprint": tokenizer_fingerprint,
        "verified_object_count": len(manifest_objects),
        "objects": manifest_objects,
    }
    manifest_path = pool_root / "skillsbench_manifest.json"
    atomic_write_json(manifest_path, payload)
    print(
        "[verified] "
        f"objects={len(manifest_objects)} "
        f"catalog={catalog_path} manifest={manifest_path} "
        f"allocated_gib={raw_stat.st_blocks * 512 / 1024**3:.3f}"
    )


if __name__ == "__main__":
    main()
