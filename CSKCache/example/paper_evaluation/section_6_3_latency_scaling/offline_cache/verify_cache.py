#!/usr/bin/env python3
"""Verify the dedicated six-object pool and publish its frozen manifest."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import time

from transformers import AutoConfig

import config as cfg
from prepare_sources import atomic_write_json, build_plan, REPOSITORY_ROOT


sys.path.insert(0, str(REPOSITORY_ROOT / "CSKCache"))

from cskcache import MetadataManager, fingerprint_model, fingerprint_tokenizer  # noqa: E402


def main() -> None:
    plan = build_plan()
    pool_root = (cfg.POOL_ROOT / cfg.POOL_MODEL_DIR).resolve()
    catalog_path = pool_root / "raw" / "catalog.json"
    if not catalog_path.is_file():
        raise FileNotFoundError(f"Catalog is missing: {catalog_path}")
    model_config = AutoConfig.from_pretrained(
        cfg.MODEL_PATH, local_files_only=True, trust_remote_code=True
    )
    expected_layers = int(model_config.num_hidden_layers)
    if expected_layers != cfg.EXPECTED_LAYERS:
        raise RuntimeError("configured layer count differs from the model")
    manager = MetadataManager(catalog_path, expected_layers=expected_layers)
    objects = manager.list_objects()
    expected = {
        (str(row["skill_name"]), str(row["skill_version"])): row
        for row in plan["workloads"]
    }
    actual = {(item.skill_name, item.skill_version): item for item in objects}
    if set(expected) != set(actual):
        raise RuntimeError(
            f"Catalog identity mismatch: missing={sorted(set(expected)-set(actual))} "
            f"unexpected={sorted(set(actual)-set(expected))}"
        )
    model_fingerprint = fingerprint_model(cfg.MODEL_PATH.resolve())
    tokenizer_fingerprint = fingerprint_tokenizer(cfg.MODEL_PATH.resolve())
    workloads: list[dict[str, object]] = []
    for key, planned in sorted(expected.items(), key=lambda pair: int(pair[1]["selection_order"])):
        cached = actual[key]
        expected_object_id = (
            f"{cached.skill_name}:{str(planned['skill_token_ids_sha256'])[:16]}:"
            f"{model_fingerprint[:16]}"
        )
        checks = {
            "object_id": (cached.object_id, expected_object_id),
            "token_ids_sha256": (
                cached.token_ids_sha256, planned["skill_token_ids_sha256"]
            ),
            "token_count": (cached.token_count, planned["skill_tokens"]),
            "model_fingerprint": (cached.model_fingerprint, model_fingerprint),
            "tokenizer_fingerprint": (
                cached.tokenizer_fingerprint, tokenizer_fingerprint
            ),
            "layer_count": (len(cached.layers), expected_layers),
        }
        failed = {name: value for name, value in checks.items() if value[0] != value[1]}
        if failed:
            raise RuntimeError(f"invalid object {cached.object_id}: {failed}")
        workloads.append(
            {
                **planned,
                "object_id": cached.object_id,
            }
        )
    containers = manager.list_containers()
    if len(containers) != 1:
        raise RuntimeError(f"expected one raw container, found {len(containers)}")
    container = containers[0]
    if container.container_id != cfg.RAW_CONTAINER_ID:
        raise RuntimeError("raw container identity differs from the frozen config")
    if container.capacity_bytes != cfg.RAW_CAPACITY_BYTES:
        raise RuntimeError("raw container capacity differs from the frozen config")
    raw_path = Path(container.raw_file_path)
    if not raw_path.is_file() or raw_path.stat().st_size != container.capacity_bytes:
        raise RuntimeError(f"raw container is missing or truncated: {raw_path}")
    catalog_sha256 = hashlib.sha256(catalog_path.read_bytes()).hexdigest()
    payload = {
        "schema_version": 1,
        "artifact_type": "fixed_length_cskcache_verified_manifest",
        "status": "verified",
        "verified_at_unix_ns": time.time_ns(),
        "model_id": "Qwen3-14B",
        "model_path": str(cfg.MODEL_PATH.resolve()),
        "catalog": str(catalog_path),
        "catalog_sha256": catalog_sha256,
        "container_id": container.container_id,
        "raw_file": str(raw_path),
        "raw_logical_bytes": raw_path.stat().st_size,
        "raw_allocated_bytes": raw_path.stat().st_blocks * 512,
        "model_fingerprint": model_fingerprint,
        "tokenizer_fingerprint": tokenizer_fingerprint,
        "workload_count": len(workloads),
        "workloads": workloads,
    }
    manifest_path = pool_root / "fixed_length_manifest.json"
    atomic_write_json(manifest_path, payload)
    print(f"[verified] workloads=6 catalog={catalog_path} manifest={manifest_path}")


if __name__ == "__main__":
    main()
