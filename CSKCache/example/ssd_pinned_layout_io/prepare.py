#!/usr/bin/env python3
"""Materialize the four deterministic, aligned SSD layout artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import config as cfg
from common import (
    ARTIFACT_TYPE,
    LAYOUTS,
    artifact_for_layout,
    atomic_write_json,
    dataset_spec,
    iter_region_blocks,
    manifest_path,
    require_writable_data_mount,
)


def create_raw_file(path: Path, capacity_bytes: int) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.stat().st_size != capacity_bytes:
            raise RuntimeError(f"incompatible existing raw artifact: {path}")
        return False
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.posix_fallocate(fd, 0, capacity_bytes)
        os.fsync(fd)
    except BaseException:
        os.close(fd)
        path.unlink(missing_ok=True)
        raise
    else:
        os.close(fd)
    return True


def prepare_layout(layout: str) -> dict[str, Any]:
    artifact = artifact_for_layout(cfg, layout)
    raw_path = Path(artifact.raw_file)
    manifest_file = manifest_path(cfg, layout)
    created = create_raw_file(raw_path, artifact.capacity_bytes)
    expected_prefix = {
        "artifact_type": ARTIFACT_TYPE,
        "dataset": dataset_spec(cfg),
        "layout_artifact": artifact.to_dict(),
    }
    if manifest_file.exists():
        existing = json.loads(manifest_file.read_text(encoding="utf-8"))
        for key, value in expected_prefix.items():
            if existing.get(key) != value:
                raise RuntimeError(f"incompatible existing manifest: {manifest_file}")
        if existing.get("status") == "completed":
            if created:
                raise RuntimeError("completed manifest exists but raw file was missing")
            print(f"[skip] {layout}: compatible completed artifact")
            return existing
    manifest: dict[str, Any] = {
        **expected_prefix,
        "status": "preparing",
        "regions": [],
    }
    atomic_write_json(manifest_file, manifest)
    fd = os.open(raw_path, os.O_RDWR)
    try:
        records = []
        for extent in artifact.extents:
            digest = hashlib.sha256()
            cursor = 0
            for block in iter_region_blocks(
                extent,
                hidden_dim=cfg.KV_HEAD_DIM,
                dtype_bytes=cfg.DTYPE_BYTES,
                seed=cfg.DATASET_SEED,
            ):
                written = os.pwrite(fd, block, extent.offset_bytes + cursor)
                if written != len(block):
                    raise RuntimeError(f"short write in {layout} region {extent.region_id}")
                digest.update(block)
                cursor += written
            if cursor != extent.length_bytes:
                raise AssertionError("region generator emitted the wrong byte count")
            records.append(
                {
                    "region_id": extent.region_id,
                    "offset_bytes": extent.offset_bytes,
                    "length_bytes": extent.length_bytes,
                    "payload_sha256": digest.hexdigest(),
                }
            )
            manifest["regions"] = records
            atomic_write_json(manifest_file, manifest)
            print(
                f"[write] {layout} region={extent.region_id + 1}/"
                f"{len(artifact.extents)} bytes={extent.length_bytes}"
            )
        os.fsync(fd)
    finally:
        os.close(fd)
    manifest["status"] = "completed"
    atomic_write_json(manifest_file, manifest)
    return manifest


def main() -> None:
    require_writable_data_mount(cfg)
    for layout in LAYOUTS:
        prepare_layout(layout)
    print(f"Prepared four layouts under {Path(cfg.DATA_ROOT).resolve()}")


if __name__ == "__main__":
    main()
