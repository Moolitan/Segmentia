#!/usr/bin/env python3
"""Convert the measured Skill KV pool into one LMCache raw-block file.

This is an offline, resumable conversion. It never formats or mounts storage,
and it leaves all original per-layer files unchanged.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from common import atomic_write_json, load_config, require_mounted_device
from raw_skill_kv_common import (
    build_layout,
    discover_raw_sources,
    key_spec,
    memory_object,
    open_core,
    read_payload,
    sha256_bytes,
)


def create_raw_file(path: Path, capacity_bytes: int) -> None:
    """Create and physically allocate the dedicated raw-block file once."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.stat().st_size != capacity_bytes:
            raise RuntimeError(f"existing raw file has an unexpected size: {path}")
        return
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.posix_fallocate(descriptor, 0, capacity_bytes)
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        path.unlink(missing_ok=True)
        raise
    else:
        os.close(descriptor)


def load_or_initialize_manifest(path: Path, layout: dict[str, Any]) -> dict[str, Any]:
    """Load the conversion checkpoint or create its initial state."""
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("layout") != layout:
            raise RuntimeError(
                "raw manifest layout differs from the current configuration"
            )
        return payload
    payload = {
        "schema_version": 1,
        "status": "preparing",
        "layout": layout,
        "skills": {},
    }
    atomic_write_json(path, payload)
    return payload


def main() -> None:
    config = load_config()
    fast_ssd = config["fast_ssd_skill_cache"]
    mount = require_mounted_device(
        Path(fast_ssd["mount_point"]),
        Path(fast_ssd["expected_source"]),
        writable=True,
    )
    sources = discover_raw_sources(config)
    layout = build_layout(config, sources)
    raw_path = Path(layout["raw_file"])
    manifest_path = Path(config["raw_skill_kv"]["manifest"]).resolve()
    Path(raw_path).resolve().relative_to(Path(mount["target"]).resolve())
    manifest_path.relative_to(Path(mount["target"]).resolve())
    create_raw_file(raw_path, int(layout["capacity_bytes"]))
    state = load_or_initialize_manifest(manifest_path, layout)

    core = open_core(layout)
    try:
        for source in sources:
            encoded_keys = [key_spec(layer).encoded for layer in source.layers]
            completed = state["skills"].get(source.skill)
            if completed is not None:
                if not all(core.exists_many(encoded_keys)):
                    raise RuntimeError(
                        f"checkpoint is missing completed Skill {source.skill}"
                    )
                print(f"[skip] {source.skill} already packed")
                continue

            layer_records = []
            print(f"[pack] {source.skill} layers={len(source.layers)}")
            for layer in source.layers:
                spec = key_spec(layer)
                tensor = read_payload(layer)
                digest = sha256_bytes(memoryview(tensor.numpy()).cast("B"))
                if not core.contains_key(spec.encoded):
                    result = core.put_many([spec], [memory_object(tensor, layer)])
                    if result.results != [True]:
                        raise RuntimeError(
                            f"failed to store {source.skill}/layer-{layer.layer_id}"
                        )
                layer_records.append(
                    {
                        "layer_id": layer.layer_id,
                        "cache_key": layer.cache_key,
                        "size_bytes": layer.size_bytes,
                        "sha256": digest,
                    }
                )
            core.checkpoint_now()
            state["skills"][source.skill] = {
                "task": source.task,
                "token_count": source.token_count,
                "cache_bytes": source.cache_bytes,
                "layers": layer_records,
            }
            atomic_write_json(manifest_path, state)
            print(f"[completed] {source.skill} bytes={source.cache_bytes}")

        state["status"] = "completed"
        state["total_cache_bytes"] = sum(source.cache_bytes for source in sources)
        atomic_write_json(manifest_path, state)
    finally:
        core.close()
    print(f"[done] {raw_path}")
    print(f"[manifest] {manifest_path}")


if __name__ == "__main__":
    main()
