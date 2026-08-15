#!/usr/bin/env python3
"""Convert the measured Skill KV pool into one LMCache raw-block file.

This is an offline, resumable conversion. It never formats or mounts storage,
and it leaves all original per-layer files unchanged.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any
import uuid


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
CSKCACHE_PACKAGE_ROOT = REPOSITORY_ROOT / "CSKCache"
if str(CSKCACHE_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(CSKCACHE_PACKAGE_ROOT))

from cskcache import (  # noqa: E402
    CacheBuilder,
    CacheObjectBuildInput,
    ContainerMetadata,
    LayerBuildInput,
    MetadataManager,
    RAW_BUILD_CHECKPOINT_TYPE,
    fingerprint_model,
    fingerprint_tokenizer,
    publish_cache_snapshot,
)

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


def create_raw_file(path: Path, capacity_bytes: int) -> bool:
    """Create and physically allocate the dedicated raw-block file once."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.stat().st_size != capacity_bytes:
            raise RuntimeError(f"existing raw file has an unexpected size: {path}")
        return False
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
    return True


def load_or_initialize_manifest(path: Path, layout: dict[str, Any]) -> dict[str, Any]:
    """Load the conversion checkpoint or create its initial state."""
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("artifact_type") != RAW_BUILD_CHECKPOINT_TYPE:
            raise RuntimeError(
                "raw build checkpoint is from an unsupported legacy format; "
                "remove it and rebuild the raw container"
            )
        if payload.get("layout") != layout:
            raise RuntimeError(
                "raw manifest layout differs from the current configuration"
            )
        return payload
    payload = {
        "artifact_type": RAW_BUILD_CHECKPOINT_TYPE,
        "status": "preparing",
        "layout": layout,
        "skills": {},
    }
    atomic_write_json(path, payload)
    return payload


def resolve_container(
    settings: dict[str, Any],
    layout: dict[str, Any],
    metadata_path: Path,
    *,
    raw_file_created: bool,
) -> ContainerMetadata:
    """Reuse a published generation or create one for a new metadata snapshot."""

    expected_layers = int(layout["expected_layers"])
    if metadata_path.is_file() and not raw_file_created:
        manager = MetadataManager(metadata_path, expected_layers=expected_layers)
        containers = manager.list_containers()
        if len(containers) != 1:
            raise RuntimeError("CSKCache metadata must contain exactly one container")
        container = containers[0]
    else:
        container = ContainerMetadata(
            container_id=str(settings["container_id"]),
            raw_file_path=str(Path(layout["raw_file"]).resolve()),
            container_format_version=int(settings["container_format_version"]),
            storage_generation=uuid.uuid4().hex,
            capacity_bytes=int(layout["capacity_bytes"]),
            alignment_bytes=int(layout["block_alignment_bytes"]),
            header_bytes=int(layout["header_bytes"]),
        )
    expected = (
        str(settings["container_id"]),
        str(Path(layout["raw_file"]).resolve()),
        int(settings["container_format_version"]),
        int(layout["capacity_bytes"]),
        int(layout["block_alignment_bytes"]),
        int(layout["header_bytes"]),
    )
    actual = (
        container.container_id,
        container.raw_file_path,
        container.container_format_version,
        container.capacity_bytes,
        container.alignment_bytes,
        container.header_bytes,
    )
    if actual != expected:
        raise RuntimeError("published CSKCache container differs from raw layout")
    return container


def publish_cskcache_metadata(
    config: dict[str, Any],
    layout: dict[str, Any],
    sources,
    state: dict[str, Any],
    core,
    *,
    raw_file_created: bool,
) -> Path:
    """Publish CSKCache's authoritative catalog after checkpointing."""

    settings = config["raw_skill_kv"]
    metadata_path = Path(settings["cskcache_metadata"]).resolve()
    container = resolve_container(
        settings,
        layout,
        metadata_path,
        raw_file_created=raw_file_created,
    )
    model_paths = {source.model_path for source in sources}
    if len(model_paths) != 1:
        raise RuntimeError(f"packed Skills use different model paths: {model_paths}")
    model_path = next(iter(model_paths))
    model_fingerprint = fingerprint_model(model_path)
    tokenizer_fingerprint = fingerprint_tokenizer(model_path)
    builder = CacheBuilder(
        core,
        container,
        expected_layers=int(layout["expected_layers"]),
    )

    objects = []
    for source in sources:
        packed = state["skills"].get(source.skill)
        if not isinstance(packed, dict):
            raise RuntimeError(f"raw manifest has no packed Skill {source.skill}")
        packed_layers = {
            int(layer["layer_id"]): layer for layer in packed.get("layers", [])
        }
        if set(packed_layers) != set(range(int(layout["expected_layers"]))):
            raise RuntimeError(f"raw manifest is incomplete for {source.skill}")
        build_layers = []
        for layer in source.layers:
            record = packed_layers[layer.layer_id]
            if str(record["cache_key"]) != layer.cache_key:
                raise RuntimeError(
                    f"raw cache key differs for {source.skill}/layer-{layer.layer_id}"
                )
            if int(record["size_bytes"]) != layer.size_bytes:
                raise RuntimeError(
                    "raw payload size differs for "
                    f"{source.skill}/layer-{layer.layer_id}"
                )
            build_layers.append(
                LayerBuildInput(
                    layer_id=layer.layer_id,
                    backend_key=layer.cache_key,
                    lookup_key=key_spec(layer).encoded,
                    length_bytes=layer.size_bytes,
                    dtype=str(layer.dtype).removeprefix("torch."),
                    shape=tuple(int(dim) for dim in layer.shape),
                    memory_layout=layer.memory_format.name,
                    payload_sha256=str(record["sha256"]),
                )
            )
        object_id = (
            f"{source.skill}:{source.token_ids_sha256[:16]}:"
            f"{model_fingerprint[:16]}"
        )
        objects.append(
            builder.build_object(
                CacheObjectBuildInput(
                    object_id=object_id,
                    skill_name=source.skill,
                    skill_version=source.skill_version,
                    model_fingerprint=model_fingerprint,
                    tokenizer_fingerprint=tokenizer_fingerprint,
                    token_count=source.token_count,
                    source_position_start=source.source_position_start,
                    token_ids_sha256=source.token_ids_sha256,
                    start_marker_token_ids=source.start_marker_token_ids,
                    layers=tuple(build_layers),
                )
            )
        )

    publish_cache_snapshot(
        metadata_path,
        container,
        objects,
        expected_layers=int(layout["expected_layers"]),
        replace_existing=raw_file_created,
    )
    return metadata_path


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
    raw_file_created = create_raw_file(raw_path, int(layout["capacity_bytes"]))
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
        cskcache_metadata = publish_cskcache_metadata(
            config,
            layout,
            sources,
            state,
            core,
            raw_file_created=raw_file_created,
        )
    finally:
        core.close()
    print(f"[done] {raw_path}")
    print(f"[manifest] {manifest_path}")
    print(f"[cskcache-metadata] {cskcache_metadata}")


if __name__ == "__main__":
    main()
