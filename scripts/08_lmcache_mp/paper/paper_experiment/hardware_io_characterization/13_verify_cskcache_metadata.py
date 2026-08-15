#!/usr/bin/env python3
"""Verify CSKCache Catalog extents against the real 990 PRO payloads.

Run 11_prepare_raw_skill_kv.py first. This script does not run an Agent or
vLLM; it opens the existing raw file, loads each Skill through StorageManager,
and compares all 40 destination buffers with the persisted payload hashes.
"""

from __future__ import annotations

from pathlib import Path
import sys
import time

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
CSKCACHE_PACKAGE_ROOT = REPOSITORY_ROOT / "CSKCache"
if str(CSKCACHE_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(CSKCACHE_PACKAGE_ROOT))

from cskcache import MetadataManager, StorageManager  # noqa: E402

from common import load_config, require_mounted_device, write_test_result  # noqa: E402
from raw_skill_kv_common import (  # noqa: E402
    build_layout,
    discover_raw_sources,
    memory_object,
    open_core,
    sha256_bytes,
)


def main() -> None:
    config = load_config()
    settings = config["raw_skill_kv"]
    fast_ssd = config["fast_ssd_skill_cache"]
    require_mounted_device(
        Path(fast_ssd["mount_point"]),
        Path(fast_ssd["expected_source"]),
        writable=False,
    )
    sources = discover_raw_sources(config)
    layout = build_layout(config, sources)
    expected_layers = int(layout["expected_layers"])
    metadata_path = Path(settings["cskcache_metadata"]).resolve()
    metadata = MetadataManager(metadata_path, expected_layers=expected_layers)
    objects_by_skill = {item.skill_name: item for item in metadata.list_objects()}
    if set(objects_by_skill) != {source.skill for source in sources}:
        raise RuntimeError("CSKCache metadata Skill set differs from packed sources")

    maximum_layer_bytes = int(layout["maximum_layer_bytes"])
    pinned = torch.empty(
        (expected_layers, maximum_layer_bytes),
        dtype=torch.uint8,
        pin_memory=True,
    )
    alignment = int(layout["block_alignment_bytes"])
    if any(pinned[layer].data_ptr() % alignment for layer in range(expected_layers)):
        raise RuntimeError("pinned verification buffers are not block aligned")

    core = open_core(layout)
    core.raw_device().register_fixed_buffers(
        [pinned[layer].data_ptr() for layer in range(expected_layers)],
        [maximum_layer_bytes] * expected_layers,
    )
    storage = StorageManager(metadata, core)
    rows = []
    try:
        for source in sources:
            cache_object = objects_by_skill[source.skill]
            if cache_object.token_count != source.token_count:
                raise RuntimeError(f"token count differs for {source.skill}")
            destinations = [
                memory_object(pinned[index, : layer.size_bytes], layer)
                for index, layer in enumerate(source.layers)
            ]
            started = time.perf_counter_ns()
            result = storage.read_object_into(cache_object.object_id, destinations)
            duration_ms = (time.perf_counter_ns() - started) / 1_000_000
            if not result.complete:
                raise RuntimeError(f"incomplete extent read for {source.skill}")
            for extent, destination in zip(
                cache_object.layers, destinations, strict=True
            ):
                actual = sha256_bytes(destination.byte_array)
                if actual != extent.payload_sha256:
                    raise RuntimeError(
                        f"payload mismatch for {source.skill}/layer-{extent.layer_id}"
                    )
            rows.append(
                {
                    "skill": source.skill,
                    "object_id": cache_object.object_id,
                    "layer_count": len(cache_object.layers),
                    "cache_bytes": source.cache_bytes,
                    "duration_ms": duration_ms,
                    "payload_sha256_verified": True,
                }
            )
            print(
                f"[verified] {source.skill} layers={len(cache_object.layers)} "
                f"duration={duration_ms:.3f}ms"
            )
    finally:
        core.close()

    output = write_test_result(
        "13_cskcache_metadata_verification",
        config,
        {
            "metadata_path": str(metadata_path),
            "raw_file": str(Path(layout["raw_file"])),
            "skill_count": len(rows),
            "extent_count": sum(row["layer_count"] for row in rows),
            "all_payloads_verified": True,
            "rows": rows,
        },
    )
    print(f"[completed] {output}")


if __name__ == "__main__":
    main()
