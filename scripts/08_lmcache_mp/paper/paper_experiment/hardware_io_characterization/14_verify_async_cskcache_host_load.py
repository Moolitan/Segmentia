#!/usr/bin/env python3
"""Verify asynchronous CSKCache T0 loads against the real 990 PRO container.

Run ``11_prepare_raw_skill_kv.py`` first.  This test does not start an Agent,
vLLM, or a GPU model.  It allocates and registers one long-lived pinned CPU
buffer group before timing begins, then verifies that CSKCache can submit each
Skill at T0, return while I/O runs, reach HOST_READY, expose all 40 layers, and
return the logical lease without reallocating or re-registering pinned pages.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence
import sys
import threading
import time

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
CSKCACHE_PACKAGE_ROOT = REPOSITORY_ROOT / "CSKCache"
if str(CSKCACHE_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(CSKCACHE_PACKAGE_ROOT))

from cskcache import (  # noqa: E402
    HostLoadState,
    LayerExtent,
    MetadataManager,
    StorageManager,
)

from common import load_config, require_mounted_device, write_test_result  # noqa: E402
from raw_skill_kv_common import (  # noqa: E402
    RawLayerSource,
    build_layout,
    discover_raw_sources,
    memory_object,
    open_core,
    sha256_bytes,
)


class CountingExtentBackend:
    """Count physical batches while forwarding to LMCache RawBlockCore."""

    def __init__(self, core: Any) -> None:
        self._core = core
        self.device_path = core.device_path
        self.capacity_bytes = core.capacity_bytes
        self.block_align = core.block_align
        self.header_bytes = core.header_bytes
        self.read_batches = 0
        self._lock = threading.Lock()

    def read_extents_into(
        self,
        offsets: Sequence[int],
        lengths: Sequence[int],
        objs: Sequence[Any],
    ) -> list[bool]:
        with self._lock:
            self.read_batches += 1
        return self._core.read_extents_into(offsets, lengths, objs)


class PersistentPinnedBufferPool:
    """One preallocated 40-layer pool used sequentially by the verifier.

    The CUDA page-lock and io_uring fixed-buffer registration happen once at
    process startup.  ``acquire`` only lends views into those pages; ``release``
    makes the same pages available to the next Skill and performs no pinning.
    """

    def __init__(
        self,
        pinned: torch.Tensor,
        layers_by_key: dict[str, RawLayerSource],
    ) -> None:
        self._pinned = pinned
        self._layers_by_key = layers_by_key
        self._lock = threading.Lock()
        self._in_use = False
        self.acquire_count = 0
        self.release_count = 0

    def acquire(self, extents: Sequence[LayerExtent]) -> Sequence[Any]:
        with self._lock:
            if self._in_use:
                raise RuntimeError("the single verification buffer group is in use")
            if len(extents) != self._pinned.shape[0]:
                raise RuntimeError("extent count differs from pinned layer count")
            objects = []
            for index, extent in enumerate(extents):
                try:
                    layer = self._layers_by_key[extent.backend_key]
                except KeyError as exc:
                    raise KeyError(
                        f"unknown offline layer key: {extent.backend_key}"
                    ) from exc
                if layer.size_bytes != extent.length_bytes:
                    raise RuntimeError("extent length differs from source layer")
                objects.append(
                    memory_object(
                        self._pinned[index, : extent.length_bytes],
                        layer,
                    )
                )
            self._in_use = True
            self.acquire_count += 1
            return tuple(objects)

    def release(self, memory_objects: Sequence[Any]) -> None:
        with self._lock:
            if not self._in_use:
                raise RuntimeError("pinned buffer group was released twice")
            if len(memory_objects) != self._pinned.shape[0]:
                raise RuntimeError("released group differs from pinned layer count")
            self._in_use = False
            self.release_count += 1

    @property
    def in_use(self) -> bool:
        with self._lock:
            return self._in_use


def wait_until_ready(storage: StorageManager, ticket: str) -> float:
    started = time.perf_counter_ns()
    while True:
        state = storage.poll_host_load(ticket)
        if state is HostLoadState.READY:
            return (time.perf_counter_ns() - started) / 1_000_000
        if state is HostLoadState.FAILED:
            raise RuntimeError(f"asynchronous host load failed for {ticket}")
        time.sleep(0.0005)


def verify_buffers(cache_object: Any, buffers: Sequence[Any]) -> None:
    for extent, memory_obj in zip(cache_object.layers, buffers, strict=True):
        actual = sha256_bytes(memory_obj.byte_array)
        if actual != extent.payload_sha256:
            raise RuntimeError(
                f"payload mismatch for {cache_object.skill_name}/layer-{extent.layer_id}"
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

    allocation_started = time.perf_counter_ns()
    maximum_layer_bytes = int(layout["maximum_layer_bytes"])
    pinned = torch.empty(
        (expected_layers, maximum_layer_bytes),
        dtype=torch.uint8,
        pin_memory=True,
    )
    allocation_ms = (time.perf_counter_ns() - allocation_started) / 1_000_000
    alignment = int(layout["block_alignment_bytes"])
    if any(pinned[layer].data_ptr() % alignment for layer in range(expected_layers)):
        raise RuntimeError("persistent pinned buffers are not raw-block aligned")

    layers_by_key = {
        layer.cache_key: layer for source in sources for layer in source.layers
    }
    if len(layers_by_key) != expected_layers * len(sources):
        raise RuntimeError("offline layer backend keys are not unique")
    pool = PersistentPinnedBufferPool(pinned, layers_by_key)
    core = open_core(layout)
    registration_started = time.perf_counter_ns()
    core.raw_device().register_fixed_buffers(
        [pinned[index].data_ptr() for index in range(expected_layers)],
        [maximum_layer_bytes] * expected_layers,
    )
    registration_ms = (time.perf_counter_ns() - registration_started) / 1_000_000
    backend = CountingExtentBackend(core)
    storage = StorageManager(
        metadata,
        backend,
        host_buffer_pool=pool,
        max_inflight_loads=1,
    )

    rows = []
    try:
        for index, source in enumerate(sources):
            cache_object = objects_by_skill[source.skill]
            ticket = f"verify-{index}-{source.skill}"
            metadata.create_ticket(ticket, cache_object.object_id)
            before_batches = backend.read_batches
            submitted_at = time.perf_counter_ns()
            submitted = storage.submit_host_load(ticket, cache_object.object_id)
            submit_ms = (time.perf_counter_ns() - submitted_at) / 1_000_000
            if submitted.host_load_state is not HostLoadState.LOADING:
                raise RuntimeError("T0 submission did not enter HOST_LOADING")
            ready_wait_ms = wait_until_ready(storage, ticket)
            buffers = storage.get_ready_buffers(ticket)
            verify_buffers(cache_object, buffers)
            if backend.read_batches - before_batches != 1:
                raise RuntimeError("one Skill did not use exactly one physical batch")
            storage.release_host_load(ticket)
            if pool.in_use:
                raise RuntimeError("logical release did not return the pinned group")
            rows.append(
                {
                    "skill": source.skill,
                    "cache_bytes": source.cache_bytes,
                    "layer_count": len(cache_object.layers),
                    "submit_ms": submit_ms,
                    "ready_wait_ms": ready_wait_ms,
                    "payload_sha256_verified": True,
                }
            )
            print(
                f"[verified] {source.skill} submit={submit_ms:.3f}ms "
                f"ready_wait={ready_wait_ms:.3f}ms"
            )
    finally:
        storage.close()
        core.close()

    if pool.acquire_count != len(sources) or pool.release_count != len(sources):
        raise RuntimeError("pinned pool acquire/release counts are unbalanced")
    output = write_test_result(
        "14_cskcache_async_host_load",
        config,
        {
            "metadata_path": str(metadata_path),
            "raw_file": str(Path(layout["raw_file"])),
            "skill_count": len(rows),
            "extent_count": sum(row["layer_count"] for row in rows),
            "physical_read_batches": backend.read_batches,
            "pinned_pool": {
                "groups": 1,
                "layers_per_group": expected_layers,
                "allocation_ms_excluded": allocation_ms,
                "fixed_buffer_registration_ms_excluded": registration_ms,
                "acquire_count": pool.acquire_count,
                "release_count": pool.release_count,
                "balanced": not pool.in_use,
            },
            "all_payloads_verified": True,
            "timing_scope": (
                "submit_ms measures synchronous T0 bookkeeping; ready_wait_ms "
                "measures background extent read completion. Linux page-cache "
                "state is not controlled, so this is not a cold-read benchmark."
            ),
            "rows": rows,
        },
    )
    print(f"[completed] {output}")


if __name__ == "__main__":
    main()
