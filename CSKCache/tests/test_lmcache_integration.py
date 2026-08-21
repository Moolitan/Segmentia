"""Cross-package tests for CSKCache over LMCache physical infrastructure."""

from __future__ import annotations

from concurrent.futures import Future
from types import MethodType, SimpleNamespace

import pytest
import torch

from cskcache.integrations.lmcache import runtime as lmcache_runtime
from cskcache import (
    CacheObjectMetadata,
    SingleLayerChunkBuffers,
    LMCacheHostBufferPool,
    LMCacheLayerObjectReader,
    LayerExtent,
    ReadStrategy,
    StorageBackend,
)
from cskcache.integrations.lmcache import LMCacheRuntimeBridge
from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_management import MemoryFormat
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend


class FakeMemoryObject:
    def __init__(
        self,
        size: int,
        *,
        physical_size: int | None = None,
        shape: torch.Size,
        dtype: torch.dtype,
    ) -> None:
        self._size = size
        self._physical_size = physical_size if physical_size is not None else size
        self.tensor = SimpleNamespace(shape=shape, dtype=dtype)
        self.used_sizes = []
        self.release_calls = 0

    def get_size(self) -> int:
        return self._size

    def get_physical_size(self) -> int:
        return self._physical_size

    def set_used_size(self, size: int) -> None:
        if size > self._physical_size:
            raise ValueError("logical size exceeds physical capacity")
        self._size = size
        self.used_sizes.append(size)

    def ref_count_down(self) -> None:
        self.release_calls += 1


def make_extent(layer_id: int, *, tokens: int = 8) -> LayerExtent:
    return LayerExtent(
        layer_id=layer_id,
        backend_key=f"key-{layer_id}",
        offset_bytes=4096 * (layer_id + 1),
        length_bytes=2 * tokens * 64 * 2,
        dtype="bfloat16",
        shape=(2, tokens, 64),
        memory_layout="KV_2TD",
        payload_sha256=f"{layer_id + 1:064x}",
    )


def fake_local_cpu_backend(
    *,
    allocation_succeeds: bool = True,
    page_bytes: int | None = None,
    fail_after_calls: int | None = None,
):
    backend = object.__new__(LocalCPUBackend)
    backend.use_hot = True
    backend.calls = []
    backend.last_objects = []
    backend.object_batches = []

    def batched_allocate(
        self,
        shapes,
        dtypes,
        batch_size,
        fmt=None,
        eviction=True,
        busy_loop=True,
    ):
        self.calls.append(
            (shapes, dtypes, batch_size, fmt, eviction, busy_loop)
        )
        if not allocation_succeeds or (
            fail_after_calls is not None
            and len(self.calls) > fail_after_calls
        ):
            return None
        logical_size = int(torch.tensor(shapes).prod().item()) * dtypes.itemsize
        physical_size = page_bytes if page_bytes is not None else logical_size
        self.last_objects = [
            FakeMemoryObject(
                logical_size,
                physical_size=physical_size,
                shape=torch.Size(shapes),
                dtype=dtypes,
            )
            for _ in range(batch_size)
        ]
        self.object_batches.append(self.last_objects)
        return self.last_objects

    backend.batched_allocate = MethodType(batched_allocate, backend)
    return backend


def test_lmcache_pool_uses_one_nonblocking_batched_allocation() -> None:
    backend = fake_local_cpu_backend()
    pool = LMCacheHostBufferPool(backend)
    objects = pool.acquire([make_extent(0), make_extent(1)])

    assert len(backend.calls) == 1
    shapes, dtype, count, fmt, eviction, busy_loop = backend.calls[0]
    assert shapes == torch.Size((2, 8, 64))
    assert dtype is torch.bfloat16
    assert count == 2
    assert fmt is MemoryFormat.KV_2TD
    assert eviction is True
    assert busy_loop is False
    assert all(item.memory_obj.used_sizes == [] for item in objects)

    pool.release(objects)
    assert all(item.memory_obj.release_calls == 1 for item in objects)


def test_local_disk_reader_submits_all_skill_layers_through_layerwise_api() -> None:
    base_key = CacheEngineKey("model", 1, 0, 17, torch.bfloat16, None)
    keys = base_key.split_layers(3)
    objects = []
    for _ in keys:
        memory_obj = SimpleNamespace(unpin_calls=0, release_calls=0)
        memory_obj.unpin = lambda item=memory_obj: setattr(
            item, "unpin_calls", item.unpin_calls + 1
        )
        memory_obj.ref_count_down = lambda item=memory_obj: setattr(
            item, "release_calls", item.release_calls + 1
        )
        objects.append(memory_obj)

    class FakeStorageManager:
        def __init__(self) -> None:
            self.layerwise_calls = []
            self.unpin_calls = []

        def layerwise_batched_get(self, layer_keys, *, location, lookup_id):
            self.layerwise_calls.append((layer_keys, location, lookup_id))
            for memory_obj in objects:
                future = Future()
                future.set_result([memory_obj])
                yield future

        def batched_unpin(self, unpin_keys, locations):
            self.unpin_calls.append((unpin_keys, locations))

    storage = FakeStorageManager()
    reader = LMCacheLayerObjectReader(
        storage,
        location="LocalDiskBackend",
    )

    loaded = reader.read_layer_objects([key.to_string() for key in keys])

    assert loaded == tuple(objects)
    assert storage.layerwise_calls == [
        ([[key] for key in keys], "LocalDiskBackend", "cskcache-t0")
    ]
    assert storage.unpin_calls == [(keys, ["LocalDiskBackend"])]
    assert all(item.unpin_calls == 1 for item in objects)
    assert all(item.release_calls == 0 for item in objects)


def test_local_disk_reader_registers_catalog_layers() -> None:
    calls = []
    backend = SimpleNamespace(
        validate_existing=lambda *args: None,
        register_existing=lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    storage = SimpleNamespace(
        storage_backends={"LocalDiskBackend": backend},
        layerwise_batched_get=lambda *args, **kwargs: (),
        batched_unpin=lambda *args, **kwargs: None,
    )
    keys = CacheEngineKey(
        "model", 1, 0, 17, torch.bfloat16, None
    ).split_layers(2)
    layers = tuple(
        LayerExtent(
            layer_id=layer_id,
            backend_key=key.to_string(),
            offset_bytes=None,
            length_bytes=2 * 4 * 3 * 2,
            dtype="bfloat16",
            shape=(2, 4, 3),
            memory_layout="KV_2TD",
            payload_sha256=f"{layer_id + 1:064x}",
        )
        for layer_id, key in enumerate(keys)
    )
    cache_object = CacheObjectMetadata(
        object_id="skill:v1",
        skill_name="skill",
        skill_version="v1",
        model_fingerprint="model-fingerprint",
        tokenizer_fingerprint="tokenizer-fingerprint",
        token_count=4,
        source_position_start=10,
        token_ids_sha256="a" * 64,
        start_marker_token_ids=(1,),
        container_id=None,
        read_strategy=ReadStrategy.BATCHED,
        layers=layers,
        storage_backend=StorageBackend.LOCAL_DISK,
    )

    reader = LMCacheLayerObjectReader(storage, location="LocalDiskBackend")
    reader.register_catalog_objects([cache_object])

    assert [args[0] for args, _kwargs in calls] == list(keys)
    assert all(
        args[1:5]
        == (48, torch.Size((2, 4, 3)), torch.bfloat16, MemoryFormat.KV_2TD)
        for args, _kwargs in calls
    )
    assert all(
        torch.equal(kwargs["cached_positions"], torch.arange(10, 14))
        for _args, kwargs in calls
    )


def test_lmcache_pool_fails_instead_of_waiting_when_capacity_is_unavailable() -> None:
    pool = LMCacheHostBufferPool(fake_local_cpu_backend(allocation_succeeds=False))
    with pytest.raises(MemoryError, match="complete layers"):
        pool.acquire([make_extent(0), make_extent(1)])


def test_lmcache_pool_allocates_chunk_single_layer_with_separate_method() -> None:
    backend = fake_local_cpu_backend()
    pool = LMCacheHostBufferPool(
        backend,
        layout="chunk_single_layer",
        chunking_mode="fixed_size",
        chunk_tokens=4,
    )

    groups = pool.acquire_chunk_single_layer(
        [make_extent(0, tokens=10), make_extent(1, tokens=10)]
    )

    assert len(backend.calls) == 3
    assert [call[0] for call in backend.calls] == [
        torch.Size((2, 4, 64)),
        torch.Size((2, 4, 64)),
        torch.Size((2, 2, 64)),
    ]
    assert all(call[2] == 2 for call in backend.calls)
    assert all(isinstance(group, SingleLayerChunkBuffers) for group in groups)
    assert [
        [(chunk.token_start, chunk.token_end) for chunk in group.chunks]
        for group in groups
    ] == [[(0, 4), (4, 8), (8, 10)]] * 2
    assert [chunk.memory_obj for chunk in groups[0].chunks] == [
        batch[0] for batch in backend.object_batches
    ]
    assert [chunk.memory_obj for chunk in groups[1].chunks] == [
        batch[1] for batch in backend.object_batches
    ]

    pool.release(groups)
    assert all(
        item.release_calls == 1
        for batch in backend.object_batches
        for item in batch
    )


def test_lmcache_chunk_pool_releases_partial_allocation_on_failure() -> None:
    backend = fake_local_cpu_backend(fail_after_calls=1)
    pool = LMCacheHostBufferPool(
        backend,
        layout="chunk_single_layer",
        chunking_mode="fixed_size",
        chunk_tokens=4,
    )

    with pytest.raises(MemoryError, match="chunk-layer buffers"):
        pool.acquire([make_extent(0, tokens=10), make_extent(1, tokens=10)])

    assert len(backend.object_batches) == 1
    assert all(
        item.release_calls == 1 for item in backend.object_batches[0]
    )


def test_lmcache_pool_accepts_rebound_pages_with_skill_layout() -> None:
    page_bytes = 40 * 1024 * 1024
    backend = fake_local_cpu_backend(page_bytes=page_bytes)
    pool = LMCacheHostBufferPool(backend)
    extents = [make_extent(0, tokens=1919), make_extent(1, tokens=1919)]

    objects = pool.acquire(extents)

    assert all(
        item.memory_obj.get_physical_size() == page_bytes for item in objects
    )
    assert [item.memory_obj.get_size() for item in objects] == [
        extent.length_bytes for extent in extents
    ]
    assert [item.memory_obj.tensor.shape for item in objects] == [
        torch.Size(extent.shape) for extent in extents
    ]
    assert all(item.memory_obj.used_sizes == [] for item in objects)


def test_lmcache_pool_releases_group_when_page_capacity_is_too_small() -> None:
    backend = fake_local_cpu_backend(page_bytes=2048)
    pool = LMCacheHostBufferPool(backend)

    with pytest.raises(MemoryError, match="smaller than the persisted extent"):
        pool.acquire([make_extent(0, tokens=16), make_extent(1, tokens=16)])

    objects = backend.last_objects
    assert all(item.release_calls == 1 for item in objects)


def test_csk_t0_initialization_resolves_raw_block_plugin_key(monkeypatch) -> None:
    """The plugin registry is keyed by ``raw_block``, not its class name."""

    raw_backend = object()
    local_cpu_backend = object()
    captured = {}

    class FakeMetadataManager:
        def __init__(self, path, *, expected_layers):
            captured["metadata"] = (path, expected_layers)

        def list_objects(self):
            return ()

    class FakeHostBufferPool:
        def __init__(self, backend, **kwargs):
            captured["local_cpu_backend"] = backend
            captured["host_pool_kwargs"] = kwargs

    class FakeStorageManager:
        def __init__(
            self,
            metadata_manager,
            backend,
            *,
            storage_backend,
            local_disk_backend,
            host_buffer_pool,
            max_inflight_loads,
        ):
            captured["raw_backend"] = backend
            captured["storage_backend"] = storage_backend
            captured["local_disk_backend"] = local_disk_backend
            captured["max_inflight_loads"] = max_inflight_loads

    class FakeRequestManager:
        def __init__(self, *_args, **_kwargs):
            captured["request_manager_created"] = True

    monkeypatch.setattr(lmcache_runtime, "MetadataManager", FakeMetadataManager)
    monkeypatch.setattr(
        lmcache_runtime, "LMCacheHostBufferPool", FakeHostBufferPool
    )
    monkeypatch.setattr(lmcache_runtime, "StorageManager", FakeStorageManager)
    monkeypatch.setattr(lmcache_runtime, "RequestManager", FakeRequestManager)
    monkeypatch.setattr(
        lmcache_runtime, "fingerprint_model", lambda _path: "model"
    )
    monkeypatch.setattr(
        lmcache_runtime, "fingerprint_tokenizer", lambda _path: "tokenizer"
    )

    engine = SimpleNamespace(
        storage_manager=SimpleNamespace(
            storage_backends={
                "LocalCPUBackend": local_cpu_backend,
                "raw_block": raw_backend,
            }
        ),
        num_layers=40,
        use_layerwise=True,
        metadata=SimpleNamespace(model_name="/model"),
    )
    extras = {"cskcache_metadata_path": "/metadata.json"}
    engine.config = SimpleNamespace(
        chunk_size=256,
        local_cpu=True,
        get_extra_config_value=lambda key, default=None: extras.get(
            key, default
        ),
    )
    engine.is_healthy = lambda: True

    LMCacheRuntimeBridge(engine)

    assert captured["metadata"] == ("/metadata.json", 40)
    assert captured["local_cpu_backend"] is local_cpu_backend
    assert captured["host_pool_kwargs"] == {
        "layout": "chunk_single_layer",
        "chunking_mode": "whole_skill",
        "chunk_tokens": None,
    }
    assert captured["raw_backend"] is raw_backend
    assert captured["storage_backend"] == "raw_block"
    assert captured["local_disk_backend"] is None
    assert captured["max_inflight_loads"] == 4
    assert captured["request_manager_created"] is True
