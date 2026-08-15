from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence
import json
import threading
import time

import pytest

from cskcache import (
    CacheObjectMetadata,
    ContainerMetadata,
    BindingState,
    HostLoadState,
    LayerExtent,
    MetadataManager,
    ReadStrategy,
    StorageManager,
    generation_sidecar_path,
    publish_generation_sidecar,
)


@dataclass
class Destination:
    byte_array: bytearray


class FileExtentBackend:
    """CPU-only stand-in for LMCache's physical extent interface."""

    def __init__(self, container: ContainerMetadata) -> None:
        self.device_path = container.raw_file_path
        self.capacity_bytes = container.capacity_bytes
        self.block_align = container.alignment_bytes
        self.header_bytes = container.header_bytes
        self.read_calls = 0
        self.key_index_calls = 0
        self.forced_results: list[bool] | None = None

    def entry_offset(self, _key: object) -> int:
        self.key_index_calls += 1
        raise AssertionError("online extent loading must not query the key index")

    def read_extents_into(
        self,
        offsets: Sequence[int],
        lengths: Sequence[int],
        objs: Sequence[Any],
    ) -> list[bool]:
        self.read_calls += 1
        with Path(self.device_path).open("rb") as handle:
            for offset, length, obj in zip(offsets, lengths, objs, strict=True):
                if len(obj.byte_array) < length:
                    raise ValueError("destination buffer is too small")
                handle.seek(offset)
                payload = handle.read(length)
                if len(payload) != length:
                    raise RuntimeError("short physical read")
                obj.byte_array[:length] = payload
        if self.forced_results is not None:
            return list(self.forced_results)
        return [True] * len(offsets)


class BlockingExtentBackend(FileExtentBackend):
    """Keep the physical read pending so tests can inspect T0 state."""

    def __init__(self, container: ContainerMetadata) -> None:
        super().__init__(container)
        self.started = threading.Event()
        self.allow_completion = threading.Event()

    def read_extents_into(
        self,
        offsets: Sequence[int],
        lengths: Sequence[int],
        objs: Sequence[Any],
    ) -> list[bool]:
        self.started.set()
        if not self.allow_completion.wait(timeout=5):
            raise TimeoutError("test did not release the physical read")
        return super().read_extents_into(offsets, lengths, objs)


class RecordingHostBufferPool:
    """CPU-only stand-in for LMCache's long-lived pinned allocator."""

    def __init__(self, *, omit_last: bool = False) -> None:
        self.omit_last = omit_last
        self.acquire_calls = 0
        self.release_calls = 0
        self.released_groups: list[tuple[Destination, ...]] = []

    def acquire(self, extents: Sequence[LayerExtent]) -> Sequence[Destination]:
        self.acquire_calls += 1
        selected = extents[:-1] if self.omit_last else extents
        return tuple(
            Destination(bytearray(extent.length_bytes)) for extent in selected
        )

    def release(self, memory_objects: Sequence[Any]) -> None:
        self.release_calls += 1
        self.released_groups.append(tuple(memory_objects))


def wait_for_host_state(
    manager: StorageManager,
    ticket: str,
    expected: HostLoadState,
) -> None:
    deadline = time.monotonic() + 5
    while manager.poll_host_load(ticket) is not expected:
        if time.monotonic() >= deadline:
            raise AssertionError(
                f"ticket {ticket} did not reach host state {expected.value}"
            )
        time.sleep(0.005)


def prepare_container(tmp_path: Path, *, layers: int = 40) -> tuple[
    ContainerMetadata,
    tuple[bytes, ...],
]:
    alignment = 4096
    capacity = alignment * (layers + 2)
    raw_path = tmp_path / "skill-kv.raw"
    raw_path.write_bytes(b"\0" * capacity)
    payloads = tuple(
        bytes([layer % 251 + 1]) * 512 for layer in range(layers)
    )
    with raw_path.open("r+b") as handle:
        for layer, payload in enumerate(payloads):
            handle.seek(alignment * (layer + 1))
            handle.write(payload)
    container = ContainerMetadata(
        container_id="qwen3-14b-raw-v1",
        raw_file_path=str(raw_path.resolve()),
        container_format_version=1,
        storage_generation="generation-1",
        capacity_bytes=capacity,
        alignment_bytes=alignment,
        header_bytes=alignment,
    )
    publish_generation_sidecar(container)
    return container, payloads


def make_object(
    container: ContainerMetadata,
    payloads: tuple[bytes, ...],
) -> CacheObjectMetadata:
    extents = tuple(
        LayerExtent(
            layer_id=layer,
            backend_key=f"offline-key-{layer}",
            offset_bytes=container.alignment_bytes * (layer + 1),
            length_bytes=len(payload),
            dtype="uint8",
            shape=(len(payload),),
            memory_layout="flat-test-payload",
            payload_sha256=f"{layer + 1:064x}"[-64:],
        )
        for layer, payload in enumerate(payloads)
    )
    return CacheObjectMetadata(
        object_id="internal-comms:v1:qwen3-14b",
        skill_name="internal-comms",
        skill_version="v1",
        model_fingerprint="qwen3-14b@revision",
        tokenizer_fingerprint="qwen3-tokenizer@revision",
        token_count=339,
        source_position_start=0,
        token_ids_sha256="a" * 64,
        start_marker_token_ids=(1, 2, 3),
        container_id=container.container_id,
        read_strategy=ReadStrategy.BATCHED,
        layers=extents,
    )


def prepare_manager(
    tmp_path: Path,
) -> tuple[MetadataManager, ContainerMetadata, tuple[bytes, ...]]:
    container, payloads = prepare_container(tmp_path)
    manager = MetadataManager(tmp_path / "metadata.json", expected_layers=40)
    manager.publish_container(container)
    manager.publish_object(make_object(container, payloads))
    return manager, container, payloads


def test_40_extents_use_one_physical_submit_without_key_lookup(
    tmp_path: Path,
) -> None:
    manager, container, payloads = prepare_manager(tmp_path)
    backend = FileExtentBackend(container)
    destinations = [Destination(bytearray(len(payload))) for payload in payloads]

    result = StorageManager(manager, backend).read_object_into(
        "internal-comms:v1:qwen3-14b", destinations
    )

    assert result.complete
    assert result.layer_ids == tuple(range(40))
    assert backend.read_calls == 1
    assert backend.key_index_calls == 0
    assert tuple(bytes(item.byte_array) for item in destinations) == payloads


def test_wrong_generation_is_rejected_before_physical_read(tmp_path: Path) -> None:
    manager, container, payloads = prepare_manager(tmp_path)
    backend = FileExtentBackend(container)
    sidecar = generation_sidecar_path(container)
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload["storage_generation"] = "rebuilt-generation"
    sidecar.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="generation sidecar"):
        StorageManager(manager, backend).read_object_into(
            "internal-comms:v1:qwen3-14b",
            [Destination(bytearray(len(item))) for item in payloads],
        )
    assert backend.read_calls == 0


@pytest.mark.parametrize("kind", ["unaligned", "out_of_bounds", "overlap"])
def test_invalid_extent_layout_is_rejected_by_cskcache(
    tmp_path: Path, kind: str
) -> None:
    container, payloads = prepare_container(tmp_path)
    metadata = make_object(container, payloads)
    layers = list(metadata.layers)
    if kind == "unaligned":
        layers[0] = replace(layers[0], offset_bytes=layers[0].offset_bytes + 1)
    elif kind == "out_of_bounds":
        layers[-1] = replace(
            layers[-1], offset_bytes=container.capacity_bytes
        )
    else:
        layers[1] = replace(layers[1], offset_bytes=layers[0].offset_bytes)
    manager = MetadataManager(tmp_path / "metadata.json", expected_layers=40)
    manager.publish_container(container)
    with pytest.raises(ValueError):
        manager.publish_object(replace(metadata, layers=tuple(layers)))


def test_partial_backend_result_never_reports_complete(tmp_path: Path) -> None:
    manager, container, payloads = prepare_manager(tmp_path)
    backend = FileExtentBackend(container)
    backend.forced_results = [True] * 39 + [False]
    result = StorageManager(manager, backend).read_object_into(
        "internal-comms:v1:qwen3-14b",
        [Destination(bytearray(len(item))) for item in payloads],
    )
    assert not result.complete
    assert result.per_layer_success[-1] is False


def test_destination_count_must_cover_the_complete_object(tmp_path: Path) -> None:
    manager, container, payloads = prepare_manager(tmp_path)
    backend = FileExtentBackend(container)
    with pytest.raises(ValueError, match="complete model layer count"):
        StorageManager(manager, backend).read_object_into(
            "internal-comms:v1:qwen3-14b",
            [Destination(bytearray(len(item))) for item in payloads[:-1]],
        )
    assert backend.read_calls == 0


def test_submit_returns_while_one_40_layer_read_is_pending(tmp_path: Path) -> None:
    metadata, container, payloads = prepare_manager(tmp_path)
    metadata.create_ticket("call-1", "internal-comms:v1:qwen3-14b")
    backend = BlockingExtentBackend(container)
    pool = RecordingHostBufferPool()
    storage = StorageManager(metadata, backend, host_buffer_pool=pool)

    submitted = storage.submit_host_load(
        "call-1", "internal-comms:v1:qwen3-14b"
    )

    assert submitted.host_load_state is HostLoadState.LOADING
    assert backend.started.wait(timeout=1)
    assert not backend.allow_completion.is_set()
    backend.allow_completion.set()
    wait_for_host_state(storage, "call-1", HostLoadState.READY)
    buffers = storage.get_ready_buffers("call-1")
    assert tuple(bytes(item.byte_array) for item in buffers) == payloads
    assert backend.read_calls == 1
    assert backend.key_index_calls == 0
    assert pool.acquire_calls == 1

    released = storage.release_host_load("call-1")
    assert released.binding_state is BindingState.RELEASED
    assert pool.release_calls == 1
    storage.close()


def test_same_object_tickets_share_io_but_keep_independent_leases(
    tmp_path: Path,
) -> None:
    metadata, container, _ = prepare_manager(tmp_path)
    metadata.create_ticket("call-a", "internal-comms:v1:qwen3-14b")
    metadata.create_ticket("call-b", "internal-comms:v1:qwen3-14b")
    backend = BlockingExtentBackend(container)
    pool = RecordingHostBufferPool()
    storage = StorageManager(metadata, backend, host_buffer_pool=pool)

    first = storage.submit_host_load("call-a", "internal-comms:v1:qwen3-14b")
    assert backend.started.wait(timeout=1)
    second = storage.submit_host_load("call-b", "internal-comms:v1:qwen3-14b")
    assert first.io_operation_id == second.io_operation_id
    assert first.storage_lease_id != second.storage_lease_id

    backend.allow_completion.set()
    wait_for_host_state(storage, "call-a", HostLoadState.READY)
    wait_for_host_state(storage, "call-b", HostLoadState.READY)
    assert storage.get_ready_buffers("call-a") is storage.get_ready_buffers("call-b")
    assert backend.read_calls == 1
    assert pool.acquire_calls == 1

    storage.cancel_host_load("call-a", "request_a_cancelled")
    assert metadata.get_runtime("call-a").binding_state is BindingState.FALLBACK
    assert storage.poll_host_load("call-b") is HostLoadState.READY
    assert pool.release_calls == 0
    storage.release_host_load("call-b")
    assert pool.release_calls == 1
    storage.close()


def test_incomplete_pool_allocation_fails_without_physical_read(
    tmp_path: Path,
) -> None:
    metadata, container, _ = prepare_manager(tmp_path)
    metadata.create_ticket("call-1", "internal-comms:v1:qwen3-14b")
    backend = FileExtentBackend(container)
    pool = RecordingHostBufferPool(omit_last=True)
    storage = StorageManager(metadata, backend, host_buffer_pool=pool)

    storage.submit_host_load("call-1", "internal-comms:v1:qwen3-14b")
    wait_for_host_state(storage, "call-1", HostLoadState.FAILED)

    assert backend.read_calls == 0
    assert pool.release_calls == 1
    assert len(pool.released_groups[0]) == 39
    assert metadata.get_runtime("call-1").binding_state is BindingState.FALLBACK
    storage.close()


def test_partial_physical_read_fails_and_releases_complete_buffer_group(
    tmp_path: Path,
) -> None:
    metadata, container, _ = prepare_manager(tmp_path)
    metadata.create_ticket("call-1", "internal-comms:v1:qwen3-14b")
    backend = FileExtentBackend(container)
    backend.forced_results = [True] * 39 + [False]
    pool = RecordingHostBufferPool()
    storage = StorageManager(metadata, backend, host_buffer_pool=pool)

    storage.submit_host_load("call-1", "internal-comms:v1:qwen3-14b")
    wait_for_host_state(storage, "call-1", HostLoadState.FAILED)

    assert backend.read_calls == 1
    assert pool.release_calls == 1
    assert len(pool.released_groups[0]) == 40
    with pytest.raises(RuntimeError, match="not ready"):
        storage.get_ready_buffers("call-1")
    storage.close()


def test_cancelling_only_ticket_during_io_releases_after_safe_completion(
    tmp_path: Path,
) -> None:
    metadata, container, _ = prepare_manager(tmp_path)
    metadata.create_ticket("call-1", "internal-comms:v1:qwen3-14b")
    backend = BlockingExtentBackend(container)
    pool = RecordingHostBufferPool()
    storage = StorageManager(metadata, backend, host_buffer_pool=pool)

    storage.submit_host_load("call-1", "internal-comms:v1:qwen3-14b")
    assert backend.started.wait(timeout=1)
    storage.cancel_host_load("call-1", "request_cancelled")
    assert pool.release_calls == 0
    assert metadata.get_runtime("call-1").binding_state is BindingState.FALLBACK

    backend.allow_completion.set()
    deadline = time.monotonic() + 5
    while pool.release_calls == 0:
        if time.monotonic() >= deadline:
            raise AssertionError("completed orphan load did not release its buffers")
        time.sleep(0.005)
    assert pool.release_calls == 1
    storage.close()


def test_close_waits_for_inflight_read_and_leaves_no_pool_allocation(
    tmp_path: Path,
) -> None:
    metadata, container, _ = prepare_manager(tmp_path)
    metadata.create_ticket("call-1", "internal-comms:v1:qwen3-14b")
    backend = BlockingExtentBackend(container)
    pool = RecordingHostBufferPool()
    storage = StorageManager(metadata, backend, host_buffer_pool=pool)
    storage.submit_host_load("call-1", "internal-comms:v1:qwen3-14b")
    assert backend.started.wait(timeout=1)

    close_thread = threading.Thread(target=storage.close)
    close_thread.start()
    time.sleep(0.02)
    assert close_thread.is_alive()
    backend.allow_completion.set()
    close_thread.join(timeout=5)

    assert not close_thread.is_alive()
    assert pool.release_calls == 1
    assert metadata.get_runtime("call-1").binding_state is BindingState.FALLBACK
    with pytest.raises(RuntimeError, match="closed"):
        metadata.create_ticket("call-2", "internal-comms:v1:qwen3-14b")
        storage.submit_host_load("call-2", "internal-comms:v1:qwen3-14b")
