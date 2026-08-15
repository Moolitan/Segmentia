"""CSKCache physical-container verification and batched load orchestration."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence
import json
import os
import threading
import uuid

from .cache_metadata import CacheObjectMetadata, ContainerMetadata, LayerExtent
from .metadata_manager import MetadataManager
from .profile import profile_event
from .reuse_state import BindingState, HostLoadState, RuntimeReuseState


class ExtentReadBackend(Protocol):
    """The narrow physical interface CSKCache requires from LMCache."""

    device_path: str
    capacity_bytes: int
    block_align: int
    header_bytes: int

    def read_extents_into(
        self,
        offsets: Sequence[int],
        lengths: Sequence[int],
        objs: Sequence[Any],
    ) -> list[bool]: ...


class HostBufferPool(Protocol):
    """Long-lived pinned-memory pool supplied by LMCache at service startup.

    CSKCache owns the logical lease and the Skill-level lifetime.  The pool
    owns physical pinned-memory allocation.  Implementations must return one
    destination object per extent and accept the same group on release.
    """

    def acquire(self, extents: Sequence[LayerExtent]) -> Sequence[Any]: ...

    def release(self, memory_objects: Sequence[Any]) -> None: ...


@dataclass(frozen=True)
class CSKReadBatch:
    """One complete, layer-ordered physical read request for a Skill object."""

    cache_object_id: str
    container_id: str
    extents: tuple[LayerExtent, ...]

    @classmethod
    def from_metadata(
        cls,
        metadata: CacheObjectMetadata,
        container: ContainerMetadata,
        *,
        expected_layers: int,
    ) -> "CSKReadBatch":
        metadata.validate(expected_layers, container)
        return cls(
            cache_object_id=metadata.object_id,
            container_id=container.container_id,
            extents=tuple(sorted(metadata.layers, key=lambda item: item.layer_id)),
        )

    @property
    def layer_ids(self) -> tuple[int, ...]:
        return tuple(extent.layer_id for extent in self.extents)

    @property
    def offsets(self) -> tuple[int, ...]:
        return tuple(extent.offset_bytes for extent in self.extents)

    @property
    def lengths(self) -> tuple[int, ...]:
        return tuple(extent.length_bytes for extent in self.extents)


@dataclass(frozen=True)
class CSKReadResult:
    """Per-layer evidence with an explicit whole-object completion verdict."""

    cache_object_id: str
    layer_ids: tuple[int, ...]
    per_layer_success: tuple[bool, ...]

    @classmethod
    def from_backend_results(
        cls, batch: CSKReadBatch, results: Sequence[bool]
    ) -> "CSKReadResult":
        if len(results) != len(batch.extents):
            raise RuntimeError(
                "physical backend returned a result count different from the batch"
            )
        return cls(
            cache_object_id=batch.cache_object_id,
            layer_ids=batch.layer_ids,
            per_layer_success=tuple(bool(item) for item in results),
        )

    @property
    def complete(self) -> bool:
        return bool(self.per_layer_success) and all(self.per_layer_success)


@dataclass
class _SharedHostLoad:
    """One physical SSD read shared by all live tickets for one object."""

    cache_object_id: str
    io_operation_id: str
    batch: CSKReadBatch
    tickets: set[str]
    future: Future[tuple[Any, ...]] | None = None
    memory_objects: tuple[Any, ...] | None = None


@dataclass(frozen=True)
class _StorageLease:
    """One ticket's logical claim on a shared host-resident buffer group."""

    ticket: str
    lease_id: str
    cache_object_id: str


def generation_sidecar_path(container: ContainerMetadata) -> Path:
    """Return the CSKCache-owned generation sidecar for a raw container."""

    return Path(f"{container.raw_file_path}.cskcache-generation.json")


def publish_generation_sidecar(container: ContainerMetadata) -> Path:
    """Atomically publish the identity of a newly built raw container.

    Offline packing calls this only after LMCache has written and checkpointed
    the raw file. Online readers compare it with persistent CSKCache metadata
    before issuing any extent read.
    """

    container.validate()
    path = generation_sidecar_path(container)
    payload = {
        "container_id": container.container_id,
        "raw_file_path": str(Path(container.raw_file_path).resolve()),
        "container_format_version": container.container_format_version,
        "storage_generation": container.storage_generation,
        "capacity_bytes": container.capacity_bytes,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return path


class StorageManager:
    """Validate CSKCache ownership, then delegate one physical batch to LMCache."""

    def __init__(
        self,
        metadata_manager: MetadataManager,
        backend: ExtentReadBackend,
        *,
        host_buffer_pool: HostBufferPool | None = None,
        max_inflight_loads: int = 1,
    ) -> None:
        if max_inflight_loads <= 0:
            raise ValueError("max_inflight_loads must be > 0")
        self._metadata_manager = metadata_manager
        self._backend = backend
        self._host_buffer_pool = host_buffer_pool
        self._max_inflight_loads = max_inflight_loads
        self._lock = threading.RLock()
        self._executor: ThreadPoolExecutor | None = None
        self._loads_by_object: dict[str, _SharedHostLoad] = {}
        self._leases_by_ticket: dict[str, _StorageLease] = {}
        self._closed = False

    def read_object_into(
        self,
        cache_object_id: str,
        destination_memory_objects: Sequence[Any],
    ) -> CSKReadResult:
        """Load all layers using persisted extents, never LMCache's key index."""

        metadata = self._metadata_manager.get_object(cache_object_id)
        container = self._metadata_manager.get_container(metadata.container_id)
        self._validate_open_container(container)
        batch = CSKReadBatch.from_metadata(
            metadata,
            container,
            expected_layers=self._metadata_manager.expected_layers,
        )
        if len(destination_memory_objects) != len(batch.extents):
            raise ValueError(
                "destination count must equal the complete model layer count"
            )
        results = self._backend.read_extents_into(
            batch.offsets,
            batch.lengths,
            destination_memory_objects,
        )
        return CSKReadResult.from_backend_results(batch, results)

    def submit_host_load(
        self,
        ticket: str,
        cache_object_id: str,
    ) -> RuntimeReuseState:
        """Start or join one asynchronous raw-block-to-pinned load.

        The caller creates ``ticket`` in MetadataManager first.  This method
        assigns a ticket-local storage lease, attaches it to a shared physical
        read for ``cache_object_id``, and returns while that read is still in
        progress.  No LMCache key index is consulted.
        """

        if self._host_buffer_pool is None:
            raise RuntimeError("asynchronous host loading requires a buffer pool")
        metadata = self._metadata_manager.get_object(cache_object_id)
        container = self._metadata_manager.get_container(metadata.container_id)
        self._validate_open_container(container)
        batch = CSKReadBatch.from_metadata(
            metadata,
            container,
            expected_layers=self._metadata_manager.expected_layers,
        )
        runtime = self._metadata_manager.get_runtime(ticket)
        if runtime.cache_object_id != cache_object_id:
            raise ValueError("ticket cache object differs from submitted object")

        with self._lock:
            self._require_open()
            if ticket in self._leases_by_ticket:
                raise ValueError(f"ticket already owns a storage lease: {ticket}")
            load = self._loads_by_object.get(cache_object_id)
            is_new_load = load is None
            if load is None:
                load = _SharedHostLoad(
                    cache_object_id=cache_object_id,
                    io_operation_id=f"host-io-{uuid.uuid4().hex}",
                    batch=batch,
                    tickets=set(),
                )
                self._loads_by_object[cache_object_id] = load
            lease = _StorageLease(
                ticket=ticket,
                lease_id=f"host-lease-{uuid.uuid4().hex}",
                cache_object_id=cache_object_id,
            )
            load.tickets.add(ticket)
            self._leases_by_ticket[ticket] = lease
            try:
                state = self._metadata_manager.start_host_load(
                    ticket,
                    io_operation_id=load.io_operation_id,
                    storage_lease_id=lease.lease_id,
                )
                if load.memory_objects is not None:
                    return self._metadata_manager.mark_host_ready(ticket)
                if is_new_load:
                    executor = self._get_executor_locked()
                    profile_event(
                        "csk_host_load_submitted",
                        ticket,
                        cache_object_id=cache_object_id,
                        io_operation_id=load.io_operation_id,
                        layers=len(load.batch.extents),
                        bytes=sum(load.batch.lengths),
                    )
                    load.future = executor.submit(
                        self._load_into_pool,
                        ticket,
                        load.io_operation_id,
                        load.batch,
                    )
                    load.future.add_done_callback(
                        lambda future, object_id=cache_object_id: (
                            self._complete_host_load(object_id, future)
                        )
                    )
                return state
            except Exception:
                load.tickets.discard(ticket)
                self._leases_by_ticket.pop(ticket, None)
                if is_new_load and not load.tickets:
                    self._loads_by_object.pop(cache_object_id, None)
                try:
                    self._metadata_manager.mark_host_failed(
                        ticket, "host load submission failed"
                    )
                except ValueError:
                    pass
                raise

    def poll_host_load(self, ticket: str) -> HostLoadState:
        """Return the durable host-load state for one logical ticket."""

        return self._metadata_manager.get_runtime(ticket).host_load_state

    def get_ready_buffers(self, ticket: str) -> tuple[Any, ...]:
        """Return the shared read-only buffer group owned by ``ticket``."""

        state = self._metadata_manager.get_runtime(ticket)
        if state.host_load_state is not HostLoadState.READY:
            raise RuntimeError("host buffers are not ready")
        with self._lock:
            lease = self._leases_by_ticket.get(ticket)
            if lease is None:
                raise KeyError(f"ticket has no live storage lease: {ticket}")
            load = self._loads_by_object.get(lease.cache_object_id)
            if load is None or load.memory_objects is None:
                raise RuntimeError("ready metadata has no resident buffer group")
            return load.memory_objects

    def cancel_host_load(self, ticket: str, reason: str = "host_load_cancelled") -> None:
        """Cancel one logical lease without cancelling other sharing tickets."""

        if not reason:
            raise ValueError("cancellation reason must be non-empty")
        buffers_to_release: tuple[Any, ...] | None = None
        with self._lock:
            buffers_to_release = self._detach_ticket_locked(ticket)
            state = self._metadata_manager.get_runtime(ticket)
            if state.binding_state not in (
                BindingState.FALLBACK,
                BindingState.RELEASED,
            ):
                self._metadata_manager.fallback(ticket, reason)
        self._release_buffers(buffers_to_release)

    def release_host_load(self, ticket: str) -> RuntimeReuseState:
        """Release one logical lease and free buffers after the last user."""

        buffers_to_release: tuple[Any, ...] | None = None
        with self._lock:
            buffers_to_release = self._detach_ticket_locked(ticket)
            state = self._metadata_manager.release(ticket)
        self._release_buffers(buffers_to_release)
        return state

    def close(self) -> None:
        """Stop accepting loads and release every pool allocation exactly once."""

        with self._lock:
            if self._closed:
                return
            self._closed = True
            tickets = tuple(self._leases_by_ticket)
        for ticket in tickets:
            try:
                self.cancel_host_load(ticket, "storage_manager_closed")
            except (KeyError, ValueError):
                pass
        executor = self._executor
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)
        buffers: list[tuple[Any, ...]] = []
        with self._lock:
            for load in self._loads_by_object.values():
                if load.memory_objects is not None:
                    buffers.append(load.memory_objects)
            self._loads_by_object.clear()
            self._leases_by_ticket.clear()
        for group in buffers:
            self._release_buffers(group)

    def _load_into_pool(
        self,
        ticket: str,
        io_operation_id: str,
        batch: CSKReadBatch,
    ) -> tuple[Any, ...]:
        assert self._host_buffer_pool is not None
        memory_objects: tuple[Any, ...] = ()
        try:
            profile_event(
                "csk_host_buffer_acquire_start",
                ticket,
                cache_object_id=batch.cache_object_id,
                io_operation_id=io_operation_id,
            )
            memory_objects = tuple(self._host_buffer_pool.acquire(batch.extents))
            if len(memory_objects) != len(batch.extents):
                raise RuntimeError(
                    "host buffer pool returned an incomplete layer group"
                )
            profile_event(
                "csk_host_buffer_acquire_complete",
                ticket,
                cache_object_id=batch.cache_object_id,
                io_operation_id=io_operation_id,
                buffers=len(memory_objects),
            )
            profile_event(
                "csk_host_read_start",
                ticket,
                cache_object_id=batch.cache_object_id,
                io_operation_id=io_operation_id,
                layers=len(batch.extents),
                bytes=sum(batch.lengths),
            )
            results = self._backend.read_extents_into(
                batch.offsets,
                batch.lengths,
                memory_objects,
            )
            profile_event(
                "csk_host_read_complete",
                ticket,
                cache_object_id=batch.cache_object_id,
                io_operation_id=io_operation_id,
                successful_extents=sum(bool(item) for item in results),
                extents=len(results),
            )
            read_result = CSKReadResult.from_backend_results(batch, results)
            if not read_result.complete:
                raise RuntimeError("physical backend returned an incomplete layer group")
            return memory_objects
        except Exception:
            self._release_buffers(memory_objects)
            raise

    def _complete_host_load(
        self,
        cache_object_id: str,
        future: Future[tuple[Any, ...]],
    ) -> None:
        try:
            memory_objects = future.result()
            error: Exception | None = None
        except Exception as exc:
            memory_objects = ()
            error = exc

        release_now: tuple[Any, ...] | None = None
        with self._lock:
            load = self._loads_by_object.get(cache_object_id)
            if load is None or load.future is not future:
                release_now = memory_objects or None
            elif error is not None:
                self._loads_by_object.pop(cache_object_id, None)
                for ticket in tuple(load.tickets):
                    self._leases_by_ticket.pop(ticket, None)
                    try:
                        self._metadata_manager.mark_host_failed(
                            ticket, f"{type(error).__name__}: {error}"
                        )
                    except ValueError:
                        pass
                load.tickets.clear()
            elif not load.tickets:
                self._loads_by_object.pop(cache_object_id, None)
                release_now = memory_objects
            else:
                load.memory_objects = memory_objects
                for ticket in tuple(load.tickets):
                    try:
                        self._metadata_manager.mark_host_ready(ticket)
                        profile_event(
                            "csk_host_ready",
                            ticket,
                            cache_object_id=cache_object_id,
                            io_operation_id=load.io_operation_id,
                            buffers=len(memory_objects),
                        )
                    except ValueError:
                        load.tickets.discard(ticket)
                        self._leases_by_ticket.pop(ticket, None)
                if not load.tickets:
                    self._loads_by_object.pop(cache_object_id, None)
                    load.memory_objects = None
                    release_now = memory_objects
        self._release_buffers(release_now)

    def _detach_ticket_locked(self, ticket: str) -> tuple[Any, ...] | None:
        lease = self._leases_by_ticket.pop(ticket, None)
        if lease is None:
            raise KeyError(f"ticket has no live storage lease: {ticket}")
        load = self._loads_by_object.get(lease.cache_object_id)
        if load is None:
            return None
        load.tickets.discard(ticket)
        if load.tickets or load.memory_objects is None:
            return None
        self._loads_by_object.pop(lease.cache_object_id, None)
        buffers = load.memory_objects
        load.memory_objects = None
        return buffers

    def _get_executor_locked(self) -> ThreadPoolExecutor:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=self._max_inflight_loads,
                thread_name_prefix="cskcache-host-load",
            )
        return self._executor

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("StorageManager is closed")

    def _release_buffers(self, memory_objects: Sequence[Any] | None) -> None:
        if memory_objects and self._host_buffer_pool is not None:
            self._host_buffer_pool.release(memory_objects)

    def _validate_open_container(self, container: ContainerMetadata) -> None:
        container.validate()
        expected_path = Path(container.raw_file_path).resolve()
        backend_path = Path(self._backend.device_path).resolve()
        if backend_path != expected_path:
            raise ValueError(
                "LMCache backend is open on a different raw container path"
            )
        if not expected_path.is_file():
            raise ValueError("raw container path is not a regular file")
        if expected_path.stat().st_size != container.capacity_bytes:
            raise ValueError("raw container size differs from CSKCache metadata")
        if int(self._backend.capacity_bytes) != container.capacity_bytes:
            raise ValueError("LMCache backend capacity differs from CSKCache metadata")
        if int(self._backend.block_align) != container.alignment_bytes:
            raise ValueError("LMCache backend alignment differs from CSKCache metadata")
        if int(self._backend.header_bytes) != container.header_bytes:
            raise ValueError(
                "LMCache backend header size differs from CSKCache metadata"
            )
        self._validate_generation_sidecar(container)

    @staticmethod
    def _validate_generation_sidecar(container: ContainerMetadata) -> None:
        path = generation_sidecar_path(container)
        if not path.is_file():
            raise ValueError("CSKCache generation sidecar is missing")
        payload = json.loads(path.read_text(encoding="utf-8"))
        expected = {
            "container_id": container.container_id,
            "raw_file_path": str(Path(container.raw_file_path).resolve()),
            "container_format_version": container.container_format_version,
            "storage_generation": container.storage_generation,
            "capacity_bytes": container.capacity_bytes,
        }
        if payload != expected:
            raise ValueError(
                "CSKCache generation sidecar does not match container metadata"
            )
