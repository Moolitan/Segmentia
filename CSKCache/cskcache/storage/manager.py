"""CSKCache physical-container verification and batched load orchestration."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Sequence
import threading
import uuid

from ..metadata.base import StorageBackend
from ..metadata.manager import MetadataManager
from ..profile import profile_event
from ..runtime.base import BindingState, HostLoadState, RuntimeReuseState
from .base import (
    CSKReadBatch,
    CSKReadResult,
    ExtentReadBackend,
    HostBufferPool,
    LayerObjectReadBackend,
)
from .transfers.base import StorageTransfer
from .transfers.layer_objects import LayerObjectTransfer
from .transfers.raw_extents import RawExtentTransfer


@dataclass
class _TicketHostLoad:
    """One ticket-owned SSD-to-pinned load and buffer group."""

    ticket: str
    cache_object_id: str
    io_operation_id: str
    batch: CSKReadBatch
    future: Future[tuple[Any, ...]] | None = None
    memory_objects: tuple[Any, ...] | None = None


class StorageManager:
    """Validate CSKCache ownership, then delegate one physical batch to LMCache."""

    def __init__(
        self,
        metadata_manager: MetadataManager,
        backend: ExtentReadBackend | None = None,
        *,
        storage_backend: str = "raw_block",
        local_disk_backend: LayerObjectReadBackend | None = None,
        host_buffer_pool: HostBufferPool | None = None,
        max_inflight_loads: int = 1,
    ) -> None:
        if max_inflight_loads <= 0:
            raise ValueError("max_inflight_loads must be > 0")
        self._metadata_manager = metadata_manager
        if storage_backend not in ("raw_block", "local_disk"):
            raise ValueError(f"unsupported CSKCache storage backend: {storage_backend}")
        if storage_backend == "raw_block" and backend is None:
            raise ValueError("raw_block storage requires an extent backend")
        if storage_backend == "local_disk" and local_disk_backend is None:
            raise ValueError("local_disk storage requires a layer-object backend")
        if storage_backend == "raw_block":
            assert backend is not None
            transfer: StorageTransfer = RawExtentTransfer(
                metadata_manager,
                backend,
                host_buffer_pool,
            )
        else:
            assert local_disk_backend is not None
            transfer = LayerObjectTransfer(local_disk_backend, host_buffer_pool)
        self._transfer = transfer
        self.storage_backend = transfer.storage_backend.value
        self._host_buffer_pool = host_buffer_pool
        self._max_inflight_loads = max_inflight_loads
        self._lock = threading.RLock()
        self._executor: ThreadPoolExecutor | None = None
        self._loads_by_ticket: dict[str, _TicketHostLoad] = {}
        self._closed = False

    def read_object_into(
        self,
        cache_object_id: str,
        destination_memory_objects: Sequence[Any],
    ) -> CSKReadResult:
        """Load all layers using persisted extents, never LMCache's key index."""

        metadata = self._metadata_manager.get_object(cache_object_id)
        if metadata.storage_backend is not StorageBackend.RAW_BLOCK:
            raise RuntimeError("read_object_into is only available for raw_block")
        assert metadata.container_id is not None
        container = self._metadata_manager.get_container(metadata.container_id)
        self._transfer.validate_container(container)
        batch = CSKReadBatch.from_metadata(
            metadata,
            container,
            expected_layers=self._metadata_manager.expected_layers,
        )
        if len(destination_memory_objects) != len(batch.extents):
            raise ValueError(
                "destination count must equal the complete model layer count"
            )
        read_into = getattr(self._transfer, "read_into", None)
        if not callable(read_into):
            raise RuntimeError("read_object_into is only available for raw_block")
        return read_into(batch, destination_memory_objects)

    def submit_host_load(
        self,
        ticket: str,
        cache_object_id: str,
    ) -> RuntimeReuseState:
        """Start one ticket-owned storage-to-pinned layer-group load.

        The caller creates ``ticket`` in MetadataManager first. This method
        returns while that ticket's physical read is still in progress.
        """

        if self._host_buffer_pool is None:
            raise RuntimeError("asynchronous host loading requires a buffer pool")
        metadata = self._metadata_manager.get_object(cache_object_id)
        if metadata.storage_backend.value != self.storage_backend:
            raise ValueError(
                "selected CSKCache backend differs from object metadata: "
                f"selected={self.storage_backend}, "
                f"object={metadata.storage_backend.value}"
            )
        container = None
        if metadata.storage_backend is StorageBackend.RAW_BLOCK:
            assert metadata.container_id is not None
            container = self._metadata_manager.get_container(metadata.container_id)
            self._transfer.validate_container(container)
        else:
            self._transfer.validate_container(None)
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
            if ticket in self._loads_by_ticket:
                raise ValueError(f"ticket already owns a host load: {ticket}")
            load = _TicketHostLoad(
                ticket=ticket,
                cache_object_id=cache_object_id,
                io_operation_id=f"host-io-{uuid.uuid4().hex}",
                batch=batch,
            )
            self._loads_by_ticket[ticket] = load
            try:
                state = self._metadata_manager.start_host_load(
                    ticket,
                    io_operation_id=load.io_operation_id,
                )
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
                    lambda future, owner=ticket: self._complete_host_load(
                        owner, future
                    )
                )
                return state
            except Exception:
                self._loads_by_ticket.pop(ticket, None)
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
        """Return the pinned buffer group owned by ``ticket``."""

        state = self._metadata_manager.get_runtime(ticket)
        if state.host_load_state is not HostLoadState.READY:
            raise RuntimeError("host buffers are not ready")
        with self._lock:
            load = self._loads_by_ticket.get(ticket)
            if load is None or load.memory_objects is None:
                raise RuntimeError("ready ticket has no resident buffer group")
            return load.memory_objects

    def cancel_host_load(
        self,
        ticket: str,
        reason: str = "host_load_cancelled",
    ) -> None:
        """Cancel one ticket and release its buffers when safe."""

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
        """Release one ticket-owned Host-buffer group."""

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
            tickets = tuple(self._loads_by_ticket)
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
            for load in self._loads_by_ticket.values():
                if load.memory_objects is not None:
                    buffers.append(load.memory_objects)
            self._loads_by_ticket.clear()
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
            profile_event(
                "csk_host_read_start",
                ticket,
                cache_object_id=batch.cache_object_id,
                io_operation_id=io_operation_id,
                layers=len(batch.extents),
                bytes=sum(batch.lengths),
                storage_backend=self.storage_backend,
            )
            memory_objects = self._transfer.load(batch)
            profile_event(
                "csk_host_read_complete",
                ticket,
                cache_object_id=batch.cache_object_id,
                io_operation_id=io_operation_id,
                successful_extents=len(memory_objects),
                extents=len(memory_objects),
                storage_backend=self.storage_backend,
            )
            profile_event(
                "csk_host_buffer_acquire_complete",
                ticket,
                cache_object_id=batch.cache_object_id,
                io_operation_id=io_operation_id,
                buffers=len(memory_objects),
            )
            return memory_objects
        except Exception:
            self._release_buffers(memory_objects)
            raise

    def _complete_host_load(
        self,
        ticket: str,
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
            load = self._loads_by_ticket.get(ticket)
            if load is None or load.future is not future:
                release_now = memory_objects or None
            elif error is not None:
                self._loads_by_ticket.pop(ticket, None)
                try:
                    self._metadata_manager.mark_host_failed(
                        ticket, f"{type(error).__name__}: {error}"
                    )
                except ValueError:
                    pass
            else:
                load.memory_objects = memory_objects
                try:
                    self._metadata_manager.mark_host_ready(ticket)
                    profile_event(
                        "csk_host_ready",
                        ticket,
                        cache_object_id=load.cache_object_id,
                        io_operation_id=load.io_operation_id,
                        buffers=len(memory_objects),
                    )
                except ValueError:
                    self._loads_by_ticket.pop(ticket, None)
                    load.memory_objects = None
                    release_now = memory_objects
        self._release_buffers(release_now)

    def _detach_ticket_locked(self, ticket: str) -> tuple[Any, ...] | None:
        load = self._loads_by_ticket.pop(ticket, None)
        if load is None:
            raise KeyError(f"ticket has no live host load: {ticket}")
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
