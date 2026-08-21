"""Thread-safe owner of CSKCache persistent metadata and runtime state."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any
import json
import os
import threading
import time

from .base import (
    CacheObjectMetadata,
    CacheObjectStatus,
    ContainerMetadata,
    StorageBackend,
)
from ..runtime.base import (
    BindingState,
    HostLoadState,
    ReusePlan,
    RuntimeReuseState,
)


CATALOG_VERSION = 2


class MetadataManager:
    """Single authority for cache-object metadata and reuse lifecycle state.

    Persistent mutations are serialized and atomically replaced on disk.
    Runtime records remain in memory because tickets, request IDs, I/O handles,
    and leases are meaningful only within one serving-process lifetime.
    """

    def __init__(self, metadata_path: str | Path, *, expected_layers: int) -> None:
        if expected_layers <= 0:
            raise ValueError("expected_layers must be > 0")
        self.metadata_path = Path(metadata_path)
        self.expected_layers = expected_layers
        self._lock = threading.RLock()
        self._persistent_write_lock = threading.Lock()
        self._containers: dict[str, ContainerMetadata] = {}
        self._objects: dict[str, CacheObjectMetadata] = {}
        self._runtime_by_ticket: dict[str, RuntimeReuseState] = {}
        self._ticket_by_request: dict[str, str] = {}
        if self.metadata_path.exists():
            self._containers, self._objects = self._load_file()

    # Persistent Cache Metadata operations.

    def publish_container(self, metadata: ContainerMetadata) -> None:
        """Atomically publish one immutable raw-container description."""

        metadata.validate()
        with self._persistent_write_lock:
            with self._lock:
                if metadata.container_id in self._containers:
                    raise ValueError(
                        f"container_id already exists: {metadata.container_id}"
                    )
                updated = dict(self._containers)
                updated[metadata.container_id] = metadata
            self._write_file(updated, self._objects)
            with self._lock:
                self._containers = updated

    def get_container(self, container_id: str) -> ContainerMetadata:
        with self._lock:
            return self._require_container(container_id)

    def list_containers(self) -> tuple[ContainerMetadata, ...]:
        with self._lock:
            containers = tuple(self._containers.values())
        return tuple(sorted(containers, key=lambda item: item.container_id))

    def publish_object(self, metadata: CacheObjectMetadata) -> None:
        """Atomically publish a new immutable cache-object version."""

        container = None
        if metadata.storage_backend is StorageBackend.RAW_BLOCK:
            assert metadata.container_id is not None
            with self._lock:
                container = self._require_container(metadata.container_id)
        metadata.validate(self.expected_layers, container)
        if metadata.status is not CacheObjectStatus.ACTIVE:
            raise ValueError("newly published metadata must be active")
        with self._persistent_write_lock:
            with self._lock:
                if metadata.object_id in self._objects:
                    raise ValueError(f"object_id already exists: {metadata.object_id}")
                for existing in self._objects.values():
                    if (
                        existing.status is CacheObjectStatus.ACTIVE
                        and existing.identity_key == metadata.identity_key
                    ):
                        raise ValueError(
                            "an active object already owns identity "
                            f"{metadata.identity_key}"
                        )
                updated = dict(self._objects)
                updated[metadata.object_id] = metadata
            self._write_file(self._containers, updated)
            with self._lock:
                self._objects = updated

    def invalidate_object(self, object_id: str) -> CacheObjectMetadata:
        """Persistently prevent new tickets from using an existing object."""

        with self._persistent_write_lock:
            with self._lock:
                current = self._require_object(object_id)
                if current.status is CacheObjectStatus.INVALIDATED:
                    return current
                invalidated = replace(current, status=CacheObjectStatus.INVALIDATED)
                updated = dict(self._objects)
                updated[object_id] = invalidated
            self._write_file(self._containers, updated)
            with self._lock:
                self._objects = updated
            return invalidated

    def get_object(self, object_id: str) -> CacheObjectMetadata:
        with self._lock:
            return self._require_object(object_id)

    def resolve_object(
        self,
        *,
        skill_name: str,
        model_fingerprint: str,
        tokenizer_fingerprint: str,
        skill_version: str | None = None,
    ) -> CacheObjectMetadata:
        """Resolve exactly one active object; ambiguity is a hard error."""

        with self._lock:
            matches = [
                item
                for item in self._objects.values()
                if item.status is CacheObjectStatus.ACTIVE
                and item.skill_name == skill_name
                and item.model_fingerprint == model_fingerprint
                and item.tokenizer_fingerprint == tokenizer_fingerprint
                and (skill_version is None or item.skill_version == skill_version)
            ]
        if not matches:
            raise KeyError(f"no active cache object for Skill {skill_name!r}")
        if len(matches) != 1:
            versions = sorted(item.skill_version for item in matches)
            raise ValueError(
                f"Skill {skill_name!r} is ambiguous; specify one of {versions}"
            )
        return matches[0]

    def list_objects(
        self, *, include_invalidated: bool = False
    ) -> tuple[CacheObjectMetadata, ...]:
        with self._lock:
            objects = tuple(
                item
                for item in self._objects.values()
                if include_invalidated or item.status is CacheObjectStatus.ACTIVE
            )
        return tuple(sorted(objects, key=lambda item: item.object_id))

    # Runtime Reuse State operations.

    def create_ticket(
        self,
        ticket: str,
        cache_object_id: str,
        *,
        deadline_ns: int | None = None,
        now_ns: int | None = None,
    ) -> RuntimeReuseState:
        """Create request-independent runtime state when SkillAction is parsed."""

        if not ticket:
            raise ValueError("ticket must be non-empty")
        with self._lock:
            if ticket in self._runtime_by_ticket:
                raise ValueError(f"ticket already exists: {ticket}")
            metadata = self._require_object(cache_object_id)
            if metadata.status is not CacheObjectStatus.ACTIVE:
                raise ValueError(f"cache object is invalidated: {cache_object_id}")
            created_at = time.time_ns() if now_ns is None else int(now_ns)
            if deadline_ns is not None and deadline_ns <= created_at:
                raise ValueError("deadline_ns must be later than created_at_ns")
            state = RuntimeReuseState(
                ticket=ticket,
                cache_object_id=cache_object_id,
                host_load_state=HostLoadState.NOT_STARTED,
                binding_state=BindingState.UNBOUND,
                created_at_ns=created_at,
                deadline_ns=deadline_ns,
            )
            self._runtime_by_ticket[ticket] = state
            return state

    def start_host_load(
        self,
        ticket: str,
        *,
        io_operation_id: str,
    ) -> RuntimeReuseState:
        if not io_operation_id:
            raise ValueError("I/O operation ID must be non-empty")
        with self._lock:
            state = self._require_runtime(ticket)
            self._require_live(state)
            if state.host_load_state is not HostLoadState.NOT_STARTED:
                raise ValueError("host load has already started")
            return self._store_runtime(
                state.updated(
                    host_load_state=HostLoadState.LOADING,
                    io_operation_id=io_operation_id,
                )
            )

    def mark_host_ready(self, ticket: str) -> RuntimeReuseState:
        with self._lock:
            state = self._require_runtime(ticket)
            self._require_live(state)
            if state.host_load_state is not HostLoadState.LOADING:
                raise ValueError("only a loading ticket can become host-ready")
            return self._store_runtime(
                state.updated(host_load_state=HostLoadState.READY)
            )

    def mark_host_failed(self, ticket: str, reason: str) -> RuntimeReuseState:
        if not reason:
            raise ValueError("failure reason must be non-empty")
        with self._lock:
            state = self._require_runtime(ticket)
            self._require_live(state)
            if state.host_load_state not in (
                HostLoadState.NOT_STARTED,
                HostLoadState.LOADING,
            ):
                raise ValueError("host load can no longer fail")
            return self._store_runtime(
                state.updated(
                    host_load_state=HostLoadState.FAILED,
                    binding_state=BindingState.FALLBACK,
                    fallback_reason=reason,
                )
            )

    def bind_request(
        self,
        ticket: str,
        *,
        request_id: str,
        verified_cache_object_id: str,
        segment_start: int,
        segment_end: int,
    ) -> RuntimeReuseState:
        """Bind token-authenticated request state without waiting for SSD I/O."""

        if not request_id:
            raise ValueError("request_id must be non-empty")
        if segment_start < 0 or segment_end <= segment_start:
            raise ValueError("verified Skill span is invalid")
        with self._lock:
            state = self._require_runtime(ticket)
            self._require_live(state)
            if state.binding_state is not BindingState.OBSERVED:
                raise ValueError("Skill observation must be verified before binding")
            if verified_cache_object_id != state.cache_object_id:
                raise ValueError(
                    "verified cache object does not match prefetched object"
                )
            existing_ticket = self._ticket_by_request.get(request_id)
            if existing_ticket is not None:
                raise ValueError(
                    f"request {request_id!r} is already bound to {existing_ticket!r}"
                )
            bound = state.updated(
                request_id=request_id,
                segment_start=segment_start,
                segment_end=segment_end,
                binding_state=BindingState.VERIFIED,
            )
            self._ticket_by_request[request_id] = ticket
            return self._store_runtime(bound)

    def mark_observation_verified(self, ticket: str) -> RuntimeReuseState:
        """Record that request B contains the expected successful Skill result."""

        with self._lock:
            state = self._require_runtime(ticket)
            self._require_live(state)
            if state.binding_state is BindingState.OBSERVED:
                return state
            if state.binding_state is not BindingState.UNBOUND:
                raise ValueError("ticket can no longer accept a Skill observation")
            return self._store_runtime(
                state.updated(binding_state=BindingState.OBSERVED)
            )

    def set_reuse_plan(
        self,
        ticket: str,
        plan: ReusePlan,
    ) -> RuntimeReuseState:
        """Atomically attach an already validated plan to its ticket."""

        if plan.ticket != ticket:
            raise ValueError("reuse plan ticket does not match target ticket")
        with self._lock:
            state = self._require_runtime(ticket)
            self._require_live(state)
            if state.binding_state is not BindingState.VERIFIED:
                raise ValueError("reuse planning requires a verified request")
            if state.request_id != plan.request_id:
                raise ValueError("reuse plan request does not match ticket binding")
            values = (
                plan.reuse_start,
                plan.reuse_end,
                plan.source_reuse_start,
                plan.source_reuse_end,
                plan.calibration_start,
                plan.calibration_end,
                plan.correction_alpha,
                plan.block_alignment,
            )
            existing = (
                state.reuse_start,
                state.reuse_end,
                state.source_reuse_start,
                state.source_reuse_end,
                state.calibration_start,
                state.calibration_end,
                state.correction_alpha,
                state.block_alignment,
            )
            if state.reuse_start is not None:
                if existing != values:
                    raise ValueError("ticket already has a different reuse plan")
                return state
            return self._store_runtime(
                state.updated(
                    reuse_start=plan.reuse_start,
                    reuse_end=plan.reuse_end,
                    source_reuse_start=plan.source_reuse_start,
                    source_reuse_end=plan.source_reuse_end,
                    calibration_start=plan.calibration_start,
                    calibration_end=plan.calibration_end,
                    correction_alpha=plan.correction_alpha,
                    block_alignment=plan.block_alignment,
                )
            )

    def activate(self, ticket: str) -> RuntimeReuseState:
        """Activate reuse only after both authentication and host load complete."""

        with self._lock:
            state = self._require_runtime(ticket)
            if state.binding_state is not BindingState.VERIFIED:
                raise ValueError("ticket must be verified before activation")
            if state.host_load_state is not HostLoadState.READY:
                raise ValueError("host data must be ready before activation")
            return self._store_runtime(
                state.updated(binding_state=BindingState.ACTIVE)
            )

    def mark_layer_loaded(self, ticket: str, layer_id: int) -> RuntimeReuseState:
        with self._lock:
            state = self._require_active(ticket)
            self._require_next_layer(state.loaded_through_layer, layer_id)
            return self._store_runtime(
                state.updated(loaded_through_layer=layer_id)
            )

    def mark_layer_corrected(self, ticket: str, layer_id: int) -> RuntimeReuseState:
        with self._lock:
            state = self._require_active(ticket)
            self._require_next_layer(state.corrected_through_layer, layer_id)
            if layer_id > state.loaded_through_layer:
                raise ValueError("a layer must be loaded before it is corrected")
            return self._store_runtime(
                state.updated(corrected_through_layer=layer_id)
            )

    def fallback(self, ticket: str, reason: str) -> RuntimeReuseState:
        if not reason:
            raise ValueError("fallback reason must be non-empty")
        with self._lock:
            state = self._require_runtime(ticket)
            self._require_live(state)
            return self._store_runtime(
                state.updated(
                    binding_state=BindingState.FALLBACK,
                    fallback_reason=reason,
                )
            )

    def expire(self, *, now_ns: int | None = None) -> tuple[RuntimeReuseState, ...]:
        """Move expired nonterminal tickets to FALLBACK without deleting evidence."""

        current = time.time_ns() if now_ns is None else int(now_ns)
        expired: list[RuntimeReuseState] = []
        with self._lock:
            for ticket, state in tuple(self._runtime_by_ticket.items()):
                if (
                    state.deadline_ns is not None
                    and current >= state.deadline_ns
                    and state.binding_state
                    not in (BindingState.FALLBACK, BindingState.RELEASED)
                ):
                    updated = state.updated(
                        binding_state=BindingState.FALLBACK,
                        fallback_reason="deadline_expired",
                    )
                    self._runtime_by_ticket[ticket] = updated
                    expired.append(updated)
        return tuple(expired)

    def release(self, ticket: str) -> RuntimeReuseState:
        """End the ticket lifecycle and remove its request lookup binding."""

        with self._lock:
            state = self._require_runtime(ticket)
            if state.binding_state is BindingState.RELEASED:
                return state
            if state.request_id is not None:
                self._ticket_by_request.pop(state.request_id, None)
            return self._store_runtime(
                state.updated(binding_state=BindingState.RELEASED)
            )

    def get_runtime(self, ticket: str) -> RuntimeReuseState:
        with self._lock:
            return self._require_runtime(ticket)

    def get_runtime_for_request(self, request_id: str) -> RuntimeReuseState:
        with self._lock:
            try:
                ticket = self._ticket_by_request[request_id]
            except KeyError as exc:
                raise KeyError(f"unknown request_id: {request_id}") from exc
            return self._require_runtime(ticket)

    # Internal validation and atomic persistence.

    def _require_object(self, object_id: str) -> CacheObjectMetadata:
        try:
            return self._objects[object_id]
        except KeyError as exc:
            raise KeyError(f"unknown cache object: {object_id}") from exc

    def _require_container(self, container_id: str) -> ContainerMetadata:
        try:
            return self._containers[container_id]
        except KeyError as exc:
            raise KeyError(f"unknown raw container: {container_id}") from exc

    def _require_runtime(self, ticket: str) -> RuntimeReuseState:
        try:
            return self._runtime_by_ticket[ticket]
        except KeyError as exc:
            raise KeyError(f"unknown ticket: {ticket}") from exc

    @staticmethod
    def _require_live(state: RuntimeReuseState) -> None:
        if state.binding_state in (BindingState.FALLBACK, BindingState.RELEASED):
            raise ValueError(f"ticket is terminal: {state.binding_state.value}")

    def _require_active(self, ticket: str) -> RuntimeReuseState:
        state = self._require_runtime(ticket)
        if state.binding_state is not BindingState.ACTIVE:
            raise ValueError("layer progress requires an active ticket")
        return state

    def _require_next_layer(self, previous: int, layer_id: int) -> None:
        if layer_id != previous + 1:
            raise ValueError(
                "layer progress must be consecutive: "
                f"previous={previous}, next={layer_id}"
            )
        if layer_id >= self.expected_layers:
            raise ValueError(f"layer_id exceeds model depth: {layer_id}")

    def _store_runtime(self, state: RuntimeReuseState) -> RuntimeReuseState:
        self._runtime_by_ticket[state.ticket] = state
        return state

    def _load_file(
        self,
    ) -> tuple[dict[str, ContainerMetadata], dict[str, CacheObjectMetadata]]:
        payload = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        if set(payload) != {
            "catalog_version",
            "expected_layers",
            "containers",
            "objects",
        }:
            raise ValueError("persistent metadata has unexpected top-level fields")
        version = int(payload["catalog_version"])
        if version not in (1, CATALOG_VERSION):
            raise ValueError(
                f"unsupported Catalog version: {payload['catalog_version']}"
            )
        if int(payload["expected_layers"]) != self.expected_layers:
            raise ValueError("persistent metadata model depth does not match service")
        if not isinstance(payload["containers"], list):
            raise ValueError("persistent metadata containers must be a list")
        if not isinstance(payload["objects"], list):
            raise ValueError("persistent metadata objects must be a list")
        containers: dict[str, ContainerMetadata] = {}
        for item in payload["containers"]:
            container = ContainerMetadata.from_dict(item)
            if container.container_id in containers:
                raise ValueError(
                    f"duplicate container_id: {container.container_id}"
                )
            containers[container.container_id] = container
        objects: dict[str, CacheObjectMetadata] = {}
        active_identities: set[tuple[str, str, str, str]] = set()
        for item in payload["objects"]:
            if version == 1:
                item = dict(item)
                item["storage_backend"] = StorageBackend.RAW_BLOCK.value
            metadata = CacheObjectMetadata.from_dict(item)
            container = None
            if metadata.storage_backend is StorageBackend.RAW_BLOCK:
                try:
                    assert metadata.container_id is not None
                    container = containers[metadata.container_id]
                except (AssertionError, KeyError) as exc:
                    raise ValueError(
                        f"object {metadata.object_id!r} references unknown container "
                        f"{metadata.container_id!r}"
                    ) from exc
            metadata.validate(self.expected_layers, container)
            if metadata.object_id in objects:
                raise ValueError(f"duplicate object_id: {metadata.object_id}")
            if (
                metadata.status is CacheObjectStatus.ACTIVE
                and metadata.identity_key in active_identities
            ):
                raise ValueError(
                    f"duplicate active identity: {metadata.identity_key}"
                )
            objects[metadata.object_id] = metadata
            if metadata.status is CacheObjectStatus.ACTIVE:
                active_identities.add(metadata.identity_key)
        return containers, objects

    def _write_file(
        self,
        containers: dict[str, ContainerMetadata],
        objects: dict[str, CacheObjectMetadata],
    ) -> None:
        payload: dict[str, Any] = {
            "catalog_version": CATALOG_VERSION,
            "expected_layers": self.expected_layers,
            "containers": [
                containers[key].to_dict() for key in sorted(containers)
            ],
            "objects": [objects[key].to_dict() for key in sorted(objects)],
        }
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.metadata_path.with_name(
            f".{self.metadata_path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        try:
            with temporary.open("x", encoding="utf-8") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.metadata_path)
            directory_fd = os.open(self.metadata_path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            temporary.unlink(missing_ok=True)
