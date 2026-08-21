"""Persistent cache identities and physical layouts owned by CSKCache."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any
import re

from ..chunking import ChunkingMode, ChunkingSpec, build_chunk_plan
from ..layouts import KVLayout, build_layout_plan


_SHA256 = re.compile(r"[0-9a-f]{64}")
SOURCE_ARTIFACT_TYPE = "cskcache_source_object"
RAW_BUILD_CHECKPOINT_TYPE = "cskcache_raw_build_checkpoint"
SKILL_PAYLOAD_FORMAT = "context_segment_v1"


@dataclass(frozen=True)
class SkillTokenIdentity:
    """Exact Skill payload text and token identity of one source object."""

    payload_format: str
    observation_text: str
    cache_text: str
    token_ids: tuple[int, ...]
    token_ids_sha256: str
    start_marker_text: str
    start_marker_token_ids: tuple[int, ...]
    start_marker_token_ids_sha256: str

    def validate(self) -> None:
        if self.payload_format != SKILL_PAYLOAD_FORMAT:
            raise ValueError(f"unsupported payload_format: {self.payload_format!r}")
        for field in ("observation_text", "cache_text", "start_marker_text"):
            _require_text(getattr(self, field), field)
        if self.cache_text != self.observation_text + "\n":
            raise ValueError("cache_text must add exactly one Tool boundary newline")
        if not self.token_ids or any(token < 0 for token in self.token_ids):
            raise ValueError("token_ids must contain non-negative token IDs")
        if not self.start_marker_token_ids or any(
            token < 0 for token in self.start_marker_token_ids
        ):
            raise ValueError(
                "start_marker_token_ids must contain non-negative token IDs"
            )
        if self.token_ids[: len(self.start_marker_token_ids)] != (
            self.start_marker_token_ids
        ):
            raise ValueError("start marker tokens must prefix the cache token object")
        _require_sha256(self.token_ids_sha256, "token_ids_sha256")
        _require_sha256(
            self.start_marker_token_ids_sha256,
            "start_marker_token_ids_sha256",
        )


class CacheObjectStatus(str, Enum):
    """Persistent availability of one immutable cache-object version."""

    ACTIVE = "active"
    INVALIDATED = "invalidated"


class ReadStrategy(str, Enum):
    """How the storage backend should submit one object's layer extents."""

    CONTIGUOUS = "contiguous"
    BATCHED = "batched"


class StorageBackend(str, Enum):
    """Physical backend holding one complete Skill-layer group."""

    RAW_BLOCK = "raw_block"
    LOCAL_DISK = "local_disk"


def _require_text(value: str, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")


def _require_sha256(value: str, field: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 hex digest")


@dataclass(frozen=True)
class ContainerMetadata:
    """CSKCache-owned identity and geometry of one raw-block container.

    LMCache executes reads against the file, but it does not own this identity.
    In particular, ``storage_generation`` changes whenever CSKCache rebuilds
    the physical container so that stale object extents fail closed.
    """

    container_id: str
    raw_file_path: str
    container_format_version: int
    storage_generation: str
    capacity_bytes: int
    alignment_bytes: int
    header_bytes: int

    def validate(self) -> None:
        for field in ("container_id", "raw_file_path", "storage_generation"):
            _require_text(getattr(self, field), field)
        if not Path(self.raw_file_path).is_absolute():
            raise ValueError("raw_file_path must be absolute")
        if self.container_format_version <= 0:
            raise ValueError("container_format_version must be > 0")
        if self.capacity_bytes <= 0:
            raise ValueError("capacity_bytes must be > 0")
        if self.alignment_bytes <= 0:
            raise ValueError("alignment_bytes must be > 0")
        if self.header_bytes < 0:
            raise ValueError("header_bytes must be >= 0")
        if self.header_bytes % self.alignment_bytes != 0:
            raise ValueError("header_bytes must be alignment compliant")

    def to_dict(self) -> dict[str, Any]:
        return {
            "container_id": self.container_id,
            "raw_file_path": self.raw_file_path,
            "container_format_version": self.container_format_version,
            "storage_generation": self.storage_generation,
            "capacity_bytes": self.capacity_bytes,
            "alignment_bytes": self.alignment_bytes,
            "header_bytes": self.header_bytes,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ContainerMetadata":
        expected = {
            "container_id",
            "raw_file_path",
            "container_format_version",
            "storage_generation",
            "capacity_bytes",
            "alignment_bytes",
            "header_bytes",
        }
        if set(payload) != expected:
            raise ValueError(
                f"ContainerMetadata fields differ: expected={sorted(expected)}, "
                f"actual={sorted(payload)}"
            )
        metadata = cls(
            container_id=str(payload["container_id"]),
            raw_file_path=str(payload["raw_file_path"]),
            container_format_version=int(payload["container_format_version"]),
            storage_generation=str(payload["storage_generation"]),
            capacity_bytes=int(payload["capacity_bytes"]),
            alignment_bytes=int(payload["alignment_bytes"]),
            header_bytes=int(payload["header_bytes"]),
        )
        metadata.validate()
        return metadata


@dataclass(frozen=True)
class LayerExtent:
    """One model-layer object and its optional raw-block offset."""

    layer_id: int
    backend_key: str
    offset_bytes: int | None
    length_bytes: int
    dtype: str
    shape: tuple[int, ...]
    memory_layout: str
    payload_sha256: str

    def validate(self) -> None:
        if self.layer_id < 0:
            raise ValueError("layer_id must be >= 0")
        _require_text(self.backend_key, "backend_key")
        if self.offset_bytes is not None and self.offset_bytes < 0:
            raise ValueError("offset_bytes must be >= 0")
        if self.length_bytes <= 0:
            raise ValueError("length_bytes must be > 0")
        _require_text(self.dtype, "dtype")
        if not self.shape or any(dim <= 0 for dim in self.shape):
            raise ValueError("shape must contain only positive dimensions")
        _require_text(self.memory_layout, "memory_layout")
        _require_sha256(self.payload_sha256, "payload_sha256")

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer_id": self.layer_id,
            "backend_key": self.backend_key,
            "offset_bytes": self.offset_bytes,
            "length_bytes": self.length_bytes,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "memory_layout": self.memory_layout,
            "payload_sha256": self.payload_sha256,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LayerExtent":
        expected = {
            "layer_id",
            "backend_key",
            "offset_bytes",
            "length_bytes",
            "dtype",
            "shape",
            "memory_layout",
            "payload_sha256",
        }
        if set(payload) != expected:
            raise ValueError(
                f"LayerExtent fields differ: expected={sorted(expected)}, "
                f"actual={sorted(payload)}"
            )
        extent = cls(
            layer_id=int(payload["layer_id"]),
            backend_key=str(payload["backend_key"]),
            offset_bytes=(
                None
                if payload["offset_bytes"] is None
                else int(payload["offset_bytes"])
            ),
            length_bytes=int(payload["length_bytes"]),
            dtype=str(payload["dtype"]),
            shape=tuple(int(dim) for dim in payload["shape"]),
            memory_layout=str(payload["memory_layout"]),
            payload_sha256=str(payload["payload_sha256"]),
        )
        extent.validate()
        return extent


@dataclass(frozen=True)
class CacheObjectMetadata:
    """Immutable semantic identity plus physical layout of one Skill KV object."""

    object_id: str
    skill_name: str
    skill_version: str
    model_fingerprint: str
    tokenizer_fingerprint: str
    token_count: int
    source_position_start: int
    token_ids_sha256: str
    start_marker_token_ids: tuple[int, ...]
    container_id: str | None
    read_strategy: ReadStrategy
    layers: tuple[LayerExtent, ...]
    chunking: ChunkingSpec = field(
        default_factory=lambda: ChunkingSpec(ChunkingMode.WHOLE_SKILL)
    )
    storage_layout: KVLayout = KVLayout.CHUNK_SINGLE_LAYER
    storage_backend: StorageBackend = StorageBackend.RAW_BLOCK
    status: CacheObjectStatus = CacheObjectStatus.ACTIVE

    def validate(
        self,
        expected_layers: int,
        container: ContainerMetadata | None,
    ) -> None:
        for field in (
            "object_id",
            "skill_name",
            "skill_version",
            "model_fingerprint",
            "tokenizer_fingerprint",
        ):
            _require_text(getattr(self, field), field)
        if self.storage_backend is StorageBackend.RAW_BLOCK:
            if container is None:
                raise ValueError("raw_block object requires container metadata")
            container.validate()
            if self.container_id != container.container_id:
                raise ValueError("cache object references a different container")
        elif self.storage_backend is StorageBackend.LOCAL_DISK:
            if self.container_id is not None or container is not None:
                raise ValueError("local_disk object must not reference a raw container")
        else:
            raise ValueError(f"unsupported storage backend: {self.storage_backend}")
        if self.token_count <= 0:
            raise ValueError("token_count must be > 0")
        chunk_plan = build_chunk_plan(self.token_count, self.chunking)
        layout_plan = build_layout_plan(
            self.storage_layout,
            chunk_plan,
            expected_layers,
        )
        if len(layout_plan.regions) != expected_layers or any(
            region.layer_count != 1
            or region.token_start != 0
            or region.token_end != self.token_count
            for region in layout_plan.regions
        ):
            raise ValueError(
                "the current persistent Catalog encodes one complete Skill "
                "region per model layer"
            )
        if self.source_position_start < 0:
            raise ValueError("source_position_start must be >= 0")
        _require_sha256(self.token_ids_sha256, "token_ids_sha256")
        if not self.start_marker_token_ids or any(
            token < 0 for token in self.start_marker_token_ids
        ):
            raise ValueError("start_marker_token_ids must contain non-negative IDs")
        if len(self.layers) != expected_layers:
            raise ValueError(
                f"expected {expected_layers} layer extents, found {len(self.layers)}"
            )
        ordered = tuple(sorted(self.layers, key=lambda item: item.layer_id))
        actual_ids = tuple(item.layer_id for item in ordered)
        if actual_ids != tuple(range(expected_layers)):
            raise ValueError(
                "layer IDs must be exactly "
                f"0..{expected_layers - 1}, found {actual_ids}"
            )
        backend_keys: set[str] = set()
        ranges: list[tuple[int, int]] = []
        for extent in ordered:
            extent.validate()
            if extent.backend_key in backend_keys:
                raise ValueError(f"duplicate backend_key: {extent.backend_key}")
            backend_keys.add(extent.backend_key)
            if self.storage_backend is StorageBackend.LOCAL_DISK:
                if extent.offset_bytes is not None:
                    raise ValueError("local_disk layer must not contain a raw offset")
                continue
            assert container is not None and extent.offset_bytes is not None
            if extent.offset_bytes % container.alignment_bytes != 0:
                raise ValueError(
                    f"layer {extent.layer_id} offset is not alignment compliant"
                )
            if extent.offset_bytes + extent.length_bytes > container.capacity_bytes:
                raise ValueError(
                    f"layer {extent.layer_id} exceeds container capacity"
                )
            ranges.append(
                (extent.offset_bytes, extent.offset_bytes + extent.length_bytes)
            )
        if self.storage_backend is StorageBackend.LOCAL_DISK:
            if self.read_strategy is not ReadStrategy.BATCHED:
                raise ValueError("local_disk objects use batched file reads")
            return
        ranges.sort()
        for previous, current in zip(ranges, ranges[1:]):
            if current[0] < previous[1]:
                raise ValueError("layer extents overlap")
        inferred = infer_read_strategy(ordered)
        if self.read_strategy is not inferred:
            raise ValueError(
                f"read_strategy={self.read_strategy.value} disagrees with "
                f"physical extents ({inferred.value})"
            )

    @property
    def identity_key(self) -> tuple[str, str, str, str]:
        return (
            self.skill_name,
            self.skill_version,
            self.model_fingerprint,
            self.tokenizer_fingerprint,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "skill_name": self.skill_name,
            "skill_version": self.skill_version,
            "model_fingerprint": self.model_fingerprint,
            "tokenizer_fingerprint": self.tokenizer_fingerprint,
            "token_count": self.token_count,
            "source_position_start": self.source_position_start,
            "token_ids_sha256": self.token_ids_sha256,
            "start_marker_token_ids": list(self.start_marker_token_ids),
            "container_id": self.container_id,
            "storage_backend": self.storage_backend.value,
            "read_strategy": self.read_strategy.value,
            "chunking": self.chunking.to_dict(),
            "storage_layout": self.storage_layout.value,
            "layers": [
                layer.to_dict()
                for layer in sorted(self.layers, key=lambda x: x.layer_id)
            ],
            "status": self.status.value,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CacheObjectMetadata":
        expected = {
            "object_id",
            "skill_name",
            "skill_version",
            "model_fingerprint",
            "tokenizer_fingerprint",
            "token_count",
            "source_position_start",
            "token_ids_sha256",
            "start_marker_token_ids",
            "container_id",
            "read_strategy",
            "layers",
            "status",
            "storage_backend",
            "chunking",
            "storage_layout",
        }
        legacy = expected - {"chunking", "storage_layout"}
        if set(payload) not in (expected, legacy):
            raise ValueError(
                f"CacheObjectMetadata fields differ: expected={sorted(expected)}, "
                f"actual={sorted(payload)}"
            )
        return cls(
            object_id=str(payload["object_id"]),
            skill_name=str(payload["skill_name"]),
            skill_version=str(payload["skill_version"]),
            model_fingerprint=str(payload["model_fingerprint"]),
            tokenizer_fingerprint=str(payload["tokenizer_fingerprint"]),
            token_count=int(payload["token_count"]),
            source_position_start=int(payload["source_position_start"]),
            token_ids_sha256=str(payload["token_ids_sha256"]),
            start_marker_token_ids=tuple(
                int(token) for token in payload["start_marker_token_ids"]
            ),
            container_id=(
                None
                if payload["container_id"] is None
                else str(payload["container_id"])
            ),
            storage_backend=StorageBackend(str(payload["storage_backend"])),
            read_strategy=ReadStrategy(str(payload["read_strategy"])),
            layers=tuple(LayerExtent.from_dict(item) for item in payload["layers"]),
            chunking=(
                ChunkingSpec(ChunkingMode.WHOLE_SKILL)
                if "chunking" not in payload
                else ChunkingSpec.from_dict(payload["chunking"])
            ),
            storage_layout=KVLayout(
                payload.get("storage_layout", KVLayout.CHUNK_SINGLE_LAYER.value)
            ),
            status=CacheObjectStatus(str(payload["status"])),
        )


def infer_read_strategy(layers: tuple[LayerExtent, ...]) -> ReadStrategy:
    """Infer whether all physical layer extents form one gap-free interval."""

    if any(layer.offset_bytes is None for layer in layers):
        return ReadStrategy.BATCHED
    by_offset = sorted(layers, key=lambda item: int(item.offset_bytes))
    contiguous = all(
        int(previous.offset_bytes) + previous.length_bytes
        == int(current.offset_bytes)
        for previous, current in zip(by_offset, by_offset[1:])
    )
    return ReadStrategy.CONTIGUOUS if contiguous else ReadStrategy.BATCHED
