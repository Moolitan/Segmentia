"""Direct raw-block construction and atomic CSKCache Catalog publication."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import threading
from typing import Any, Protocol, Sequence
import uuid

from .cache_metadata import (
    CacheObjectMetadata,
    CONTEXT_SEGMENT_FORMAT,
    ContainerMetadata,
    ContextSegmentTokenIdentity,
    LayerExtent,
    infer_read_strategy,
)
from .context_segment import render_context_segment
from .fingerprint import fingerprint_token_ids
from .metadata_manager import MetadataManager
from .storage_manager import publish_generation_sidecar


class RawOffsetNotFoundError(ValueError):
    """A direct-save layer is absent from LMCache's recovered raw index."""


def build_context_segment_token_identity(
    tokenizer: Any,
    skill_name: str,
    skill_text: str,
) -> ContextSegmentTokenIdentity:
    """Tokenize the exact Context Segment object used offline and online.

    The final newline is the Tool-message content boundary present in the
    current online prompt format. It belongs to the cached token object but
    not to the XML-like Context Segment itself.
    """

    observation_text = render_context_segment(skill_name, skill_text)
    cache_text = observation_text + "\n"
    token_ids = tuple(
        int(token_id)
        for token_id in tokenizer.encode(cache_text, add_special_tokens=False)
    )
    if not token_ids:
        raise RuntimeError(f"empty Context Segment token sequence for {skill_name}")
    opening_end = observation_text.find("\n") + 1
    if opening_end <= 0:
        raise RuntimeError("rendered Context Segment has no opening boundary")
    start_marker_text = observation_text[:opening_end]
    start_marker_token_ids = tuple(
        int(token_id)
        for token_id in tokenizer.encode(
            start_marker_text,
            add_special_tokens=False,
        )
    )
    if not start_marker_token_ids:
        raise RuntimeError(f"empty Context Segment marker for {skill_name}")
    return ContextSegmentTokenIdentity(
        context_format=CONTEXT_SEGMENT_FORMAT,
        observation_text=observation_text,
        cache_text=cache_text,
        token_ids=token_ids,
        token_ids_sha256=fingerprint_token_ids(token_ids),
        start_marker_text=start_marker_text,
        start_marker_token_ids=start_marker_token_ids,
        start_marker_token_ids_sha256=fingerprint_token_ids(
            start_marker_token_ids
        ),
    )


class OfflineOffsetBackend(Protocol):
    """LMCache slot information used after a direct offline raw save."""

    header_bytes: int

    def entry_offset(self, key: Any) -> int | None: ...


class DirectRawOffsetBackend(OfflineOffsetBackend, Protocol):
    """Physical facts exposed by LMCache after a direct raw-block save."""

    device_path: str
    capacity_bytes: int
    block_align: int


@dataclass(frozen=True)
class LayerBuildInput:
    """Tensor facts needed to publish one persistent layer extent."""

    layer_id: int
    backend_key: str
    lookup_key: Any
    length_bytes: int
    dtype: str
    shape: tuple[int, ...]
    memory_layout: str
    payload_sha256: str


@dataclass(frozen=True)
class CacheObjectBuildInput:
    """Semantic identity and all layers of one offline Skill KV object."""

    object_id: str
    skill_name: str
    skill_version: str
    model_fingerprint: str
    tokenizer_fingerprint: str
    token_count: int
    source_position_start: int
    token_ids_sha256: str
    start_marker_token_ids: tuple[int, ...]
    layers: tuple[LayerBuildInput, ...]


@dataclass(frozen=True)
class DirectRawLayerBuildInput:
    """Layer identity and tensor geometry before raw payload verification."""

    layer_id: int
    backend_key: str
    lookup_key: Any
    length_bytes: int
    dtype: str
    shape: tuple[int, ...]
    memory_layout: str


@dataclass(frozen=True)
class DirectRawCacheObjectBuildInput:
    """One exact-save object already written into the shared raw container."""

    object_id: str
    skill_name: str
    skill_version: str
    model_fingerprint: str
    tokenizer_fingerprint: str
    token_count: int
    source_position_start: int
    token_ids_sha256: str
    start_marker_token_ids: tuple[int, ...]
    layers: tuple[DirectRawLayerBuildInput, ...]


class CacheBuilder:
    """Translate LMCache raw-slot offsets into CSKCache-owned extents."""

    def __init__(
        self,
        backend: OfflineOffsetBackend,
        container: ContainerMetadata,
        *,
        expected_layers: int,
    ) -> None:
        if expected_layers <= 0:
            raise ValueError("expected_layers must be > 0")
        container.validate()
        if int(backend.header_bytes) != container.header_bytes:
            raise ValueError("LMCache header size differs from container metadata")
        self._backend = backend
        self._container = container
        self._expected_layers = expected_layers

    def build_object(self, source: CacheObjectBuildInput) -> CacheObjectMetadata:
        """Resolve raw offsets once; this method is never used online."""

        if len(source.layers) != self._expected_layers:
            raise ValueError(
                f"expected {self._expected_layers} build layers, "
                f"found {len(source.layers)}"
            )
        ordered = tuple(sorted(source.layers, key=lambda item: item.layer_id))
        layer_ids = tuple(item.layer_id for item in ordered)
        if layer_ids != tuple(range(self._expected_layers)):
            raise ValueError("build layer IDs must cover the complete model")

        extents: list[LayerExtent] = []
        for layer in ordered:
            slot_offset = self._backend.entry_offset(layer.lookup_key)
            if slot_offset is None:
                raise RawOffsetNotFoundError(
                    f"LMCache has no raw offset for layer {layer.layer_id}"
                )
            extents.append(
                LayerExtent(
                    layer_id=layer.layer_id,
                    backend_key=layer.backend_key,
                    offset_bytes=int(slot_offset) + self._container.header_bytes,
                    length_bytes=layer.length_bytes,
                    dtype=layer.dtype,
                    shape=layer.shape,
                    memory_layout=layer.memory_layout,
                    payload_sha256=layer.payload_sha256,
                )
            )

        extent_tuple = tuple(extents)
        metadata = CacheObjectMetadata(
            object_id=source.object_id,
            skill_name=source.skill_name,
            skill_version=source.skill_version,
            model_fingerprint=source.model_fingerprint,
            tokenizer_fingerprint=source.tokenizer_fingerprint,
            token_count=source.token_count,
            source_position_start=source.source_position_start,
            token_ids_sha256=source.token_ids_sha256,
            start_marker_token_ids=source.start_marker_token_ids,
            container_id=self._container.container_id,
            read_strategy=infer_read_strategy(extent_tuple),
            layers=extent_tuple,
        )
        metadata.validate(self._expected_layers, self._container)
        return metadata


class DirectRawCacheBuilder:
    """Verify direct LMCache raw writes and atomically publish the Catalog.

    vLLM and LMCache have already copied every model layer into one raw-block
    container. This builder never reads or creates intermediate ``.pt`` files.
    It resolves each key's final slot once, hashes the payload bytes at that
    extent, validates the complete layer group, and only then replaces the
    persistent Catalog.
    """

    def __init__(
        self,
        backend: DirectRawOffsetBackend,
        container: ContainerMetadata,
        *,
        expected_layers: int,
    ) -> None:
        if expected_layers <= 0:
            raise ValueError("expected_layers must be > 0")
        container.validate()
        raw_path = Path(container.raw_file_path).resolve()
        if Path(backend.device_path).resolve() != raw_path:
            raise ValueError("LMCache raw path differs from CSKCache container")
        if int(backend.capacity_bytes) != container.capacity_bytes:
            raise ValueError("LMCache capacity differs from container metadata")
        if int(backend.block_align) != container.alignment_bytes:
            raise ValueError("LMCache alignment differs from container metadata")
        if int(backend.header_bytes) != container.header_bytes:
            raise ValueError("LMCache header size differs from container metadata")
        if not raw_path.is_file():
            raise FileNotFoundError(f"raw-block container is missing: {raw_path}")
        if raw_path.stat().st_size != container.capacity_bytes:
            raise ValueError("raw-block file size differs from container capacity")
        self._backend = backend
        self._container = container
        self._expected_layers = expected_layers
        self._raw_path = raw_path

    def _payload_sha256(self, offset_bytes: int, length_bytes: int) -> str:
        digest = hashlib.sha256()
        remaining = length_bytes
        with self._raw_path.open("rb", buffering=0) as handle:
            handle.seek(offset_bytes)
            while remaining:
                block = handle.read(min(1024 * 1024, remaining))
                if not block:
                    raise ValueError(
                        "raw-block payload ended before the recorded extent"
                    )
                digest.update(block)
                remaining -= len(block)
        return digest.hexdigest()

    def build_object(
        self, source: DirectRawCacheObjectBuildInput
    ) -> CacheObjectMetadata:
        """Resolve and verify one complete exact-save object from raw storage."""

        if len(source.layers) != self._expected_layers:
            raise ValueError(
                f"expected {self._expected_layers} direct raw layers, "
                f"found {len(source.layers)}"
            )
        ordered = tuple(sorted(source.layers, key=lambda item: item.layer_id))
        layer_ids = tuple(item.layer_id for item in ordered)
        if layer_ids != tuple(range(self._expected_layers)):
            raise ValueError("direct raw layer IDs must cover the complete model")

        verified_layers: list[LayerBuildInput] = []
        for layer in ordered:
            slot_offset = self._backend.entry_offset(layer.lookup_key)
            if slot_offset is None:
                raise RawOffsetNotFoundError(
                    f"LMCache has no direct raw offset for layer {layer.layer_id}"
                )
            payload_offset = int(slot_offset) + self._container.header_bytes
            if payload_offset % self._container.alignment_bytes != 0:
                raise ValueError(
                    f"layer {layer.layer_id} payload offset is not aligned"
                )
            if (
                payload_offset < self._container.header_bytes
                or payload_offset + layer.length_bytes
                > self._container.capacity_bytes
            ):
                raise ValueError(
                    f"layer {layer.layer_id} payload exceeds the raw container"
                )
            verified_layers.append(
                LayerBuildInput(
                    layer_id=layer.layer_id,
                    backend_key=layer.backend_key,
                    lookup_key=layer.lookup_key,
                    length_bytes=layer.length_bytes,
                    dtype=layer.dtype,
                    shape=layer.shape,
                    memory_layout=layer.memory_layout,
                    payload_sha256=self._payload_sha256(
                        payload_offset, layer.length_bytes
                    ),
                )
            )

        verified_source = CacheObjectBuildInput(
            object_id=source.object_id,
            skill_name=source.skill_name,
            skill_version=source.skill_version,
            model_fingerprint=source.model_fingerprint,
            tokenizer_fingerprint=source.tokenizer_fingerprint,
            token_count=source.token_count,
            source_position_start=source.source_position_start,
            token_ids_sha256=source.token_ids_sha256,
            start_marker_token_ids=source.start_marker_token_ids,
            layers=tuple(verified_layers),
        )
        return CacheBuilder(
            self._backend,
            self._container,
            expected_layers=self._expected_layers,
        ).build_object(verified_source)

    def publish_objects(
        self,
        metadata_path: str | Path,
        sources: Sequence[DirectRawCacheObjectBuildInput],
    ) -> tuple[CacheObjectMetadata, ...]:
        """Verify a direct-save batch and publish one append-only snapshot.

        Every source is fully verified before the existing Catalog is touched.
        Rebuilding a Skill replaces the active object for the same
        Skill/model/tokenizer identity; its old raw slot becomes unreachable
        and can be reclaimed by a future offline compaction.
        """

        if not sources:
            raise ValueError("sources must be non-empty")
        built = tuple(self.build_object(source) for source in sources)
        identities = [
            (
                item.skill_name,
                item.model_fingerprint,
                item.tokenizer_fingerprint,
            )
            for item in built
        ]
        if len(set(identities)) != len(identities):
            raise ValueError("direct raw batch contains duplicate Skill identities")

        destination = Path(metadata_path)
        retained: tuple[CacheObjectMetadata, ...] = ()
        if destination.exists():
            existing = MetadataManager(
                destination, expected_layers=self._expected_layers
            )
            if existing.list_containers() != (self._container,):
                raise ValueError("existing Catalog describes a different container")
            replaced = set(identities)
            retained = tuple(
                item
                for item in existing.list_objects(include_invalidated=True)
                if (
                    item.skill_name,
                    item.model_fingerprint,
                    item.tokenizer_fingerprint,
                )
                not in replaced
            )

        complete_snapshot = tuple(
            sorted((*retained, *built), key=lambda item: item.object_id)
        )
        publish_cache_snapshot(
            destination,
            self._container,
            complete_snapshot,
            expected_layers=self._expected_layers,
            replace_existing=destination.exists(),
        )
        return built


def publish_cache_snapshot(
    metadata_path: str | Path,
    container: ContainerMetadata,
    objects: Sequence[CacheObjectMetadata],
    *,
    expected_layers: int,
    replace_existing: bool = False,
) -> None:
    """Publish one complete container snapshot without partial visibility."""

    if not objects:
        raise ValueError("objects must be non-empty")
    ordered = tuple(sorted(objects, key=lambda item: item.object_id))
    for metadata in ordered:
        metadata.validate(expected_layers, container)

    destination = Path(metadata_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        existing = MetadataManager(destination, expected_layers=expected_layers)
        same_container = existing.list_containers() == (container,)
        same_objects = existing.list_objects(include_invalidated=True) == ordered
        if same_container and same_objects:
            publish_generation_sidecar(container)
            return
        if not replace_existing and not same_container:
            raise ValueError("existing metadata contains a different container")
        if not replace_existing and not same_objects:
            raise ValueError("existing metadata contains different cache objects")

    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{threading.get_ident()}."
        f"{uuid.uuid4().hex}.tmp"
    )
    try:
        staging = MetadataManager(temporary, expected_layers=expected_layers)
        staging.publish_container(container)
        for metadata in ordered:
            staging.publish_object(metadata)
        verified = MetadataManager(temporary, expected_layers=expected_layers)
        if verified.list_containers() != (container,):
            raise RuntimeError("staged container metadata failed round-trip")
        if verified.list_objects(include_invalidated=True) != ordered:
            raise RuntimeError("staged cache objects failed round-trip")

        publish_generation_sidecar(container)
        os.replace(temporary, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
