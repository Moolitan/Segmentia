"""Offline metadata construction for LMCache LocalDisk Skill layers."""

from __future__ import annotations

import os
from pathlib import Path
import threading
from collections.abc import Sequence
import uuid
import hashlib

from ..base import CacheObjectMetadata, LayerExtent, ReadStrategy, StorageBackend
from ..manager import MetadataManager
from .base import LocalDiskCacheObjectBuildInput
from ...storage.formats.torch_pt import validate_region_file


class LocalDiskCacheBuilder:
    """Verify LMCache layer files and publish a key-based metadata snapshot."""

    def __init__(self, *, expected_layers: int) -> None:
        if expected_layers <= 0:
            raise ValueError("expected_layers must be > 0")
        self._expected_layers = expected_layers

    @staticmethod
    def _payload_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb", buffering=0) as handle:
            while block := handle.read(1024 * 1024):
                digest.update(block)
        return digest.hexdigest()

    def build_object(
        self,
        source: LocalDiskCacheObjectBuildInput,
    ) -> CacheObjectMetadata:
        if len(source.layers) != self._expected_layers:
            raise ValueError(
                f"expected {self._expected_layers} LocalDisk layers, "
                f"found {len(source.layers)}"
            )
        ordered = tuple(sorted(source.layers, key=lambda item: item.layer_id))
        if tuple(item.layer_id for item in ordered) != tuple(
            range(self._expected_layers)
        ):
            raise ValueError("LocalDisk layer IDs must cover the complete model")
        layers: list[LayerExtent] = []
        for layer in ordered:
            path = Path(layer.data_path)
            validate_region_file(path, layer.backend_key, layer.length_bytes)
            layers.append(
                LayerExtent(
                    layer_id=layer.layer_id,
                    backend_key=layer.backend_key,
                    offset_bytes=None,
                    length_bytes=layer.length_bytes,
                    dtype=layer.dtype,
                    shape=layer.shape,
                    memory_layout=layer.memory_layout,
                    payload_sha256=self._payload_sha256(path),
                )
            )
        metadata = CacheObjectMetadata(
            object_id=source.object_id,
            skill_name=source.skill_name,
            skill_version=source.skill_version,
            model_fingerprint=source.model_fingerprint,
            tokenizer_fingerprint=source.tokenizer_fingerprint,
            token_count=source.token_count,
            source_position_start=source.source_position_start,
            token_ids_sha256=source.token_ids_sha256,
            chunk_token_ids_sha256=source.chunk_token_ids_sha256,
            start_marker_token_ids=source.start_marker_token_ids,
            container_id=None,
            read_strategy=ReadStrategy.BATCHED,
            layers=tuple(layers),
            chunking=source.chunking,
            storage_layout=source.storage_layout,
            storage_backend=StorageBackend.LOCAL_DISK,
        )
        metadata.validate(self._expected_layers, None)
        return metadata

    def publish_objects(
        self,
        metadata_path: str | Path,
        sources: Sequence[LocalDiskCacheObjectBuildInput],
    ) -> tuple[CacheObjectMetadata, ...]:
        if not sources:
            raise ValueError("sources must be non-empty")
        built = tuple(self.build_object(source) for source in sources)
        identities = {
            (
                item.skill_name,
                item.model_fingerprint,
                item.tokenizer_fingerprint,
            )
            for item in built
        }
        if len(identities) != len(built):
            raise ValueError("LocalDisk batch contains duplicate Skill identities")
        destination = Path(metadata_path)
        retained: tuple[CacheObjectMetadata, ...] = ()
        if destination.exists():
            existing = MetadataManager(
                destination,
                expected_layers=self._expected_layers,
            )
            if existing.list_containers():
                raise ValueError("LocalDisk metadata must not contain raw containers")
            retained = tuple(
                item
                for item in existing.list_objects(include_invalidated=True)
                if (
                    item.skill_name,
                    item.model_fingerprint,
                    item.tokenizer_fingerprint,
                )
                not in identities
            )
        complete = tuple(sorted((*retained, *built), key=lambda item: item.object_id))
        publish_local_disk_snapshot(
            destination,
            complete,
            expected_layers=self._expected_layers,
            replace_existing=destination.exists(),
        )
        return built


def publish_local_disk_snapshot(
    metadata_path: str | Path,
    objects: Sequence[CacheObjectMetadata],
    *,
    expected_layers: int,
    replace_existing: bool = False,
) -> None:
    """Atomically publish a key-based LocalDisk metadata snapshot."""

    if not objects:
        raise ValueError("objects must be non-empty")
    ordered = tuple(sorted(objects, key=lambda item: item.object_id))
    for metadata in ordered:
        if metadata.storage_backend is not StorageBackend.LOCAL_DISK:
            raise ValueError("LocalDisk metadata contains a non-LocalDisk object")
        metadata.validate(expected_layers, None)

    destination = Path(metadata_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        existing = MetadataManager(destination, expected_layers=expected_layers)
        same_metadata = (
            not existing.list_containers()
            and existing.list_objects(include_invalidated=True) == ordered
        )
        if same_metadata:
            return
        if not replace_existing:
            raise ValueError("existing metadata contains different cache objects")

    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{threading.get_ident()}."
        f"{uuid.uuid4().hex}.tmp"
    )
    try:
        staging = MetadataManager(temporary, expected_layers=expected_layers)
        for metadata in ordered:
            staging.publish_object(metadata)
        verified = MetadataManager(temporary, expected_layers=expected_layers)
        if verified.list_containers():
            raise RuntimeError("staged LocalDisk metadata contains raw containers")
        if verified.list_objects(include_invalidated=True) != ordered:
            raise RuntimeError("staged LocalDisk objects failed round-trip")

        os.replace(temporary, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
