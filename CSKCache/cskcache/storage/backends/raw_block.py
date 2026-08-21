"""Raw-block SSD-to-pinned loading for complete Skill-layer groups."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any
import json
import os

from ...metadata.base import ContainerMetadata, StorageBackend
from ...metadata.manager import MetadataManager
from ..base import (
    CSKReadBatch,
    CSKReadResult,
    ExtentReadBackend,
    HostBufferPool,
)


def generation_sidecar_path(container: ContainerMetadata) -> Path:
    """Return the CSKCache-owned generation sidecar for a raw container."""

    return Path(f"{container.raw_file_path}.cskcache-generation.json")


def publish_generation_sidecar(container: ContainerMetadata) -> Path:
    """Atomically publish the identity of a newly built raw container."""

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


class RawBlockLoader:
    """Validate one raw container and read all layer extents as one workset."""

    storage_backend = StorageBackend.RAW_BLOCK

    def __init__(
        self,
        metadata_manager: MetadataManager,
        backend: ExtentReadBackend,
        host_buffer_pool: HostBufferPool | None,
    ) -> None:
        self._metadata_manager = metadata_manager
        self._backend = backend
        self._host_buffer_pool = host_buffer_pool

    def validate_container(self, container: ContainerMetadata | None) -> None:
        """Verify that LMCache is open on the Catalog-owned raw generation."""

        if container is None:
            raise ValueError("raw_block storage requires container metadata")
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
            raise ValueError(
                "LMCache backend capacity differs from CSKCache metadata"
            )
        if int(self._backend.block_align) != container.alignment_bytes:
            raise ValueError("LMCache backend alignment differs from CSKCache metadata")
        if int(self._backend.header_bytes) != container.header_bytes:
            raise ValueError(
                "LMCache backend header size differs from CSKCache metadata"
            )
        self._validate_generation_sidecar(container)

    def read_into(
        self,
        batch: CSKReadBatch,
        destination_memory_objects: Sequence[Any],
    ) -> CSKReadResult:
        """Read offset-addressed extents into caller-owned pinned buffers."""

        results = self._backend.read_extents_into(
            batch.offsets,
            batch.lengths,
            destination_memory_objects,
        )
        return CSKReadResult.from_backend_results(batch, results)

    def load(self, batch: CSKReadBatch) -> tuple[Any, ...]:
        """Allocate pinned layers and load the complete raw-block workset."""

        if self._host_buffer_pool is None:
            raise RuntimeError("raw_block loading has no host buffer pool")
        metadata = self._metadata_manager.get_object(batch.cache_object_id)
        assert metadata.container_id is not None
        container = self._metadata_manager.get_container(metadata.container_id)
        self.validate_container(container)
        source_objects = tuple(
            self._host_buffer_pool.acquire_persistent(batch.extents)
        )
        if len(source_objects) != len(batch.extents):
            self._host_buffer_pool.release(source_objects)
            raise RuntimeError("host buffer pool returned an incomplete layer group")
        read_result = self.read_into(batch, source_objects)
        if not read_result.complete:
            self._host_buffer_pool.release(source_objects)
            raise RuntimeError("raw_block returned an incomplete layer group")
        return tuple(
            self._host_buffer_pool.arrange_loaded_layers(
                batch.extents,
                source_objects,
            )
        )

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
