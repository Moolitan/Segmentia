"""Offset-addressed raw extent transfer into pinned host buffers."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any
import json

from ...metadata.base import ContainerMetadata, StorageBackend
from ...metadata.manager import MetadataManager
from ..backends.raw_block import generation_sidecar_path
from ..base import (
    CSKReadBatch,
    CSKReadResult,
    ExtentReadBackend,
    HostBufferPool,
)


class RawExtentTransfer:
    """Validate one raw container and transfer all extents as one workset."""

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
        """Verify that the backend is open on the Catalog-owned generation."""

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
        """Transfer offset-addressed extents into caller-owned pinned buffers."""

        results = self._backend.read_extents_into(
            batch.offsets,
            batch.lengths,
            destination_memory_objects,
        )
        return CSKReadResult.from_backend_results(batch, results)

    def load(self, batch: CSKReadBatch) -> tuple[Any, ...]:
        """Allocate pinned layers and transfer the complete raw workset."""

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

    def load_layer(self, batch: CSKReadBatch, layer_id: int) -> Any:
        """Allocate, read, and arrange one layer from a validated object."""

        if self._host_buffer_pool is None:
            raise RuntimeError("raw_block loading has no host buffer pool")
        if not 0 <= layer_id < len(batch.extents):
            raise ValueError("raw_block layer_id is outside the read batch")
        extent = batch.extents[layer_id]
        if extent.layer_id != layer_id:
            raise ValueError("raw_block batch is not in model-layer order")
        single = CSKReadBatch(
            cache_object_id=batch.cache_object_id,
            container_id=batch.container_id,
            extents=(extent,),
        )
        source_objects = tuple(
            self._host_buffer_pool.acquire_persistent(single.extents)
        )
        if len(source_objects) != 1:
            self._host_buffer_pool.release(source_objects)
            raise RuntimeError("host buffer pool did not return one layer")
        try:
            read_result = self.read_into(single, source_objects)
            if not read_result.complete:
                raise RuntimeError("raw_block returned an incomplete layer")
        except Exception:
            self._host_buffer_pool.release(source_objects)
            raise
        arranged = tuple(
            self._host_buffer_pool.arrange_loaded_layers(
                single.extents,
                source_objects,
            )
        )
        if len(arranged) != 1:
            self._host_buffer_pool.release(arranged)
            raise RuntimeError("host layout did not return one arranged layer")
        return arranged[0]

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
