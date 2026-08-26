"""Public contract for one selected storage-to-pinned transfer."""

from __future__ import annotations

from typing import Any, Protocol

from ...metadata.base import ContainerMetadata, StorageBackend
from ..base import CSKReadBatch


class StorageTransfer(Protocol):
    """Transfer one validated physical batch through a selected backend."""

    storage_backend: StorageBackend

    def validate_container(self, container: ContainerMetadata | None) -> None: ...

    def load(self, batch: CSKReadBatch) -> tuple[Any, ...]: ...
