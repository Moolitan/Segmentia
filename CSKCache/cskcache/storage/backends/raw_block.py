"""Raw-block container identity and generation publication."""

from __future__ import annotations

from pathlib import Path
import json
import os

from ...metadata.base import ContainerMetadata


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
