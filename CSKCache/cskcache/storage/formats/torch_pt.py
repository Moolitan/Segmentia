"""LMCache LocalDisk naming for torch-serialized region files."""

from __future__ import annotations

from pathlib import Path


def expected_region_filename(backend_key: str) -> str:
    if not backend_key:
        raise ValueError("backend_key must be non-empty")
    return backend_key.replace("/", "-") + ".pt"


def validate_region_file(path: Path, backend_key: str, length_bytes: int) -> None:
    if not path.is_absolute() or not path.is_file():
        raise FileNotFoundError(f"LocalDisk region file is missing: {path}")
    if path.name != expected_region_filename(backend_key):
        raise ValueError("LocalDisk region path does not match its backend key")
    if path.stat().st_size != length_bytes:
        raise ValueError("LocalDisk region file has an unexpected size")
