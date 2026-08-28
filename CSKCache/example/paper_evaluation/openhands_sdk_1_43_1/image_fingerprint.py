"""Helpers for auditing the immutable latest-SDK image."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent


def source_files() -> tuple[Path, ...]:
    return (
        ROOT / "Dockerfile",
        ROOT / "official-lock-requirements.txt",
        *(sorted((ROOT / "adapter").glob("*.py"))),
    )


def source_sha256() -> str:
    digest = hashlib.sha256()
    for path in source_files():
        relative = path.relative_to(ROOT).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def load_lock(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Image lock is not an object: {path}")
    return payload


def inspect_image(reference: str) -> dict[str, Any]:
    result = subprocess.run(
        ["docker", "image", "inspect", reference],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    if not isinstance(payload, list) or len(payload) != 1:
        raise RuntimeError(f"Unexpected docker inspect output for {reference}")
    image = payload[0]
    return {
        "image_id": image["Id"],
        "repo_digests": image.get("RepoDigests") or [],
        "architecture": image["Architecture"],
        "os": image["Os"],
        "labels": (image.get("Config") or {}).get("Labels") or {},
    }
