#!/usr/bin/env python3
"""Copy the 11 measured Skill KV objects onto the mounted Samsung 990 PRO.

The script is intentionally separate from the loading benchmark: copying data
would warm Linux's page cache and invalidate the first cold-read measurement.
It never formats or mounts a device and refuses to write when ``/mnt/990_pro``
is merely an ordinary directory on the root filesystem.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from common import (
    atomic_write_json,
    discover_manifest_layers,
    load_config,
    require_mounted_device,
    resolve_skill_manifest,
)


COPY_BUFFER_BYTES = 8 * 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        while chunk := handle.read(COPY_BUFFER_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def copy_verified(source: Path, destination: Path) -> str:
    """Copy one file atomically and verify its complete SHA-256."""
    source_digest = sha256_file(source)
    if (
        destination.is_file()
        and destination.stat().st_size == source.stat().st_size
        and sha256_file(destination) == source_digest
    ):
        return source_digest

    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with source.open("rb", buffering=0) as source_handle, os.fdopen(
            descriptor, "wb", buffering=0
        ) as destination_handle:
            while chunk := source_handle.read(COPY_BUFFER_BYTES):
                destination_handle.write(chunk)
            destination_handle.flush()
            os.fsync(destination_handle.fileno())
        if temporary.stat().st_size != source.stat().st_size:
            raise IOError(f"copied file has the wrong size: {source}")
        if sha256_file(temporary) != source_digest:
            raise IOError(f"copied file failed SHA-256 verification: {source}")
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return source_digest


def selected_skills(config: dict[str, Any]) -> list[str]:
    cases = config["agent_schedule"].get("cases")
    settings = config["agent_kv_loading_actual"]
    excluded = {str(value) for value in settings["excluded_skills"]}
    if excluded != {"docx", "writing-systems-papers"}:
        raise ValueError(
            "agent_kv_loading_actual.excluded_skills must contain exactly "
            "docx and writing-systems-papers"
        )
    if not isinstance(cases, list):
        raise ValueError("agent_schedule.cases must be a list")
    skills = [
        str(case["skill"])
        for case in cases
        if str(case["skill"]) not in excluded
    ]
    if len(skills) != 11 or len(set(skills)) != 11:
        raise ValueError(f"expected 11 unique selected Skills, found {skills}")
    return skills


def copy_skill(
    source_pool: Path,
    destination_pool: Path,
    skill: str,
    expected_layers: int,
) -> dict[str, Any]:
    source_manifest_path = resolve_skill_manifest(source_pool, skill)
    source_manifest, source_layers = discover_manifest_layers(
        source_manifest_path, expected_layers
    )
    relative_directory = source_manifest_path.parent.relative_to(source_pool)
    destination_directory = destination_pool / relative_directory
    destination_kv = destination_directory / "kv"

    copied_files: list[dict[str, Any]] = []
    for layer in source_layers:
        layer_sources = [layer.path]
        historical_sidecar = Path(f"{layer.path}.meta.json")
        if historical_sidecar.is_file():
            layer_sources.append(historical_sidecar)
        for source in layer_sources:
            destination = destination_kv / source.name
            digest = copy_verified(source, destination)
            copied_files.append(
                {
                    "source": str(source),
                    "destination": str(destination),
                    "bytes": source.stat().st_size,
                    "sha256": digest,
                }
            )

    completed_marker = source_manifest_path.parent / "COMPLETED"
    if completed_marker.is_file():
        copy_verified(completed_marker, destination_directory / "COMPLETED")

    destination_manifest = dict(source_manifest)
    destination_manifest["cache_dir"] = str(destination_kv)
    destination_manifest["staging_dir"] = str(destination_pool / ".staging")
    destination_manifest_path = destination_directory / "manifest.json"
    atomic_write_json(destination_manifest_path, destination_manifest)

    checked_manifest, checked_layers = discover_manifest_layers(
        destination_manifest_path, expected_layers
    )
    source_bytes = sum(layer.size_bytes for layer in source_layers)
    destination_bytes = sum(layer.size_bytes for layer in checked_layers)
    if destination_bytes != source_bytes:
        raise ValueError(f"copied Skill byte count disagrees for {skill}")
    if checked_manifest.get("token_count") != source_manifest.get("token_count"):
        raise ValueError(f"copied Skill token count disagrees for {skill}")
    return {
        "skill": skill,
        "source_manifest": str(source_manifest_path),
        "destination_manifest": str(destination_manifest_path),
        "token_count": source_manifest.get("token_count"),
        "layer_count": len(checked_layers),
        "cache_bytes": destination_bytes,
        "files": copied_files,
    }


def main() -> None:
    config = load_config()
    fast_ssd = config["fast_ssd_skill_cache"]
    mount = require_mounted_device(
        Path(fast_ssd["mount_point"]),
        Path(fast_ssd["expected_source"]),
        writable=True,
    )
    source_pool = Path(config["skill_cache"]["pool_dir"]).resolve()
    destination_pool = Path(fast_ssd["pool_dir"]).resolve()
    mount_point = Path(fast_ssd["mount_point"]).resolve()
    destination_pool.relative_to(mount_point)
    expected_layers = int(config["skill_cache"]["expected_layers"])

    destination_pool.mkdir(parents=True, exist_ok=True)
    copied = []
    for skill in selected_skills(config):
        print(f"[copy] {skill}")
        copied.append(
            copy_skill(source_pool, destination_pool, skill, expected_layers)
        )
        print(f"[verified] {skill} layers={expected_layers}")

    summary_path = destination_pool / "990pro_copy_manifest.json"
    atomic_write_json(
        summary_path,
        {
            "schema_version": 1,
            "mount": mount,
            "source_pool": str(source_pool),
            "destination_pool": str(destination_pool),
            "skills": copied,
            "total_cache_bytes": sum(item["cache_bytes"] for item in copied),
        },
    )
    print(f"[completed] {summary_path} skills={len(copied)}")


if __name__ == "__main__":
    main()
