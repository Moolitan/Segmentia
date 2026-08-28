"""Authenticate the six fixed-length task--Skill workloads."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class Workload:
    task_id: str
    source_type: str
    skill_name: str
    skill_version: str
    object_id: str
    skill_tokens: int
    length_bucket: str
    task_path: Path
    skill_path: Path
    relative_skill_path: str


@dataclass(frozen=True)
class CuratedTask:
    task_id: str
    source_type: str
    task_path: Path
    skill_name: str
    skill_path: Path
    skill_tokens: int
    length_bucket: str
    skill_token_ids_sha256: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def load_curated_task(metadata_path: Path) -> CuratedTask:
    """Load one frozen repository-Skill task and authenticate its source text."""

    metadata_path = metadata_path.resolve()
    value = _read_object(metadata_path)
    if value.get("schema_version") != 1:
        raise RuntimeError("unsupported curated task schema")
    if value.get("source_type") != "curated_repository_task":
        raise RuntimeError("curated task has an unexpected source type")
    task_path = (metadata_path.parent / str(value["task_path"])).resolve()
    skill_path = (metadata_path.parent / str(value["skill_path"])).resolve()
    for label, path in (("task", task_path), ("Skill", skill_path)):
        if not path.is_file():
            raise FileNotFoundError(f"curated {label} source is missing: {path}")
    if sha256_file(task_path) != str(value["task_text_sha256"]):
        raise RuntimeError("curated task text differs from frozen metadata")
    if sha256_file(skill_path) != str(value["skill_text_sha256"]):
        raise RuntimeError("curated Skill text differs from frozen metadata")
    skill_tokens = int(value["qwen3_14b_skill_tokens"])
    length_bucket = str(value["length_bucket"])
    if length_bucket != ">10K" or skill_tokens <= 10_000:
        raise RuntimeError("curated proof task does not belong to the >10K bucket")
    return CuratedTask(
        task_id=str(value["task_id"]),
        source_type=str(value["source_type"]),
        task_path=task_path,
        skill_name=str(value["skill_name"]),
        skill_path=skill_path,
        skill_tokens=skill_tokens,
        length_bucket=length_bucket,
        skill_token_ids_sha256=str(value["skill_token_ids_sha256"]),
    )


def bucket_for_tokens(
    token_count: int,
    buckets: Sequence[tuple[str, int, int | None]],
) -> str:
    """Return the unique left-closed, right-open fixed-length bucket."""

    if token_count <= 0:
        raise ValueError("token_count must be positive")
    matches = [
        name
        for name, lower, upper in buckets
        if token_count >= lower and (upper is None or token_count < upper)
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"token count {token_count} belongs to {len(matches)} buckets"
        )
    return matches[0]


def eligible_for_all_ratios(
    token_count: int,
    *,
    max_ratio: float,
    minimum_full_recompute_tokens: int,
    block_alignment: int,
    minimum_reuse_tokens: int,
) -> bool:
    if block_alignment <= 0:
        raise ValueError("block_alignment must be positive")
    calibration = max(1, math.ceil(token_count * max_ratio))
    reusable_by_residue = []
    for segment_start in range(block_alignment):
        nominal_start = segment_start + minimum_full_recompute_tokens + calibration
        reuse_start = (
            (nominal_start + block_alignment - 1) // block_alignment
        ) * block_alignment
        reuse_end = (
            (segment_start + token_count) // block_alignment
        ) * block_alignment
        reusable_by_residue.append(reuse_end - reuse_start)
    return min(reusable_by_residue) >= minimum_reuse_tokens


def load_fixed_workloads(
    *,
    pool_root: Path,
    expected_model_id: str,
    buckets: Sequence[tuple[str, int, int | None]],
    max_ratio: float,
    minimum_full_recompute_tokens: int,
    block_alignment: int,
    minimum_reuse_tokens: int,
) -> tuple[list[Workload], dict[str, Any], dict[str, Any]]:
    """Load the verified six-object pool and fail on any source drift."""

    manifest_path = pool_root / "fixed_length_manifest.json"
    catalog_path = pool_root / "raw" / "catalog.json"
    for path in (manifest_path, catalog_path):
        if not path.is_file():
            raise FileNotFoundError(f"fixed-length offline artifact is missing: {path}")
    manifest = _read_object(manifest_path)
    catalog = _read_object(catalog_path)
    catalog_digest = sha256_file(catalog_path)
    if manifest.get("status") != "verified":
        raise RuntimeError("fixed-length offline manifest is not verified")
    if manifest.get("model_id") != expected_model_id:
        raise RuntimeError("fixed-length pool belongs to another model")
    if manifest.get("catalog_sha256") != catalog_digest:
        raise RuntimeError("fixed-length manifest and Catalog digest disagree")
    catalog_objects = {
        str(item["object_id"]): item for item in catalog.get("objects", [])
    }
    expected_layers = int(catalog["expected_layers"])
    workloads = []
    for record in manifest.get("workloads", []):
        task_path = Path(str(record["task_path"])).resolve()
        skill_path = Path(str(record["skill_path"])).resolve()
        for label, path, expected_hash in (
            ("task", task_path, str(record["task_text_sha256"])),
            ("Skill", skill_path, str(record["skill_text_sha256"])),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"fixed-length {label} source is missing: {path}")
            if sha256_file(path) != expected_hash:
                raise RuntimeError(f"fixed-length {label} source hash changed: {path}")
        token_count = int(record["skill_tokens"])
        assigned_bucket = str(record["length_bucket"])
        if bucket_for_tokens(token_count, buckets) != assigned_bucket:
            raise RuntimeError(f"workload is assigned to the wrong bucket: {record}")
        if not eligible_for_all_ratios(
            token_count,
            max_ratio=max_ratio,
            minimum_full_recompute_tokens=minimum_full_recompute_tokens,
            block_alignment=block_alignment,
            minimum_reuse_tokens=minimum_reuse_tokens,
        ):
            raise RuntimeError(f"workload leaves too few reusable tokens: {record}")
        object_id = str(record["object_id"])
        cached = catalog_objects.get(object_id)
        if cached is None:
            raise RuntimeError(f"Catalog has no selected object: {object_id}")
        for manifest_key, catalog_key in (
            ("skill_name", "skill_name"),
            ("skill_version", "skill_version"),
            ("skill_tokens", "token_count"),
            ("skill_token_ids_sha256", "token_ids_sha256"),
        ):
            if record.get(manifest_key) != cached.get(catalog_key):
                raise RuntimeError(f"manifest/Catalog mismatch: {object_id} {manifest_key}")
        if len(cached.get("layers", [])) != expected_layers:
            raise RuntimeError(f"incomplete layer extents: {object_id}")
        workloads.append(
            Workload(
                task_id=str(record["task_id"]),
                source_type=str(record["source_type"]),
                skill_name=str(record["skill_name"]),
                skill_version=str(record["skill_version"]),
                object_id=object_id,
                skill_tokens=token_count,
                length_bucket=assigned_bucket,
                task_path=task_path,
                skill_path=skill_path,
                relative_skill_path=str(record["relative_skill_path"]),
            )
        )
    bucket_order = [item[0] for item in buckets]
    by_bucket = {workload.length_bucket: workload for workload in workloads}
    if len(by_bucket) != len(workloads):
        raise RuntimeError("fixed-length manifest has duplicate buckets")
    if set(by_bucket) != set(bucket_order):
        raise RuntimeError("fixed-length manifest does not cover every bucket")
    ordered = [by_bucket[name] for name in bucket_order]
    for field, values in (
        ("task", [item.task_id for item in ordered]),
        ("Skill", [item.skill_name for item in ordered]),
        ("object", [item.object_id for item in ordered]),
    ):
        if len(values) != len(set(values)):
            raise RuntimeError(f"fixed-length selection repeats a {field}")
    metadata = {
        "manifest_path": str(manifest_path),
        "catalog_sha256": catalog_digest,
        "bucket_order": bucket_order,
        "workload_count": len(ordered),
        "repetitions_are_measurement_repeats": True,
    }
    return ordered, catalog, metadata


def write_catalog_view(
    master_catalog: dict[str, Any], workloads: Iterable[Workload], output_path: Path
) -> None:
    object_ids = {workload.object_id for workload in workloads}
    selected = [
        item
        for item in master_catalog.get("objects", [])
        if str(item.get("object_id")) in object_ids
    ]
    if len(selected) != len(object_ids):
        raise RuntimeError("Catalog view does not contain every selected object")
    names = [str(item.get("skill_name")) for item in selected]
    if len(names) != len(set(names)):
        raise RuntimeError("selected Catalog view contains ambiguous Skill names")
    view = {
        "catalog_version": master_catalog["catalog_version"],
        "expected_layers": master_catalog["expected_layers"],
        "containers": master_catalog["containers"],
        "objects": selected,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(view, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output_path)
