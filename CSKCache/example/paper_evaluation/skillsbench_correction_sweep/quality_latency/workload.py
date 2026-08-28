"""Validate frozen workloads and build exact one-object Catalog views."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Workload:
    task_id: str
    skill_name: str
    tier: str
    smoke: bool
    task_path: Path
    skill_path: Path
    relative_skill_path: str
    skill_version: str
    object_id: str
    skill_tokens: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def skillsbench_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return payload


def load_workloads(
    *,
    workload_path: Path,
    skillsbench_root: Path,
    manifest_path: Path,
    catalog_path: Path,
    expected_commit: str,
    expected_catalog_sha256: str,
) -> tuple[list[Workload], dict[str, Any]]:
    if skillsbench_commit(skillsbench_root) != expected_commit:
        raise RuntimeError("SkillsBench checkout differs from the frozen commit")
    actual_catalog_sha256 = sha256_file(catalog_path)
    if actual_catalog_sha256 != expected_catalog_sha256:
        raise RuntimeError(
            "offline Catalog SHA-256 differs from the verified v2 pool"
        )

    manifest = _read_object(manifest_path)
    catalog = _read_object(catalog_path)
    if manifest.get("status") != "verified":
        raise RuntimeError("offline manifest is not verified")
    if manifest.get("skillsbench_commit") != expected_commit:
        raise RuntimeError("offline manifest uses a different SkillsBench commit")
    if manifest.get("catalog_sha256") != actual_catalog_sha256:
        raise RuntimeError("offline manifest and Catalog digest disagree")

    catalog_objects = {
        str(item["object_id"]): item for item in catalog.get("objects", [])
    }
    source_index: dict[str, dict[str, Any]] = {}
    for item in manifest.get("objects", []):
        for source_path in item.get("source_paths", []):
            source_path = str(source_path)
            if source_path in source_index:
                raise RuntimeError(f"duplicate manifest source path: {source_path}")
            source_index[source_path] = item

    configured = json.loads(workload_path.read_text(encoding="utf-8"))
    if not isinstance(configured, list) or not configured:
        raise ValueError("workloads.json must contain a non-empty list")
    workloads: list[Workload] = []
    seen_tasks: set[str] = set()
    smoke_count = 0
    for raw in configured:
        if not isinstance(raw, dict):
            raise TypeError("each workload must be a JSON object")
        task_id = str(raw.get("task_id", "")).strip()
        skill_name = str(raw.get("skill_name", "")).strip()
        tier = str(raw.get("tier", "")).strip()
        smoke = bool(raw.get("smoke", False))
        if not task_id or task_id in seen_tasks:
            raise ValueError(f"invalid or duplicate task_id: {task_id!r}")
        if not skill_name or tier not in {"core", "extension"}:
            raise ValueError(f"invalid workload: {raw}")
        task_dir = skillsbench_root / "tasks" / task_id
        task_path = task_dir / "task.md"
        skill_path = task_dir / "environment/skills" / skill_name / "SKILL.md"
        if not task_path.is_file() or not skill_path.is_file():
            raise FileNotFoundError(f"incomplete workload: {task_id}/{skill_name}")
        task_skills = tuple(
            sorted(task_dir.glob("environment/skills/*/SKILL.md"))
        )
        if task_skills != (skill_path,):
            raise RuntimeError(f"workload is not a single-Skill task: {task_id}")
        relative = skill_path.relative_to(skillsbench_root / "tasks").as_posix()
        manifest_object = source_index.get(relative)
        if manifest_object is None:
            raise RuntimeError(f"manifest has no exact Skill source: {relative}")
        object_id = str(manifest_object["object_id"])
        catalog_object = catalog_objects.get(object_id)
        if catalog_object is None:
            raise RuntimeError(f"Catalog has no object {object_id}")
        for key in ("skill_name", "skill_version", "token_count"):
            if catalog_object.get(key) != manifest_object.get(key):
                raise RuntimeError(f"manifest/Catalog mismatch for {object_id}: {key}")
        if str(manifest_object["skill_name"]) != skill_name:
            raise RuntimeError(f"manifest Skill name mismatch for {relative}")
        if len(catalog_object.get("layers", [])) != int(catalog["expected_layers"]):
            raise RuntimeError(f"incomplete layer extents for {object_id}")
        workloads.append(
            Workload(
                task_id=task_id,
                skill_name=skill_name,
                tier=tier,
                smoke=smoke,
                task_path=task_path,
                skill_path=skill_path,
                relative_skill_path=relative,
                skill_version=str(manifest_object["skill_version"]),
                object_id=object_id,
                skill_tokens=int(manifest_object["token_count"]),
            )
        )
        seen_tasks.add(task_id)
        smoke_count += int(smoke)
    if smoke_count != 1 or not workloads[0].smoke:
        raise RuntimeError("exactly one smoke workload must appear first")
    return workloads, catalog


def write_catalog_view(
    master_catalog: dict[str, Any], object_id: str, output_path: Path
) -> None:
    selected = [
        item
        for item in master_catalog.get("objects", [])
        if str(item.get("object_id")) == object_id
    ]
    if len(selected) != 1:
        raise RuntimeError(f"expected one Catalog object for {object_id}")
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
