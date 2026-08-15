#!/usr/bin/env python3
"""Shared configuration and result helpers for hardware I/O characterization."""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPOSITORY_ROOT / "CSKCache"))
from cskcache import SOURCE_ARTIFACT_TYPE

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "config.yaml"
LAYER_PATTERN = re.compile(r"@(\d+)\.pt$")


@dataclass(frozen=True)
class LayerFile:
    layer_id: int
    path: Path
    size_bytes: int
    metadata: dict[str, Any] | None = None


def load_config() -> dict[str, Any]:
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"configuration is not a mapping: {CONFIG_PATH}")
    for section in (
        "run",
        "skill_cache",
        "ssd",
        "dram",
        "pcie",
        "end_to_end",
        "agent_schedule",
    ):
        if section not in payload or not isinstance(payload[section], dict):
            raise ValueError(f"missing configuration section: {section}")
    run_id = str(payload["run"].get("id", "")).strip()
    if not run_id or "/" in run_id:
        raise ValueError("run.id must be a non-empty path-safe name")
    return payload


def config_sha256() -> str:
    return hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()


def raw_run_dir(config: dict[str, Any]) -> Path:
    return Path(config["run"]["raw_output_root"]) / str(config["run"]["id"])


def result_dir(config: dict[str, Any]) -> Path:
    return Path(config["run"]["result_dir"])


def require_mounted_device(
    mount_point: Path,
    expected_source: Path,
    *,
    writable: bool,
) -> dict[str, str]:
    """Fail unless *mount_point* is the expected standalone filesystem.

    ``findmnt --target`` also succeeds for an ordinary directory by returning
    the containing root filesystem.  Comparing the reported target with the
    requested mount point prevents a missing NVMe mount from silently filling
    the system disk.
    """
    mount_point = mount_point.resolve()
    command = [
        "findmnt",
        "--json",
        "--target",
        str(mount_point),
        "--output",
        "SOURCE,TARGET,FSTYPE,OPTIONS",
    ]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    filesystems = payload.get("filesystems")
    if not isinstance(filesystems, list) or len(filesystems) != 1:
        raise RuntimeError(f"cannot resolve one filesystem for {mount_point}")
    filesystem = filesystems[0]
    target = Path(str(filesystem.get("target", ""))).resolve()
    source = Path(str(filesystem.get("source", ""))).resolve()
    expected_source = expected_source.resolve()
    options = str(filesystem.get("options", ""))
    if target != mount_point:
        raise RuntimeError(
            f"{mount_point} is not a mount point; findmnt resolved its "
            f"containing filesystem at {target}"
        )
    if source != expected_source:
        raise RuntimeError(
            f"{mount_point} is backed by {source}, expected {expected_source}"
        )
    if writable and "rw" not in options.split(","):
        raise RuntimeError(f"{mount_point} is not mounted read-write: {options}")
    return {
        "source": str(source),
        "target": str(target),
        "fstype": str(filesystem.get("fstype", "")),
        "options": options,
    }


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def result_envelope(test_name: str, config: dict[str, Any], **fields: Any) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "test": test_name,
        "run_id": str(config["run"]["id"]),
        "created_unix_ns": time.time_ns(),
        "config_path": str(CONFIG_PATH),
        "config_sha256": config_sha256(),
        **fields,
    }


def write_test_result(test_name: str, config: dict[str, Any], payload: dict[str, Any]) -> Path:
    path = raw_run_dir(config) / f"{test_name}.json"
    atomic_write_json(path, result_envelope(test_name, config, **payload))
    return path


def discover_layers(config: dict[str, Any]) -> tuple[dict[str, Any], list[LayerFile]]:
    cache = config["skill_cache"]
    manifest_path = (
        Path(cache["pool_dir"]) / str(cache["cache_id"]) / "manifest.json"
    )
    return discover_manifest_layers(manifest_path, int(cache["expected_layers"]))


def resolve_skill_manifest(pool_dir: Path, skill: str) -> Path:
    """Resolve one offline manifest using the same collection precedence as 07."""
    preferred = (
        pool_dir / skill / "manifest.json",
        pool_dir / "Auto-claude-code-research-in-sleep" / skill / "manifest.json",
        pool_dir / "superpowers" / skill / "manifest.json",
    )
    manifest_path = next((path for path in preferred if path.is_file()), None)
    if manifest_path is not None:
        return manifest_path
    candidates = sorted(pool_dir.glob(f"**/{skill}/manifest.json"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"cannot resolve one offline manifest for {skill}: {candidates}"
        )
    return candidates[0]


def discover_manifest_layers(
    manifest_path: Path, expected_layers: int
) -> tuple[dict[str, Any], list[LayerFile]]:
    """Validate a completed manifest and return its layer files in model order."""
    if not manifest_path.is_file():
        raise FileNotFoundError(f"offline manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("artifact_type") != SOURCE_ARTIFACT_TYPE:
        raise ValueError(
            f"offline object is not a CSKCache source object: {manifest_path}"
        )
    if manifest.get("status") != "completed":
        raise ValueError(f"offline cache is not completed: {manifest_path}")
    if manifest.get("layer_count") != expected_layers:
        raise ValueError(
            f"expected {expected_layers} manifest layers: {manifest_path}"
        )
    data_files = manifest.get("data_files")
    if not isinstance(data_files, list) or len(data_files) != expected_layers:
        raise ValueError(
            f"manifest data_files must contain {expected_layers} entries: "
            f"{manifest_path}"
        )
    kv_dir = Path(str(manifest.get("cache_dir", "")))
    if not kv_dir.is_dir():
        raise FileNotFoundError(f"offline cache directory is missing: {kv_dir}")
    layer_records = manifest.get("layers")
    if not isinstance(layer_records, list) or len(layer_records) != expected_layers:
        raise ValueError(
            f"manifest layers must contain {expected_layers} entries: "
            f"{manifest_path}"
        )
    records_by_name: dict[str, dict[str, Any]] = {}
    for item in layer_records:
        if not isinstance(item, dict):
            raise ValueError(f"manifest layer record is malformed: {manifest_path}")
        filename = str(item.get("data_file", ""))
        if not filename or filename in records_by_name:
            raise ValueError(
                f"manifest layer data_file is missing or duplicated: {manifest_path}"
            )
        records_by_name[filename] = item

    files: list[LayerFile] = []
    for filename in data_files:
        path = kv_dir / str(filename)
        if not path.is_file():
            raise FileNotFoundError(f"offline layer file is missing: {path}")
        match = LAYER_PATTERN.search(path.name)
        if match is None:
            raise ValueError(f"cannot parse layer id from {path.name}")
        layer_id = int(match.group(1))
        size = path.stat().st_size
        layer_metadata = records_by_name.get(path.name)
        if layer_metadata is None:
            raise ValueError(f"manifest has no layer metadata for {path.name}")
        if int(layer_metadata.get("layer_id", -1)) != layer_id:
            raise ValueError(f"layer ID disagrees with manifest: {path}")
        if int(layer_metadata.get("size_bytes", -1)) != size:
            raise ValueError(f"layer size disagrees with manifest: {path}")
        files.append(LayerFile(layer_id, path, size, layer_metadata))
    files.sort(key=lambda item: item.layer_id)
    actual_ids = [item.layer_id for item in files]
    if actual_ids != list(range(expected_layers)):
        raise ValueError(
            f"expected layer ids 0..{expected_layers - 1}, found {actual_ids}"
        )
    manifest_files = set(data_files)
    actual_files = {item.path.name for item in files}
    if manifest_files != actual_files:
        raise ValueError("manifest data_files disagree with the layer files on disk")
    return manifest, files


def percentile(values: Iterable[float], q: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot compute a percentile of an empty sequence")
    if not 0.0 <= q <= 1.0:
        raise ValueError("percentile q must be in [0, 1]")
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def gib_per_second(size_bytes: int, duration_ms: float) -> float:
    if duration_ms <= 0:
        raise ValueError("duration must be positive")
    return float(size_bytes) / (1024**3) / (duration_ms / 1000.0)


def parse_cpu_list(specification: str) -> list[int]:
    cpus: list[int] = []
    for part in specification.strip().split(","):
        if not part:
            continue
        if "-" in part:
            start, end = (int(value) for value in part.split("-", 1))
            cpus.extend(range(start, end + 1))
        else:
            cpus.append(int(part))
    return sorted(set(cpus))


def numa_nodes() -> dict[int, list[int]]:
    nodes: dict[int, list[int]] = {}
    for path in sorted(Path("/sys/devices/system/node").glob("node[0-9]*")):
        node_id = int(path.name.removeprefix("node"))
        cpulist = path / "cpulist"
        if cpulist.is_file():
            cpus = parse_cpu_list(cpulist.read_text(encoding="utf-8"))
            if cpus:
                nodes[node_id] = cpus
    if not nodes:
        nodes[0] = sorted(os.sched_getaffinity(0))
    return nodes


def metadata_for_layers(manifest: dict[str, Any], layers: list[LayerFile]) -> dict[str, Any]:
    sizes = [layer.size_bytes for layer in layers]
    return {
        "cache_id": manifest.get("cache_id"),
        "model_path": manifest.get("model_path"),
        "token_count": manifest.get("token_count"),
        "layer_count": len(layers),
        "total_bytes": sum(sizes),
        "minimum_layer_bytes": min(sizes),
        "maximum_layer_bytes": max(sizes),
        "uniform_layer_size": len(set(sizes)) == 1,
    }
