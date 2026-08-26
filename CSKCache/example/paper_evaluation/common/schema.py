"""Stable, cross-platform CSV and JSONL result schema."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = 1

# A deliberately wide schema lets results from different subsections, models,
# and machines be concatenated without guessing column types or meanings.
SAMPLE_COLUMNS = (
    "schema_version",
    "run_id",
    "section",
    "case_id",
    "status",
    "platform_id",
    "hostname",
    "gpu_name",
    "model_id",
    "model_path",
    "tensor_parallel_size",
    "system",
    "skill_name",
    "skill_tokens",
    "task_id",
    "chunk_tokens",
    "mutation",
    "mutation_position",
    "storage_layout",
    "host_layout",
    "io_engine",
    "use_odirect",
    "correction_strategy",
    "correction_budget_tokens",
    "correction_ratio",
    "concurrency",
    "replica",
    "repetition",
    "warmup",
    "prompt_tokens",
    "reused_tokens",
    "reuse_ratio",
    "ttft_ms",
    "latency_ms",
    "batch_elapsed_ms",
    "throughput_requests_per_s",
    "rule_adherence",
    "rule_passed",
    "rule_total",
    "output_tokens",
    "fallback",
    "fallback_reason",
    "input_fingerprint",
    "started_utc",
    "completed_utc",
)

SUMMARY_COLUMNS = (
    "schema_version",
    "section",
    "platform_id",
    "model_id",
    "system",
    "skill_name",
    "skill_tokens",
    "task_id",
    "chunk_tokens",
    "mutation",
    "mutation_position",
    "storage_layout",
    "host_layout",
    "io_engine",
    "use_odirect",
    "correction_strategy",
    "correction_budget_tokens",
    "correction_ratio",
    "concurrency",
    "sample_count",
    "median_ttft_ms",
    "median_latency_ms",
    "median_throughput_requests_per_s",
    "median_reuse_ratio",
    "rule_adherence",
    "fallback_count",
    "input_fingerprint",
)


def normalize_row(
    row: Mapping[str, Any], columns: Sequence[str] = SAMPLE_COLUMNS
) -> dict[str, Any]:
    unknown = sorted(set(row) - set(columns))
    if unknown:
        raise ValueError(f"unknown CSV columns: {unknown}")
    normalized = {name: "" for name in columns}
    normalized.update(row)
    normalized["schema_version"] = SCHEMA_VERSION
    for name, value in tuple(normalized.items()):
        if value is None:
            normalized[name] = ""
        elif isinstance(value, bool):
            normalized[name] = "true" if value else "false"
        elif isinstance(value, (list, tuple, dict)):
            normalized[name] = json.dumps(
                value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
    return normalized


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n").encode()
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        os.write(descriptor, encoded)
    finally:
        os.close(descriptor)


def append_csv(
    path: Path,
    row: Mapping[str, Any],
    columns: Sequence[str] = SAMPLE_COLUMNS,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = normalize_row(row, columns)
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
        if needs_header:
            writer.writeheader()
        writer.writerow(normalized)
        handle.flush()
        os.fsync(handle.fileno())


def write_csv(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    columns: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow(normalize_row(row, columns))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
