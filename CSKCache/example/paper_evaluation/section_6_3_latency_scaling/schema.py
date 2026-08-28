"""Stable artifacts for the host-resident SkillsBench TTFT run."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

from paper_evaluation.common.schema import write_csv


SAMPLE_COLUMNS = (
    "schema_version",
    "run_id",
    "section",
    "case_id",
    "status",
    "invalid_reason",
    "platform_id",
    "gpu_name",
    "model_id",
    "model_path",
    "task_id",
    "source_type",
    "skill_name",
    "skill_version",
    "object_id",
    "skill_tokens",
    "length_bucket",
    "repetition",
    "system",
    "system_family",
    "correction_strategy",
    "requested_calibration_ratio",
    "expected_calibration_tokens",
    "actual_calibration_tokens",
    "requested_recompute_ratio",
    "expected_recomputed_tokens",
    "actual_recomputed_tokens",
    "deviation_check_layer",
    "prompt_tokens",
    "vllm_cached_tokens",
    "profile_reused_tokens",
    "reuse_ratio",
    "host_cache_mode",
    "host_cache_prepared",
    "ttft_ms",
    "client_ttft_ms",
    "request_latency_ms",
    "output_tokens",
    "fallback",
    "fallback_reason",
    "attempt_dir",
    "server_dir",
    "started_utc",
    "completed_utc",
)

SUMMARY_COLUMNS = (
    "schema_version",
    "platform_id",
    "model_id",
    "length_bucket",
    "system",
    "sample_count",
    "task_count",
    "mean_ttft_ms",
    "ci95_low_ttft_ms",
    "ci95_high_ttft_ms",
    "std_ttft_ms",
    "mean_prompt_tokens",
    "mean_skill_tokens",
    "mean_reuse_ratio",
    "fallback_count",
)

WORKLOAD_COLUMNS = (
    "schema_version",
    "selection_order",
    "source_type",
    "length_bucket",
    "task_id",
    "skill_name",
    "skill_version",
    "object_id",
    "skill_tokens",
    "relative_skill_path",
)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def collect_samples(run_dir: Path) -> list[dict[str, Any]]:
    newest: dict[str, tuple[str, dict[str, Any]]] = {}
    for path in sorted((run_dir / "cases").glob("*/attempt-*/sample.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise TypeError(f"sample is not a JSON object: {path}")
        case_id = str(value.get("case_id", ""))
        if not case_id:
            raise RuntimeError(f"sample has no case_id: {path}")
        attempt = path.parent.name
        current = newest.get(case_id)
        if current is None or attempt > current[0]:
            newest[case_id] = (attempt, value)
    return [newest[key][1] for key in sorted(newest)]


def write_sample_tables(run_dir: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    materialized = [dict(row) for row in rows]
    write_csv(run_dir / "samples.csv", materialized, SAMPLE_COLUMNS)
    temporary = run_dir / f".samples.jsonl.{os.getpid()}.tmp"
    temporary.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in materialized
        ),
        encoding="utf-8",
    )
    os.replace(temporary, run_dir / "samples.jsonl")
