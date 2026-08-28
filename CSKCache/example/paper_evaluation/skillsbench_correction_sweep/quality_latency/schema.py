"""Stable result columns and crash-safe JSON helpers for this sweep."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

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
    "tier",
    "smoke",
    "skill_name",
    "skill_version",
    "object_id",
    "skill_tokens",
    "system",
    "system_family",
    "correction_strategy",
    "requested_calibration_ratio",
    "expected_calibration_tokens",
    "actual_calibration_tokens",
    "actual_calibration_ratio",
    "matched_tokens",
    "reused_tokens",
    "reuse_start",
    "reuse_end",
    "profile_layer",
    "calibration_forward_ms",
    "residual_correction_ms",
    "calibration_compute_ms",
    "layer_gpu_ms",
    "prompt_tokens",
    "vllm_cached_tokens",
    "ttft_ms",
    "request_latency_ms",
    "output_tokens",
    "finish_reason",
    "thinking_extraction_source",
    "thinking_chars",
    "thinking_words",
    "fallback",
    "fallback_reason",
    "attempt_dir",
    "response_path",
    "thinking_path",
    "content_path",
    "catalog_view_path",
    "started_utc",
    "completed_utc",
)

PAIRED_COLUMNS = SAMPLE_COLUMNS + (
    "reference_case_id",
    "reference_thinking_path",
    "reference_thinking_words",
    "rouge_l_recall",
)

SUMMARY_COLUMNS = (
    "schema_version",
    "requested_calibration_ratio",
    "system",
    "sample_count",
    "task_count",
    "median_rouge_l_recall",
    "q1_rouge_l_recall",
    "q3_rouge_l_recall",
    "median_calibration_compute_ms",
    "q1_calibration_compute_ms",
    "q3_calibration_compute_ms",
    "median_calibration_forward_ms",
    "median_residual_correction_ms",
    "median_actual_calibration_tokens",
    "median_reused_tokens",
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
    """Keep the newest completed sample for each deterministic case ID."""

    newest: dict[str, tuple[str, dict[str, Any]]] = {}
    for path in sorted((run_dir / "cases").glob("*/attempt-*/sample.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError(f"sample is not a JSON object: {path}")
        case_id = str(payload.get("case_id", ""))
        if not case_id:
            raise RuntimeError(f"sample has no case_id: {path}")
        attempt_key = path.parent.name
        current = newest.get(case_id)
        if current is None or attempt_key > current[0]:
            newest[case_id] = (attempt_key, payload)
    return [newest[key][1] for key in sorted(newest)]


def write_sample_tables(run_dir: Path, samples: list[Mapping[str, Any]]) -> None:
    write_csv(run_dir / "samples.csv", samples, SAMPLE_COLUMNS)
    temporary = run_dir / f".samples.jsonl.{os.getpid()}.tmp"
    temporary.write_text(
        "".join(
            json.dumps(dict(sample), ensure_ascii=False, sort_keys=True) + "\n"
            for sample in samples
        ),
        encoding="utf-8",
    )
    os.replace(temporary, run_dir / "samples.jsonl")
