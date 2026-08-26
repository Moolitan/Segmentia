"""Measure layout completion latency without the final model layer."""

from __future__ import annotations

import csv
import importlib
import json
import os
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from config import MAX_MODEL_LEN, MAX_TOKENS, PREFIX_TOKENS, TAIL_TOKENS


SWEEP_CONFIG_MODULE = os.environ.get(
    "CSKCACHE_COMPLETION_SWEEP_CONFIG",
    "chunk_single_layer_completion_config",
)
SWEEP_CONFIG = importlib.import_module(SWEEP_CONFIG_MODULE)
CALIBRATION_TOKENS = SWEEP_CONFIG.CALIBRATION_TOKENS
CASE_RETRIES = SWEEP_CONFIG.CASE_RETRIES
CHUNK_SIZE_TOKEN_VALUES = SWEEP_CONFIG.CHUNK_SIZE_TOKEN_VALUES
COMPLETION_LAYER_STOP = SWEEP_CONFIG.COMPLETION_LAYER_STOP
EXECUTION_ORDER = SWEEP_CONFIG.EXECUTION_ORDER
EXPECTED_LAYERS = SWEEP_CONFIG.EXPECTED_LAYERS
HOST_LAYOUT = SWEEP_CONFIG.HOST_LAYOUT
MAX_CONSECUTIVE_FAILURES = SWEEP_CONFIG.MAX_CONSECUTIVE_FAILURES
REPETITIONS = SWEEP_CONFIG.REPETITIONS
RUN_NAME = SWEEP_CONFIG.RUN_NAME
SKILL_TOKEN_VALUES = SWEEP_CONFIG.SKILL_TOKEN_VALUES
STORAGE_LAYOUT = SWEEP_CONFIG.STORAGE_LAYOUT
SWEEP_OUTPUT_ROOT = SWEEP_CONFIG.SWEEP_OUTPUT_ROOT
WARMUP_REQUESTS = SWEEP_CONFIG.WARMUP_REQUESTS

EXAMPLE_DIR = Path(__file__).resolve().parent
RUN_SCRIPT = EXAMPLE_DIR / "run.py"
REQUIRED_CASE_FILES = (
    "cskcache_profile.jsonl",
    "request_result.json",
    "summary.json",
    "warmup_profile.jsonl",
    "warmup_request_result.json",
)
COMPLETION_METRICS = (
    "completion_through_layer_ms",
    "profiled_union_through_layer_ms",
    "unattributed_through_layer_ms",
    "h2d_gpu_through_layer_ms",
    "stage_transform_gpu_through_layer_ms",
    "compute_gpu_through_layer_ms",
    "request_elapsed_ms",
)


@dataclass(frozen=True)
class CaseSpec:
    skill_tokens: int
    chunk_size_tokens: int
    repetition: int

    @property
    def case_id(self) -> str:
        return (
            f"b{self.skill_tokens}_c{self.chunk_size_tokens}_"
            f"{HOST_LAYOUT}_{EXECUTION_ORDER}_r{self.repetition}"
        )

    def child_payload(self, run_dir: Path) -> dict[str, Any]:
        return {
            "run_dir": str(run_dir),
            "skill_tokens": self.skill_tokens,
            "calibration_ratio": CALIBRATION_TOKENS / self.skill_tokens,
            "calibration_tokens": CALIBRATION_TOKENS,
            "execution_order": EXECUTION_ORDER,
            "chunk_size_tokens": self.chunk_size_tokens,
            "storage_layout": STORAGE_LAYOUT,
            "host_layout": HOST_LAYOUT,
            "warmup_requests": WARMUP_REQUESTS,
        }


def _build_specs() -> list[CaseSpec]:
    arms = [
        (skill_tokens, chunk_size_tokens)
        for skill_tokens in SKILL_TOKEN_VALUES
        for chunk_size_tokens in CHUNK_SIZE_TOKEN_VALUES
    ]
    specs = []
    for repetition in range(REPETITIONS):
        run_order = arms if repetition % 2 == 0 else list(reversed(arms))
        specs.extend(
            CaseSpec(skill_tokens, chunk_size_tokens, repetition)
            for skill_tokens, chunk_size_tokens in run_order
        )
    expected = len(arms) * REPETITIONS
    if len(specs) != expected or len({spec.case_id for spec in specs}) != expected:
        raise ValueError("sweep matrix is incomplete or contains duplicate cases")
    return specs


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_profile(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _interval_union_ms(intervals: list[dict[str, Any]]) -> float:
    merged: list[list[float]] = []
    for item in sorted(intervals, key=lambda value: float(value["start_ms"])):
        start = float(item["start_ms"])
        end = float(item["end_ms"])
        if end < start:
            raise RuntimeError("profile interval ends before it starts")
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return sum(end - start for start, end in merged)


def _indexed_layers(
    records: list[dict[str, Any]], event: str
) -> dict[int, dict[str, Any]]:
    selected = [record for record in records if record.get("event") == event]
    indexed = {int(record["layer"]): record for record in selected}
    expected = set(range(EXPECTED_LAYERS))
    if len(selected) != EXPECTED_LAYERS or set(indexed) != expected:
        raise RuntimeError(f"{event} does not cover each model layer exactly once")
    return indexed


def _completion_metrics(path: Path) -> dict[str, float]:
    records = _load_profile(path)
    h2d = _indexed_layers(records, "cskcache_h2d_layer")
    transform = _indexed_layers(records, "cskcache_stage_transform_layer")
    compute_records = [
        record
        for record in records
        if record.get("event") == "cskcache_layer_compute"
    ]
    if len(compute_records) != 1:
        raise RuntimeError("profile does not contain one layer-compute record")
    compute_items = compute_records[0].get("calibration_correct_install", [])
    compute = {int(record["layer"]): record for record in compute_items}
    if len(compute_items) != EXPECTED_LAYERS or set(compute) != set(
        range(EXPECTED_LAYERS)
    ):
        raise RuntimeError("compute profile does not cover each model layer exactly once")

    correction = [
        record
        for record in records
        if record.get("event") == "csk_correction_complete"
    ]
    release = [
        record for record in records if record.get("event") == "csk_reuse_release"
    ]
    fallback = [
        record
        for record in records
        if "fallback" in str(record.get("event", "")).lower()
    ]
    if (
        len(correction) != 1
        or correction[0].get("layers") != EXPECTED_LAYERS
        or not release
        or not release[-1].get("released")
        or fallback
    ):
        raise RuntimeError("profile did not complete CSKCache reuse cleanly")

    included_layers = range(COMPLETION_LAYER_STOP)
    h2d_selected = [h2d[layer] for layer in included_layers]
    transform_selected = [transform[layer] for layer in included_layers]
    compute_selected = [compute[layer] for layer in included_layers]
    intervals = [*h2d_selected, *transform_selected, *compute_selected]
    origin = min(float(item["start_ms"]) for item in intervals)
    completion = max(float(item["end_ms"]) for item in intervals) - origin
    profiled_union = _interval_union_ms(intervals)
    return {
        "completion_through_layer_ms": completion,
        "profiled_union_through_layer_ms": profiled_union,
        "unattributed_through_layer_ms": completion - profiled_union,
        "h2d_gpu_through_layer_ms": sum(
            float(item["end_ms"]) - float(item["start_ms"])
            for item in h2d_selected
        ),
        "stage_transform_gpu_through_layer_ms": sum(
            float(item["end_ms"]) - float(item["start_ms"])
            for item in transform_selected
        ),
        "compute_gpu_through_layer_ms": sum(
            float(item["end_ms"]) - float(item["start_ms"])
            for item in compute_selected
        ),
    }


def _measure_case(spec: CaseSpec, run_dir: Path) -> dict[str, Any]:
    missing = [name for name in REQUIRED_CASE_FILES if not (run_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"case is missing outputs: {', '.join(missing)}")
    request = _load_json(run_dir / "request_result.json")
    expected = spec.child_payload(run_dir)
    for key, value in expected.items():
        if key != "run_dir" and request.get(key) != value:
            raise RuntimeError(f"case output has a different {key}")
    warmup_results = _load_json(run_dir / "warmup_request_result.json")
    if len(warmup_results) != WARMUP_REQUESTS:
        raise RuntimeError("case did not execute the configured warm-up")

    measured = _completion_metrics(run_dir / "cskcache_profile.jsonl")
    _completion_metrics(run_dir / "warmup_profile.jsonl")
    return {
        **asdict(spec),
        "case_id": spec.case_id,
        "calibration_tokens": CALIBRATION_TOKENS,
        "storage_layout": STORAGE_LAYOUT,
        "host_layout": HOST_LAYOUT,
        "execution_order": EXECUTION_ORDER,
        "included_layers": COMPLETION_LAYER_STOP,
        "excluded_layers": str(list(range(COMPLETION_LAYER_STOP, EXPECTED_LAYERS))),
        **measured,
        "request_elapsed_ms": float(request["request_elapsed_ms"]),
        "run_dir": str(run_dir),
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(int(row["skill_tokens"]), int(row["chunk_size_tokens"]))].append(row)
    output = []
    for skill_tokens in SKILL_TOKEN_VALUES:
        for chunk_size_tokens in CHUNK_SIZE_TOKEN_VALUES:
            samples = groups.get((skill_tokens, chunk_size_tokens), [])
            if not samples:
                continue
            item: dict[str, Any] = {
                "skill_tokens": skill_tokens,
                "chunk_size_tokens": chunk_size_tokens,
                "repetitions": len(samples),
                "included_layers": COMPLETION_LAYER_STOP,
                "excluded_layers": str(
                    list(range(COMPLETION_LAYER_STOP, EXPECTED_LAYERS))
                ),
            }
            for metric in COMPLETION_METRICS:
                values = [float(sample[metric]) for sample in samples]
                item[f"{metric}_median"] = statistics.median(values)
                item[f"{metric}_mean"] = statistics.mean(values)
                item[f"{metric}_std"] = (
                    statistics.stdev(values) if len(values) > 1 else 0.0
                )
                item[f"{metric}_min"] = min(values)
                item[f"{metric}_max"] = max(values)
            output.append(item)
    return output


def _plot(aggregate: list[dict[str, Any]]) -> None:
    positions = list(range(len(SKILL_TOKEN_VALUES)))
    labels = [f"{tokens // 1024}K" for tokens in SKILL_TOKEN_VALUES]
    colors = ("#4C78A8", "#59A14F", "#F28E2B", "#E15759")
    fig, axis = plt.subplots(figsize=(5.8, 3.2))
    for chunk_size_tokens, color in zip(
        CHUNK_SIZE_TOKEN_VALUES, colors, strict=True
    ):
        rows = [
            row
            for row in aggregate
            if int(row["chunk_size_tokens"]) == chunk_size_tokens
        ]
        axis.plot(
            positions,
            [float(row["completion_through_layer_ms_median"]) for row in rows],
            marker="o",
            markersize=4.5,
            linewidth=1.5,
            color=color,
            label=f"{chunk_size_tokens} tokens/chunk",
        )
    axis.set_xticks(positions, labels=labels)
    axis.set_xlabel("Skill length (tokens)")
    axis.set_ylabel("Completion latency through layer 38 (ms)")
    axis.grid(axis="y", alpha=0.18)
    axis.legend(frameon=False, fontsize=7.5, ncol=2)
    fig.tight_layout()
    fig.savefig(
        SWEEP_OUTPUT_ROOT / "completion_latency_without_final_layer.png",
        dpi=240,
    )
    fig.savefig(
        SWEEP_OUTPUT_ROOT / "completion_latency_without_final_layer.pdf"
    )
    plt.close(fig)


def _archive_incomplete(case_dir: Path) -> None:
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    target = case_dir.with_name(f"{case_dir.name}.failed-{timestamp}")
    suffix = 1
    while target.exists():
        target = case_dir.with_name(f"{case_dir.name}.failed-{timestamp}-{suffix}")
        suffix += 1
    case_dir.rename(target)


def _run_case(
    spec: CaseSpec,
    specs_dir: Path,
    cases_dir: Path,
    logs_dir: Path,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    case_dir = cases_dir / spec.case_id
    spec_path = specs_dir / f"{spec.case_id}.json"
    _write_json(spec_path, spec.child_payload(case_dir))
    if case_dir.exists():
        try:
            return _measure_case(spec, case_dir), []
        except Exception as error:
            print(f"archiving incomplete {spec.case_id}: {error}", flush=True)
            _archive_incomplete(case_dir)

    failures = []
    for attempt in range(CASE_RETRIES + 1):
        log_path = logs_dir / f"{spec.case_id}.attempt-{attempt}.log"
        child_env = os.environ.copy()
        child_env["CSKCACHE_PINNED_CASE_CONFIG"] = str(spec_path)
        print(
            f"running {spec.case_id} attempt={attempt + 1}/{CASE_RETRIES + 1}",
            flush=True,
        )
        with log_path.open("w", encoding="utf-8") as log_handle:
            completed = subprocess.run(
                [sys.executable, str(RUN_SCRIPT)],
                cwd=EXAMPLE_DIR,
                env=child_env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        try:
            if completed.returncode != 0:
                raise RuntimeError(f"child exited with status {completed.returncode}")
            return _measure_case(spec, case_dir), failures
        except Exception as error:
            failures.append(
                {
                    "case_id": spec.case_id,
                    "attempt": attempt,
                    "error": str(error),
                    "log_path": str(log_path),
                    "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
            )
            if case_dir.exists():
                _archive_incomplete(case_dir)
    return None, failures


def _prepare_root(specs: list[CaseSpec]) -> tuple[Path, Path, Path]:
    SWEEP_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for name in ("case_specs", "cases", "logs"):
        (SWEEP_OUTPUT_ROOT / name).mkdir(exist_ok=True)
    experiment = {
        "run_name": RUN_NAME,
        "skill_token_values": list(SKILL_TOKEN_VALUES),
        "chunk_size_token_values": list(CHUNK_SIZE_TOKEN_VALUES),
        "calibration_tokens": CALIBRATION_TOKENS,
        "storage_layout": STORAGE_LAYOUT,
        "host_layout": HOST_LAYOUT,
        "execution_order": EXECUTION_ORDER,
        "warmup_requests": WARMUP_REQUESTS,
        "repetitions": REPETITIONS,
        "expected_layers": EXPECTED_LAYERS,
        "included_layers": list(range(COMPLETION_LAYER_STOP)),
        "excluded_layers": list(range(COMPLETION_LAYER_STOP, EXPECTED_LAYERS)),
        "primary_metric": "completion_through_layer_ms",
        "cases": [asdict(spec) for spec in specs],
    }
    config_path = SWEEP_OUTPUT_ROOT / "experiment_config.json"
    if config_path.exists() and _load_json(config_path) != experiment:
        raise RuntimeError("RUN_NAME already contains a different experiment")
    _write_json(config_path, experiment)
    return tuple(
        SWEEP_OUTPUT_ROOT / name for name in ("case_specs", "cases", "logs")
    )


def _write_results(rows: list[dict[str, Any]], failed: list[str]) -> None:
    aggregate = _aggregate(rows)
    _write_csv(SWEEP_OUTPUT_ROOT / "per_run.csv", rows)
    _write_csv(SWEEP_OUTPUT_ROOT / "aggregate.csv", aggregate)
    _write_json(
        SWEEP_OUTPUT_ROOT / "summary.json",
        {
            "planned_cases": len(SKILL_TOKEN_VALUES)
            * len(CHUNK_SIZE_TOKEN_VALUES)
            * REPETITIONS,
            "successful_cases": len(rows),
            "failed_cases": failed,
            "primary_metric": "completion_through_layer_ms",
            "included_layers": list(range(COMPLETION_LAYER_STOP)),
            "excluded_layers": list(range(COMPLETION_LAYER_STOP, EXPECTED_LAYERS)),
        },
    )
    if len(rows) == len(SKILL_TOKEN_VALUES) * len(CHUNK_SIZE_TOKEN_VALUES) * REPETITIONS:
        _plot(aggregate)


def main() -> None:
    if PREFIX_TOKENS + max(SKILL_TOKEN_VALUES) + TAIL_TOKENS + MAX_TOKENS > MAX_MODEL_LEN:
        raise ValueError("MAX_MODEL_LEN is smaller than the sweep request")
    specs = _build_specs()
    specs_dir, cases_dir, logs_dir = _prepare_root(specs)
    rows = []
    failed = []
    consecutive_failures = 0
    failures_path = SWEEP_OUTPUT_ROOT / "failures.jsonl"
    print(f"planned warmed cases: {len(specs)}", flush=True)
    for index, spec in enumerate(specs, start=1):
        print(f"[{index}/{len(specs)}] {spec.case_id}", flush=True)
        row, attempts = _run_case(spec, specs_dir, cases_dir, logs_dir)
        if attempts:
            with failures_path.open("a", encoding="utf-8") as handle:
                for attempt in attempts:
                    handle.write(json.dumps(attempt, sort_keys=True) + "\n")
        if row is None:
            failed.append(spec.case_id)
            consecutive_failures += 1
        else:
            rows.append(row)
            consecutive_failures = 0
        _write_results(rows, failed)
        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            print("stopping after three consecutive failed cases", flush=True)
            break
    print(f"sweep results: {SWEEP_OUTPUT_ROOT}", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
