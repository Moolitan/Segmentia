"""Run the warmed critical pinned-KV pipeline sweep."""

from __future__ import annotations

import csv
import json
import os
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt

from config import MAX_MODEL_LEN, MAX_TOKENS, PREFIX_TOKENS, TAIL_TOKENS
from sweep_config import (
    CALIBRATION_BY_SKILL,
    CASE_RETRIES,
    CHUNK_SIZE_TOKENS,
    EXECUTION_ORDERS,
    MAX_CONSECUTIVE_FAILURES,
    REPETITIONS,
    RUN_NAME,
    SKILL_TOKEN_VALUES,
    STABLE_LAYER_START,
    STABLE_LAYER_STOP,
    SWEEP_OUTPUT_ROOT,
    WARMUP_REQUESTS,
)


EXAMPLE_DIR = Path(__file__).resolve().parent
RUN_SCRIPT = EXAMPLE_DIR / "run.py"
EXPECTED_LAYERS = 40
TOTAL_CASES = len(SKILL_TOKEN_VALUES) * 6 * REPETITIONS
REQUIRED_CASE_FILES = (
    "cskcache_profile.jsonl",
    "request_result.json",
    "summary.json",
    "warmup_profile.jsonl",
    "warmup_request_result.json",
)
NUMERIC_METRICS = (
    "median_h2d_next_ms",
    "median_layer_adaptation_ms",
    "stable_pair_span_ms",
    "stable_pair_overlap_ms",
    "layer0_adaptation_ms",
    "warmup_layer0_adaptation_ms",
    "warmup_layer0_reduction_percent",
    "h2d_gpu_ms",
    "h2d_cpu_submit_ms",
    "adaptation_gpu_ms",
    "h2d_overlap_percent",
    "pipeline_span_ms",
    "request_elapsed_ms",
)


@dataclass(frozen=True)
class CaseSpec:
    skill_tokens: int
    calibration_tokens: int
    execution_order: str
    chunk_size_tokens: int
    storage_layout: str
    host_layout: str
    loading_policy: str
    repetition: int

    @property
    def chunk_label(self) -> str:
        return (
            "One"
            if self.loading_policy == "one_chunk_layer"
            else str(self.chunk_size_tokens)
        )

    @property
    def case_id(self) -> str:
        return (
            f"b{self.skill_tokens}_c{self.chunk_label.lower()}_"
            f"{self.loading_policy}_{self.execution_order}_r{self.repetition}"
        )

    def child_payload(self, run_dir: Path) -> dict[str, Any]:
        return {
            "run_dir": str(run_dir),
            "skill_tokens": self.skill_tokens,
            "calibration_tokens": self.calibration_tokens,
            "execution_order": self.execution_order,
            "chunk_size_tokens": self.chunk_size_tokens,
            "storage_layout": self.storage_layout,
            "host_layout": self.host_layout,
            "warmup_requests": WARMUP_REQUESTS,
        }


def _build_specs() -> list[CaseSpec]:
    specs = []
    for skill_tokens in SKILL_TOKEN_VALUES:
        calibration_tokens = CALIBRATION_BY_SKILL[skill_tokens]
        for repetition in range(REPETITIONS):
            for loading_policy, host_layout in (
                ("chunkwise", "chunk_single_layer"),
                ("packed_layer", "packed_chunks_single_layer"),
            ):
                for execution_order in EXECUTION_ORDERS:
                    specs.append(
                        CaseSpec(
                            skill_tokens=skill_tokens,
                            calibration_tokens=calibration_tokens,
                            execution_order=execution_order,
                            chunk_size_tokens=CHUNK_SIZE_TOKENS,
                            storage_layout="packed_chunks_single_layer",
                            host_layout=host_layout,
                            loading_policy=loading_policy,
                            repetition=repetition,
                        )
                    )
            for execution_order in EXECUTION_ORDERS:
                specs.append(
                    CaseSpec(
                        skill_tokens=skill_tokens,
                        calibration_tokens=calibration_tokens,
                        execution_order=execution_order,
                        chunk_size_tokens=skill_tokens,
                        storage_layout="packed_chunks_single_layer",
                        host_layout="packed_chunks_single_layer",
                        loading_policy="one_chunk_layer",
                        repetition=repetition,
                    )
                )
    if len(specs) != TOTAL_CASES or len({spec.case_id for spec in specs}) != len(specs):
        raise ValueError("critical sweep matrix is incomplete or contains duplicates")
    return specs


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _profile_metrics(path: Path) -> dict[str, float]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    h2d = {
        int(record["layer"]): record
        for record in records
        if record.get("event") == "cskcache_h2d_layer"
    }
    compute_records = [
        record
        for record in records
        if record.get("event") == "cskcache_layer_compute"
    ]
    if len(h2d) != EXPECTED_LAYERS or len(compute_records) != 1:
        raise RuntimeError("profile does not contain one complete layer pipeline")
    adaptation = {
        int(record["layer"]): record
        for record in compute_records[0]["calibration_correct_install"]
    }
    if len(adaptation) != EXPECTED_LAYERS:
        raise RuntimeError("profile does not contain all adapted layers")
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

    stable_layers = range(STABLE_LAYER_START, STABLE_LAYER_STOP)
    h2d_times = []
    adaptation_times = []
    pair_spans = []
    pair_overlaps = []
    for layer in stable_layers:
        left = adaptation[layer]
        right = h2d[layer + 1]
        adaptation_times.append(left["end_ms"] - left["start_ms"])
        h2d_times.append(right["end_ms"] - right["start_ms"])
        pair_spans.append(
            max(left["end_ms"], right["end_ms"])
            - min(left["start_ms"], right["start_ms"])
        )
        pair_overlaps.append(
            max(
                0.0,
                min(left["end_ms"], right["end_ms"])
                - max(left["start_ms"], right["start_ms"]),
            )
        )
    layer0 = adaptation[0]["end_ms"] - adaptation[0]["start_ms"]
    return {
        "median_h2d_next_ms": statistics.median(h2d_times),
        "median_layer_adaptation_ms": statistics.median(adaptation_times),
        "stable_pair_span_ms": statistics.median(pair_spans),
        "stable_pair_overlap_ms": statistics.median(pair_overlaps),
        "layer0_adaptation_ms": layer0,
    }


def _measure_case(spec: CaseSpec, run_dir: Path) -> dict[str, Any]:
    missing = [name for name in REQUIRED_CASE_FILES if not (run_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"case is missing outputs: {', '.join(missing)}")
    request = _load_json(run_dir / "request_result.json")
    expected = spec.child_payload(run_dir)
    for key in expected:
        if key != "run_dir" and request.get(key) != expected[key]:
            raise RuntimeError(f"case output has a different {key}")
    warmup_results = _load_json(run_dir / "warmup_request_result.json")
    if len(warmup_results) != WARMUP_REQUESTS:
        raise RuntimeError("case did not execute the configured warm-up")

    measured = _profile_metrics(run_dir / "cskcache_profile.jsonl")
    warmup = _profile_metrics(run_dir / "warmup_profile.jsonl")
    summary = _load_json(run_dir / "summary.json")
    h2d_total = float(summary["h2d_gpu_ms"])
    overlap_ms = float(summary["overlap_ms"])
    warmup_layer0 = warmup["layer0_adaptation_ms"]
    measured_layer0 = measured["layer0_adaptation_ms"]
    return {
        **asdict(spec),
        "case_id": spec.case_id,
        "chunk_label": spec.chunk_label,
        **measured,
        "warmup_layer0_adaptation_ms": warmup_layer0,
        "warmup_layer0_reduction_percent": (
            100.0 * (warmup_layer0 - measured_layer0) / warmup_layer0
        ),
        "h2d_gpu_ms": h2d_total,
        "h2d_cpu_submit_ms": float(summary["h2d_cpu_submit_ms"]),
        "adaptation_gpu_ms": float(summary["compute_gpu_ms"]),
        "h2d_overlap_percent": (
            min(100.0, 100.0 * overlap_ms / h2d_total)
            if h2d_total > 0.0
            else 0.0
        ),
        "pipeline_span_ms": float(summary["pipeline_span_ms"]),
        "request_elapsed_ms": float(summary["request_elapsed_ms"]),
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
    keys = (
        "skill_tokens",
        "calibration_tokens",
        "chunk_label",
        "chunk_size_tokens",
        "loading_policy",
        "execution_order",
    )
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    output = []
    for group, samples in sorted(groups.items(), key=lambda item: str(item[0])):
        item = dict(zip(keys, group, strict=True))
        item["repetitions"] = len(samples)
        for metric in NUMERIC_METRICS:
            values = [float(sample[metric]) for sample in samples]
            item[f"{metric}_median"] = statistics.median(values)
            item[f"{metric}_mean"] = statistics.mean(values)
            item[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        output.append(item)
    return output


def _plot_pipeline(rows: list[dict[str, Any]]) -> None:
    variants = (
        (
            str(CHUNK_SIZE_TOKENS),
            "chunkwise",
            "Per-chunk layer\n256 tokens/object",
        ),
        (
            str(CHUNK_SIZE_TOKENS),
            "packed_layer",
            "Packed layer\nall chunks/object",
        ),
        (
            "One",
            "one_chunk_layer",
            "One-chunk layer\nSkill tokens/object",
        ),
    )
    colors = {"h2d_first": "#4C78A8", "compute_first": "#E45756"}
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 5.2), sharex=True)
    for axis, skill_tokens in zip(axes.flat, SKILL_TOKEN_VALUES, strict=True):
        for order in EXECUTION_ORDERS:
            points = []
            for chunk_label, policy, _ in variants:
                points.append(
                    next(
                        row
                        for row in rows
                        if row["skill_tokens"] == skill_tokens
                        and row["chunk_label"] == chunk_label
                        and row["loading_policy"] == policy
                        and row["execution_order"] == order
                    )
                )
            axis.plot(
                range(len(points)),
                [row["pipeline_span_ms_median"] for row in points],
                marker="o",
                linewidth=1.6,
                color=colors[order],
                label=order.replace("_", "-").title(),
            )
        axis.set_title(f"{skill_tokens:,} Skill tokens", fontsize=9)
        axis.set_xticks(range(len(variants)), [item[2] for item in variants])
        axis.tick_params(axis="x", labelsize=7, pad=2)
        axis.set_ylabel("Warmed pipeline time (ms)")
        axis.grid(alpha=0.18)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.supxlabel("Pinned-memory H2D source layout", fontsize=9, y=0.01)
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    fig.savefig(SWEEP_OUTPUT_ROOT / "warmed_pipeline.png", dpi=240)
    fig.savefig(SWEEP_OUTPUT_ROOT / "warmed_pipeline.pdf")
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
        "warmup_requests": WARMUP_REQUESTS,
        "repetitions": REPETITIONS,
        "chunk_size_tokens": CHUNK_SIZE_TOKENS,
        "skill_token_values": list(SKILL_TOKEN_VALUES),
        "calibration_by_skill": {
            str(key): value for key, value in CALIBRATION_BY_SKILL.items()
        },
        "execution_orders": list(EXECUTION_ORDERS),
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
            "planned_cases": TOTAL_CASES,
            "successful_cases": len(rows),
            "failed_cases": failed,
            "warmup_requests_per_case": WARMUP_REQUESTS,
            "repetitions_per_configuration": REPETITIONS,
        },
    )
    if len(rows) == TOTAL_CASES:
        _plot_pipeline(aggregate)


def main() -> None:
    if PREFIX_TOKENS + max(SKILL_TOKEN_VALUES) + TAIL_TOKENS + MAX_TOKENS > MAX_MODEL_LEN:
        raise ValueError("MAX_MODEL_LEN is smaller than the largest request")
    specs = _build_specs()
    specs_dir, cases_dir, logs_dir = _prepare_root(specs)
    rows = []
    failed = []
    consecutive_failures = 0
    failures_path = SWEEP_OUTPUT_ROOT / "failures.jsonl"
    print(f"planned warmed cases: {TOTAL_CASES}", flush=True)
    for index, spec in enumerate(specs, start=1):
        print(f"[{index}/{TOTAL_CASES}] {spec.case_id}", flush=True)
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
