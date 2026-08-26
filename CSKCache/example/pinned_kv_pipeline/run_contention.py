"""Run and aggregate the first reuse-induced bandwidth-inversion diagnosis."""

from __future__ import annotations

import csv
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from contention_config import (
    ARMS,
    CALIBRATION_TOKENS,
    CASE_CONFIG_ENV,
    CASE_SPECS,
    CHUNK_SIZE_TOKENS,
    EXECUTION_ORDER,
    HOST_LAYOUT,
    LOG_DIR,
    REPETITIONS,
    RUN_ROOT,
    RUNNER,
    SKILL_TOKENS,
    STABLE_LAYER_START,
    STABLE_LAYER_STOP,
    STORAGE_LAYOUT,
    WARMUP_REQUESTS,
)


def _records(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _case_complete(run_dir: Path) -> bool:
    profile_path = run_dir / "cskcache_profile.jsonl"
    if (
        not (run_dir / "request_result.json").is_file()
        or not profile_path.is_file()
    ):
        return False
    return any(
        record.get("event") == "cskcache_contention_diagnostic"
        for record in _records(profile_path)
    )


def _run_cases() -> None:
    CASE_SPECS.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    total = len(ARMS) * REPETITIONS
    position = 0
    for arm in ARMS:
        for repetition in range(REPETITIONS):
            position += 1
            case_id = f"{arm}_r{repetition}"
            run_dir = RUN_ROOT / "cases" / case_id
            if _case_complete(run_dir):
                print(f"[{position}/{total}] reuse {case_id}", flush=True)
                continue
            if run_dir.exists():
                run_dir.rename(
                    run_dir.with_name(
                        f"{case_id}.incomplete-"
                        f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
                    )
                )
            spec = {
                "run_dir": str(run_dir),
                "skill_tokens": SKILL_TOKENS,
                "calibration_tokens": CALIBRATION_TOKENS,
                "calibration_ratio": CALIBRATION_TOKENS / SKILL_TOKENS,
                "chunk_size_tokens": CHUNK_SIZE_TOKENS,
                "storage_layout": STORAGE_LAYOUT,
                "host_layout": HOST_LAYOUT,
                "execution_order": EXECUTION_ORDER,
                "contention_arm": arm,
                "warmup_requests": WARMUP_REQUESTS,
            }
            spec_path = CASE_SPECS / f"{case_id}.json"
            spec_path.write_text(
                json.dumps(spec, indent=2) + "\n", encoding="utf-8"
            )
            environment = os.environ.copy()
            environment[CASE_CONFIG_ENV] = str(spec_path)
            print(f"[{position}/{total}] run {case_id}", flush=True)
            with (LOG_DIR / f"{case_id}.log").open(
                "w", encoding="utf-8"
            ) as log:
                subprocess.run(
                    [sys.executable, str(RUNNER)],
                    cwd=RUNNER.parent,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )


def _case_row(arm: str, repetition: int) -> dict[str, Any]:
    run_dir = RUN_ROOT / "cases" / f"{arm}_r{repetition}"
    records = _records(run_dir / "cskcache_profile.jsonl")
    diagnostic = next(
        record
        for record in records
        if record.get("event") == "cskcache_contention_diagnostic"
    )
    if arm == "full":
        h2d_layers = [
            {
                "layer": int(record["layer"]),
                "gpu_ms": float(record["end_ms"] - record["start_ms"]),
            }
            for record in records
            if record.get("event") == "cskcache_h2d_layer"
        ]
        compute_record = next(
            record
            for record in records
            if record.get("event") == "cskcache_layer_compute"
        )
        compute_layers = [
            {
                "layer": int(record["layer"]),
                "gpu_ms": float(record["calibration_forward_ms"]),
            }
            for record in compute_record["calibration_correct_install"]
        ]
        gpu_span_ms = json.loads(
            (run_dir / "summary.json").read_text(encoding="utf-8")
        )["pipeline_span_ms"]
    else:
        h2d_layers = diagnostic["h2d_layers"]
        compute_layers = diagnostic["compute_layers"]
        gpu_span_ms = float(diagnostic["gpu_span_ms"])

    stable = range(STABLE_LAYER_START, STABLE_LAYER_STOP)
    h2d_by_layer = {
        int(record["layer"]): float(record["gpu_ms"])
        for record in h2d_layers
    }
    compute_by_layer = {
        int(record["layer"]): float(record["gpu_ms"])
        for record in compute_layers
    }
    return {
        "arm": arm,
        "repetition": repetition,
        "h2d_sum_ms": sum(h2d_by_layer.values()),
        "h2d_stable_layer_ms": (
            statistics.median(h2d_by_layer[layer] for layer in stable)
            if h2d_by_layer
            else 0.0
        ),
        "compute_sum_ms": sum(compute_by_layer.values()),
        "compute_stable_layer_ms": (
            statistics.median(compute_by_layer[layer] for layer in stable)
            if compute_by_layer
            else 0.0
        ),
        "gpu_span_ms": gpu_span_ms,
        "wall_ms": float(diagnostic["wall_ms"]),
    }


def _aggregate() -> None:
    rows = [
        _case_row(arm, repetition)
        for arm in ARMS
        for repetition in range(REPETITIONS)
    ]
    fields = tuple(rows[0])
    with (RUN_ROOT / "raw.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    metrics = fields[2:]
    summary = {
        arm: {
            metric: statistics.median(
                float(row[metric]) for row in rows if row["arm"] == arm
            )
            for metric in metrics
        }
        for arm in ARMS
    }
    h2d_slowdown = (
        summary["concurrent"]["h2d_stable_layer_ms"]
        / summary["h2d_only"]["h2d_stable_layer_ms"]
    )
    compute_slowdown = (
        summary["concurrent"]["compute_stable_layer_ms"]
        / summary["calibration_only"]["compute_stable_layer_ms"]
    )
    ideal_span = max(
        summary["h2d_only"]["gpu_span_ms"],
        summary["calibration_only"]["gpu_span_ms"],
    )
    result = {
        "config": {
            "skill_tokens": SKILL_TOKENS,
            "calibration_tokens": CALIBRATION_TOKENS,
            "layout": HOST_LAYOUT,
            "repetitions": REPETITIONS,
            "stable_layers": [STABLE_LAYER_START, STABLE_LAYER_STOP - 1],
        },
        "median": summary,
        "diagnosis": {
            "h2d_slowdown": h2d_slowdown,
            "compute_slowdown": compute_slowdown,
            "ideal_concurrent_span_ms": ideal_span,
            "observed_concurrent_span_ms": summary["concurrent"][
                "gpu_span_ms"
            ],
            "span_over_ideal": (
                summary["concurrent"]["gpu_span_ms"] / ideal_span
            ),
            "counter_gate": (
                h2d_slowdown >= 1.15 or compute_slowdown >= 1.15
            ),
        },
    }
    (RUN_ROOT / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(
        "H2D slowdown: " f"{h2d_slowdown:.3f}x\n"
        "Calibration slowdown: " f"{compute_slowdown:.3f}x\n"
        "Concurrent span / isolated ideal: "
        f"{result['diagnosis']['span_over_ideal']:.3f}x\n"
        "Nsight counter gate: "
        f"{'GO' if result['diagnosis']['counter_gate'] else 'NO-GO'}\n"
        f"results: {RUN_ROOT}",
        flush=True,
    )


def main() -> None:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    _run_cases()
    _aggregate()


if __name__ == "__main__":
    main()
