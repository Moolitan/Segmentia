#!/usr/bin/env python3
"""Validate and summarize isolated CSKCache H2D benchmark cases."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable


BASE_STAGE_NAMES = (
    "key_h2d",
    "value_h2d",
    "scatter_span",
)
SUMMARY_STAGE_NAMES = (*BASE_STAGE_NAMES[:2], "rope", BASE_STAGE_NAMES[2])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-cases", type=int, default=8)
    parser.add_argument("--expected-repetitions", type=int, default=30)
    return parser.parse_args()


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def metric_stats(values: Iterable[float]) -> dict[str, float]:
    samples = [float(value) for value in values]
    if not samples:
        return {}
    mean = statistics.fmean(samples)
    std = statistics.stdev(samples) if len(samples) > 1 else 0.0
    return {
        "mean": mean,
        "std": std,
        "cv": std / mean if mean else 0.0,
        "min": min(samples),
        "p10": percentile(samples, 0.10),
        "p50": percentile(samples, 0.50),
        "p90": percentile(samples, 0.90),
        "p95": percentile(samples, 0.95),
        "max": max(samples),
    }


def read_case(path: Path, expected_repetitions: int) -> tuple[dict, list[dict]]:
    records = [json.loads(line) for line in path.read_text().splitlines() if line]
    cases = [record for record in records if record.get("record_type") == "case"]
    iterations = [
        record for record in records if record.get("record_type") == "iteration"
    ]
    if len(cases) != 1:
        raise ValueError(f"{path} must contain exactly one case record")
    if len(iterations) != expected_repetitions:
        raise ValueError(
            f"{path} has {len(iterations)} iterations; expected {expected_repetitions}"
        )
    for record in iterations:
        if record["expected_layers"] != record["scattered_layers"]:
            raise ValueError(f"Incomplete layer scatter in {path}")
        if record["skipped_layers"] != 0:
            raise ValueError(f"Nonzero skipped layers in {path}")
        if cases[0]["profiling"] == "on":
            required = set(BASE_STAGE_NAMES)
            if int(cases[0]["position_shift"]) != 0:
                required.add("rope")
            missing = required - set(record["cuda_stage_ms"])
            if missing:
                raise ValueError(f"Missing CUDA stages {sorted(missing)} in {path}")
    return cases[0], iterations


def flatten_stats(prefix: str, stats: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{name}": value for name, value in stats.items()}


def summarize_case(case: dict, iterations: list[dict]) -> dict[str, Any]:
    row: dict[str, Any] = {
        key: case[key]
        for key in (
            "case_id",
            "profiling",
            "memory",
            "position_shift",
            "warmup",
            "repetitions",
            "cache_id",
            "tokens",
            "bytes",
            "layers",
            "disk_load_ms",
            "pin_setup_ms",
            "gpu_name",
        )
    }
    for metric in (
        "operation_wall_ms",
        "end_to_end_wall_ms",
        "outer_cuda_ms",
        "path_gbps",
    ):
        row.update(
            flatten_stats(metric, metric_stats(record[metric] for record in iterations))
        )
    if case["profiling"] == "on":
        for stage in SUMMARY_STAGE_NAMES:
            row.update(
                flatten_stats(
                    f"cuda_{stage}_ms",
                    metric_stats(
                        record["cuda_stage_ms"].get(stage, 0.0)
                        for record in iterations
                    ),
                )
            )
    return row


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def comparison_row(
    comparison: str,
    baseline: dict[str, Any],
    variant: dict[str, Any],
) -> dict[str, Any]:
    baseline_ms = float(baseline["end_to_end_wall_ms_p50"])
    variant_ms = float(variant["end_to_end_wall_ms_p50"])
    return {
        "comparison": comparison,
        "baseline_case": baseline["case_id"],
        "variant_case": variant["case_id"],
        "baseline_p50_ms": baseline_ms,
        "variant_p50_ms": variant_ms,
        "delta_ms": variant_ms - baseline_ms,
        "delta_percent": (variant_ms / baseline_ms - 1.0) * 100.0,
    }


def build_comparisons(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_condition = {
        (
            str(row["profiling"]),
            str(row["memory"]),
            int(row["position_shift"]),
        ): row
        for row in rows
    }
    comparisons: list[dict[str, Any]] = []
    memories = sorted({str(row["memory"]) for row in rows})
    shifts = sorted({int(row["position_shift"]) for row in rows})
    profiles = sorted({str(row["profiling"]) for row in rows})
    for memory in memories:
        for shift in shifts:
            off = by_condition.get(("off", memory, shift))
            on = by_condition.get(("on", memory, shift))
            if off is not None and on is not None:
                comparisons.append(comparison_row("profiling_on_vs_off", off, on))
    for profiling in profiles:
        for shift in shifts:
            pageable = by_condition.get((profiling, "pageable", shift))
            pinned = by_condition.get((profiling, "pinned", shift))
            if pageable is not None and pinned is not None:
                comparisons.append(
                    comparison_row("pinned_vs_pageable", pageable, pinned)
                )
    if 0 in shifts:
        nonzero_shifts = [shift for shift in shifts if shift != 0]
        for profiling in profiles:
            for memory in memories:
                zero = by_condition.get((profiling, memory, 0))
                for shift in nonzero_shifts:
                    shifted = by_condition.get((profiling, memory, shift))
                    if zero is not None and shifted is not None:
                        comparisons.append(
                            comparison_row("position_shift_vs_zero", zero, shifted)
                        )
    return comparisons


def main() -> None:
    args = parse_args()
    case_paths = sorted(args.case_dir.glob("*.jsonl"))
    if len(case_paths) != args.expected_cases:
        raise ValueError(
            f"Found {len(case_paths)} completed cases in {args.case_dir}; "
            f"expected {args.expected_cases}"
        )
    loaded = [read_case(path, args.expected_repetitions) for path in case_paths]
    case_ids = [case["case_id"] for case, _ in loaded]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Duplicate case IDs")
    rows = [summarize_case(case, iterations) for case, iterations in loaded]
    rows.sort(
        key=lambda row: (
            int(row["position_shift"]),
            str(row["memory"]),
            str(row["profiling"]),
        )
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_output = args.output_dir / "raw_iterations.jsonl"
    with raw_output.open("w", encoding="utf-8") as output:
        for path in case_paths:
            for line in path.read_text().splitlines():
                record = json.loads(line)
                if record.get("record_type") == "iteration":
                    output.write(json.dumps(record, sort_keys=True) + "\n")
    summary_csv = args.output_dir / "summary.csv"
    write_csv(rows, summary_csv)
    comparisons_csv = args.output_dir / "comparisons.csv"
    write_csv(build_comparisons(rows), comparisons_csv)
    config = {
        "schema_version": 1,
        "expected_cases": args.expected_cases,
        "expected_repetitions": args.expected_repetitions,
        "case_ids": case_ids,
        "cases": [case for case, _ in loaded],
    }
    (args.output_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n"
    )
    print(f"cases={len(rows)}")
    print(f"raw_iterations={raw_output}")
    print(f"summary_csv={summary_csv}")
    print(f"comparisons_csv={comparisons_csv}")


if __name__ == "__main__":
    main()
