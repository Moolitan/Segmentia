"""Run isolated pinned-KV cases and find the C/R/I--H2D balance point."""

from __future__ import annotations

import csv
import json
import math
import os
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from config import MAX_MODEL_LEN, MAX_TOKENS, PREFIX_TOKENS, TAIL_TOKENS
from sweep_config import (
    CALIBRATION_TOKEN_VALUES,
    REPETITIONS,
    SKILL_TOKEN_VALUES,
    STABLE_LAYER_START,
    STABLE_LAYER_STOP,
    SWEEP_OUTPUT_ROOT,
)


EXAMPLE_DIR = Path(__file__).resolve().parent
RUN_SCRIPT = EXAMPLE_DIR / "run.py"


def _load_profile(run_dir: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (run_dir / "cskcache_profile.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line
    ]


def _measure_stable_layers(run_dir: Path) -> dict[str, float | int]:
    records = _load_profile(run_dir)
    h2d = {
        int(record["layer"]): float(record["end_ms"] - record["start_ms"])
        for record in records
        if record.get("event") == "cskcache_h2d_layer"
    }
    compute_record = next(
        record
        for record in records
        if record.get("event") == "cskcache_layer_compute"
    )
    cri = {
        int(record["layer"]): float(record["end_ms"] - record["start_ms"])
        for record in compute_record["calibration_correct_install"]
    }
    layers = [
        layer
        for layer in range(STABLE_LAYER_START, STABLE_LAYER_STOP)
        if layer in cri and layer + 1 in h2d
    ]
    if not layers:
        raise RuntimeError("profile contains no stable C/R/I--H2D layer pairs")
    median_h2d = statistics.median(h2d[layer + 1] for layer in layers)
    median_cri = statistics.median(cri[layer] for layer in layers)
    return {
        "stable_layer_start": layers[0],
        "stable_layer_stop": layers[-1] + 1,
        "stable_layer_pairs": len(layers),
        "median_h2d_next_ms": median_h2d,
        "median_cri_ms": median_cri,
        "absolute_gap_ms": abs(median_cri - median_h2d),
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["skill_tokens"], row["calibration_tokens"])].append(row)

    aggregate = []
    for (skill_tokens, calibration_tokens), samples in sorted(groups.items()):
        median_h2d = statistics.median(
            sample["median_h2d_next_ms"] for sample in samples
        )
        median_cri = statistics.median(
            sample["median_cri_ms"] for sample in samples
        )
        aggregate.append(
            {
                "skill_tokens": skill_tokens,
                "calibration_tokens": calibration_tokens,
                "repetitions": len(samples),
                "median_h2d_next_ms": median_h2d,
                "median_cri_ms": median_cri,
                "absolute_gap_ms": abs(median_cri - median_h2d),
            }
        )
    return aggregate


def _plot(run_root: Path, rows: list[dict], balance: list[dict]) -> None:
    skills = sorted({row["skill_tokens"] for row in rows})
    balance_by_skill = {row["skill_tokens"]: row for row in balance}
    columns = 2
    rows_count = math.ceil(len(skills) / columns)
    fig, axes = plt.subplots(
        rows_count,
        columns,
        figsize=(7.4, 2.55 * rows_count),
        squeeze=False,
    )
    for axis, skill_tokens in zip(axes.flat, skills, strict=False):
        points = sorted(
            (row for row in rows if row["skill_tokens"] == skill_tokens),
            key=lambda row: row["calibration_tokens"],
        )
        x = [row["calibration_tokens"] for row in points]
        axis.plot(
            x,
            [row["median_h2d_next_ms"] for row in points],
            marker="o",
            linewidth=1.8,
            label=r"H2D($l+1$)",
        )
        axis.plot(
            x,
            [row["median_cri_ms"] for row in points],
            marker="s",
            linewidth=1.8,
            label=r"C/R/I($l$)",
        )
        best = balance_by_skill[skill_tokens]
        axis.axvline(
            best["calibration_tokens"], color="#777777", linestyle="--", linewidth=1
        )
        axis.set_title(f"Skill tokens = {skill_tokens}")
        axis.set_xlabel("Calibration tokens")
        axis.set_ylabel("Median layer latency (ms)")
        axis.grid(alpha=0.25)
    for axis in axes.flat[len(skills) :]:
        axis.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(run_root / "balance_curves.png", dpi=240)
    fig.savefig(run_root / "balance_curves.pdf")
    plt.close(fig)


def main() -> None:
    largest_prompt = PREFIX_TOKENS + max(SKILL_TOKEN_VALUES) + TAIL_TOKENS
    if largest_prompt + MAX_TOKENS > MAX_MODEL_LEN:
        raise ValueError(
            "MAX_MODEL_LEN is smaller than the largest configured sweep request"
        )
    if not 0 <= STABLE_LAYER_START < STABLE_LAYER_STOP:
        raise ValueError("stable layer range is invalid")

    run_root = SWEEP_OUTPUT_ROOT / time.strftime(
        "%Y%m%dT%H%M%SZ", time.gmtime()
    )
    specs_dir = run_root / "case_specs"
    cases_dir = run_root / "cases"
    specs_dir.mkdir(parents=True)
    cases_dir.mkdir()

    per_run_rows: list[dict] = []
    for skill_tokens in SKILL_TOKEN_VALUES:
        for calibration_tokens in CALIBRATION_TOKEN_VALUES:
            for repetition in range(REPETITIONS):
                case_name = (
                    f"skill-{skill_tokens}_cal-{calibration_tokens}"
                    f"_rep-{repetition}"
                )
                case_dir = cases_dir / case_name
                spec_path = specs_dir / f"{case_name}.json"
                spec_path.write_text(
                    json.dumps(
                        {
                            "run_dir": str(case_dir),
                            "skill_tokens": skill_tokens,
                            "calibration_tokens": calibration_tokens,
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                child_env = os.environ.copy()
                child_env["CSKCACHE_PINNED_CASE_CONFIG"] = str(spec_path)
                print(f"running {case_name}", flush=True)
                subprocess.run(
                    [sys.executable, str(RUN_SCRIPT)],
                    cwd=EXAMPLE_DIR,
                    env=child_env,
                    check=True,
                )
                metrics = _measure_stable_layers(case_dir)
                per_run_rows.append(
                    {
                        "skill_tokens": skill_tokens,
                        "calibration_tokens": calibration_tokens,
                        "repetition": repetition,
                        **metrics,
                        "run_dir": str(case_dir),
                    }
                )

    aggregate = _aggregate(per_run_rows)
    balance = []
    for skill_tokens in sorted(set(SKILL_TOKEN_VALUES)):
        candidates = [
            row for row in aggregate if row["skill_tokens"] == skill_tokens
        ]
        balance.append(min(candidates, key=lambda row: row["absolute_gap_ms"]))

    _write_csv(run_root / "per_run.csv", per_run_rows)
    _write_csv(run_root / "aggregate.csv", aggregate)
    _write_csv(run_root / "balance_points.csv", balance)
    (run_root / "summary.json").write_text(
        json.dumps(
            {
                "stable_layer_range": [
                    STABLE_LAYER_START,
                    STABLE_LAYER_STOP,
                ],
                "balance_rule": "minimum absolute median latency gap",
                "balance_points": balance,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _plot(run_root, aggregate, balance)
    print(f"sweep results: {run_root}")


if __name__ == "__main__":
    main()
