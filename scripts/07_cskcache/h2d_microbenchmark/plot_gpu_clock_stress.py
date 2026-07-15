#!/usr/bin/env python3
"""Plot the GPU-clock diagnostic stress run for pageable+shift=0.

This is a one-off diagnostic, not part of the regular 8-case matrix: it
checks whether the fast/slow step seen within a single pageable+shift=0 run
lines up with the GPU leaving its idle power state (P8 -> P0). Reads the
--sample-gpu-clock output of run_case.py directly (one JSONL, one case) and
plots per-iteration wall time against the per-iteration SM clock reading.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cskcache-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/cskcache-xdg-cache")

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULT_DIR = REPO_ROOT / "results" / "problem_exploration" / "h2d_microbenchmark"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-jsonl", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--name", default="gpu_clock_stress")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            if record.get("record_type") == "iteration":
                rows.append(record)
    rows.sort(key=lambda r: r["iteration"])
    if not rows or "gpu_sm_clock_mhz" not in rows[0]:
        raise ValueError(f"{path} has no gpu_sm_clock_mhz field — was it run with --sample-gpu-clock?")
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["iteration", "end_to_end_wall_ms", "gpu_sm_clock_mhz", "gpu_pstate", "gpu_temperature_c", "gpu_power_w"])
        for r in rows:
            writer.writerow(
                [
                    r["iteration"],
                    r["end_to_end_wall_ms"],
                    r["gpu_sm_clock_mhz"],
                    r["gpu_pstate"],
                    r["gpu_temperature_c"],
                    r["gpu_power_w"],
                ]
            )


def plot(rows: list[dict], output_stem: Path) -> None:
    iterations = [r["iteration"] for r in rows]
    wall = [r["end_to_end_wall_ms"] for r in rows]
    sm_clock = [r["gpu_sm_clock_mhz"] for r in rows]

    fig, ax_wall = plt.subplots(figsize=(9.0, 3.4))
    ax_wall.plot(iterations, wall, color="#E45756", marker="o", markersize=2.5, linewidth=0.8, label="end_to_end_wall_ms")
    ax_wall.set_xlabel("Iteration")
    ax_wall.set_ylabel("End-to-end wall time (ms)", color="#E45756")
    ax_wall.tick_params(axis="y", labelcolor="#E45756")

    ax_clock = ax_wall.twinx()
    ax_clock.plot(iterations, sm_clock, color="#4C78A8", linewidth=1.2, label="GPU SM clock (MHz)")
    ax_clock.set_ylabel("GPU SM clock (MHz)", color="#4C78A8")
    ax_clock.tick_params(axis="y", labelcolor="#4C78A8")
    ax_clock.set_ylim(0, max(sm_clock) * 1.2)

    ax_wall.set_title(
        "pageable, shift=0, profiling=off — 100 reps, GPU started idle (P8)"
    )
    fig.tight_layout()
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{output_stem}.png", dpi=200)
    fig.savefig(f"{output_stem}.pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rows = read_rows(args.case_jsonl)
    figure_dir = args.result_dir / "figures"
    data_path = args.result_dir / "data" / f"{args.name}.csv"
    write_csv(rows, data_path)
    output_stem = figure_dir / args.name
    plot(rows, output_stem)
    distinct_clocks = sorted(set(r["gpu_sm_clock_mhz"] for r in rows))
    distinct_pstates = sorted(set(r["gpu_pstate"] for r in rows))
    print(f"data_csv={data_path}")
    print(f"figure_png={output_stem}.png")
    print(f"distinct_sm_clock_mhz={distinct_clocks}")
    print(f"distinct_pstate={distinct_pstates}")


if __name__ == "__main__":
    main()
