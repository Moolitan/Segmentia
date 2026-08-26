#!/usr/bin/env python3
"""Summarize and plot independent-read versus readv latency."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cskcache-scatter-gather-mpl")
import matplotlib.pyplot as plt

try:
    from .common import DIRECT_MODES, IO_ENGINES, LAYOUTS, summarize_samples
    from .run import atomic_write_json
except ImportError:
    from common import DIRECT_MODES, IO_ENGINES, LAYOUTS, summarize_samples
    from run import atomic_write_json


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_latency(rows: list[dict[str, Any]], output: Path, token_count: int) -> None:
    by_key = {
        (
            str(row["layout"]),
            str(row["io_engine"]),
            bool(row["use_odirect"]),
            str(row["submission_mode"]),
        ): row
        for row in rows
    }
    figure, axes = plt.subplots(2, 2, figsize=(13.5, 7.0), sharey=True)
    colors = {"multi_read": "#4c78a8", "readv": "#e45756"}
    labels = {"multi_read": "Independent reads", "readv": "Vectored read"}
    positions = list(range(len(LAYOUTS)))
    width = 0.36
    for axis, (engine, direct) in zip(
        axes.flat,
        ((engine, direct) for engine in IO_ENGINES for direct in DIRECT_MODES),
        strict=True,
    ):
        for mode, offset in (("multi_read", -width / 2), ("readv", width / 2)):
            values = [
                float(by_key[(layout, engine, direct, mode)]["p50_ms"])
                for layout in LAYOUTS
            ]
            bars = axis.bar(
                [position + offset for position in positions],
                values,
                width=width,
                color=colors[mode],
                edgecolor="black",
                linewidth=1.0,
                label=labels[mode],
            )
            axis.bar_label(bars, fmt="%.1f", padding=2, fontsize=7)
        access = "O_DIRECT" if direct else "Buffered"
        axis.set_title(f"{engine} · {access}")
        axis.set_xticks(
            positions,
            [layout.replace("_", " ") for layout in LAYOUTS],
            rotation=12,
            ha="right",
            fontsize=8,
        )
        axis.grid(axis="y", linestyle="--", alpha=0.3)
        axis.set_ylim(bottom=0)
    axes[0, 0].set_ylabel("Latency (ms)")
    axes[1, 0].set_ylabel("Latency (ms)")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        loc="lower left",
        bbox_to_anchor=(0.30, 0.92, 0.40, 0.05),
        ncol=2,
        mode="expand",
        frameon=False,
    )
    figure.suptitle(
        f"SSD scatter/gather read latency ({token_count:,} tokens)", y=0.995
    )
    figure.tight_layout(rect=(0, 0, 1, 0.90))
    figure.savefig(output, dpi=180)
    plt.close(figure)


def analyze(samples_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(samples_path.read_text(encoding="utf-8"))
    if payload.get("status") != "completed":
        raise RuntimeError("refusing to summarize an incomplete run")
    rows = summarize_samples(list(payload["samples"]))
    output_dir = samples_path.parent
    write_csv(output_dir / "summary.csv", rows)
    atomic_write_json(output_dir / "summary.json", rows)
    plot_latency(
        rows,
        output_dir / "latency_bar_chart.png",
        int(payload["benchmark"]["token_count"]),
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("samples", type=Path)
    args = parser.parse_args()
    analyze(args.samples.resolve())


if __name__ == "__main__":
    main()
