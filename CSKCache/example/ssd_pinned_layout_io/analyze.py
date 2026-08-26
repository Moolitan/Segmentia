#!/usr/bin/env python3
"""Summarize and plot SSD-to-pinned layout I/O samples."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cskcache-layout-io-mpl")
import matplotlib.pyplot as plt

import config as cfg
from common import LAYOUTS, atomic_write_json, summarize_samples


POLICIES = (
    ("posix", False, "POSIX\nbuffered"),
    ("posix", True, "POSIX\nO_DIRECT"),
    ("io_uring", False, "io_uring\nbuffered"),
    ("io_uring", True, "io_uring\nO_DIRECT"),
)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_latency_bars(
    rows: list[dict[str, Any]], output: Path, *, token_count: int
) -> None:
    by_key = {
        (row["layout"], row["io_engine"], bool(row["use_odirect"])): row
        for row in rows
    }
    figure, axis = plt.subplots(figsize=(12.5, 4.6))
    group_positions = list(range(len(LAYOUTS)))
    bar_width = 0.19
    offsets = (-1.5, -0.5, 0.5, 1.5)
    for policy_index, (engine, direct, label) in enumerate(POLICIES):
        values = [
            float(by_key[(layout, engine, direct)]["p50_ms"])
            for layout in LAYOUTS
        ]
        positions = [
            position + offsets[policy_index] * bar_width
            for position in group_positions
        ]
        bars = axis.bar(
            positions,
            values,
            width=bar_width,
            label=label.replace("\n", " "),
            edgecolor="black",
            linewidth=1.0,
        )
        axis.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
    axis.set_xticks(group_positions, [layout.replace("_", " ") for layout in LAYOUTS])
    axis.set_ylabel("Latency (ms)")
    axis.set_title(
        f"SSD to matched pinned layout latency ({token_count:,} tokens)",
        pad=34,
    )
    axis.grid(axis="y", linestyle="--", alpha=0.35)
    axis.legend(
        loc="lower left",
        bbox_to_anchor=(0.06, 1.01, 0.88, 0.1),
        ncol=4,
        mode="expand",
        frameon=False,
    )
    axis.set_ylim(bottom=0)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def analyze(samples_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(samples_path.read_text(encoding="utf-8"))
    if payload.get("status") != "completed":
        raise RuntimeError("refusing to summarize an incomplete run")
    rows = summarize_samples(list(payload["samples"]))
    token_count = int(payload.get("benchmark", {}).get("token_count", cfg.TOKEN_COUNT))
    output_dir = samples_path.parent
    write_csv(output_dir / "summary.csv", rows)
    atomic_write_json(output_dir / "summary.json", rows)
    plot_latency_bars(
        rows,
        output_dir / "latency_bar_chart.png",
        token_count=token_count,
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("samples", type=Path)
    args = parser.parse_args()
    analyze(args.samples.resolve())


if __name__ == "__main__":
    main()
