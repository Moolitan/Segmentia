#!/usr/bin/env python3
"""Plot the H2D microbenchmark's three-factor result (memory, position
shift, profiling on/off) from summarize.py's summary.csv.

Reads only summary.csv (already-aggregated per-case percentiles/means); it
never re-derives numbers from raw_iterations.jsonl. Writes a long-form CSV
next to the figure so the numbers behind each bar are inspectable without
re-running the benchmark.

Two panels sharing the same four (memory, position_shift) groups on the
x-axis:
  (a) end-to-end wall time (ms), profiling on vs. off, per group — shows the
      overall effect of pinning and of the RoPE position shift, and the
      (small) added cost of profiling itself.
  (b) GPU-side stage breakdown (ms) for the profiling=on cases only, since
      only those runs have per-stage LoadTrace data — shows *why* (a) looks
      the way it does: pinning shrinks Key/Value H2D, a nonzero shift adds a
      RoPE segment.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cskcache-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/cskcache-xdg-cache")

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paper_plot_style import COLORS, apply_publication_style, save_figure

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = Path(
    os.environ.get(
        "CSKCACHE_H2D_OUTPUT_ROOT",
        "/mnt/Large_Language_Model_Lab_1/wsh/Segmentia/output/07_cskcache/h2d_microbenchmark",
    )
)
DEFAULT_RESULT_DIR = REPO_ROOT / "results" / "problem_exploration" / "h2d_microbenchmark"

# (memory, position_shift) groups, in the order shown on the x-axis: pinned
# vs. pageable side by side, split further by whether a position shift
# forces a RoPE recompute.
GROUPS = [("pageable", 0), ("pinned", 0), ("pageable", 17000), ("pinned", 17000)]
STAGE_FIELDS = [
    ("Key H2D", "cuda_key_h2d_ms_mean"),
    ("Value H2D", "cuda_value_h2d_ms_mean"),
    ("RoPE", "cuda_rope_ms_mean"),
    ("Scatter", "cuda_scatter_span_ms_mean"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-csv",
        type=Path,
        help="Path to summarize.py's summary.csv. If omitted, use the newest "
        "run under --output-root.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--name", default="h2d_benchmark")
    return parser.parse_args()


def newest_summary(output_root: Path) -> Path:
    candidates = list(output_root.glob("*/summary.csv"))
    if not candidates:
        raise FileNotFoundError(f"No summary.csv found under {output_root}")
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def read_rows(path: Path) -> dict[tuple[str, int, str], dict[str, str]]:
    rows: dict[tuple[str, int, str], dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            key = (row["memory"], int(row["position_shift"]), row["profiling"])
            rows[key] = row
    missing = [
        (memory, shift, profiling)
        for memory, shift in GROUPS
        for profiling in ("off", "on")
        if (memory, shift, profiling) not in rows
    ]
    if missing:
        raise ValueError(f"summary.csv is missing expected cases: {missing}")
    return rows


def write_long_csv(rows: dict[tuple[str, int, str], dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["memory", "position_shift", "profiling", "metric", "value_ms"])
        for (memory, shift), profiling in ((g, p) for g in GROUPS for p in ("off", "on")):
            row = rows[(memory, shift, profiling)]
            writer.writerow(
                [memory, shift, profiling, "end_to_end_wall_ms_p50", row["end_to_end_wall_ms_p50"]]
            )
            writer.writerow(
                [memory, shift, profiling, "end_to_end_wall_ms_p10", row["end_to_end_wall_ms_p10"]]
            )
            writer.writerow(
                [memory, shift, profiling, "end_to_end_wall_ms_p90", row["end_to_end_wall_ms_p90"]]
            )
            if profiling == "on":
                for stage_name, field in STAGE_FIELDS:
                    writer.writerow([memory, shift, profiling, stage_name, row[field]])


def plot_wall_time(ax: plt.Axes, rows: dict[tuple[str, int, str], dict[str, str]]) -> None:
    x = range(len(GROUPS))
    width = 0.35
    for offset, profiling, color in ((-1, "off", "#4C78A8"), (1, "on", "#E45756")):
        p50 = [float(rows[(m, s, profiling)]["end_to_end_wall_ms_p50"]) for m, s in GROUPS]
        p10 = [float(rows[(m, s, profiling)]["end_to_end_wall_ms_p10"]) for m, s in GROUPS]
        p90 = [float(rows[(m, s, profiling)]["end_to_end_wall_ms_p90"]) for m, s in GROUPS]
        lower = [max(v - lo, 0.0) for v, lo in zip(p50, p10)]
        upper = [max(hi - v, 0.0) for v, hi in zip(p50, p90)]
        positions = [xi + offset * width / 2 for xi in x]
        bars = ax.bar(
            positions,
            p50,
            width=width,
            color=color,
            label=f"profiling {profiling}",
            zorder=3,
        )
        ax.errorbar(
            positions,
            p50,
            yerr=[lower, upper],
            fmt="none",
            ecolor="#333333",
            elinewidth=1.0,
            capsize=3,
            zorder=4,
        )
        for bar, value in zip(bars, p50):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(upper) * 0.15 + 0.5,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{m}\nshift={s}" for m, s in GROUPS])
    ax.set_ylabel("End-to-end wall time (ms)\nmedian, p10–p90 whiskers")
    # Extra headroom above the tallest bar+whisker+label so the legend sits
    # in clear space instead of overlapping the value labels.
    ax.set_ylim(0, max(p90 for p50, p10, p90 in _wall_time_bounds(rows)) * 1.4)
    ax.legend(frameon=False, loc="upper left", ncol=2)
    ax.yaxis.grid(True, color="#E4E4E0", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title("(a) Overall latency: memory × position shift × profiling")


def _wall_time_bounds(
    rows: dict[tuple[str, int, str], dict[str, str]]
) -> list[tuple[float, float, float]]:
    return [
        (
            float(rows[(m, s, p)]["end_to_end_wall_ms_p50"]),
            float(rows[(m, s, p)]["end_to_end_wall_ms_p10"]),
            float(rows[(m, s, p)]["end_to_end_wall_ms_p90"]),
        )
        for m, s in GROUPS
        for p in ("off", "on")
    ]


def plot_stage_breakdown(ax: plt.Axes, rows: dict[tuple[str, int, str], dict[str, str]]) -> None:
    x = range(len(GROUPS))
    bottoms = [0.0] * len(GROUPS)
    for stage_name, field in STAGE_FIELDS:
        values = [float(rows[(m, s, "on")][field]) for m, s in GROUPS]
        ax.bar(
            list(x),
            values,
            bottom=bottoms,
            color=COLORS[stage_name],
            edgecolor="white",
            linewidth=0.8,
            label=stage_name,
            zorder=3,
        )
        bottoms = [b + v for b, v in zip(bottoms, values)]
    for xi, total in zip(x, bottoms):
        ax.text(xi, total + 0.6, f"{total:.1f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{m}\nshift={s}" for m, s in GROUPS])
    ax.set_ylabel("GPU stage time (ms)\nmean, profiling=on runs only")
    # Same headroom trick as panel (a): keep the legend clear of the
    # tallest stack's total-time label.
    ax.set_ylim(0, max(bottoms) * 1.4)
    ax.legend(frameon=False, loc="upper left", ncol=2, fontsize=7)
    ax.yaxis.grid(True, color="#E4E4E0", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title("(b) Where the GPU time goes (to_gpu() stage breakdown)")


def plot(rows: dict[tuple[str, int, str], dict[str, str]], output_stem: Path) -> None:
    apply_publication_style()
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(9.5, 3.6))
    plot_wall_time(ax_a, rows)
    plot_stage_breakdown(ax_b, rows)
    fig.tight_layout()
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, str(output_stem))
    plt.close(fig)


def main() -> None:
    args = parse_args()
    summary_csv = (
        args.summary_csv.resolve()
        if args.summary_csv is not None
        else newest_summary(args.output_root).resolve()
    )
    rows = read_rows(summary_csv)
    figure_dir = args.result_dir / "figures"
    data_path = args.result_dir / "data" / f"{args.name}.csv"
    write_long_csv(rows, data_path)
    output_stem = figure_dir / args.name
    plot(rows, output_stem)
    print(f"summary_csv={summary_csv}")
    print(f"data_csv={data_path}")
    print(f"figure_pdf={output_stem}.pdf")
    print(f"figure_png={output_stem}.png")


if __name__ == "__main__":
    main()
