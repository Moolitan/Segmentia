"""Plot per-case thinking/action embedding cosine for the temp=0.6 audit."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_INPUT = (
    ROOT
    / "results/problem_exploration/raw_decode_token_sequences/tables"
    / "temp0.6_without_occ12_embedding_cosine.csv"
)
DEFAULT_OUTPUT_DIR = (
    ROOT / "results/problem_exploration/raw_decode_token_sequences/figures"
)

COMPARISONS = (
    "sampling_baseline",
    "same_seed_recompute_vs_rope",
    "cross_seed_recompute_vs_rope",
)
COMPARISON_LABELS = {
    "sampling_baseline": "RC1–RC2\nsampling baseline",
    "same_seed_recompute_vs_rope": "RC1–RoPE",
    "cross_seed_recompute_vs_rope": "RC2–RoPE",
}
COMPARISON_COLORS = {
    "sampling_baseline": "#0072B2",
    "same_seed_recompute_vs_rope": "#D55E00",
    "cross_seed_recompute_vs_rope": "#009E73",
}
EXPECTED_ARMS = {
    "sampling_baseline": ("recompute_run1", "recompute_run2"),
    "same_seed_recompute_vs_rope": ("recompute_run1", "rope"),
    "cross_seed_recompute_vs_rope": ("recompute_run2", "rope"),
}
METRIC_COLORS = {
    "thinking_cosine": "#56B4E9",
    "action_cosine": "#CC79A7",
}
OUTPUT_STEM = "temp0.6_without_occ12_embedding_cosine"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str | float]]:
    required = {
        "comparison",
        "left_arm",
        "right_arm",
        "filename",
        "thinking_cosine",
        "action_cosine",
    }
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"missing required columns: {sorted(missing)}")
        rows: list[dict[str, str | float]] = []
        for line_number, raw in enumerate(reader, start=2):
            row: dict[str, str | float] = dict(raw)
            for metric in ("thinking_cosine", "action_cosine"):
                value = float(raw[metric])
                if not math.isfinite(value) or not -1.0 <= value <= 1.0:
                    raise ValueError(
                        f"{path}:{line_number}: invalid {metric}={value}"
                    )
                row[metric] = value
            rows.append(row)

    groups = {comparison: [] for comparison in COMPARISONS}
    for row in rows:
        comparison = str(row["comparison"])
        if comparison not in groups:
            raise ValueError(f"unexpected comparison: {comparison}")
        groups[comparison].append(row)

    for comparison, group in groups.items():
        if len(group) != 12:
            raise ValueError(f"{comparison}: expected 12 rows, found {len(group)}")
        filenames = {str(row["filename"]) for row in group}
        if len(filenames) != len(group):
            raise ValueError(f"{comparison}: duplicate filename")
        actual_arms = {
            (str(row["left_arm"]), str(row["right_arm"])) for row in group
        }
        if actual_arms != {EXPECTED_ARMS[comparison]}:
            raise ValueError(
                f"{comparison}: unexpected arm pairs {sorted(actual_arms)}"
            )
    return rows


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def rows_for(
    rows: list[dict[str, str | float]], comparison: str
) -> list[dict[str, str | float]]:
    return [row for row in rows if row["comparison"] == comparison]


def plot(rows: list[dict[str, str | float]], output_dir: Path) -> None:
    configure_style()
    fig, (ax_scatter, ax_distribution) = plt.subplots(
        1, 2, figsize=(7.2, 3.25), gridspec_kw={"width_ratios": (1.0, 1.18)}
    )

    for comparison in COMPARISONS:
        group = rows_for(rows, comparison)
        thinking = [float(row["thinking_cosine"]) for row in group]
        action = [float(row["action_cosine"]) for row in group]
        color = COMPARISON_COLORS[comparison]
        ax_scatter.scatter(
            thinking,
            action,
            s=28,
            color=color,
            alpha=0.78,
            edgecolor="white",
            linewidth=0.45,
            label=COMPARISON_LABELS[comparison].replace("\n", " "),
        )
        ax_scatter.scatter(
            sum(thinking) / len(thinking),
            sum(action) / len(action),
            marker="D",
            s=58,
            color=color,
            edgecolor="black",
            linewidth=0.7,
            zorder=4,
        )

    ax_scatter.plot(
        [0.90, 1.0],
        [0.90, 1.0],
        linestyle="--",
        color="#777777",
        linewidth=0.8,
        zorder=0,
    )
    ax_scatter.set_xlim(0.90, 1.005)
    ax_scatter.set_ylim(0.52, 1.015)
    ax_scatter.set_xlabel("Thinking embedding cosine")
    ax_scatter.set_ylabel("Action embedding cosine")
    ax_scatter.grid(axis="both", color="#D9D9D9", linewidth=0.5, alpha=0.7)
    ax_scatter.legend(
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(-0.03, 1.01),
        ncol=1,
        handletextpad=0.45,
        borderaxespad=0,
    )
    ax_scatter.text(
        0.015,
        0.975,
        "A",
        transform=ax_scatter.transAxes,
        va="top",
        fontweight="bold",
    )

    positions: list[float] = []
    data: list[list[float]] = []
    colors: list[str] = []
    offsets = {"thinking_cosine": -0.18, "action_cosine": 0.18}
    deterministic_jitter = [
        -0.075,
        -0.061,
        -0.048,
        -0.034,
        -0.020,
        -0.007,
        0.007,
        0.020,
        0.034,
        0.048,
        0.061,
        0.075,
    ]
    for index, comparison in enumerate(COMPARISONS, start=1):
        group = sorted(rows_for(rows, comparison), key=lambda row: str(row["filename"]))
        for metric in ("thinking_cosine", "action_cosine"):
            position = index + offsets[metric]
            values = [float(row[metric]) for row in group]
            positions.append(position)
            data.append(values)
            colors.append(METRIC_COLORS[metric])
            ax_distribution.scatter(
                [position + jitter for jitter in deterministic_jitter],
                values,
                s=13,
                color=METRIC_COLORS[metric],
                alpha=0.72,
                edgecolor="none",
                zorder=3,
            )

    boxplots = ax_distribution.boxplot(
        data,
        positions=positions,
        widths=0.29,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.1},
        whiskerprops={"color": "#555555", "linewidth": 0.8},
        capprops={"color": "#555555", "linewidth": 0.8},
        boxprops={"color": "#555555", "linewidth": 0.8},
    )
    for patch, color in zip(boxplots["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.24)

    ax_distribution.set_xlim(0.55, 3.45)
    ax_distribution.set_ylim(0.52, 1.015)
    ax_distribution.set_xticks(range(1, 4))
    ax_distribution.set_xticklabels(
        [COMPARISON_LABELS[comparison] for comparison in COMPARISONS]
    )
    ax_distribution.set_ylabel("Embedding cosine")
    ax_distribution.grid(axis="y", color="#D9D9D9", linewidth=0.5, alpha=0.7)
    ax_distribution.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                color=METRIC_COLORS["thinking_cosine"],
                label="Thinking",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                color=METRIC_COLORS["action_cosine"],
                label="Action",
            ),
        ],
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01),
        ncol=2,
        handletextpad=0.35,
        columnspacing=1.2,
        borderaxespad=0,
    )
    ax_distribution.text(
        0.015,
        0.975,
        "B",
        transform=ax_distribution.transAxes,
        va="top",
        fontweight="bold",
    )

    fig.subplots_adjust(left=0.085, right=0.995, bottom=0.20, top=0.78, wspace=0.34)
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(
            output_dir / f"{OUTPUT_STEM}.{extension}",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.04,
        )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input)
    plot(rows, args.output_dir)
    print(f"Validated {len(rows)} rows from {args.input}")
    print(f"Wrote {args.output_dir / (OUTPUT_STEM + '.png')}")
    print(f"Wrote {args.output_dir / (OUTPUT_STEM + '.pdf')}")


if __name__ == "__main__":
    main()
