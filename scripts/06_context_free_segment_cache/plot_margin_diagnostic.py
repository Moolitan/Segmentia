"""Plot P0 logprob margin diagnostic figures.

This script only reads post-run CSV artifacts under
results/problem_exploration/logprob_margin_diagnostic. It does not start vLLM
or regenerate experiment outputs.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from config import RESULTS_DIR  # noqa: E402

DEFAULT_ROOT = RESULTS_DIR / "logprob_margin_diagnostic"

COLORS = {
    "direct": "#8593A8",
    "rope": "#4C72B0",
    "diverged": "#C65B5B",
    "stable": "#5BA672",
    "risk": "#D9C26A",
    "text": "#2E3440",
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.edgecolor": "black",
        "axes.linewidth": 1,
        "text.color": "black",
        "xtick.color": "black",
        "ytick.color": "black",
    }
)


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def to_bool(value: str) -> bool:
    return str(value).lower() == "true"


def to_float(value: str | None) -> float | None:
    if value in (None, "", "None"):
        return None
    return float(value)


def short_case(row: dict[str, str]) -> str:
    task = row["task"].replace("_", " ")
    task = "".join(part[0].upper() for part in task.split()[:3])
    skill = row["skill"].replace("-", " ")
    skill = "".join(part[0].upper() for part in skill.split()[:2])
    return f"{task}/{skill}/o{row['occurrence']}"


def fmt(value: float | str | None) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, str):
        value = float(value)
    return f"{value:.2f}"


def plot_case_risk_map(root: Path) -> Path:
    rows = load_csv(root / "tables" / "margin_case_diagnostic_summary.csv")
    rows = sorted(rows, key=lambda r: (r["task"], r["skill"], int(r["occurrence"])))

    labels = [short_case(r) for r in rows]
    direct_div = [to_bool(r["direct_diverged"]) for r in rows]
    rope_div = [to_bool(r["rope_diverged"]) for r in rows]
    any_div = [d or q for d, q in zip(direct_div, rope_div)]
    rec_min_zero = [
        (to_float(r["recompute_min_margin_decision_window"]) or 0.0) == 0.0
        for r in rows
    ]
    struct = [to_float(r["recompute_min_structural_margin_decision_window"]) for r in rows]
    struct_plot = [np.nan if v is None else min(v, 8.0) for v in struct]

    matrix = np.array([direct_div, rope_div, rec_min_zero], dtype=float)

    fig = plt.figure(figsize=(15, 6.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.05, 1.0], hspace=0.42)
    ax = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax)

    cmap = matplotlib.colors.ListedColormap(["#F4F1EA", COLORS["diverged"]])
    ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["direct action diverged", "rope action diverged", "recompute min margin = 0"], fontsize=10)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([])
    ax.set_title(
        "P0 case risk map: every action-divergent case sits in a zero-margin recompute window",
        fontsize=12,
        fontweight="bold",
    )
    for x, diverged in enumerate(any_div):
        if diverged:
            ax.axvspan(x - 0.5, x + 0.5, color=COLORS["diverged"], alpha=0.08, zorder=0)
    for x in range(len(labels) + 1):
        ax.axvline(x - 0.5, color="white", lw=0.8)
    for y in range(4):
        ax.axhline(y - 0.5, color="white", lw=0.8)

    x = np.arange(len(labels))
    bars = ax2.bar(
        x,
        struct_plot,
        color=[COLORS["diverged"] if d else COLORS["stable"] for d in any_div],
        edgecolor="black",
        linewidth=0.6,
        zorder=3,
    )
    for bar, raw in zip(bars, struct):
        if raw is None:
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                0.2,
                "NA",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )
        elif raw > 8:
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                8.15,
                f"{raw:.1f}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )
    ax2.set_ylim(0, 9.2)
    ax2.set_ylabel("recompute structural\nmin margin (capped at 8)", fontsize=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=55, ha="right", fontsize=8)
    ax.tick_params(axis="x", bottom=False, labelbottom=False)
    ax2.grid(axis="y", alpha=0.25, lw=0.5, ls="--", zorder=0)
    ax2.set_axisbelow(True)
    ax2.text(
        0.995,
        0.96,
        "red bars/columns = at least one reuse mode changed action",
        transform=ax2.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color=COLORS["text"],
    )

    out = root / "figures" / "margin_case_risk_map.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_group_comparison(root: Path) -> Path:
    rows = load_csv(root / "tables" / "margin_group_summary.csv")
    by = {(r["mode"], r["diverged_from_recompute"]): r for r in rows}
    groups = [("direct", "False"), ("direct", "True"), ("rope", "False"), ("rope", "True")]
    labels = ["direct\nstable", "direct\ndiverged", "rope\nstable", "rope\ndiverged"]
    colors = [
        COLORS["stable"],
        COLORS["diverged"],
        COLORS["stable"],
        COLORS["diverged"],
    ]

    mean_margin = [to_float(by[g]["mean_mode_mean_margin_decision_window"]) for g in groups]
    struct_margin = [to_float(by[g]["mean_mode_min_structural_margin_decision_window"]) for g in groups]
    rec_zero = [int(by[g]["zero_recompute_min_margin_decision_window"]) for g in groups]
    n = [int(by[g]["n"]) for g in groups]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8))
    x = np.arange(len(groups))

    bars1 = ax1.bar(x, mean_margin, color=colors, edgecolor="black", linewidth=0.7, zorder=3)
    for bar, val in zip(bars1, mean_margin):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.15, fmt(val), ha="center", va="bottom", fontsize=9)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel("mean margin in decision window", fontsize=10)
    ax1.set_title("Diverged rows have lower average decision-window margin", fontsize=11, fontweight="bold")
    ax1.grid(axis="y", alpha=0.25, lw=0.5, ls="--", zorder=0)
    ax1.set_axisbelow(True)

    bars2 = ax2.bar(x, struct_margin, color=colors, edgecolor="black", linewidth=0.7, zorder=3)
    for bar, val, zeros, total in zip(bars2, struct_margin, rec_zero, n):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.12, fmt(val), ha="center", va="bottom", fontsize=9)
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            0.06,
            f"zero-min\n{zeros}/{total}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=COLORS["text"],
        )
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel("mean structural-token min margin", fontsize=10)
    ax2.set_title("Structural-margin separation is weak, especially for rope", fontsize=11, fontweight="bold")
    ax2.grid(axis="y", alpha=0.25, lw=0.5, ls="--", zorder=0)
    ax2.set_axisbelow(True)

    fig.suptitle(
        "P0 margin grouping: low-margin risk is visible, but not sufficient for prediction",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    out = root / "figures" / "margin_group_comparison.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_first_diff_scatter(root: Path) -> Path:
    rows = load_csv(root / "tables" / "margin_first_diff_summary.csv")
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    mode_marker = {"direct": "o", "rope": "^"}

    for mode in ["direct", "rope"]:
        for diverged, label, color in [
            (False, "same action", COLORS["stable"]),
            (True, "action diverged", COLORS["diverged"]),
        ]:
            selected = [
                r
                for r in rows
                if r["mode"] == mode and to_bool(r["headline_diverged_from_recompute"]) == diverged
            ]
            xs = [to_float(r["first_diff_token_index"]) for r in selected]
            pairs = [
                (to_float(r["first_diff_token_index"]), to_float(r["recompute_margin_at_first_diff"]))
                for r in selected
            ]
            pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
            if not pairs:
                continue
            xs = [x for x, _ in pairs]
            ys = [min(y, 5.0) for _, y in pairs]
            jitter = -0.7 if mode == "direct" else 0.7
            ax.scatter(
                [x + jitter for x in xs],
                ys,
                s=70,
                marker=mode_marker[mode],
                color=color,
                edgecolor="black",
                linewidth=0.7,
                alpha=0.9,
                label=f"{mode}: {label}",
                zorder=3,
            )
            for x0, raw_y in pairs:
                if raw_y > 5.0:
                    ax.text(
                        x0 + jitter,
                        5.08,
                        f"{raw_y:.1f}",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        color=COLORS["text"],
                    )

    ax.axhline(1.0, color="#666666", lw=1.0, ls=(0, (5, 3)), zorder=1)
    ax.axhline(0.25, color="#666666", lw=0.8, ls=(0, (2, 3)), zorder=1)
    ax.text(0.99, 1.03, "margin = 1.0", transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=8)
    ax.text(0.99, 0.28, "margin = 0.25", transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=8)
    ax.set_xlabel("first generated-token divergence index vs recompute", fontsize=10)
    ax.set_ylabel("recompute margin at first divergence token", fontsize=10)
    ax.set_ylim(-0.35, 5.55)
    ax.set_title(
        "First token divergence happens early and is not a reliable action-decision alignment",
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(alpha=0.25, lw=0.5, ls="--", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8.5, loc="upper right", framealpha=0.95, ncol=2)

    out = root / "figures" / "margin_first_diff_scatter.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    args = parser.parse_args()

    root = Path(args.root)
    outputs = [
        plot_case_risk_map(root),
        plot_group_comparison(root),
        plot_first_diff_scatter(root),
    ]
    for out in outputs:
        print(f"[done] {out}")


if __name__ == "__main__":
    main()
