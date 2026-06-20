"""Plot Segmentia thinking-to-action diagnostic figures."""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt

from config import RESULTS_DIR

DEFAULT_ROOT = RESULTS_DIR / "thinking_to_action_divergence"

CATEGORY_LABELS = {
    "A_thinking_similar_action_same": "A: similar thinking\nsame action",
    "B_thinking_similar_action_diverged": "B: similar thinking\ndiverged action",
    "C_thinking_different_action_diverged": "C: different thinking\ndiverged action",
    "D_thinking_different_action_same": "D: different thinking\nsame action",
}
CATEGORY_ORDER = [
    "A_thinking_similar_action_same",
    "B_thinking_similar_action_diverged",
    "C_thinking_different_action_diverged",
    "D_thinking_different_action_same",
]
CATEGORY_COLORS = {
    "A_thinking_similar_action_same": "#4C78A8",
    "B_thinking_similar_action_diverged": "#D64F4F",
    "C_thinking_different_action_diverged": "#8E5EA2",
    "D_thinking_different_action_same": "#72A24D",
}


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def to_float(value: str | None) -> float | None:
    if value in (None, "", "None"):
        return None
    return float(value)


def plot_category_counts(rows: list[dict[str, str]], root: Path) -> Path:
    counts = Counter(row["category"] for row in rows)
    values = [counts.get(cat, 0) for cat in CATEGORY_ORDER]
    colors = [CATEGORY_COLORS[cat] for cat in CATEGORY_ORDER]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    bars = ax.bar(range(len(values)), values, color=colors, edgecolor="black", linewidth=0.8)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.25, str(value), ha="center", va="bottom", fontsize=11)
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels([CATEGORY_LABELS[cat] for cat in CATEGORY_ORDER], fontsize=9)
    ax.set_ylabel("case pairs", fontsize=10)
    ax.set_ylim(0, max(values) + 3)
    ax.set_title("Thinking-to-action categories: most divergences keep similar thinking", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.25, linewidth=0.7)
    fig.tight_layout()
    out = root / "figures" / "thinking_action_category_counts.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_embedding_margin(rows: list[dict[str, str]], root: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    for cat in CATEGORY_ORDER:
        selected = [row for row in rows if row["category"] == cat]
        xs = [to_float(row["embedding_cosine"]) for row in selected]
        ys = [to_float(row["boundary_margin_delta_rope_minus_recompute"]) for row in selected]
        points = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
        if not points:
            continue
        ax.scatter(
            [p[0] for p in points],
            [p[1] for p in points],
            s=58,
            color=CATEGORY_COLORS[cat],
            edgecolor="black",
            linewidth=0.6,
            alpha=0.9,
            label=CATEGORY_LABELS[cat].replace("\n", " "),
        )
    ax.axhline(0, color="#222222", linewidth=0.9, linestyle="--")
    ax.axvline(0.80, color="#777777", linewidth=0.8, linestyle=":", label="embedding threshold 0.80")
    ax.set_xlabel("thinking embedding cosine (recompute vs rope)", fontsize=10)
    ax.set_ylabel("boundary margin delta: rope - recompute", fontsize=10)
    ax.set_title("Divergence usually occurs despite high thinking embedding similarity", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.25, linewidth=0.7)
    ax.legend(loc="best", fontsize=8, frameon=True)
    fig.tight_layout()
    out = root / "figures" / "embedding_vs_boundary_margin_delta.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_boundary_margin_pairs(rows: list[dict[str, str]], root: Path) -> Path:
    sorted_rows = sorted(
        rows,
        key=lambda r: (
            r["action_diverged"] != "True",
            r["task"],
            r["skill"],
            int(r["occurrence"]),
        ),
    )
    labels = [f"{r['task'].split('_')[0]}:{r['skill']}:{r['occurrence']}" for r in sorted_rows]
    rec = [to_float(r["recompute_boundary_margin"]) for r in sorted_rows]
    rope = [to_float(r["rope_boundary_margin"]) for r in sorted_rows]
    colors = ["#D64F4F" if r["action_diverged"] == "True" else "#4C78A8" for r in sorted_rows]
    x = list(range(len(sorted_rows)))

    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    for i, (rv, pv, color) in enumerate(zip(rec, rope, colors)):
        if rv is None or pv is None:
            continue
        ax.plot([i, i], [rv, pv], color=color, alpha=0.45, linewidth=1.2)
    ax.scatter(x, rec, marker="o", s=38, color="#F2C14E", edgecolor="black", linewidth=0.5, label="recompute")
    ax.scatter(x, rope, marker="s", s=38, color=colors, edgecolor="black", linewidth=0.5, label="rope")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=7)
    ax.set_ylabel("action-boundary margin", fontsize=10)
    ax.set_title("Boundary margins by case: red rope markers are action-divergent", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.25, linewidth=0.7)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    out = root / "figures" / "boundary_margin_pairs.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    args = parser.parse_args()
    root = Path(args.root)
    rows = load_rows(root / "tables" / "thinking_pair_summary.csv")
    outputs = [
        plot_category_counts(rows, root),
        plot_embedding_margin(rows, root),
        plot_boundary_margin_pairs(rows, root),
    ]
    for out in outputs:
        print(f"[plot] {out}")


if __name__ == "__main__":
    main()
