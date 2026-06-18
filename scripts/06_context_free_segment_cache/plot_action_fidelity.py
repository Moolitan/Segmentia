"""Plot action-level / trajectory fidelity per reuse mode.

Bars are the mean over cases of the first-class behavior metrics produced by
evaluate_outputs.py (trajectory / modality match rate and full-sequence
embedding cosine). Works for both the main metrics_rows.csv (direct, rope) and
the value-repair metrics CSV (rope, vrep, krep, oracle).
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from config import DEFAULT_METRICS_CSV, RESULTS_DIR  # noqa: E402

DEFAULT_PLOT_PATH = (
    RESULTS_DIR
    / "headline_semantic_action_gap"
    / "figures"
    / "headline_action_fidelity.png"
)

# Stable left-to-right mode order; only modes present in the CSV are drawn.
MODE_ORDER = ["direct", "rope", "vrep", "krep", "oracle"]
MODE_LABEL = {
    "direct": "direct",
    "rope": "rope\n(key fix)",
    "vrep": "vrep\n(+value)",
    "krep": "krep\n(oracle key)",
    "oracle": "oracle\n(both)",
}

METRICS = [
    ("trajectory_match_rate", "Trajectory match"),
    ("modality_match_rate", "Modality match"),
    ("full_embedding_cos", "Full-seq cosine"),
]
COLORS = ["#4C72B0", "#DD8452", "#55A868"]


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def mean_by_mode(rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    buckets: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        for col, _ in METRICS:
            raw = row.get(col, "")
            if raw and raw != "None":
                buckets[row["mode"]][col].append(float(raw))
    return {
        mode: {col: sum(v) / len(v) for col, v in cols.items()}
        for mode, cols in buckets.items()
    }


def main() -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_METRICS_CSV))
    parser.add_argument("--output", default=str(DEFAULT_PLOT_PATH))
    args = parser.parse_args()

    rows = load_csv(Path(args.input))
    data = mean_by_mode(rows)
    modes = [m for m in MODE_ORDER if m in data]
    if not modes:
        raise SystemExit(f"No known modes found in {args.input}")

    x = np.arange(len(modes))
    n = len(METRICS)
    bar_w = 0.24
    offsets = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * bar_w

    fig, ax = plt.subplots(figsize=(1.7 * len(modes) + 3, 5))
    for (col, label), off, color in zip(METRICS, offsets, COLORS):
        vals = [data[m].get(col, 0.0) for m in modes]
        bars = ax.bar(x + off, vals, bar_w * 0.9, label=label, color=color,
                      edgecolor="black", linewidth=0.6, alpha=0.9)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([MODE_LABEL.get(m, m) for m in modes], fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.axhline(1.0, color="#888888", lw=0.7, ls="--", alpha=0.6)
    ax.set_ylabel("Mean over cases", fontsize=10)
    ax.set_title("Behavior fidelity vs recompute (higher = more faithful)",
                 fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.25, lw=0.5, ls="--")
    ax.legend(fontsize=9, loc="lower right", framealpha=0.9)

    plt.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[done] {out}")


if __name__ == "__main__":
    main()
