"""Plot per-(skill, task) behavior-fidelity metrics for the headline experiment.

Two subplots (direct / rope), one bar group per skill×task, five metrics per
group: trajectory match rate, tool-set match rate, full-sequence semantic
cosine, BLEU-4 and ROUGE-L.  The layout and style follow the metrics_by_skill
convention.
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

from config import RESULTS_DIR  # noqa: E402

DEFAULT_INPUT = (
    RESULTS_DIR
    / "headline_semantic_action_gap"
    / "tables"
    / "headline_metrics_rows.csv"
)
DEFAULT_OUTPUT = (
    RESULTS_DIR
    / "headline_semantic_action_gap"
    / "figures"
    / "headline_by_skill.png"
)

TASK_ABBREV: dict[str, str] = {
    "doc_coauthoring_design_doc":     "DocCoauth",
    "internal_comms_incident_update": "IntComms",
    "launch_poster_page_pack":        "Poster",
    "mcp_server_and_spec":            "MCPSpec",
    "slack_launch_pack":              "Slack",
    "web_artifact_with_theme":        "WebArt",
}

SKILL_TASK_ORDER: list[tuple[str, str]] = [
    ("brand-guidelines",      "slack_launch_pack"),
    ("canvas-design",         "launch_poster_page_pack"),
    ("doc-coauthoring",       "doc_coauthoring_design_doc"),
    ("doc-coauthoring",       "mcp_server_and_spec"),
    ("internal-comms",        "internal_comms_incident_update"),
    ("internal-comms",        "slack_launch_pack"),
    ("mcp-builder",           "mcp_server_and_spec"),
    ("slack-gif-creator",     "slack_launch_pack"),
    ("theme-factory",         "launch_poster_page_pack"),
    ("theme-factory",         "web_artifact_with_theme"),
    ("web-artifacts-builder", "launch_poster_page_pack"),
    ("web-artifacts-builder", "web_artifact_with_theme"),
]

SKILL_ABBREV: dict[str, str] = {
    "brand-guidelines":     "BrandGd",
    "canvas-design":        "Canvas",
    "doc-coauthoring":      "DocCoauth",
    "internal-comms":       "IntComms",
    "mcp-builder":          "MCPBuild",
    "slack-gif-creator":    "SlackGIF",
    "theme-factory":        "ThemeFact",
    "web-artifacts-builder": "WebArt",
}

# Three discrete/semantic scores + two surface-overlap scores
METRICS: list[tuple[str, str]] = [
    ("trajectory_match_rate", "Trajectory match"),
    ("tool_set_match_rate",   "Tool-set match"),
    ("full_embedding_cos",    "Semantic cosine"),
    ("full_bleu4",            "BLEU-4"),
    ("full_rouge_l",          "ROUGE-L"),
]

# Muted, cohesive qualitative palette (soft scientific look)
COLORS = ["#4C72B0", "#E1934B", "#5BA672", "#C65B5B", "#8579B0"]

MODE_LABEL = {"direct": "direct  ·  inject KV as-is",
              "rope":   "rope  ·  inject + RoPE key correction"}


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def aggregate(rows: list[dict[str, str]]) -> dict[str, dict[tuple[str, str], dict[str, float]]]:
    """Return {mode: {(skill, task): {col: mean_over_occurrences}}}."""
    buckets: dict[str, dict[tuple[str, str], dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for row in rows:
        key = (row["skill"], row["task"])
        for col, _ in METRICS:
            raw = row.get(col, "")
            if raw and raw not in ("None", ""):
                buckets[row["mode"]][key][col].append(float(raw))
    return {
        mode: {k: {col: sum(v) / len(v) for col, v in cols.items()}
               for k, cols in keys.items()}
        for mode, keys in buckets.items()
    }


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import numpy as np

    # ---- global style: clean, light, modern -----------------------------
    plt.rcParams.update({
        "font.family":      "DejaVu Sans",
        "axes.edgecolor":   "black",
        "axes.linewidth":   1,
        "axes.titlesize":   11,
        "xtick.color":      "black",
        "ytick.color":      "black",
        "text.color":       "black",
    })

    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    rows = load_csv(Path(args.input))
    data = aggregate(rows)

    n_groups  = len(SKILL_TASK_ORDER)
    n_metrics = len(METRICS)
    bar_w  = 0.15
    x      = np.arange(n_groups)
    offsets = np.linspace(-(n_metrics - 1) / 2, (n_metrics - 1) / 2, n_metrics) * bar_w

    xlabels = [f"{SKILL_ABBREV[s]}\n({TASK_ABBREV[t]})" for s, t in SKILL_TASK_ORDER]

    shared_skills = {s for s, _ in SKILL_TASK_ORDER
                     if sum(1 for sk, _ in SKILL_TASK_ORDER if sk == s) > 1}

    fig, axes = plt.subplots(
        2, 1, figsize=(19, 9.5), sharex=True,
        gridspec_kw={"hspace": 0.30},
    )
    fig.suptitle(
        "Behavior fidelity vs recompute, per skill × task",
        fontsize=14, fontweight="bold", y=0.94,
    )

    legend_handles = None

    for ax, mode in zip(axes, ["direct", "rope"]):
        mode_data = data.get(mode, {})

        # subtle alternating background bands for multi-task skills
        prev_skill, shade = None, False
        for i, (skill, _) in enumerate(SKILL_TASK_ORDER):
            if skill != prev_skill:
                shade = not shade
                prev_skill = skill
            if skill in shared_skills and shade:
                ax.axvspan(i - 0.5, i + 0.5, color="black", alpha=0.035, zorder=0)

        bars_by_metric = []
        for (col, label), offset, color in zip(METRICS, offsets, COLORS):
            vals = [mode_data.get(key, {}).get(col, 0.0) for key in SKILL_TASK_ORDER]
            bars = ax.bar(
                x + offset, vals, bar_w * 0.90,
                label=label, color=color, alpha=0.92,
                edgecolor="black", linewidth=0.8, zorder=3,
            )
            bars_by_metric.append(bars)
            # vertical value labels avoid the horizontal label collisions
            for bar, val in zip(bars, vals):
                if val > 0.02:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.02,
                        f"{val:.2f}",
                        ha="center", va="bottom", rotation=90,
                        fontsize=10, color="black", zorder=4,
                    )
                else:
                    # 真 0：画一个贴着 x 轴的小标记，避免和"没柱子"混淆
                    ax.scatter(
                        bar.get_x() + bar.get_width() / 2, 0.0,
                        marker="v", s=18, color=bar.get_facecolor(),
                        edgecolor="black", linewidth=0.5, zorder=5,
                    )
                    ax.text(
                        bar.get_x() + bar.get_width() / 2, 0.015,
                        "0", ha="center", va="bottom", 
                        fontsize=10, color="black", zorder=4,
                    )

        if legend_handles is None:
            legend_handles = [b[0] for b in bars_by_metric]

        # reference line for "perfect match"
        ax.axhline(1.0, color="black", lw=0.9, ls=(0, (4, 3)), alpha=0.7, zorder=1)
        ax.text(n_groups - 0.55, 1.012, "1.0",
                fontsize=7, color="black", ha="right", va="bottom")

        ax.set_title(MODE_LABEL.get(mode, mode), fontsize=11,
                     fontweight="bold", color="black", pad=10, loc="left")
        ax.set_ylim(0, 1.22)
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.set_ylabel("Score   (1.0 = matches recompute)", fontsize=12, labelpad=8)
        ax.grid(axis="y", alpha=0.35, lw=0.6, linestyle="--", color="black", zorder=0)
        ax.set_axisbelow(True)
        ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.1))

        # keep only the left + bottom spines for an airy look
        ax.tick_params(axis="y", labelsize=9, length=3)
        ax.tick_params(axis="x", length=0)
        ax.set_xlim(-0.6, n_groups - 0.4)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(xlabels, fontsize=8.5, linespacing=1.4, color="black")
    axes[-1].set_xlabel("Skill  ·  (Task)", fontsize=10, labelpad=8)

    # single shared legend, centered above both panels
    fig.legend(
        legend_handles, [lbl for _, lbl in METRICS],
        loc="upper center", bbox_to_anchor=(0.5, 0.918),
        ncol=n_metrics, frameon=False, fontsize=13,
        handlelength=1.3, columnspacing=2.0, handletextpad=0.6,
    )

    fig.subplots_adjust(top=0.88, bottom=0.10, left=0.055, right=0.985)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    print(f"[done] {out}")


if __name__ == "__main__":
    main()
