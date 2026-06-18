"""Generate the headline / stability / KV-substitution figures for result summary.md.

Produces four PNGs under results/06_context_free_segment_cache/:
  1. headline_summary.png      -- aggregate 5-metric bars (direct/rope) showing
                                  the "semantic high, action low" gap (the hook).
  2. stability.png             -- left: action self-consistency per mode vs the
                                  recompute noise floor; right: decomposition of
                                  each reuse mode's 24 cases into agree / systematic
                                  shift / sampling noise.
  3. value_repair_2x2_full.png -- aggregate 5-metric bars across the 4 KV-source
                                  conditions (rope / vrep / krep / recompute-splice).
  4. value_repair_heatmap.png  -- per-task trajectory-match heatmap, 6 tasks x 4
                                  conditions.

The per-(skill,task) headline detail figure is produced separately by
plot_metrics.py (headline_by_skill.png).
"""
from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from config import RESULTS_DIR  # noqa: E402

R = RESULTS_DIR
HEADLINE = R / "headline_semantic_action_gap"
STABILITY = R / "stability_systematic_vs_noise"
VALUE_REPAIR = R / "value_repair_key_value_diagnosis"

# Muted, cohesive palette shared with plot_metrics.py
COLORS = ["#4C72B0", "#E1934B", "#5BA672", "#C65B5B", "#8579B0"]
plt.rcParams.update({
    "font.family":    "DejaVu Sans",
    "axes.edgecolor": "black",
    "axes.linewidth": 1,
    "text.color":     "black",
    "xtick.color":    "black",
    "ytick.color":    "black",
})


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def fv(row: dict, col: str):
    v = row.get(col, "")
    return float(v) if v not in ("", "None") else None


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def case_key(r):
    return (r["task"], r["skill"], r["occurrence"])


TASK_ABBREV = {
    "doc_coauthoring_design_doc":     "DocCoauth",
    "internal_comms_incident_update": "IntComms",
    "launch_poster_page_pack":        "Poster",
    "mcp_server_and_spec":            "MCPSpec",
    "slack_launch_pack":              "Slack",
    "web_artifact_with_theme":        "WebArt",
}


# ---------------------------------------------------------------------------
# 1. headline_summary.png
# ---------------------------------------------------------------------------
def fig_headline_summary() -> None:
    rows = load_csv(HEADLINE / "tables" / "headline_metrics_rows.csv")
    metrics = [
        ("trajectory_match_rate", "Trajectory\nmatch"),
        ("tool_set_match_rate",   "Tool-set\nmatch"),
        ("modality_match_rate",   "Modality\nmatch"),
        ("full_bleu4",            "BLEU-4"),
        ("full_rouge_l",          "ROUGE-L"),
        ("full_embedding_cos",    "Semantic\ncosine"),
    ]
    modes = ["direct", "rope"]
    mode_color = {"direct": "#8593A8", "rope": "#4C72B0"}

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(metrics))
    bw = 0.38
    for i, mode in enumerate(modes):
        sel = [r for r in rows if r["mode"] == mode]
        vals = [mean(fv(r, col) for r in sel) for col, _ in metrics]
        off = (i - 0.5) * bw
        bars = ax.bar(x + off, vals, bw * 0.92, label=mode, color=mode_color[mode],
                      edgecolor="black", linewidth=0.8, zorder=3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=9, zorder=4)

    # shade three metric families: actions diverge, surface wording varies,
    # but the underlying meaning is preserved.
    ax.axvspan(-0.5, 2.5, color="#C65B5B", alpha=0.06, zorder=0)   # actions
    ax.axvspan(2.5, 4.5, color="#8593A8", alpha=0.06, zorder=0)    # surface text
    ax.axvspan(4.5, 5.5, color="#5BA672", alpha=0.08, zorder=0)    # meaning
    ax.text(1.0, 1.14, "discrete ACTIONS\n(diverge ~40%)", ha="center",
            fontsize=9.5, color="#9c3b3b", fontweight="bold")
    ax.text(3.5, 1.14, "surface wording\n(varies)", ha="center",
            fontsize=9.5, color="#4a566b", fontweight="bold")
    ax.text(5.0, 1.14, "MEANING\n(preserved)", ha="center",
            fontsize=9.5, color="#2f6b46", fontweight="bold")

    ax.axhline(1.0, color="black", lw=0.9, ls=(0, (4, 3)), alpha=0.6, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in metrics], fontsize=9)
    ax.set_ylim(0, 1.28)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel("Score  (1.0 = matches recompute)", fontsize=11)
    ax.set_title("Stage 1 (headline): reuse preserves meaning but shifts discrete actions\n"
                 "24 cases, 6 tasks, greedy decode (temperature 0)",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, lw=0.6, ls="--", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(title="reuse mode", fontsize=10, loc="center",
              bbox_to_anchor=(0.60, 0.62), framealpha=0.95)
    plt.tight_layout()
    out = HEADLINE / "figures" / "headline_summary.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[done] {out}")


# ---------------------------------------------------------------------------
# 2. stability.png
# ---------------------------------------------------------------------------
def fig_stability() -> None:
    rows = load_csv(STABILITY / "tables" / "stability_stability_rows.csv")
    by = defaultdict(dict)
    for r in rows:
        by[case_key(r)][r["mode"]] = r

    modes = ["recompute", "direct", "rope"]
    sc = {m: [float(by[k][m]["action_self_consistency"]) for k in by if m in by[k]]
          for m in modes}

    # decomposition of each reuse mode's cases vs recompute
    decomp = {}
    for mode in ["direct", "rope"]:
        cats = Counter()
        for k in by:
            rec, m = by[k]["recompute"], by[k][mode]
            if m["majority_action"] == rec["majority_action"]:
                cats["agree"] += 1
            elif float(m["action_self_consistency"]) == 1.0:
                cats["systematic"] += 1
            else:
                cats["noise"] += 1
        decomp[mode] = cats

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5),
                                   gridspec_kw={"width_ratios": [1, 1.15]})

    # -- left: self-consistency per mode, with recompute floor line --
    mode_color = {"recompute": "#5BA672", "direct": "#8593A8", "rope": "#4C72B0"}
    means = [mean(sc[m]) for m in modes]
    bars = axL.bar(range(len(modes)), means, 0.6,
                   color=[mode_color[m] for m in modes],
                   edgecolor="black", linewidth=0.8, zorder=3)
    for b, v in zip(bars, means):
        axL.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.008,
                 f"{v:.3f}", ha="center", va="bottom", fontsize=10, zorder=4)
    floor = mean(sc["recompute"])
    axL.axhline(floor, color="#2f6b46", lw=1.2, ls=(0, (5, 3)), zorder=2)
    axL.text(2.45, floor + 0.004, "recompute noise floor",
             ha="right", va="bottom", fontsize=9, color="#2f6b46")
    axL.set_xticks(range(len(modes)))
    axL.set_xticklabels(modes, fontsize=10)
    axL.set_ylim(0.6, 1.0)
    axL.set_ylabel("Action self-consistency\n(fraction repeating own majority over 3 samples)",
                   fontsize=10)
    axL.set_title("Reuse is slightly less self-consistent than recompute\n"
                  "(temperature 0.7, 3 samples / case)", fontsize=11, fontweight="bold")
    axL.grid(axis="y", alpha=0.3, lw=0.6, ls="--", zorder=0)
    axL.set_axisbelow(True)

    # -- right: stacked decomposition of the 24 cases --
    order = ["agree", "systematic", "noise"]
    labels = {
        "agree":      "agrees with recompute",
        "systematic": "SYSTEMATIC shift\n(reuse fully self-consistent, yet differs)",
        "noise":      "noise\n(reuse itself unstable)",
    }
    cat_color = {"agree": "#5BA672", "systematic": "#C65B5B", "noise": "#D9C26A"}
    dmodes = ["direct", "rope"]
    bottoms = np.zeros(len(dmodes))
    for cat in order:
        vals = np.array([decomp[m][cat] for m in dmodes], dtype=float)
        bars = axR.bar(range(len(dmodes)), vals, 0.55, bottom=bottoms,
                       label=labels[cat], color=cat_color[cat],
                       edgecolor="black", linewidth=0.8, zorder=3)
        for b, v, bot in zip(bars, vals, bottoms):
            if v > 0:
                axR.text(b.get_x() + b.get_width() / 2, bot + v / 2,
                         f"{int(v)}", ha="center", va="center",
                         fontsize=11, fontweight="bold", zorder=4)
        bottoms += vals
    axR.set_xticks(range(len(dmodes)))
    axR.set_xticklabels(dmodes, fontsize=10)
    axR.set_ylim(0, 24)
    axR.set_ylabel("number of cases (out of 24)", fontsize=10)
    axR.set_title("Not all divergence is noise: rope produces confident,\n"
                  "repeatable behavioral shifts", fontsize=11, fontweight="bold")
    axR.legend(fontsize=8.5, loc="lower right", framealpha=0.95)
    axR.grid(axis="y", alpha=0.3, lw=0.6, ls="--", zorder=0)
    axR.set_axisbelow(True)

    fig.suptitle("Stage 2 (stability): is the Stage-1 action shift systematic or sampling noise?",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = STABILITY / "figures" / "stability.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[done] {out}")


# ---------------------------------------------------------------------------
# 3 + 4. value-repair (KV recompute substitution) figures
# ---------------------------------------------------------------------------
def fig_value_repair() -> None:
    rows = load_csv(VALUE_REPAIR / "tables" / "value_repair_metrics_rows.csv")
    arms = ["rope", "vrep", "krep", "oracle"]
    arms_by = defaultdict(set)
    for r in rows:
        arms_by[case_key(r)].add(r["mode"])
    matched = {k for k, a in arms_by.items() if set(arms) <= a}

    arm_label = {
        "rope":   "rope\n(skill K + skill V)",
        "vrep":   "vrep\n(skill K + recompute V)\n← substitute VALUE",
        "krep":   "krep\n(recompute K + skill V)\n← substitute KEY",
        "oracle": "recompute-splice\n(recompute K + recompute V)\n≈ upper bound",
    }
    metrics = [
        ("trajectory_match_rate", "Trajectory match"),
        ("tool_set_match_rate",   "Tool-set match"),
        ("modality_match_rate",   "Modality match"),
        ("full_bleu4",            "BLEU-4"),
        ("full_rouge_l",          "ROUGE-L"),
    ]

    # --- aggregate bars ---
    fig, ax = plt.subplots(figsize=(11, 5.4))
    x = np.arange(len(arms))
    bw = 0.15
    offs = np.linspace(-(len(metrics) - 1) / 2, (len(metrics) - 1) / 2, len(metrics)) * bw
    for (col, label), off, color in zip(metrics, offs, COLORS):
        vals = []
        for arm in arms:
            sel = [r for r in rows if r["mode"] == arm and case_key(r) in matched]
            vals.append(mean(fv(r, col) for r in sel))
        bars = ax.bar(x + off, vals, bw * 0.9, label=label, color=color,
                      edgecolor="black", linewidth=0.7, zorder=3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=7.5, rotation=90, zorder=4)
    ax.axhline(1.0, color="black", lw=0.9, ls=(0, (4, 3)), alpha=0.6, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([arm_label[a] for a in arms], fontsize=8.5)
    ax.set_ylim(0, 1.22)
    ax.set_ylabel("Score  (1.0 = matches recompute)", fontsize=11)
    ax.set_title("Stage 3 (KV recompute substitution, 2×2): which half of the KV carries the action gap?\n"
                 "rope→vrep isolates VALUE substitution · rope→krep isolates KEY substitution · 24 cases, 6 tasks",
                 fontsize=11.5, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, lw=0.6, ls="--", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8.5, loc="upper left", ncol=2, framealpha=0.95)
    plt.tight_layout()
    out = VALUE_REPAIR / "figures" / "value_repair_2x2_full.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[done] {out}")

    # --- per-task trajectory heatmap ---
    tasks = ["doc_coauthoring_design_doc", "internal_comms_incident_update",
             "launch_poster_page_pack", "mcp_server_and_spec",
             "slack_launch_pack", "web_artifact_with_theme"]
    heat_label = {
        "rope":   "rope\n(skill K+V)",
        "vrep":   "vrep\n(skill K\n+recompute V)",
        "krep":   "krep\n(recompute K\n+skill V)",
        "oracle": "recompute-\nsplice\n(recompute K+V)",
    }
    mat = np.full((len(tasks), len(arms)), np.nan)
    for ti, t in enumerate(tasks):
        for ai, arm in enumerate(arms):
            sel = [r for r in rows if r["mode"] == arm and r["task"] == t]
            mat[ti, ai] = mean(fv(r, "trajectory_match_rate") for r in sel)

    fig2, ax2 = plt.subplots(figsize=(8, 4.4))
    im = ax2.imshow(mat, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax2.set_xticks(range(len(arms)))
    ax2.set_xticklabels([heat_label[a] for a in arms], fontsize=8.5)
    ax2.set_yticks(range(len(tasks)))
    ax2.set_yticklabels([TASK_ABBREV[t] for t in tasks], fontsize=9)
    for ti in range(len(tasks)):
        for ai in range(len(arms)):
            ax2.text(ai, ti, f"{mat[ti, ai]:.2f}", ha="center", va="center",
                     fontsize=9, fontweight="bold", color="black")
    plt.colorbar(im, ax=ax2, label="trajectory match rate")
    ax2.set_title("Stage 3: trajectory fidelity per task × KV-source condition\n(1.0 = matches recompute)",
                  fontsize=11, fontweight="bold")
    plt.tight_layout()
    out2 = VALUE_REPAIR / "figures" / "value_repair_heatmap.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"[done] {out2}")


def main() -> None:
    fig_headline_summary()
    fig_stability()
    fig_value_repair()


if __name__ == "__main__":
    main()
