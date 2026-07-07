"""Plot logit-divergence results: per-token JSD trajectories + branch-point map.

Reads the analysis outputs produced by run_logit_divergence.py:
  <div_dir>/divergence_summary.csv
  <div_dir>/per_token/<label>.json

Produces, under the research directory:
  fig_a_jsd_trajectory.png   逐 token JSD 轨迹（4 个代表 case，标 </think> 与分叉点）
  fig_b_branch_point_map.png 12 个分叉点的 margin vs JSD 散点（临界翻转 vs 分布改动）
  branch_point_summary.csv   每个 case 的分叉点关键量（供 doc 引用）

用法:
  python plot_logit_divergence.py [--div-dir <dir>] [--out-dir <dir>]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LN2 = math.log(2.0)

DEFAULT_DIV_DIR = Path(
    "/mnt/Large_Language_Model_Lab_1/wsh/Segmentia/output/06_context_free_segment_cache"
    "/cross_occurrence_function_vector/logit_divergence"
)
DEFAULT_OUT_DIR = Path(
    "/home/wsh/openhands_code_research/results/problem_exploration/logit_divergence"
)

# 4 个代表 case：2 个近似平局翻转 + 2 个分布实质改动
TRAJECTORY_CASES = [
    ("inv026--slack_launch_pack--brand-guidelines--occ3", "near-tie flip"),
    ("inv020--doc_coauthoring_design_doc--doc-coauthoring--occ3", "near-tie flip"),
    ("inv019--internal_comms_incident_update--internal-comms--occ3", "distribution shift"),
    ("inv021--web_artifact_with_theme--web-artifacts-builder--occ3", "distribution shift"),
]

# 色板（色盲友好）
C_TRAJ = "#3B6FB6"
C_THINK = "#8A8F98"
C_BRANCH = "#D1495B"
C_CEIL = "#B8B8B8"
C_TIE = "#3B6FB6"
C_SHIFT = "#D1495B"

MARGIN_THRESHOLD = 0.20  # margin < 0.20 归为“近似平局翻转”


def short_label(label: str) -> str:
    # inv019--task--skill--occ3  ->  inv019 / skill
    parts = label.split("--")
    return f"{parts[0]} / {parts[2]}" if len(parts) >= 3 else label


def load_summary_rows(div_dir: Path) -> list[dict]:
    csv_path = div_dir / "divergence_summary.csv"
    with csv_path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_per_token(div_dir: Path, label: str) -> dict:
    with (div_dir / "per_token" / f"{label}.json").open(encoding="utf-8") as f:
        return json.load(f)


def plot_trajectories(div_dir: Path, out_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.2), sharey=True)
    axes = axes.ravel()

    for ax, (label, regime) in zip(axes, TRAJECTORY_CASES):
        data = load_per_token(div_dir, label)
        summ = data["summary"]
        jsd = np.array([e["jsd"] for e in data["per_token_jsd"]])
        x = np.arange(len(jsd))

        ax.plot(x, jsd, color=C_TRAJ, lw=0.9, alpha=0.9)
        ax.axhline(LN2, color=C_CEIL, ls=":", lw=1.0)
        ax.text(len(jsd) * 0.995, LN2 + 0.008, "ln2 (max)", ha="right", va="bottom",
                fontsize=8, color="#666")

        # </think> 边界（取两条路径 think_end 的较小值）
        rc_te, rp_te = summ.get("rc_think_end"), summ.get("rp_think_end")
        te = min([v for v in (rc_te, rp_te) if v is not None], default=None)
        if te is not None:
            ax.axvline(te, color=C_THINK, ls="--", lw=1.1)
            ax.text(te, 0.02, " </think>", rotation=90, va="bottom", ha="left",
                    fontsize=8, color=C_THINK)

        # 分叉点
        bp = summ.get("branch_point") or {}
        bidx = bp.get("index")
        if bidx is not None:
            ax.axvline(bidx, color=C_BRANCH, ls="-", lw=1.2, alpha=0.8)
            ax.plot([bidx], [jsd[bidx]], "o", color=C_BRANCH, ms=6, zorder=5)
            ax.annotate(
                f"branch @ {bidx}\nJSD={bp.get('jsd_at_branch', 0):.3f}",
                xy=(bidx, jsd[bidx]),
                xytext=(bidx + len(jsd) * 0.05, 0.35),
                fontsize=8, color=C_BRANCH,
                arrowprops=dict(arrowstyle="->", color=C_BRANCH, lw=0.8),
            )

        ax.set_title(f"{short_label(label)}   [{regime}]", fontsize=10)
        ax.set_ylim(-0.02, 0.72)
        ax.set_xlabel("token position (decode step)", fontsize=9)
        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel("per-token JSD (recompute vs rope)", fontsize=9)
    axes[2].set_ylabel("per-token JSD (recompute vs rope)", fontsize=9)
    fig.suptitle(
        "Per-token JSD trajectory: identical before the branch, saturates to ln2 after",
        fontsize=12, y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = out_dir / "fig_a_jsd_trajectory.png"
    fig.savefig(out, dpi=160)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return out


def plot_branch_map(rows: list[dict], out_dir: Path) -> tuple[Path, list[dict]]:
    recs = []
    for r in rows:
        if not r["first_div_idx"] or r["first_div_idx"] == "None":
            continue
        rc_prob = float(r["rc_prob"])
        rc_in_rp = float(r["rc_in_rp"])
        margin = abs(rc_prob - rc_in_rp)
        recs.append({
            "label": r["label"],
            "short": short_label(r["label"]),
            "first_div_idx": int(r["first_div_idx"]),
            "jsd_at_branch": float(r["jsd_at_branch"]),
            "rc_prob": rc_prob,
            "rc_in_rp": rc_in_rp,
            "margin": margin,
            "rc_chosen": r["rc_chosen"].strip("'"),
            "rp_chosen": r["rp_chosen"].strip("'"),
            "regime": "near-tie flip" if margin < MARGIN_THRESHOLD else "distribution shift",
        })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.4))

    # 左：margin vs jsd_at_branch 散点
    for reg, color in (("near-tie flip", C_TIE), ("distribution shift", C_SHIFT)):
        pts = [d for d in recs if d["regime"] == reg]
        ax1.scatter([d["margin"] for d in pts], [d["jsd_at_branch"] for d in pts],
                    c=color, s=70, label=reg, edgecolor="white", linewidth=0.6, zorder=3)
    ax1.axvline(MARGIN_THRESHOLD, color="#999", ls="--", lw=1.0)
    ax1.text(MARGIN_THRESHOLD + 0.005, 0.34, f"margin={MARGIN_THRESHOLD}",
             fontsize=8, color="#666", rotation=90, va="top")
    for d in recs:
        ax1.annotate(d["short"], (d["margin"], d["jsd_at_branch"]),
                     fontsize=7, xytext=(4, 3), textcoords="offset points", color="#444")
    ax1.set_xlabel("branch margin  |P_recompute(rc token) − P_rope(rc token)|", fontsize=9)
    ax1.set_ylabel("JSD at branch point", fontsize=9)
    ax1.set_title("Branch point: near-tie flip vs distribution shift", fontsize=11)
    ax1.grid(True, alpha=0.25)
    ax1.legend(fontsize=9, loc="upper left")

    # 右：分叉位置（都在 thinking 早期）
    recs_sorted = sorted(recs, key=lambda d: d["first_div_idx"])
    ys = np.arange(len(recs_sorted))
    colors = [C_TIE if d["regime"] == "near-tie flip" else C_SHIFT for d in recs_sorted]
    ax2.barh(ys, [d["first_div_idx"] for d in recs_sorted], color=colors, alpha=0.85)
    ax2.set_yticks(ys)
    ax2.set_yticklabels([d["short"] for d in recs_sorted], fontsize=7.5)
    ax2.set_xlabel("first divergence token index (all inside <think>)", fontsize=9)
    ax2.set_title("Divergence is seeded early, always in thinking", fontsize=11)
    ax2.grid(True, axis="x", alpha=0.25)
    for y, d in zip(ys, recs_sorted):
        ax2.text(d["first_div_idx"] + 0.5, y, str(d["first_div_idx"]),
                 va="center", fontsize=7.5, color="#333")

    fig.tight_layout()
    out = out_dir / "fig_b_branch_point_map.png"
    fig.savefig(out, dpi=160)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return out, recs


def write_branch_csv(recs: list[dict], out_dir: Path) -> Path:
    out = out_dir / "branch_point_summary.csv"
    cols = ["label", "first_div_idx", "rc_chosen", "rp_chosen",
            "rc_prob", "rc_in_rp", "margin", "jsd_at_branch", "regime"]
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for d in sorted(recs, key=lambda d: d["margin"]):
            w.writerow({k: d[k] for k in cols})
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--div-dir", type=Path, default=DEFAULT_DIV_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_summary_rows(args.div_dir)
    fa = plot_trajectories(args.div_dir, args.out_dir)
    fb, recs = plot_branch_map(rows, args.out_dir)
    csv_out = write_branch_csv(recs, args.out_dir)

    n_tie = sum(1 for d in recs if d["regime"] == "near-tie flip")
    n_shift = len(recs) - n_tie
    print(f"[fig a] {fa}")
    print(f"[fig b] {fb}")
    print(f"[csv ]  {csv_out}")
    print(f"[stat]  {len(recs)} pairs: {n_tie} near-tie flips, {n_shift} distribution shifts")
    print(f"[stat]  first_div_idx: min={min(d['first_div_idx'] for d in recs)}, "
          f"max={max(d['first_div_idx'] for d in recs)}, "
          f"all in thinking (by construction of these cases)")


if __name__ == "__main__":
    main()
