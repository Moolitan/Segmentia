"""三路对比分析：rope KV 效应 vs 模型固有方差基线（temp=0.6 采样档）。

读取两份 divergence_summary CSV（rc1 vs rope、rc1 vs rc2）和原始序列文件，
生成对比图和 CSV。

用法:
  python plot_baseline_comparison.py [--div-dir <dir>] [--seq-dir <dir>] [--out-dir <dir>]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LN2 = math.log(2.0)

DEFAULT_DIV_DIR = Path(
    "/mnt/Large_Language_Model_Lab_1/wsh/Segmentia/output/06_context_free_segment_cache"
    "/cross_occurrence_function_vector/logit_divergence/temp0.6/without_occ12"
)
DEFAULT_SEQ_DIR = Path(
    "/home/wsh/openhands_code_research/results/problem_exploration"
    "/raw_decode_token_sequences/temp0.6/without_occ12"
)
DEFAULT_OUT_DIR = Path(
    "/home/wsh/openhands_code_research/results/problem_exploration"
    "/logit_divergence/temp0.6_baseline_comparison"
)

C_BASELINE = "#5B8C5A"
C_ROPE = "#D1495B"
C_NEUTRAL = "#3B6FB6"


def load_summary(csv_path: Path) -> list[dict]:
    rows = []
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            for k in ("rc_tokens", "rp_tokens", "first_div_idx"):
                r[k] = int(r[k])
            for k in ("think_jsd_mean", "think_jsd_max", "action_jsd_mean",
                       "action_jsd_max", "overall_jsd_mean", "rc_prob",
                       "rp_prob", "rc_in_rp", "rp_in_rc", "jsd_at_branch"):
                r[k] = float(r[k])
            r["div_in_thinking"] = r["div_in_thinking"] == "True"
            rows.append(r)
    return rows


def load_per_token_jsd(json_path: Path) -> list[float]:
    with json_path.open(encoding="utf-8") as f:
        data = json.load(f)
    entries = data.get("per_token_jsd", data) if isinstance(data, dict) else data
    return [float(e.get("jsd", 0.0)) if isinstance(e, dict) else float(e) for e in entries]


def get_action_type(text: str) -> str:
    parts = text.split("</think>", 1)
    action = parts[1] if len(parts) > 1 else text
    return "tool_call" if "<tool_call>" in action else "text"


def short_label(label: str) -> str:
    m = re.match(r"(inv\d+)--(.+?)--(.+?)--occ\d+", label)
    if m:
        return f"{m.group(1)}\n{m.group(3)[:18]}"
    return label[:25]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--div-dir", type=Path, default=DEFAULT_DIV_DIR)
    parser.add_argument("--seq-dir", type=Path, default=DEFAULT_SEQ_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    kv_csv = args.div_dir / "divergence_summary_recompute_run1__vs__rope.csv"
    bl_csv = args.div_dir / "divergence_summary_recompute_run1__vs__recompute_run2.csv"
    kv_rows = load_summary(kv_csv)
    bl_rows = load_summary(bl_csv)

    labels = [r["label"] for r in kv_rows]
    n = len(labels)
    x = np.arange(n)

    # --- Action type comparison ---
    seq_runs = {}
    for run_name in ("recompute_run1", "recompute_run2", "rope"):
        run_dir = args.seq_dir / run_name
        seq_runs[run_name] = {}
        for txt in run_dir.glob("*.txt"):
            content = txt.read_text(encoding="utf-8")
            seq_runs[run_name][txt.stem + ".txt"] = get_action_type(content)

    # Build match arrays
    match_table = []
    for label in labels:
        fname = label + ".txt"
        stem = label
        rc1_t = seq_runs.get("recompute_run1", {}).get(fname, "?")
        rc2_t = seq_runs.get("recompute_run2", {}).get(fname, "?")
        rp_t = seq_runs.get("rope", {}).get(fname, "?")
        if rc1_t == "?":
            for k, v in seq_runs.get("recompute_run1", {}).items():
                if label in k:
                    rc1_t = v; break
        if rc2_t == "?":
            for k, v in seq_runs.get("recompute_run2", {}).items():
                if label in k:
                    rc2_t = v; break
        if rp_t == "?":
            for k, v in seq_runs.get("rope", {}).items():
                if label in k:
                    rp_t = v; break
        match_table.append({
            "label": label,
            "rc1": rc1_t, "rc2": rc2_t, "rope": rp_t,
            "rc1_rc2_match": rc1_t == rc2_t,
            "rc1_rope_match": rc1_t == rp_t,
            "rc2_rope_match": rc2_t == rp_t,
        })

    bl_match = sum(1 for m in match_table if m["rc1_rc2_match"])
    kv_match = sum(1 for m in match_table if m["rc1_rope_match"])
    cr_match = sum(1 for m in match_table if m["rc2_rope_match"])

    # ============================
    # Figure 1: JSD at branch point comparison
    # ============================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel A: JSD at branch point
    ax = axes[0, 0]
    jsd_kv = [r["jsd_at_branch"] for r in kv_rows]
    jsd_bl = [r["jsd_at_branch"] for r in bl_rows]
    bar_w = 0.35
    ax.bar(x - bar_w/2, jsd_bl, bar_w, color=C_BASELINE, alpha=0.85, label="Baseline (rc1 vs rc2)")
    ax.bar(x + bar_w/2, jsd_kv, bar_w, color=C_ROPE, alpha=0.85, label="KV effect (rc1 vs rope)")
    ax.set_xticks(x)
    ax.set_xticklabels([short_label(l) for l in labels], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("JSD at branch point (nat)")
    ax.set_title("A. JSD at Branch Point: Baseline = 0 everywhere")
    ax.legend(fontsize=8)
    ax.axhline(0, color="gray", lw=0.5)

    # Panel B: First divergence index
    ax = axes[0, 1]
    div_kv = [r["first_div_idx"] for r in kv_rows]
    div_bl = [r["first_div_idx"] for r in bl_rows]
    ax.bar(x - bar_w/2, div_bl, bar_w, color=C_BASELINE, alpha=0.85, label="Baseline")
    ax.bar(x + bar_w/2, div_kv, bar_w, color=C_ROPE, alpha=0.85, label="KV effect")
    ax.set_xticks(x)
    ax.set_xticklabels([short_label(l) for l in labels], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("First divergence index (token)")
    ax.set_title("B. First Divergence Position: Baseline always diverges earlier")
    ax.legend(fontsize=8)

    # Panel C: Action type match rate
    ax = axes[1, 0]
    pairs = ["Baseline\n(rc1 vs rc2)", "KV effect\n(rc1 vs rope)", "Cross\n(rc2 vs rope)"]
    rates = [bl_match/n*100, kv_match/n*100, cr_match/n*100]
    colors = [C_BASELINE, C_ROPE, C_NEUTRAL]
    bars = ax.bar(pairs, rates, color=colors, alpha=0.85, width=0.5)
    for bar, rate, cnt in zip(bars, rates, [bl_match, kv_match, cr_match]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{cnt}/{n}\n({rate:.0f}%)", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.set_ylabel("Action type match (%)")
    ax.set_title("C. Action Type Match: Rope = Baseline")
    ax.axhline(75, color="gray", ls="--", lw=0.8, alpha=0.5)

    # Panel D: Overall JSD mean
    ax = axes[1, 1]
    ojsd_kv = [r["overall_jsd_mean"] for r in kv_rows]
    ojsd_bl = [r["overall_jsd_mean"] for r in bl_rows]
    ax.bar(x - bar_w/2, ojsd_bl, bar_w, color=C_BASELINE, alpha=0.85, label="Baseline")
    ax.bar(x + bar_w/2, ojsd_kv, bar_w, color=C_ROPE, alpha=0.85, label="KV effect")
    ax.set_xticks(x)
    ax.set_xticklabels([short_label(l) for l in labels], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Overall JSD mean (nat)")
    ax.axhline(LN2, color="gray", ls=":", lw=0.8, label=f"ln2 = {LN2:.3f}")
    ax.set_title("D. Overall JSD Mean: Both pairs saturate to ln2")
    ax.legend(fontsize=8)

    fig.suptitle(
        "Rope KV Reuse vs Model Inherent Variance (temp=0.6, Qwen3-14B)\n"
        "Conclusion: Action divergence is model sampling variance, NOT rope damage",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(args.out_dir / f"fig_baseline_comparison.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ============================
    # Figure 2: Per-case action type heatmap
    # ============================
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    type_map = {"tool_call": 1, "text": 0}
    data = np.array([[type_map.get(m["rc1"], -1), type_map.get(m["rc2"], -1),
                       type_map.get(m["rope"], -1)] for m in match_table])
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["#4A90D9", "#E8A838"])
    im = ax2.imshow(data, cmap=cmap, aspect="auto", interpolation="nearest")
    ax2.set_xticks([0, 1, 2])
    ax2.set_xticklabels(["recompute_run1\n(seed=1111)", "recompute_run2\n(seed=2222)",
                          "rope\n(seed=1111)"], fontsize=10)
    ax2.set_yticks(range(n))
    ax2.set_yticklabels([short_label(l).replace("\n", " / ") for l in labels], fontsize=8)
    for i in range(n):
        for j in range(3):
            val = "TC" if data[i, j] == 1 else "TXT"
            ax2.text(j, i, val, ha="center", va="center", fontsize=8,
                     color="white" if data[i, j] == 1 else "black", fontweight="bold")
    ax2.set_title("Action Type per Run (blue=tool_call, orange=text)\n"
                  "No systematic pattern — rope mismatch ≈ baseline mismatch",
                  fontsize=11, fontweight="bold")
    fig2.tight_layout()
    for ext in ("png", "pdf"):
        fig2.savefig(args.out_dir / f"fig_action_type_heatmap.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig2)

    # ============================
    # Figure 3: JSD trajectory comparison (pick 2 cases)
    # ============================
    kv_per_token_dir = args.div_dir / "per_token_recompute_run1__vs__rope"
    bl_per_token_dir = args.div_dir / "per_token_recompute_run1__vs__recompute_run2"

    example_cases = [labels[0], labels[2]]
    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))
    for idx_c, case_label in enumerate(example_cases):
        ax = axes3[idx_c]
        kv_json = kv_per_token_dir / f"{case_label}.json"
        bl_json = bl_per_token_dir / f"{case_label}.json"
        if kv_json.exists() and bl_json.exists():
            jsd_kv_traj = load_per_token_jsd(kv_json)
            jsd_bl_traj = load_per_token_jsd(bl_json)
            min_len = min(len(jsd_kv_traj), len(jsd_bl_traj), 500)
            ax.plot(range(min_len), jsd_bl_traj[:min_len], color=C_BASELINE,
                    alpha=0.7, lw=0.8, label="Baseline (rc1 vs rc2)")
            ax.plot(range(min_len), jsd_kv_traj[:min_len], color=C_ROPE,
                    alpha=0.7, lw=0.8, label="KV effect (rc1 vs rope)")
            ax.axhline(LN2, color="gray", ls=":", lw=0.5)
            ax.set_xlabel("Token index")
            ax.set_ylabel("JSD (nat)")
            ax.set_title(short_label(case_label).replace("\n", " / "), fontsize=10)
            ax.legend(fontsize=8)
    fig3.suptitle("JSD Trajectory: Both pairs saturate to ln2 after divergence",
                  fontsize=12, fontweight="bold")
    fig3.tight_layout()
    for ext in ("png", "pdf"):
        fig3.savefig(args.out_dir / f"fig_jsd_trajectory_comparison.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig3)

    # ============================
    # Summary CSV
    # ============================
    csv_path = args.out_dir / "baseline_comparison_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "label", "rc1_type", "rc2_type", "rope_type",
            "rc1_rc2_type_match", "rc1_rope_type_match",
            "baseline_first_div", "kv_first_div",
            "baseline_jsd_at_branch", "kv_jsd_at_branch",
            "baseline_overall_jsd_mean", "kv_overall_jsd_mean",
            "baseline_branch_tokens", "kv_branch_tokens",
        ])
        for i, label in enumerate(labels):
            writer.writerow([
                label,
                match_table[i]["rc1"], match_table[i]["rc2"], match_table[i]["rope"],
                match_table[i]["rc1_rc2_match"], match_table[i]["rc1_rope_match"],
                bl_rows[i]["first_div_idx"], kv_rows[i]["first_div_idx"],
                bl_rows[i]["jsd_at_branch"], kv_rows[i]["jsd_at_branch"],
                bl_rows[i]["overall_jsd_mean"], kv_rows[i]["overall_jsd_mean"],
                f'{bl_rows[i]["rc_chosen"]}/{bl_rows[i]["rp_chosen"]}',
                f'{kv_rows[i]["rc_chosen"]}/{kv_rows[i]["rp_chosen"]}',
            ])

    print(f"\n=== Summary ===")
    print(f"Action type match rate:")
    print(f"  Baseline (rc1 vs rc2):  {bl_match}/{n} = {bl_match/n*100:.1f}%")
    print(f"  KV effect (rc1 vs rope): {kv_match}/{n} = {kv_match/n*100:.1f}%")
    print(f"  Cross (rc2 vs rope):     {cr_match}/{n} = {cr_match/n*100:.1f}%")
    print(f"\nFirst divergence index (mean):")
    print(f"  Baseline: {np.mean(div_bl):.1f}  (median {np.median(div_bl):.0f})")
    print(f"  KV effect: {np.mean(div_kv):.1f}  (median {np.median(div_kv):.0f})")
    print(f"\nJSD at branch point (mean):")
    print(f"  Baseline: {np.mean(jsd_bl):.4f}")
    print(f"  KV effect: {np.mean(jsd_kv):.4f}")
    print(f"\nOverall JSD mean:")
    print(f"  Baseline: {np.mean(ojsd_bl):.4f}")
    print(f"  KV effect: {np.mean(ojsd_kv):.4f}")
    print(f"\nOutputs: {args.out_dir}")


if __name__ == "__main__":
    main()
