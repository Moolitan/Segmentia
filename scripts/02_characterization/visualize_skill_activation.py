"""
可视化 Skill 激活模拟结果。

读取 results/characterization/skill_activation_stats.json，
生成论文级图表保存到 results/characterization/figures/。

用法：
    python scripts/02_characterization/visualize_skill_activation.py
    python scripts/02_characterization/visualize_skill_activation.py --input results/characterization/skill_activation_stats.json
"""

import argparse
import json
import math
import os
import sys
from collections import Counter
from itertools import combinations

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

# 论文风格：清爽、无多余装饰
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

# 配色：来自 Tableau 10
COLORS = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
    "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac",
    "#5778a4", "#e49444",
]


def load_data(path: str) -> dict:
    with open(path) as f:
        return json.load(f)

def _is_swebench_real_run(data: dict) -> bool:
    return data.get("metadata", {}).get("source") == "swebench_real_run"


# ---------------------------------------------------------------------------
# Fig 1: Skill 激活频率（水平条形图）
# ---------------------------------------------------------------------------

def plot_skill_frequency(data: dict, out_dir: str) -> None:
    freq = data["aggregate"]["skill_frequency"]
    names = sorted(freq.keys(), key=lambda k: freq[k]["count"])
    counts = [freq[n]["count"] for n in names]
    pcts = [freq[n]["pct"] for n in names]
    total = data["metadata"]["total_instances"]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.barh(names, counts, color=COLORS[0], edgecolor="white", height=0.65)

    for bar, pct in zip(bars, pcts):
        w = bar.get_width()
        ax.text(w + total * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{pct:.1f}%", va="center", fontsize=9, color="#333")

    ax.set_xlabel("Number of Instances Triggered")
    ax.set_title("Skill Activation Frequency across SWE-Bench Instances")
    ax.set_xlim(0, max(counts) * 1.15)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig1_skill_frequency.pdf"))
    fig.savefig(os.path.join(out_dir, "fig1_skill_frequency.png"))
    plt.close(fig)
    print("  fig1_skill_frequency.pdf")


# ---------------------------------------------------------------------------
# Fig 2: 每实例激活 Skill 数分布（柱形图）
# ---------------------------------------------------------------------------

def plot_activation_distribution(data: dict, out_dir: str) -> None:
    dist = data["aggregate"]["activation_count_distribution"]
    keys = sorted(int(k) for k in dist.keys())
    vals = [dist[str(k)] for k in keys]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(keys, vals, color=COLORS[1], edgecolor="white", width=0.7)

    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + max(vals) * 0.01,
                    str(int(h)), ha="center", va="bottom", fontsize=9, color="#333")

    ax.set_xlabel("Number of Skills Activated per Instance")
    ax.set_ylabel("Number of Instances")
    ax.set_title("Distribution of Skill Activation Count")
    ax.set_xticks(keys)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig2_activation_distribution.pdf"))
    fig.savefig(os.path.join(out_dir, "fig2_activation_distribution.png"))
    plt.close(fig)
    print("  fig2_activation_distribution.pdf")


# ---------------------------------------------------------------------------
# Fig 3: <EXTRA_INFO> Token 分布（直方图 + 箱线图组合）
# ---------------------------------------------------------------------------

def plot_token_distribution(data: dict, out_dir: str) -> None:
    extra_tokens = [inst["extra_info_tokens"] for inst in data["per_instance"]]
    issue_tokens = [inst["issue_text_tokens"] for inst in data["per_instance"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [3, 1]})

    # 直方图
    n_bins = min(30, max(10, len(extra_tokens) // 10))
    ax1.hist(extra_tokens, bins=n_bins, color=COLORS[0], alpha=0.85, edgecolor="white", label="EXTRA_INFO tokens")
    ax1.hist(issue_tokens, bins=n_bins, color=COLORS[1], alpha=0.7, edgecolor="white", label="Issue text tokens")
    ax1.set_xlabel("Token Count")
    ax1.set_ylabel("Number of Instances")
    ax1.set_title("Token Distribution: Skill Injection vs Issue Text")
    ax1.legend()

    # 箱线图
    bp = ax2.boxplot(
        [extra_tokens, issue_tokens],
        tick_labels=["EXTRA_INFO", "Issue"],
        patch_artist=True,
        widths=0.5,
        medianprops={"color": "#333", "linewidth": 1.5},
    )
    bp["boxes"][0].set_facecolor(COLORS[0])
    bp["boxes"][0].set_alpha(0.7)
    bp["boxes"][1].set_facecolor(COLORS[1])
    bp["boxes"][1].set_alpha(0.7)
    ax2.set_ylabel("Token Count")
    ax2.set_title("Token Box Plot")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig3_token_distribution.pdf"))
    fig.savefig(os.path.join(out_dir, "fig3_token_distribution.png"))
    plt.close(fig)
    print("  fig3_token_distribution.pdf")


# ---------------------------------------------------------------------------
# Fig 4: Skill 共现热力图
# ---------------------------------------------------------------------------

def plot_co_occurrence(data: dict, out_dir: str) -> None:
    skill_names = data["metadata"]["skill_names"]
    co = data["aggregate"]["co_occurrence_matrix"]
    n = len(skill_names)
    matrix = np.zeros((n, n), dtype=int)

    # 对角线 = 自身激活次数
    freq = data["aggregate"]["skill_frequency"]
    for i, name in enumerate(skill_names):
        matrix[i][i] = freq[name]["count"]
        for j, other in enumerate(skill_names):
            if i != j and other in co.get(name, {}):
                matrix[i][j] = co[name][other]

    # 短名称
    short_names = [n.replace("-", "\n") if len(n) > 10 else n for n in skill_names]

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(short_names, fontsize=8)

    # 数值标注
    for i in range(n):
        for j in range(n):
            val = matrix[i][j]
            if val > 0:
                color = "white" if val > matrix.max() * 0.6 else "#333"
                ax.text(j, i, str(val), ha="center", va="center", fontsize=7, color=color)

    ax.set_title("Skill Co-occurrence Matrix")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Count")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig4_co_occurrence.pdf"))
    fig.savefig(os.path.join(out_dir, "fig4_co_occurrence.png"))
    plt.close(fig)
    print("  fig4_co_occurrence.pdf")


# ---------------------------------------------------------------------------
# Fig 5: Top Skill 组合（水平条形图）
# ---------------------------------------------------------------------------

def plot_top_combinations(data: dict, out_dir: str) -> None:
    combos = data["aggregate"]["skill_set_frequency"][:15]
    if not combos:
        return

    labels = []
    counts = []
    for c in reversed(combos):
        skills = c["skill_set"]
        label = ", ".join(skills) if skills else "(none)"
        # 截断过长的标签
        if len(label) > 60:
            label = label[:57] + "..."
        labels.append(label)
        counts.append(c["count"])

    fig, ax = plt.subplots(figsize=(9, max(4, len(labels) * 0.35)))
    bars = ax.barh(range(len(labels)), counts, color=COLORS[3], edgecolor="white", height=0.65)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)

    for bar in bars:
        w = bar.get_width()
        ax.text(w + max(counts) * 0.02, bar.get_y() + bar.get_height() / 2,
                str(int(w)), va="center", fontsize=9, color="#333")

    ax.set_xlabel("Number of Instances")
    ax.set_title("Top 15 Most Common Skill Combinations")
    ax.set_xlim(0, max(counts) * 1.2)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig5_top_combinations.pdf"))
    fig.savefig(os.path.join(out_dir, "fig5_top_combinations.png"))
    plt.close(fig)
    print("  fig5_top_combinations.pdf")


# ---------------------------------------------------------------------------
# Fig 6: 跨请求 Jaccard 距离分布（CDF）
# ---------------------------------------------------------------------------

def plot_jaccard_cdf(data: dict, out_dir: str) -> None:
    per_instance = data["per_instance"]
    n = len(per_instance)
    if n < 2:
        print("  (跳过 fig6: 实例数不足)")
        return

    skill_sets = [set(inst["activated_skills"]) for inst in per_instance]

    # 计算所有 pair 的 Jaccard 距离
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            a, b = skill_sets[i], skill_sets[j]
            if not a and not b:
                dists.append(0.0)
            else:
                dists.append(1.0 - len(a & b) / len(a | b))

    dists_sorted = np.sort(dists)
    cdf = np.arange(1, len(dists_sorted) + 1) / len(dists_sorted)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(dists_sorted, cdf, color=COLORS[2], linewidth=2)
    ax.axvline(x=np.mean(dists), color=COLORS[0], linestyle="--", linewidth=1,
               label=f"Mean = {np.mean(dists):.3f}")
    ax.axvline(x=np.median(dists), color=COLORS[1], linestyle=":", linewidth=1,
               label=f"Median = {np.median(dists):.3f}")

    ax.set_xlabel("Jaccard Distance between Instance Pairs")
    ax.set_ylabel("CDF")
    ax.set_title("Cross-Request Skill Set Diversity (Jaccard Distance)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig6_jaccard_cdf.pdf"))
    fig.savefig(os.path.join(out_dir, "fig6_jaccard_cdf.png"))
    plt.close(fig)
    print("  fig6_jaccard_cdf.pdf")


# ---------------------------------------------------------------------------
# Fig 7: Dynamic Ratio（动态 token 占比分布）
# ---------------------------------------------------------------------------

def plot_dynamic_ratio(data: dict, out_dir: str) -> None:
    """每个实例的 extra_info_tokens / (extra_info_tokens + issue_text_tokens) 分布。"""
    ratios = []
    for inst in data["per_instance"]:
        extra = inst["extra_info_tokens"]
        issue = inst["issue_text_tokens"]
        total = extra + issue
        ratios.append(extra / total if total > 0 else 0.0)

    fig, ax = plt.subplots(figsize=(6, 4))

    # 直方图
    n_bins = min(25, max(8, len(ratios) // 10))
    ax.hist(ratios, bins=n_bins, color=COLORS[4], alpha=0.85, edgecolor="white")
    ax.axvline(x=np.mean(ratios), color=COLORS[2], linestyle="--", linewidth=1.5,
               label=f"Mean = {np.mean(ratios):.2f}")
    ax.axvline(x=np.median(ratios), color=COLORS[0], linestyle=":", linewidth=1.5,
               label=f"Median = {np.median(ratios):.2f}")

    ax.set_xlabel("Dynamic Ratio (EXTRA_INFO / Total Tokens)")
    ax.set_ylabel("Number of Instances")
    ax.set_title("Skill Injection Token Overhead Ratio")
    ax.set_xlim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig7_dynamic_ratio.pdf"))
    fig.savefig(os.path.join(out_dir, "fig7_dynamic_ratio.png"))
    plt.close(fig)
    print("  fig7_dynamic_ratio.pdf")


# ---------------------------------------------------------------------------
# Fig R1: SWE-Bench 真实运行 —— 是否触发 Skills（柱形图）
# ---------------------------------------------------------------------------

def plot_real_run_any_activation(data: dict, out_dir: str) -> None:
    agg = data["aggregate"]
    any_cnt = agg["any_skill_activation"]["count"]
    none_cnt = agg["no_skill_activation"]["count"]
    any_pct = agg["any_skill_activation"]["pct"]
    n = data["metadata"]["total_instances"]
    model = data["metadata"].get("model", "")

    fig, ax = plt.subplots(figsize=(6.2, 4))
    labels = ["Any skill\nactivated", "No skill\nactivated"]
    counts = [any_cnt, none_cnt]
    colors = [COLORS[2], COLORS[0]]
    bars = ax.bar(labels, counts, color=colors, edgecolor="white", width=0.65)

    ymax = max(counts) if counts else 1
    for bar, c in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            c + ymax * 0.03,
            str(int(c)),
            ha="center",
            va="bottom",
            fontsize=10,
            color="#333",
        )

    ax.set_ylabel("Number of Instances")
    ax.set_title(f"SWE-Bench Real Run: Skill Activation Rate\n(model={model}, n={n}, any={any_pct:.1f}%)")
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "figR1_real_run_any_activation.pdf"))
    fig.savefig(os.path.join(out_dir, "figR1_real_run_any_activation.png"))
    plt.close(fig)
    print("  figR1_real_run_any_activation.pdf")


# ---------------------------------------------------------------------------
# Fig R2: 激活 vs 使用 对比（分组柱形图）
# ---------------------------------------------------------------------------

def plot_activation_vs_usage(data: dict, out_dir: str) -> None:
    """对比：关键词激活(keyword trigger) vs 实际使用(agent读取SKILL.md)。"""
    agg = data["aggregate"]
    n = data["metadata"]["total_instances"]
    model = data["metadata"].get("model", "")

    act_any = agg["any_skill_activation"]["count"]
    act_none = agg["no_skill_activation"]["count"]

    # 兼容没有 usage 数据的旧格式
    if "any_skill_usage" not in agg:
        print("  (跳过 figR2: 无 skill usage 数据)")
        return

    use_any = agg["any_skill_usage"]["count"]
    use_none = agg["no_skill_usage"]["count"]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    x = np.arange(2)
    width = 0.32
    bars1 = ax.bar(x - width / 2, [act_any, act_none], width,
                   label="Activated (keyword trigger)", color=COLORS[1], edgecolor="white")
    bars2 = ax.bar(x + width / 2, [use_any, use_none], width,
                   label="Used (agent reads SKILL.md)", color=COLORS[2], edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(["Any skill", "No skill"])
    ax.set_ylabel("Number of Instances")
    ax.set_title(f"Skill Activation vs Actual Usage\n(model={model}, n={n})")
    ax.legend()
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    ymax = max(act_any, act_none, use_any, use_none) if n else 1
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + ymax * 0.02,
                        str(int(h)), ha="center", va="bottom", fontsize=9, color="#333")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "figR2_activation_vs_usage.pdf"))
    fig.savefig(os.path.join(out_dir, "figR2_activation_vs_usage.png"))
    plt.close(fig)
    print("  figR2_activation_vs_usage.pdf")


# ---------------------------------------------------------------------------
# Fig R3: 每个 Skill 的激活 vs 使用 频率对比
# ---------------------------------------------------------------------------

def plot_skill_frequency_comparison(data: dict, out_dir: str) -> None:
    """每个 skill 的激活次数 vs 使用次数并排对比。"""
    agg = data["aggregate"]
    act_freq = agg["skill_frequency"]

    if "skill_usage_frequency" not in agg:
        print("  (跳过 figR3: 无 skill usage 数据)")
        return

    use_freq = agg["skill_usage_frequency"]
    names = sorted(act_freq.keys(), key=lambda k: act_freq[k]["count"], reverse=False)
    act_counts = [act_freq[n]["count"] for n in names]
    use_counts = [use_freq.get(n, {}).get("count", 0) for n in names]

    fig, ax = plt.subplots(figsize=(8, 5.5))

    y = np.arange(len(names))
    height = 0.35
    bars1 = ax.barh(y - height / 2, act_counts, height,
                    label="Activated (keyword trigger)", color=COLORS[1], edgecolor="white")
    bars2 = ax.barh(y + height / 2, use_counts, height,
                    label="Used (agent reads SKILL.md)", color=COLORS[2], edgecolor="white")

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Number of Instances")
    ax.set_title("Per-Skill: Activation vs Actual Usage")
    ax.legend(loc="lower right")

    xmax = max(max(act_counts, default=0), max(use_counts, default=0))
    ax.set_xlim(0, xmax * 1.2 if xmax > 0 else 1)

    for bars in [bars1, bars2]:
        for bar in bars:
            w = bar.get_width()
            if w > 0:
                ax.text(w + xmax * 0.01, bar.get_y() + bar.get_height() / 2,
                        str(int(w)), va="center", fontsize=8, color="#333")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "figR3_skill_freq_comparison.pdf"))
    fig.savefig(os.path.join(out_dir, "figR3_skill_freq_comparison.png"))
    plt.close(fig)
    print("  figR3_skill_freq_comparison.pdf")


# ---------------------------------------------------------------------------
# Fig R4: 使用漏斗图（激活→使用的转化率）
# ---------------------------------------------------------------------------

def plot_usage_funnel(data: dict, out_dir: str) -> None:
    """漏斗图：available → activated → used。"""
    agg = data["aggregate"]
    n = data["metadata"]["total_instances"]
    model = data["metadata"].get("model", "")

    if "any_skill_usage" not in agg:
        print("  (跳过 figR4: 无 skill usage 数据)")
        return

    available = n  # 所有实例都有 skills 可用
    activated = agg["any_skill_activation"]["count"]
    used = agg["any_skill_usage"]["count"]

    fig, ax = plt.subplots(figsize=(6, 4))
    stages = ["Skills\nAvailable", "Skills\nActivated\n(keyword)", "Skills\nActually Used\n(read SKILL.md)"]
    values = [available, activated, used]
    colors = [COLORS[0], COLORS[1], COLORS[2]]

    bars = ax.bar(stages, values, color=colors, edgecolor="white", width=0.6)

    for bar, v in zip(bars, values):
        pct = v / n * 100 if n else 0
        ax.text(bar.get_x() + bar.get_width() / 2, v + max(values) * 0.02,
                f"{v}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=10, color="#333")

    ax.set_ylabel("Number of Instances")
    ax.set_title(f"Skill Usage Funnel\n(model={model}, n={n})")
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.set_ylim(0, max(values) * 1.2)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "figR4_usage_funnel.pdf"))
    fig.savefig(os.path.join(out_dir, "figR4_usage_funnel.png"))
    plt.close(fig)
    print("  figR4_usage_funnel.pdf")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input",
        default="results/characterization/skill_activation_stats.json",
        help="输入 JSON 路径",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="图表输出目录（默认与输入同目录下的 figures/）",
    )
    args = parser.parse_args()

    data = load_data(args.input)
    out_dir = args.outdir or os.path.join(os.path.dirname(args.input), "figures")
    os.makedirs(out_dir, exist_ok=True)

    n = data["metadata"]["total_instances"]
    k = data["metadata"]["total_skills"]
    print(f"数据: {n} 个实例, {k} 个 Skills")
    print(f"输出目录: {out_dir}\n")

    print("生成图表:")
    if _is_swebench_real_run(data):
        # 真实运行数据：激活 + 使用相关的图
        plot_real_run_any_activation(data, out_dir)
        plot_skill_frequency(data, out_dir)
        plot_activation_distribution(data, out_dir)
        # 激活 vs 实际使用对比图
        plot_activation_vs_usage(data, out_dir)
        plot_skill_frequency_comparison(data, out_dir)
        plot_usage_funnel(data, out_dir)
    else:
        # 离线模拟数据：画完整 Figure 1-7
        plot_skill_frequency(data, out_dir)
        plot_activation_distribution(data, out_dir)
        plot_token_distribution(data, out_dir)
        plot_co_occurrence(data, out_dir)
        plot_top_combinations(data, out_dir)
        plot_jaccard_cdf(data, out_dir)
        plot_dynamic_ratio(data, out_dir)

    print(f"\n完成！图表保存至 {out_dir}/")


if __name__ == "__main__":
    main()
