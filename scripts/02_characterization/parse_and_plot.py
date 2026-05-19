"""
读取 run_web_skills_test.py 输出的 JSON，生成 prefix cache 特征化图表。

用法：
    python scripts/02_web/parse_and_plot.py [--input PATH] [--outdir DIR]
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# ---------------------------------------------------------------------------
# 配色 & 样式
# ---------------------------------------------------------------------------

COLORS = {
    "input": "#2563EB",
    "output": "#F59E0B",
    "cache_hit": "#10B981",
    "cache_miss": "#EF4444",
    "ideal": "#8B5CF6",
    "wasted": "#FB923C",
    "grid": "#E5E7EB",
    "bg": "#FAFAFA",
    "sys_prompt": "#6366F1",
    "conv_history": "#F97316",
}


def setup_style():
    plt.rcParams.update({
        "figure.facecolor": COLORS["bg"],
        "axes.facecolor": "#FFFFFF",
        "axes.grid": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": COLORS["grid"],
        "grid.linewidth": 0.5,
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
    })


def save_fig(fig, outdir: str, name: str):
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"{name}.png"), dpi=200)
    fig.savefig(os.path.join(outdir, f"{name}.pdf"))
    plt.close(fig)
    print(f"  -> {name}.png/pdf")


# ---------------------------------------------------------------------------
# Figure 1: Per-Turn Token Usage
# ---------------------------------------------------------------------------

def plot_per_turn_tokens(turns: list[dict], outdir: str):
    """柱状图：每轮的 prompt / completion / cache_read tokens。"""
    fig, ax = plt.subplots(figsize=(10, 5))
    n = len(turns)
    x = np.arange(n)

    prompt = [t["prompt_tokens"] for t in turns]
    completion = [t["completion_tokens"] for t in turns]
    cache_read = [t["cache_read_tokens"] for t in turns]

    w = 0.25
    ax.bar(x - w, prompt, width=w, color=COLORS["input"], label="Prompt tokens", edgecolor="white")
    ax.bar(x, completion, width=w, color=COLORS["output"], label="Completion tokens", edgecolor="white")
    ax.bar(x + w, cache_read, width=w, color=COLORS["cache_hit"], label="Cache read tokens", edgecolor="white")

    for i, v in enumerate(prompt):
        ax.text(i - w, v + max(prompt) * 0.01, f"{v/1000:.1f}K", ha="center", va="bottom",
                fontsize=8, color=COLORS["input"], fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"Turn {t['turn_number']}" for t in turns])
    ax.set_ylabel("Tokens")
    ax.set_title("Per-Turn Token Usage (Repeated Identical Messages)")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v/1000:.0f}K"))
    ax.legend(loc="upper left", framealpha=0.9)

    save_fig(fig, outdir, "fig1_per_turn_tokens")


# ---------------------------------------------------------------------------
# Figure 2: Cumulative Prompt Tokens & Cache Efficiency
# ---------------------------------------------------------------------------

def plot_cumulative_and_cache(turns: list[dict], outdir: str):
    """双图：左 = 累计 prompt tokens，右 = cache hit rate。"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    n = len(turns)
    x = np.arange(1, n + 1)

    cumulative = [t["cumulative_prompt_tokens"] for t in turns]
    cache_rates = [t["vllm_prefix_cache_hit_rate"] * 100 if t["vllm_prefix_cache_hit_rate"] >= 0
                   else 0.0 for t in turns]

    # 理想曲线：如果 prefix 完全复用，每轮只新增 completion + 新 user message 的 prompt
    # 第一轮全部是新的，之后复用前一轮的全部 prompt 作为前缀
    prompt_per_turn = [t["prompt_tokens"] for t in turns]
    ideal_cumulative = []
    for i in range(n):
        if i == 0:
            ideal_cumulative.append(prompt_per_turn[0])
        else:
            # 理想情况：只有新增部分需要 prefill
            new_only = prompt_per_turn[i] - prompt_per_turn[i - 1] if prompt_per_turn[i] > prompt_per_turn[i - 1] else prompt_per_turn[i] * 0.1
            ideal_cumulative.append(ideal_cumulative[-1] + new_only)

    # --- 左图：累计 prompt tokens ---
    ax1.plot(x, cumulative, "o-", color=COLORS["input"], linewidth=2, markersize=6,
             label="Actual cumulative")
    ax1.plot(x, ideal_cumulative, "s--", color=COLORS["ideal"], linewidth=2, markersize=6,
             label="Ideal (full prefix reuse)")
    ax1.fill_between(x, ideal_cumulative, cumulative, alpha=0.15, color=COLORS["wasted"],
                     label="Wasted prefill")

    ax1.set_xlabel("Turn")
    ax1.set_ylabel("Cumulative Prompt Tokens")
    ax1.set_title("Cumulative Prefill Cost")
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v/1000:.0f}K"))
    ax1.legend(loc="upper left", framealpha=0.9)
    ax1.set_xticks(x)

    # --- 右图：Cache hit rate ---
    bars = ax2.bar(x, cache_rates, width=0.5, color=COLORS["cache_miss"], edgecolor="white")

    # 理想 hit rate
    ideal_hit = []
    for i in range(n):
        if i == 0 or prompt_per_turn[i] == 0:
            ideal_hit.append(0)
        else:
            ideal_hit.append(min(prompt_per_turn[i - 1] / prompt_per_turn[i] * 100, 100))

    ax2.bar(x + 0.35, ideal_hit, width=0.3, color=COLORS["ideal"], alpha=0.6,
            edgecolor="white", label="Ideal hit rate")

    for i, v in enumerate(ideal_hit):
        if v > 0:
            ax2.text(x[i] + 0.35, v + 1, f"{v:.0f}%", ha="center", va="bottom",
                     fontsize=8, color=COLORS["ideal"], fontweight="bold")

    ax2.set_xlabel("Turn")
    ax2.set_ylabel("Cache Hit Rate (%)")
    ax2.set_title("vLLM Prefix Cache Hit Rate")
    ax2.set_ylim(0, 105)
    ax2.set_xticks(x)
    ax2.legend(["Actual", "Ideal"], loc="upper left", framealpha=0.9)

    save_fig(fig, outdir, "fig2_cumulative_and_cache")


# ---------------------------------------------------------------------------
# Figure 3: Prefix Reuse Potential (堆叠柱状图)
# ---------------------------------------------------------------------------

def plot_prefix_reuse(turns: list[dict], outdir: str):
    """每轮 prompt tokens 中可复用前缀 vs 新增部分。"""
    fig, ax = plt.subplots(figsize=(10, 5))

    n = len(turns)
    x = np.arange(1, n + 1)
    prompt = [t["prompt_tokens"] for t in turns]

    reusable = []
    new_tokens = []
    for i in range(n):
        if i == 0:
            reusable.append(0)
            new_tokens.append(prompt[0])
        else:
            r = min(prompt[i - 1], prompt[i])
            reusable.append(r)
            new_tokens.append(prompt[i] - r)

    ax.bar(x, reusable, width=0.6, color=COLORS["cache_hit"],
           label="Reusable prefix (wasted without cache)", edgecolor="white")
    ax.bar(x, new_tokens, width=0.6, bottom=reusable, color=COLORS["input"],
           label="New tokens (must compute)", edgecolor="white")

    for i in range(n):
        if prompt[i] > 0 and reusable[i] > 0:
            pct = reusable[i] / prompt[i] * 100
            ax.text(x[i], reusable[i] / 2, f"{pct:.0f}%", ha="center", va="center",
                    fontsize=10, fontweight="bold", color="white")

    ax.set_xlabel("Turn")
    ax.set_ylabel("Tokens")
    ax.set_title("Per-Turn Prefix Reuse Potential (Actual Cache Hit = 0%)")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v/1000:.0f}K"))
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_xticks(x)

    total_reusable = sum(reusable)
    total_prompt = sum(prompt)
    ax.annotate(
        f"Total reusable: {total_reusable/1000:.0f}K / {total_prompt/1000:.0f}K "
        f"({total_reusable/max(1,total_prompt)*100:.1f}%)",
        xy=(0.98, 0.95), xycoords="axes fraction", ha="right", va="top",
        fontsize=10, color=COLORS["cache_miss"],
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#FEE2E2",
                  edgecolor=COLORS["cache_miss"], alpha=0.8),
    )

    save_fig(fig, outdir, "fig3_prefix_reuse")


# ---------------------------------------------------------------------------
# Figure 4: LLM 调用次数 & 延迟
# ---------------------------------------------------------------------------

def plot_calls_and_latency(turns: list[dict], outdir: str):
    """双轴图：每轮 LLM 调用次数 + 轮执行时间。"""
    fig, ax1 = plt.subplots(figsize=(10, 5))

    n = len(turns)
    x = np.arange(1, n + 1)
    calls = [t["num_llm_calls"] for t in turns]
    elapsed = [t["elapsed_seconds"] for t in turns]

    color1 = COLORS["input"]
    color2 = COLORS["wasted"]

    bars = ax1.bar(x, calls, width=0.5, color=color1, alpha=0.7, label="LLM calls", edgecolor="white")
    ax1.set_xlabel("Turn")
    ax1.set_ylabel("# LLM Calls", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_xticks(x)

    for bar, v in zip(bars, calls):
        ax1.text(bar.get_x() + bar.get_width() / 2, v + 0.1, str(v),
                 ha="center", va="bottom", fontsize=9, color=color1, fontweight="bold")

    ax2 = ax1.twinx()
    ax2.plot(x, elapsed, "o-", color=color2, linewidth=2, markersize=7, label="Elapsed time")
    ax2.set_ylabel("Elapsed (seconds)", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    for i, v in enumerate(elapsed):
        ax2.text(x[i], v + max(elapsed) * 0.02, f"{v:.0f}s", ha="center", va="bottom",
                 fontsize=8, color=color2)

    ax1.set_title("Per-Turn LLM Calls and Execution Time")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", framealpha=0.9)

    save_fig(fig, outdir, "fig4_calls_and_latency")


# ---------------------------------------------------------------------------
# 汇总表
# ---------------------------------------------------------------------------

def print_summary(data: dict):
    turns = data["turns"]
    agg = data["aggregate"]

    print(f"\n{'=' * 72}")
    print(f"实验: {data['metadata']['experiment']}")
    print(f"轮数: {data['metadata']['num_turns']}, Skills: {data['metadata']['skill_names']}")
    print(f"{'=' * 72}")
    print(f"{'Turn':>5}  {'Prompt':>8}  {'Compl':>8}  {'CacheRd':>8}  "
          f"{'LLM#':>5}  {'CacheHit%':>9}  {'Time':>6}")
    print("-" * 72)
    for t in turns:
        hit = t["vllm_prefix_cache_hit_rate"]
        hit_str = f"{hit*100:.2f}%" if hit >= 0 else "N/A"
        print(f"{t['turn_number']:>5}  {t['prompt_tokens']:>8}  {t['completion_tokens']:>8}  "
              f"{t['cache_read_tokens']:>8}  {t['num_llm_calls']:>5}  "
              f"{hit_str:>9}  {t['elapsed_seconds']:>5.1f}s")
    print("-" * 72)
    print(f"Total  {agg['total_prompt_tokens']:>8}  {agg['total_completion_tokens']:>8}  "
          f"{agg['total_cache_read_tokens']:>8}  {agg['total_llm_calls']:>5}  "
          f"ratio={agg['cache_read_ratio']:.4f}  {agg['total_elapsed_seconds']:>5.1f}s")

    # 理论浪费
    prompts = [t["prompt_tokens"] for t in turns]
    reusable = sum(min(prompts[i-1], prompts[i]) for i in range(1, len(prompts)))
    total = sum(prompts)
    print(f"\n理论可复用前缀: {reusable:,} / {total:,} tokens ({reusable/max(1,total)*100:.1f}%)")
    print(f"{'=' * 72}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=os.path.join(ROOT, "results", "01_web", "repeated_message_traces.json"),
                        help="run_web_skills_test.py 输出的 JSON")
    parser.add_argument("--outdir", default=os.path.join(ROOT, "results", "01_web", "figures"),
                        help="图表输出目录")
    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    print(f"读取数据: {input_path}")
    with open(input_path) as f:
        data = json.load(f)

    turns = data["turns"]
    print(f"共 {len(turns)} 轮数据")

    if not turns:
        print("[错误] 无轮次数据")
        return

    print_summary(data)

    setup_style()
    print(f"\n生成图表 -> {outdir}/")
    plot_per_turn_tokens(turns, outdir)
    plot_cumulative_and_cache(turns, outdir)
    plot_prefix_reuse(turns, outdir)
    plot_calls_and_latency(turns, outdir)
    print("\n完成!")


if __name__ == "__main__":
    main()
