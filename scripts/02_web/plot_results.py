"""
plot_results.py - 02_web 实验可视化

读取 repeated_message_traces.json 和(可选)llm_call_prompts.jsonl,
生成以下图表,保存到 results/01_web/figures/:

  fig1_token_breakdown.pdf/png        Per-turn token 分解 (stacked bar)
  fig2_prefix_cache_hit_rate.pdf/png  Per-turn vLLM prefix cache 命中率
  fig3_latency_and_calls.pdf/png      Per-turn 延迟与 LLM 调用次数
  fig4_prompt_growth.pdf/png          Cumulative prompt tokens 增长
  fig5_percall_tokens.pdf/png         Per-call prompt tokens 瀑布图 (需要 JSONL)
  fig6_percall_cache_hit.pdf/png      Per-call vLLM cache 命中率 (需要 JSONL)
  fig7_cache_write_vs_read.pdf/png    Per-turn cache_write vs cache_read grouped bar
  fig8_reasoning_tokens.pdf/png       Per-call reasoning vs completion tokens (需要 JSONL)
  fig9_gpu_cache_usage.pdf/png        Per-call GPU KV cache 饱和度 (需要 JSONL)
  fig10_percall_latency.pdf/png       Per-call 延迟 bar，按 turn 着色 (需要 JSONL)
  fig11_context_growth.pdf/png        Per-call context 字符数与消息数增长 (需要 JSONL)
  fig12_cache_efficiency.pdf/png      Per-turn effective cache ratio 折线

用法:
    python scripts/02_web/plot_results.py [选项]

选项:
    --input PATH           主 JSON 路径
    --calls-input PATH     per-call JSONL 路径
    --outdir DIR           图表输出目录
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
})

COLORS = {
    "prompt": "#4C72B0",
    "completion": "#DD8452",
    "cache_read": "#55A868",
    "cache_write": "#C44E52",
    "hit_rate": "#8172B2",
    "latency": "#937860",
    "calls": "#DA8BC3",
    "cumulative": "#4C72B0",
}


def save_fig(fig, outdir: str, name: str):
    """保存 PDF + PNG."""
    for ext in ("pdf", "png"):
        path = os.path.join(outdir, f"{name}.{ext}")
        fig.savefig(path)
    plt.close(fig)
    print(f"  saved {name}")


# ---------------------------------------------------------------------------
# Turn-level 图表
# ---------------------------------------------------------------------------

def load_turns(json_path: str) -> tuple[dict, list[dict]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("metadata", {}), data["turns"]


def fig1_token_breakdown(turns: list[dict], outdir: str):
    """Per-turn token 分解: stacked bar (prompt / completion / cache_read)."""
    xs = [t["turn_number"] for t in turns]
    prompt = [t["prompt_tokens"] for t in turns]
    completion = [t["completion_tokens"] for t in turns]
    cache_read = [t["cache_read_tokens"] for t in turns]

    fig, ax = plt.subplots(figsize=(7, 4))
    bar_w = 0.6
    ax.bar(xs, prompt, bar_w, label="Prompt tokens", color=COLORS["prompt"])
    ax.bar(xs, completion, bar_w, bottom=prompt, label="Completion tokens", color=COLORS["completion"])
    if any(c > 0 for c in cache_read):
        ax.bar(xs, cache_read, bar_w,
               bottom=[p + c for p, c in zip(prompt, completion)],
               label="Cache read tokens", color=COLORS["cache_read"])

    ax.set_xlabel("Turn")
    ax.set_ylabel("Tokens")
    ax.set_title("Per-Turn Token Breakdown")
    ax.set_xticks(xs)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k" if x >= 1000 else f"{x:.0f}"))
    ax.legend()
    save_fig(fig, outdir, "fig1_token_breakdown")


def fig2_prefix_cache_hit_rate(turns: list[dict], outdir: str):
    """Per-turn vLLM prefix cache 命中率 (line + bar)."""
    xs = [t["turn_number"] for t in turns]
    hit_rate = [t["vllm_prefix_cache_hit_rate"] for t in turns]
    hits = [t["vllm_prefix_cache_hits_tokens"] for t in turns]
    queries = [t["vllm_prefix_cache_queries_tokens"] for t in turns]

    fig, ax1 = plt.subplots(figsize=(7, 4))

    # 左轴: 命中率折线
    color_hr = COLORS["hit_rate"]
    ax1.plot(xs, hit_rate, "o-", color=color_hr, linewidth=2, markersize=7, label="Hit rate", zorder=3)
    ax1.set_xlabel("Turn")
    ax1.set_ylabel("Prefix Cache Hit Rate", color=color_hr)
    ax1.set_ylim(0, 1.05)
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax1.tick_params(axis="y", labelcolor=color_hr)
    ax1.set_xticks(xs)

    # 右轴: hits / queries tokens (双 bar)
    ax2 = ax1.twinx()
    bar_w = 0.3
    ax2.bar([x - bar_w / 2 for x in xs], queries, bar_w, alpha=0.35,
            color=COLORS["prompt"], label="Query tokens")
    ax2.bar([x + bar_w / 2 for x in xs], hits, bar_w, alpha=0.35,
            color=COLORS["cache_read"], label="Hit tokens")
    ax2.set_ylabel("Tokens")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k" if x >= 1000 else f"{x:.0f}"))

    # 合并 legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="lower right")
    ax1.set_title("vLLM Prefix Cache Hit Rate per Turn")

    save_fig(fig, outdir, "fig2_prefix_cache_hit_rate")


def fig3_latency_and_calls(turns: list[dict], outdir: str):
    """Per-turn 延迟 (bar) 与 LLM 调用次数 (line)."""
    xs = [t["turn_number"] for t in turns]
    latency = [t["elapsed_seconds"] for t in turns]
    calls = [t["num_llm_calls"] for t in turns]

    fig, ax1 = plt.subplots(figsize=(7, 4))

    ax1.bar(xs, latency, 0.6, color=COLORS["latency"], alpha=0.7, label="Latency (s)")
    ax1.set_xlabel("Turn")
    ax1.set_ylabel("Elapsed (s)", color=COLORS["latency"])
    ax1.tick_params(axis="y", labelcolor=COLORS["latency"])
    ax1.set_xticks(xs)

    ax2 = ax1.twinx()
    ax2.plot(xs, calls, "s-", color=COLORS["calls"], linewidth=2, markersize=7, label="LLM calls")
    ax2.set_ylabel("# LLM Calls", color=COLORS["calls"])
    ax2.tick_params(axis="y", labelcolor=COLORS["calls"])
    ax2.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right")
    ax1.set_title("Per-Turn Latency & LLM Call Count")

    save_fig(fig, outdir, "fig3_latency_and_calls")


def fig4_prompt_growth(turns: list[dict], outdir: str):
    """Cumulative prompt tokens 增长曲线."""
    xs = [t["turn_number"] for t in turns]
    cumulative = [t["cumulative_prompt_tokens"] for t in turns]
    per_turn = [t["prompt_tokens"] for t in turns]

    fig, ax1 = plt.subplots(figsize=(7, 4))

    ax1.fill_between(xs, cumulative, alpha=0.2, color=COLORS["cumulative"])
    ax1.plot(xs, cumulative, "o-", color=COLORS["cumulative"], linewidth=2, markersize=7, label="Cumulative prompt")
    ax1.set_xlabel("Turn")
    ax1.set_ylabel("Cumulative Prompt Tokens", color=COLORS["cumulative"])
    ax1.tick_params(axis="y", labelcolor=COLORS["cumulative"])
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
    ax1.set_xticks(xs)

    ax2 = ax1.twinx()
    ax2.bar(xs, per_turn, 0.4, alpha=0.4, color=COLORS["completion"], label="Per-turn prompt")
    ax2.set_ylabel("Per-Turn Prompt Tokens", color=COLORS["completion"])
    ax2.tick_params(axis="y", labelcolor=COLORS["completion"])
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k" if x >= 1000 else f"{x:.0f}"))

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left")
    ax1.set_title("Prompt Token Growth across Turns")

    save_fig(fig, outdir, "fig4_prompt_growth")


# ---------------------------------------------------------------------------
# Per-call 图表 (需要 JSONL 或 JSON 中的 llm_calls)
# ---------------------------------------------------------------------------

def load_per_call_data(json_path: str, jsonl_path: str | None) -> list[dict]:
    """优先从 JSONL 加载 per-call 数据,回退到主 JSON 中的 llm_calls 字段。"""
    # 尝试 JSONL
    if jsonl_path and os.path.isfile(jsonl_path) and os.path.getsize(jsonl_path) > 10:
        records = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        if records:
            return records

    # 回退: 主 JSON 中的 turns[].llm_calls
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    records = []
    for t in data.get("turns", []):
        for c in t.get("llm_calls", []):
            c["turn_number"] = t["turn_number"]
            records.append(c)
    return records


def fig5_percall_tokens(calls: list[dict], outdir: str):
    """Per-call prompt tokens 瀑布图: x=call_index, 颜色=turn。"""
    if not calls:
        return

    indices = list(range(len(calls)))
    prompt_tokens = [c.get("prompt_tokens", 0) for c in calls]
    turn_numbers = [c.get("turn_number", 0) for c in calls]

    # 颜色映射: turn_number -> color
    unique_turns = sorted(set(turn_numbers))
    cmap = plt.cm.get_cmap("tab10", max(len(unique_turns), 1))
    turn_to_color = {t: cmap(i) for i, t in enumerate(unique_turns)}
    colors = [turn_to_color[t] for t in turn_numbers]

    fig, ax = plt.subplots(figsize=(max(8, len(calls) * 0.5), 4.5))
    bars = ax.bar(indices, prompt_tokens, color=colors, edgecolor="white", linewidth=0.5)

    ax.set_xlabel("LLM Call Index (chronological)")
    ax.set_ylabel("Prompt Tokens")
    ax.set_title("Per-Call Prompt Tokens (colored by turn)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k" if x >= 1000 else f"{x:.0f}"))

    # 添加 turn 边界线
    for i in range(1, len(calls)):
        if turn_numbers[i] != turn_numbers[i - 1]:
            ax.axvline(i - 0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

    # Legend
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=turn_to_color[t], label=f"Turn {t}") for t in unique_turns]
    ax.legend(handles=legend_handles, loc="upper left", ncol=min(len(unique_turns), 6))

    save_fig(fig, outdir, "fig5_percall_tokens")


def fig6_percall_cache_hit(calls: list[dict], outdir: str):
    """Per-call vLLM prefix cache 命中率 (scatter + line)."""
    if not calls:
        return

    # 过滤掉没有 vllm 数据的记录
    valid = [(i, c) for i, c in enumerate(calls)
             if c.get("vllm_prefix_cache_hit_rate", -1) >= 0]
    if not valid:
        return

    indices = [v[0] for v in valid]
    hit_rates = [v[1]["vllm_prefix_cache_hit_rate"] for v in valid]
    turn_numbers = [v[1].get("turn_number", 0) for v in valid]

    unique_turns = sorted(set(turn_numbers))
    cmap = plt.cm.get_cmap("tab10", max(len(unique_turns), 1))
    turn_to_color = {t: cmap(i) for i, t in enumerate(unique_turns)}
    colors = [turn_to_color[t] for t in turn_numbers]

    fig, ax = plt.subplots(figsize=(max(8, len(calls) * 0.5), 4.5))
    ax.scatter(indices, hit_rates, c=colors, s=60, zorder=3, edgecolors="white", linewidths=0.5)
    ax.plot(indices, hit_rates, "-", color="gray", alpha=0.4, linewidth=1, zorder=2)

    ax.set_xlabel("LLM Call Index (chronological)")
    ax.set_ylabel("Prefix Cache Hit Rate")
    ax.set_title("Per-Call vLLM Prefix Cache Hit Rate")
    ax.set_ylim(-0.05, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    # turn 边界线
    all_turns = [c.get("turn_number", 0) for c in calls]
    for i in range(1, len(calls)):
        if all_turns[i] != all_turns[i - 1]:
            ax.axvline(i - 0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=turn_to_color[t], label=f"Turn {t}") for t in unique_turns]
    ax.legend(handles=legend_handles, loc="lower right", ncol=min(len(unique_turns), 6))

    save_fig(fig, outdir, "fig6_percall_cache_hit")


def fig7_cache_write_vs_read(turns: list[dict], outdir: str):
    """Per-turn cache_write vs cache_read tokens (grouped bar).

    展示 KV cache 从"写主导"变为"读主导"的过渡点。
    """
    xs = np.array([t["turn_number"] for t in turns])
    cache_write = np.array([t.get("cache_write_tokens", 0) for t in turns])
    cache_read = np.array([t.get("cache_read_tokens", 0) for t in turns])

    if not any(cache_write > 0) and not any(cache_read > 0):
        print("  fig7: no cache_write/read data, skipping")
        return

    bar_w = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(xs - bar_w / 2, cache_write, bar_w, label="Cache write tokens", color=COLORS["cache_write"], alpha=0.85)
    ax.bar(xs + bar_w / 2, cache_read, bar_w, label="Cache read tokens", color=COLORS["cache_read"], alpha=0.85)

    ax.set_xlabel("Turn")
    ax.set_ylabel("Tokens")
    ax.set_title("Cache Write vs Cache Read Tokens per Turn")
    ax.set_xticks(xs)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k" if x >= 1000 else f"{x:.0f}"))
    ax.legend()

    save_fig(fig, outdir, "fig7_cache_write_vs_read")


def fig8_reasoning_tokens(calls: list[dict], outdir: str):
    """Per-call reasoning vs completion tokens (stacked bar).

    量化 Qwen3 reasoning_effort=high 带来的 thinking overhead。
    """
    if not calls:
        return

    reasoning = [c.get("reasoning_tokens", 0) for c in calls]
    completion = [c.get("completion_tokens", 0) for c in calls]

    if not any(r > 0 for r in reasoning):
        print("  fig8: no reasoning_tokens data, skipping")
        return

    indices = list(range(len(calls)))
    turn_numbers = [c.get("turn_number", 0) for c in calls]
    unique_turns = sorted(set(turn_numbers))
    cmap = plt.cm.get_cmap("tab10", max(len(unique_turns), 1))
    turn_to_color = {t: cmap(i) for i, t in enumerate(unique_turns)}

    fig, ax = plt.subplots(figsize=(max(8, len(calls) * 0.5), 4.5))
    ax.bar(indices, reasoning, label="Reasoning tokens", color="#E8A838", alpha=0.9)
    ax.bar(indices, completion, bottom=reasoning, label="Completion tokens", color=COLORS["completion"], alpha=0.85)

    # turn 边界线
    for i in range(1, len(calls)):
        if turn_numbers[i] != turn_numbers[i - 1]:
            ax.axvline(i - 0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

    ax.set_xlabel("LLM Call Index (chronological)")
    ax.set_ylabel("Tokens")
    ax.set_title("Per-Call Reasoning vs Completion Tokens")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k" if x >= 1000 else f"{x:.0f}"))

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="#E8A838", label="Reasoning tokens"),
        Patch(facecolor=COLORS["completion"], label="Completion tokens"),
    ] + [Patch(facecolor=turn_to_color[t], alpha=0.3, label=f"Turn {t}") for t in unique_turns]
    ax.legend(handles=legend_handles, loc="upper left", ncol=min(2 + len(unique_turns), 8))

    save_fig(fig, outdir, "fig8_reasoning_tokens")


def fig9_gpu_cache_usage(calls: list[dict], outdir: str):
    """Per-call vLLM GPU KV cache usage 折线图.

    观察 prefix cache 是否趋近饱和（饱和后命中率可能下降）。
    """
    if not calls:
        return

    usage = [c.get("vllm_gpu_cache_usage", -1) for c in calls]
    valid_mask = [u >= 0 for u in usage]
    if not any(valid_mask):
        print("  fig9: no vllm_gpu_cache_usage data, skipping")
        return

    indices = list(range(len(calls)))
    turn_numbers = [c.get("turn_number", 0) for c in calls]

    fig, ax = plt.subplots(figsize=(max(8, len(calls) * 0.5), 4))

    valid_x = [i for i, v in zip(indices, valid_mask) if v]
    valid_y = [u for u, v in zip(usage, valid_mask) if v]
    ax.plot(valid_x, valid_y, "o-", color="#2E86AB", linewidth=2, markersize=6, label="GPU cache usage")
    ax.fill_between(valid_x, valid_y, alpha=0.15, color="#2E86AB")

    # 90% 警戒线
    ax.axhline(0.9, color="red", linestyle="--", linewidth=1, alpha=0.7, label="90% threshold")

    # turn 边界线
    for i in range(1, len(calls)):
        if turn_numbers[i] != turn_numbers[i - 1]:
            ax.axvline(i - 0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

    ax.set_xlabel("LLM Call Index (chronological)")
    ax.set_ylabel("GPU KV Cache Usage")
    ax.set_title("vLLM GPU KV Cache Usage per Call")
    ax.set_ylim(-0.05, 1.1)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.legend(loc="upper left")

    save_fig(fig, outdir, "fig9_gpu_cache_usage")


def fig10_percall_latency(calls: list[dict], outdir: str):
    """Per-call elapsed_seconds bar，按 turn 着色.

    识别最慢的调用（首次 prefill vs 后续 decode）。
    """
    if not calls:
        return

    latency = [c.get("elapsed_seconds", 0) for c in calls]
    turn_numbers = [c.get("turn_number", 0) for c in calls]
    indices = list(range(len(calls)))

    unique_turns = sorted(set(turn_numbers))
    cmap = plt.cm.get_cmap("tab10", max(len(unique_turns), 1))
    turn_to_color = {t: cmap(i) for i, t in enumerate(unique_turns)}
    colors = [turn_to_color[t] for t in turn_numbers]

    fig, ax = plt.subplots(figsize=(max(8, len(calls) * 0.5), 4.5))
    ax.bar(indices, latency, color=colors, edgecolor="white", linewidth=0.5)

    # turn 边界线
    for i in range(1, len(calls)):
        if turn_numbers[i] != turn_numbers[i - 1]:
            ax.axvline(i - 0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

    ax.set_xlabel("LLM Call Index (chronological)")
    ax.set_ylabel("Elapsed (s)")
    ax.set_title("Per-Call Latency (colored by turn)")

    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=turn_to_color[t], label=f"Turn {t}") for t in unique_turns]
    ax.legend(handles=legend_handles, loc="upper right", ncol=min(len(unique_turns), 6))

    save_fig(fig, outdir, "fig10_percall_latency")


def fig11_context_growth(calls: list[dict], outdir: str):
    """Per-call 总 context 字符数折线图（来自 messages_summary）.

    直观展示 context 膨胀速度，以及 LLMSummarizingCondenser 截断效果。
    """
    if not calls:
        return

    total_chars = []
    num_messages = []
    for c in calls:
        summary = c.get("messages_summary", [])
        total_chars.append(sum(m.get("chars", 0) for m in summary))
        num_messages.append(len(summary))

    if not any(tc > 0 for tc in total_chars):
        print("  fig11: no messages_summary data, skipping")
        return

    indices = list(range(len(calls)))
    turn_numbers = [c.get("turn_number", 0) for c in calls]

    fig, ax1 = plt.subplots(figsize=(max(8, len(calls) * 0.5), 4.5))

    color_chars = "#4C72B0"
    color_msgs = "#DD8452"

    ax1.plot(indices, total_chars, "o-", color=color_chars, linewidth=2, markersize=6, label="Total chars")
    ax1.fill_between(indices, total_chars, alpha=0.1, color=color_chars)
    ax1.set_xlabel("LLM Call Index (chronological)")
    ax1.set_ylabel("Total Context Chars", color=color_chars)
    ax1.tick_params(axis="y", labelcolor=color_chars)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k" if x >= 1000 else f"{x:.0f}"))

    ax2 = ax1.twinx()
    ax2.plot(indices, num_messages, "s--", color=color_msgs, linewidth=1.5, markersize=5, label="# Messages")
    ax2.set_ylabel("# Messages in Context", color=color_msgs)
    ax2.tick_params(axis="y", labelcolor=color_msgs)
    ax2.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # turn 边界线 + condenser 截断标注
    for i in range(1, len(calls)):
        if turn_numbers[i] != turn_numbers[i - 1]:
            ax1.axvline(i - 0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
        # 字符数骤降 > 30% 认为发生了 condenser 截断
        if total_chars[i] < total_chars[i - 1] * 0.7 and total_chars[i - 1] > 0:
            ax1.annotate("condense", xy=(i, total_chars[i]),
                         xytext=(i, total_chars[i] * 1.1),
                         fontsize=7, color="red", ha="center",
                         arrowprops=dict(arrowstyle="->", color="red", lw=0.8))

    ax1.set_title("Context Size Growth (chars & message count)")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left")

    save_fig(fig, outdir, "fig11_context_growth")


def fig12_cache_efficiency(turns: list[dict], outdir: str):
    """Per-turn effective cache ratio = cache_read / (prompt + cache_read) 折线图.

    衡量"有效输入中有多少来自缓存"，综合反映前缀缓存收益。
    """
    xs = [t["turn_number"] for t in turns]
    ratios = []
    for t in turns:
        prompt = t.get("prompt_tokens", 0)
        cache_read = t.get("cache_read_tokens", 0)
        total = prompt + cache_read
        ratios.append(cache_read / total if total > 0 else 0.0)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xs, ratios, "o-", color=COLORS["cache_read"], linewidth=2, markersize=7)
    ax.fill_between(xs, ratios, alpha=0.15, color=COLORS["cache_read"])

    ax.set_xlabel("Turn")
    ax.set_ylabel("Effective Cache Ratio")
    ax.set_title("Cache Read / (Prompt + Cache Read) per Turn")
    ax.set_ylim(-0.05, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_xticks(xs)

    # 标出每个点的数值
    for x, r in zip(xs, ratios):
        ax.annotate(f"{r:.1%}", xy=(x, r), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=8)

    save_fig(fig, outdir, "fig12_cache_efficiency")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default=os.path.join(ROOT, "results", "01_web", "repeated_message_traces.json"))
    parser.add_argument("--calls-input", default=os.path.join(ROOT, "results", "01_web", "llm_call_prompts.jsonl"))
    parser.add_argument("--outdir", default=os.path.join(ROOT, "results", "01_web", "figures"))
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # --- Turn-level ---
    print("Loading turn-level data...")
    metadata, turns = load_turns(args.input)
    n_turns = len(turns)
    print(f"  {n_turns} turns, {metadata.get('experiment', '?')}")

    print("Generating turn-level figures...")
    fig1_token_breakdown(turns, args.outdir)
    fig2_prefix_cache_hit_rate(turns, args.outdir)
    fig3_latency_and_calls(turns, args.outdir)
    fig4_prompt_growth(turns, args.outdir)
    fig7_cache_write_vs_read(turns, args.outdir)
    fig12_cache_efficiency(turns, args.outdir)

    # --- Per-call ---
    print("Loading per-call data...")
    calls = load_per_call_data(args.input, args.calls_input)
    if calls:
        print(f"  {len(calls)} LLM calls across {len(set(c.get('turn_number', 0) for c in calls))} turns")
        print("Generating per-call figures...")
        fig5_percall_tokens(calls, args.outdir)
        fig6_percall_cache_hit(calls, args.outdir)
        fig8_reasoning_tokens(calls, args.outdir)
        fig9_gpu_cache_usage(calls, args.outdir)
        fig10_percall_latency(calls, args.outdir)
        fig11_context_growth(calls, args.outdir)
    else:
        print("  No per-call data available (run experiment with new code to generate JSONL)")

    print(f"\nAll figures saved to: {args.outdir}")


if __name__ == "__main__":
    main()
