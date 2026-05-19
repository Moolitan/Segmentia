#!/usr/bin/env python3
"""
Skill vs No-Skill Comparison Figures for SegKV Paper.

Generates publication-quality comparison figures between with-skill and
no-skill (control group) experiment traces to quantify the overhead
introduced by the skill mechanism.

Usage:
  python scripts/03/skill_comparison.py \
    --results-dir results/03/multurn_bench \
    --out-dir results/03/multurn_bench/segkv_figures
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import numpy as np


# ── Style ─────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "savefig.facecolor": "white",
})

SEQUENCES = ["T4_B", "T6_F", "T8_A", "T10_A"]
SEQ_LABELS = {
    "T4_B": "T4-B (Excel→Word→PPT→PDF)",
    "T6_F": "T6-F (Find→Create→Frontend→Word→PPT→PDF)",
    "T8_A": "T8_A (Find → MCP → Frontend → Vercel → Excel → Word → PPT → PDF)",
    "T10_A": "T10_A (Find → Read → MCP → Frontend → Vercel → Excel → Skill → Word → PPT → PDF)",
}
SEQ_SHORT = {"T4_B": "T4-B", "T6_F": "T6_F", "T8_A": "T8_A", "T10_A": "T10_A"}

# Paired colors: darker for with-skill, lighter for no-skill
# COLORS_SKILL = {"T4_B": "#2563EB", "T4_F": "#DC2626", "T6_F": "#059669"}
# COLORS_NOSKILL = {"T4_B": "#93C5FD", "T4_F": "#FCA5A5", "T6_F": "#6EE7B7"}
VARIANT_COLORS = {
    ("with-skill", 16384): "#DC2626",
    ("with-skill", 32768): "#059669",
    ("no-skill", 16384): "#2563EB",
    ("no-skill", 32768): "#7C3AED",
}


# ── Data loading ──────────────────────────────────────────────────────────────

@dataclass
class SeqData:
    key: str
    label: str
    color: str
    num_turns: int = 0
    total_elapsed: float = 0.0
    calls: list = field(default_factory=list)
    timeline: list = field(default_factory=list)
    no_skills: bool = False
    context_len: int | None = None


def load_seq(results_dir: str, seq_key: str, context_len: int, noskill: bool = False) -> SeqData | None:
    suffix = "_noskill" if noskill else ""
    path = os.path.join(
        results_dir,
        f"ctx_{context_len}",
        f"{seq_key}{suffix}",
        "multiturn_sequence_traces.json",
    )
    if not os.path.exists(path):
        print(f"[SKIP] {path} not found")
        return None

    with open(path) as f:
        data = json.load(f)

    s = data["sequence"]
    group = "no-skill" if noskill else "with-skill"

    sd = SeqData(
        key=seq_key,
        label=f"{SEQ_LABELS.get(seq_key, seq_key)} [{group}, ctx={context_len}]",
        color="#333",   # 这里只占位，真正画图时动态决定
        num_turns=s["num_turns"],
        total_elapsed=s["total_elapsed_seconds"],
        calls=s["llm_calls"],
        timeline=s.get("vllm_timeline", []),
        no_skills=noskill,
        context_len=context_len,
    )
    return sd


# ── Helpers ───────────────────────────────────────────────────────────────────

def per_turn_stats(calls: list[dict]) -> dict:
    turns = {}
    for c in calls:

        t = c["turn_number"]
        print(t)
        if t not in turns:
            turns[t] = {
                "calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                "total_latency": 0.0, "latencies": [],
            }
        turns[t]["calls"] += 1
        turns[t]["prompt_tokens"] += c["prompt_tokens"]
        turns[t]["completion_tokens"] += c["completion_tokens"]
        turns[t]["total_latency"] += c["request_latency_seconds"]
        turns[t]["latencies"].append(c["request_latency_seconds"])
    for d in turns.values():
        d["efficiency_ms_per_tok"] = (
            d["total_latency"] / d["prompt_tokens"] * 1000
            if d["prompt_tokens"] > 0 else 0
        )
    return turns


def detect_truncations(calls: list[dict], threshold: float = 0.5) -> list[dict]:
    truncs = []
    for i in range(1, len(calls)):
        prev_pt = calls[i - 1]["prompt_tokens"]
        curr_pt = calls[i]["prompt_tokens"]
        if prev_pt > 5000 and curr_pt < prev_pt * (1 - threshold):
            truncs.append({
                "index": i, "before": prev_pt, "after": curr_pt,
                "fraction_lost": 1 - curr_pt / prev_pt,
            })
    return truncs


def detect_kv_drops(timeline: list[dict], drop_ratio: float = 0.3) -> int:
    """Count KV cache usage drops (zero-value idle samples excluded)."""
    nonzero = [(s["elapsed_seconds"], s.get("kv_cache_usage_perc", 0) or 0)
               for s in timeline
               if (s.get("kv_cache_usage_perc", 0) or 0) > 0]
    drops = 0
    for i in range(1, len(nonzero)):
        prev = nonzero[i - 1][1]
        curr = nonzero[i][1]
        if prev > 0.001 and curr < prev * drop_ratio:
            drops += 1
    return drops


def get_turn_boundaries(calls: list[dict]) -> list[int]:
    bounds = []
    prev = None
    for i, c in enumerate(calls):
        t = c.get("turn_number", 1)
        if prev is not None and t != prev:
            bounds.append(i)
        prev = t
    return bounds

def group_variants_by_sequence(all_variants: list[SeqData]) -> dict[str, list[SeqData]]:
    grouped = {}
    for sd in all_variants:
        grouped.setdefault(sd.key, []).append(sd)
    return grouped

def get_variant_color(sd: SeqData) -> str:
    group = "no-skill" if sd.no_skills else "with-skill"
    return VARIANT_COLORS.get((group, sd.context_len), "#333")

def get_variant_style(sd: SeqData) -> dict:
    group = "no-skill" if sd.no_skills else "with-skill"

    if group == "with-skill":
        return {
            "linestyle": "-",
            "marker": "o",
            "markersize": 3.2,
        }
    else:
        return {
            "linestyle": "--",
            "marker": None,
            "markersize": 0,
        }



# ── Figure C1: Side-by-Side Prompt Token Growth ──────────────────────────────
def figC1_prompt_growth_comparison(
    grouped_variants: dict[str, list[SeqData]],
    out_dir: str,
):
    """
    One subplot per sequence.
    Overlay all variants in the same subplot:
      - with-skill / no-skill
      - ctx=16384 / 32768 / ...
    Legend format: T4-B-skill-16384
    X-axis length: use the longest skill-run call count in that sequence;
    if no skill-run exists, fallback to the longest variant.
    """
    seq_keys = [k for k in SEQUENCES if k in grouped_variants and grouped_variants[k]]
    if not seq_keys:
        print("[WARN] figC1: no grouped variants found")
        return

    n = len(seq_keys)
    ncols = 2 if n > 1 else 1
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(7.2 * ncols, 3.8 * nrows),
        sharex=False,
        squeeze=False,
    )


    def variant_label(sd: SeqData) -> str:
        seq_short = SEQ_SHORT.get(sd.key, sd.key).replace(" ", "")
        mode = "noskill" if sd.no_skills else "skill"
        ctx = sd.context_len if sd.context_len is not None else "NA"
        return f"{seq_short}-{mode}-{ctx}"

    def subplot_title(seq_key: str, variants: list[SeqData]) -> str:
        total_runs = len(variants)
        skill_runs = sum(0 if v.no_skills else 1 for v in variants)
        noskill_runs = sum(1 if v.no_skills else 0 for v in variants)
        return (
            f"{SEQ_SHORT.get(seq_key, seq_key)} — "
            f"{total_runs} runs ({skill_runs} skill / {noskill_runs} no-skill)"
        )

    for idx, seq_key in enumerate(seq_keys):
        ax = axes[idx // ncols][idx % ncols]
        variants = grouped_variants[seq_key]

        # 先按 no-skill / skill，再按 ctx 排序，图例更稳定
        variants = sorted(
            variants,
            key=lambda sd: (sd.no_skills, sd.context_len if sd.context_len is not None else -1)
        )

        # 横坐标：优先取最长的 skill 请求数
        skill_lengths = [len(sd.calls) for sd in variants if not sd.no_skills]
        if skill_lengths:
            x_max = max(skill_lengths)
        else:
            x_max = max(len(sd.calls) for sd in variants)

        any_trunc = False

        for sd in variants:
            calls = sd.calls
            if not calls:
                continue

            xs = list(range(len(calls)))
            pts = [c["prompt_tokens"] for c in calls]
            bounds = get_turn_boundaries(calls)
            truncs = detect_truncations(calls)

            line_color = get_variant_color(sd)
            line_style = get_variant_style(sd)


            ax.plot(
                xs,
                pts,
                color=line_color,
                linewidth=1.8,
                linestyle=line_style["linestyle"],
                marker=line_style["marker"],
                markersize=line_style["markersize"],
                alpha=0.95,
                zorder=3,
                label=variant_label(sd),
            )

            # turn boundaries: 只画一套，避免多条重复边界太乱
            # 这里选“该序列最长的一条曲线”的边界作为参考
            if len(calls) == max(len(v.calls) for v in variants):
                for bx in bounds:
                    ax.axvline(
                        x=bx - 0.5,
                        color="#94A3B8",
                        linestyle="--",
                        linewidth=0.8,
                        alpha=0.5,
                        zorder=1,
                    )

            # truncation 标记：保留原风格，但注释更紧凑，防止多曲线遮挡
            for tr in truncs:
                any_trunc = True
                ax.axvspan(
                    tr["index"] - 0.45,
                    tr["index"] + 0.45,
                    alpha=0.10,
                    color="#DC2626",
                    zorder=0,
                )
                ax.annotate(
                    f"{variant_label(sd)}\n{tr['before']//1000}K→{tr['after']//1000}K",
                    xy=(tr["index"], tr["after"]),
                    xytext=(tr["index"] + 0.8, tr["after"] + 2500),
                    fontsize=6.8,
                    color="#DC2626",
                    fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="#DC2626", lw=0.7),
                    bbox=dict(
                        boxstyle="round,pad=0.18",
                        facecolor="#EDF437",
                        edgecolor="#DC2626",
                        alpha=0.75,
                    ),
                    zorder=5,
                )

        # 上限参考线：原来写死 32768，现在按“该序列出现过的最大 ctx”画主参考线
        seq_ctxs = sorted({sd.context_len for sd in variants if sd.context_len is not None})
        if seq_ctxs:
            for ctx in seq_ctxs:
                alpha = 0.45 if ctx == max(seq_ctxs) else 0.25
                ax.axhline(
                    y=ctx,
                    color="#EF4444",
                    linestyle=":",
                    linewidth=0.9,
                    alpha=alpha,
                    zorder=1,
                )
                ax.text(
                    x_max - 0.1,
                    ctx + 350,
                    f"ctx={ctx}",
                    fontsize=7,
                    color="#EF4444",
                    ha="right",
                    va="bottom",
                    alpha=alpha,
                )

        ax.set_xlim(-0.5, x_max - 0.5 if x_max > 0 else 0.5)
        ax.set_title(subplot_title(seq_key, variants), fontsize=10.5, fontweight="bold")
        ax.set_xlabel("LLM Call Index")
        ax.set_ylabel("Prompt Tokens")
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f}K")
        )
        ax.grid(True, alpha=0.2, linewidth=0.5)

        # 图例放子图内右上角
        ax.legend(
            fontsize=8.2,
            loc="upper left",
            frameon=True,
            framealpha=0.92,
            ncol=1,
        )

        if any_trunc:
            ax.text(
                0.99, 0.02,
                "Red shaded area: truncation detected",
                transform=ax.transAxes,
                ha="right", va="bottom",
                fontsize=7.5, color="#B91C1C",
            )

    # 清掉空白子图
    total_axes = nrows * ncols
    for j in range(len(seq_keys), total_axes):
        fig.delaxes(axes[j // ncols][j % ncols])

    fig.suptitle(
        "Fig C1: Prompt Token Growth Comparison Across Skill Setting and Context Length",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(
            os.path.join(out_dir, f"figC1_prompt_growth_comparison.{ext}"),
            bbox_inches="tight",
        )
    plt.close(fig)
    print("[OK] figC1_prompt_growth_comparison")


# ── Figure C2: Aggregate Comparison Bar Chart ────────────────────────────────

def figC2_aggregate_comparison(grouped_variants: dict[str, list[SeqData]], out_dir: str):
    """Aggregate bar charts comparing all variants within each sequence:
    with-skill / no-skill × different context lengths.
    """
    seq_keys = [k for k in SEQUENCES if k in grouped_variants and grouped_variants[k]]
    if not seq_keys:
        print("[WARN] figC2: no grouped variants found")
        return

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    axes = axes.flatten()

    metric_defs = [
        ("total_prompt", "Total Prompt Tokens", lambda sd: sum(c["prompt_tokens"] for c in sd.calls),
         lambda v, _: f"{v / 1000:.0f}K", "(a)"),
        ("total_elapsed", "Total Elapsed Time (s)", lambda sd: sd.total_elapsed,
         None, "(b)"),
        ("num_calls", "Number of LLM Calls", lambda sd: len(sd.calls),
         None, "(c)"),
        ("num_truncations", "Truncation Events", lambda sd: len(detect_truncations(sd.calls)),
         None, "(d)"),
    ]

    # 统一 4 种 variant 顺序
    variant_order = [
        ("with-skill", 16384),
        ("no-skill", 16384),
        ("with-skill", 32768),
        ("no-skill", 32768),
    ]

    def get_variant(sd: SeqData):
        group = "no-skill" if sd.no_skills else "with-skill"
        return (group, sd.context_len)

    def variant_label(group: str, ctx: int) -> str:
        mode = "skill" if group == "with-skill" else "noskill"
        return f"{mode}-{ctx}"

    def variant_color(group: str, ctx: int) -> str:
        # 这里按你前面定下来的风格来：
        # 颜色表示 ctx；skill/noskill 主要在图例名字里体现
        # 你要是更想 4 根柱子都不同色，也可以在这里改
        if ctx == 16384:
            return "#2563EB"   # 橙色
        elif ctx == 32768:
            return "#EEEA20DE"   # 深紫色
        return "#333333"

    x = np.arange(len(seq_keys))
    w = 0.18
    offsets = [-1.5 * w, -0.5 * w, 0.5 * w, 1.5 * w]

    for ax, (metric_key, ylabel, metric_fn, fmt, panel) in zip(axes, metric_defs):
        # 先把每个 variant 在所有 sequence 上的值收集起来
        values_by_variant = {v: [] for v in variant_order}

        for seq_key in seq_keys:
            seq_variants = grouped_variants[seq_key]
            variant_map = {get_variant(sd): sd for sd in seq_variants}

            for v in variant_order:
                sd = variant_map.get(v)
                if sd is None:
                    values_by_variant[v].append(np.nan)
                else:
                    values_by_variant[v].append(metric_fn(sd))

        # 画 4 组柱子
        for offset, v in zip(offsets, variant_order):
            group, ctx = v
            vals = values_by_variant[v]
            def variant_hatch(group: str) -> str:
                return "///" if group == "with-skill" else ""

            ax.bar(
                x + offset,
                vals,
                width=w,
                color=variant_color(group, ctx),
                alpha=0.9,
                edgecolor="black",
                linewidth=0.8,
                hatch=variant_hatch(group),
                label=variant_label(group, ctx),
                zorder=3,
            )


        ax.set_xticks(x)
        ax.set_xticklabels([SEQ_SHORT.get(k, k) for k in seq_keys])
        ax.set_ylabel(ylabel)
        ax.set_title(f"{panel} {ylabel}", fontweight="bold")
        if fmt:
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt))
        ax.legend(fontsize=8.5, loc="upper right", ncol=1)
        ax.grid(True, alpha=0.2, linewidth=0.5, axis="y")

    fig.suptitle(
        "Fig C2: Aggregate Metrics Comparison Across Skill Setting and Context Length",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    for ext in ["pdf", "png"]:
        fig.savefig(
            os.path.join(out_dir, f"figC2_aggregate_comparison.{ext}"),
            bbox_inches="tight",
        )
    plt.close(fig)
    print("[OK] figC2_aggregate_comparison")


# ── Figure C3: Per-Turn Efficiency Comparison ─────────────────────────────────

def figC3_turn_efficiency_comparison(
    grouped_variants: dict[str, list[SeqData]],
    out_dir: str,
):
    """Grouped bar chart: per-turn ms/tok for each sequence,
    comparing with-skill / no-skill across context lengths.
    """
    seq_keys = [k for k in SEQUENCES if k in grouped_variants and grouped_variants[k]]
    if not seq_keys:
        print("[WARN] figC3: no grouped variants found")
        return

    n = len(seq_keys)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    variant_order = [
        ("with-skill", 16384),
        ("with-skill", 32768),
        ("no-skill", 16384),
        ("no-skill", 32768),
    ]

    def get_variant(sd: SeqData):
        group = "no-skill" if sd.no_skills else "with-skill"
        return (group, sd.context_len)

    def variant_label(group: str, ctx: int) -> str:
        mode = "skill" if group == "with-skill" else "noskill"
        return f"{mode}-{ctx}"

    def variant_color(ctx: int) -> str:
        if ctx == 16384:
            return "#D97706"   # orange
        elif ctx == 32768:
            return "#7C3AED"   # deep purple
        return "#333333"

    def variant_hatch(group: str) -> str:
        return "///" if group == "with-skill" else ""

    for ax, seq_key in zip(axes, seq_keys):
        seq_variants = grouped_variants[seq_key]
        variant_map = {get_variant(sd): sd for sd in seq_variants}

        # 先统计所有 variant 的 per-turn efficiency
        turn_stats_map = {}
        max_turns = 0
        for v in variant_order:
            sd = variant_map.get(v)
            if sd is None:
                turn_stats_map[v] = {}
                continue
            print(sd.key,", ",sd.context_len)
            turns = per_turn_stats(sd.calls)
            turn_stats_map[v] = turns
            if turns:
                max_turns = max(max_turns, max(turns.keys()))

        if max_turns == 0:
            ax.text(
                0.5, 0.5, "No turn data",
                transform=ax.transAxes,
                ha="center", va="center",
                fontsize=12, color="#999",
            )
            ax.set_title(SEQ_SHORT.get(seq_key, seq_key), fontweight="bold")
            continue

        t_range = range(1, max_turns + 1)
        x = np.arange(len(list(t_range)))

        # 4 根柱子一组
        w = 0.18
        offsets = [-1.5 * w, -0.5 * w, 0.5 * w, 1.5 * w]

        for offset, v in zip(offsets, variant_order):
            group, ctx = v
            turns = turn_stats_map[v]
            vals = [
                turns[t]["efficiency_ms_per_tok"] if t in turns else np.nan
                for t in t_range
            ]

            bars = ax.bar(
                x + offset,
                vals,
                width=w,
                color=variant_color(ctx),
                edgecolor="black",
                linewidth=0.8,
                hatch=variant_hatch(group),
                alpha=0.9,
                label=variant_label(group, ctx),
                zorder=3,
            )

            # Annotate values
            for bar, val in zip(bars, vals):
                if np.isfinite(val) and val > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.08,
                        f"{val:.1f}",
                        ha="center",
                        va="bottom",
                        fontsize=6.5,
                        fontweight="bold",
                    )

        ax.set_xticks(x)
        ax.set_xticklabels([f"T{t}" for t in t_range])
        ax.set_xlabel("Turn")
        ax.set_title(SEQ_SHORT.get(seq_key, seq_key), fontweight="bold")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.2, linewidth=0.5, axis="y")

    axes[0].set_ylabel("Efficiency (ms/tok) — lower is better")
    fig.suptitle(
        "Fig C3: Per-Turn Inference Efficiency Across Skill Setting and Context Length",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    for ext in ["pdf", "png"]:
        fig.savefig(
            os.path.join(out_dir, f"figC3_turn_efficiency_comparison.{ext}"),
            bbox_inches="tight",
        )
    plt.close(fig)
    print("[OK] figC3_turn_efficiency_comparison")


# ── Figure C4: KV Cache Comparison ────────────────────────────────────────────

def figC4_kv_cache_comparison(pairs: list[tuple[SeqData, SeqData]], out_dir: str):
    """Side-by-side KV cache usage timeline: skill vs no-skill."""
    n = len(pairs)
    fig, axes = plt.subplots(n, 2, figsize=(14, 3.5 * n), sharex=False)
    if n == 1:
        axes = [axes]

    for row, (sd_skill, sd_noskill) in enumerate(pairs):
        for col, sd in enumerate([sd_noskill, sd_skill]):
            ax = axes[row][col] if n > 1 else axes[col]
            timeline = sd.timeline
            if not timeline:
                ax.text(0.5, 0.5, "No timeline data", transform=ax.transAxes,
                        ha="center", va="center", fontsize=12, color="#999")
            else:
                raw_elapsed = [s["elapsed_seconds"] for s in timeline]
                raw_kv = [s.get("kv_cache_usage_perc", 0) or 0 for s in timeline]
                elapsed = [t for t, v in zip(raw_elapsed, raw_kv) if v > 0]
                kv = [v for v in raw_kv if v > 0]
                ax.fill_between(elapsed, kv, alpha=0.3, color=sd.color)
                ax.plot(elapsed, kv, color=sd.color, linewidth=1.2)
                drops = detect_kv_drops(timeline)
                group = "No-Skill" if sd.no_skills else "With-Skill"
                ax.set_title(
                    f"{SEQ_SHORT[sd.key]} [{group}] — {drops} KV drops",
                    fontsize=10, fontweight="bold")

            ax.set_ylabel("KV Cache Usage (%)")
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
            ax.grid(True, alpha=0.2, linewidth=0.5)
            ax.set_xlabel("Elapsed Time (s)")

    fig.suptitle("Fig C4: KV Cache Usage — No-Skill (left) vs With-Skill (right)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(out_dir, f"figC4_kv_cache_comparison.{ext}"),
                    bbox_inches="tight")
    plt.close(fig)
    print("[OK] figC4_kv_cache_comparison")


# ── Figure C5: Skill Overhead Waterfall ───────────────────────────────────────

def figC5_skill_overhead_waterfall(pairs: list[tuple[SeqData, SeqData]],
                                    out_dir: str):
    """Waterfall chart decomposing the overhead introduced by skills:
    no-skill baseline + skill token overhead + additional truncation cost."""
    fig, ax = plt.subplots(figsize=(10, 6))

    seq_labels = [SEQ_SHORT[p[0].key] for p in pairs]
    x = np.arange(len(pairs))
    w = 0.5

    noskill_totals = []
    skill_totals = []
    for sd_skill, sd_noskill in pairs:
        noskill_totals.append(sum(c["prompt_tokens"] for c in sd_noskill.calls))
        skill_totals.append(sum(c["prompt_tokens"] for c in sd_skill.calls))

    skill_overhead = [s - n for s, n in zip(skill_totals, noskill_totals)]

    # Stack: noskill baseline + skill overhead
    bars_base = ax.bar(x, noskill_totals, w, color="#94A3B8", alpha=0.85,
                       label="Baseline (no-skill)", zorder=3)
    bars_oh = ax.bar(x, [max(0, o) for o in skill_overhead], w,
                     bottom=noskill_totals,
                     color="#F59E0B", alpha=0.85,
                     label="Skill overhead", zorder=3)
    # Handle negative overhead (skill might be more efficient in some cases)
    for i, oh in enumerate(skill_overhead):
        if oh < 0:
            ax.bar(x[i], abs(oh), w, bottom=noskill_totals[i] + oh,
                   color="#059669", alpha=0.85, zorder=3)

    # Annotate
    for i, (nt, st, oh) in enumerate(zip(noskill_totals, skill_totals,
                                          skill_overhead)):
        pct = oh / nt * 100 if nt > 0 else 0
        sign = "+" if pct > 0 else ""
        color = "#DC2626" if pct > 0 else "#059669"
        ax.text(i, st + 10000, f"{sign}{pct:.0f}% ({sign}{oh // 1000}K tok)",
                ha="center", va="bottom", fontsize=10, fontweight="bold",
                color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(seq_labels)
    ax.set_ylabel("Total Prompt Tokens")
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f}K"))
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.2, linewidth=0.5, axis="y")
    ax.set_title("Fig C5: Skill Mechanism Token Overhead — "
                 "Baseline vs With-Skill Total Prompt Tokens",
                 fontsize=13, fontweight="bold")

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(out_dir, f"figC5_skill_overhead_waterfall.{ext}"),
                    bbox_inches="tight")
    plt.close(fig)
    print("[OK] figC5_skill_overhead_waterfall")


# ── Figure C6: Prompt Tokens + KV Cache Overlay (with-skill only) ────────────

def figC6_prompt_kv_overlay(pairs: list[tuple[SeqData, SeqData]], out_dir: str):
    """Overlay prompt_tokens (per-request) and KV cache usage (timeline)
    on a shared elapsed-time x-axis for with-skill experiments.

    Each LLM request is drawn as a horizontal bar spanning
    [request_start, request_start + latency] with height = prompt_tokens.
    KV cache usage is drawn as a line on the secondary y-axis.
    Turn boundaries are shown as vertical dashed lines.
    """
    skill_seqs = [p[0] for p in pairs]  # with-skill only

    n = len(skill_seqs)
    fig, axes = plt.subplots(n, 1, figsize=(14, 4.5 * n), squeeze=False)

    for row, sd in enumerate(skill_seqs):
        ax_tok = axes[row, 0]

        calls = sd.calls
        timeline = sd.timeline
        if not calls:
            ax_tok.text(0.5, 0.5, "No LLM calls", transform=ax_tok.transAxes,
                        ha="center", va="center", fontsize=12, color="#999")
            continue

        # Compute elapsed time for each request relative to timeline t0 or
        # first request, whichever is earlier.
        t0 = timeline[0]["timestamp"] if timeline else calls[0]["request_started_at"]
        t0 = min(t0, calls[0]["request_started_at"])

        # --- Left y-axis: prompt tokens per request ---
        bar_colors_map = {
            "T4_B": ("#2563EB", "#93C5FD"),  # fill, edge
            "T4_F": ("#DC2626", "#FCA5A5"),
            "T6_F": ("#059669", "#6EE7B7"),
        }
        fill_c, edge_c = bar_colors_map.get(sd.key, ("#2563EB", "#93C5FD"))

        for c in calls:
            t_start = c["request_started_at"] - t0
            duration = c["request_latency_seconds"]
            pt = c["prompt_tokens"]
            ax_tok.barh(
                y=pt, width=duration, left=t_start, height=pt * 0.06,
                color=fill_c, alpha=0.7, edgecolor=edge_c, linewidth=0.5,
                zorder=3,
            )
            # Also plot a marker at center of the bar for readability
            ax_tok.plot(t_start + duration / 2, pt, marker="o", markersize=4,
                        color=fill_c, markeredgecolor="white", markeredgewidth=0.5,
                        zorder=4)

        # Connect the dots with a light line to show trend
        req_times = [c["request_started_at"] - t0 + c["request_latency_seconds"] / 2
                     for c in calls]
        req_pts = [c["prompt_tokens"] for c in calls]
        ax_tok.plot(req_times, req_pts, color=fill_c, linewidth=1.0, alpha=0.4,
                    linestyle="-", zorder=2)

        ax_tok.set_ylabel("Prompt Tokens", color=fill_c, fontweight="bold")
        ax_tok.tick_params(axis="y", labelcolor=fill_c)
        ax_tok.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f}K"))

        # --- Right y-axis: KV cache usage ---
        ax_kv = ax_tok.twinx()

        if timeline:
            raw_elapsed = [s["elapsed_seconds"] for s in timeline]
            raw_kv = [s.get("kv_cache_usage_perc", 0) or 0 for s in timeline]
            # Keep non-zero samples for KV plot
            kv_elapsed = [t for t, v in zip(raw_elapsed, raw_kv) if v > 0]
            kv_vals = [v for v in raw_kv if v > 0]
            if kv_vals:
                ax_kv.fill_between(kv_elapsed, kv_vals, alpha=0.15,
                                   color="#7C3AED")
                ax_kv.plot(kv_elapsed, kv_vals, color="#7C3AED", linewidth=1.5,
                           alpha=0.85, zorder=2, label="KV Cache Usage")
        else:
            ax_kv.text(0.95, 0.5, "No timeline data", transform=ax_kv.transAxes,
                       ha="right", va="center", fontsize=10, color="#999")

        ax_kv.set_ylabel("KV Cache Usage (%)", color="#7C3AED", fontweight="bold")
        ax_kv.tick_params(axis="y", labelcolor="#7C3AED")
        ax_kv.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax_kv.set_ylim(-0.02, max(0.3, max(kv_vals, default=0.1) * 1.3))

        # --- Turn boundaries ---
        prev_turn = None
        for c in calls:
            tn = c.get("turn_number", 1)
            if prev_turn is not None and tn != prev_turn:
                t_boundary = c["request_started_at"] - t0
                ax_tok.axvline(x=t_boundary, color="#94A3B8", linestyle="--",
                               linewidth=1, alpha=0.7, zorder=1)
                ax_tok.text(t_boundary, ax_tok.get_ylim()[1] * 0.97,
                            f" T{tn}", fontsize=8, color="#64748B",
                            va="top", ha="left")
            prev_turn = tn

        # --- Annotate high-latency requests ---
        if calls:
            latencies = [c["request_latency_seconds"] for c in calls]
            median_lat = sorted(latencies)[len(latencies) // 2]
            for c in calls:
                lat = c["request_latency_seconds"]
                if lat > median_lat * 3 and lat > 30:
                    t_start = c["request_started_at"] - t0
                    ax_tok.annotate(
                        f"{lat:.0f}s",
                        xy=(t_start + lat / 2, c["prompt_tokens"]),
                        xytext=(0, 15), textcoords="offset points",
                        fontsize=7, color="#DC2626", fontweight="bold",
                        ha="center",
                        arrowprops=dict(arrowstyle="->", color="#DC2626",
                                        lw=0.6),
                    )

        # --- Title and grid ---
        total_pt = sum(c["prompt_tokens"] for c in calls)
        ax_tok.set_title(
            f"{SEQ_SHORT[sd.key]} [With-Skill] — "
            f"Prompt Tokens vs KV Cache Usage  "
            f"({len(calls)} calls, {total_pt // 1000}K total tok, "
            f"{sd.total_elapsed:.0f}s elapsed)",
            fontsize=11, fontweight="bold",
        )
        ax_tok.set_xlabel("Elapsed Time (s)")
        ax_tok.grid(True, alpha=0.15, linewidth=0.5)

        # Combined legend
        from matplotlib.lines import Line2D
        legend_elements = [
            mpatches.Patch(facecolor=fill_c, alpha=0.7, label="Prompt Tokens"),
            Line2D([0], [0], color="#7C3AED", linewidth=1.5, label="KV Cache Usage"),
        ]
        ax_tok.legend(handles=legend_elements, loc="upper left", fontsize=9)

    fig.suptitle(
        "Fig C6: Prompt Tokens × KV Cache Usage — Time-Aligned Overlay (With-Skill)",
        fontsize=14, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(out_dir, f"figC6_prompt_kv_overlay.{ext}"),
                    bbox_inches="tight")
    plt.close(fig)
    print("[OK] figC6_prompt_kv_overlay")


# ── Summary Statistics ────────────────────────────────────────────────────────

def print_comparison_summary(pairs: list[tuple[SeqData, SeqData]], out_dir: str):
    """Print and save comparison summary statistics."""
    lines = []
    lines.append("=" * 100)
    lines.append("Skill vs No-Skill Comparison — Summary Statistics")
    lines.append("=" * 100)

    latex_rows = []

    for sd_skill, sd_noskill in pairs:
        sk_total_pt = sum(c["prompt_tokens"] for c in sd_skill.calls)
        ns_total_pt = sum(c["prompt_tokens"] for c in sd_noskill.calls)
        sk_total_ct = sum(c["completion_tokens"] for c in sd_skill.calls)
        ns_total_ct = sum(c["completion_tokens"] for c in sd_noskill.calls)
        sk_truncs = len(detect_truncations(sd_skill.calls))
        ns_truncs = len(detect_truncations(sd_noskill.calls))
        sk_kv_drops = detect_kv_drops(sd_skill.timeline) if sd_skill.timeline else "N/A"
        ns_kv_drops = detect_kv_drops(sd_noskill.timeline) if sd_noskill.timeline else "N/A"

        sk_turns = per_turn_stats(sd_skill.calls)
        ns_turns = per_turn_stats(sd_noskill.calls)
        sk_effs = [d["efficiency_ms_per_tok"] for d in sk_turns.values()]
        ns_effs = [d["efficiency_ms_per_tok"] for d in ns_turns.values()]
        sk_var = max(sk_effs) / min(sk_effs) if sk_effs and min(sk_effs) > 0 else 0
        ns_var = max(ns_effs) / min(ns_effs) if ns_effs and min(ns_effs) > 0 else 0

        pt_delta = (sk_total_pt - ns_total_pt) / ns_total_pt * 100 if ns_total_pt > 0 else 0
        elapsed_delta = (sd_skill.total_elapsed - sd_noskill.total_elapsed) / sd_noskill.total_elapsed * 100 if sd_noskill.total_elapsed > 0 else 0

        lines.append(f"\n--- {SEQ_SHORT[sd_skill.key]} ---")
        lines.append(f"  {'Metric':<30} {'No-Skill':>15} {'With-Skill':>15} {'Delta':>10}")
        lines.append(f"  {'-' * 72}")
        lines.append(f"  {'LLM Calls':<30} {len(sd_noskill.calls):>15} {len(sd_skill.calls):>15} {len(sd_skill.calls) - len(sd_noskill.calls):>+10}")
        lines.append(f"  {'Total Prompt Tokens':<30} {ns_total_pt:>15,} {sk_total_pt:>15,} {pt_delta:>+9.0f}%")
        lines.append(f"  {'Total Completion Tokens':<30} {ns_total_ct:>15,} {sk_total_ct:>15,}")
        lines.append(f"  {'Total Elapsed (s)':<30} {sd_noskill.total_elapsed:>15.1f} {sd_skill.total_elapsed:>15.1f} {elapsed_delta:>+9.0f}%")
        lines.append(f"  {'Truncation Events':<30} {ns_truncs:>15} {sk_truncs:>15} {sk_truncs - ns_truncs:>+10}")
        lines.append(f"  {'KV Cache Drops':<30} {str(ns_kv_drops):>15} {str(sk_kv_drops):>15}")
        lines.append(f"  {'Efficiency Variance':<30} {ns_var:>14.1f}x {sk_var:>14.1f}x")

        # LaTeX row
        latex_rows.append(
            f"  {SEQ_SHORT[sd_skill.key]} & "
            f"{len(sd_noskill.calls)}/{len(sd_skill.calls)} & "
            f"{ns_total_pt // 1000}K/{sk_total_pt // 1000}K & "
            f"{pt_delta:+.0f}\\% & "
            f"{sd_noskill.total_elapsed:.0f}s/{sd_skill.total_elapsed:.0f}s & "
            f"{elapsed_delta:+.0f}\\% & "
            f"{ns_truncs}/{sk_truncs} & "
            f"{ns_var:.1f}x/{sk_var:.1f}x \\\\"
        )

    summary_text = "\n".join(lines)
    print(summary_text)

    with open(os.path.join(out_dir, "skill_comparison_stats.txt"), "w") as f:
        f.write(summary_text)

    # LaTeX table
    latex = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Comparison of multi-turn agent workloads with and without skill mechanism "
        "(No-Skill / With-Skill). Skill loading introduces significant token overhead "
        "and additional truncation events.}",
        "\\label{tab:skill_comparison}",
        "\\small",
        "\\begin{tabular}{lccccccc}",
        "\\toprule",
        "Seq & Calls & Prompt Tok & $\\Delta$PT & Elapsed & $\\Delta$Time & "
        "Truncs & Eff. Var. \\\\",
        "\\midrule",
    ]
    latex.extend(latex_rows)
    latex.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    with open(os.path.join(out_dir, "table_skill_comparison.tex"), "w") as f:
        f.write("\n".join(latex))

    print(f"\n[OK] skill_comparison_stats.txt + table_skill_comparison.tex")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Skill vs No-Skill comparison figures for SegKV paper")
    parser.add_argument("--results-dir",
                        default="results/03-8B/multurn_bench",
                        help="Root results directory")
    parser.add_argument("--out-dir",
                        default="results/03-8B/multurn_bench/segkv_figures",
                        help="Output directory for figures")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 这里先写死，后面也可以再改成自动扫描 ctx_*
    context_lengths = [16384, 32768]

    # 1) 给 Fig C1 用：收集所有 variant
    all_variants = []
    for seq_key in SEQUENCES:
        for ctx in context_lengths:
            for noskill in [False, True]:
                sd = load_seq(args.results_dir, seq_key, context_len=ctx, noskill=noskill)
                if sd is not None:
                    all_variants.append(sd)

    grouped_variants = group_variants_by_sequence(all_variants)

    if grouped_variants:
        print(f"Loaded Fig C1 grouped variants: {list(grouped_variants.keys())}")
        figC1_prompt_growth_comparison(grouped_variants, args.out_dir)
    else:
        print("[WARN] No grouped variants found for Fig C1")

    # 2) 旧的 C2~C6 先继续按单一 ctx 逻辑保留，后面再统一升级
    pairs = []
    single_ctx_for_old_figs = 32768
    for seq_key in SEQUENCES:
        sd_skill = load_seq(args.results_dir, seq_key, context_len=single_ctx_for_old_figs, noskill=False)
        sd_noskill = load_seq(args.results_dir, seq_key, context_len=single_ctx_for_old_figs, noskill=True)
        if sd_skill and sd_noskill:
            pairs.append((sd_skill, sd_noskill))

    if not pairs:
        print("ERROR: no complete (skill, no-skill) pairs found", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(pairs)} comparison pairs for legacy figures: "
          f"{[SEQ_SHORT[p[0].key] for p in pairs]}\n")

    figC2_aggregate_comparison(grouped_variants, args.out_dir)
    figC3_turn_efficiency_comparison(grouped_variants, args.out_dir)
    # figC4_kv_cache_comparison(pairs, args.out_dir)
    # figC5_skill_overhead_waterfall(pairs, args.out_dir)
    # figC6_prompt_kv_overlay(pairs, args.out_dir)
    print()
    # print_comparison_summary(pairs, args.out_dir)

if __name__ == "__main__":
    main()
