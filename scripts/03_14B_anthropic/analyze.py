"""Analyze a multi-Turn trace produced by run_mulTurn.py.

Consumes ``multiTurn_sequence_traces.json`` and emits:

* A text summary on stdout.
* PNG figures under ``--out-dir`` (defaults to ``<input_dir>/figures``).

核心问题：当 SkillTool 第 N 次返回 skill 文档时，vLLM 是否重新做了 prefill？
判据：prefill_tokens = prompt_tokens - hits_tokens（真实算力信号）
位置判据已废弃。

Usage::

    python analyze.py \\
        --input results/03_14B_anthropic/<repo>/multiTurn_sequence_traces.json
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ─── helpers ────────────────────────────────────────────────────────────────

def _load(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _prefill_tokens(call: dict) -> int | None:
    """实际需要 prefill 的 token 数 = prompt_tokens − hits_tokens。"""
    pt = call.get("prompt_tokens")
    ht = call.get("vllm_prefix_cache_hits_tokens")
    if pt is None or ht is None:
        return None
    return max(0, pt - ht)


def _delta_new(calls: list[dict], idx: int) -> int | None:
    """本次 call 相比上一次 call 新增的 prompt token 数。"""
    if idx == 0:
        return None
    cur = calls[idx].get("prompt_tokens")
    prev = calls[idx - 1].get("prompt_tokens")
    if cur is None or prev is None:
        return None
    return max(0, cur - prev)


def _detect_skill_loads(calls: list[dict]) -> list[int]:
    """
    检测 SkillTool 被调用的 call 序号。
    启发式：该 call 的 prefill_tokens 远大于 Δnew（新内容小但 prefill 大）
    以及/或者 hit_rate 突然下降。
    返回 call 索引列表。
    """
    skill_load_indices = []
    for i, c in enumerate(calls):
        pf = _prefill_tokens(c)
        dn = _delta_new(calls, i)
        if pf is None or dn is None:
            continue
        # prefill 超出新增量 1000 token 以上，视为 skill 被重新加载/cache 断裂
        waste = pf - dn
        if waste > 1000:
            skill_load_indices.append(i)
    return skill_load_indices


# ─── text summary ────────────────────────────────────────────────────────────

def _print_summary(data: dict[str, Any]) -> None:
    repo = data.get("benchmark_repo", "?")
    calls = data.get("llm_calls", []) or []

    print(f"\n== {repo} ==")
    print(f"  LLM calls total : {len(calls)}")

    Turns = sorted({c.get("Turn_number") for c in calls if c.get("Turn_number") is not None})
    print(f"  Turns           : {len(Turns)}  {Turns}")

    # 1. prefill vs Δnew 浪费统计
    total_prefill = 0
    total_new = 0
    total_waste = 0
    waste_calls = 0
    for i, c in enumerate(calls):
        pf = _prefill_tokens(c)
        dn = _delta_new(calls, i)
        if pf is None:
            continue
        total_prefill += pf
        if dn is not None:
            total_new += dn
            w = pf - dn
            if w > 0:
                total_waste += w
                waste_calls += 1

    print(f"\n  [Waste analysis]")
    print(f"  total prefill_tokens (actual compute)  : {total_prefill:,}")
    print(f"  total delta_new_tokens (ideal minimum) : {total_new:,}")
    print(f"  over-computed tokens (waste)            : {total_waste:,}  ({waste_calls} calls)")
    if total_prefill > 0:
        print(f"  waste ratio                      : {total_waste/total_prefill*100:.1f}%")

    # 2. prefill 时间
    prefill_times = [c.get("vllm_request_prefill_time_seconds") for c in calls
                     if c.get("vllm_request_prefill_time_seconds") is not None]
    if prefill_times:
        total_pf_time = sum(prefill_times)
        print(f"\n  [prefill time]")
        print(f"  total   : {total_pf_time:.2f}s")
        print(f"  mean    : {total_pf_time/len(prefill_times):.3f}s")
        print(f"  max     : {max(prefill_times):.3f}s")

    # 3. hit rate
    hit_rates = [c.get("vllm_prefix_cache_hit_rate") for c in calls
                 if c.get("vllm_prefix_cache_hit_rate") is not None]
    if hit_rates:
        print(f"\n  [prefix cache hit_rate]")
        print(f"  mean={sum(hit_rates)/len(hit_rates):.3f}  "
              f"min={min(hit_rates):.3f}  max={max(hit_rates):.3f}")

    # 4. TTFT
    ttfts = [c.get("vllm_time_to_first_token_seconds") for c in calls
             if c.get("vllm_time_to_first_token_seconds") is not None]
    if ttfts:
        print(f"\n  [TTFT]")
        print(f"  mean={sum(ttfts)/len(ttfts):.3f}s  "
              f"min={min(ttfts):.3f}s  max={max(ttfts):.3f}s")

    # 5. skill 副本数（old analysis, kept as text）
    analysis = data.get("skill_doc_analysis") or {}
    if analysis.get("skills"):
        print(f"\n  [skill KV analysis (old position heuristic, reference only)]")
        for sname, sdata in analysis["skills"].items():
            s = sdata.get("summary", {})
            total_present = s.get("first_use", 0) + s.get("exact_repeat", 0) + s.get("prefix_broken_repeat", 0)
            print(f"  skill '{sname}': present {total_present} times  "
                  f"(first={s.get('first_use',0)} exact={s.get('exact_repeat',0)} "
                  f"broken={s.get('prefix_broken_repeat',0)})")


# ─── figure 1: Δnew vs prefill 浪费 ─────────────────────────────────────────

def _plot_waste_analysis(data: dict[str, Any], out_path: str) -> None:
    """
    条形图：每 call 的 prefill_tokens（蓝）和 Δnew_tokens（绿）。
    二者之差（红色填充）= 本次多算的浪费量。
    Turn 边界用竖虚线标注。
    """
    calls = data.get("llm_calls", []) or []
    if not calls:
        reTurn

    n = len(calls)
    xs = np.arange(n)
    pf_vals = [(_prefill_tokens(c) or 0) for c in calls]
    dn_vals = [(_delta_new(calls, i) if i > 0 else 0) or 0 for i, c in enumerate(calls)]

    fig, ax = plt.subplots(figsize=(max(12, n * 0.3), 5))

    # 浪费区域（prefill 超出 Δnew 的部分）
    waste_bottoms = dn_vals
    waste_heights = [max(0, p - d) for p, d in zip(pf_vals, dn_vals)]
    ax.bar(xs, waste_heights, bottom=waste_bottoms,
           color="tab:red", alpha=0.5, label="Waste (prefill - delta_new)", zorder=2)
    ax.bar(xs, dn_vals, color="tab:green", alpha=0.7, label="delta_new tokens (ideal minimum)", zorder=3)

    # prefill total line
    ax.plot(xs, pf_vals, "-o", color="tab:blue", markersize=3, linewidth=1.5,
            label="actual prefill_tokens", zorder=4)

    # Turn boundaries
    prev_Turn = None
    for i, c in enumerate(calls):
        t = c.get("Turn_number")
        if prev_Turn is not None and t != prev_Turn:
            ax.axvline(i - 0.5, color="black", linestyle=":", alpha=0.4)
            ax.text(i - 0.5, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else 1,
                    f"T{t}", fontsize=7, ha="center", color="black")
        prev_Turn = t

    ax.set_xlabel("LLM call index")
    ax.set_ylabel("tokens")
    ax.set_title(f"{data.get('benchmark_repo', '?')}: Prefill waste analysis (prefill_tokens vs delta_new)")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ─── figure 2: prefill spike + skill load 对齐 ───────────────────────────────

def _plot_prefill_spike_with_skill_events(data: dict[str, Any], out_path: str) -> None:
    """
    双轴图：
    - 上：prefill_time_seconds（条形），spike 对应 skill 重加载事件打红标
    - 下：hit_rate 折线，展示 cache 断裂时机
    Turn 边界虚线贯穿两子图。
    """
    calls = data.get("llm_calls", []) or []
    if not calls:
        reTurn

    n = len(calls)
    xs = np.arange(n)
    pf_times = [c.get("vllm_request_prefill_time_seconds") or 0.0 for c in calls]
    hit_rates = [c.get("vllm_prefix_cache_hit_rate") or 0.0 for c in calls]
    waste_indices = _detect_skill_loads(calls)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(12, n * 0.3), 7),
                                    sharex=True, gridspec_kw={"height_ratios": [3, 1]})

    # 上图：prefill 时间
    colors = ["tab:red" if i in waste_indices else "tab:blue" for i in range(n)]
    ax1.bar(xs, pf_times, color=colors, alpha=0.8)
    ax1.set_ylabel("prefill_time (s)")
    ax1.set_title(f"{data.get('benchmark_repo', '?')}: Prefill spike aligned with skill reload events\n"
                  f"(red = suspected skill reload: prefill > delta_new by >1000 tokens)")
    ax1.grid(True, axis="y", alpha=0.3)

    # label spikes
    for i in waste_indices:
        pf = _prefill_tokens(calls[i]) or 0
        dn = _delta_new(calls, i) or 0
        waste = pf - dn
        ax1.text(i, pf_times[i], f"+{waste//1000}k\nwaste", ha="center", va="bottom",
                 fontsize=6.5, color="tab:red", fontweight="bold")

    red_patch = mpatches.Patch(color="tab:red", alpha=0.8, label="suspected skill reload")
    blue_patch = mpatches.Patch(color="tab:blue", alpha=0.8, label="normal call")
    ax1.legend(handles=[red_patch, blue_patch], fontsize=8)

    # lower panel: hit_rate
    ax2.plot(xs, hit_rates, "-", color="tab:orange", linewidth=1.5)
    ax2.fill_between(xs, 0, hit_rates, color="tab:orange", alpha=0.15)
    ax2.set_ylabel("hit_rate")
    ax2.set_ylim(0, 1.05)
    ax2.set_xlabel("LLM call index")
    ax2.grid(True, axis="y", alpha=0.3)

    # Turn 边界
    prev_Turn = None
    for i, c in enumerate(calls):
        t = c.get("Turn_number")
        if prev_Turn is not None and t != prev_Turn:
            ax1.axvline(i - 0.5, color="black", linestyle=":", alpha=0.35)
            ax2.axvline(i - 0.5, color="black", linestyle=":", alpha=0.35)
            ax1.text(i - 0.5, max(pf_times) * 0.97 if max(pf_times) > 0 else 0.01,
                     f"T{t}", fontsize=7, ha="center", color="black")
        prev_Turn = t

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ─── figure 3: skill 副本数递增曲线 ─────────────────────────────────────────

def _plot_skill_copy_count(data: dict[str, Any], out_path: str) -> None:
    """
    检测 skill 文档在 prompt 中出现的次数（字符串搜索），画出随 call 序号的累计副本数。
    每出现一次 spike（prompt 变长但 hits 没跟上），视为又加载了一份。
    实际做法：统计每个 call 的 request_prompt_text 中 skill 文档出现次数。
    """
    calls = data.get("llm_calls", []) or []
    analysis = data.get("skill_doc_analysis") or {}
    skills_meta = analysis.get("skills", {}) or {}

    if not calls or not skills_meta:
        # fallback: 只画浪费累计曲线
        _plot_waste_cumulative(data, out_path)
        reTurn

    n = len(calls)
    xs = np.arange(n)

    fig, ax = plt.subplots(figsize=(max(12, n * 0.3), 5))

    cmap_colors = ["tab:orange", "tab:purple", "tab:brown", "tab:pink", "tab:cyan"]
    for k, (skill_name, _) in enumerate(skills_meta.items()):
        copy_counts = []
        for c in calls:
            skill_info = (c.get("skills") or {}).get(skill_name, {})
            present = skill_info.get("present", False)
            copy_counts.append(1 if present else 0)

        # 累计副本数（每次 first_use 或 prefix_broken_repeat = 新副本）
        cumulative_copies = []
        total = 0
        for i, c in enumerate(calls):
            skill_info = (c.get("skills") or {}).get(skill_name, {})
            label = skill_info.get("label", "not_present")
            if label in ("first_use", "prefix_broken_repeat"):
                total += 1
            cumulative_copies.append(total)

        color = cmap_colors[k % len(cmap_colors)]
        ax.step(xs, cumulative_copies, where="post", color=color, linewidth=2,
                label=f"skill: {skill_name}")
        # 在新副本加入的时刻打点
        for i, cnt in enumerate(cumulative_copies):
            if i > 0 and cnt > cumulative_copies[i - 1]:
                ax.scatter(i, cnt, color=color, s=60, zorder=5)

    # Turn 边界
    prev_Turn = None
    for i, c in enumerate(calls):
        t = c.get("Turn_number")
        if prev_Turn is not None and t != prev_Turn:
            ax.axvline(i - 0.5, color="black", linestyle=":", alpha=0.4)
            ax.text(i - 0.5, ax.get_ylim()[1] * 0.98 if ax.get_ylim()[1] > 0 else 0.01,
                    f"T{t}", fontsize=7, ha="center")
        prev_Turn = t

    ax.set_xlabel("LLM call index")
    ax.set_ylabel("cumulative skill copies")
    ax.set_title(f"{data.get('benchmark_repo', '?')}: Skill copy count growth per call\n"
                 f"(each dot = new copy inserted; slope = reload frequency)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_waste_cumulative(data: dict[str, Any], out_path: str) -> None:
    """Fallback: cumulative wasted tokens (no skill info available)。"""
    calls = data.get("llm_calls", []) or []
    n = len(calls)
    xs = np.arange(n)

    cumulative_waste = []
    total = 0
    for i, c in enumerate(calls):
        pf = _prefill_tokens(c)
        dn = _delta_new(calls, i)
        if pf is not None and dn is not None:
            total += max(0, pf - dn)
        cumulative_waste.append(total)

    fig, ax = plt.subplots(figsize=(max(10, n * 0.25), 4))
    ax.plot(xs, cumulative_waste, "-", color="tab:red", linewidth=2)
    ax.fill_between(xs, 0, cumulative_waste, color="tab:red", alpha=0.1)
    ax.set_xlabel("LLM call index")
    ax.set_ylabel("cumulative wasted tokens")
    ax.set_title(f"{data.get('benchmark_repo', '?')}: Cumulative over-computed tokens (prefill - delta_new)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ─── figure 4: what-if 节省估算 ──────────────────────────────────────────────

def _plot_whatif_savings(data: dict[str, Any], out_path: str) -> None:
    """
    What-if 分析：假设 skill 只加载一次（KV 完全复用），总 prefill 时间会节省多少？
    - 找出所有"疑似 skill 重加载"的 call（prefill - Δnew > 1000）
    - 估算如果那部分多算的 tokens 不用 prefill，对应节省的时间
    - 画条形图：实际 vs what-if 理想 prefill 时间
    """
    calls = data.get("llm_calls", []) or []
    pf_times = [c.get("vllm_request_prefill_time_seconds") or 0.0 for c in calls]

    if not pf_times or not any(pf_times):
        reTurn

    # 估算 prefill 速度：tokens/second（用非浪费 call 的样本）
    speeds = []
    for i, c in enumerate(calls):
        pf_tok = _prefill_tokens(c)
        pf_time = c.get("vllm_request_prefill_time_seconds")
        dn = _delta_new(calls, i)
        if pf_tok and pf_time and dn is not None:
            waste = pf_tok - dn
            if waste <= 500:  # 基本正常的 call
                speeds.append(pf_tok / pf_time)
    avg_speed = sum(speeds) / len(speeds) if speeds else None

    # 构建 what-if 时间序列
    whatif_times = []
    for i, c in enumerate(calls):
        pf_tok = _prefill_tokens(c)
        pf_time = c.get("vllm_request_prefill_time_seconds") or 0.0
        dn = _delta_new(calls, i)
        if pf_tok and dn is not None and avg_speed:
            waste_tok = max(0, pf_tok - dn)
            saved_time = waste_tok / avg_speed
            whatif_times.append(max(0, pf_time - saved_time))
        else:
            whatif_times.append(pf_time)

    actual_total = sum(pf_times)
    ideal_total = sum(whatif_times)
    saved = actual_total - ideal_total

    n = len(calls)
    xs = np.arange(n)

    fig, ax = plt.subplots(figsize=(max(12, n * 0.3), 5))
    ax.bar(xs, pf_times, color="tab:red", alpha=0.6, label=f"Actual prefill_time (total {actual_total:.1f}s)")
    ax.bar(xs, whatif_times, color="tab:green", alpha=0.8,
           label=f"What-if ideal time (total {ideal_total:.1f}s, saves {saved:.1f}s)")

    ax.set_xlabel("LLM call index")
    ax.set_ylabel("prefill_time (s)")
    ax.set_title(f"{data.get('benchmark_repo', '?')}: What-if: prefill time saved with skill KV reuse\n"
                 f"(avg_prefill_speed = {avg_speed:.0f} tokens/s)" if avg_speed else
                 f"{data.get('benchmark_repo', '?')}: What-if: prefill time saved with skill KV reuse")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ─── figure 5: TTFT 分布对比 ─────────────────────────────────────────────────

def _plot_ttft_distribution(data: dict[str, Any], out_path: str) -> None:
    """
    箱线图对比：skill 重加载 call vs 正常 call 的 TTFT 分布。
    """
    calls = data.get("llm_calls", []) or []
    waste_indices = set(_detect_skill_loads(calls))

    ttft_waste = [c.get("vllm_time_to_first_token_seconds")
                  for i, c in enumerate(calls)
                  if i in waste_indices and c.get("vllm_time_to_first_token_seconds") is not None]
    ttft_normal = [c.get("vllm_time_to_first_token_seconds")
                   for i, c in enumerate(calls)
                   if i not in waste_indices and c.get("vllm_time_to_first_token_seconds") is not None]

    if not ttft_waste and not ttft_normal:
        reTurn

    fig, ax = plt.subplots(figsize=(6, 5))
    data_groups = []
    labels = []
    if ttft_normal:
        data_groups.append(ttft_normal)
        labels.append(f"normal call\n(n={len(ttft_normal)})")
    if ttft_waste:
        data_groups.append(ttft_waste)
        labels.append(f"suspected reload\n(n={len(ttft_waste)})")

    bp = ax.boxplot(data_groups, labels=labels, patch_artist=True,
                    medianprops={"color": "black", "linewidth": 2})
    colors = ["tab:blue", "tab:red"]
    for patch, color in zip(bp["boxes"], colors[:len(bp["boxes"])]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel("TTFT (s)")
    ax.set_title(f"{data.get('benchmark_repo', '?')}: TTFT distribution: normal vs suspected skill reload calls")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ─── figure 6: Turn 维度聚合 ─────────────────────────────────────────────────

def _plot_Turn_aggregation(data: dict[str, Any], out_path: str) -> None:
    """
    每 Turn 的聚合：total prefill_tokens、total waste_tokens、skill reload count。
    三个子图纵向排列。
    """
    calls = data.get("llm_calls", []) or []
    if not calls:
        reTurn

    waste_indices = set(_detect_skill_loads(calls))
    Turns = sorted({c.get("Turn_number") for c in calls if c.get("Turn_number") is not None})

    Turn_prefill = {t: 0 for t in Turns}
    Turn_waste = {t: 0 for t in Turns}
    Turn_skill_loads = {t: 0 for t in Turns}
    Turn_pf_time = {t: 0.0 for t in Turns}

    for i, c in enumerate(calls):
        t = c.get("Turn_number")
        if t is None:
            continue
        pf = _prefill_tokens(c) or 0
        dn = _delta_new(calls, i) or 0
        Turn_prefill[t] += pf
        Turn_waste[t] += max(0, pf - dn)
        if i in waste_indices:
            Turn_skill_loads[t] += 1
        pf_time = c.get("vllm_request_prefill_time_seconds") or 0.0
        Turn_pf_time[t] += pf_time

    xs = np.arange(len(Turns))
    labels = [f"T{t}" for t in Turns]

    fig, axes = plt.subplots(3, 1, figsize=(max(8, len(Turns) * 0.8), 9), sharex=True)

    axes[0].bar(xs, [Turn_prefill[t] for t in Turns], color="tab:blue", alpha=0.8)
    axes[0].set_ylabel("prefill_tokens")
    axes[0].set_title(f"{data.get('benchmark_repo', '?')}: Per-Turn aggregated stats")
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].bar(xs, [Turn_waste[t] for t in Turns], color="tab:red", alpha=0.8)
    axes[1].set_ylabel("wasted tokens")
    axes[1].grid(True, axis="y", alpha=0.3)

    axes[2].bar(xs, [Turn_skill_loads[t] for t in Turns], color="tab:orange", alpha=0.8)
    axes[2].set_ylabel("skill reload count")
    axes[2].set_xlabel("Turn")
    axes[2].grid(True, axis="y", alpha=0.3)

    plt.xticks(xs, labels)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ─── main entry ──────────────────────────────────────────────────────────────

def analyze_one(input_path: str, out_dir: str) -> None:
    data = _load(input_path)
    os.makedirs(out_dir, exist_ok=True)

    _print_summary(data)

    # 核心三图（按建议优先级）
    _plot_waste_analysis(data, os.path.join(out_dir, "1_waste_prefill_vs_delta_new.png"))
    _plot_prefill_spike_with_skill_events(data, os.path.join(out_dir, "2_prefill_spike_skill_aligned.png"))
    _plot_skill_copy_count(data, os.path.join(out_dir, "3_skill_copy_count.png"))

    # 补充分析
    _plot_whatif_savings(data, os.path.join(out_dir, "4_whatif_savings.png"))
    _plot_ttft_distribution(data, os.path.join(out_dir, "5_ttft_distribution.png"))
    _plot_Turn_aggregation(data, os.path.join(out_dir, "6_Turn_aggregation.png"))

    print(f"  figures -> {out_dir}")
    print(f"    1_waste_prefill_vs_delta_new.png  <- core: waste quantification")
    print(f"    2_prefill_spike_skill_aligned.png <- core: when waste happens")
    print(f"    3_skill_copy_count.png            <- core: why waste (copy count)")
    print(f"    4_whatif_savings.png              <- what-if savings")
    print(f"    5_ttft_distribution.png           <- TTFT impact of skill reload")
    print(f"    6_Turn_aggregation.png            <- per-turn aggregation")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="multiTurn_sequence_traces.json")
    parser.add_argument(
        "--results-dir",
        help="scan for <repo>/multiturn_sequence_traces.json under this dir",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="figures output dir, default <input_dir>/figures",
    )
    args = parser.parse_args()

    if args.input:
        out_dir = args.out_dir or os.path.join(os.path.dirname(args.input), "figures")
        analyze_one(args.input, out_dir)
    elif args.results_dir:
        pattern = os.path.join(args.results_dir, "*", "multiTurn_sequence_traces.json")
        paths = sorted(glob.glob(pattern))
        if not paths:
            print(f"[WARN] no files match {pattern}")
            reTurn
        for p in paths:
            out = args.out_dir or os.path.join(os.path.dirname(p), "figures")
            analyze_one(p, out)
    else:
        parser.error("must pass --input or --results-dir")


if __name__ == "__main__":
    main()