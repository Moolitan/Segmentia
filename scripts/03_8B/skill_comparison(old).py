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

SEQUENCES = ["T4_B", "T4_F", "T6_F"]
SEQ_LABELS = {
    "T4_B": "T4-B (Excel→Word→PPT→PDF)",
    "T4_F": "T4-F (Frontend→Vercel→Word→PDF)",
    "T6_F": "T6-F (Find→Create→Frontend→Word→PPT→PDF)",
}
SEQ_SHORT = {"T4_B": "T4-B", "T4_F": "T4-F", "T6_F": "T6-F"}

# Paired colors: darker for with-skill, lighter for no-skill
COLORS_SKILL = {"T4_B": "#2563EB", "T4_F": "#DC2626", "T6_F": "#059669"}
COLORS_NOSKILL = {"T4_B": "#93C5FD", "T4_F": "#FCA5A5", "T6_F": "#6EE7B7"}


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


def load_seq(results_dir: str, seq_key: str, noskill: bool = False) -> SeqData | None:
    suffix = "_noskill" if noskill else ""
    path = os.path.join(results_dir, f"{seq_key}{suffix}",
                        "multiturn_sequence_traces.json")
    if not os.path.exists(path):
        print(f"[SKIP] {path} not found")
        return None
    with open(path) as f:
        data = json.load(f)
    s = data["sequence"]
    group = "no-skill" if noskill else "with-skill"
    sd = SeqData(
        key=seq_key,
        label=f"{SEQ_LABELS.get(seq_key, seq_key)} [{group}]",
        color=(COLORS_NOSKILL if noskill else COLORS_SKILL).get(seq_key, "#333"),
        num_turns=s["num_turns"],
        total_elapsed=s["total_elapsed_seconds"],
        calls=s["llm_calls"],
        timeline=s.get("vllm_timeline", []),
        no_skills=noskill,
    )
    return sd


# ── Helpers ───────────────────────────────────────────────────────────────────

def per_turn_stats(calls: list[dict]) -> dict:
    turns = {}
    for c in calls:
        t = c["turn_number"]
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


# ── Figure C1: Side-by-Side Prompt Token Growth ──────────────────────────────

def figC1_prompt_growth_comparison(pairs: list[tuple[SeqData, SeqData]], out_dir: str):
    """Side-by-side comparison of prompt token growth: skill vs no-skill."""
    n = len(pairs)
    fig, axes = plt.subplots(n, 2, figsize=(14, 3.5 * n), sharex=False)
    if n == 1:
        axes = [axes]

    for row, (sd_skill, sd_noskill) in enumerate(pairs):
        for col, sd in enumerate([sd_noskill, sd_skill]):
            ax = axes[row][col] if n > 1 else axes[col]
            calls = sd.calls
            xs = list(range(len(calls)))
            pts = [c["prompt_tokens"] for c in calls]
            bounds = get_turn_boundaries(calls)
            truncs = detect_truncations(calls)

            ax.plot(xs, pts, color=sd.color, linewidth=1.8, marker="o",
                    markersize=3, zorder=3)

            for bx in bounds:
                ax.axvline(x=bx - 0.5, color="#94A3B8", linestyle="--",
                           linewidth=0.8, alpha=0.6)

            for tr in truncs:
                ax.axvspan(tr["index"] - 0.5, tr["index"] + 0.5,
                           alpha=0.15, color="#DC2626", zorder=1)
                ax.annotate(
                    f"TRUNC\n{tr['before'] // 1000}K→{tr['after'] // 1000}K",
                    xy=(tr["index"], tr["after"]),
                    xytext=(tr["index"] + 1.5, tr["after"] + 4000),
                    fontsize=7, color="#DC2626", fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="#DC2626", lw=0.8),
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#FEE2E2",
                              edgecolor="#DC2626", alpha=0.8),
                    zorder=5,
                )

            ax.axhline(y=32768, color="#EF4444", linestyle=":", linewidth=0.8,
                       alpha=0.4)

            total_pt = sum(pts)
            group = "No-Skill" if sd.no_skills else "With-Skill"
            ax.set_title(f"{SEQ_SHORT[sd.key]} [{group}] — "
                         f"{len(calls)} calls, {total_pt // 1000}K total prompt tok",
                         fontsize=10, fontweight="bold")
            ax.set_ylabel("Prompt Tokens")
            ax.yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f}K"))
            ax.grid(True, alpha=0.2, linewidth=0.5)

    for ax_row in (axes if n > 1 else [axes]):
        if isinstance(ax_row, np.ndarray):
            ax_row[-1].set_xlabel("LLM Call Index")
            ax_row[0].set_xlabel("LLM Call Index")
        else:
            ax_row.set_xlabel("LLM Call Index")

    fig.suptitle("Fig C1: Prompt Token Growth — No-Skill (left) vs With-Skill (right)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(out_dir, f"figC1_prompt_growth_comparison.{ext}"),
                    bbox_inches="tight")
    plt.close(fig)
    print("[OK] figC1_prompt_growth_comparison")


# ── Figure C2: Aggregate Comparison Bar Chart ────────────────────────────────

def figC2_aggregate_comparison(pairs: list[tuple[SeqData, SeqData]], out_dir: str):
    """Bar chart comparing total prompt tokens, total latency, LLM calls,
    and truncation count between skill and no-skill groups."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    seq_labels = [SEQ_SHORT[p[0].key] for p in pairs]
    x = np.arange(len(pairs))
    w = 0.35

    # Collect metrics
    metrics = {
        "total_prompt": ([], []),      # (skill, noskill)
        "total_elapsed": ([], []),
        "num_calls": ([], []),
        "num_truncations": ([], []),
    }
    for sd_skill, sd_noskill in pairs:
        metrics["total_prompt"][0].append(
            sum(c["prompt_tokens"] for c in sd_skill.calls))
        metrics["total_prompt"][1].append(
            sum(c["prompt_tokens"] for c in sd_noskill.calls))
        metrics["total_elapsed"][0].append(sd_skill.total_elapsed)
        metrics["total_elapsed"][1].append(sd_noskill.total_elapsed)
        metrics["num_calls"][0].append(len(sd_skill.calls))
        metrics["num_calls"][1].append(len(sd_noskill.calls))
        metrics["num_truncations"][0].append(
            len(detect_truncations(sd_skill.calls)))
        metrics["num_truncations"][1].append(
            len(detect_truncations(sd_noskill.calls)))

    configs = [
        (axes[0, 0], "Total Prompt Tokens", "total_prompt",
         lambda v, _: f"{v / 1000:.0f}K", "(a)"),
        (axes[0, 1], "Total Elapsed Time (s)", "total_elapsed", None, "(b)"),
        (axes[1, 0], "Number of LLM Calls", "num_calls", None, "(c)"),
        (axes[1, 1], "Truncation Events", "num_truncations", None, "(d)"),
    ]

    for ax, ylabel, key, fmt, panel in configs:
        skill_vals = metrics[key][0]
        noskill_vals = metrics[key][1]

        bars_ns = ax.bar(x - w / 2, noskill_vals, w, color="#94A3B8",
                         alpha=0.85, label="No-Skill", zorder=3)
        bars_sk = ax.bar(x + w / 2, skill_vals, w, color="#2563EB",
                         alpha=0.85, label="With-Skill", zorder=3)

        # Annotate delta
        for i, (sv, nv) in enumerate(zip(skill_vals, noskill_vals)):
            if nv > 0:
                delta = (sv - nv) / nv * 100
                sign = "+" if delta > 0 else ""
                color = "#DC2626" if delta > 0 else "#059669"
                y_top = max(sv, nv)
                ax.text(i, y_top * 1.05, f"{sign}{delta:.0f}%",
                        ha="center", va="bottom", fontsize=9,
                        fontweight="bold", color=color)

        ax.set_xticks(x)
        ax.set_xticklabels(seq_labels)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{panel} {ylabel}", fontweight="bold")
        if fmt:
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt))
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(True, alpha=0.2, linewidth=0.5, axis="y")

    fig.suptitle("Fig C2: Skill vs No-Skill — Aggregate Metrics Comparison",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(out_dir, f"figC2_aggregate_comparison.{ext}"),
                    bbox_inches="tight")
    plt.close(fig)
    print("[OK] figC2_aggregate_comparison")


# ── Figure C3: Per-Turn Efficiency Comparison ─────────────────────────────────

def figC3_turn_efficiency_comparison(pairs: list[tuple[SeqData, SeqData]],
                                      out_dir: str):
    """Grouped bar chart: per-turn ms/tok for each sequence, skill vs no-skill."""
    n = len(pairs)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, (sd_skill, sd_noskill) in zip(axes, pairs):
        turns_sk = per_turn_stats(sd_skill.calls)
        turns_ns = per_turn_stats(sd_noskill.calls)
        max_turns = max(max(turns_sk.keys(), default=0),
                        max(turns_ns.keys(), default=0))
        t_range = range(1, max_turns + 1)
        w = 0.35

        eff_sk = [turns_sk[t]["efficiency_ms_per_tok"] if t in turns_sk else 0
                  for t in t_range]
        eff_ns = [turns_ns[t]["efficiency_ms_per_tok"] if t in turns_ns else 0
                  for t in t_range]
        x = np.arange(len(list(t_range)))

        bars_ns = ax.bar(x - w / 2, eff_ns, w, color="#94A3B8", alpha=0.85,
                         label="No-Skill", zorder=3)
        bars_sk = ax.bar(x + w / 2, eff_sk, w,
                         color=COLORS_SKILL[sd_skill.key], alpha=0.85,
                         label="With-Skill", zorder=3)

        # Annotate values
        for bar_arr, vals in [(bars_ns, eff_ns), (bars_sk, eff_sk)]:
            for bar, val in zip(bar_arr, vals):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.1,
                            f"{val:.1f}", ha="center", va="bottom",
                            fontsize=7, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels([f"T{t}" for t in t_range])
        ax.set_xlabel("Turn")
        ax.set_title(SEQ_SHORT[sd_skill.key], fontweight="bold")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.2, linewidth=0.5, axis="y")

    axes[0].set_ylabel("Efficiency (ms/tok) — lower is better")
    fig.suptitle("Fig C3: Per-Turn Inference Efficiency — Skill vs No-Skill",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(out_dir, f"figC3_turn_efficiency_comparison.{ext}"),
                    bbox_inches="tight")
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
                        default="results/03/multurn_bench",
                        help="Root results directory")
    parser.add_argument("--out-dir",
                        default="results/03/multurn_bench/segkv_figures",
                        help="Output directory for figures")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load paired data: (with-skill, no-skill) for each sequence
    pairs = []
    for seq_key in SEQUENCES:
        sd_skill = load_seq(args.results_dir, seq_key, noskill=False)
        sd_noskill = load_seq(args.results_dir, seq_key, noskill=True)
        if sd_skill and sd_noskill:
            pairs.append((sd_skill, sd_noskill))
        else:
            missing = []
            if not sd_skill:
                missing.append("with-skill")
            if not sd_noskill:
                missing.append("no-skill")
            print(f"[WARN] {seq_key}: missing {', '.join(missing)} data, skipping pair")

    if not pairs:
        print("ERROR: no complete (skill, no-skill) pairs found", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(pairs)} comparison pairs: "
          f"{[SEQ_SHORT[p[0].key] for p in pairs]}\n")

    # Generate all comparison figures
    figC1_prompt_growth_comparison(pairs, args.out_dir)
    figC2_aggregate_comparison(pairs, args.out_dir)
    figC3_turn_efficiency_comparison(pairs, args.out_dir)
    figC4_kv_cache_comparison(pairs, args.out_dir)
    figC5_skill_overhead_waterfall(pairs, args.out_dir)
    figC6_prompt_kv_overlay(pairs, args.out_dir)
    print()
    print_comparison_summary(pairs, args.out_dir)


if __name__ == "__main__":
    main()
