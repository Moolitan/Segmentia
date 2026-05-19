#!/usr/bin/env python3
"""
SegKV Paper — Motivation / Characterization Figures.

Generates publication-quality figures from T4_B, T4_F, T6_F experiment traces
to illustrate the three core inefficiencies:
  1. Context accumulation overhead & catastrophic truncation
  2. KV cache sawtooth pattern with drops
  3. Prefix cache hit rate dynamics at turn boundaries
  4. Per-turn efficiency variance (ms/tok)
  5. Context segment composition breakdown
  6. Summary statistics table (LaTeX)

Usage:
  python scripts/03/segkv_characterization.py \
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

COLORS = {
    "T4_B": "#2563EB",   # blue
    "T4_F": "#DC2626",   # red
    "T6_F": "#059669",   # green
}
SEQ_LABELS = {
    "T4_B": "T4-B (Excel→Word→PPT→PDF)",
    "T4_F": "T4-F (Frontend→Vercel→Word→PDF)",
    "T6_F": "T6-F (Find→Create→Frontend→Word→PPT→PDF)",
}
SEQUENCES = ["T4_B", "T4_F", "T6_F"]


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


def load_seq(results_dir: str, seq_key: str) -> SeqData | None:
    path = os.path.join(results_dir, seq_key, "multiturn_sequence_traces.json")
    if not os.path.exists(path):
        print(f"[SKIP] {path} not found")
        return None
    with open(path) as f:
        data = json.load(f)
    s = data["sequence"]
    sd = SeqData(
        key=seq_key,
        label=SEQ_LABELS.get(seq_key, seq_key),
        color=COLORS.get(seq_key, "#333333"),
        num_turns=s["num_turns"],
        total_elapsed=s["total_elapsed_seconds"],
        calls=s["llm_calls"],
        timeline=s.get("vllm_timeline", []),
    )
    return sd


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_turn_boundaries(calls: list[dict]) -> list[int]:
    """Return indices where turn changes."""
    bounds = []
    prev = None
    for i, c in enumerate(calls):
        t = c.get("turn_number", 1)
        if prev is not None and t != prev:
            bounds.append(i)
        prev = t
    return bounds


def detect_truncations(calls: list[dict], threshold: float = 0.5) -> list[dict]:
    """Detect prompt token drops > threshold fraction."""
    truncs = []
    for i in range(1, len(calls)):
        prev_pt = calls[i - 1]["prompt_tokens"]
        curr_pt = calls[i]["prompt_tokens"]
        if prev_pt > 5000 and curr_pt < prev_pt * (1 - threshold):
            truncs.append({
                "index": i,
                "before": prev_pt,
                "after": curr_pt,
                "fraction_lost": 1 - curr_pt / prev_pt,
            })
    return truncs


def per_turn_stats(calls: list[dict]) -> dict:
    """Compute per-turn aggregated stats."""
    turns = {}
    for c in calls:
        t = c["turn_number"]
        if t not in turns:
            turns[t] = {
                "calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                "total_latency": 0.0, "max_prompt": 0, "latencies": [],
            }
        turns[t]["calls"] += 1
        turns[t]["prompt_tokens"] += c["prompt_tokens"]
        turns[t]["completion_tokens"] += c["completion_tokens"]
        turns[t]["total_latency"] += c["request_latency_seconds"]
        turns[t]["max_prompt"] = max(turns[t]["max_prompt"], c["prompt_tokens"])
        turns[t]["latencies"].append(c["request_latency_seconds"])
    # Compute efficiency
    for t, d in turns.items():
        d["efficiency_ms_per_tok"] = (
            d["total_latency"] / d["prompt_tokens"] * 1000
            if d["prompt_tokens"] > 0 else 0
        )
    return turns


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


# ── Figure 1: Prompt Token Growth with Truncation Events ─────────────────────

def fig1_prompt_growth_with_truncations(seqs: list[SeqData], out_dir: str):
    """Multi-panel figure showing prompt token growth per call, with truncation
    events annotated as red zones."""
    fig, axes = plt.subplots(len(seqs), 1, figsize=(12, 3.2 * len(seqs)),
                              sharex=False)
    if len(seqs) == 1:
        axes = [axes]

    for ax, sd in zip(axes, seqs):
        calls = sd.calls
        xs = list(range(len(calls)))
        pts = [c["prompt_tokens"] for c in calls]
        bounds = get_turn_boundaries(calls)
        truncs = detect_truncations(calls)

        # Plot prompt tokens
        ax.plot(xs, pts, color=sd.color, linewidth=1.8, marker="o",
                markersize=3, markerfacecolor=sd.color, markeredgecolor="white",
                markeredgewidth=0.5, zorder=3)

        # Turn boundaries
        for bx in bounds:
            ax.axvline(x=bx - 0.5, color="#94A3B8", linestyle="--",
                       linewidth=0.8, alpha=0.6)

        # Annotate truncation events
        for tr in truncs:
            idx = tr["index"]
            ax.annotate(
                f"TRUNCATION\n{tr['before']:,}→{tr['after']:,}\n({tr['fraction_lost']:.0%} lost)",
                xy=(idx, tr["after"]),
                xytext=(idx + 2, tr["after"] + 5000),
                fontsize=8, color="#DC2626", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#DC2626", lw=1.2),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#FEE2E2",
                          edgecolor="#DC2626", alpha=0.9),
                zorder=5,
            )
            # Red vertical band
            ax.axvspan(idx - 0.5, idx + 0.5, alpha=0.15, color="#DC2626", zorder=1)

        # 32K context limit line
        ax.axhline(y=32768, color="#EF4444", linestyle=":", linewidth=1.0,
                   alpha=0.5, label="32K context limit")

        ax.set_ylabel("Prompt Tokens")
        ax.set_title(sd.label, fontsize=11, fontweight="bold")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: f"{x / 1000:.0f}K"))
        ax.grid(True, alpha=0.2, linewidth=0.5)
        ax.legend(loc="upper left", fontsize=8)

    axes[-1].set_xlabel("LLM Call Index")
    fig.suptitle("Fig 1: Prompt Token Growth with Catastrophic Truncation Events",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(out_dir, f"fig1_prompt_growth_truncation.{ext}"))
    plt.close(fig)
    print("[OK] fig1_prompt_growth_truncation")


# ── Figure 2: KV Cache Sawtooth Pattern ──────────────────────────────────────

def fig2_kv_cache_sawtooth(seqs: list[SeqData], out_dir: str):
    """Multi-panel KV cache usage over time showing sawtooth drops."""
    fig, axes = plt.subplots(len(seqs), 1, figsize=(12, 3.2 * len(seqs)),
                              sharex=False)
    if len(seqs) == 1:
        axes = [axes]

    for ax, sd in zip(axes, seqs):
        timeline = sd.timeline
        if not timeline:
            ax.text(0.5, 0.5, "No timeline data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=14, color="#999")
            ax.set_title(sd.label)
            continue

        raw_elapsed = [s["elapsed_seconds"] for s in timeline]
        raw_kv = [s.get("kv_cache_usage_perc", 0) or 0 for s in timeline]
        elapsed = [t for t, v in zip(raw_elapsed, raw_kv) if v > 0]
        kv = [v for v in raw_kv if v > 0]

        ax.fill_between(elapsed, kv, alpha=0.3, color=sd.color, zorder=2)
        ax.plot(elapsed, kv, color=sd.color, linewidth=1.5, zorder=3)

        # Turn boundaries on timeline
        calls = sd.calls
        prev_turn = None
        t0 = timeline[0]["timestamp"] if timeline else 0
        for c in calls:
            tn = c.get("turn_number", 1)
            if prev_turn is not None and tn != prev_turn:
                t = c.get("request_started_at")
                if t is not None:
                    ax.axvline(x=t - t0, color="#94A3B8", linestyle="--",
                               linewidth=0.8, alpha=0.6)
            prev_turn = tn

        # Count and annotate drops
        drops = detect_kv_drops(timeline)
        ax.set_title(f"{sd.label}  ({drops} KV cache drops)",
                     fontsize=11, fontweight="bold")
        ax.set_ylabel("KV Cache Usage (%)")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax.grid(True, alpha=0.2, linewidth=0.5)

    axes[-1].set_xlabel("Elapsed Time (seconds)")
    fig.suptitle("Fig 2: KV Cache Usage — Sawtooth Pattern from Cache Destruction",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(out_dir, f"fig2_kv_cache_sawtooth.{ext}"))
    plt.close(fig)
    print("[OK] fig2_kv_cache_sawtooth")


# ── Figure 3: Prefix Cache Hit Rate Dynamics ─────────────────────────────────

def fig3_prefix_cache_dynamics(seqs: list[SeqData], out_dir: str):
    """Overlay prefix cache hit rate for all sequences on one plot."""
    fig, ax = plt.subplots(figsize=(10, 5))

    for sd in seqs:
        timeline = sd.timeline
        if not timeline:
            continue
        elapsed = [s["elapsed_seconds"] for s in timeline]
        hr = [s.get("prefix_cache_hit_rate", 0) or 0 for s in timeline]
        ax.plot(elapsed, hr, color=sd.color, linewidth=1.5, label=sd.label,
                alpha=0.85)

    ax.set_xlabel("Elapsed Time (seconds)")
    ax.set_ylabel("Prefix Cache Hit Rate")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_ylim(-0.05, 1.15)
    ax.grid(True, alpha=0.2, linewidth=0.5)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title("Fig 3: Prefix Cache Hit Rate — Warmup and Turn-Boundary Drops",
                 fontsize=13, fontweight="bold")

    # Annotate cold start region
    ax.axvspan(0, 100, alpha=0.08, color="#DC2626", zorder=0)
    ax.text(50, 0.15, "Cold Start\nRegion", ha="center", va="center",
            fontsize=9, color="#DC2626", fontstyle="italic")

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(out_dir, f"fig3_prefix_cache_dynamics.{ext}"))
    plt.close(fig)
    print("[OK] fig3_prefix_cache_dynamics")


# ── Figure 4: Per-Turn Efficiency Comparison ─────────────────────────────────

def fig4_turn_efficiency(seqs: list[SeqData], out_dir: str):
    """Grouped bar chart of ms/tok efficiency per turn, per sequence."""
    fig, ax = plt.subplots(figsize=(10, 5))

    max_turns = max(sd.num_turns for sd in seqs)
    x_base = np.arange(1, max_turns + 1)
    bar_width = 0.8 / len(seqs)

    for i, sd in enumerate(seqs):
        turns = per_turn_stats(sd.calls)
        efficiencies = []
        x_positions = []
        for t in range(1, max_turns + 1):
            if t in turns:
                efficiencies.append(turns[t]["efficiency_ms_per_tok"])
                x_positions.append(t + (i - len(seqs) / 2 + 0.5) * bar_width)
        bars = ax.bar(x_positions, efficiencies, width=bar_width * 0.9,
                      color=sd.color, alpha=0.85, label=sd.label, zorder=3)
        # Annotate values
        for bar, eff in zip(bars, efficiencies):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                    f"{eff:.1f}", ha="center", va="bottom", fontsize=7,
                    fontweight="bold", color=sd.color)

    ax.set_xlabel("Turn Number")
    ax.set_ylabel("Efficiency (ms/tok) — lower is better")
    ax.set_xticks(x_base)
    ax.grid(True, alpha=0.2, linewidth=0.5, axis="y")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title("Fig 4: Per-Turn Inference Efficiency — 6.3x Variance from Skill Switching",
                 fontsize=13, fontweight="bold")

    # Annotate worst/best
    all_effs = []
    for sd in seqs:
        turns = per_turn_stats(sd.calls)
        for t, d in turns.items():
            all_effs.append((d["efficiency_ms_per_tok"], sd.key, t))
    worst = max(all_effs, key=lambda x: x[0])
    best = min(all_effs, key=lambda x: x[0])
    variance = worst[0] / best[0]
    ax.text(0.02, 0.95,
            f"Worst: {worst[0]:.1f} ms/tok ({worst[1]} Turn {worst[2]})\n"
            f"Best:  {best[0]:.1f} ms/tok ({best[1]} Turn {best[2]})\n"
            f"Variance: {variance:.1f}x",
            transform=ax.transAxes, fontsize=9, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="#F0F9FF", edgecolor="#2563EB",
                      alpha=0.9))

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(out_dir, f"fig4_turn_efficiency.{ext}"))
    plt.close(fig)
    print("[OK] fig4_turn_efficiency")


# ── Figure 5: Context Overhead Breakdown ─────────────────────────────────────

def fig5_overhead_breakdown(seqs: list[SeqData], out_dir: str):
    """Stacked bar chart showing ideal vs overhead prompt tokens per sequence."""
    fig, ax = plt.subplots(figsize=(8, 5))

    seq_labels = [sd.label.split(" (")[0] for sd in seqs]
    x = np.arange(len(seqs))

    ideal_tokens = []
    overhead_tokens = []
    total_tokens = []

    for sd in seqs:
        total = sum(c["prompt_tokens"] for c in sd.calls)
        # "Ideal" = first call's prompt * number of calls (no growth)
        ideal = sd.calls[0]["prompt_tokens"] * len(sd.calls)
        overhead = total - ideal
        total_tokens.append(total)
        ideal_tokens.append(ideal)
        overhead_tokens.append(overhead)

    bars1 = ax.bar(x, ideal_tokens, width=0.5, color="#059669", alpha=0.85,
                   label="Useful (unique content)", zorder=3)
    bars2 = ax.bar(x, overhead_tokens, width=0.5, bottom=ideal_tokens,
                   color="#DC2626", alpha=0.7, label="Overhead (re-sent context)",
                   zorder=3)

    # Annotate percentages
    for i, (tot, ideal, ovhd) in enumerate(zip(total_tokens, ideal_tokens, overhead_tokens)):
        pct = ovhd / tot * 100
        ax.text(i, tot + 10000, f"{pct:.0f}% overhead\n({tot / 1000:.0f}K total)",
                ha="center", va="bottom", fontsize=9, fontweight="bold",
                color="#DC2626")

    ax.set_xticks(x)
    ax.set_xticklabels(seq_labels, fontsize=10)
    ax.set_ylabel("Total Prompt Tokens")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda val, _: f"{val / 1000:.0f}K"))
    ax.grid(True, alpha=0.2, linewidth=0.5, axis="y")
    ax.legend(loc="upper left", fontsize=10)
    ax.set_title("Fig 5: Context Accumulation Overhead — 52–61% of Tokens are Redundant",
                 fontsize=13, fontweight="bold")

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(out_dir, f"fig5_overhead_breakdown.{ext}"))
    plt.close(fig)
    print("[OK] fig5_overhead_breakdown")


# ── Figure 6: Latency Spikes Correlated with Truncation ──────────────────────

def fig6_latency_vs_truncation(seqs: list[SeqData], out_dir: str):
    """Dual-axis plot: prompt tokens (left) and latency (right) on same x-axis,
    showing correlation between truncation events and latency spikes."""
    # Use T4_B as the primary example (most truncation events)
    sd = seqs[0]  # T4_B
    calls = sd.calls
    xs = list(range(len(calls)))
    pts = [c["prompt_tokens"] for c in calls]
    lats = [c["request_latency_seconds"] for c in calls]
    bounds = get_turn_boundaries(calls)
    truncs = detect_truncations(calls)

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax2 = ax1.twinx()

    # Prompt tokens (left axis)
    ln1 = ax1.plot(xs, pts, color="#2563EB", linewidth=1.5, marker="o",
                   markersize=3, label="Prompt Tokens", zorder=3)
    ax1.set_ylabel("Prompt Tokens", color="#2563EB", fontweight="bold")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda val, _: f"{val / 1000:.0f}K"))
    ax1.tick_params(axis="y", labelcolor="#2563EB")

    # Latency (right axis)
    ln2 = ax2.bar(xs, lats, width=0.6, color="#DC2626", alpha=0.5,
                  label="Latency (s)", zorder=2)
    ax2.set_ylabel("Latency (seconds)", color="#DC2626", fontweight="bold")
    ax2.tick_params(axis="y", labelcolor="#DC2626")

    # Turn boundaries
    for bx in bounds:
        ax1.axvline(x=bx - 0.5, color="#94A3B8", linestyle="--",
                    linewidth=0.8, alpha=0.6)

    # Truncation events
    for tr in truncs:
        ax1.axvspan(tr["index"] - 0.5, tr["index"] + 0.5, alpha=0.2,
                    color="#FCD34D", zorder=1)

    ax1.set_xlabel("LLM Call Index")
    ax1.set_title(f"Fig 6: {sd.label} — Latency Spikes Correlate with Context Truncation",
                  fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.15, linewidth=0.5)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(lines1 + [mpatches.Patch(color="#DC2626", alpha=0.5, label="Latency (s)"),
                          mpatches.Patch(color="#FCD34D", alpha=0.5, label="Truncation event")],
               labels1 + ["Latency (s)", "Truncation event"],
               loc="upper left", fontsize=9)

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(out_dir, f"fig6_latency_truncation_corr.{ext}"))
    plt.close(fig)
    print("[OK] fig6_latency_truncation_corr")


# ── Figure 6.2: Latency × KV Cache Usage Combined ────────────────────────────

def fig6_2_latency_kv_combined(seqs: list[SeqData], out_dir: str):
    """Multi-panel: per-call latency bars + KV cache usage on shared time axis,
    showing how KV drops directly precede latency spikes."""
    n = len(seqs)
    fig, axes = plt.subplots(n, 1, figsize=(14, 4.2 * n), sharex=False)
    if n == 1:
        axes = [axes]

    for ax, sd in zip(axes, seqs):
        calls = sd.calls
        timeline = sd.timeline
        if not timeline:
            ax.text(0.5, 0.5, "No timeline data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=14, color="#999")
            ax.set_title(sd.label)
            continue

        # Derive absolute t0 from timeline so calls and timeline share the same x-axis
        t0_abs = timeline[0]["timestamp"] - timeline[0]["elapsed_seconds"]

        # KV cache usage (filtered zeros) on left axis
        raw_elapsed = [s["elapsed_seconds"] for s in timeline]
        raw_kv = [s.get("kv_cache_usage_perc", 0) or 0 for s in timeline]
        kv_elapsed = [t for t, v in zip(raw_elapsed, raw_kv) if v > 0]
        kv_vals = [v for v in raw_kv if v > 0]

        ax.fill_between(kv_elapsed, kv_vals, alpha=0.25, color=sd.color, zorder=2)
        ax.plot(kv_elapsed, kv_vals, color=sd.color, linewidth=1.2, zorder=3,
                label="KV Cache Usage")
        ax.set_ylabel("KV Cache Usage (%)", color=sd.color, fontweight="bold")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax.tick_params(axis="y", labelcolor=sd.color)
        ax.set_ylim(-0.02, max(kv_vals) * 1.15 if kv_vals else 0.6)

        # Latency bars on right axis (convert call abs timestamps to elapsed seconds)
        ax2 = ax.twinx()
        call_times = []
        call_lats = []
        for c in calls:
            t = c.get("request_started_at")
            if t is not None:
                call_times.append(t - t0_abs)
                call_lats.append(c["request_latency_seconds"])

        bar_width = max(5, (max(call_times) - min(call_times)) / len(call_times) * 0.4) if call_times else 5
        ax2.bar(call_times, call_lats, width=bar_width, color="#DC2626",
                alpha=0.45, zorder=4, label="Latency (s)")
        ax2.set_ylabel("Latency (seconds)", color="#DC2626", fontweight="bold")
        ax2.tick_params(axis="y", labelcolor="#DC2626")

        # Truncation event highlights
        truncs = detect_truncations(calls)
        for tr in truncs:
            t = calls[tr["index"]].get("request_started_at")
            if t is not None:
                te = t - t0_abs
                ax.axvspan(te - 5, te + 5, alpha=0.2,
                           color="#FCD34D", zorder=1, label="Truncation" if tr == truncs[0] else "")

        # Turn boundaries
        prev_turn = None
        for c in calls:
            tn = c.get("turn_number", 1)
            if prev_turn is not None and tn != prev_turn:
                t = c.get("request_started_at")
                if t is not None:
                    ax.axvline(x=t - t0_abs, color="#94A3B8", linestyle="--",
                               linewidth=0.8, alpha=0.5)
            prev_turn = tn

        drops = detect_kv_drops(timeline)
        ax.set_title(f"{sd.label}  ({drops} KV drops, {len(truncs)} truncations)",
                     fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.15, linewidth=0.5)

        # Combined legend
        handles1, labels1 = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        seen = set()
        unique_handles, unique_labels = [], []
        for h, l in zip(handles1 + handles2, labels1 + labels2):
            if l not in seen:
                seen.add(l)
                unique_handles.append(h)
                unique_labels.append(l)
        ax.legend(unique_handles, unique_labels, loc="upper left", fontsize=8,
                  framealpha=0.9)

    axes[-1].set_xlabel("Elapsed Time (seconds)")
    fig.suptitle("Fig 6.2: Latency Spikes × KV Cache Drops — Temporal Correlation",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(out_dir, f"fig6_2_latency_kv_combined.{ext}"),
                    bbox_inches="tight")
    plt.close(fig)
    print("[OK] fig6_2_latency_kv_combined")


# ── Summary Statistics Table ─────────────────────────────────────────────────

def print_summary_table(seqs: list[SeqData], out_dir: str):
    """Print summary statistics and save as LaTeX table."""
    lines = []
    lines.append("=" * 100)
    lines.append("SegKV Characterization — Summary Statistics")
    lines.append("=" * 100)

    latex_rows = []

    for sd in seqs:
        calls = sd.calls
        total_prompt = sum(c["prompt_tokens"] for c in calls)
        total_comp = sum(c["completion_tokens"] for c in calls)
        ideal_prompt = calls[0]["prompt_tokens"] * len(calls)
        overhead_pct = (total_prompt - ideal_prompt) / total_prompt * 100

        truncs = detect_truncations(calls)
        kv_drops = detect_kv_drops(sd.timeline) if sd.timeline else "N/A"

        turns = per_turn_stats(calls)
        effs = [d["efficiency_ms_per_tok"] for d in turns.values()]
        eff_variance = max(effs) / min(effs) if min(effs) > 0 else float("inf")

        # Prefix cache stats
        if sd.timeline:
            hrs = [s.get("prefix_cache_hit_rate", 0) or 0 for s in sd.timeline]
            avg_hr = np.mean(hrs)
            min_hr = min(hrs)
        else:
            avg_hr = min_hr = 0

        lines.append(f"\n--- {sd.label} ---")
        lines.append(f"  Turns: {sd.num_turns}")
        lines.append(f"  LLM Calls: {len(calls)}")
        lines.append(f"  Total Elapsed: {sd.total_elapsed:.1f}s ({sd.total_elapsed / 60:.1f}min)")
        lines.append(f"  Total Prompt Tokens: {total_prompt:,}")
        lines.append(f"  Total Completion Tokens: {total_comp:,}")
        lines.append(f"  Ideal Prompt Tokens: {ideal_prompt:,}")
        lines.append(f"  Context Overhead: {overhead_pct:.1f}%")
        lines.append(f"  Truncation Events: {len(truncs)}")
        lines.append(f"  KV Cache Drops: {kv_drops}")
        lines.append(f"  Prefix Cache Hit Rate: avg={avg_hr:.1%}, min={min_hr:.1%}")
        lines.append(f"  Efficiency Variance: {eff_variance:.1f}x "
                     f"(best={min(effs):.2f}, worst={max(effs):.2f} ms/tok)")
        for t, d in sorted(turns.items()):
            lines.append(f"    Turn {t}: {d['calls']} calls, "
                         f"prompt={d['prompt_tokens']:,}, "
                         f"eff={d['efficiency_ms_per_tok']:.2f} ms/tok, "
                         f"max_lat={max(d['latencies']):.1f}s")

        # LaTeX row
        latex_rows.append(
            f"  {sd.key.replace('_', '-')} & {sd.num_turns} & {len(calls)} & "
            f"{total_prompt / 1000:.0f}K & {overhead_pct:.0f}\\% & "
            f"{len(truncs)} & {kv_drops} & {avg_hr:.0%} & "
            f"{eff_variance:.1f}$\\times$ \\\\"
        )

    # Print to console
    summary_text = "\n".join(lines)
    print(summary_text)

    # Save to file
    with open(os.path.join(out_dir, "summary_stats.txt"), "w") as f:
        f.write(summary_text)

    # LaTeX table
    latex = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Characterization of multi-turn skill-augmented agent workloads "
        "on vLLM with prefix caching (Qwen3-Coder-30B-A3B, 32K context).}",
        "\\label{tab:characterization}",
        "\\small",
        "\\begin{tabular}{lcccccccc}",
        "\\toprule",
        "Seq & Turns & Calls & Prompt Tok & Overhead & Truncations & "
        "KV Drops & Cache Hit & Eff. Var. \\\\",
        "\\midrule",
    ]
    latex.extend(latex_rows)
    latex.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    latex_text = "\n".join(latex)
    with open(os.path.join(out_dir, "table_characterization.tex"), "w") as f:
        f.write(latex_text)

    print(f"\n[OK] summary_stats.txt + table_characterization.tex")


# ── Figure 7: Combined Motivation Figure (2x2 for paper) ────────────────────

def fig7_combined_motivation(seqs: list[SeqData], out_dir: str):
    """2x2 combined figure for paper: (a) prompt growth, (b) KV sawtooth,
    (c) efficiency variance, (d) overhead breakdown."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # Use T4_B as primary example for (a) and (b)
    sd = seqs[0]

    # (a) Prompt token growth with truncation
    ax = axes[0, 0]
    calls = sd.calls
    xs = list(range(len(calls)))
    pts = [c["prompt_tokens"] for c in calls]
    bounds = get_turn_boundaries(calls)
    truncs = detect_truncations(calls)

    ax.plot(xs, pts, color=sd.color, linewidth=1.5, marker="o", markersize=2.5,
            zorder=3)
    for bx in bounds:
        ax.axvline(x=bx - 0.5, color="#94A3B8", linestyle="--", linewidth=0.7, alpha=0.5)
    for tr in truncs:
        ax.axvspan(tr["index"] - 0.5, tr["index"] + 0.5, alpha=0.2,
                   color="#DC2626", zorder=1)
    ax.axhline(y=32768, color="#EF4444", linestyle=":", linewidth=0.8, alpha=0.4)
    ax.set_ylabel("Prompt Tokens")
    ax.set_xlabel("LLM Call Index")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f}K"))
    ax.set_title("(a) Prompt Token Growth (T4-B)", fontweight="bold")
    ax.grid(True, alpha=0.15)

    # (b) KV cache sawtooth
    ax = axes[0, 1]
    if sd.timeline:
        raw_elapsed = [s["elapsed_seconds"] for s in sd.timeline]
        raw_kv = [s.get("kv_cache_usage_perc", 0) or 0 for s in sd.timeline]
        elapsed_b = [t for t, v in zip(raw_elapsed, raw_kv) if v > 0]
        kv_b = [v for v in raw_kv if v > 0]
        ax.fill_between(elapsed_b, kv_b, alpha=0.3, color=sd.color)
        ax.plot(elapsed_b, kv_b, color=sd.color, linewidth=1.2)
        drops = detect_kv_drops(sd.timeline)
        ax.set_title(f"(b) KV Cache Sawtooth (T4-B, {drops} drops)", fontweight="bold")
    ax.set_ylabel("KV Cache Usage (%)")
    ax.set_xlabel("Elapsed Time (s)")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.grid(True, alpha=0.15)

    # (c) Per-turn efficiency
    ax = axes[1, 0]
    max_turns = max(sd.num_turns for sd in seqs)
    bar_width = 0.8 / len(seqs)
    for i, sd2 in enumerate(seqs):
        turns = per_turn_stats(sd2.calls)
        x_pos = []
        effs = []
        for t in range(1, max_turns + 1):
            if t in turns:
                x_pos.append(t + (i - len(seqs) / 2 + 0.5) * bar_width)
                effs.append(turns[t]["efficiency_ms_per_tok"])
        ax.bar(x_pos, effs, width=bar_width * 0.9, color=sd2.color, alpha=0.8,
               label=sd2.label.split(" (")[0])
    ax.set_xlabel("Turn Number")
    ax.set_ylabel("ms/tok (lower = better)")
    ax.set_xticks(range(1, max_turns + 1))
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("(c) Per-Turn Inference Efficiency", fontweight="bold")
    ax.grid(True, alpha=0.15, axis="y")

    # (d) Overhead breakdown
    ax = axes[1, 1]
    labels = [sd2.key.replace("_", "-") for sd2 in seqs]
    ideal = []
    overhead = []
    for sd2 in seqs:
        total = sum(c["prompt_tokens"] for c in sd2.calls)
        id_val = sd2.calls[0]["prompt_tokens"] * len(sd2.calls)
        ideal.append(id_val)
        overhead.append(total - id_val)
    x = np.arange(len(seqs))
    ax.bar(x, ideal, width=0.5, color="#059669", alpha=0.8, label="Useful")
    ax.bar(x, overhead, width=0.5, bottom=ideal, color="#DC2626", alpha=0.7,
           label="Overhead")
    for i, (id_v, oh_v) in enumerate(zip(ideal, overhead)):
        total = id_v + oh_v
        pct = oh_v / total * 100
        ax.text(i, total + 5000, f"{pct:.0f}%", ha="center", fontsize=9,
                fontweight="bold", color="#DC2626")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f}K"))
    ax.set_ylabel("Total Prompt Tokens")
    ax.legend(fontsize=9)
    ax.set_title("(d) Context Accumulation Overhead", fontweight="bold")
    ax.grid(True, alpha=0.15, axis="y")

    fig.suptitle("SegKV Motivation: Characterization of Skill-Augmented Agent Inference",
                 fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(out_dir, f"fig7_combined_motivation.{ext}"))
    plt.close(fig)
    print("[OK] fig7_combined_motivation")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SegKV characterization / motivation figures")
    parser.add_argument("--results-dir",
                        default="results/03/multurn_bench",
                        help="Root results directory")
    parser.add_argument("--out-dir",
                        default="results/03/multurn_bench/segkv_figures",
                        help="Output directory for figures")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load all sequences
    seqs = []
    for seq_key in SEQUENCES:
        sd = load_seq(args.results_dir, seq_key)
        if sd:
            seqs.append(sd)
    if not seqs:
        print("ERROR: no sequence data found", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded {len(seqs)} sequences: {[s.key for s in seqs]}\n")

    # Generate all figures
    fig1_prompt_growth_with_truncations(seqs, args.out_dir)
    fig2_kv_cache_sawtooth(seqs, args.out_dir)
    fig3_prefix_cache_dynamics(seqs, args.out_dir)
    fig4_turn_efficiency(seqs, args.out_dir)
    fig5_overhead_breakdown(seqs, args.out_dir)
    fig6_latency_vs_truncation(seqs, args.out_dir)
    fig6_2_latency_kv_combined(seqs, args.out_dir)
    fig7_combined_motivation(seqs, args.out_dir)
    print()
    print_summary_table(seqs, args.out_dir)


if __name__ == "__main__":
    main()
