#!/usr/bin/env python3
"""
SegKV Paper — 14B Anthropic Benchmark Characterization Figures.

Generates publication-quality figures from Qwen3-14B × Anthropic multi-turn
benchmark traces (8 tasks × 2 context lengths) to motivate segment-aware
KV Cache management.

Figures:
  Fig 1  (2×2)  Combined motivation: prompt growth, 32K-vs-16K bar, KV timeline, prefix hit rate
  Fig 2  (3×1)  Truncation deep-dive on launch_poster_page_pack (32K)
  Fig 3  (2×1)  Context overhead breakdown + system-prompt dominance

Usage:
  python scripts/03_14B_anthropic/segkv_characterization_14B.py \
    --results-dir results/03_14B_anthropic/multurn_bench \
    --out-dir     results/03_14B_anthropic/multurn_bench/paper_figures
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

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

BENCHMARKS = [
    "baseline_feature_brainstorm",
    "baseline_update_polish",
    "doc_coauthoring_design_doc",
    "internal_comms_incident_update",
    "web_artifact_with_theme",
    "mcp_server_and_spec",
    "launch_poster_page_pack",
    "slack_launch_pack",
]

SHORT_NAMES = {
    "baseline_feature_brainstorm": "brainstorm",
    "baseline_update_polish": "update",
    "doc_coauthoring_design_doc": "design-doc",
    "internal_comms_incident_update": "incident",
    "web_artifact_with_theme": "web-artifact",
    "mcp_server_and_spec": "mcp-server",
    "launch_poster_page_pack": "poster-pack",
    "slack_launch_pack": "slack-pack",
}

SKILL_COUNTS = {
    "baseline_feature_brainstorm": 0,
    "baseline_update_polish": 0,
    "doc_coauthoring_design_doc": 2,
    "internal_comms_incident_update": 1,
    "web_artifact_with_theme": 2,
    "mcp_server_and_spec": 2,
    "launch_poster_page_pack": 3,
    "slack_launch_pack": 2,
}

PALETTE_8 = [
    "#2563EB", "#DC2626", "#059669", "#D97706",
    "#7C3AED", "#DB2777", "#0891B2", "#4B5563",
]
BENCH_COLORS = {b: PALETTE_8[i] for i, b in enumerate(BENCHMARKS)}


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class BenchData:
    repo: str
    short: str
    color: str
    ctx_len: int
    num_turns: int = 0
    total_elapsed: float = 0.0
    calls: list = field(default_factory=list)
    timeline: list = field(default_factory=list)


def load_bench(results_dir: str, ctx: int, repo: str) -> Optional[BenchData]:
    path = os.path.join(results_dir, f"ctx_{ctx}", repo,
                        "multiturn_sequence_traces.json")
    if not os.path.isfile(path):
        print(f"[SKIP] {path}")
        return None
    with open(path) as f:
        data = json.load(f)
    s = data["sequence"]
    return BenchData(
        repo=repo,
        short=SHORT_NAMES.get(repo, repo),
        color=BENCH_COLORS.get(repo, "#333"),
        ctx_len=ctx,
        num_turns=s["num_turns"],
        total_elapsed=s["total_elapsed_seconds"],
        calls=s["llm_calls"],
        timeline=s.get("vllm_timeline", []),
    )


# ── Helpers ──────────────────────────────────────────────────────────────────

def get_turn_boundaries(calls):
    bounds = []
    prev = None
    for i, c in enumerate(calls):
        t = c.get("turn_number", 1)
        if prev is not None and t != prev:
            bounds.append(i)
        prev = t
    return bounds


def detect_truncations(calls, threshold=0.5):
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


def detect_kv_drops(timeline, drop_ratio=0.3):
    nonzero = [(s["elapsed_seconds"], s.get("kv_cache_usage_perc", 0) or 0)
               for s in timeline
               if (s.get("kv_cache_usage_perc", 0) or 0) > 0]
    drops = 0
    for i in range(1, len(nonzero)):
        if nonzero[i - 1][1] > 0.001 and nonzero[i][1] < nonzero[i - 1][1] * drop_ratio:
            drops += 1
    return drops


def _save(fig, out_dir, name):
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"),
                    bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {name}")


# ── Figure 1: Combined Motivation (2×2) ─────────────────────────────────────

def fig1_combined_motivation(d32: list[BenchData], d16: list[BenchData],
                             out_dir: str):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # ── (a) Prompt token growth for 4 representative 32K benchmarks ──
    ax = axes[0, 0]
    show_set = ["web_artifact_with_theme", "launch_poster_page_pack",
                "doc_coauthoring_design_doc", "baseline_feature_brainstorm"]
    for bd in d32:
        if bd.repo not in show_set:
            continue
        pts = [c["prompt_tokens"] for c in bd.calls]
        xs = list(range(len(pts)))
        ax.plot(xs, pts, color=bd.color, linewidth=1.6, marker="o",
                markersize=2.5, label=bd.short, zorder=3)
        bounds = get_turn_boundaries(bd.calls)
        for bx in bounds:
            ax.axvline(x=bx - 0.5, color="#94A3B8", linestyle="--",
                       linewidth=0.6, alpha=0.4)
        for tr in detect_truncations(bd.calls):
            idx = tr["index"]
            ax.axvspan(idx - 0.4, idx + 0.4, alpha=0.18, color="#DC2626",
                       zorder=1)
            ax.annotate(
                f"TRUNC\n{tr['before']//1000}K→{tr['after']//1000}K",
                xy=(idx, tr["after"]), xytext=(idx + 1.5, tr["after"] + 6000),
                fontsize=7, color="#DC2626", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#DC2626", lw=1),
                bbox=dict(boxstyle="round,pad=0.2", fc="#FEE2E2",
                          ec="#DC2626", alpha=0.9),
                zorder=5)
    ax.axhline(y=32768, color="#EF4444", linestyle=":", linewidth=0.8,
               alpha=0.4, label="32K limit")
    ax.set_ylabel("Prompt Tokens")
    ax.set_xlabel("LLM Call Index")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{v / 1000:.0f}K"))
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    ax.set_title("(a) Prompt Growth — 32K Context", fontweight="bold")
    ax.grid(True, alpha=0.15)

    # ── (b) 32K vs 16K comparison bar chart ──
    ax = axes[0, 1]
    repos_common = [b.repo for b in d32 if any(x.repo == b.repo for x in d16)]
    n = len(repos_common)
    x = np.arange(n)
    w = 0.35

    tok32 = [sum(c["prompt_tokens"] for c in next(b for b in d32 if b.repo == r).calls)
             for r in repos_common]
    tok16 = [sum(c["prompt_tokens"] for c in next(b for b in d16 if b.repo == r).calls)
             for r in repos_common]
    labels = [SHORT_NAMES[r] for r in repos_common]

    bars1 = ax.bar(x - w / 2, [t / 1000 for t in tok32], w, color="#2563EB",
                   alpha=0.8, label="32K ctx")
    bars2 = ax.bar(x + w / 2, [t / 1000 for t in tok16], w, color="#059669",
                   alpha=0.8, label="16K ctx")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=30, ha="right")
    ax.set_ylabel("Total Prompt Tokens (K)")
    ax.legend(fontsize=9)
    ax.set_title("(b) 32K vs 16K — Token Volume", fontweight="bold")
    ax.grid(True, alpha=0.15, axis="y")
    total_32 = sum(tok32)
    total_16 = sum(tok16)
    ax.text(0.97, 0.95,
            f"32K total: {total_32/1e6:.2f}M\n16K total: {total_16/1e6:.2f}M\n"
            f"Ratio: {total_32/total_16:.1f}×",
            transform=ax.transAxes, fontsize=8, va="top", ha="right",
            bbox=dict(boxstyle="round", fc="#F0F9FF", ec="#2563EB", alpha=0.9))

    # ── (c) KV Cache usage timeline for representative benchmarks (32K) ──
    ax = axes[1, 0]
    show_kv = ["web_artifact_with_theme", "launch_poster_page_pack",
               "doc_coauthoring_design_doc"]
    for bd in d32:
        if bd.repo not in show_kv or not bd.timeline:
            continue
        raw_e = [s["elapsed_seconds"] for s in bd.timeline]
        raw_kv = [s.get("kv_cache_usage_perc", 0) or 0 for s in bd.timeline]
        e = [t for t, v in zip(raw_e, raw_kv) if v > 0]
        kv = [v for v in raw_kv if v > 0]
        ax.fill_between(e, kv, alpha=0.2, color=bd.color)
        ax.plot(e, kv, color=bd.color, linewidth=1.3, label=bd.short)
    ax.set_ylabel("KV Cache Usage (%)")
    ax.set_xlabel("Elapsed Time (s)")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("(c) KV Cache Timeline — 32K Context", fontweight="bold")
    ax.grid(True, alpha=0.15)

    # ── (d) Prefix cache hit rate dynamics (32K) ──
    ax = axes[1, 1]
    for bd in d32:
        if bd.repo not in show_kv or not bd.timeline:
            continue
        e = [s["elapsed_seconds"] for s in bd.timeline]
        hr = [s.get("prefix_cache_hit_rate", 0) or 0 for s in bd.timeline]
        ax.plot(e, hr, color=bd.color, linewidth=1.3, label=bd.short,
                alpha=0.85)
    ax.axvspan(0, 60, alpha=0.08, color="#DC2626", zorder=0)
    ax.text(30, 0.12, "Cold Start", ha="center", fontsize=8,
            color="#DC2626", fontstyle="italic")
    ax.set_ylabel("Prefix Cache Hit Rate")
    ax.set_xlabel("Elapsed Time (s)")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("(d) Prefix Cache Hit Rate Dynamics", fontweight="bold")
    ax.grid(True, alpha=0.15)

    fig.suptitle(
        "Fig 1: Multi-Turn Agent Inference Characterization "
        "(Qwen3-14B, Anthropic Benchmark, vLLM + Prefix Caching)",
        fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, out_dir, "fig1_combined_motivation")


# ── Figure 2: Truncation Deep-Dive ──────────────────────────────────────────

def fig2_truncation_deep_dive(d32: list[BenchData], out_dir: str):
    bd = next((b for b in d32 if b.repo == "launch_poster_page_pack"), None)
    if bd is None:
        print("[SKIP] fig2 — launch_poster_page_pack not found")
        return

    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=False)
    calls = bd.calls
    xs = list(range(len(calls)))
    pts = [c["prompt_tokens"] for c in calls]
    lats = [c.get("request_latency_seconds", 0) for c in calls]
    bounds = get_turn_boundaries(calls)
    truncs = detect_truncations(calls)

    # ── (a) Prompt tokens ──
    ax = axes[0]
    ax.plot(xs, pts, color="#2563EB", linewidth=1.8, marker="o",
            markersize=4, markerfacecolor="#2563EB", markeredgecolor="white",
            markeredgewidth=0.5, zorder=3)
    for bx in bounds:
        ax.axvline(x=bx - 0.5, color="#94A3B8", linestyle="--",
                   linewidth=0.7, alpha=0.5)
    ax.axhline(y=32768, color="#EF4444", linestyle=":", linewidth=0.9,
               alpha=0.5, label="32K limit")
    for tr in truncs:
        idx = tr["index"]
        ax.axvspan(idx - 0.5, idx + 0.5, alpha=0.18, color="#DC2626", zorder=1)
        ax.annotate(
            f"TRUNCATION\n{tr['before']:,} → {tr['after']:,}\n"
            f"({tr['fraction_lost']:.1%} lost)",
            xy=(idx, tr["after"]),
            xytext=(idx + 2, tr["after"] + 8000),
            fontsize=9, color="#DC2626", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#DC2626", lw=1.2),
            bbox=dict(boxstyle="round,pad=0.3", fc="#FEE2E2", ec="#DC2626",
                      alpha=0.9),
            zorder=5)
    ax.set_ylabel("Prompt Tokens")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{v / 1000:.0f}K"))
    ax.legend(fontsize=9, loc="upper left")
    ax.set_title("(a) Prompt Token Growth — Catastrophic Truncation at 32K",
                 fontweight="bold")
    ax.grid(True, alpha=0.15)

    # ── (b) Per-call latency ──
    ax = axes[1]
    colors_lat = []
    for i, lat in enumerate(lats):
        is_trunc_zone = any(abs(i - tr["index"]) <= 2 for tr in truncs)
        colors_lat.append("#DC2626" if is_trunc_zone else "#2563EB")
    ax.bar(xs, lats, color=colors_lat, alpha=0.75, zorder=3)
    for bx in bounds:
        ax.axvline(x=bx - 0.5, color="#94A3B8", linestyle="--",
                   linewidth=0.7, alpha=0.5)
    for tr in truncs:
        ax.axvspan(tr["index"] - 0.5, tr["index"] + 2.5, alpha=0.12,
                   color="#FCD34D", zorder=1)
    median_lat = np.median(lats)
    ax.axhline(y=median_lat, color="#059669", linestyle="--", linewidth=1,
               alpha=0.6, label=f"Median = {median_lat:.1f}s")
    max_lat_idx = int(np.argmax(lats))
    ax.annotate(f"Peak: {lats[max_lat_idx]:.0f}s\n(call[{max_lat_idx}])",
                xy=(max_lat_idx, lats[max_lat_idx]),
                xytext=(max_lat_idx + 2, lats[max_lat_idx] - 10),
                fontsize=8, fontweight="bold", color="#DC2626",
                arrowprops=dict(arrowstyle="->", color="#DC2626", lw=1))
    ax.set_ylabel("Latency (seconds)")
    ax.legend(fontsize=9, loc="upper right")
    ax.set_title("(b) Per-Call Latency — Spikes in Truncation Recovery Zone",
                 fontweight="bold")
    ax.grid(True, alpha=0.15, axis="y")

    # ── (c) KV Cache + prefix hit rate on shared timeline ──
    ax = axes[2]
    if bd.timeline:
        raw_e = [s["elapsed_seconds"] for s in bd.timeline]
        raw_kv = [s.get("kv_cache_usage_perc", 0) or 0 for s in bd.timeline]
        raw_hr = [s.get("prefix_cache_hit_rate", 0) or 0 for s in bd.timeline]

        e_kv = [t for t, v in zip(raw_e, raw_kv) if v > 0]
        kv = [v for v in raw_kv if v > 0]

        ax.fill_between(e_kv, kv, alpha=0.25, color="#7C3AED")
        ax.plot(e_kv, kv, color="#7C3AED", linewidth=1.3,
                label="KV Cache Usage")
        ax.set_ylabel("KV Cache Usage (%)", color="#7C3AED")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

        ax2 = ax.twinx()
        ax2.plot(raw_e, raw_hr, color="#D97706", linewidth=1.3, alpha=0.8,
                 label="Prefix Hit Rate")
        ax2.set_ylabel("Prefix Cache Hit Rate", color="#D97706")
        ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax2.set_ylim(-0.05, 1.05)
        ax2.tick_params(axis="y", labelcolor="#D97706")

        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
    ax.set_xlabel("Elapsed Time (s)")
    ax.set_title("(c) KV Cache & Prefix Hit Rate Timeline",
                 fontweight="bold")
    ax.grid(True, alpha=0.15)

    fig.suptitle(
        "Fig 2: Truncation Deep-Dive — launch_poster_page_pack (32K, 3 Skills)",
        fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, out_dir, "fig2_truncation_deep_dive")


# ── Figure 3: Context Overhead & 32K-vs-16K Comparison ───────────────────────

def fig3_context_overhead(d32: list[BenchData], d16: list[BenchData],
                          out_dir: str):
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # ── (a) 32K overhead breakdown for all 8 benchmarks ──
    ax = axes[0]
    repos = [bd.repo for bd in d32]
    labels = [bd.short for bd in d32]
    n = len(repos)
    x = np.arange(n)

    ideal_toks = []
    overhead_toks = []
    total_toks = []
    for bd in d32:
        total = sum(c["prompt_tokens"] for c in bd.calls)
        ideal = bd.calls[0]["prompt_tokens"] * len(bd.calls)
        total_toks.append(total)
        ideal_toks.append(ideal)
        overhead_toks.append(total - ideal)

    ax.bar(x, [t / 1000 for t in ideal_toks], width=0.55, color="#059669",
           alpha=0.8, label="First-call baseline", zorder=3)
    ax.bar(x, [t / 1000 for t in overhead_toks], width=0.55,
           bottom=[t / 1000 for t in ideal_toks], color="#DC2626", alpha=0.7,
           label="Context accumulation overhead", zorder=3)

    for i in range(n):
        total = total_toks[i]
        ovhd = overhead_toks[i]
        pct = ovhd / total * 100 if total > 0 else 0
        ax.text(i, total / 1000 + 8, f"{pct:.0f}%",
                ha="center", fontsize=9, fontweight="bold", color="#DC2626")

    ax.set_xticks(x)
    skill_labels = [f"{labels[i]}\n({SKILL_COUNTS.get(repos[i], '?')} skills)"
                    for i in range(n)]
    ax.set_xticklabels(skill_labels, fontsize=8)
    ax.set_ylabel("Total Prompt Tokens (K)")
    ax.legend(fontsize=9, loc="upper left")
    ax.set_title(
        "(a) Context Accumulation Overhead — 32K Window, 8 Benchmarks",
        fontweight="bold")
    ax.grid(True, alpha=0.15, axis="y")

    avg_pct = np.mean([overhead_toks[i] / total_toks[i] * 100
                       for i in range(n) if total_toks[i] > 0])
    ax.text(0.97, 0.95,
            f"Avg overhead: {avg_pct:.0f}%\n"
            f"System prompt: ~16.9K tok/call\n"
            f"(51% of 32K window)",
            transform=ax.transAxes, fontsize=9, va="top", ha="right",
            bbox=dict(boxstyle="round", fc="#F0F9FF", ec="#2563EB", alpha=0.9))

    # ── (b) 32K vs 16K: elapsed time + call count comparison ──
    ax = axes[1]
    repos_common = [b.repo for b in d32 if any(x.repo == b.repo for x in d16)]
    nc = len(repos_common)
    x2 = np.arange(nc)
    w = 0.35

    elapsed_32 = [next(b for b in d32 if b.repo == r).total_elapsed
                  for r in repos_common]
    elapsed_16 = [next(b for b in d16 if b.repo == r).total_elapsed
                  for r in repos_common]
    calls_32 = [len(next(b for b in d32 if b.repo == r).calls)
                for r in repos_common]
    calls_16 = [len(next(b for b in d16 if b.repo == r).calls)
                for r in repos_common]
    labels_c = [SHORT_NAMES[r] for r in repos_common]

    ax.bar(x2 - w / 2, elapsed_32, w, color="#2563EB", alpha=0.8,
           label="32K ctx (elapsed)")
    ax.bar(x2 + w / 2, elapsed_16, w, color="#059669", alpha=0.8,
           label="16K ctx (elapsed)")

    for i in range(nc):
        diff_pct = (elapsed_16[i] - elapsed_32[i]) / elapsed_32[i] * 100
        sign = "+" if diff_pct > 0 else ""
        clr = "#DC2626" if diff_pct > 0 else "#059669"
        y_pos = max(elapsed_32[i], elapsed_16[i]) + 15
        ax.text(i, y_pos,
                f"16K {sign}{diff_pct:.0f}%\n({calls_32[i]}→{calls_16[i]} calls)",
                ha="center", fontsize=7, color=clr, fontweight="bold")

    ax.set_xticks(x2)
    ax.set_xticklabels(labels_c, fontsize=8, rotation=30, ha="right")
    ax.set_ylabel("Total Elapsed Time (seconds)")
    ax.legend(fontsize=9, loc="upper left")
    ax.set_title(
        "(b) 32K vs 16K Elapsed Time — 6.3x More Tokens, Mixed Speed Results",
        fontweight="bold")
    ax.grid(True, alpha=0.15, axis="y")

    faster_32_count = sum(1 for i in range(nc) if elapsed_32[i] < elapsed_16[i])
    sum_tok32 = sum(sum(c["prompt_tokens"] for c in
                        next(b for b in d32 if b.repo == r).calls)
                    for r in repos_common)
    sum_tok16 = sum(sum(c["prompt_tokens"] for c in
                        next(b for b in d16 if b.repo == r).calls)
                    for r in repos_common)
    ax.text(0.97, 0.95,
            f"32K faster in {faster_32_count}/{nc} tasks\n"
            f"32K: {sum_tok32/1e6:.1f}M prompt tok\n"
            f"16K: {sum_tok16/1e6:.1f}M prompt tok\n"
            f"Ratio: {sum_tok32/sum_tok16:.1f}x",
            transform=ax.transAxes, fontsize=8, va="top", ha="right",
            bbox=dict(boxstyle="round", fc="#F0F9FF", ec="#2563EB", alpha=0.9))

    fig.suptitle(
        "Fig 3: Context Overhead & Window-Size Paradox "
        "(Qwen3-14B, Anthropic Benchmark)",
        fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, out_dir, "fig3_context_overhead")


# ── Summary statistics ───────────────────────────────────────────────────────

def print_summary(d32: list[BenchData], d16: list[BenchData], out_dir: str):
    lines = []
    lines.append("=" * 110)
    lines.append("SegKV 14B Anthropic Characterization — Summary Statistics")
    lines.append("=" * 110)

    for tag, dset in [("32K", d32), ("16K", d16)]:
        lines.append(f"\n{'─' * 50} {tag} Context {'─' * 50}")
        for bd in dset:
            total_p = sum(c["prompt_tokens"] for c in bd.calls)
            total_c = sum(c["completion_tokens"] for c in bd.calls)
            pts = [c["prompt_tokens"] for c in bd.calls]
            ideal = bd.calls[0]["prompt_tokens"] * len(bd.calls)
            ovhd = (total_p - ideal) / total_p * 100 if total_p else 0
            truncs = detect_truncations(bd.calls)
            drops = detect_kv_drops(bd.timeline) if bd.timeline else "N/A"

            lines.append(
                f"  {bd.short:15s} | calls={len(bd.calls):2d} | "
                f"elapsed={bd.total_elapsed:7.1f}s | "
                f"prompt={total_p:>8,} | compl={total_c:>6,} | "
                f"overhead={ovhd:4.0f}% | "
                f"range=[{min(pts):,}..{max(pts):,}] | "
                f"truncations={len(truncs)} | kv_drops={drops} | "
                f"skills={SKILL_COUNTS.get(bd.repo, '?')}"
            )

    text = "\n".join(lines)
    print(text)
    with open(os.path.join(out_dir, "summary_stats_14B.txt"), "w") as f:
        f.write(text)
    print(f"\n[OK] summary_stats_14B.txt")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SegKV 14B Anthropic characterization figures")
    parser.add_argument("--results-dir",
                        default="results/03_14B_anthropic/multurn_bench")
    parser.add_argument("--out-dir",
                        default="results/03_14B_anthropic/multurn_bench/paper_figures")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    d32, d16 = [], []
    for repo in BENCHMARKS:
        bd = load_bench(args.results_dir, 32768, repo)
        if bd:
            d32.append(bd)
        bd = load_bench(args.results_dir, 16384, repo)
        if bd:
            d16.append(bd)

    if not d32:
        print("ERROR: no 32K data found", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded {len(d32)} benchmarks @ 32K, {len(d16)} @ 16K\n")

    fig1_combined_motivation(d32, d16, args.out_dir)
    fig2_truncation_deep_dive(d32, args.out_dir)
    fig3_context_overhead(d32, d16, args.out_dir)
    print()
    print_summary(d32, d16, args.out_dir)


if __name__ == "__main__":
    main()
