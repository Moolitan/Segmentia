#!/usr/bin/env python3
"""
Visualize multiturn sequence traces — per-request level charts.

Generates separate PNG files:
  Per-request:
    1. latency.png
    2. prompt_tokens.png
    3. completion_tokens.png
  Engine timeline (from vllm_timeline):
    4. timeline_kv_cache_usage.png
    5. timeline_prompt_throughput.png
    6. timeline_generation_throughput.png
    7. timeline_prefix_cache_hit_rate.png

X-axis: global request index 0, 1, 2, ...
Turn boundaries are shown as vertical dashed lines.
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def load_traces(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_series(llm_calls: list[dict]) -> dict:
    xs = list(range(len(llm_calls)))
    latencies, prompt_toks, comp_toks = [], [], []

    for c in llm_calls:
        # Per-request latency is collected at the transport boundary.
        latencies.append(c.get("request_latency_seconds"))
        prompt_toks.append(c.get("prompt_tokens", 0))
        comp_toks.append(c.get("completion_tokens", 0))

    return {
        "xs": xs,
        "latencies": latencies,
        "prompt_tokens": prompt_toks,
        "completion_tokens": comp_toks,
    }


def get_turn_boundaries(llm_calls: list[dict]) -> list[int]:
    boundaries = []
    prev_turn = None
    for i, c in enumerate(llm_calls):
        tn = c.get("turn_number", 1)
        if prev_turn is not None and tn != prev_turn:
            boundaries.append(i)
        prev_turn = tn
    return boundaries


def _plot_single(xs, ys, boundaries, title, ylabel, color, fmt_fn, out_path,
                 seq_id, ylim=None, y_formatter=None, xlabel="Request Index",
                 boundary_offset=-0.5):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_title(f"{seq_id}  —  {title}", fontsize=13, fontweight="bold")

    marker_size = 6 if len(xs) <= 80 else 3
    ax.plot(xs, ys, marker="o", markersize=marker_size, linewidth=2,
            color=color, markerfacecolor=color, markeredgecolor="white",
            markeredgewidth=0.8, zorder=3)

    # Only annotate individual points when there are few enough to read
    if len(xs) <= 30:
        for x, y in zip(xs, ys):
            ax.annotate(fmt_fn(y), (x, y), textcoords="offset points", xytext=(0, 10),
                        ha="center", fontsize=8, color=color, fontweight="bold")

    for bx in boundaries:
        ax.axvline(x=bx + boundary_offset, color="#94A3B8", linestyle="--", linewidth=1, alpha=0.7)

    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10, fontweight="bold")
    if len(xs) <= 30:
        ax.set_xticks(xs)
    ax.grid(True, alpha=0.25, linewidth=0.5)

    if ylim:
        ax.set_ylim(ylim)
    if y_formatter:
        ax.yaxis.set_major_formatter(y_formatter)

    plt.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[OK] saved: {out_path}")


def _plot_timeline(traces: dict, out_dir: str, seq_id: str):
    """Plot timeline-based charts from vllm_timeline data (follow vllm.log)."""
    timeline = traces["sequence"].get("vllm_timeline", [])
    if not timeline:
        print("[SKIP] vllm_timeline: no data available (need re-run to collect)")
        return

    elapsed = [s["elapsed_seconds"] for s in timeline]

    # Compute turn boundaries on the timeline using per-request start times.
    llm_calls = traces["sequence"].get("llm_calls", [])
    turn_time_boundaries = []
    prev_turn = None
    for c in llm_calls:
        tn = c.get("turn_number", 1)
        if prev_turn is not None and tn != prev_turn:
            t = c.get("request_started_at")
            if t is not None and timeline:
                t0 = timeline[0]["timestamp"]
                turn_time_boundaries.append(t - t0)
        prev_turn = tn

    # 1. KV Cache Usage (timeline) — exclude zero values
    kv = [s.get("kv_cache_usage_perc") for s in timeline]
    if any(v is not None and v > 0 for v in kv):
        mask = [(v is not None and v > 0) for v in kv]
        kv_elapsed = [t for t, m in zip(elapsed, mask) if m]
        ys = [v for v, m in zip(kv, mask) if m]
        _plot_single(
            kv_elapsed, ys, turn_time_boundaries,
            title="KV Cache Usage (timeline)",
            ylabel="Usage (%)",
            color="#7C3AED",
            fmt_fn=lambda y: f"{y * 100:.1f}%",
            out_path=os.path.join(out_dir, "timeline_kv_cache_usage.png"),
            seq_id=seq_id,
            ylim=(-0.05, max(0.2, max(ys) * 1.3 + 0.05)),
            y_formatter=mticker.PercentFormatter(1.0),
            xlabel="Elapsed Time (s)",
            boundary_offset=0,
        )

    # 2. Avg Prompt Throughput (timeline)
    pt = [s.get("avg_prompt_throughput") for s in timeline]
    if any(v is not None for v in pt):
        ys = [v if v is not None else 0.0 for v in pt]
        _plot_single(
            elapsed, ys, turn_time_boundaries,
            title="Avg Prompt Throughput (timeline)",
            ylabel="tokens/s",
            color="#059669",
            fmt_fn=lambda y: f"{y:.1f}",
            out_path=os.path.join(out_dir, "timeline_prompt_throughput.png"),
            seq_id=seq_id,
            xlabel="Elapsed Time (s)",
            boundary_offset=0,
        )

    # 3. Avg Generation Throughput (timeline)
    gt = [s.get("avg_generation_throughput") for s in timeline]
    if any(v is not None for v in gt):
        ys = [v if v is not None else 0.0 for v in gt]
        _plot_single(
            elapsed, ys, turn_time_boundaries,
            title="Avg Generation Throughput (timeline)",
            ylabel="tokens/s",
            color="#D97706",
            fmt_fn=lambda y: f"{y:.1f}",
            out_path=os.path.join(out_dir, "timeline_generation_throughput.png"),
            seq_id=seq_id,
            xlabel="Elapsed Time (s)",
            boundary_offset=0,
        )

    # 4. Prefix Cache Hit Rate (timeline)
    hr = [s.get("prefix_cache_hit_rate") for s in timeline]
    if any(v is not None for v in hr):
        ys = [v if v is not None else 0.0 for v in hr]
        _plot_single(
            elapsed, ys, turn_time_boundaries,
            title="Prefix Cache Hit Rate (timeline)",
            ylabel="Hit Rate",
            color="#2563EB",
            fmt_fn=lambda y: f"{y:.1%}",
            out_path=os.path.join(out_dir, "timeline_prefix_cache_hit_rate.png"),
            seq_id=seq_id,
            ylim=(-0.05, 1.15),
            y_formatter=mticker.PercentFormatter(1.0),
            xlabel="Elapsed Time (s)",
            boundary_offset=0,
        )


def plot_traces(traces: dict, out_dir: str):
    seq = traces["sequence"]
    seq_id = seq.get("sequence_id", "unknown")
    llm_calls = seq["llm_calls"]
    n = len(llm_calls)
    if n == 0:
        print("[WARN] no llm_calls, nothing to plot.")
        return

    series = build_series(llm_calls)
    boundaries = get_turn_boundaries(llm_calls)
    xs = series["xs"]
    os.makedirs(out_dir, exist_ok=True)

    # 1. Latency
    if any(v is not None for v in series["latencies"]):
        ys = [v if v is not None else 0.0 for v in series["latencies"]]
        _plot_single(
            xs, ys, boundaries,
            title="Latency",
            ylabel="Seconds",
            color="#DC2626",
            fmt_fn=lambda y: f"{y:.1f}s",
            out_path=os.path.join(out_dir, "latency.png"),
            seq_id=seq_id,
        )
    else:
        print("[SKIP] latency: no data available")

    # 2. Prompt Tokens
    _plot_single(
        xs, series["prompt_tokens"], boundaries,
        title="Prompt Tokens (input)",
        ylabel="Tokens",
        color="#059669",
        fmt_fn=lambda y: f"{y:,.0f}",
        out_path=os.path.join(out_dir, "prompt_tokens.png"),
        seq_id=seq_id,
    )

    # 3. Completion Tokens
    _plot_single(
        xs, series["completion_tokens"], boundaries,
        title="Completion Tokens (output)",
        ylabel="Tokens",
        color="#D97706",
        fmt_fn=lambda y: f"{y:,.0f}",
        out_path=os.path.join(out_dir, "completion_tokens.png"),
        seq_id=seq_id,
    )

    # 4-7. Timeline charts (kv_cache, prompt throughput, generation throughput, prefix cache hit rate)
    _plot_timeline(traces, out_dir, seq_id)


def main():
    parser = argparse.ArgumentParser(description="Visualize multiturn per-request metrics")
    parser.add_argument("--input", required=True, help="Path to multiturn_sequence_traces.json")
    parser.add_argument("--out-dir", required=True, help="Output directory for figures")
    args = parser.parse_args()

    traces = load_traces(args.input)
    plot_traces(traces, args.out_dir)


if __name__ == "__main__":
    main()
