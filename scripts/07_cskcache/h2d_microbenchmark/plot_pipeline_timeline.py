#!/usr/bin/env python3
"""Gantt-style visualization of to_gpu()'s pipelined path.

Reads the JSON produced by run_case.py's --timeline-output (one row per
prefetch_stream H2D+RoPE interval and one row per default_stream scatter
interval, per layer) and draws two horizontal swimlanes so overlap -- or its
absence -- between the two streams is visible directly, instead of only
inferable from aggregate before/after stage totals.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cskcache-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/cskcache-xdg-cache")

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULT_DIR = REPO_ROOT / "results" / "problem_exploration" / "h2d_microbenchmark"

LANES = {"prefetch_stream": 1, "default_stream": 0}
LANE_LABELS = {
    "prefetch_stream": "prefetch_stream\n(H2D + RoPE)",
    "default_stream": "default_stream\n(scatter into paged cache)",
}
STAGE_COLORS = {"h2d_rope": "#4C78A8", "scatter": "#54A24B"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeline-json", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--name", default="pipeline_timeline")
    parser.add_argument(
        "--max-layers",
        type=int,
        default=12,
        help="Only draw the first N layers (full 40 layers makes each bar "
        "too thin to read); the overlap pattern repeats every layer, so a "
        "handful is representative.",
    )
    return parser.parse_args()


def compute_overlap_ms(events: list[dict]) -> float:
    """Total time where a prefetch_stream interval and a default_stream
    interval are simultaneously active, computed by sweeping merged
    per-stream busy intervals against each other."""

    def merge(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
        merged: list[tuple[float, float]] = []
        for start, end in sorted(intervals):
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        return merged

    prefetch = merge(
        [(e["start_ms"], e["end_ms"]) for e in events if e["stream"] == "prefetch_stream"]
    )
    default = merge(
        [(e["start_ms"], e["end_ms"]) for e in events if e["stream"] == "default_stream"]
    )
    overlap = 0.0
    i = j = 0
    while i < len(prefetch) and j < len(default):
        lo = max(prefetch[i][0], default[j][0])
        hi = min(prefetch[i][1], default[j][1])
        if lo < hi:
            overlap += hi - lo
        if prefetch[i][1] < default[j][1]:
            i += 1
        else:
            j += 1
    return overlap


def plot(events: list[dict], num_layers_shown: int, output_stem: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 2.8))
    for event in events:
        lane = LANES[event["stream"]]
        ax.barh(
            lane,
            event["end_ms"] - event["start_ms"],
            left=event["start_ms"],
            height=0.5,
            color=STAGE_COLORS[event["stage"]],
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
    ax.set_yticks([0, 1])
    ax.set_yticklabels([LANE_LABELS["default_stream"], LANE_LABELS["prefetch_stream"]])
    ax.set_xlabel("Time since to_gpu() call start (ms)")
    ax.set_title(
        f"Pipelined to_gpu(): first {num_layers_shown} of 40 layers "
        "(blue=H2D+RoPE on prefetch stream, green=scatter on default stream)"
    )
    ax.xaxis.grid(True, color="#E4E4E0", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{output_stem}.png", dpi=200)
    fig.savefig(f"{output_stem}.pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    payload = json.loads(args.timeline_json.read_text())
    all_events = payload["events"]

    overlap_ms = compute_overlap_ms(all_events)
    span_ms = max(e["end_ms"] for e in all_events) - min(e["start_ms"] for e in all_events)

    shown_layers = sorted({e["layer"] for e in all_events})[: args.max_layers]
    shown_events = [e for e in all_events if e["layer"] in shown_layers]

    figure_dir = args.result_dir / "figures"
    data_path = args.result_dir / "data" / f"{args.name}.csv"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    with data_path.open("w", encoding="utf-8") as f:
        f.write("layer,stream,stage,start_ms,end_ms,duration_ms\n")
        for e in all_events:
            f.write(
                f"{e['layer']},{e['stream']},{e['stage']},"
                f"{e['start_ms']:.4f},{e['end_ms']:.4f},{e['end_ms']-e['start_ms']:.4f}\n"
            )

    output_stem = figure_dir / args.name
    plot(shown_events, len(shown_layers), output_stem)

    print(f"total_layers={payload['num_layers']}")
    print(f"total_events={len(all_events)}")
    print(f"call_span_ms={span_ms:.3f}")
    print(f"measured_overlap_ms={overlap_ms:.3f}")
    print(f"overlap_fraction_of_span={overlap_ms / span_ms:.3f}")
    print(f"data_csv={data_path}")
    print(f"figure_png={output_stem}.png")


if __name__ == "__main__":
    main()
