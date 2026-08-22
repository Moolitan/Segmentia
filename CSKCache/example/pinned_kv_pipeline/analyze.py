"""Draw the measured H2D and calibration/correction/install timeline."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


COMPUTE_PHASE_FIELDS = (
    "calibration_forward_ms",
    "calibration_commit_ms",
    "residual_correction_ms",
    "suffix_commit_ms",
)


def _load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _interval_union_ms(intervals: list[dict]) -> float:
    merged: list[list[float]] = []
    for item in sorted(intervals, key=lambda value: value["start_ms"]):
        start = float(item["start_ms"])
        end = float(item["end_ms"])
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return sum(end - start for start, end in merged)


def analyze(run_dir: Path) -> None:
    records = _load(run_dir / "cskcache_profile.jsonl")
    request_result = json.loads(
        (run_dir / "request_result.json").read_text(encoding="utf-8")
    )
    h2d = sorted(
        (item for item in records if item["event"] == "cskcache_h2d_layer"),
        key=lambda item: item["layer"],
    )
    compute_record = next(
        item
        for item in records
        if item["event"] == "cskcache_layer_compute"
    )
    compute = compute_record["calibration_correct_install"]
    transform = sorted(
        (
            item
            for item in records
            if item["event"] == "cskcache_stage_transform_layer"
        ),
        key=lambda item: item["layer"],
    )

    origin = min(
        min(item["start_ms"] for item in h2d),
        min(item["start_ms"] for item in compute),
        min(item["start_ms"] for item in transform),
    )
    fig, ax = plt.subplots(figsize=(8.2, 2.75))
    colors = ("#4C78A8", "#59A14F", "#E45756")
    for lane, intervals in enumerate((h2d, transform, compute)):
        for item in intervals:
            start = item["start_ms"] - origin
            width = item["end_ms"] - item["start_ms"]
            ax.broken_barh([(start, width)], (lane * 12, 8), facecolors=colors[lane])
    ax.set_yticks(
        (4, 16, 28),
        labels=("H2D", "RoPE/stage", "Calibration + KV install"),
    )
    ax.set_xlabel("Time (ms)")
    ax.set_ylim(-2, 38)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(run_dir / "pipeline.png", dpi=220)
    plt.close(fig)

    overlap_ms = 0.0
    for left in h2d:
        for right in compute:
            overlap_ms += max(
                0.0,
                min(left["end_ms"], right["end_ms"])
                - max(left["start_ms"], right["start_ms"]),
            )
    pipeline_end = max(
        max(item["end_ms"] for item in h2d),
        max(item["end_ms"] for item in compute),
        max(item["end_ms"] for item in transform),
    )
    profiled_union_ms = _interval_union_ms([*h2d, *transform, *compute])
    transform_record = next(
        item for item in records if item["event"] == "cskcache_stage_transform"
    )
    h2d_by_layer = {int(item["layer"]): item for item in h2d}
    compute_phase_totals = {
        field: sum(float(item[field]) for item in compute)
        for field in COMPUTE_PHASE_FIELDS
    }
    compute_gpu_ms = sum(
        item["end_ms"] - item["start_ms"] for item in compute
    )
    compute_phase_sum_ms = sum(compute_phase_totals.values())
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "host_layout": request_result["host_layout"],
                "h2d_layers": len(h2d),
                "compute_layers": len(compute),
                "h2d_gpu_ms": sum(
                    item["end_ms"] - item["start_ms"] for item in h2d
                ),
                "h2d_cpu_submit_ms": sum(
                    float(item.get("cpu_submit_ms", 0.0)) for item in h2d
                ),
                "compute_gpu_ms": compute_gpu_ms,
                **{
                    f"{field.removesuffix('_ms')}_gpu_ms": value
                    for field, value in compute_phase_totals.items()
                },
                "compute_phase_sum_ms": compute_phase_sum_ms,
                "compute_phase_unattributed_ms": (
                    compute_gpu_ms - compute_phase_sum_ms
                ),
                "stage_transform_gpu_ms": sum(
                    item["end_ms"] - item["start_ms"] for item in transform
                ),
                "sync_wait_cpu_ms": transform_record["sync_wait_cpu_ms"],
                "overlap_ms": overlap_ms,
                "pipeline_span_ms": pipeline_end - origin,
                "profiled_union_ms": profiled_union_ms,
                "unattributed_span_ms": (
                    pipeline_end - origin - profiled_union_ms
                ),
                "tail_h2d_ms": {
                    str(layer): (
                        h2d_by_layer[layer]["end_ms"]
                        - h2d_by_layer[layer]["start_ms"]
                    )
                    for layer in (38, 39)
                    if layer in h2d_by_layer
                },
                "request_elapsed_ms": request_result["request_elapsed_ms"],
                "generated_token_ids": request_result["generated_token_ids"],
                "execution_order": compute_record.get(
                    "execution_order", "legacy_compute_first"
                ),
                "synchronization": compute_record["synchronization"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
