#!/usr/bin/env python3
"""Validate controlled shared-dominant Full/materialized/Shared points."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from capture_common import atomic_write_json
from shared_bank_gpu_closure.validate_closure import (
    exactly_one,
    request_matches,
    structured_events,
)


SHAPES = ("long-6k", "long-8k")
MODES = ("full", "materialized", "shared")
FOLLOWER_POINTS = (1, 4)
CALIBRATION_TOKENS = 124
ADMISSION_CAP = 2


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def response_events(
    events: list[dict[str, Any]], response_id: str
) -> list[dict[str, Any]]:
    return [
        event
        for event in events
        if request_matches(event.get("request_id"), response_id)
    ]


def validate_point(
    run_dir: Path,
    shape: str,
    mode: str,
    followers: int,
    geometry: dict[str, Any],
) -> dict[str, Any]:
    point_dir = run_dir / shape / mode / f"n{followers}"
    point = load_json(point_dir / "followers" / "manifest.json")
    if point.get("mode") != mode or point.get("followers") != followers:
        raise ValueError(f"point identity mismatch: {point_dir}")
    if point.get("completed") != followers or point.get("failed") != 0:
        raise ValueError(f"incomplete point: {point_dir}")
    if point.get("kv_usage_peak_delta") is None:
        raise ValueError(f"missing KV-pool signal: {point_dir}")

    log_path = point_dir / "vllm.log"
    log = log_path.read_text(encoding="utf-8", errors="replace")
    if (
        "EngineCore encountered an issue" in log
        or "Traceback (most recent call last)" in log
    ):
        raise ValueError(f"server failure: {log_path}")
    scheduler_events = structured_events(log, "SEGMENTIA_EVENT")
    profile_events = structured_events(log, "SEGMENTIA_PROFILE_EVENT")

    h2d_tokens: list[int] = []
    shared_blocks: list[int] | None = None
    max_reservations = 0
    max_leases = 0
    for index in range(followers):
        record = load_json(
            point_dir / "followers" / f"follower-{index:03d}.json"
        )
        response_id = record.get("response_id")
        if record.get("status") != "completed" or not response_id:
            raise ValueError(f"incomplete follower record: {point_dir}")
        own_scheduler = response_events(scheduler_events, response_id)
        own_profile = response_events(profile_events, response_id)
        fallbacks = [
            event
            for event in own_scheduler
            if event.get("event")
            in {"segmentia_lookup_overshot", "segmentia_lookup_local_fallback"}
            or event.get("phase") == "local_fallback"
        ]
        if fallbacks:
            raise ValueError(f"follower fallback in {point_dir}")

        if mode == "full":
            if any(
                str(event.get("event", "")).startswith("segmentia_")
                for event in own_scheduler + own_profile
            ):
                raise ValueError("Full arm unexpectedly entered Segmentia")
            continue

        exactly_one(scheduler_events, "segmentia_lookup_complete", response_id)
        h2d = exactly_one(profile_events, "segmentia_h2d_breakdown", response_id)
        h2d_tokens.append(int(h2d["tokens"]))
        activations = [
            event
            for event in own_scheduler
            if event.get("event") == "segmentia_shared_bank_activate"
        ]
        if mode == "materialized":
            if activations:
                raise ValueError("materialized arm unexpectedly used Shared Bank")
            continue

        activation = exactly_one(
            scheduler_events, "segmentia_shared_bank_activate", response_id
        )
        if activation.get("activation_mode") != "follower_correction_only":
            raise ValueError("Shared follower did not use correction-only mode")
        blocks = activation.get("shared_block_ids")
        expected_blocks = int(geometry["shared_b1_tokens"]) // 16
        if not isinstance(blocks, list) or len(blocks) != expected_blocks:
            raise ValueError("Shared Bank block geometry mismatch")
        if shared_blocks is None:
            shared_blocks = blocks
        elif shared_blocks != blocks:
            raise ValueError("followers did not share one physical Bank")
        if h2d.get("tokens") != CALIBRATION_TOKENS:
            raise ValueError("Shared follower H2D was not calibration-only")
        max_leases = max(max_leases, int(activation["lease_count"]))
        admission = exactly_one(
            scheduler_events, "segmentia_shared_pre_p_admit", response_id
        )
        if admission.get("cap") != ADMISSION_CAP:
            raise ValueError("Shared arm did not use durable cap=2 admission")
        max_reservations = max(
            max_reservations, int(admission["reservation_count"])
        )

    return {
        "shape": shape,
        "mode": mode,
        "followers": followers,
        "private_0_p_tokens": geometry["private_0_p_tokens"],
        "shared_b1_tokens": geometry["shared_b1_tokens"],
        "rho": geometry["rho_shared_over_private"],
        "theoretical_kv_gain": (
            followers * (1 + geometry["rho_shared_over_private"])
            / (followers + geometry["rho_shared_over_private"])
        ),
        "wall_s": point["wall_s"],
        "throughput_req_s": point["throughput_req_s"],
        "latency_p50_ms": point["latency_p50_ms"],
        "latency_p95_ms": point["latency_p95_ms"],
        "kv_usage_peak_delta": point["kv_usage_peak_delta"],
        "h2d_tokens_total": sum(h2d_tokens),
        "shared_blocks": len(shared_blocks or []),
        "max_reservations": max_reservations,
        "max_leases": max_leases,
        "log": str(log_path.resolve()),
    }


def write_figure(path: Path, comparisons: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [row["shape"] for row in comparisons]
    x = range(len(labels))
    width = 0.24
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    axes[0].bar([i - width for i in x], [r["n1_latency_speedup"] for r in comparisons], width, label="N=1 latency")
    axes[0].bar(list(x), [r["n4_throughput_speedup"] for r in comparisons], width, label="N=4 throughput")
    axes[0].bar([i + width for i in x], [r["theoretical_kv_gain_n4"] for r in comparisons], width, label="N=4 KV capacity")
    axes[0].axhline(2.0, color="black", linewidth=0.8, linestyle="--")
    axes[0].set_xticks(list(x), labels)
    axes[0].set_ylabel("Improvement over Full (×)")
    axes[0].legend(fontsize=8)

    axes[1].bar([i - width / 2 for i in x], [r["materialized_h2d_n4"] for r in comparisons], width, label="Materialized")
    axes[1].bar([i + width / 2 for i in x], [r["shared_h2d_n4"] for r in comparisons], width, label="Shared")
    axes[1].set_xticks(list(x), labels)
    axes[1].set_ylabel("Follower H2D tokens (N=4)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    geometries = {
        shape: load_json(args.run_dir / "requests" / shape / "manifest.json")
        for shape in SHAPES
    }
    rows = [
        validate_point(args.run_dir, shape, mode, followers, geometries[shape])
        for shape in SHAPES
        for mode in MODES
        for followers in FOLLOWER_POINTS
    ]
    indexed = {
        (row["shape"], row["mode"], row["followers"]): row for row in rows
    }
    comparisons = []
    for shape in SHAPES:
        full_n1 = indexed[(shape, "full", 1)]
        full_n4 = indexed[(shape, "full", 4)]
        materialized_n4 = indexed[(shape, "materialized", 4)]
        shared_n1 = indexed[(shape, "shared", 1)]
        shared_n4 = indexed[(shape, "shared", 4)]
        comparisons.append(
            {
                "shape": shape,
                "rho": shared_n4["rho"],
                "theoretical_kv_gain_n4": shared_n4["theoretical_kv_gain"],
                "n1_latency_speedup": (
                    full_n1["latency_p50_ms"] / shared_n1["latency_p50_ms"]
                ),
                "n4_throughput_speedup": (
                    shared_n4["throughput_req_s"] / full_n4["throughput_req_s"]
                ),
                "n4_wall_speedup": full_n4["wall_s"] / shared_n4["wall_s"],
                "materialized_h2d_n4": materialized_n4["h2d_tokens_total"],
                "shared_h2d_n4": shared_n4["h2d_tokens_total"],
                "measured_kv_peak_ratio_n4": (
                    materialized_n4["kv_usage_peak_delta"]
                    / shared_n4["kv_usage_peak_delta"]
                    if shared_n4["kv_usage_peak_delta"] > 0
                    else None
                ),
            }
        )

    mechanical_checks = {
        "geometry_shared_dominant": all(row["rho"] >= 2 for row in comparisons),
        "theoretical_n4_capacity_at_least_2x": all(
            row["theoretical_kv_gain_n4"] >= 2 for row in comparisons
        ),
        "shared_bank_is_one_copy": all(
            row["shared_blocks"] == row["shared_b1_tokens"] // 16
            for row in rows
            if row["mode"] == "shared"
        ),
        "shared_h2d_is_calibration_only": all(
            row["h2d_tokens_total"] == row["followers"] * CALIBRATION_TOKENS
            for row in rows
            if row["mode"] == "shared"
        ),
        "materialized_h2d_exceeds_shared": all(
            row["materialized_h2d_n4"] > row["shared_h2d_n4"]
            for row in comparisons
        ),
        "admission_cap_observed": all(
            row["max_reservations"] <= ADMISSION_CAP
            and row["max_leases"] <= ADMISSION_CAP
            for row in rows
            if row["mode"] == "shared"
        ),
    }
    long8 = next(row for row in comparisons if row["shape"] == "long-8k")
    performance_check = (
        long8["n1_latency_speedup"] >= 1.8
        or long8["n4_throughput_speedup"] >= 1.8
    )
    summary = {
        "schema_version": 1,
        "gate": (
            "go"
            if all(mechanical_checks.values()) and performance_check
            else "no_go"
        ),
        "mechanical_gate": (
            "go" if all(mechanical_checks.values()) else "no_go"
        ),
        "performance_promising_gate": "go" if performance_check else "no_go",
        "single_repetition_only": True,
        "run_dir": str(args.run_dir.resolve()),
        "checks": mechanical_checks,
        "comparisons": comparisons,
        "points": rows,
    }
    atomic_write_json(args.run_dir / "manifest.json", summary)
    for subdir in ("data", "tables", "figures"):
        (args.output_dir / subdir).mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output_dir / "data" / "summary.json", summary)

    point_fields = [
        "shape", "mode", "followers", "private_0_p_tokens", "shared_b1_tokens",
        "rho", "theoretical_kv_gain", "wall_s", "throughput_req_s",
        "latency_p50_ms", "latency_p95_ms", "kv_usage_peak_delta",
        "h2d_tokens_total", "shared_blocks", "max_reservations", "max_leases",
    ]
    with (args.output_dir / "tables" / "points.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=point_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in point_fields})
    comparison_fields = list(comparisons[0])
    with (args.output_dir / "tables" / "speedups.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=comparison_fields)
        writer.writeheader()
        writer.writerows(comparisons)
    write_figure(args.output_dir / "figures" / "shared_dominant_sanity.png", comparisons)

    lines = [
        "# Shared-dominant Skill scaling sanity",
        "",
        f"- Overall gate: **{summary['gate']}**",
        f"- Mechanical gate: **{summary['mechanical_gate']}**",
        f"- Performance promising gate: **{summary['performance_promising_gate']}**",
        "- Workload type: controlled token-geometry stress built by cycling real captured Skill tokens; it is not a naturally occurring single Skill.",
        "- The ratio is rho = |B1| / |[0,P)|, and G_KV(N) = N(1+rho)/(N+rho).",
        "",
    ]
    for row in comparisons:
        lines.append(
            f"- {row['shape']}: rho={row['rho']:.2f}, theoretical N=4 KV gain={row['theoretical_kv_gain_n4']:.2f}×, "
            f"N=1 latency speedup={row['n1_latency_speedup']:.2f}×, N=4 throughput speedup={row['n4_throughput_speedup']:.2f}×."
        )
    lines.extend(
        [
            "",
            "该结果只有单次 sanity。即使达到 2×，也只能进入多重复正式矩阵，不能直接写成最终性能结论。",
            "",
        ]
    )
    (args.output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    with (args.output_dir / "source_manifest.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(("artifact", "source"))
        writer.writerow(("data/summary.json", args.run_dir / "manifest.json"))
        writer.writerow(("tables/points.csv", args.run_dir))
        writer.writerow(("tables/speedups.csv", args.run_dir))
        writer.writerow(("figures/shared_dominant_sanity.png", args.run_dir))
        writer.writerow(("summary.md", args.run_dir))
    print(
        f"[validated] shared_dominant_gate={summary['gate']} "
        f"mechanical={summary['mechanical_gate']} "
        f"performance={summary['performance_promising_gate']} "
        f"output={args.output_dir}"
    )


if __name__ == "__main__":
    main()
