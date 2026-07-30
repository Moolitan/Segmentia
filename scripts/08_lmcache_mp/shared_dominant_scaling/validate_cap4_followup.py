#!/usr/bin/env python3
"""Validate the shared-dominant N=4, pre-P cap=4 follow-up."""
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
FOLLOWERS = 4
CAP = 4
CALIBRATION_TOKENS = 124


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


def validate_cap4_point(
    baseline_run_dir: Path,
    run_dir: Path,
    shape: str,
) -> dict[str, Any]:
    geometry = load_json(
        baseline_run_dir / "requests" / shape / "manifest.json"
    )
    point_dir = run_dir / shape / "cap4"
    point = load_json(point_dir / "followers" / "manifest.json")
    if point.get("mode") != "shared" or point.get("followers") != FOLLOWERS:
        raise ValueError(f"point identity mismatch: {point_dir}")
    if point.get("completed") != FOLLOWERS or point.get("failed") != 0:
        raise ValueError(f"incomplete point: {point_dir}")

    log_path = point_dir / "vllm.log"
    log = log_path.read_text(encoding="utf-8", errors="replace")
    if (
        "EngineCore encountered an issue" in log
        or "Traceback (most recent call last)" in log
    ):
        raise ValueError(f"server failure: {log_path}")
    scheduler_events = structured_events(log, "SEGMENTIA_EVENT")
    profile_events = structured_events(log, "SEGMENTIA_PROFILE_EVENT")

    owner = load_json(point_dir / "owner.json")
    owner_response_id = owner.get("response_id")
    if owner.get("status") != "completed" or not owner_response_id:
        raise ValueError(f"owner did not complete: {point_dir}")
    publish = exactly_one(
        scheduler_events, "segmentia_shared_bank_publish", owner_response_id
    )
    if publish.get("bank_state") != "ready" or publish.get("success") is not True:
        raise ValueError(f"owner did not publish one READY Bank: {point_dir}")

    expected_blocks = int(geometry["shared_b1_tokens"]) // 16
    published_blocks = publish.get("shared_block_ids")
    if not isinstance(published_blocks, list) or len(published_blocks) != expected_blocks:
        raise ValueError(f"published Bank geometry mismatch: {point_dir}")

    max_reservations = 0
    max_leases = 0
    h2d_tokens_total = 0
    follower_blocks: list[int] | None = None
    pre_p_waits = 0
    for index in range(FOLLOWERS):
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

        exactly_one(scheduler_events, "segmentia_lookup_complete", response_id)
        activation = exactly_one(
            scheduler_events, "segmentia_shared_bank_activate", response_id
        )
        if activation.get("activation_mode") != "follower_correction_only":
            raise ValueError("Shared follower did not use correction-only mode")
        blocks = activation.get("shared_block_ids")
        if blocks != published_blocks:
            raise ValueError("followers did not reference the published Bank")
        if follower_blocks is None:
            follower_blocks = blocks
        elif follower_blocks != blocks:
            raise ValueError("followers did not share one physical Bank")
        max_leases = max(max_leases, int(activation["lease_count"]))

        admission = exactly_one(
            scheduler_events, "segmentia_shared_pre_p_admit", response_id
        )
        if admission.get("cap") != CAP:
            raise ValueError("Shared arm did not use cap=4 admission")
        max_reservations = max(
            max_reservations, int(admission["reservation_count"])
        )
        pre_p_waits += sum(
            event.get("event") == "segmentia_shared_pre_p_wait"
            for event in own_scheduler
        )

        h2d = exactly_one(profile_events, "segmentia_h2d_breakdown", response_id)
        if h2d.get("tokens") != CALIBRATION_TOKENS:
            raise ValueError("Shared follower H2D was not calibration-only")
        h2d_tokens_total += int(h2d["tokens"])

    return {
        "shape": shape,
        "private_0_p_tokens": int(geometry["private_0_p_tokens"]),
        "shared_b1_tokens": int(geometry["shared_b1_tokens"]),
        "wall_s": float(point["wall_s"]),
        "throughput_req_s": float(point["throughput_req_s"]),
        "latency_p50_ms": float(point["latency_p50_ms"]),
        "latency_p95_ms": float(point["latency_p95_ms"]),
        "kv_usage_peak_delta": point.get("kv_usage_peak_delta"),
        "h2d_tokens_total": h2d_tokens_total,
        "shared_blocks": len(follower_blocks or []),
        "max_reservations": max_reservations,
        "max_leases": max_leases,
        "pre_p_waits": pre_p_waits,
        "log": str(log_path.resolve()),
    }


def build_comparison(
    shape: str,
    cap4: dict[str, Any],
    baseline_summary: dict[str, Any],
) -> dict[str, Any]:
    baseline = {
        (row["shape"], row["mode"], row["followers"]): row
        for row in baseline_summary["points"]
    }
    materialized = baseline[(shape, "materialized", FOLLOWERS)]
    cap2 = baseline[(shape, "shared", FOLLOWERS)]
    return {
        "shape": shape,
        "materialized_wall_s": float(materialized["wall_s"]),
        "cap2_wall_s": float(cap2["wall_s"]),
        "cap4_wall_s": float(cap4["wall_s"]),
        "materialized_throughput_req_s": float(
            materialized["throughput_req_s"]
        ),
        "cap2_throughput_req_s": float(cap2["throughput_req_s"]),
        "cap4_throughput_req_s": float(cap4["throughput_req_s"]),
        "cap4_vs_cap2_wall_speedup": float(cap2["wall_s"]) / cap4["wall_s"],
        "cap4_vs_materialized_wall_ratio": (
            cap4["wall_s"] / float(materialized["wall_s"])
        ),
    }


def write_figure(path: Path, comparisons: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [row["shape"] for row in comparisons]
    x = list(range(len(labels)))
    width = 0.24
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    ax.bar(
        [index - width for index in x],
        [row["materialized_wall_s"] for row in comparisons],
        width,
        label="Materialized",
    )
    ax.bar(
        x,
        [row["cap2_wall_s"] for row in comparisons],
        width,
        label="Shared cap=2",
    )
    ax.bar(
        [index + width for index in x],
        [row["cap4_wall_s"] for row in comparisons],
        width,
        label="Shared cap=4",
    )
    ax.set_xticks(x, labels)
    ax.set_ylabel("N=4 wall time (s)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    baseline_summary = load_json(args.baseline_run_dir / "manifest.json")
    if baseline_summary.get("gate") != "go":
        raise ValueError("baseline shared-dominant sanity did not pass")
    rows = [
        validate_cap4_point(args.baseline_run_dir, args.run_dir, shape)
        for shape in SHAPES
    ]
    comparisons = [
        build_comparison(row["shape"], row, baseline_summary) for row in rows
    ]
    indexed = {row["shape"]: row for row in comparisons}
    mechanical_checks = {
        "four_durable_admissions_observed": all(
            row["max_reservations"] == CAP for row in rows
        ),
        "no_pre_p_cap_wait": all(row["pre_p_waits"] == 0 for row in rows),
        "one_physical_bank": all(
            row["shared_blocks"] == row["shared_b1_tokens"] // 16
            for row in rows
        ),
        "calibration_only_h2d": all(
            row["h2d_tokens_total"] == FOLLOWERS * CALIBRATION_TOKENS
            for row in rows
        ),
    }
    performance_checks = {
        "long8_cap4_improves_over_cap2_by_20pct": (
            indexed["long-8k"]["cap4_vs_cap2_wall_speedup"] >= 1.20
        ),
        "cap4_within_10pct_of_materialized": all(
            row["cap4_vs_materialized_wall_ratio"] <= 1.10
            for row in comparisons
        ),
    }
    summary = {
        "schema_version": 1,
        "gate": (
            "go"
            if all(mechanical_checks.values())
            and all(performance_checks.values())
            else "no_go"
        ),
        "single_repetition_only": True,
        "baseline_run_dir": str(args.baseline_run_dir.resolve()),
        "run_dir": str(args.run_dir.resolve()),
        "checks": {
            "mechanical": mechanical_checks,
            "performance": performance_checks,
        },
        "comparisons": comparisons,
        "points": rows,
    }
    atomic_write_json(args.run_dir / "manifest.json", summary)
    for subdir in ("data", "tables", "figures"):
        (args.output_dir / subdir).mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output_dir / "data" / "summary.json", summary)

    with (args.output_dir / "tables" / "cap4_points.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "tables" / "cap4_comparison.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)
    write_figure(
        args.output_dir / "figures" / "cap4_wall_time.png", comparisons
    )

    lines = [
        "# Shared-dominant dynamic-admission diagnostic",
        "",
        f"- Gate: **{summary['gate']}**",
        "- This is a single-repetition cap=4 diagnostic, not a final paper result.",
        "- `wall_s` covers the four concurrently released followers; owner Bank construction is measured separately and excluded.",
        "",
        "| Shape | Materialized (s) | Shared cap=2 (s) | Shared cap=4 (s) | cap=4 vs cap=2 | cap=4 / Materialized |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in comparisons:
        lines.append(
            f"| {row['shape']} | {row['materialized_wall_s']:.3f} | "
            f"{row['cap2_wall_s']:.3f} | {row['cap4_wall_s']:.3f} | "
            f"{row['cap4_vs_cap2_wall_speedup']:.3f}× | "
            f"{row['cap4_vs_materialized_wall_ratio']:.3f}× |"
        )
    lines.extend(
        [
            "",
            "Go means cap=4 admitted all four requests without duplicating the Bank or expanding H2D beyond four calibration windows, improved Long-8K wall time by at least 20%, and stayed within 10% of Materialized on both shapes.",
            "No-Go means the fixed cap is not the sole explanation; the next diagnosis must separate Cascade Attention execution from shared-batch scheduling overhead.",
            "",
        ]
    )
    (args.output_dir / "summary.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    with (args.output_dir / "source_manifest.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(("artifact", "source"))
        writer.writerow(("data/summary.json", args.run_dir / "manifest.json"))
        writer.writerow(("tables/cap4_points.csv", args.run_dir))
        writer.writerow(
            ("tables/cap4_comparison.csv", args.baseline_run_dir)
        )
        writer.writerow(("figures/cap4_wall_time.png", args.run_dir))
        writer.writerow(("summary.md", args.run_dir))
    print(
        f"[validated] shared_dominant_admission_gate={summary['gate']} "
        f"mechanical={'go' if all(mechanical_checks.values()) else 'no_go'} "
        f"performance={'go' if all(performance_checks.values()) else 'no_go'} "
        f"output={args.output_dir}"
    )


if __name__ == "__main__":
    main()
