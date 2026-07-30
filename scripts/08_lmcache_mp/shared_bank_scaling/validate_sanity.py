#!/usr/bin/env python3
"""Validate the N=1/4 materialized-versus-shared scaling sanity."""
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


MODES = ("materialized", "shared")
FOLLOWER_POINTS = (1, 4)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_point(run_dir: Path, mode: str, followers: int) -> dict[str, Any]:
    point_dir = run_dir / mode / f"n{followers}"
    point = load_json(point_dir / "followers" / "manifest.json")
    if point.get("mode") != mode or point.get("followers") != followers:
        raise ValueError(f"point manifest identity mismatch: {point_dir}")
    if point.get("completed") != followers or point.get("failed") != 0:
        raise ValueError(f"point did not complete every follower: {point_dir}")
    if point.get("kv_usage_peak") is None or point.get("kv_usage_before") is None:
        raise ValueError(f"KV-pool metrics unavailable: {point_dir}")

    log_path = point_dir / "vllm.log"
    log = log_path.read_text(encoding="utf-8", errors="replace")
    if "EngineCore encountered an issue" in log or "Traceback (most recent call last)" in log:
        raise ValueError(f"server failure in {log_path}")
    scheduler_events = structured_events(log, "SEGMENTIA_EVENT")
    profile_events = structured_events(log, "SEGMENTIA_PROFILE_EVENT")
    fallback_events = [
        event
        for event in scheduler_events
        if event.get("event")
        in {"segmentia_lookup_overshot", "segmentia_lookup_local_fallback"}
        or event.get("phase") == "local_fallback"
    ]
    if fallback_events:
        raise ValueError(f"fallback events={len(fallback_events)} in {point_dir}")

    h2d_tokens: list[int] = []
    bank_blocks: list[int] | None = None
    lease_counts: list[int] = []
    for index in range(followers):
        record = load_json(
            point_dir / "followers" / f"follower-{index:03d}.json"
        )
        response_id = record.get("response_id")
        if record.get("status") != "completed" or not response_id:
            raise ValueError(f"incomplete follower record in {point_dir}")
        exactly_one(scheduler_events, "segmentia_lookup_complete", response_id)
        h2d = exactly_one(profile_events, "segmentia_h2d_breakdown", response_id)
        h2d_tokens.append(int(h2d["tokens"]))
        if mode == "shared":
            activation = exactly_one(
                scheduler_events, "segmentia_shared_bank_activate", response_id
            )
            if activation.get("activation_mode") != "follower_correction_only":
                raise ValueError("shared follower did not use correction-only mode")
            blocks = activation.get("shared_block_ids")
            if not isinstance(blocks, list) or not blocks:
                raise ValueError("shared follower has no physical Bank blocks")
            if bank_blocks is None:
                bank_blocks = blocks
            elif blocks != bank_blocks:
                raise ValueError("shared followers used different Bank blocks")
            lease_counts.append(int(activation["lease_count"]))
        else:
            shared_activations = [
                event
                for event in scheduler_events
                if event.get("event") == "segmentia_shared_bank_activate"
                and request_matches(event.get("request_id"), response_id)
            ]
            if shared_activations:
                raise ValueError("materialized follower unexpectedly used Shared Bank")

    return {
        "mode": mode,
        "followers": followers,
        "wall_s": point["wall_s"],
        "throughput_req_s": point["throughput_req_s"],
        "latency_p50_ms": point["latency_p50_ms"],
        "latency_p95_ms": point["latency_p95_ms"],
        "kv_usage_before": point["kv_usage_before"],
        "kv_usage_peak": point["kv_usage_peak"],
        "kv_usage_peak_delta": point["kv_usage_peak_delta"],
        "h2d_tokens_per_follower": h2d_tokens,
        "h2d_tokens_total": sum(h2d_tokens),
        "shared_blocks": len(bank_blocks) if bank_blocks is not None else 0,
        "max_lease_count": max(lease_counts) if lease_counts else 0,
        "log": str(log_path.resolve()),
    }


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    shared_n4 = summary["points"]["shared_n4"]
    materialized_n4 = summary["points"]["materialized_n4"]
    lines = [
        "# Shared Skill scaling sanity",
        "",
        f"- Measurement gate: **{summary['gate']}**",
        "- Modes: per-request materialization vs. non-materialized Shared Bank",
        "- Concurrency points: N=1,4; one run per point",
        f"- Shared N=4 physical Bank blocks: {shared_n4['shared_blocks']}",
        f"- Shared N=4 maximum leases: {shared_n4['max_lease_count']}",
        f"- Materialized/shared N=4 follower H2D tokens: "
        f"{materialized_n4['h2d_tokens_total']}/{shared_n4['h2d_tokens_total']}",
        "",
        "本 sanity 只验证测量链、复用路径和资源信号是否可比较。每点只有一次运行，"
        "p95、吞吐和 KV-pool 使用率均不得作为 3× 性能结论。",
        "",
        "未生成核心图：四个单次采样点不足以支持趋势图；完整 N=1/2/4/8、三重复矩阵通过后再出图。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = [
        validate_point(args.run_dir, mode, followers)
        for mode in MODES
        for followers in FOLLOWER_POINTS
    ]
    points = {f"{row['mode']}_n{row['followers']}": row for row in rows}
    shared_n1 = points["shared_n1"]
    shared_n4 = points["shared_n4"]
    materialized_n1 = points["materialized_n1"]
    materialized_n4 = points["materialized_n4"]

    checks = {
        "shared_bank_blocks_constant": (
            shared_n1["shared_blocks"] == shared_n4["shared_blocks"] == 27
        ),
        "shared_h2d_lower_at_n4": (
            shared_n4["h2d_tokens_total"] < materialized_n4["h2d_tokens_total"]
        ),
        "materialized_h2d_scales_with_n": (
            materialized_n4["h2d_tokens_total"]
            == 4 * materialized_n1["h2d_tokens_total"]
        ),
        "shared_h2d_scales_by_calibration_only": (
            shared_n4["h2d_tokens_total"]
            == 4 * shared_n1["h2d_tokens_total"]
        ),
        "kv_usage_signal_direction": (
            materialized_n4["kv_usage_peak_delta"]
            > shared_n4["kv_usage_peak_delta"]
        ),
    }
    summary = {
        "schema_version": 1,
        "gate": "go" if all(checks.values()) else "no_go",
        "run_dir": str(args.run_dir.resolve()),
        "checks": checks,
        "points": points,
    }
    atomic_write_json(args.run_dir / "manifest.json", summary)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "data").mkdir(exist_ok=True)
    (args.output_dir / "tables").mkdir(exist_ok=True)
    (args.output_dir / "figures").mkdir(exist_ok=True)
    atomic_write_json(args.output_dir / "data" / "summary.json", summary)
    with (args.output_dir / "tables" / "sanity.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fields = [
            "mode", "followers", "wall_s", "throughput_req_s",
            "latency_p50_ms", "latency_p95_ms", "kv_usage_before",
            "kv_usage_peak", "kv_usage_peak_delta", "h2d_tokens_total",
            "shared_blocks", "max_lease_count",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})
    write_summary(args.output_dir / "summary.md", summary)
    with (args.output_dir / "source_manifest.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(("artifact", "source"))
        writer.writerow(("data/summary.json", args.run_dir / "manifest.json"))
        writer.writerow(("tables/sanity.csv", args.run_dir))
        writer.writerow(("summary.md", args.run_dir))
    print(
        f"[validated] scaling_sanity_gate={summary['gate']} "
        f"checks={sum(checks.values())}/{len(checks)} output={args.output_dir}"
    )


if __name__ == "__main__":
    main()
