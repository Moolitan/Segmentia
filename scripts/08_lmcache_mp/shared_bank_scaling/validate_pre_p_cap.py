#!/usr/bin/env python3
"""Validate the N=4 Shared Bank pre-P admission cap diagnostic."""
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


CAPS = (1, 2, 4)
FOLLOWERS = 4


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def point_summary(run_dir: Path, cap: int) -> dict[str, Any]:
    point_dir = run_dir / f"cap{cap}"
    point = load_json(point_dir / "followers" / "manifest.json")
    if point.get("completed") != FOLLOWERS or point.get("failed") != 0:
        raise ValueError(f"cap={cap} did not complete every follower")
    if point.get("kv_usage_peak_delta") is None:
        raise ValueError(f"cap={cap} has no KV-pool samples")

    log_path = point_dir / "vllm.log"
    log = log_path.read_text(encoding="utf-8", errors="replace")
    if (
        "EngineCore encountered an issue" in log
        or "Traceback (most recent call last)" in log
    ):
        raise ValueError(f"cap={cap} server failed")
    scheduler_events = structured_events(log, "SEGMENTIA_EVENT")
    profile_events = structured_events(log, "SEGMENTIA_PROFILE_EVENT")
    fallbacks = [
        event
        for event in scheduler_events
        if event.get("event") == "segmentia_lookup_overshot"
        or event.get("phase") == "local_fallback"
    ]
    if fallbacks:
        raise ValueError(f"cap={cap} fallback events={len(fallbacks)}")

    admissions = []
    waits = []
    activations = []
    h2d_tokens = []
    reference_blocks: list[int] | None = None
    for index in range(FOLLOWERS):
        record = load_json(
            point_dir / "followers" / f"follower-{index:03d}.json"
        )
        response_id = record.get("response_id")
        if record.get("status") != "completed" or not response_id:
            raise ValueError(f"cap={cap} follower-{index:03d} incomplete")
        admission = exactly_one(
            scheduler_events, "segmentia_shared_pre_p_admit", response_id
        )
        if admission.get("cap") != cap:
            raise ValueError(f"cap={cap} admission recorded another cap")
        if admission.get("private_tokens") != 7312:
            raise ValueError(f"cap={cap} private-token estimate changed")
        if admission.get("private_blocks") != 457:
            raise ValueError(f"cap={cap} private-block estimate changed")
        admissions.append(admission)
        waits.extend(
            event
            for event in scheduler_events
            if event.get("event") == "segmentia_shared_pre_p_wait"
            and request_matches(event.get("request_id"), response_id)
        )

        activation = exactly_one(
            scheduler_events, "segmentia_shared_bank_activate", response_id
        )
        if activation.get("activation_mode") != "follower_correction_only":
            raise ValueError(f"cap={cap} follower did not use READY Bank")
        blocks = activation.get("shared_block_ids")
        if not isinstance(blocks, list) or len(blocks) != 27:
            raise ValueError(f"cap={cap} shared block geometry changed")
        if reference_blocks is None:
            reference_blocks = blocks
        elif blocks != reference_blocks:
            raise ValueError(f"cap={cap} followers used different Bank blocks")
        activations.append(activation)

        h2d = exactly_one(profile_events, "segmentia_h2d_breakdown", response_id)
        if h2d.get("tokens") != 124 or h2d.get("correction_only") is not True:
            raise ValueError(f"cap={cap} follower H2D is not calibration-only")
        h2d_tokens.append(int(h2d["tokens"]))

    max_reservations = max(int(row["reservation_count"]) for row in admissions)
    max_leases = max(int(row["lease_count"]) for row in activations)
    if max_reservations > cap or max_leases > cap:
        raise ValueError(f"cap={cap} was exceeded")
    return {
        "cap": cap,
        "followers": FOLLOWERS,
        "wall_s": point["wall_s"],
        "throughput_req_s": point["throughput_req_s"],
        "latency_p50_ms": point["latency_p50_ms"],
        "latency_p95_ms": point["latency_p95_ms"],
        "kv_usage_before": point["kv_usage_before"],
        "kv_usage_peak": point["kv_usage_peak"],
        "kv_usage_peak_delta": point["kv_usage_peak_delta"],
        "max_reservations": max_reservations,
        "max_leases": max_leases,
        "waited_followers": len(waits),
        "h2d_tokens_total": sum(h2d_tokens),
        "shared_blocks": len(reference_blocks or []),
        "log": str(log_path.resolve()),
    }


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    rows = summary["points"]
    lines = [
        "# Private-live-set-aware pre-P admission sanity",
        "",
        f"- Gate: **{summary['gate']}**",
        "- Workload: four distinct contexts using one READY 27-block Bank",
        "- Admission caps: 1, 2, 4",
        f"- KV-pool peak deltas: "
        f"{rows['cap1']['kv_usage_peak_delta']}/"
        f"{rows['cap2']['kv_usage_peak_delta']}/"
        f"{rows['cap4']['kv_usage_peak_delta']}",
        f"- Wall times: {rows['cap1']['wall_s']}/"
        f"{rows['cap2']['wall_s']}/{rows['cap4']['wall_s']} s",
        "",
        "该单重复诊断只判断 pre-P cap 是否控制私有 KV live set，不能作为正式吞吐或 p95 结果。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = [point_summary(args.run_dir, cap) for cap in CAPS]
    points = {f"cap{row['cap']}": row for row in rows}
    cap1, cap2, cap4 = (points[f"cap{cap}"] for cap in CAPS)
    checks = {
        "caps_observed": all(
            row["max_reservations"] == row["cap"]
            and row["max_leases"] <= row["cap"]
            for row in rows
        ),
        "shared_data_plane_constant": all(
            row["shared_blocks"] == 27
            and row["h2d_tokens_total"] == FOLLOWERS * 124
            for row in rows
        ),
        "kv_peak_monotonic": (
            cap1["kv_usage_peak_delta"]
            < cap2["kv_usage_peak_delta"]
            < cap4["kv_usage_peak_delta"]
        ),
        "cap2_reduces_cap4_peak": (
            cap2["kv_usage_peak_delta"]
            <= 0.75 * cap4["kv_usage_peak_delta"]
        ),
        "cap2_wall_not_regressed": cap2["wall_s"] <= 1.10 * cap4["wall_s"],
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
    with (args.output_dir / "tables" / "cap_sanity.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fields = [
            "cap", "followers", "wall_s", "throughput_req_s",
            "latency_p50_ms", "latency_p95_ms", "kv_usage_before",
            "kv_usage_peak", "kv_usage_peak_delta", "max_reservations",
            "max_leases", "waited_followers", "h2d_tokens_total",
            "shared_blocks",
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
        writer.writerow(("tables/cap_sanity.csv", args.run_dir))
        writer.writerow(("summary.md", args.run_dir))
    print(
        f"[validated] pre_p_cap_gate={summary['gate']} "
        f"checks={sum(checks.values())}/{len(checks)} output={args.output_dir}"
    )


if __name__ == "__main__":
    main()
