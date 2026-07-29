#!/usr/bin/env python3
"""Aggregate fixed-prefix latency runs and determine sustained break-even."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from benchmark_common import ARMS, atomic_write_json
from validate_run import events, profile_events, request_events


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def optional_one(
    records: list[dict[str, Any]], event_name: str
) -> dict[str, Any] | None:
    matches = [record for record in records if record.get("event") == event_name]
    if len(matches) > 1:
        raise ValueError(f"expected at most one {event_name}, got {len(matches)}")
    return matches[0] if matches else None


def collect_cpu_pipeline_rows(
    leaf: Path,
    timing_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Join request rows with SSD->CPU, P-wait, CPU-read and H2D events."""
    log_path = leaf / "vllm.log"
    control = events(log_path)
    profile = profile_events(log_path)
    pipeline_rows: list[dict[str, Any]] = []
    for row in timing_rows:
        response_id = row.get("response_id")
        if not isinstance(response_id, str) or not response_id:
            raise ValueError(f"missing response_id in {leaf / 'timings.jsonl'}")
        current_control = request_events(control, response_id)
        current_profile = request_events(profile, response_id)
        prefetch = optional_one(current_profile, "segmentia_cpu_prefetch_complete")
        cpu_hit = optional_one(current_profile, "segmentia_cpu_cache_hit")
        activation = optional_one(current_profile, "segmentia_cpu_activate")
        h2d = optional_one(current_profile, "segmentia_h2d_breakdown")
        cpu_reads = [
            event
            for event in current_profile
            if event.get("event") == "segmentia_storage_read"
            and event.get("storage_tier") == "cpu"
        ]
        waiting = optional_one(current_control, "segmentia_lookup_waiting")
        complete = optional_one(current_control, "segmentia_lookup_complete")
        wait_ms = 0.0
        if waiting is not None:
            if complete is None:
                raise ValueError(f"waiting request has no completion: {response_id}")
            wait_ms = (
                int(complete["monotonic_ns"]) - int(waiting["monotonic_ns"])
            ) / 1_000_000
            if wait_ms < 0:
                raise ValueError(f"negative P-boundary wait for {response_id}")
        pipeline_rows.append(
            {
                "arm": row["arm"],
                "skill_tokens": int(row["skill_tokens"]),
                "kind": row["kind"],
                "ordinal": int(row["ordinal"]),
                "source_tier": (
                    activation.get("source_tier", "") if activation else "fallback"
                ),
                "ssd_to_cpu_ms": (
                    round(float(prefetch["duration_ms"]), 6) if prefetch else ""
                ),
                "cpu_probe_ms": (
                    round(float(cpu_hit["duration_ms"]), 6) if cpu_hit else ""
                ),
                "p_boundary_wait_ms": round(wait_ms, 6),
                "cpu_read_ms": round(
                    sum(float(event["duration_ms"]) for event in cpu_reads), 6
                ),
                "cpu_read_events": len(cpu_reads),
                "h2d_gpu_ms": (
                    round(float(h2d["pure_h2d_gpu_ms"]), 6) if h2d else ""
                ),
            }
        )
    return pipeline_rows


def plot(rows: list[dict[str, Any]], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_arm[row["arm"]].append(row)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for arm in ARMS:
        arm_rows = sorted(by_arm.get(arm, []), key=lambda item: int(item["skill_tokens"]))
        if arm_rows:
            ax.plot(
                [row["skill_tokens"] for row in arm_rows],
                [row["median_ms"] for row in arm_rows],
                marker="o",
                linewidth=1.8,
                label=arm,
            )
    ax.set_xlabel("Skill length (tokens)")
    ax.set_ylabel("One-token end-to-end latency (ms)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "latency_vs_skill_length.png", dpi=180)
    plt.close(fig)

    prefix_rows = sorted(by_arm.get("prefix_256", []), key=lambda item: int(item["skill_tokens"]))
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.axhline(0.0, color="#555555", linewidth=1.0)
    ax.axhline(3.0, color="#888888", linewidth=1.0, linestyle="--")
    ax.plot(
        [row["skill_tokens"] for row in prefix_rows],
        [row["speedup_vs_full_pct"] for row in prefix_rows],
        marker="o",
        linewidth=1.8,
        color="#d55e00",
    )
    ax.set_xlabel("Skill length (tokens)")
    ax.set_ylabel("Prefix-256 speedup over Full (%)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "prefix_speedup_vs_skill_length.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--replicas", type=int, required=True)
    args = parser.parse_args()
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    replica_grouped: dict[tuple[int, str, int], list[float]] = defaultdict(list)
    sources: list[dict[str, str]] = []
    cpu_pipeline: list[dict[str, Any]] = []
    for replica in range(args.replicas):
        for arm in ARMS:
            leaf = args.run_dir / f"replica_{replica}" / arm
            manifest = json.loads((leaf / "manifest.json").read_text(encoding="utf-8"))
            if manifest.get("status") != "completed" or manifest.get("arm") != arm:
                raise ValueError(f"invalid manifest: {leaf}")
            sources.append({"artifact": f"replica_{replica}/{arm}", "source": str(leaf.resolve())})
            timing_rows = read_rows(leaf / "timings.jsonl")
            if bool(manifest.get("cpu_prefetch", False)):
                cpu_pipeline.extend(collect_cpu_pipeline_rows(leaf, timing_rows))
            for row in timing_rows:
                if row.get("kind") != "measure":
                    continue
                key = (arm, int(row["skill_tokens"]))
                grouped[key].append(float(row["elapsed_ms"]))
                replica_grouped[(replica, *key)].append(float(row["elapsed_ms"]))
    lengths = sorted({length for _, length in grouped})
    aggregate: list[dict[str, Any]] = []
    medians: dict[tuple[str, int], float] = {}
    for arm in ARMS:
        for length in lengths:
            values = grouped[(arm, length)]
            if not values:
                raise ValueError(f"missing measurements arm={arm} length={length}")
            median = statistics.median(values)
            medians[(arm, length)] = median
            aggregate.append(
                {
                    "arm": arm,
                    "skill_tokens": length,
                    "samples": len(values),
                    "median_ms": round(median, 6),
                    "p95_ms": round(percentile(values, 0.95), 6),
                    "speedup_vs_full_pct": 0.0,
                    "faster_replicas": "",
                }
            )
    for row in aggregate:
        arm = row["arm"]
        length = int(row["skill_tokens"])
        full = medians[("full", length)]
        current = medians[(arm, length)]
        row["speedup_vs_full_pct"] = round((full - current) / full * 100.0, 6)
        faster = 0
        for replica in range(args.replicas):
            full_replica = statistics.median(replica_grouped[(replica, "full", length)])
            arm_replica = statistics.median(replica_grouped[(replica, arm, length)])
            faster += int(arm_replica < full_replica)
        row["faster_replicas"] = faster

    prefix = {
        int(row["skill_tokens"]): row
        for row in aggregate
        if row["arm"] == "prefix_256"
    }
    valid_lengths = [length for length in lengths if length >= 640]
    required_replicas = math.ceil(args.replicas * 2 / 3)
    break_even: int | None = None
    for index, length in enumerate(valid_lengths):
        window = valid_lengths[index : index + 3]
        if len(window) < 3:
            continue
        first = prefix[length]
        sustained = all(float(prefix[item]["speedup_vs_full_pct"]) >= -2.0 for item in window)
        if (
            float(first["speedup_vs_full_pct"]) >= 3.0
            and int(first["faster_replicas"]) >= required_replicas
            and sustained
        ):
            break_even = length
            break
    gate = "go" if break_even is not None else "no_go"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table_fields = [
        "arm", "skill_tokens", "samples", "median_ms", "p95_ms",
        "speedup_vs_full_pct", "faster_replicas",
    ]
    write_csv(args.output_dir / "tables" / "latency_by_length.csv", aggregate, table_fields)
    if cpu_pipeline:
        write_csv(
            args.output_dir / "tables" / "cpu_pipeline_by_request.csv",
            cpu_pipeline,
            [
                "arm", "skill_tokens", "kind", "ordinal", "source_tier",
                "ssd_to_cpu_ms", "cpu_probe_ms", "p_boundary_wait_ms",
                "cpu_read_ms", "cpu_read_events", "h2d_gpu_ms",
            ],
        )
    decision = {
        "schema_version": 1,
        "gate": gate,
        "break_even_tokens": break_even,
        "criteria": {
            "minimum_speedup_pct": 3.0,
            "minimum_faster_replicas": required_replicas,
            "sustained_points": 3,
            "maximum_sustained_regression_pct": 2.0,
        },
        "replicas": args.replicas,
        "lengths": lengths,
    }
    atomic_write_json(args.output_dir / "tables" / "break_even.json", decision)
    write_csv(
        args.output_dir / "source_manifest.csv",
        sources,
        ["artifact", "source"],
    )
    plot(aggregate, args.output_dir)
    summary = (
        "# Fixed Prefix-256 latency and length break-even\n\n"
        f"- Gate: **{gate}**\n"
        f"- Break-even Skill length: **{break_even if break_even is not None else 'not observed'}**\n"
        f"- Independent service replicas: **{args.replicas}**\n"
        "- Metric: non-streaming end-to-end latency for one generated token; it is not labeled exact TTFT.\n\n"
        + (
            "Reuse arms use the SSD→CPU→GPU pipeline. Per-request promotion, P-boundary wait, CPU lookup, and pure H2D timings are recorded in `tables/cpu_pipeline_by_request.csv`.\n\n"
            if cpu_pipeline
            else ""
        )
        +
        "The gate requires at least 3% aggregate median improvement over Full, improvement in at least two-thirds of service replicas, and no regression worse than 2% at the candidate and next two larger tested lengths.\n"
    )
    (args.output_dir / "summary.md").write_text(summary, encoding="utf-8")
    print(f"[analyzed] gate={gate} break_even={break_even} output={args.output_dir}")


if __name__ == "__main__":
    main()
