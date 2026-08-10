#!/usr/bin/env python3
"""Validate cross-mode identity and publish the 8K-Skill latency summary."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from workload import MODES, atomic_write_json


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot compute a percentile of an empty sample")
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_run(run_dir: Path, replicas: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    all_rows: list[dict[str, Any]] = []
    validations: list[dict[str, Any]] = []
    for replica in range(replicas):
        for mode in MODES:
            leaf = run_dir / f"replica_{replica}" / mode
            manifest = json.loads((leaf / "manifest.json").read_text(encoding="utf-8"))
            validation = json.loads((leaf / "validation.json").read_text(encoding="utf-8"))
            if (
                manifest.get("status") != "completed"
                or validation.get("status") != "valid"
                or manifest.get("mode") != mode
                or int(manifest.get("replica", -1)) != replica
            ):
                raise ValueError(f"invalid leaf state: {leaf}")
            rows = read_jsonl(leaf / "timings.jsonl")
            all_rows.extend(rows)
            validation["replica"] = replica
            validations.append(validation)
    return all_rows, validations


def validate_cross_mode(rows: list[dict[str, Any]], replicas: int) -> None:
    grouped: dict[tuple[int, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["replica"]), str(row["kind"]), int(row["ordinal"]))].append(row)
    for sample, current in grouped.items():
        if len(current) != len(MODES) or {row["mode"] for row in current} != set(MODES):
            raise ValueError(f"sample does not contain all modes: {sample}")
        for field in (
            "prompt_sha256",
            "prefix_sha256",
            "skill_sha256",
            "prompt_tokens",
            "segment_start",
            "segment_end",
        ):
            if len({row[field] for row in current}) != 1:
                raise ValueError(f"cross-mode mismatch for {sample}: field={field}")
    observed_replicas = {sample[0] for sample in grouped}
    if observed_replicas != set(range(replicas)):
        raise ValueError("run is missing one or more replicas")


def aggregate(rows: list[dict[str, Any]], replicas: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    measured = [row for row in rows if row["kind"] == "measure"]
    summary_rows: list[dict[str, Any]] = []
    mode_values: dict[str, list[float]] = {}
    replica_medians: dict[str, dict[int, float]] = {mode: {} for mode in MODES}
    for mode in MODES:
        values = [float(row["elapsed_ms"]) for row in measured if row["mode"] == mode]
        mode_values[mode] = values
        for replica in range(replicas):
            replica_values = [
                float(row["elapsed_ms"])
                for row in measured
                if row["mode"] == mode and int(row["replica"]) == replica
            ]
            if not replica_values:
                raise ValueError(f"mode={mode} replica={replica} has no measurements")
            replica_medians[mode][replica] = statistics.median(replica_values)
        summary_rows.append(
            {
                "mode": mode,
                "n": len(values),
                "median_ms": statistics.median(values),
                "p95_ms": percentile(values, 0.95),
                "mean_ms": statistics.fmean(values),
                "min_ms": min(values),
                "max_ms": max(values),
            }
        )

    medians = {row["mode"]: float(row["median_ms"]) for row in summary_rows}
    derived = {
        "direct_speedup_vs_full": medians["full"] / medians["direct"],
        "correction_speedup_vs_full": medians["full"] / medians["correction"],
        "correction_overhead_vs_direct_percent":
            (medians["correction"] / medians["direct"] - 1.0) * 100.0,
        "direct_faster_replicas": sum(
            replica_medians["direct"][replica] < replica_medians["full"][replica]
            for replica in range(replicas)
        ),
        "correction_faster_replicas": sum(
            replica_medians["correction"][replica] < replica_medians["full"][replica]
            for replica in range(replicas)
        ),
        "replica_medians_ms": replica_medians,
    }
    for row in summary_rows:
        row["speedup_vs_full"] = medians["full"] / float(row["median_ms"])
    return summary_rows, derived


def plot(summary_rows: list[dict[str, Any]], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    labels = ["Recompute", "Direct reuse", "Correction"]
    colors = ["#6B7280", "#2A9D8F", "#E9C46A"]
    medians = [float(row["median_ms"]) for row in summary_rows]
    p95 = [float(row["p95_ms"]) for row in summary_rows]
    upper = [high - median for median, high in zip(medians, p95)]
    fig, axis = plt.subplots(figsize=(7.0, 2.8))
    bars = axis.bar(
        labels,
        medians,
        width=0.56,
        color=colors,
        edgecolor="black",
        linewidth=1.4,
        yerr=[[0.0] * len(medians), upper],
        capsize=5,
    )
    axis.set_ylabel("One-token latency (ms)", fontsize=13)
    axis.tick_params(axis="both", labelsize=12)
    axis.grid(axis="y", color="#D1D5DB", linewidth=0.8, alpha=0.75)
    axis.set_axisbelow(True)
    for bar, value in zip(bars, medians):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=11,
        )
    fig.tight_layout(pad=0.4)
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / "latency_comparison.pdf", bbox_inches="tight")
    fig.savefig(figure_dir / "latency_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_summary(
    output_dir: Path,
    run_dir: Path,
    summary_rows: list[dict[str, Any]],
    derived: dict[str, Any],
    validations: list[dict[str, Any]],
) -> None:
    by_mode = {row["mode"]: row for row in summary_rows}
    external: dict[str, tuple[int, int]] = {}
    for mode in ("direct", "correction"):
        selected = [row for row in validations if row["mode"] == mode]
        external[mode] = (
            min(int(row["external_tokens_min"]) for row in selected),
            max(int(row["external_tokens_max"]) for row in selected),
        )
    text = f"""# 8K Agent Skill 三模式延迟

## 结论

本实验使用真实离线 `paper-write` Skill（8021 tokens），比较完整重计算、直接复用和 K-only 前缀纠错。正式统计包含每种模式 3 个独立 vLLM replicas、每个 replica 10 个动态 prompt；cold 与 warmup 请求不进入 headline 统计。

- Recompute：median {by_mode['full']['median_ms']:.3f} ms，P95 {by_mode['full']['p95_ms']:.3f} ms。
- Direct reuse：median {by_mode['direct']['median_ms']:.3f} ms，P95 {by_mode['direct']['p95_ms']:.3f} ms，相对 Recompute 加速 {derived['direct_speedup_vs_full']:.3f}×。
- Correction：median {by_mode['correction']['median_ms']:.3f} ms，P95 {by_mode['correction']['p95_ms']:.3f} ms，相对 Recompute 加速 {derived['correction_speedup_vs_full']:.3f}×；相对 Direct 增加 {derived['correction_overhead_vs_direct_percent']:.2f}% 延迟。
- Direct 在 {derived['direct_faster_replicas']}/3 个独立 replicas 中快于 Recompute；Correction 在 {derived['correction_faster_replicas']}/3 个 replicas 中快于 Recompute。

## 实验设置

每个请求由 1024-token 动态前缀、8021-token 真实缓存 Skill 对象和 32-token 动态后缀组成，非流式生成 1 token。相同 sample 在三个模式中的 token IDs 与 prompt SHA256 完全一致；不同 sample 使用不同前缀，避免 leaf 内 automatic prefix cache 命中。每个 `(replica, mode, task)` 使用独立 vLLM server。

Direct 的每个请求均通过日志验证 external KV apply，实际复用 token 范围为 {external['direct'][0]}–{external['direct'][1]}；Correction 的范围为 {external['correction'][0]}–{external['correction'][1]}。纠错配置为 Skill 前 256 tokens 在线计算、相对区间 `[132,256)` 校准逐层逐 KV-head K 残差、`α=0.6`，V 直接复用。

## 指标边界

图表中的 latency 是从 HTTP 请求发出到包含一个生成 token 的非流式响应返回的 wall-clock 时间，包含调度、Prefill、缓存读取/纠错以及一次 decode，不等同于纯 GPU Prefill 时间或流式 TTFT。该结果只覆盖一个约 8K-token Skill 长度点。

## 数据来源

- 原始运行：`{run_dir.resolve()}`
- 请求级数据：`tables/per_request_latency.csv`
- 聚合表：`tables/latency_summary.csv`
- 核心图：`figures/latency_comparison.pdf`
"""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--replicas", type=int, default=3)
    args = parser.parse_args()
    rows, validations = load_run(args.run_dir, args.replicas)
    validate_cross_mode(rows, args.replicas)
    summary_rows, derived = aggregate(rows, args.replicas)

    measured = [row for row in rows if row["kind"] == "measure"]
    request_fields = [
        "mode", "replica", "kind", "ordinal", "sample_id", "elapsed_ms",
        "prompt_tokens", "skill_tokens", "segment_start", "segment_end",
        "prompt_sha256", "response_id",
    ]
    write_csv(
        args.output_dir / "tables" / "per_request_latency.csv",
        request_fields,
        ({field: row.get(field) for field in request_fields} for row in measured),
    )
    summary_fields = [
        "mode", "n", "median_ms", "p95_ms", "mean_ms", "min_ms", "max_ms",
        "speedup_vs_full",
    ]
    write_csv(
        args.output_dir / "tables" / "latency_summary.csv",
        summary_fields,
        ({field: row.get(field) for field in summary_fields} for row in summary_rows),
    )
    atomic_write_json(
        args.output_dir / "data" / "run_metadata.json",
        {
            "schema_version": 1,
            "run_dir": str(args.run_dir.resolve()),
            "replicas": args.replicas,
            "summary": summary_rows,
            "derived": derived,
            "leaf_validation": validations,
        },
    )
    plot(summary_rows, args.output_dir)
    write_summary(
        args.output_dir, args.run_dir, summary_rows, derived, validations
    )
    source_rows = [
        {
            "artifact": artifact,
            "source": str(args.run_dir.resolve()),
            "description": description,
        }
        for artifact, description in (
            ("summary.md", "Three-mode latency result and interpretation"),
            ("figures/latency_comparison.pdf", "Median with one-sided P95 whisker"),
            ("figures/latency_comparison.png", "Raster copy of the core figure"),
            ("tables/latency_summary.csv", "Mode-level latency aggregates"),
            ("tables/per_request_latency.csv", "Measured request-level latency"),
            ("data/run_metadata.json", "Derived metrics and leaf validation"),
        )
    ]
    write_csv(
        args.output_dir / "source_manifest.csv",
        ["artifact", "source", "description"],
        source_rows,
    )
    print(f"[analyzed] run={args.run_dir} output={args.output_dir}")


if __name__ == "__main__":
    main()
