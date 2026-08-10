#!/usr/bin/env python3
"""Publish lightweight tables, figures, and a summary from the five tests."""
from __future__ import annotations

import csv
import io
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/segmentia-mpl")
import matplotlib.pyplot as plt
import numpy as np

from common import atomic_write_text, load_config, percentile, raw_run_dir, result_dir


TEST_NAMES = (
    "01_hardware_topology",
    "02_ssd_to_cpu",
    "03_cpu_memory",
    "04_pcie_h2d",
    "05_skill_kv_path",
)


def load_results(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    root = raw_run_dir(config)
    missing = [name for name in TEST_NAMES if not (root / f"{name}.json").is_file()]
    if missing:
        raise FileNotFoundError(f"run the missing tests before analysis: {missing}")
    return {
        name: json.loads((root / f"{name}.json").read_text(encoding="utf-8"))
        for name in TEST_NAMES
    }


def median_rows(
    rows: list[dict[str, Any]],
    key_fields: tuple[str, ...],
    value_field: str,
) -> dict[tuple[Any, ...], float]:
    grouped: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in key_fields)].append(float(row[value_field]))
    return {key: percentile(values, 0.50) for key, values in grouped.items()}


def build_summary_rows(results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ssd = median_rows(
        results["02_ssd_to_cpu"]["measurements"],
        ("state", "threads"),
        "bandwidth_gib_s",
    )
    for (state, threads), value in sorted(ssd.items()):
        rows.append(
            {
                "component": "SSD/page cache to CPU staging",
                "configuration": f"{state}, {threads} thread(s)",
                "metric": "bandwidth_gib_s",
                "value": value,
            }
        )
    for row in results["03_cpu_memory"]["measurements"]:
        if row.get("status") == "ok":
            rows.append(
                {
                    "component": "CPU memory",
                    "configuration": row["path"],
                    "metric": "bandwidth_gib_s",
                    "value": float(row["bandwidth_gib_s"]),
                }
            )
    for row in results["04_pcie_h2d"]["measurements"]:
        if int(row["copies"]) == 1:
            rows.append(
                {
                    "component": "PCIe H2D",
                    "configuration": row["mode"],
                    "metric": "wall_bandwidth_gib_s",
                    "value": float(row["wall_bandwidth_gib_s"]),
                }
            )
    path = median_rows(
        results["05_skill_kv_path"]["measurements"],
        ("state",),
        "path_bandwidth_gib_s",
    )
    for (state,), value in sorted(path.items()):
        rows.append(
            {
                "component": "Serial Skill KV path",
                "configuration": state,
                "metric": "path_bandwidth_gib_s",
                "value": value,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer, fieldnames=("component", "configuration", "metric", "value")
    )
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(path, buffer.getvalue())


def plot_component_bandwidth(rows: list[dict[str, Any]], figure_dir: Path) -> None:
    selected = [row for row in rows if row["component"] != "Serial Skill KV path"]
    labels = [
        f"{row['component']}\n{row['configuration']}".replace(" to ", "→")
        for row in selected
    ]
    values = [float(row["value"]) for row in selected]
    colors = ["#D6A20B" if "SSD" in row["component"] else "#2E7D6B" if "CPU" in row["component"] else "#3B6EA8" for row in selected]
    fig, axis = plt.subplots(figsize=(12.0, 4.2))
    positions = np.arange(len(values))
    axis.bar(positions, values, color=colors, edgecolor="black", linewidth=1.2)
    axis.set_ylabel("Effective bandwidth (GiB/s)")
    axis.set_xticks(positions, labels, rotation=25, ha="right")
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)
    fig.tight_layout()
    for suffix, dpi in (("pdf", None), ("png", 300)):
        fig.savefig(figure_dir / f"component_bandwidth.{suffix}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_path_latency(results: dict[str, dict[str, Any]], figure_dir: Path) -> None:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in results["05_skill_kv_path"]["measurements"]:
        for field in ("disk_read_ms", "h2d_wall_ms", "other_ms", "total_ms"):
            grouped[str(row["state"])][field].append(float(row[field]))
    states = [state for state in ("cold", "warm") if state in grouped]
    disk = [percentile(grouped[state]["disk_read_ms"], 0.50) for state in states]
    h2d = [percentile(grouped[state]["h2d_wall_ms"], 0.50) for state in states]
    other = [percentile(grouped[state]["other_ms"], 0.50) for state in states]
    positions = np.arange(len(states))
    fig, axis = plt.subplots(figsize=(6.4, 3.8))
    axis.bar(positions, disk, label="File read", color="#D6A20B", edgecolor="black")
    axis.bar(positions, h2d, bottom=disk, label="H2D", color="#3B6EA8", edgecolor="black")
    axis.bar(
        positions,
        other,
        bottom=np.asarray(disk) + np.asarray(h2d),
        label="Control/other",
        color="#8AAE92",
        edgecolor="black",
    )
    axis.set_xticks(positions, [state.capitalize() for state in states])
    axis.set_ylabel("Median latency (ms)")
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)
    axis.legend(frameon=False)
    fig.tight_layout()
    for suffix, dpi in (("pdf", None), ("png", 300)):
        fig.savefig(figure_dir / f"skill_kv_path_latency.{suffix}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def summary_markdown(
    config: dict[str, Any],
    results: dict[str, dict[str, Any]],
    rows: list[dict[str, Any]],
) -> str:
    cache = results["02_ssd_to_cpu"]["skill_cache"]
    lines = [
        "# Hardware I/O Characterization",
        "",
        "## 实验对象",
        "",
        f"- Run: `{config['run']['id']}`",
        f"- Cache: `{cache['cache_id']}`",
        f"- Tokens: {cache['token_count']}",
        f"- Layers: {cache['layer_count']}",
        f"- Total bytes: {cache['total_bytes']}",
        "- Cold 定义：只对目标层文件使用 `POSIX_FADV_DONTNEED`，未清空全局 page cache。",
        "- 完整路径：逐层串行 file read → pinned CPU buffer → synchronized async H2D；不包含 Transformer 计算。",
        "",
        "## 汇总带宽",
        "",
        "| Component | Configuration | Metric | Value |",
        "|---|---|---|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['component']} | {row['configuration']} | {row['metric']} | {float(row['value']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## 图表",
            "",
            "- `figures/component_bandwidth.pdf`：SSD、DRAM 和 PCIe 的有效带宽。",
            "- `figures/skill_kv_path_latency.pdf`：完整 40 层 Skill KV 串行路径的 cold/warm 延迟分解。",
            "",
            "## 解释边界",
            "",
            "这些结果提供独立数据移动链路和串行完整路径的硬件基线。它们不能证明 vLLM 中预取与 Transformer 计算已经发生重叠；真实 overlap 仍需端到端事件时间线验证。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    config = load_config()
    results = load_results(config)
    output = result_dir(config)
    figure_dir = output / "figures"
    table_dir = output / "tables"
    data_dir = output / "data"
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    rows = build_summary_rows(results)
    write_csv(table_dir / "hardware_io_summary.csv", rows)
    plot_component_bandwidth(rows, figure_dir)
    plot_path_latency(results, figure_dir)
    atomic_write_text(output / "summary.md", summary_markdown(config, results, rows))
    source_root = raw_run_dir(config)
    manifest_lines = [
        "artifact,source",
        f"summary.md,{source_root}",
        f"figures/component_bandwidth.pdf,{source_root}/02_ssd_to_cpu.json;{source_root}/03_cpu_memory.json;{source_root}/04_pcie_h2d.json",
        f"figures/skill_kv_path_latency.pdf,{source_root}/05_skill_kv_path.json",
        f"tables/hardware_io_summary.csv,{source_root}",
    ]
    atomic_write_text(output / "source_manifest.csv", "\n".join(manifest_lines) + "\n")
    atomic_write_text(
        data_dir / "run_pointer.txt", f"{source_root}\n"
    )
    print(f"[completed] {output}")


if __name__ == "__main__":
    main()
