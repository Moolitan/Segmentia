#!/usr/bin/env python3
"""Validate, aggregate, plot, and publish the three-arm TTFT sweep."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
import statistics


METHODS = ("normal_prefill", "direct_reuse", "deviation_topk")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(
    path: Path, rows: list[dict[str, object]], fields: tuple[str, ...]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(
    rows: list[dict[str, str]],
    *,
    token_lengths: tuple[int, ...],
    repetitions: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    measured = [row for row in rows if row["warmup"].lower() == "false"]
    expected = len(token_lengths) * repetitions * len(METHODS)
    if len(measured) != expected:
        raise RuntimeError(
            f"expected {expected} measured rows, found {len(measured)}"
        )
    aggregates: list[dict[str, object]] = []
    for segment_tokens in token_lengths:
        for method in METHODS:
            selected = [
                row
                for row in measured
                if int(row["segment_tokens"]) == segment_tokens
                and row["method"] == method
            ]
            if len(selected) != repetitions:
                raise RuntimeError(
                    f"tokens={segment_tokens} method={method}: expected "
                    f"{repetitions} rows, found {len(selected)}"
                )
            if method != "normal_prefill" and any(
                row["host_ready_before_timer"].lower() != "true"
                or row["reuse_validated"].lower() != "true"
                for row in selected
            ):
                raise RuntimeError(
                    f"tokens={segment_tokens} method={method}: evidence gate failed"
                )
            values = [float(row["ttft_ms"]) for row in selected]
            aggregates.append(
                {
                    "segment_tokens": segment_tokens,
                    "prompt_tokens": int(selected[0]["prompt_tokens"]),
                    "method": method,
                    "sample_count": len(values),
                    "median_ttft_ms": statistics.median(values),
                    "mean_ttft_ms": statistics.mean(values),
                    "stdev_ttft_ms": (
                        statistics.stdev(values) if len(values) > 1 else 0.0
                    ),
                    "min_ttft_ms": min(values),
                    "max_ttft_ms": max(values),
                }
            )

    comparisons: list[dict[str, object]] = []
    for segment_tokens in token_lengths:
        by_method = {
            str(row["method"]): row
            for row in aggregates
            if int(row["segment_tokens"]) == segment_tokens
        }
        normal = float(by_method["normal_prefill"]["median_ttft_ms"])
        direct = float(by_method["direct_reuse"]["median_ttft_ms"])
        selective = float(by_method["deviation_topk"]["median_ttft_ms"])
        comparisons.append(
            {
                "segment_tokens": segment_tokens,
                "prompt_tokens": int(
                    by_method["normal_prefill"]["prompt_tokens"]
                ),
                "normal_prefill_median_ttft_ms": normal,
                "direct_reuse_median_ttft_ms": direct,
                "direct_saved_ms": normal - direct,
                "direct_reduction_pct": (normal - direct) / normal * 100.0,
                "direct_speedup_x": normal / direct,
                "selective_recompute_median_ttft_ms": selective,
                "selective_saved_ms": normal - selective,
                "selective_reduction_pct": (
                    (normal - selective) / normal * 100.0
                ),
                "selective_speedup_x": normal / selective,
                "selective_overhead_vs_direct_ms": selective - direct,
                "selective_overhead_vs_direct_pct": (
                    (selective - direct) / direct * 100.0
                ),
            }
        )
    return aggregates, comparisons


def _plot(comparisons: list[dict[str, object]], output: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    tokens = [int(row["segment_tokens"]) for row in comparisons]
    normal = [
        float(row["normal_prefill_median_ttft_ms"]) for row in comparisons
    ]
    direct = [
        float(row["direct_reuse_median_ttft_ms"]) for row in comparisons
    ]
    selective = [
        float(row["selective_recompute_median_ttft_ms"])
        for row in comparisons
    ]
    x = np.arange(len(tokens), dtype=float)
    width = 0.25
    figure, axis = plt.subplots(figsize=(8.4, 4.0))
    axis.bar(
        x - width,
        normal,
        width,
        label="Normal prefill",
        color="#4C78A8",
        edgecolor="black",
        linewidth=0.7,
    )
    axis.bar(
        x,
        direct,
        width,
        label="Direct reuse",
        color="#F28E2B",
        edgecolor="black",
        linewidth=0.7,
    )
    axis.bar(
        x + width,
        selective,
        width,
        label="CacheBlend 15% selective recompute",
        color="#59A14F",
        edgecolor="black",
        linewidth=0.7,
    )
    tallest = max(normal + direct + selective)
    axis.set_xticks(x, [str(token) for token in tokens])
    axis.set_xlabel("Token size")
    axis.set_ylabel("TTFT (ms)")
    axis.set_ylim(0, tallest * 1.12)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, ncol=3, loc="upper left")
    figure.tight_layout()
    figure.savefig(output.with_suffix(".png"), dpi=240)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def _publish(
    run_dir: Path,
    publish_root: Path,
    comparisons: list[dict[str, object]],
) -> None:
    figures = publish_root / "figures"
    tables = publish_root / "tables"
    data = publish_root / "data"
    for directory in (figures, tables, data):
        directory.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        shutil.copy2(
            run_dir / f"ttft_by_token_size.{suffix}",
            figures / f"ttft_by_token_size.{suffix}",
        )
    shutil.copy2(run_dir / "comparison.csv", tables / "comparison.csv")
    shutil.copy2(run_dir / "aggregate.csv", data / "aggregate.csv")

    lines = [
        "# Prefill and KV Reuse TTFT",
        "",
        "## 实验方法",
        "",
        "- 模型与硬件：Qwen3-14B，单张 NVIDIA RTX A6000。",
        "- Token size：512、1024、2048、4096、8192。",
        "- 三组方法分别使用独立 engine；每组预热 1 次、正式测量 5 次。",
        "- 每个请求前清空 prefix cache，最大输出为 1 token。",
        "- Direct reuse 不运行辅助纠错模型。",
        "- CacheBlend 15% selective recompute 在第 1 层按 key 平方 L2 偏差选出 top 15%，",
        "  后续层只重算同一批 token。",
        "- 复用数据计时前已处于 Pinned CPU；TTFT 从 generate 调用计至首 token。",
        "",
        "## 数据",
        "",
        "| Token size | Normal prefill | Direct reuse | Direct reduction | "
        "CacheBlend 15% selective recompute | Selective reduction |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparisons:
        lines.append(
            f"| {int(row['segment_tokens'])} | "
            f"{float(row['normal_prefill_median_ttft_ms']):.3f} ms | "
            f"{float(row['direct_reuse_median_ttft_ms']):.3f} ms | "
            f"{float(row['direct_reduction_pct']):.2f}% | "
            f"{float(row['selective_recompute_median_ttft_ms']):.3f} ms | "
            f"{float(row['selective_reduction_pct']):.2f}% |"
        )
    lines.extend(
        [
            "",
            "核心图为 `figures/ttft_by_token_size.png`。柱高为 5 次正式样本的",
            "中位 TTFT，具体数值与提升比例见上表。",
            "",
            "## 限制",
            "",
            "该实验只测延迟，不评价生成质量。合成 KV 保留真实 shape、Pinned H2D、",
            "PagedKV 安装及选择性重算路径，但不代表真实 Skill 内容，也不包含 SSD、",
            "Agent 或 Tool 调用时间。",
            "",
            f"原始 run：`{run_dir}`。",
        ]
    )
    (publish_root / "summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    manifest_rows = [
        {
            "artifact": f"figures/ttft_by_token_size.{suffix}",
            "source": str(run_dir / f"ttft_by_token_size.{suffix}"),
        }
        for suffix in ("png", "pdf")
    ]
    manifest_rows.extend(
        [
            {
                "artifact": "tables/comparison.csv",
                "source": str(run_dir / "comparison.csv"),
            },
            {
                "artifact": "data/aggregate.csv",
                "source": str(run_dir / "aggregate.csv"),
            },
            {
                "artifact": "summary.md",
                "source": str(run_dir / "run_config.json"),
            },
        ]
    )
    _write_csv(
        publish_root / "source_manifest.csv",
        manifest_rows,
        ("artifact", "source"),
    )


def analyze(run_dir: Path, *, publish_root: Path | None = None) -> None:
    config = json.loads(
        (run_dir / "run_config.json").read_text(encoding="utf-8")
    )
    if tuple(config["methods"]) != METHODS:
        raise RuntimeError("run method order differs from analyzer contract")
    rows = _read_csv(run_dir / "samples.csv")
    aggregates, comparisons = summarize(
        rows,
        token_lengths=tuple(int(v) for v in config["token_lengths"]),
        repetitions=int(config["repetitions"]),
    )
    _write_csv(
        run_dir / "aggregate.csv",
        aggregates,
        (
            "segment_tokens",
            "prompt_tokens",
            "method",
            "sample_count",
            "median_ttft_ms",
            "mean_ttft_ms",
            "stdev_ttft_ms",
            "min_ttft_ms",
            "max_ttft_ms",
        ),
    )
    _write_csv(
        run_dir / "comparison.csv",
        comparisons,
        (
            "segment_tokens",
            "prompt_tokens",
            "normal_prefill_median_ttft_ms",
            "direct_reuse_median_ttft_ms",
            "direct_saved_ms",
            "direct_reduction_pct",
            "direct_speedup_x",
            "selective_recompute_median_ttft_ms",
            "selective_saved_ms",
            "selective_reduction_pct",
            "selective_speedup_x",
            "selective_overhead_vs_direct_ms",
            "selective_overhead_vs_direct_pct",
        ),
    )
    _plot(comparisons, run_dir / "ttft_by_token_size")
    if publish_root is not None:
        _publish(run_dir, publish_root, comparisons)


if __name__ == "__main__":
    raise SystemExit("analyze.py is invoked by run.py with the active run directory")
