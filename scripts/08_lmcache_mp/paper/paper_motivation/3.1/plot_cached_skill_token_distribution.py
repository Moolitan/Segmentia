#!/usr/bin/env python3
"""Plot the token-length distribution of completed offline Skill KV caches."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_POOL_DIR = Path(
    "/mnt/Large_Language_Model_Lab_1/wsh/skill_save_pool/Qwen3-14B"
)
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "results"
    / "problem_exploration"
    / "skill_token_length_distribution"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-dir", type=Path, default=DEFAULT_POOL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_completed_manifests(pool_dir: Path) -> list[dict[str, Any]]:
    if not pool_dir.is_dir():
        raise FileNotFoundError(f"offline Skill pool does not exist: {pool_dir}")

    records: list[dict[str, Any]] = []
    cache_ids: set[str] = set()
    for manifest_path in sorted(pool_dir.rglob("manifest.json")):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("status") != "completed":
            continue
        completed_marker = manifest_path.with_name("COMPLETED")
        if not completed_marker.is_file():
            raise RuntimeError(
                f"completed manifest has no COMPLETED marker: {manifest_path}"
            )

        cache_id = payload.get("cache_id")
        token_count = payload.get("token_count")
        if not isinstance(cache_id, str) or not cache_id:
            raise RuntimeError(f"invalid cache_id in {manifest_path}")
        if (
            not isinstance(token_count, int)
            or isinstance(token_count, bool)
            or token_count <= 0
        ):
            raise RuntimeError(f"invalid token_count in {manifest_path}")
        if cache_id in cache_ids:
            raise RuntimeError(f"duplicate cache_id in offline pool: {cache_id}")
        cache_ids.add(cache_id)

        raw_count = payload.get("raw_skill_token_count")
        if raw_count is not None and (
            not isinstance(raw_count, int)
            or isinstance(raw_count, bool)
            or raw_count <= 0
        ):
            raise RuntimeError(f"invalid raw_skill_token_count in {manifest_path}")

        records.append(
            {
                "cache_id": cache_id,
                "skill_name": payload.get("skill_name", ""),
                "cached_tokens": token_count,
                "raw_skill_tokens": raw_count,
                "wrapper_tokens": (
                    token_count - raw_count if raw_count is not None else None
                ),
                "skill_path": payload.get("skill_path", ""),
                "manifest_path": str(manifest_path.resolve()),
            }
        )

    if not records:
        raise RuntimeError(f"no completed Skill manifests found under {pool_dir}")
    return sorted(records, key=lambda row: (-row["cached_tokens"], row["cache_id"]))


def summarize(records: list[dict[str, Any]], pool_dir: Path) -> dict[str, Any]:
    values = np.asarray([row["cached_tokens"] for row in records], dtype=np.int64)
    return {
        "data_source": str(pool_dir.resolve()),
        "metric": "manifest.token_count",
        "count": int(values.size),
        "total_tokens": int(values.sum()),
        "minimum": int(values.min()),
        "p25": round(float(np.percentile(values, 25)), 1),
        "median": round(float(np.median(values)), 1),
        "mean": round(float(values.mean()), 1),
        "p75": round(float(np.percentile(values, 75)), 1),
        "p90": round(float(np.percentile(values, 90)), 1),
        "p95": round(float(np.percentile(values, 95)), 1),
        "maximum": int(values.max()),
    }


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = [
        "cache_id",
        "skill_name",
        "cached_tokens",
        "raw_skill_tokens",
        "wrapper_tokens",
        "skill_path",
        "manifest_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def plot_distribution(
    output_dir: Path,
    records: list[dict[str, Any]],
) -> None:
    values = np.asarray([row["cached_tokens"] for row in records], dtype=np.int64)
    bin_count = max(10, min(24, math.ceil(math.sqrt(values.size))))
    if values.min() == values.max():
        bins = np.asarray([values.min() - 0.5, values.max() + 0.5])
    else:
        bins = np.geomspace(values.min(), values.max(), bin_count + 1)

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(7.0, 2.2))
    ax.hist(
        values,
        bins=bins,
        color="#5B8E7D",
        edgecolor="black",
        linewidth=2.0,
    )
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xlabel("Cached Skill length (tokens)")
    ax.set_ylabel("Number of Skills")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(output_dir / "skill_token_length_distribution.png", dpi=300)
    fig.savefig(output_dir / "skill_token_length_distribution.pdf")
    plt.close(fig)


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    content = f"""# 离线 Skill Token 长度分布

## 结论

本统计覆盖 {summary['count']} 个成功离线并带有 `COMPLETED` 标记的 Skill KV 缓存。实际缓存长度的中位数为 {summary['median']:,.0f} token，P90 为 {summary['p90']:,.0f} token，范围为 {summary['minimum']:,}–{summary['maximum']:,} token。

## 统计口径

- 数据源：`{summary['data_source']}/**/manifest.json`。
- 指标：manifest 中的 `token_count`，即实际离线保存的完整 context-segment KV token 数。
- 纳入条件：`status == "completed"`、同目录存在 `COMPLETED`、`token_count` 为正整数。
- 本统计不读取 `SKILL.md`，不重新 tokenize，也不启动 vLLM。

## 汇总

| 指标 | Token 数 |
|---|---:|
| 最小值 | {summary['minimum']:,} |
| P25 | {summary['p25']:,.1f} |
| 中位数 | {summary['median']:,.1f} |
| 均值 | {summary['mean']:,.1f} |
| P75 | {summary['p75']:,.1f} |
| P90 | {summary['p90']:,.1f} |
| P95 | {summary['p95']:,.1f} |
| 最大值 | {summary['maximum']:,} |
| 总计 | {summary['total_tokens']:,} |

## 产物

- `data/skill_token_lengths.csv`：逐 Skill 明细，包含每条记录的 manifest 来源。
- `data/summary.json`：机器可读汇总。
- `figures/skill_token_length_distribution.png` 和 `.pdf`：对数横轴直方分布图。

## 限制

该分布仅描述成功离线的缓存对象，不包含离线失败或未生成 manifest 的 Skill，因此不能解释失败条目的文本长度。
"""
    path.write_text(content, encoding="utf-8")


def write_source_manifest(path: Path, pool_dir: Path) -> None:
    rows = [
        ("data/skill_token_lengths.csv", f"{pool_dir.resolve()}/**/manifest.json"),
        ("data/summary.json", "data/skill_token_lengths.csv"),
        ("figures/skill_token_length_distribution.png", "data/skill_token_lengths.csv"),
        ("figures/skill_token_length_distribution.pdf", "data/skill_token_lengths.csv"),
        ("summary.md", "data/summary.json; data/skill_token_lengths.csv"),
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["artifact", "source"])
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    records = load_completed_manifests(args.pool_dir)
    summary = summarize(records, args.pool_dir)

    data_dir = args.output_dir / "data"
    figures_dir = args.output_dir / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    write_csv(data_dir / "skill_token_lengths.csv", records)
    (data_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    plot_distribution(figures_dir, records)
    write_summary(args.output_dir / "summary.md", summary)
    write_source_manifest(args.output_dir / "source_manifest.csv", args.pool_dir)
    print(
        f"[done] count={summary['count']} min={summary['minimum']} "
        f"median={summary['median']:.1f} p90={summary['p90']:.1f} "
        f"max={summary['maximum']} output={args.output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
