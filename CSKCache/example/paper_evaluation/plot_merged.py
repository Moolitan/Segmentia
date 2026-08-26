"""Regenerate cross-platform paper figures from merged sample CSV rows."""

from __future__ import annotations

from collections import defaultdict
import statistics

from config import MERGED_OUTPUT_DIR
from common.plotting import COLORS, configure_matplotlib, save_figure
from common.schema import read_csv


def _completed(rows, section):
    return [
        row for row in rows
        if row["section"] == section and row["status"] == "completed"
        and row["warmup"] != "true"
    ]


def _quality(rows) -> None:
    selected = _completed(rows, "section_6_2_correction_quality")
    if not selected:
        return
    configure_matplotlib()
    import matplotlib.pyplot as plt
    import numpy as np

    platforms = list(dict.fromkeys(row["platform_id"] for row in selected))
    systems = list(dict.fromkeys(row["system"] for row in selected))
    x = np.arange(len(systems), dtype=float)
    width = 0.8 / len(platforms)
    figure, axis = plt.subplots(figsize=(7.2, 2.7))
    for index, platform in enumerate(platforms):
        values = []
        for system in systems:
            matches = [
                row for row in selected
                if row["platform_id"] == platform and row["system"] == system
            ]
            passed = sum(int(row["rule_passed"]) for row in matches)
            total = sum(int(row["rule_total"]) for row in matches)
            values.append(passed / total if total else 0.0)
        axis.bar(
            x + (index - (len(platforms) - 1) / 2) * width,
            values,
            width,
            label=platform,
            color=COLORS[index % len(COLORS)],
            edgecolor="black",
            linewidth=0.8,
        )
    axis.set_xticks(x, systems, rotation=20, ha="right")
    axis.set_ylabel("Rule Adherence")
    axis.set_ylim(0, 1.05)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(
        loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=len(platforms),
        frameon=False, columnspacing=1.8,
    )
    figure.subplots_adjust(top=0.80, bottom=0.30)
    save_figure(figure, MERGED_OUTPUT_DIR / "quality_cross_platform.png")
    plt.close(figure)


def _latency(rows) -> None:
    selected = _completed(rows, "section_6_3_latency_scaling")
    selected = [row for row in selected if row["system"] == "CSKCache"]
    if not selected:
        return
    configure_matplotlib()
    import matplotlib.pyplot as plt

    grouped = defaultdict(list)
    for row in selected:
        grouped[(row["platform_id"], row["skill_name"])].append(
            float(row["ttft_ms"])
        )
    skills = list(dict.fromkeys(row["skill_name"] for row in selected))
    platforms = list(dict.fromkeys(row["platform_id"] for row in selected))
    figure, axis = plt.subplots(figsize=(7.2, 2.55))
    for index, platform in enumerate(platforms):
        available = [skill for skill in skills if (platform, skill) in grouped]
        axis.plot(
            available,
            [statistics.median(grouped[(platform, skill)]) for skill in available],
            marker="o",
            label=platform,
            color=COLORS[index % len(COLORS)],
        )
    axis.set_ylabel("Latency (ms)")
    axis.set_xlabel("Skill")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(
        loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=len(platforms),
        frameon=False, columnspacing=1.8,
    )
    figure.subplots_adjust(top=0.79, bottom=0.20)
    save_figure(figure, MERGED_OUTPUT_DIR / "ttft_cross_platform.png")
    plt.close(figure)


def _concurrency(rows) -> None:
    selected = _completed(rows, "section_6_6_concurrency")
    selected = [row for row in selected if row["system"] == "CSKCache"]
    if not selected:
        return
    configure_matplotlib()
    import matplotlib.pyplot as plt

    ttft = defaultdict(list)
    throughput = defaultdict(list)
    for row in selected:
        key = (row["platform_id"], int(row["concurrency"]))
        ttft[key].append(float(row["ttft_ms"]))
        throughput[key].append(float(row["throughput_requests_per_s"]))
    platforms = list(dict.fromkeys(row["platform_id"] for row in selected))
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.55))
    for index, platform in enumerate(platforms):
        concurrencies = sorted(
            concurrency for owner, concurrency in ttft if owner == platform
        )
        color = COLORS[index % len(COLORS)]
        axes[0].plot(
            concurrencies,
            [statistics.median(ttft[(platform, value)]) for value in concurrencies],
            marker="o", label=platform, color=color,
        )
        axes[1].plot(
            concurrencies,
            [
                statistics.median(throughput[(platform, value)])
                for value in concurrencies
            ],
            marker="o", label=platform, color=color,
        )
    axes[0].set_ylabel("Latency (ms)")
    axes[1].set_ylabel("Throughput (requests/s)")
    for axis in axes:
        axis.set_xlabel("Concurrent requests")
        axis.grid(alpha=0.25)
    figure.legend(
        loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(platforms),
        frameon=False, columnspacing=1.8,
    )
    figure.subplots_adjust(top=0.79, bottom=0.20, wspace=0.34)
    save_figure(figure, MERGED_OUTPUT_DIR / "concurrency_cross_platform.png")
    plt.close(figure)


def main() -> None:
    path = MERGED_OUTPUT_DIR / "combined_samples.csv"
    if not path.is_file():
        raise FileNotFoundError(f"run merge_results.py first: {path}")
    rows = read_csv(path)
    _quality(rows)
    _latency(rows)
    _concurrency(rows)
    print(f"figures={MERGED_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
