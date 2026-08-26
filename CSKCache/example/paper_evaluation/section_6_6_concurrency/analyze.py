"""Aggregate concurrent batches and draw TTFT/throughput scaling."""

from __future__ import annotations

from pathlib import Path

from common.plotting import COLORS, configure_matplotlib, save_figure
from common.schema import SUMMARY_COLUMNS, read_csv, write_csv
from common.statistics import group_rows, median_or_none


GROUP_KEYS = (
    "section", "platform_id", "model_id", "system", "skill_name",
    "skill_tokens", "chunk_tokens", "correction_strategy",
    "correction_budget_tokens", "correction_ratio", "concurrency",
    "input_fingerprint",
)


def analyze(run_dir: Path) -> None:
    rows = [
        row for row in read_csv(run_dir / "samples.csv")
        if row["status"] == "completed" and row["warmup"] == "false"
    ]
    summaries = []
    for key, values in group_rows(rows, GROUP_KEYS).items():
        record = dict(zip(GROUP_KEYS, key, strict=True))
        summaries.append(
            {
                **record,
                "sample_count": len(values),
                "median_ttft_ms": median_or_none(
                    row["ttft_ms"] for row in values
                ),
                "median_latency_ms": median_or_none(
                    row["latency_ms"] for row in values
                ),
                "median_throughput_requests_per_s": median_or_none(
                    row["throughput_requests_per_s"] for row in values
                ),
                "median_reuse_ratio": median_or_none(
                    row["reuse_ratio"] for row in values
                ),
                "fallback_count": sum(
                    row["fallback"] == "true" for row in values
                ),
            }
        )
    write_csv(run_dir / "summary.csv", summaries, SUMMARY_COLUMNS)
    if not summaries:
        return
    configure_matplotlib()
    import matplotlib.pyplot as plt

    for platform_id in dict.fromkeys(row["platform_id"] for row in summaries):
        selected = [
            row for row in summaries if row["platform_id"] == platform_id
        ]
        systems = list(dict.fromkeys(row["system"] for row in selected))
        concurrencies = sorted({int(row["concurrency"]) for row in selected})
        figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.55))
        for index, system in enumerate(systems):
            values = [
                next(
                    row for row in selected
                    if row["system"] == system
                    and int(row["concurrency"]) == concurrency
                )
                for concurrency in concurrencies
            ]
            axes[0].plot(
                concurrencies,
                [float(row["median_ttft_ms"]) for row in values],
                marker="o",
                label=system,
                color=COLORS[index % len(COLORS)],
            )
            axes[1].plot(
                concurrencies,
                [
                    float(row["median_throughput_requests_per_s"])
                    for row in values
                ],
                marker="o",
                label=system,
                color=COLORS[index % len(COLORS)],
            )
        axes[0].set_ylabel("Latency (ms)")
        axes[1].set_ylabel("Throughput (requests/s)")
        for axis in axes:
            axis.set_xlabel("Concurrent requests")
            axis.set_xticks(concurrencies)
            axis.grid(alpha=0.25)
        figure.legend(
            loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(systems),
            frameon=False, columnspacing=2.0, handletextpad=0.5,
        )
        figure.subplots_adjust(top=0.79, bottom=0.20, wspace=0.34)
        save_figure(figure, run_dir / f"concurrency_{platform_id}.png")
        plt.close(figure)


if __name__ == "__main__":
    raise SystemExit("analyze.py is invoked by run.py with the active run directory")
