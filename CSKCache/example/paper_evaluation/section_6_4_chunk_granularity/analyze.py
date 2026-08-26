"""Summarize logical reuse and TTFT without a heatmap."""

from __future__ import annotations

from pathlib import Path

from common.plotting import COLORS, configure_matplotlib, save_figure
from common.schema import SUMMARY_COLUMNS, read_csv, write_csv
from common.statistics import group_rows, median_or_none


GROUP_KEYS = (
    "section", "platform_id", "model_id", "system", "skill_name",
    "skill_tokens", "chunk_tokens", "mutation", "mutation_position",
    "correction_strategy", "correction_budget_tokens", "correction_ratio",
    "input_fingerprint",
)


def analyze(run_dir: Path) -> None:
    rows = [row for row in read_csv(run_dir / "samples.csv") if row["status"] == "completed"]
    summaries = []
    for key, values in group_rows(rows, GROUP_KEYS).items():
        record = dict(zip(GROUP_KEYS, key, strict=True))
        summaries.append(
            {
                **record,
                "sample_count": len(values),
                "median_ttft_ms": median_or_none(row["ttft_ms"] for row in values),
                "median_latency_ms": median_or_none(row["latency_ms"] for row in values),
                "median_reuse_ratio": median_or_none(row["reuse_ratio"] for row in values),
                "fallback_count": sum(row["fallback"] == "true" for row in values),
            }
        )
    write_csv(run_dir / "summary.csv", summaries, SUMMARY_COLUMNS)
    if not summaries:
        return
    longest = max(summaries, key=lambda row: int(row["skill_tokens"] or 0))["skill_name"]
    selected = [row for row in summaries if row["skill_name"] == longest]
    chunk_systems = [
        name for name in dict.fromkeys(row["system"] for row in selected)
        if name.startswith("Chunk-")
    ]
    mutation_order = [
        ("exact", "0.0", "Exact"),
        ("replace", "0.25", "Edit 25%"),
        ("replace", "0.5", "Edit 50%"),
        ("replace", "0.75", "Edit 75%"),
        ("append", "1.0", "Append"),
    ]
    configure_matplotlib()
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.65))
    for index, system in enumerate(chunk_systems):
        reuse_values = []
        ttft_values = []
        for mutation, position, _label in mutation_order:
            matches = [
                row for row in selected
                if row["system"] == system and row["mutation"] == mutation
                and float(row["mutation_position"] or 0) == float(position)
            ]
            reuse_values.append(float(matches[0]["median_reuse_ratio"]))
            timed = [row for row in matches if row["median_ttft_ms"] != ""]
            ttft_values.append(float(timed[0]["median_ttft_ms"]) if timed else float("nan"))
        labels = [label for _mutation, _position, label in mutation_order]
        axes[0].plot(
            labels, reuse_values, marker="o", label=system,
            color=COLORS[index % len(COLORS)], markeredgecolor="black",
        )
        axes[1].plot(
            labels, ttft_values, marker="o", label=system,
            color=COLORS[index % len(COLORS)], markeredgecolor="black",
        )
    axes[0].set_ylabel("Reusable Token Fraction")
    axes[0].set_ylim(-0.03, 1.05)
    axes[1].set_ylabel("Latency (ms)")
    for axis in axes:
        axis.tick_params(axis="x", rotation=20)
        axis.grid(axis="y", alpha=0.25)
    figure.legend(
        loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=max(1, len(chunk_systems)),
        frameon=False, columnspacing=1.6,
    )
    figure.suptitle(f"{longest}: logical Chunk granularity", y=1.12)
    figure.subplots_adjust(top=0.78, bottom=0.25, wspace=0.32)
    save_figure(figure, run_dir / "chunk_granularity.png")
    plt.close(figure)


if __name__ == "__main__":
    raise SystemExit("analyze.py is invoked by run.py with the active run directory")
