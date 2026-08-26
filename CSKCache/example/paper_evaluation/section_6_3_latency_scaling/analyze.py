"""Aggregate median TTFT and draw one grouped bar chart per platform."""

from __future__ import annotations

import statistics
from pathlib import Path

from common.plotting import COLORS, configure_matplotlib, save_figure
from common.schema import SUMMARY_COLUMNS, read_csv, write_csv
from common.statistics import group_rows, median_or_none


GROUP_KEYS = (
    "section", "platform_id", "model_id", "system", "skill_name",
    "skill_tokens", "correction_strategy", "correction_budget_tokens",
    "correction_ratio", "input_fingerprint",
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
                "median_ttft_ms": median_or_none(row["ttft_ms"] for row in values),
                "median_latency_ms": median_or_none(
                    row["latency_ms"] for row in values
                ),
                "median_reuse_ratio": median_or_none(
                    row["reuse_ratio"] for row in values
                ),
                "fallback_count": sum(row["fallback"] == "true" for row in values),
            }
        )
    write_csv(run_dir / "summary.csv", summaries, SUMMARY_COLUMNS)
    if not summaries:
        return
    configure_matplotlib()
    import matplotlib.pyplot as plt
    import numpy as np

    for platform_id in dict.fromkeys(row["platform_id"] for row in summaries):
        selected = [row for row in summaries if row["platform_id"] == platform_id]
        skills = sorted(
            {row["skill_name"]: int(row["skill_tokens"]) for row in selected}.items(),
            key=lambda item: item[1],
        )
        systems = list(dict.fromkeys(row["system"] for row in selected))
        x = np.arange(len(skills), dtype=float)
        width = 0.8 / len(systems)
        figure, axis = plt.subplots(figsize=(7.2, 2.75))
        for index, system in enumerate(systems):
            heights = []
            for skill_name, _tokens in skills:
                matches = [
                    row for row in selected
                    if row["system"] == system and row["skill_name"] == skill_name
                ]
                heights.append(float(matches[0]["median_ttft_ms"]))
            axis.bar(
                x + (index - (len(systems) - 1) / 2) * width,
                heights,
                width,
                label=system,
                color=COLORS[index % len(COLORS)],
                edgecolor="black",
                linewidth=0.8,
            )
        axis.set_xticks(
            x,
            [f"{name} ({tokens} tokens)" for name, tokens in skills],
            rotation=0,
        )
        axis.set_ylabel("Latency (ms)")
        axis.grid(axis="y", alpha=0.25)
        axis.legend(
            loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=len(systems),
            frameon=False, columnspacing=1.8, handletextpad=0.5,
        )
        figure.subplots_adjust(top=0.80, bottom=0.18)
        save_figure(figure, run_dir / f"ttft_{platform_id}.png")
        plt.close(figure)


if __name__ == "__main__":
    raise SystemExit("analyze.py is invoked by run.py with the active run directory")
