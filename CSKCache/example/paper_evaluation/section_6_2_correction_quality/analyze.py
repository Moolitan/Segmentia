"""Aggregate Rule Adherence and draw the Section 6.2 grouped bars."""

from __future__ import annotations

from pathlib import Path

from common.plotting import COLORS, configure_matplotlib, save_figure
from common.schema import SUMMARY_COLUMNS, read_csv, write_csv
from common.statistics import group_rows


GROUP_KEYS = (
    "section", "platform_id", "model_id", "system", "skill_name",
    "skill_tokens", "task_id", "correction_strategy", "correction_budget_tokens",
    "correction_ratio", "input_fingerprint",
)


def analyze(run_dir: Path) -> None:
    rows = [row for row in read_csv(run_dir / "samples.csv") if row["status"] == "completed"]
    summaries = []
    for key, values in group_rows(rows, GROUP_KEYS).items():
        record = dict(zip(GROUP_KEYS, key, strict=True))
        passed = sum(int(row["rule_passed"]) for row in values)
        total = sum(int(row["rule_total"]) for row in values)
        summaries.append(
            {
                **record,
                "sample_count": len(values),
                "rule_adherence": passed / total if total else 0.0,
                "fallback_count": sum(row["fallback"] == "true" for row in values),
            }
        )
    write_csv(run_dir / "summary.csv", summaries, SUMMARY_COLUMNS)
    if not summaries:
        return
    configure_matplotlib()
    import matplotlib.pyplot as plt

    systems = list(dict.fromkeys(row["system"] for row in summaries))
    values = []
    for system in systems:
        selected = [row for row in summaries if row["system"] == system]
        values.append(sum(float(row["rule_adherence"]) for row in selected) / len(selected))
    figure, axis = plt.subplots(figsize=(7.2, 2.7))
    axis.bar(
        range(len(systems)), values,
        color=[COLORS[i % len(COLORS)] for i in range(len(systems))],
        edgecolor="black", linewidth=0.8,
    )
    axis.set_xticks(range(len(systems)), systems, rotation=20, ha="right")
    axis.set_ylabel("Rule Adherence")
    axis.set_ylim(0, 1.05)
    axis.grid(axis="y", alpha=0.25)
    figure.subplots_adjust(top=0.95, bottom=0.28)
    save_figure(figure, run_dir / "rule_adherence.png")
    plt.close(figure)


if __name__ == "__main__":
    raise SystemExit("analyze.py is invoked by run.py with the active run directory")
