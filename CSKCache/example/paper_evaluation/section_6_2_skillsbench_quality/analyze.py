"""Aggregate deterministic SkillsBench quality outcomes."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from common.plotting import COLORS, configure_matplotlib, save_figure
from common.schema import read_csv, write_csv
from common.statistics import group_rows


GROUP_KEYS = (
    "section",
    "platform_id",
    "model_id",
    "system",
    "prefill_mode",
    "task_id",
    "skillsbench_commit",
    "agent",
    "agent_model",
    "skill_mode",
    "input_fingerprint",
)
SUMMARY_COLUMNS = (
    "schema_version",
    *GROUP_KEYS,
    "sample_count",
    "healthy_count",
    "task_success_count",
    "task_success_rate",
    "mean_reward",
    "mean_tool_calls",
    "mean_skill_invocations",
    "median_total_tokens",
    "mean_vllm_requests",
)


def _truth(value: str) -> bool:
    return value.strip().lower() == "true"


def analyze(run_dir: Path) -> None:
    rows = read_csv(run_dir / "samples.csv")
    summaries = []
    for key, values in group_rows(rows, GROUP_KEYS).items():
        healthy = [row for row in values if _truth(row["pipeline_healthy"])]
        successes = sum(_truth(row["task_success"]) for row in healthy)
        record = dict(zip(GROUP_KEYS, key, strict=True))
        summaries.append(
            {
                "schema_version": 1,
                **record,
                "sample_count": len(values),
                "healthy_count": len(healthy),
                "task_success_count": successes,
                "task_success_rate": successes / len(healthy) if healthy else 0.0,
                "mean_reward": (
                    statistics.fmean(float(row["reward"]) for row in healthy)
                    if healthy
                    else 0.0
                ),
                "mean_tool_calls": (
                    statistics.fmean(int(row["n_tool_calls"]) for row in healthy)
                    if healthy
                    else 0.0
                ),
                "mean_skill_invocations": (
                    statistics.fmean(
                        int(row["n_skill_invocations"]) for row in healthy
                    )
                    if healthy
                    else 0.0
                ),
                "median_total_tokens": (
                    statistics.median(int(row["total_tokens"]) for row in healthy)
                    if healthy
                    else 0.0
                ),
                "mean_vllm_requests": (
                    statistics.fmean(
                        int(row["vllm_request_count"]) for row in healthy
                    )
                    if healthy
                    else 0.0
                ),
            }
        )
    write_csv(run_dir / "summary.csv", summaries, SUMMARY_COLUMNS)
    (run_dir / "analysis.json").write_text(
        json.dumps(
            {"sample_count": len(rows), "groups": summaries},
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    if not summaries:
        return
    configure_matplotlib()
    import matplotlib.pyplot as plt

    labels = [f"{row['system']}\n{row['task_id']}" for row in summaries]
    success = [float(row["task_success_rate"]) for row in summaries]
    figure, axis = plt.subplots(figsize=(max(4.8, 1.7 * len(labels)), 2.8))
    bars = axis.bar(
        range(len(labels)),
        success,
        color=[COLORS[index % len(COLORS)] for index in range(len(labels))],
        edgecolor="black",
        linewidth=0.8,
    )
    axis.set_xticks(range(len(labels)), labels)
    axis.set_ylabel("SkillsBench task success")
    axis.set_ylim(0, 1.08)
    axis.grid(axis="y", alpha=0.25)
    for bar, row in zip(bars, summaries, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.025,
            f"{int(row['task_success_count'])}/{int(row['healthy_count'])}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    figure.subplots_adjust(bottom=0.22)
    save_figure(figure, run_dir / "task_success.png")
    plt.close(figure)


if __name__ == "__main__":
    raise SystemExit("analyze.py is invoked by run.py with the active run directory")
