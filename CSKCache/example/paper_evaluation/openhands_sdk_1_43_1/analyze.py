"""Aggregate paired current-vs-latest OpenHands quality outcomes."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

from common.plotting import COLORS, configure_matplotlib, save_figure
from common.schema import read_csv, write_csv
from common.statistics import group_rows


GROUP_KEYS = (
    "section",
    "platform_id",
    "model_id",
    "harness_id",
    "harness_label",
    "agent",
    "sdk_version",
    "tools_version",
    "task_id",
    "skillsbench_commit",
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
    "required_skill_used_count",
    "required_skill_use_rate",
    "mean_reward",
    "mean_tool_calls",
    "mean_skill_invocations",
    "median_total_tokens",
    "mean_vllm_requests",
)
PAIR_COLUMNS = (
    "schema_version",
    "platform_id",
    "model_id",
    "task_id",
    "repetition",
    "current_reward",
    "latest_reward",
    "current_success",
    "latest_success",
    "current_required_skill_used",
    "latest_required_skill_used",
    "success_delta_latest_minus_current",
)


def _truth(value: str) -> bool:
    return value.strip().lower() == "true"


def _exact_mcnemar_p(current_only: int, latest_only: int) -> float:
    discordant = current_only + latest_only
    if discordant == 0:
        return 1.0
    lower = min(current_only, latest_only)
    tail = sum(math.comb(discordant, index) for index in range(lower + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def analyze(run_dir: Path) -> None:
    rows = read_csv(run_dir / "samples.csv")
    summaries = []
    for key, values in group_rows(rows, GROUP_KEYS).items():
        healthy = [row for row in values if _truth(row["pipeline_healthy"])]
        successes = sum(_truth(row["task_success"]) for row in healthy)
        skill_used = sum(_truth(row["required_skill_used"]) for row in healthy)
        record = dict(zip(GROUP_KEYS, key, strict=True))
        summaries.append(
            {
                "schema_version": 1,
                **record,
                "sample_count": len(values),
                "healthy_count": len(healthy),
                "task_success_count": successes,
                "task_success_rate": successes / len(healthy) if healthy else 0.0,
                "required_skill_used_count": skill_used,
                "required_skill_use_rate": skill_used / len(healthy) if healthy else 0.0,
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

    pairs: dict[tuple[str, str, str, str], dict[str, dict[str, str]]] = {}
    for row in rows:
        if not _truth(row["pipeline_healthy"]):
            continue
        key = (
            row["platform_id"],
            row["model_id"],
            row["task_id"],
            row["repetition"],
        )
        pairs.setdefault(key, {})[row["harness_id"]] = row
    paired_rows = []
    for key, arms in sorted(pairs.items()):
        if set(arms) != {"current", "sdk_1_43_1"}:
            continue
        current = arms["current"]
        latest = arms["sdk_1_43_1"]
        current_success = _truth(current["task_success"])
        latest_success = _truth(latest["task_success"])
        paired_rows.append(
            {
                "schema_version": 1,
                "platform_id": key[0],
                "model_id": key[1],
                "task_id": key[2],
                "repetition": int(key[3]),
                "current_reward": float(current["reward"]),
                "latest_reward": float(latest["reward"]),
                "current_success": current_success,
                "latest_success": latest_success,
                "current_required_skill_used": _truth(
                    current["required_skill_used"]
                ),
                "latest_required_skill_used": _truth(
                    latest["required_skill_used"]
                ),
                "success_delta_latest_minus_current": int(latest_success)
                - int(current_success),
            }
        )
    write_csv(run_dir / "paired.csv", paired_rows, PAIR_COLUMNS)
    current_only = sum(
        row["current_success"] and not row["latest_success"] for row in paired_rows
    )
    latest_only = sum(
        row["latest_success"] and not row["current_success"] for row in paired_rows
    )
    analysis = {
        "sample_count": len(rows),
        "groups": summaries,
        "paired": {
            "pair_count": len(paired_rows),
            "both_success": sum(
                row["current_success"] and row["latest_success"]
                for row in paired_rows
            ),
            "both_failure": sum(
                not row["current_success"] and not row["latest_success"]
                for row in paired_rows
            ),
            "current_only_success": current_only,
            "latest_only_success": latest_only,
            "exact_mcnemar_p": _exact_mcnemar_p(current_only, latest_only),
            "small_sample_warning": (
                "Three repetitions are a smoke-scale comparison; report raw paired "
                "outcomes and do not claim statistical equivalence or superiority."
            ),
        },
    }
    (run_dir / "analysis.json").write_text(
        json.dumps(analysis, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not summaries:
        return
    configure_matplotlib()
    import matplotlib.pyplot as plt

    labels = [
        f"{row['harness_label']}\n{row['task_id']}" for row in summaries
    ]
    success = [float(row["task_success_rate"]) for row in summaries]
    skill_use = [float(row["required_skill_use_rate"]) for row in summaries]
    x_values = list(range(len(labels)))
    figure, axis = plt.subplots(figsize=(max(6.0, 2.3 * len(labels)), 3.1))
    width = 0.36
    success_bars = axis.bar(
        [value - width / 2 for value in x_values],
        success,
        width=width,
        label="Task success",
        color=COLORS[0],
        edgecolor="black",
        linewidth=0.8,
    )
    axis.bar(
        [value + width / 2 for value in x_values],
        skill_use,
        width=width,
        label="Required Skill used",
        color=COLORS[1],
        edgecolor="black",
        linewidth=0.8,
    )
    axis.set_xticks(x_values, labels)
    axis.set_ylabel("Rate")
    axis.set_ylim(0, 1.12)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    for bar, row in zip(success_bars, summaries, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.025,
            f"{int(row['task_success_count'])}/{int(row['healthy_count'])}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    figure.subplots_adjust(bottom=0.25)
    save_figure(figure, run_dir / "quality_comparison.png")
    plt.close(figure)


if __name__ == "__main__":
    raise SystemExit("analyze.py is invoked by run.py with the active run directory")
