"""Pair Full thinking with CSK thinking and draw quality--latency curves."""

from __future__ import annotations

import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from paper_evaluation.common.plotting import configure_matplotlib, save_figure
from paper_evaluation.common.schema import write_csv

from .metrics import rouge_l_recall
from .schema import (
    PAIRED_COLUMNS,
    SUMMARY_COLUMNS,
    collect_samples,
    write_sample_tables,
)


TASK_STYLES = {
    "3d-scan-calc": ("3D Scan", "#4477AA", "o"),
    "azure-bgp-oscillation-route-leak": ("Azure BGP", "#EE6677", "s"),
    "citation-check": ("Citation Check", "#228833", "^"),
    "data-to-d3": ("Data-to-D3", "#CCBB44", "D"),
    "jax-computing-basics": ("JAX", "#66CCEE", "v"),
    "protein-expression-analysis": ("Protein Expression", "#AA3377", "P"),
}


def _percentile(values: Iterable[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot compute a percentile of an empty sequence")
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _thinking(run_dir: Path, sample: dict[str, Any]) -> str:
    path = run_dir / str(sample["thinking_path"])
    return path.read_text(encoding="utf-8")


def _pair_samples(
    run_dir: Path, samples: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    references = {
        (str(sample["platform_id"]), str(sample["task_id"])): sample
        for sample in samples
        if sample.get("system_family") == "full" and sample.get("status") == "valid"
    }
    paired: list[dict[str, Any]] = []
    for sample in samples:
        if sample.get("system_family") != "cskcache":
            continue
        if sample.get("status") != "valid":
            continue
        key = (str(sample["platform_id"]), str(sample["task_id"]))
        reference = references.get(key)
        if reference is None:
            continue
        reference_text = _thinking(run_dir, reference)
        candidate_text = _thinking(run_dir, sample)
        paired.append(
            {
                **sample,
                "reference_case_id": reference["case_id"],
                "reference_thinking_path": reference["thinking_path"],
                "reference_thinking_words": reference["thinking_words"],
                "rouge_l_recall": rouge_l_recall(
                    reference_text, candidate_text
                ),
            }
        )
    return sorted(
        paired,
        key=lambda row: (
            str(row["platform_id"]),
            str(row["task_id"]),
            float(row["requested_calibration_ratio"]),
        ),
    )


def _summaries(paired: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[float, str], list[dict[str, Any]]] = defaultdict(list)
    for row in paired:
        groups[(float(row["requested_calibration_ratio"]), str(row["system"]))].append(
            row
        )
    summaries = []
    for (ratio, system), rows in sorted(groups.items()):
        quality = [float(row["rouge_l_recall"]) for row in rows]
        latency = [float(row["calibration_compute_ms"]) for row in rows]
        summaries.append(
            {
                "requested_calibration_ratio": ratio,
                "system": system,
                "sample_count": len(rows),
                "task_count": len({str(row["task_id"]) for row in rows}),
                "median_rouge_l_recall": statistics.median(quality),
                "q1_rouge_l_recall": _percentile(quality, 0.25),
                "q3_rouge_l_recall": _percentile(quality, 0.75),
                "median_calibration_compute_ms": statistics.median(latency),
                "q1_calibration_compute_ms": _percentile(latency, 0.25),
                "q3_calibration_compute_ms": _percentile(latency, 0.75),
                "median_calibration_forward_ms": statistics.median(
                    float(row["calibration_forward_ms"]) for row in rows
                ),
                "median_residual_correction_ms": statistics.median(
                    float(row["residual_correction_ms"]) for row in rows
                ),
                "median_actual_calibration_tokens": statistics.median(
                    int(row["actual_calibration_tokens"]) for row in rows
                ),
                "median_reused_tokens": statistics.median(
                    int(row["reused_tokens"]) for row in rows
                ),
            }
        )
    return summaries


def _draw_ratio_metrics(
    run_dir: Path,
    paired: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
) -> None:
    if not summaries:
        return
    configure_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    figure, latency_axis = plt.subplots(figsize=(9.2, 4.35))
    quality_axis = latency_axis.twinx()
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in paired:
        grouped[(str(row["platform_id"]), str(row["task_id"]))].append(row)
    task_handles = []
    for (_, task_id), rows in sorted(grouped.items()):
        rows.sort(key=lambda row: float(row["requested_calibration_ratio"]))
        ratios = [100 * float(row["requested_calibration_ratio"]) for row in rows]
        label, color, marker = TASK_STYLES.get(
            task_id,
            (task_id.replace("-", " ").title(), "#777777", "x"),
        )
        latency_axis.plot(
            ratios,
            [float(row["calibration_compute_ms"]) for row in rows],
            color=color,
            linestyle="-",
            marker=marker,
            markersize=5.0,
            linewidth=1.55,
            alpha=0.94,
            zorder=2,
        )
        quality_axis.plot(
            ratios,
            [float(row["rouge_l_recall"]) for row in rows],
            color=color,
            linestyle="--",
            marker=marker,
            markerfacecolor="white",
            markeredgewidth=1.1,
            markersize=5.0,
            linewidth=1.45,
            alpha=0.94,
            zorder=3,
        )
        task_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                marker=marker,
                linestyle="none",
                markersize=5.5,
                label=label,
            )
        )
    ratios = sorted(
        {100 * float(row["requested_calibration_ratio"]) for row in paired}
    )
    latency_axis.set_xlabel("Calibration ratio (%)")
    latency_axis.set_ylabel("Layer-8 calibration compute (ms)")
    quality_axis.set_ylabel("ROUGE-L Recall")
    latency_axis.set_xticks(ratios)
    latency_axis.grid(axis="y", alpha=0.22, linewidth=0.7)
    quality_axis.grid(False)
    quality_axis.spines["right"].set_visible(True)
    metric_handles = [
        Line2D(
            [0],
            [0],
            color="#333333",
            linestyle="-",
            linewidth=1.6,
            label="Layer-8 latency (left axis)",
        ),
        Line2D(
            [0],
            [0],
            color="#333333",
            linestyle="--",
            linewidth=1.5,
            label="ROUGE-L Recall (right axis)",
        ),
    ]
    figure.legend(
        handles=task_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=6,
        frameon=False,
        handlelength=1.2,
        columnspacing=1.05,
    )
    figure.legend(
        handles=metric_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=2,
        frameon=False,
        handlelength=2.5,
        columnspacing=2.0,
    )
    figure.subplots_adjust(top=0.78, left=0.09, right=0.91, bottom=0.14)
    save_figure(figure, run_dir / "ratio_quality_latency.png")
    plt.close(figure)


def _draw_quality_latency(
    run_dir: Path,
    paired: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
) -> None:
    if not summaries:
        return
    configure_matplotlib()
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(5.4, 3.5))
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in paired:
        grouped[(str(row["platform_id"]), str(row["task_id"]))].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: float(row["requested_calibration_ratio"]))
        axis.plot(
            [float(row["calibration_compute_ms"]) for row in rows],
            [float(row["rouge_l_recall"]) for row in rows],
            color="#A0A0A0",
            alpha=0.4,
            linewidth=0.8,
        )
    x = [float(row["median_calibration_compute_ms"]) for row in summaries]
    y = [float(row["median_rouge_l_recall"]) for row in summaries]
    axis.plot(x, y, color="#4C78A8", marker="o", linewidth=2.0)
    for x_value, y_value, row in zip(x, y, summaries, strict=True):
        axis.annotate(
            f"{100 * float(row['requested_calibration_ratio']):g}%",
            (x_value, y_value),
            xytext=(4, 4),
            textcoords="offset points",
        )
    axis.set_xlabel("Layer-8 calibration compute (ms)")
    axis.set_ylabel("ROUGE-L Recall vs. Full thinking")
    axis.grid(alpha=0.25)
    save_figure(figure, run_dir / "quality_latency_curve.png")
    plt.close(figure)


def _write_analysis(
    run_dir: Path,
    samples: list[dict[str, Any]],
    paired: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
) -> None:
    valid = sum(sample.get("status") == "valid" for sample in samples)
    invalid = len(samples) - valid
    tasks = len({str(row["task_id"]) for row in paired})
    lines = [
        "# SkillsBench correction quality--latency analysis",
        "",
        f"- Completed request-pair cases: {len(samples)}",
        f"- Valid / invalid cases: {valid} / {invalid}",
        f"- Tasks with Full-to-CSK pairs: {tasks}",
        f"- Valid paired ratio samples: {len(paired)}",
        "- Quality: word-level ROUGE-L Recall with Full-prefill thinking as reference.",
        "- Latency: layer 8 calibration_forward_ms + residual_correction_ms.",
        "",
        "This is a thinking-fidelity diagnostic, not task correctness. A high score only",
        "means that the calibrated run follows text similar to the Full-prefill run.",
        "",
        "## Ratio medians",
        "",
        "| Ratio | Tasks | ROUGE-L Recall | Calibration compute (ms) |",
        "|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            "| "
            f"{100 * float(row['requested_calibration_ratio']):g}% | "
            f"{row['task_count']} | "
            f"{float(row['median_rouge_l_recall']):.4f} | "
            f"{float(row['median_calibration_compute_ms']):.4f} |"
        )
    (run_dir / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(run_dir: Path) -> None:
    samples = collect_samples(run_dir)
    write_sample_tables(run_dir, samples)
    paired = _pair_samples(run_dir, samples)
    summaries = _summaries(paired)
    write_csv(run_dir / "paired_samples.csv", paired, PAIRED_COLUMNS)
    write_csv(run_dir / "summary.csv", summaries, SUMMARY_COLUMNS)
    _draw_ratio_metrics(run_dir, paired, summaries)
    _draw_quality_latency(run_dir, paired, summaries)
    _write_analysis(run_dir, samples, paired, summaries)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    arguments = parser.parse_args()
    analyze(arguments.run_dir.resolve())


if __name__ == "__main__":
    main()
