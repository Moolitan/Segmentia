"""Aggregate fixed-length TTFT repetitions and draw grouped-bar panels."""

from __future__ import annotations

import hashlib
import statistics
from pathlib import Path
from typing import Iterable

from paper_evaluation.common.plotting import configure_matplotlib
from paper_evaluation.common.schema import read_csv, write_csv

from . import config as local
from .schema import SUMMARY_COLUMNS


BAR_COLORS = ("#7F7F7F", "#E69F00", "#56B4E9")
BAR_HATCHES = ("//", "xx", "")
SHORT_LABELS = {
    "Full": "Full Prefill",
    "CacheBlend-15%": "CB-15%",
    "CSKCache-5%": "CSK-5%",
}


def bootstrap_mean_ci(
    values: Iterable[float], *, resamples: int, seed: int
) -> tuple[float, float]:
    import numpy as np

    data = np.asarray(tuple(float(value) for value in values), dtype=float)
    if data.size == 0:
        raise ValueError("cannot bootstrap an empty sample")
    if data.size == 1:
        value = float(data[0])
        return value, value
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, data.size, size=(resamples, data.size))
    means = data[indices].mean(axis=1)
    low, high = np.quantile(means, (0.025, 0.975))
    return float(low), float(high)


def _group_seed(platform_id: str, bucket: str, system: str) -> int:
    digest = hashlib.sha256(
        f"{local.BOOTSTRAP_SEED}:{platform_id}:{bucket}:{system}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big")


def _summaries(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        key = (
            row["platform_id"],
            row["model_id"],
            row["length_bucket"],
            row["system"],
        )
        grouped.setdefault(key, []).append(row)
    result = []
    bucket_rank = {name: index for index, name in enumerate(local.BUCKET_ORDER)}
    system_rank = {
        variant.name: index for index, variant in enumerate(local.SYSTEMS)
    }
    ordered_groups = sorted(
        grouped.items(),
        key=lambda item: (
            item[0][0], bucket_rank[item[0][2]], system_rank[item[0][3]]
        ),
    )
    for (platform_id, model_id, bucket, system), values in ordered_groups:
        repetitions = [int(row["repetition"]) for row in values]
        if len(values) != local.REPETITIONS or set(repetitions) != set(
            range(1, local.REPETITIONS + 1)
        ):
            raise RuntimeError(
                f"incomplete repetitions for {platform_id}/{bucket}/{system}"
            )
        task_ids = {row["task_id"] for row in values}
        if len(task_ids) != 1:
            raise RuntimeError(f"bucket {bucket} must contain exactly one task")
        ttft = [float(row["ttft_ms"]) for row in values]
        low, high = bootstrap_mean_ci(
            ttft,
            resamples=local.BOOTSTRAP_RESAMPLES,
            seed=_group_seed(platform_id, bucket, system),
        )
        result.append(
            {
                "platform_id": platform_id,
                "model_id": model_id,
                "length_bucket": bucket,
                "system": system,
                "sample_count": len(values),
                "task_count": len(task_ids),
                "mean_ttft_ms": statistics.fmean(ttft),
                "ci95_low_ttft_ms": low,
                "ci95_high_ttft_ms": high,
                "std_ttft_ms": statistics.stdev(ttft),
                "mean_prompt_tokens": statistics.fmean(
                    float(row["prompt_tokens"]) for row in values
                ),
                "mean_skill_tokens": statistics.fmean(
                    float(row["skill_tokens"]) for row in values
                ),
                "mean_reuse_ratio": statistics.fmean(
                    float(row["reuse_ratio"]) for row in values
                ),
                "fallback_count": sum(
                    row["fallback"].lower() == "true" for row in values
                ),
            }
        )
    return result


def _complete_platform(
    summaries: list[dict[str, object]], platform_id: str
) -> bool:
    selected = [row for row in summaries if row["platform_id"] == platform_id]
    identities = {
        (str(row["length_bucket"]), str(row["system"])) for row in selected
    }
    expected = {
        (bucket, variant.name)
        for bucket in local.BUCKET_ORDER
        for variant in local.SYSTEMS
    }
    return len(selected) == len(expected) and identities == expected and all(
        int(row["sample_count"]) == local.REPETITIONS
        and int(row["task_count"]) == 1
        for row in selected
    )


def plot_model_panels(
    output_dir: Path,
    summaries: list[dict[str, object]],
    platform_ids: tuple[str, ...],
    *,
    stem: str,
) -> None:
    """Plot one interim model or the final three-model 1x3 figure."""

    if not platform_ids or any(
        not _complete_platform(summaries, platform_id) for platform_id in platform_ids
    ):
        raise RuntimeError("cannot plot an incomplete model panel")
    configure_matplotlib()
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "savefig.dpi": 300,
        }
    )
    figure, axes_value = plt.subplots(
        1,
        len(platform_ids),
        figsize=(4.3 * len(platform_ids), 3.45),
        squeeze=False,
        constrained_layout=False,
    )
    axes = axes_value[0]
    systems = [variant.name for variant in local.SYSTEMS]
    x = np.arange(len(local.BUCKET_ORDER), dtype=float)
    width = 0.24
    for panel_index, (axis, platform_id) in enumerate(
        zip(axes, platform_ids, strict=True)
    ):
        platform_rows = {
            (str(row["length_bucket"]), str(row["system"])): row
            for row in summaries
            if row["platform_id"] == platform_id
        }
        for system_index, system in enumerate(systems):
            selected = [
                platform_rows[(bucket, system)] for bucket in local.BUCKET_ORDER
            ]
            means = np.asarray([float(row["mean_ttft_ms"]) for row in selected])
            lows = np.asarray([float(row["ci95_low_ttft_ms"]) for row in selected])
            highs = np.asarray([float(row["ci95_high_ttft_ms"]) for row in selected])
            axis.bar(
                x + (system_index - 1) * width,
                means,
                width,
                yerr=np.vstack((means - lows, highs - means)),
                capsize=2.0,
                label=SHORT_LABELS[system],
                color=BAR_COLORS[system_index],
                hatch=BAR_HATCHES[system_index],
                edgecolor="black",
                linewidth=0.7,
                error_kw={"elinewidth": 0.8, "capthick": 0.8},
            )
        model_id = next(
            str(row["model_id"])
            for row in summaries
            if row["platform_id"] == platform_id
        )
        axis.set_title(f"({chr(ord('a') + panel_index)}) {model_id}", fontsize=10)
        axis.set_xticks(x, local.BUCKET_ORDER, rotation=24, ha="right")
        axis.set_xlabel("Skill length (tokens)")
        axis.set_ylabel("Server-side TTFT (ms)")
        axis.grid(axis="y", alpha=0.22, linewidth=0.6)
        axis.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.015),
    )
    figure.subplots_adjust(
        left=0.09 / len(platform_ids),
        right=0.995,
        top=0.84,
        bottom=0.22,
        wspace=0.28,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    figure.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def _write_analysis(run_dir: Path, summaries: list[dict[str, object]]) -> None:
    lines = [
        "# Section 6.3 fixed-length TTFT analysis",
        "",
        "Bars report arithmetic mean server-side TTFT. Each bucket contains one "
        "frozen task--Skill pair and five repeated measurements per system. The "
        "95% bootstrap interval therefore describes measurement variability for "
        "that one pair; it is not a confidence interval over a task population.",
        "",
        "| Platform | Length bucket | System | Repeats | Mean TTFT (ms) | 95% CI (ms) |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['platform_id']} | {row['length_bucket']} | {row['system']} | "
            f"{row['sample_count']} | {float(row['mean_ttft_ms']):.3f} | "
            f"[{float(row['ci95_low_ttft_ms']):.3f}, "
            f"{float(row['ci95_high_ttft_ms']):.3f}] |"
        )
    (run_dir / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(run_dir: Path) -> None:
    sample_path = run_dir / "samples.csv"
    if not sample_path.is_file():
        return
    rows = [
        row
        for row in read_csv(sample_path)
        if row["status"] == "valid" and row["invalid_reason"] == ""
    ]
    if not rows:
        return
    try:
        summaries = _summaries(rows)
    except RuntimeError:
        # Smoke and interrupted runs are retained, but are not summarized as a
        # complete five-repeat result.
        return
    write_csv(run_dir / "summary.csv", summaries, SUMMARY_COLUMNS)
    _write_analysis(run_dir, summaries)
    for platform_id in local.ACTIVE_PLATFORM_IDS:
        if _complete_platform(summaries, platform_id):
            plot_model_panels(
                run_dir,
                summaries,
                (platform_id,),
                stem=f"ttft_{platform_id}",
            )


if __name__ == "__main__":
    raise SystemExit("analyze.py is invoked by run.py with the active run directory")
