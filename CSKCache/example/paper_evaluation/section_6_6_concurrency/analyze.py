"""Aggregate fixed-concurrency throughput replicas and draw two Skill panels."""

from __future__ import annotations

import json
from pathlib import Path

from common.plotting import configure_matplotlib, save_figure
from common.schema import read_csv, write_csv
from common.statistics import group_rows


REPLICA_KEYS = (
    "section",
    "platform_id",
    "model_id",
    "system",
    "skill_name",
    "skill_tokens",
    "chunk_tokens",
    "correction_strategy",
    "correction_ratio",
    "concurrency",
    "replica",
    "input_fingerprint",
)
SUMMARY_KEYS = tuple(key for key in REPLICA_KEYS if key != "replica")
REPLICA_COLUMNS = (
    "schema_version",
    *REPLICA_KEYS,
    "completed_requests",
    "measurement_seconds",
    "throughput_requests_per_s",
    "p50_ttft_ms",
    "p95_ttft_ms",
    "p50_latency_ms",
    "fallback_count",
)
SUMMARY_COLUMNS = (
    "schema_version",
    *SUMMARY_KEYS,
    "replica_count",
    "completed_requests",
    "median_throughput_requests_per_s",
    "min_throughput_requests_per_s",
    "max_throughput_requests_per_s",
    "median_p50_ttft_ms",
    "median_p95_ttft_ms",
    "fallback_count",
)
MEMORY_REPLICA_COLUMNS = (
    "schema_version",
    "platform_id",
    "system",
    "skill_name",
    "skill_tokens",
    "concurrency",
    "replica",
    "host_pool_capacity_bytes",
    "measurement_sample_count",
    "measurement_error_count",
    "peak_lmcache_host_allocated_bytes",
    "peak_lmcache_active_memory_objects",
    "peak_server_process_group_rss_bytes",
    "minimum_system_available_bytes",
)
MEMORY_SUMMARY_COLUMNS = (
    "schema_version",
    "platform_id",
    "system",
    "skill_name",
    "skill_tokens",
    "concurrency",
    "replica_count",
    "median_peak_lmcache_host_allocated_bytes",
    "min_peak_lmcache_host_allocated_bytes",
    "max_peak_lmcache_host_allocated_bytes",
    "median_peak_lmcache_active_memory_objects",
    "median_peak_server_process_group_rss_bytes",
    "measurement_error_count",
)
SYSTEM_STYLES = {
    "Full Prefill": {"color": "#54A24B", "marker": "^", "label": "Full Prefill"},
    "SkillCache-5%": {"color": "#3475B7", "marker": "o", "label": "SkillCache"},
}


def _percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot compute a percentile of an empty sequence")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _replica_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for key, values in group_rows(rows, REPLICA_KEYS).items():
        record = dict(zip(REPLICA_KEYS, key, strict=True))
        throughputs = {
            float(row["throughput_requests_per_s"]) for row in values
        }
        durations = {float(row["batch_elapsed_ms"]) / 1000.0 for row in values}
        if len(throughputs) != 1 or len(durations) != 1:
            raise RuntimeError("one replica contains inconsistent window metadata")
        ttfts = [float(row["ttft_ms"]) for row in values]
        latencies = [float(row["latency_ms"]) for row in values]
        output.append(
            {
                **record,
                "completed_requests": len(values),
                "measurement_seconds": durations.pop(),
                "throughput_requests_per_s": throughputs.pop(),
                "p50_ttft_ms": _percentile(ttfts, 0.50),
                "p95_ttft_ms": _percentile(ttfts, 0.95),
                "p50_latency_ms": _percentile(latencies, 0.50),
                "fallback_count": sum(
                    row["fallback"] == "true" for row in values
                ),
            }
        )
    return output


def _summary_rows(
    replicas: list[dict[str, object]],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    string_rows = [
        {key: str(value) for key, value in row.items()} for row in replicas
    ]
    for key, values in group_rows(string_rows, SUMMARY_KEYS).items():
        record = dict(zip(SUMMARY_KEYS, key, strict=True))
        throughputs = [
            float(row["throughput_requests_per_s"]) for row in values
        ]
        output.append(
            {
                **record,
                "replica_count": len(values),
                "completed_requests": sum(
                    int(row["completed_requests"]) for row in values
                ),
                "median_throughput_requests_per_s": _percentile(
                    throughputs, 0.50
                ),
                "min_throughput_requests_per_s": min(throughputs),
                "max_throughput_requests_per_s": max(throughputs),
                "median_p50_ttft_ms": _percentile(
                    [float(row["p50_ttft_ms"]) for row in values], 0.50
                ),
                "median_p95_ttft_ms": _percentile(
                    [float(row["p95_ttft_ms"]) for row in values], 0.50
                ),
                "fallback_count": sum(
                    int(row["fallback_count"]) for row in values
                ),
            }
        )
    return output


def _memory_replica_rows(run_dir: Path) -> list[dict[str, object]]:
    state = json.loads((run_dir / "run_state.json").read_text(encoding="utf-8"))
    output: list[dict[str, object]] = []
    for point in state["cases"].values():
        if point.get("status") != "completed":
            continue
        result_path = Path(str(point["attempt_dir"])) / "result.json"
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        metadata = payload["point_metadata"]
        memory = payload["memory"]
        output.append(
            {
                **metadata,
                **{column: memory.get(column) for column in MEMORY_REPLICA_COLUMNS
                   if column not in metadata},
            }
        )
    return output


def _memory_summary_rows(
    replicas: list[dict[str, object]],
) -> list[dict[str, object]]:
    keys = (
        "platform_id",
        "system",
        "skill_name",
        "skill_tokens",
        "concurrency",
    )
    string_rows = [
        {key: "" if value is None else str(value) for key, value in row.items()}
        for row in replicas
    ]
    output: list[dict[str, object]] = []
    for key, values in group_rows(string_rows, keys).items():
        record = dict(zip(keys, key, strict=True))

        def numbers(field: str) -> list[float]:
            return [float(row[field]) for row in values if row[field] != ""]

        host_bytes = numbers("peak_lmcache_host_allocated_bytes")
        active_objects = numbers("peak_lmcache_active_memory_objects")
        rss_bytes = numbers("peak_server_process_group_rss_bytes")
        output.append(
            {
                **record,
                "replica_count": len(values),
                "median_peak_lmcache_host_allocated_bytes": (
                    "" if not host_bytes else _percentile(host_bytes, 0.50)
                ),
                "min_peak_lmcache_host_allocated_bytes": (
                    "" if not host_bytes else min(host_bytes)
                ),
                "max_peak_lmcache_host_allocated_bytes": (
                    "" if not host_bytes else max(host_bytes)
                ),
                "median_peak_lmcache_active_memory_objects": (
                    "" if not active_objects
                    else _percentile(active_objects, 0.50)
                ),
                "median_peak_server_process_group_rss_bytes": (
                    "" if not rss_bytes else _percentile(rss_bytes, 0.50)
                ),
                "measurement_error_count": sum(
                    int(row["measurement_error_count"]) for row in values
                ),
            }
        )
    return output


def _plot(run_dir: Path, summaries: list[dict[str, object]]) -> None:
    if not summaries:
        return
    configure_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib.transforms import ScaledTranslation

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 7.5,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "axes.linewidth": 0.75,
            "axes.spines.top": True,
            "axes.spines.right": True,
        }
    )
    for platform_id in dict.fromkeys(
        str(row["platform_id"]) for row in summaries
    ):
        selected = [
            row for row in summaries if row["platform_id"] == platform_id
        ]
        skills = sorted(
            {str(row["skill_name"]): int(row["skill_tokens"]) for row in selected}.items(),
            key=lambda item: item[1],
        )
        figure, axes = plt.subplots(
            1, len(skills), figsize=(3.45, 1.10), sharey=True
        )
        if len(skills) == 1:
            axes = [axes]
        for panel, (axis, (skill_name, skill_tokens)) in enumerate(
            zip(axes, skills, strict=True)
        ):
            panel_rows = [
                row for row in selected if row["skill_name"] == skill_name
            ]
            for system, style in SYSTEM_STYLES.items():
                values = sorted(
                    (row for row in panel_rows if row["system"] == system),
                    key=lambda row: int(row["concurrency"]),
                )
                if not values:
                    continue
                xs = [int(row["concurrency"]) for row in values]
                ys = [
                    float(row["median_throughput_requests_per_s"])
                    for row in values
                ]
                axis.plot(
                    xs,
                    ys,
                    label=style["label"],
                    color=style["color"],
                    marker=style["marker"],
                    linewidth=1.35,
                    markersize=3.6,
                )
                axis.fill_between(
                    xs,
                    [float(row["min_throughput_requests_per_s"]) for row in values],
                    [float(row["max_throughput_requests_per_s"]) for row in values],
                    color=style["color"],
                    alpha=0.10,
                    linewidth=0,
                )
            axis.text(
                0.5,
                0.0,
                f"({chr(ord('a') + panel)}) {skill_tokens / 1000:.1f}K-token Skill",
                transform=axis.transAxes
                + ScaledTranslation(0.0, -0.42, figure.dpi_scale_trans),
                ha="center",
                va="top",
                fontsize=7.5,
            )
            axis.set_xlabel("Concurrent requests", labelpad=2)
            axis.set_xticks(sorted({int(row["concurrency"]) for row in panel_rows}))
            axis.grid(axis="y", alpha=0.22, linewidth=0.6)
        axes[0].set_ylabel("Throughput (req/s)", labelpad=2)
        handles, labels = axes[0].get_legend_handles_labels()
        figure.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.905),
            ncol=len(labels),
            frameon=False,
            columnspacing=1.4,
            handletextpad=0.45,
        )
        figure.subplots_adjust(
            left=0.135, right=0.985, bottom=0.383, top=0.899, wspace=0.18
        )
        save_figure(figure, run_dir / f"concurrency_throughput_{platform_id}.png")
        plt.close(figure)


def analyze(run_dir: Path) -> None:
    rows = [
        row for row in read_csv(run_dir / "samples.csv")
        if row["status"] == "completed" and row["warmup"] == "false"
    ]
    replicas = _replica_rows(rows)
    summaries = _summary_rows(replicas)
    memory_replicas = _memory_replica_rows(run_dir)
    memory_summaries = _memory_summary_rows(memory_replicas)
    write_csv(run_dir / "replica_summary.csv", replicas, REPLICA_COLUMNS)
    write_csv(run_dir / "summary.csv", summaries, SUMMARY_COLUMNS)
    write_csv(
        run_dir / "memory_replica_summary.csv",
        memory_replicas,
        MEMORY_REPLICA_COLUMNS,
    )
    write_csv(
        run_dir / "memory_summary.csv",
        memory_summaries,
        MEMORY_SUMMARY_COLUMNS,
    )
    _plot(run_dir, summaries)


if __name__ == "__main__":
    raise SystemExit("analyze.py is invoked by run.py with the active run directory")
