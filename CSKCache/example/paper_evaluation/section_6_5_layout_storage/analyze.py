"""Draw SSD-layout latency and SSD-prefetch TTFT as separate figures."""

from __future__ import annotations

from pathlib import Path

from common.plotting import COLORS, configure_matplotlib, save_figure
from common.schema import SUMMARY_COLUMNS, read_csv, write_csv
from common.statistics import group_rows, median_or_none


GROUP_KEYS = (
    "section", "platform_id", "model_id", "system", "skill_name",
    "skill_tokens", "chunk_tokens", "storage_layout", "host_layout",
    "io_engine", "use_odirect", "correction_strategy",
    "correction_budget_tokens", "input_fingerprint",
)


def analyze(run_dir: Path) -> None:
    rows = [
        row for row in read_csv(run_dir / "samples.csv")
        if row["status"] == "completed" and row["warmup"] != "true"
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

    layout_order = [
        "chunk_all_layers", "chunk_single_layer",
        "packed_chunks_single_layer", "packed_chunks_all_layers",
    ]
    layout_labels = [value.replace("_", " ") for value in layout_order]
    strategies = [
        ("posix", "false", "POSIX Buffered"),
        ("posix", "true", "POSIX O_DIRECT"),
        ("io_uring", "false", "io_uring Buffered"),
        ("io_uring", "true", "io_uring O_DIRECT"),
    ]
    ssd = [row for row in summaries if row["system"] == "SSD-to-Pinned"]
    if ssd:
        x = np.arange(len(layout_order), dtype=float)
        width = 0.19
        figure, axis = plt.subplots(figsize=(7.2, 2.45))
        for index, (engine, odirect, label) in enumerate(strategies):
            heights = []
            for layout in layout_order:
                match = next(
                    row for row in ssd
                    if row["storage_layout"] == layout
                    and row["io_engine"] == engine
                    and row["use_odirect"] == odirect
                )
                heights.append(float(match["median_latency_ms"]))
            axis.bar(
                x + (index - 1.5) * width,
                heights,
                width,
                label=label,
                color=COLORS[index],
                edgecolor="black",
                linewidth=0.8,
            )
        axis.set_xticks(x, layout_labels)
        axis.set_ylabel("Latency (ms)")
        axis.grid(axis="y", alpha=0.25)
        axis.legend(
            loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=4,
            frameon=False, columnspacing=1.7, handletextpad=0.45,
        )
        axis.set_title("12518 tokens", pad=35)
        figure.subplots_adjust(top=0.76, bottom=0.20)
        save_figure(figure, run_dir / "ssd_layout_latency.png")
        plt.close(figure)

    hierarchy = [
        row for row in summaries
        if row["system"] in {"Blocking SSD", "Prefetched SSD"}
    ]
    if hierarchy:
        order = ("Blocking SSD", "Prefetched SSD")
        figure, axis = plt.subplots(figsize=(3.7, 2.35))
        axis.bar(
            range(len(order)),
            [
                float(next(row for row in hierarchy if row["system"] == name)[
                    "median_ttft_ms"
                ])
                for name in order
            ],
            width=0.58,
            color=(COLORS[1], COLORS[0]),
            edgecolor="black",
            linewidth=0.8,
        )
        axis.set_xticks(range(len(order)), order)
        axis.set_ylabel("Latency (ms)")
        axis.grid(axis="y", alpha=0.25)
        figure.subplots_adjust(top=0.93, bottom=0.20, left=0.20, right=0.97)
        save_figure(figure, run_dir / "storage_hierarchy_ttft.png")
        plt.close(figure)


if __name__ == "__main__":
    raise SystemExit("analyze.py is invoked by run.py with the active run directory")
