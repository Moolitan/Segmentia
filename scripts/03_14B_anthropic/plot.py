#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BYTES_PER_GIB = 1024 ** 3

PERF_EVENTS = [
    "cycles",
    "instructions",
    "cache-references",
    "cache-misses",
]


def load_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"文件不存在: {path}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"CSV 文件为空: {path}")

    return df


def clean_phase(series: pd.Series) -> pd.Series:
    return (
        series.fillna("UNKNOWN")
        .astype(str)
        .str.strip("_")
        .replace("", "UNKNOWN")
    )


def phase_boundaries(df: pd.DataFrame) -> pd.DataFrame:
    phase_changed = df["phase"].ne(df["phase"].shift())

    return df.loc[
        phase_changed,
        ["elapsed_seconds", "phase"],
    ].copy()


def add_phase_lines(
    ax: plt.Axes,
    boundaries: pd.DataFrame,
) -> None:
    # 只展示有分析价值的主要阶段，避免标签挤在一起。
    important_phases = {
        "MODEL_LOADING",
        "IDLE_STEADY_WAIT",
        "IDLE_STEADY",
        "POST_INFERENCE",
        "STOPPING",
    }

    for _, row in boundaries.iterrows():
        x = float(row["elapsed_seconds"])
        phase = str(row["phase"])

        if phase.startswith("INFERENCE_"):
            display_phase = "INFERENCE"
        elif phase in important_phases:
            display_phase = phase
        else:
            continue

        ax.axvline(
            x=x,
            linestyle="--",
            linewidth=0.8,
            alpha=0.6,
        )

        ax.text(
            x,
            0.98,
            display_phase,
            rotation=90,
            verticalalignment="top",
            horizontalalignment="right",
            transform=ax.get_xaxis_transform(),
            fontsize=8,
        )


def save_figure(
    fig: plt.Figure,
    output: Path,
) -> None:
    fig.tight_layout()
    fig.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"[OK] {output}")


# =============================================================================
# cgroup 主机内存图
# =============================================================================

def plot_host_memory(
    memory_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    required = [
        "elapsed_seconds",
        "phase",
        "memory_current_bytes",
        "anon_bytes",
        "file_bytes",
        "kernel_bytes",
    ]

    missing = [
        column
        for column in required
        if column not in memory_df.columns
    ]

    if missing:
        raise ValueError(
            f"主机内存 CSV 缺少字段: {missing}"
        )

    df = memory_df.copy()
    boundaries = phase_boundaries(df)

    fig, ax = plt.subplots(figsize=(11, 5.5))

    ax.plot(
        df["elapsed_seconds"],
        df["memory_current_bytes"] / BYTES_PER_GIB,
        label="Total cgroup memory",
        linewidth=2.0,
    )

    ax.plot(
        df["elapsed_seconds"],
        df["anon_bytes"] / BYTES_PER_GIB,
        label="Anonymous memory",
        linewidth=1.5,
    )

    ax.plot(
        df["elapsed_seconds"],
        df["file_bytes"] / BYTES_PER_GIB,
        label="File-backed memory",
        linewidth=1.5,
    )

    ax.plot(
        df["elapsed_seconds"],
        df["kernel_bytes"] / BYTES_PER_GIB,
        label="Kernel memory",
        linewidth=1.5,
    )

    add_phase_lines(ax, boundaries)

    ax.set_xlabel("Elapsed time (s)")
    ax.set_ylabel("Host memory (GiB)")
    ax.set_title("vLLM Host Memory over Time")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    save_figure(
        fig,
        output_dir / "01_host_memory_timeline.png",
    )


def plot_swap_and_faults(
    memory_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    required = [
        "elapsed_seconds",
        "phase",
        "memory_swap_current_bytes",
        "pgmajfault_delta",
    ]

    missing = [
        column
        for column in required
        if column not in memory_df.columns
    ]

    if missing:
        raise ValueError(
            f"Swap CSV 缺少字段: {missing}"
        )

    df = memory_df.copy()
    boundaries = phase_boundaries(df)

    fig, ax1 = plt.subplots(figsize=(11, 5.5))

    ax1.plot(
        df["elapsed_seconds"],
        df["memory_swap_current_bytes"] / BYTES_PER_GIB,
        label="cgroup swap",
        linewidth=2.0,
    )

    ax1.set_xlabel("Elapsed time (s)")
    ax1.set_ylabel("Swap usage (GiB)")
    ax1.set_ylim(bottom=0)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()

    ax2.plot(
        df["elapsed_seconds"],
        df["pgmajfault_delta"],
        label="Major faults per sample",
        linewidth=1.2,
    )

    ax2.set_ylabel("Major page faults per sample")
    ax2.set_ylim(bottom=0)

    add_phase_lines(ax1, boundaries)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="best",
    )

    ax1.set_title(
        "vLLM Swap Usage and Major Page Faults"
    )

    save_figure(
        fig,
        output_dir / "02_swap_and_major_faults.png",
    )


# =============================================================================
# GPU 图
# =============================================================================

def aggregate_gpu_samples(
    gpu_df: pd.DataFrame,
) -> pd.DataFrame:
    required = [
        "elapsed_seconds",
        "phase",
        "gpu_memory_used_mib",
        "gpu_utilization_percent",
    ]

    missing = [
        column
        for column in required
        if column not in gpu_df.columns
    ]

    if missing:
        raise ValueError(
            f"GPU CSV 缺少字段: {missing}"
        )

    df = gpu_df.copy()

    for column in [
        "gpu_memory_used_mib",
        "gpu_utilization_percent",
    ]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    grouped = (
        df.groupby(
            ["elapsed_seconds", "phase"],
            as_index=False,
        )
        .agg(
            gpu_memory_used_mib=(
                "gpu_memory_used_mib",
                "max",
            ),
            gpu_utilization_percent=(
                "gpu_utilization_percent",
                "max",
            ),
        )
        .sort_values("elapsed_seconds")
    )

    return grouped


def plot_gpu(
    gpu_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    df = aggregate_gpu_samples(gpu_df)
    boundaries = phase_boundaries(df)

    fig, ax1 = plt.subplots(figsize=(11, 5.5))

    ax1.plot(
        df["elapsed_seconds"],
        df["gpu_memory_used_mib"] / 1024,
        label="GPU memory",
        linewidth=2.0,
    )

    ax1.set_xlabel("Elapsed time (s)")
    ax1.set_ylabel("GPU memory (GiB)")
    ax1.set_ylim(bottom=0)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()

    gpu_util_smooth = (
        df["gpu_utilization_percent"]
        .rolling(
            window=5,
            center=True,
            min_periods=1,
        )
        .mean()
    )

    ax2.plot(
        df["elapsed_seconds"],
        gpu_util_smooth,
        label="GPU utilization (5-sample mean)",
        linewidth=1.2,
        linestyle="--",
    )

    ax2.set_ylabel("GPU utilization (%)")
    ax2.set_ylim(0, 105)

    add_phase_lines(ax1, boundaries)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="best",
    )

    ax1.set_title(
        "GPU Memory and Utilization over Time"
    )

    save_figure(
        fig,
        output_dir / "03_gpu_memory_and_utilization.png",
    )


# =============================================================================
# 阶段内存汇总
# =============================================================================

def simplify_phase(phase: str) -> str:
    if phase.startswith("INFERENCE_"):
        return "INFERENCE"

    mapping = {
        "PRE_START": "STARTUP",
        "CGROUP_CREATING": "STARTUP",
        "WAITING_START_GATE": "STARTUP",
        "MODEL_LOADING": "MODEL_LOADING",
        "READY": "READY",
        "IDLE_STEADY_WAIT": "IDLE_STEADY",
        "IDLE_STEADY": "IDLE_STEADY",
        "POST_INFERENCE": "POST_INFERENCE",
        "STOPPING": "STOPPING",
        "STOPPED": "STOPPED",
    }

    return mapping.get(phase, phase)


def plot_phase_summary(
    memory_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    required = [
        "phase",
        "memory_current_bytes",
        "anon_bytes",
        "file_bytes",
        "kernel_bytes",
    ]

    missing = [
        column
        for column in required
        if column not in memory_df.columns
    ]

    if missing:
        raise ValueError(
            f"阶段汇总缺少字段: {missing}"
        )

    df = memory_df.copy()
    df["summary_phase"] = (
        df["phase"].map(simplify_phase)
    )

    selected_phases = [
        "MODEL_LOADING",
        "IDLE_STEADY",
        "INFERENCE",
        "POST_INFERENCE",
    ]

    df = df[
        df["summary_phase"].isin(selected_phases)
    ]

    summary = (
        df.groupby("summary_phase")
        .agg(
            mean_memory_bytes=(
                "memory_current_bytes",
                "mean",
            ),
            peak_memory_bytes=(
                "memory_current_bytes",
                "max",
            ),
            mean_anon_bytes=(
                "anon_bytes",
                "mean",
            ),
            mean_file_bytes=(
                "file_bytes",
                "mean",
            ),
            mean_kernel_bytes=(
                "kernel_bytes",
                "mean",
            ),
        )
        .reindex(selected_phases)
        .dropna(how="all")
    )

    summary_gib = summary / BYTES_PER_GIB

    fig, ax = plt.subplots(figsize=(9, 5.5))

    x = np.arange(len(summary_gib))
    width = 0.35

    ax.bar(
        x - width / 2,
        summary_gib["mean_memory_bytes"],
        width=width,
        label="Mean total memory",
    )

    ax.bar(
        x + width / 2,
        summary_gib["peak_memory_bytes"],
        width=width,
        label="Peak total memory",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        summary_gib.index,
        rotation=15,
    )

    ax.set_ylabel("Host memory (GiB)")
    ax.set_title(
        "Mean and Peak Host Memory by Phase"
    )
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    save_figure(
        fig,
        output_dir / "04_phase_memory_summary.png",
    )

    summary_output = (
        output_dir / "phase_memory_summary.csv"
    )

    summary_gib.to_csv(
        summary_output,
        index_label="phase",
    )

    print(f"[OK] {summary_output}")


# =============================================================================
# perf 数据读取
# =============================================================================

def parse_perf_stat_csv(
    path: Path,
    phase_name: str,
) -> pd.DataFrame:
    """
    解析 perf stat -I ... -x, 输出。

    当前实验中的有效行示例：

    1.001047933,3671392,,cycles,
    system.slice/vllm-qwen14b.service,
    8708714,100.00,,

    其中主要使用：
      第 1 列：相对时间，单位秒
      第 2 列：事件值
      第 4 列：事件名称

    perf 输出并不是带表头的标准 CSV，因此需要手动解析。
    """

    if not path.is_file():
        raise FileNotFoundError(
            f"perf 文件不存在: {path}"
        )

    records: list[dict[str, object]] = []

    with path.open(
        "r",
        encoding="utf-8",
        errors="replace",
    ) as file:
        for line_number, raw_line in enumerate(
            file,
            start=1,
        ):
            line = raw_line.strip()

            if not line:
                continue

            if line.startswith("#"):
                continue

            if any(
                marker in line.lower()
                for marker in [
                    "sudo:",
                    "not supported",
                    "not counted",
                    "permission denied",
                    "no permission",
                    "failed",
                    "error:",
                ]
            ):
                continue

            fields = [
                field.strip()
                for field in line.split(",")
            ]

            if len(fields) < 4:
                continue

            time_text = fields[0]
            value_text = fields[1]
            event_text = fields[3]

            # 去除 perf 可能附加的修饰符。
            event_name = event_text.split(":")[0]

            if event_name not in PERF_EVENTS:
                continue

            try:
                elapsed_seconds = float(time_text)
            except ValueError:
                continue

            # 某些 perf 版本会使用分组符号或空格。
            normalized_value = re.sub(
                r"[\s,]",
                "",
                value_text,
            )

            try:
                value = float(normalized_value)
            except ValueError:
                continue

            records.append(
                {
                    "elapsed_seconds": elapsed_seconds,
                    "event": event_name,
                    "value": value,
                    "phase": phase_name,
                    "source_line": line_number,
                }
            )

    if not records:
        raise ValueError(
            f"perf 文件中没有解析到有效数据: {path}"
        )

    df = pd.DataFrame(records)

    df = (
        df.sort_values(
            ["elapsed_seconds", "event"]
        )
        .reset_index(drop=True)
    )

    return df


def perf_to_wide(
    perf_df: pd.DataFrame,
) -> pd.DataFrame:
    wide = (
        perf_df.pivot_table(
            index="elapsed_seconds",
            columns="event",
            values="value",
            aggfunc="sum",
        )
        .reset_index()
        .sort_values("elapsed_seconds")
    )

    wide.columns.name = None

    for event in PERF_EVENTS:
        if event not in wide.columns:
            wide[event] = np.nan

    return wide


def smooth_series(
    series: pd.Series,
    window: int = 5,
) -> pd.Series:
    return (
        series.rolling(
            window=window,
            center=True,
            min_periods=1,
        )
        .mean()
    )


# =============================================================================
# perf 时间序列图
# =============================================================================

def plot_perf_cpu_activity(
    idle_df: pd.DataFrame,
    inference_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    idle = perf_to_wide(idle_df)
    inference = perf_to_wide(inference_df)

    fig, ax = plt.subplots(figsize=(11, 5.5))

    ax.plot(
        idle["elapsed_seconds"],
        smooth_series(idle["cycles"]),
        label="Idle cycles/s",
        linewidth=1.5,
    )

    ax.plot(
        idle["elapsed_seconds"],
        smooth_series(idle["instructions"]),
        label="Idle instructions/s",
        linewidth=1.5,
        linestyle="--",
    )

    inference_time_offset = (
        idle["elapsed_seconds"].max() + 5
    )

    inference_x = (
        inference["elapsed_seconds"]
        + inference_time_offset
    )

    ax.plot(
        inference_x,
        smooth_series(inference["cycles"]),
        label="Inference cycles/s",
        linewidth=1.5,
    )

    ax.plot(
        inference_x,
        smooth_series(
            inference["instructions"]
        ),
        label="Inference instructions/s",
        linewidth=1.5,
        linestyle="--",
    )

    ax.axvline(
        inference_time_offset,
        linestyle=":",
        linewidth=1.2,
    )

    ax.text(
        inference_time_offset,
        0.98,
        "Inference starts",
        rotation=90,
        verticalalignment="top",
        horizontalalignment="right",
        transform=ax.get_xaxis_transform(),
        fontsize=8,
    )

    ax.set_xlabel(
        "Relative sampling time (s)"
    )
    ax.set_ylabel(
        "Hardware events per sampling interval"
    )
    ax.set_title(
        "CPU Activity during Idle and Multi-Turn Inference"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    save_figure(
        fig,
        output_dir / "05_perf_cpu_activity_timeline.png",
    )


def plot_perf_cache_activity(
    idle_df: pd.DataFrame,
    inference_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    idle = perf_to_wide(idle_df)
    inference = perf_to_wide(inference_df)

    fig, ax = plt.subplots(figsize=(11, 5.5))

    ax.plot(
        idle["elapsed_seconds"],
        smooth_series(
            idle["cache-references"]
        ),
        label="Idle cache references/s",
        linewidth=1.5,
    )

    ax.plot(
        idle["elapsed_seconds"],
        smooth_series(
            idle["cache-misses"]
        ),
        label="Idle cache misses/s",
        linewidth=1.5,
        linestyle="--",
    )

    inference_time_offset = (
        idle["elapsed_seconds"].max() + 5
    )

    inference_x = (
        inference["elapsed_seconds"]
        + inference_time_offset
    )

    ax.plot(
        inference_x,
        smooth_series(
            inference["cache-references"]
        ),
        label="Inference cache references/s",
        linewidth=1.5,
    )

    ax.plot(
        inference_x,
        smooth_series(
            inference["cache-misses"]
        ),
        label="Inference cache misses/s",
        linewidth=1.5,
        linestyle="--",
    )

    ax.axvline(
        inference_time_offset,
        linestyle=":",
        linewidth=1.2,
    )

    ax.text(
        inference_time_offset,
        0.98,
        "Inference starts",
        rotation=90,
        verticalalignment="top",
        horizontalalignment="right",
        transform=ax.get_xaxis_transform(),
        fontsize=8,
    )

    ax.set_xlabel(
        "Relative sampling time (s)"
    )
    ax.set_ylabel(
        "Cache events per sampling interval"
    )
    ax.set_title(
        "CPU Cache Activity during Idle and Multi-Turn Inference"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    save_figure(
        fig,
        output_dir / "06_perf_cache_activity_timeline.png",
    )


# =============================================================================
# perf 汇总统计
# =============================================================================

def summarize_perf_phase(
    perf_df: pd.DataFrame,
    phase_name: str,
) -> dict[str, float | str]:
    wide = perf_to_wide(perf_df)

    start_time = float(
        wide["elapsed_seconds"].min()
    )

    end_time = float(
        wide["elapsed_seconds"].max()
    )

    sample_count = len(wide)

    # -I 1000 时，每行 value 本身基本就是该 1 秒区间的事件数。
    # 使用所有区间的均值作为 events/s。
    mean_cycles = float(
        wide["cycles"].mean()
    )

    mean_instructions = float(
        wide["instructions"].mean()
    )

    mean_cache_references = float(
        wide["cache-references"].mean()
    )

    mean_cache_misses = float(
        wide["cache-misses"].mean()
    )

    total_cycles = float(
        wide["cycles"].sum()
    )

    total_instructions = float(
        wide["instructions"].sum()
    )

    total_cache_references = float(
        wide["cache-references"].sum()
    )

    total_cache_misses = float(
        wide["cache-misses"].sum()
    )

    ipc = (
        total_instructions / total_cycles
        if total_cycles > 0
        else np.nan
    )

    cache_miss_rate = (
        100.0
        * total_cache_misses
        / total_cache_references
        if total_cache_references > 0
        else np.nan
    )

    return {
        "phase": phase_name,
        "first_sample_second": start_time,
        "last_sample_second": end_time,
        "sample_count": sample_count,
        "cycles_per_second": mean_cycles,
        "instructions_per_second": mean_instructions,
        "cache_references_per_second": (
            mean_cache_references
        ),
        "cache_misses_per_second": (
            mean_cache_misses
        ),
        "ipc": ipc,
        "cache_miss_rate_percent": (
            cache_miss_rate
        ),
    }


def build_perf_summary(
    idle_df: pd.DataFrame,
    inference_df: pd.DataFrame,
) -> pd.DataFrame:
    records = [
        summarize_perf_phase(
            idle_df,
            "IDLE",
        ),
        summarize_perf_phase(
            inference_df,
            "INFERENCE",
        ),
    ]

    summary = pd.DataFrame(records)
    summary = summary.set_index("phase")

    idle_row = summary.loc["IDLE"]
    inference_row = summary.loc["INFERENCE"]

    comparison: dict[str, float | str] = {
        "phase": "INFERENCE/IDLE",
        "first_sample_second": np.nan,
        "last_sample_second": np.nan,
        "sample_count": np.nan,
    }

    ratio_columns = [
        "cycles_per_second",
        "instructions_per_second",
        "cache_references_per_second",
        "cache_misses_per_second",
        "ipc",
        "cache_miss_rate_percent",
    ]

    for column in ratio_columns:
        idle_value = float(idle_row[column])
        inference_value = float(
            inference_row[column]
        )

        comparison[column] = (
            inference_value / idle_value
            if idle_value != 0
            else np.nan
        )

    comparison_df = (
        pd.DataFrame([comparison])
        .set_index("phase")
    )

    return pd.concat(
        [summary, comparison_df],
        axis=0,
    )


def plot_perf_idle_vs_inference(
    summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    phase_summary = summary.loc[
        ["IDLE", "INFERENCE"]
    ]

    metrics = [
        "cycles_per_second",
        "instructions_per_second",
        "cache_references_per_second",
        "cache_misses_per_second",
    ]

    labels = [
        "Cycles",
        "Instructions",
        "Cache references",
        "Cache misses",
    ]

    # 各事件数量级差异较大，因此相对于 idle 做归一化。
    normalized = pd.DataFrame(
        index=phase_summary.index
    )

    for metric in metrics:
        idle_value = float(
            phase_summary.loc["IDLE", metric]
        )

        if idle_value == 0:
            normalized[metric] = np.nan
        else:
            normalized[metric] = (
                phase_summary[metric] / idle_value
            )

    fig, ax = plt.subplots(figsize=(10, 5.5))

    x = np.arange(len(metrics))
    width = 0.35

    ax.bar(
        x - width / 2,
        normalized.loc["IDLE", metrics],
        width=width,
        label="Idle",
    )

    ax.bar(
        x + width / 2,
        normalized.loc["INFERENCE", metrics],
        width=width,
        label="Inference",
    )

    ax.axhline(
        1.0,
        linestyle="--",
        linewidth=1.0,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        labels,
        rotation=15,
    )

    ax.set_ylabel(
        "Normalized event rate (Idle = 1.0)"
    )

    ax.set_title(
        "Host CPU and Cache Activity: Idle vs. Inference"
    )

    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    save_figure(
        fig,
        output_dir / "07_perf_idle_vs_inference.png",
    )


def plot_perf_derived_metrics(
    summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    phase_summary = summary.loc[
        ["IDLE", "INFERENCE"]
    ]

    fig, ax1 = plt.subplots(figsize=(8.5, 5.5))

    x = np.arange(2)
    width = 0.35

    ipc_values = phase_summary["ipc"].to_numpy()

    miss_rate_values = (
        phase_summary[
            "cache_miss_rate_percent"
        ].to_numpy()
    )

    bars1 = ax1.bar(
        x - width / 2,
        ipc_values,
        width=width,
        label="IPC",
    )

    ax1.set_ylabel(
        "Instructions per cycle"
    )
    ax1.set_ylim(
        bottom=0,
        top=max(ipc_values) * 1.25
        if np.isfinite(ipc_values).any()
        else 1,
    )

    ax2 = ax1.twinx()

    bars2 = ax2.bar(
        x + width / 2,
        miss_rate_values,
        width=width,
        label="Cache miss rate",
    )

    ax2.set_ylabel(
        "Cache miss rate (%)"
    )
    ax2.set_ylim(
        bottom=0,
        top=max(miss_rate_values) * 1.25
        if np.isfinite(miss_rate_values).any()
        else 1,
    )

    ax1.set_xticks(x)
    ax1.set_xticklabels(
        ["Idle", "Inference"]
    )

    ax1.set_title(
        "IPC and Cache Miss Rate by Phase"
    )
    ax1.grid(axis="y", alpha=0.3)

    ax1.legend(
        [bars1, bars2],
        ["IPC", "Cache miss rate"],
        loc="best",
    )

    save_figure(
        fig,
        output_dir / "08_perf_derived_metrics.png",
    )


def print_perf_summary(
    summary: pd.DataFrame,
) -> None:
    print()
    print("=" * 72)
    print("perf 阶段汇总")
    print("=" * 72)

    display_columns = [
        "cycles_per_second",
        "instructions_per_second",
        "cache_references_per_second",
        "cache_misses_per_second",
        "ipc",
        "cache_miss_rate_percent",
    ]

    print(
        summary[display_columns].to_string(
            float_format=lambda value: (
                f"{value:,.4f}"
            )
        )
    )

    if (
        "IDLE" not in summary.index
        or "INFERENCE" not in summary.index
    ):
        return

    idle = summary.loc["IDLE"]
    inference = summary.loc["INFERENCE"]

    print()
    print("推理阶段相对于空闲阶段：")

    for column, label in [
        (
            "cycles_per_second",
            "cycles/s",
        ),
        (
            "instructions_per_second",
            "instructions/s",
        ),
        (
            "cache_references_per_second",
            "cache references/s",
        ),
        (
            "cache_misses_per_second",
            "cache misses/s",
        ),
    ]:
        idle_value = float(idle[column])
        inference_value = float(
            inference[column]
        )

        if idle_value == 0:
            print(
                f"  {label}: 空闲阶段为 0，"
                "无法计算倍数"
            )
            continue

        ratio = inference_value / idle_value

        print(
            f"  {label}: "
            f"{ratio:.3f}×"
        )

    print(
        "  IPC: "
        f"{idle['ipc']:.3f} "
        "→ "
        f"{inference['ipc']:.3f}"
    )

    print(
        "  Cache miss rate: "
        f"{idle['cache_miss_rate_percent']:.3f}% "
        "→ "
        f"{inference['cache_miss_rate_percent']:.3f}%"
    )


# =============================================================================
# 主函数
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot vLLM cgroup, GPU and perf results."
        )
    )

    parser.add_argument(
        "result_dir",
        type=Path,
        help=(
            "实验结果目录，例如 "
            "results/memory/20260702-072518"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "图片输出目录，默认是 "
            "<result_dir>/figures"
        ),
    )

    args = parser.parse_args()

    result_dir = args.result_dir.resolve()

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else result_dir / "figures"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    memory_path = (
        result_dir / "vllm_cgroup_memory.csv"
    )

    gpu_path = (
        result_dir / "vllm_gpu_memory.csv"
    )

    perf_idle_path = (
        result_dir / "perf_idle.csv"
    )

    perf_inference_path = (
        result_dir / "perf_inference.csv"
    )

    # -------------------------------------------------------------------------
    # cgroup 内存数据
    # -------------------------------------------------------------------------

    memory_df = load_csv(memory_path)

    memory_df["phase"] = clean_phase(
        memory_df["phase"]
    )

    # -------------------------------------------------------------------------
    # GPU 数据
    # -------------------------------------------------------------------------

    gpu_df = load_csv(gpu_path)

    gpu_df["phase"] = clean_phase(
        gpu_df["phase"]
    )

    # -------------------------------------------------------------------------
    # perf 数据
    # -------------------------------------------------------------------------

    perf_idle_df = parse_perf_stat_csv(
        perf_idle_path,
        "IDLE",
    )

    perf_inference_df = parse_perf_stat_csv(
        perf_inference_path,
        "INFERENCE",
    )

    # -------------------------------------------------------------------------
    # 原有四张图
    # -------------------------------------------------------------------------

    plot_host_memory(
        memory_df,
        output_dir,
    )

    plot_swap_and_faults(
        memory_df,
        output_dir,
    )

    plot_gpu(
        gpu_df,
        output_dir,
    )

    plot_phase_summary(
        memory_df,
        output_dir,
    )

    # -------------------------------------------------------------------------
    # 新增 perf 图
    # -------------------------------------------------------------------------

    plot_perf_cpu_activity(
        perf_idle_df,
        perf_inference_df,
        output_dir,
    )

    plot_perf_cache_activity(
        perf_idle_df,
        perf_inference_df,
        output_dir,
    )

    perf_summary = build_perf_summary(
        perf_idle_df,
        perf_inference_df,
    )

    plot_perf_idle_vs_inference(
        perf_summary,
        output_dir,
    )

    plot_perf_derived_metrics(
        perf_summary,
        output_dir,
    )

    perf_summary_output = (
        output_dir / "perf_phase_summary.csv"
    )

    perf_summary.to_csv(
        perf_summary_output,
        index_label="phase",
    )

    print(
        f"[OK] {perf_summary_output}"
    )

    print_perf_summary(perf_summary)

    print(
        f"\n全部图片已保存到: {output_dir}"
    )


if __name__ == "__main__":
    main()