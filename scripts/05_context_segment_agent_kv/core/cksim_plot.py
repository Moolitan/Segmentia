"""Plot helpers for trace replay CKSim results."""
from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path
from typing import Any


def load_summary(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_layer_rows(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["occurrence"] = int(row["occurrence"])
            row["skill_tokens"] = int(row["skill_tokens"])
            row["key_cksim"] = float(row["key_cksim"])
            row["value_cksim"] = float(row["value_cksim"])
            row["key_token_mean"] = float(row["key_token_mean"])
            row["value_token_mean"] = float(row["value_token_mean"])
            row["layer_index"] = layer_index(row["layer"])
            rows.append(row)
    return rows


def layer_index(layer: str) -> int:
    match = re.search(r"layers\.(\d+)\.", layer)
    if not match:
        raise ValueError(f"Could not parse layer index from {layer!r}")
    return int(match.group(1))


def short_task(task: str) -> str:
    aliases = {
        "internal_comms_incident_update": "internal-comms task",
        "doc_coauthoring_design_doc": "doc-coauthoring task",
        "mcp_server_and_spec": "mcp/spec task",
        "web_artifact_with_theme": "web artifact task",
        "launch_poster_page_pack": "launch poster task",
        "slack_launch_pack": "slack launch task",
    }
    return aliases.get(task, task.replace("_", " "))


def case_label(case: dict[str, Any]) -> str:
    return f"{short_task(case['task'])} | {case['skill_name']} | occ{case['occurrence']}"


def layer_means(rows: list[dict[str, Any]]) -> list[dict[str, float]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["layer_index"], []).append(row)
    means = []
    for idx in sorted(grouped):
        subset = grouped[idx]
        means.append(
            {
                "layer": idx,
                "key_cksim": sum(r["key_cksim"] for r in subset) / len(subset),
                "value_cksim": sum(r["value_cksim"] for r in subset) / len(subset),
            }
        )
    return means


def layer_means_by_comparison(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, float]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row.get("comparison", "recompute_vs_reuse"), []).append(row)
    return {comparison: layer_means(subset) for comparison, subset in grouped.items()}


def comparison_label(comparison: str) -> str:
    labels = {
        "recompute_vs_reuse": "Reuse (RoPE)",
        "recompute_vs_reuse_no_rope": "Reuse (no RoPE)",
    }
    return labels.get(comparison, comparison.replace("_", " "))


def case_comparison_metrics(case: dict[str, Any]) -> dict[str, dict[str, float]]:
    if "comparisons" in case:
        return case["comparisons"]
    return {
        case.get("comparison", "recompute_vs_reuse"): {
            "mean_key_cksim": case["mean_key_cksim"],
            "mean_value_cksim": case["mean_value_cksim"],
        }
    }


def case_sort_value(case: dict[str, Any]) -> float:
    metrics = case_comparison_metrics(case)
    if "recompute_vs_reuse" in metrics:
        return float(metrics["recompute_vs_reuse"]["mean_value_cksim"])
    return min(float(v["mean_value_cksim"]) for v in metrics.values())


def metric_range(*series: list[float], pad: float = 0.04) -> tuple[float, float]:
    values = [v for seq in series for v in seq]
    return max(0.0, min(values) - pad), min(1.02, max(values) + pad)


def setup_matplotlib() -> Any:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.pyplot as plt

    return plt


def save_layer_curve(rows: list[dict[str, Any]], output_path: str | Path) -> Path:
    plt = setup_matplotlib()
    by_comparison = layer_means_by_comparison(rows)
    comparisons = [
        c
        for c in ("recompute_vs_reuse", "recompute_vs_reuse_no_rope")
        if c in by_comparison
    ] + sorted(
        c
        for c in by_comparison
        if c not in {"recompute_vs_reuse", "recompute_vs_reuse_no_rope"}
    )
    all_vals: list[list[float]] = []
    for comparison in comparisons:
        all_vals.append([r["key_cksim"] for r in by_comparison[comparison]])
        all_vals.append([r["value_cksim"] for r in by_comparison[comparison]])
    y_min, y_max = metric_range(*all_vals)

    styles = {
        "recompute_vs_reuse": {
            "key_color": "#2f6f9f",
            "value_color": "#c65f3b",
            "linestyle": "-",
        },
        "recompute_vs_reuse_no_rope": {
            "key_color": "#6a7f2b",
            "value_color": "#7a5aa6",
            "linestyle": "--",
        },
    }

    fig, ax = plt.subplots(figsize=(11, 5.8), constrained_layout=True)
    for comparison in comparisons:
        layer_avg = by_comparison[comparison]
        layers = [r["layer"] for r in layer_avg]
        key_vals = [r["key_cksim"] for r in layer_avg]
        value_vals = [r["value_cksim"] for r in layer_avg]
        style = styles.get(
            comparison,
            {"key_color": "#2f6f9f", "value_color": "#c65f3b", "linestyle": ":"},
        )
        label = comparison_label(comparison)
        ax.plot(
            layers,
            key_vals,
            marker="o",
            markersize=4,
            linewidth=2.2,
            linestyle=style["linestyle"],
            color=style["key_color"],
            label=f"Key CKSim - {label}",
        )
        ax.plot(
            layers,
            value_vals,
            marker="s",
            markersize=4,
            linewidth=2.2,
            linestyle=style["linestyle"],
            color=style["value_color"],
            label=f"Value CKSim - {label}",
        )
    ax.set_title("Layer-wise CKSim: Recompute vs Reuse Variants", fontsize=15)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Average CKSim across cases")
    ax.set_ylim(y_min, y_max)
    ax.grid(color="#d7dde5", linewidth=0.8)
    ax.legend(frameon=False, loc="upper right")
    ax.text(
        0.01,
        0.03,
        "Value KV diverges much more than Key KV in middle and late layers.",
        transform=ax.transAxes,
        fontsize=10,
        color="#4d5562",
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)
    return output


def save_case_gap(summary: dict[str, Any], output_path: str | Path) -> Path:
    plt = setup_matplotlib()
    cases = sorted(summary["cases"], key=case_sort_value)
    labels = [case_label(c) for c in cases]
    comparisons = [
        c
        for c in ("recompute_vs_reuse", "recompute_vs_reuse_no_rope")
        if any(c in case_comparison_metrics(case) for case in cases)
    ]
    if not comparisons:
        comparisons = sorted(
            {cmp_name for case in cases for cmp_name in case_comparison_metrics(case)}
        )
    series: dict[str, dict[str, list[float]]] = {}
    for comparison in comparisons:
        series[comparison] = {
            "key": [
                float(case_comparison_metrics(c).get(comparison, {}).get("mean_key_cksim", 0.0))
                for c in cases
            ],
            "value": [
                float(case_comparison_metrics(c).get(comparison, {}).get("mean_value_cksim", 0.0))
                for c in cases
            ],
        }
    y = list(range(len(cases)))
    x_min, x_max = metric_range(
        *[series[c][metric] for c in comparisons for metric in ("key", "value")]
    )

    fig, ax = plt.subplots(figsize=(11.5, max(5.2, len(cases) * 0.55)), constrained_layout=True)
    palette = {
        "recompute_vs_reuse": ("#2f6f9f", "#c65f3b"),
        "recompute_vs_reuse_no_rope": ("#6a7f2b", "#7a5aa6"),
    }
    height = 0.18 if len(comparisons) > 1 else 0.34
    offsets = {
        "recompute_vs_reuse": (0.27, 0.09),
        "recompute_vs_reuse_no_rope": (-0.09, -0.27),
    }
    for comparison in comparisons:
        key_color, value_color = palette.get(comparison, ("#2f6f9f", "#c65f3b"))
        key_offset, value_offset = offsets.get(comparison, (0.18, -0.18))
        label = comparison_label(comparison)
        key_vals = series[comparison]["key"]
        value_vals = series[comparison]["value"]
        ax.barh(
            [i + key_offset for i in y],
            key_vals,
            height=height,
            color=key_color,
            label=f"Key - {label}",
        )
        ax.barh(
            [i + value_offset for i in y],
            value_vals,
            height=height,
            color=value_color,
            label=f"Value - {label}",
        )
        for i, (key, value) in enumerate(zip(key_vals, value_vals)):
            if key:
                ax.text(key + 0.004, i + key_offset, f"{key:.3f}", va="center", fontsize=8)
            if value:
                ax.text(value + 0.004, i + value_offset, f"{value:.3f}", va="center", fontsize=8)

    ax.set_title("CKSim by Trace Case", fontsize=15)
    ax.set_xlabel("Mean CKSim across layers")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(x_min, x_max)
    ax.grid(axis="x", color="#d7dde5", linewidth=0.8)
    ax.legend(frameon=False, loc="lower right")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)
    return output


def plot_trace_cksim(
    summary_path: str | Path,
    csv_path: str | Path,
    output_dir: str | Path,
) -> list[Path]:
    summary = load_summary(summary_path)
    rows = load_layer_rows(csv_path)
    if not summary.get("cases"):
        raise ValueError(f"No cases found in {summary_path}")
    if not rows:
        raise ValueError(f"No layer rows found in {csv_path}")

    out = Path(output_dir)
    return [
        save_layer_curve(rows, out / "cksim_layer_curve.png"),
        save_case_gap(summary, out / "cksim_case_gap.png"),
    ]
