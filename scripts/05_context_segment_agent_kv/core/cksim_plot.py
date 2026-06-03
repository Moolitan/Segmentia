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


def metric_range(*series: list[float], pad: float = 0.04) -> tuple[float, float]:
    values = [v for seq in series for v in seq]
    return max(0.0, min(values) - pad), min(1.02, max(values) + pad)


def setup_matplotlib() -> Any:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.pyplot as plt

    return plt


def save_layer_curve(rows: list[dict[str, Any]], output_path: str | Path) -> Path:
    plt = setup_matplotlib()
    layer_avg = layer_means(rows)
    layers = [r["layer"] for r in layer_avg]
    key_vals = [r["key_cksim"] for r in layer_avg]
    value_vals = [r["value_cksim"] for r in layer_avg]
    y_min, y_max = metric_range(key_vals, value_vals)

    fig, ax = plt.subplots(figsize=(11, 5.8), constrained_layout=True)
    ax.plot(layers, key_vals, marker="o", markersize=4, linewidth=2.2, color="#2f6f9f", label="Key CKSim")
    ax.plot(layers, value_vals, marker="s", markersize=4, linewidth=2.2, color="#c65f3b", label="Value CKSim")
    ax.set_title("Layer-wise CKSim: Recompute vs Reuse", fontsize=15)
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
    cases = sorted(summary["cases"], key=lambda c: c["mean_value_cksim"])
    labels = [case_label(c) for c in cases]
    key_vals = [c["mean_key_cksim"] for c in cases]
    value_vals = [c["mean_value_cksim"] for c in cases]
    y = list(range(len(cases)))
    x_min, x_max = metric_range(key_vals, value_vals)

    fig, ax = plt.subplots(figsize=(11.5, max(5.2, len(cases) * 0.55)), constrained_layout=True)
    ax.barh([i + 0.18 for i in y], key_vals, height=0.34, color="#2f6f9f", label="Key CKSim")
    ax.barh([i - 0.18 for i in y], value_vals, height=0.34, color="#c65f3b", label="Value CKSim")
    ax.set_title("CKSim by Trace Case", fontsize=15)
    ax.set_xlabel("Mean CKSim across layers")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(x_min, x_max)
    ax.grid(axis="x", color="#d7dde5", linewidth=0.8)
    ax.legend(frameon=False, loc="lower right")
    for i, (key, value) in enumerate(zip(key_vals, value_vals)):
        ax.text(key + 0.004, i + 0.18, f"{key:.3f}", va="center", fontsize=9)
        ax.text(value + 0.004, i - 0.18, f"{value:.3f}", va="center", fontsize=9)

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
