#!/usr/bin/env python3
"""Plot layer-wise Direct-to-Full cosine for every captured Skill."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch

from analyze_context_free_residual import read_layer, relocate_neox_rope, write_csv
from analyze_layer_axis import (
    HEAD_DIM,
    NUM_KV_HEADS,
    NUM_LAYERS,
    CaseRecord,
    validate_case,
)


COLORS = ("#0072B2", "#E69F00", "#009E73", "#D55E00")
LINE_STYLES = ("-", "--", "-.", ":")


def fidelity_summary(
    direct: torch.Tensor, full: torch.Tensor
) -> tuple[float, float, float, float, float]:
    """Return cosine and global normalized-L2 statistics."""
    direct = direct.to(torch.float32)
    full = full.to(torch.float32)
    dot = (direct * full).sum(dim=-1)
    denominator = torch.sqrt(
        direct.square().sum(dim=-1) * full.square().sum(dim=-1)
    ).clamp_min(torch.finfo(torch.float32).eps)
    cosine = dot / denominator
    sse = float((direct - full).square().sum())
    full_sq_norm = float(full.square().sum())
    normalized_l2 = (sse / full_sq_norm) ** 0.5
    return (
        float(cosine.mean()),
        float(cosine.median()),
        sse,
        full_sq_norm,
        normalized_l2,
    )


def analyze_case(
    case: CaseRecord, theta: float
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    expected_shape = (2, case.token_count, NUM_KV_HEADS * HEAD_DIM)
    for layer in range(NUM_LAYERS):
        offline_kv, offline_meta = read_layer(case.offline_layers[layer])
        online_kv, online_meta = read_layer(case.online_layers[layer])
        if tuple(offline_kv.shape) != expected_shape or offline_kv.shape != online_kv.shape:
            raise ValueError(
                f"KV shape mismatch for {case.case_id} layer={layer}: "
                f"offline={tuple(offline_kv.shape)} online={tuple(online_kv.shape)}"
            )
        if (
            offline_meta.get("cached_positions", {}).get("start") != 0
            or online_meta.get("cached_positions", {}).get("start") != case.shift
        ):
            raise ValueError(
                f"cached position mismatch for {case.case_id} layer={layer}"
            )

        offline_key = offline_kv[0].reshape(
            case.token_count, NUM_KV_HEADS, HEAD_DIM
        )
        direct_key = relocate_neox_rope(offline_key, case.shift, theta)
        full_key = online_kv[0].reshape(
            case.token_count, NUM_KV_HEADS, HEAD_DIM
        )
        direct_value = offline_kv[1].reshape(
            case.token_count, NUM_KV_HEADS, HEAD_DIM
        )
        full_value = online_kv[1].reshape(
            case.token_count, NUM_KV_HEADS, HEAD_DIM
        )
        for component, direct, full in (
            ("K", direct_key, full_key),
            ("V", direct_value, full_value),
        ):
            mean, median, sse, full_sq_norm, normalized_l2 = fidelity_summary(
                direct, full
            )
            rows.append(
                {
                    "case_id": case.case_id,
                    "skill": case.skill,
                    "token_count": case.token_count,
                    "component": component,
                    "layer": layer,
                    "direct_to_full_cosine_mean": mean,
                    "direct_to_full_cosine_median": median,
                    "direct_to_full_sse": sse,
                    "full_sq_norm": full_sq_norm,
                    "direct_to_full_normalized_l2": normalized_l2,
                }
            )
    return rows


def plot(rows: list[dict[str, Any]], output_dir: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    skills = list(dict.fromkeys(str(row["skill"]) for row in rows))
    fig, axes = plt.subplots(1, 2, figsize=(7, 2), constrained_layout=True)
    for axis, component, ylabel in zip(
        axes,
        ("K", "V"),
        ("Key cosine to Recompute", "Value cosine to Recompute"),
        strict=True,
    ):
        component_values = []
        for index, skill in enumerate(skills):
            skill_rows = sorted(
                (
                    row
                    for row in rows
                    if row["component"] == component and row["skill"] == skill
                ),
                key=lambda row: int(row["layer"]),
            )
            layers = [int(row["layer"]) + 1 for row in skill_rows]
            values = [float(row["direct_to_full_cosine_mean"]) for row in skill_rows]
            component_values.extend(values)
            axis.plot(
                layers,
                values,
                color=COLORS[index],
                linestyle=LINE_STYLES[index],
                linewidth=1.6,
                label=skill,
            )
        value_min = min(component_values)
        value_max = max(component_values)
        padding = max((value_max - value_min) * 0.08, 0.002)
        axis.set_ylim(value_min - padding, min(1.002, value_max + padding))
        axis.set_xlim(1, NUM_LAYERS)
        axis.set_xticks([1, 10, 20, 30, 40])
        axis.set_xlabel("Model layer")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", color="#D0D0D0", linewidth=0.5, alpha=0.65)
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(1.0)
    axes[1].legend(frameon=False, loc="best")
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_dir / "direct_full_layerwise_cosine.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.1,
    )
    fig.savefig(
        output_dir / "direct_full_layerwise_cosine.pdf",
        bbox_inches="tight",
        pad_inches=0.03,
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--pool-dir", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rope-theta", type=float, default=1_000_000.0)
    args = parser.parse_args()
    case_specs = json.loads(args.cases.read_text(encoding="utf-8"))["cases"]
    cases = [validate_case(case, args.run_dir, args.pool_dir) for case in case_specs]
    rows: list[dict[str, Any]] = []
    for case in cases:
        rows.extend(analyze_case(case, args.rope_theta))
    write_csv(args.output_csv, rows)
    plot(rows, args.output_dir)
    print(
        f"[direct-full-cosine] cases={len(cases)} rows={len(rows)} "
        f"output={args.output_dir}"
    )


if __name__ == "__main__":
    main()
