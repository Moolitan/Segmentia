#!/usr/bin/env python3
"""Ablate calibration location and length on one captured Skill KV pair.

The deployable sweep assumes Segmentia locally computes a contiguous Skill
prefix [0, P) and restores [P, E).  Offsets are estimated from different
subsets of that already-computed prefix and evaluated only on [P, E).
Middle/random windows are diagnostic controls: they test representativeness,
but would require non-prefix recomputation in an online system.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from analyze_single_case import (
    DEFAULT_MODEL_CONFIG,
    atomic_write_json,
    decode_positions,
    load_sidecar,
    read_raw_kv,
    relocate_neox_rope,
    require_populated_raw_kv,
    tensor_metrics,
    validate_case,
    write_csv,
)


DEFAULT_BUDGETS = (8, 16, 32, 64, 128, 256)
DEFAULT_RANDOM_SEED = 20260728


def deployable_windows(
    budget: int, header_tokens: int
) -> list[tuple[str, int, int]]:
    """Return calibration windows contained in the local prefix [0, budget)."""

    if budget <= 0:
        raise ValueError("budget must be positive")
    windows = [("full_prefix", 0, budget)]
    if budget > header_tokens:
        windows.append(("body_prefix", header_tokens, budget))
        body_midpoint = header_tokens + (budget - header_tokens) // 2
        windows.append(("tail_body", body_midpoint, budget))
    return windows


def diagnostic_windows(
    tokens: int,
    calibration_tokens: int,
    header_tokens: int,
    random_seed: int,
) -> list[tuple[str, int, int]]:
    """Return fixed-length location controls over the whole Skill."""

    if not 0 < calibration_tokens < tokens:
        raise ValueError("calibration_tokens must be within the Skill")
    maximum_start = tokens - calibration_tokens
    body_start = min(header_tokens, maximum_start)
    middle_start = maximum_start // 2
    generator = random.Random(random_seed + calibration_tokens)
    random_start = generator.randint(0, maximum_start)
    return [
        ("natural_start", 0, calibration_tokens),
        ("body_start", body_start, body_start + calibration_tokens),
        ("middle", middle_start, middle_start + calibration_tokens),
        ("random_fixed", random_start, random_start + calibration_tokens),
    ]


def offset_predictions_for_window(
    source: torch.Tensor,
    target: torch.Tensor,
    calibration_start: int,
    calibration_end: int,
    evaluation_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if not 0 <= calibration_start < calibration_end <= source.shape[0]:
        raise ValueError("Invalid calibration window")
    if evaluation_mask.dtype != torch.bool or evaluation_mask.shape != (
        source.shape[0],
    ):
        raise ValueError("evaluation_mask must be a boolean token mask")
    if not bool(evaluation_mask.any()):
        raise ValueError("Evaluation set cannot be empty")
    residual = (
        target[calibration_start:calibration_end]
        - source[calibration_start:calibration_end]
    )
    shared_offset = residual.mean(dim=(0, 1), keepdim=True)
    headwise_offset = residual.mean(dim=0, keepdim=True)
    evaluation_source = source[evaluation_mask]
    evaluation_target = target[evaluation_mask]
    return evaluation_target, {
        "direct": evaluation_source,
        "layer_shared_offset": evaluation_source + shared_offset,
        "headwise_offset": evaluation_source + headwise_offset,
    }


def _append_metrics(
    rows: list[dict[str, Any]],
    common: dict[str, Any],
    predictions: dict[str, torch.Tensor],
    target: torch.Tensor,
) -> None:
    direct = tensor_metrics(predictions["direct"], target)
    for baseline, prediction in predictions.items():
        metrics = tensor_metrics(prediction, target)
        rows.append(
            {
                **common,
                "baseline": baseline,
                "squared_error": metrics["squared_error"],
                "direct_squared_error": direct["squared_error"],
                "target_squared_norm": metrics["target_squared_norm"],
                "relative_l2": metrics["relative_l2"],
                "rmse": metrics["rmse"],
                "cosine": metrics["cosine"],
            }
        )


def summarize_rows(
    rows: list[dict[str, Any]], group_fields: tuple[str, ...]
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in group_fields)].append(row)
    summaries: list[dict[str, Any]] = []
    for key, values in sorted(groups.items(), key=lambda item: item[0]):
        summaries.append(
            {
                **dict(zip(group_fields, key, strict=True)),
                "layers": len(values),
                "macro_relative_l2": sum(row["relative_l2"] for row in values)
                / len(values),
                "macro_rmse": sum(row["rmse"] for row in values) / len(values),
                "macro_cosine": sum(row["cosine"] for row in values) / len(values),
                "aggregate_improvement_vs_direct": 1.0
                - sum(row["squared_error"] for row in values)
                / max(sum(row["direct_squared_error"] for row in values), 1e-30),
                "improved_layers": sum(
                    row["squared_error"] < row["direct_squared_error"]
                    for row in values
                ),
            }
        )
    return summaries


def make_plots(
    output_dir: Path,
    deployable_summary: list[dict[str, Any]],
    diagnostic_summary: list[dict[str, Any]],
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True, sharey=True)
    for axis, kv_type in zip(axes, ("K", "V"), strict=True):
        subset = [
            row
            for row in deployable_summary
            if row["kv_type"] == kv_type and row["baseline"] != "direct"
        ]
        for estimator in ("full_prefix", "body_prefix", "tail_body"):
            for baseline, linestyle in (
                ("layer_shared_offset", "-"),
                ("headwise_offset", "--"),
            ):
                values = [
                    row
                    for row in subset
                    if row["estimator"] == estimator
                    and row["baseline"] == baseline
                ]
                if not values:
                    continue
                axis.plot(
                    [row["compute_budget"] for row in values],
                    [
                        100.0 * row["aggregate_improvement_vs_direct"]
                        for row in values
                    ],
                    marker="o",
                    linewidth=1.2,
                    linestyle=linestyle,
                    label=f"{estimator}/{baseline.replace('_offset', '')}",
                )
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_xscale("log", base=2)
        axis.set_yscale("symlog", linthresh=5.0)
        axis.set_title(kv_type)
        axis.set_xlabel("Locally computed prefix tokens")
        axis.set_ylabel("Squared-error improvement vs Direct (%)")
        axis.grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=7, ncol=2)
    fig.suptitle("Deployable prefix calibration sweep")
    fig.tight_layout()
    path = figures_dir / "deployable_calibration_sweep.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path.resolve()))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True, sharey=True)
    for axis, kv_type in zip(axes, ("K", "V"), strict=True):
        for location in ("natural_start", "body_start", "middle", "random_fixed"):
            values = [
                row
                for row in diagnostic_summary
                if row["kv_type"] == kv_type
                and row["location"] == location
                and row["baseline"] == "headwise_offset"
            ]
            axis.plot(
                [row["calibration_tokens"] for row in values],
                [
                    100.0 * row["aggregate_improvement_vs_direct"]
                    for row in values
                ],
                marker="o",
                linewidth=1.2,
                label=location,
            )
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_xscale("log", base=2)
        axis.set_yscale("symlog", linthresh=5.0)
        axis.set_title(kv_type)
        axis.set_xlabel("Calibration tokens")
        axis.set_ylabel("Squared-error improvement vs Direct (%)")
        axis.grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Diagnostic calibration-location controls (headwise)")
    fig.tight_layout()
    path = figures_dir / "diagnostic_location_controls.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path.resolve()))
    return paths


def analyze_calibration_ablation(
    case_dir: Path,
    model_config_path: Path,
    output_dir: Path,
    budgets: tuple[int, ...] = DEFAULT_BUDGETS,
    random_seed: int = DEFAULT_RANDOM_SEED,
    plots: bool = True,
) -> dict[str, Any]:
    manifest_path = case_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    model_config = json.loads(model_config_path.read_text(encoding="utf-8"))
    source_files, target_files, tokens, kv_heads, header_tokens = validate_case(
        manifest, model_config
    )
    budgets = tuple(sorted(set(budgets)))
    if not budgets or budgets[0] < header_tokens or budgets[-1] >= tokens:
        raise ValueError(
            f"Budgets must be unique values in [{header_tokens}, {tokens})"
        )
    head_dim = int(model_config["head_dim"])
    rope_theta = float(model_config["rope_theta"])
    deployable_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []

    for source_path, target_path in zip(source_files, target_files, strict=True):
        layer = int(source_path.stem.rsplit("@", 1)[1])
        source_meta = load_sidecar(source_path)
        target_meta = load_sidecar(target_path)
        require_populated_raw_kv(source_path)
        require_populated_raw_kv(target_path)
        source_shape = source_meta["shape"]
        target_shape = target_meta["shape"]
        expected_shape = [2, tokens, kv_heads * head_dim]
        if source_shape != expected_shape or target_shape != expected_shape:
            raise ValueError(f"Layer {layer} shape mismatch")
        source_positions = decode_positions(source_meta["cached_positions"], tokens)
        target_positions = decode_positions(target_meta["cached_positions"], tokens)
        source_raw = read_raw_kv(source_path, source_shape)
        target_raw = read_raw_kv(target_path, target_shape)
        sources = {
            "K": relocate_neox_rope(
                source_raw[0].reshape(tokens, kv_heads, head_dim),
                source_positions,
                target_positions,
                rope_theta,
            ),
            "V": source_raw[1]
            .reshape(tokens, kv_heads, head_dim)
            .to(torch.float32),
        }
        targets = {
            "K": target_raw[0]
            .reshape(tokens, kv_heads, head_dim)
            .to(torch.float32),
            "V": target_raw[1]
            .reshape(tokens, kv_heads, head_dim)
            .to(torch.float32),
        }

        for kv_type in ("K", "V"):
            source = sources[kv_type]
            target = targets[kv_type]
            for budget in budgets:
                evaluation_mask = torch.arange(tokens) >= budget
                for estimator, start, end in deployable_windows(
                    budget, header_tokens
                ):
                    evaluation_target, predictions = offset_predictions_for_window(
                        source, target, start, end, evaluation_mask
                    )
                    _append_metrics(
                        deployable_rows,
                        {
                            "layer": layer,
                            "kv_type": kv_type,
                            "compute_budget": budget,
                            "estimator": estimator,
                            "calibration_start": start,
                            "calibration_end": end,
                            "calibration_tokens": end - start,
                            "evaluation_start": budget,
                            "evaluation_end": tokens,
                            "evaluation_tokens": tokens - budget,
                        },
                        predictions,
                        evaluation_target,
                    )

                for location, start, end in diagnostic_windows(
                    tokens, budget, header_tokens, random_seed
                ):
                    evaluation_mask = torch.ones(tokens, dtype=torch.bool)
                    evaluation_mask[start:end] = False
                    evaluation_target, predictions = offset_predictions_for_window(
                        source, target, start, end, evaluation_mask
                    )
                    _append_metrics(
                        diagnostic_rows,
                        {
                            "layer": layer,
                            "kv_type": kv_type,
                            "calibration_tokens": budget,
                            "location": location,
                            "calibration_start": start,
                            "calibration_end": end,
                            "evaluation_policy": "all_tokens_except_calibration",
                            "evaluation_tokens": tokens - budget,
                        },
                        predictions,
                        evaluation_target,
                    )

    deployable_group_fields = (
        "kv_type",
        "compute_budget",
        "estimator",
        "calibration_start",
        "calibration_end",
        "calibration_tokens",
        "evaluation_start",
        "evaluation_end",
        "evaluation_tokens",
        "baseline",
    )
    diagnostic_group_fields = (
        "kv_type",
        "calibration_tokens",
        "location",
        "calibration_start",
        "calibration_end",
        "evaluation_policy",
        "evaluation_tokens",
        "baseline",
    )
    deployable_summary = summarize_rows(deployable_rows, deployable_group_fields)
    diagnostic_summary = summarize_rows(diagnostic_rows, diagnostic_group_fields)
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = output_dir / "tables"
    write_csv(tables_dir / "deployable_layer_metrics.csv", deployable_rows)
    write_csv(tables_dir / "deployable_summary.csv", deployable_summary)
    write_csv(tables_dir / "diagnostic_layer_metrics.csv", diagnostic_rows)
    write_csv(tables_dir / "diagnostic_summary.csv", diagnostic_summary)
    figure_paths = (
        make_plots(output_dir, deployable_summary, diagnostic_summary)
        if plots
        else []
    )

    corrected_deployable = [
        row for row in deployable_summary if row["baseline"] != "direct"
    ]
    best_by_kv: dict[str, dict[str, Any]] = {}
    for kv_type in ("K", "V"):
        candidates = [
            row for row in corrected_deployable if row["kv_type"] == kv_type
        ]
        best_by_kv[kv_type] = max(
            candidates, key=lambda row: row["aggregate_improvement_vs_direct"]
        )
    summary = {
        "schema_version": 1,
        "status": "descriptive_single_case_calibration_ablation",
        "case_id": manifest["case_id"],
        "skill": manifest["skill"],
        "tokens": tokens,
        "layers": len(source_files),
        "header_tokens": header_tokens,
        "budgets": list(budgets),
        "random_seed": random_seed,
        "deployable_evaluation": "contiguous suffix [compute_budget, skill_end)",
        "diagnostic_evaluation": "all Skill tokens except calibration window",
        "best_deployable_correction_by_kv": best_by_kv,
        "interpretation_limits": [
            "One source-target pair cannot establish cross-request generalization.",
            "Layer values are correlated structure, not independent replicates.",
            "Middle and random calibration windows are diagnostic controls, not "
            "directly deployable prefix-only policies.",
            "Uniform offsets test boundary representativeness; this ablation does "
            "not train or evaluate a low-rank predictor.",
        ],
        "artifacts": {
            "manifest": str(manifest_path.resolve()),
            "tables": [
                str((tables_dir / name).resolve())
                for name in (
                    "deployable_layer_metrics.csv",
                    "deployable_summary.csv",
                    "diagnostic_layer_metrics.csv",
                    "diagnostic_summary.csv",
                )
            ],
            "figures": figure_paths,
        },
    }
    atomic_write_json(output_dir / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--budgets", type=int, nargs="+", default=list(DEFAULT_BUDGETS))
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.case_dir / "analysis/calibration_ablation"
    summary = analyze_calibration_ablation(
        case_dir=args.case_dir,
        model_config_path=args.model_config,
        output_dir=output_dir,
        budgets=tuple(args.budgets),
        random_seed=args.random_seed,
        plots=not args.no_plots,
    )
    print(
        f"[analyzed] case={summary['case_id']} budgets={summary['budgets']} "
        f"output={output_dir}"
    )


if __name__ == "__main__":
    main()
