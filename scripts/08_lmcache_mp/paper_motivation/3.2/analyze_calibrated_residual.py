#!/usr/bin/env python3
"""Evaluate prefix-direction correction with held-out scale calibration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from analyze_context_free_residual import (
    layer_sidecars,
    read_layer,
    relocate_neox_rope,
    write_csv,
)


def estimate_calibrated_offset(
    residual: torch.Tensor,
    direction_end: int = 128,
    calibration_end: int = 256,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Estimate one constant offset per KV head without reading the suffix."""
    if not 0 < direction_end < calibration_end <= residual.shape[0]:
        raise ValueError("expected 0 < direction_end < calibration_end <= token count")
    direction = residual[:direction_end].mean(dim=0)
    calibration_mean = residual[direction_end:calibration_end].mean(dim=0)
    denominator = direction.square().sum(dim=-1)
    numerator = (direction * calibration_mean).sum(dim=-1)
    alpha = torch.where(
        denominator > torch.finfo(direction.dtype).eps,
        numerator / denominator,
        torch.zeros_like(numerator),
    )
    offset = alpha.unsqueeze(-1) * direction
    return offset, alpha, direction, calibration_mean


def fidelity_metrics(approximation: torch.Tensor, full: torch.Tensor) -> dict[str, float]:
    """Measure vector direction and scale-aware error against Full."""
    approximation = approximation.to(torch.float32)
    full = full.to(torch.float32)
    difference = approximation - full
    squared_error = difference.square().sum()
    full_squared_norm = full.square().sum()
    dot = (approximation * full).sum(dim=-1)
    denominator = torch.sqrt(
        approximation.square().sum(dim=-1) * full.square().sum(dim=-1)
    ).clamp_min(torch.finfo(torch.float32).eps)
    cosine = dot / denominator
    return {
        "sse": float(squared_error),
        "full_sq_norm": float(full_squared_norm),
        "normalized_l2": float(
            torch.sqrt(squared_error / full_squared_norm.clamp_min(1e-30))
        ),
        "cosine_mean": float(cosine.mean()),
        "cosine_median": float(cosine.median()),
    }


def analyze_case(
    case_id: str,
    skill: str,
    offline_dir: Path,
    online_dir: Path,
    capture_path: Path,
    direction_end: int,
    calibration_end: int,
    theta: float,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    capture = json.loads(capture_path.read_text(encoding="utf-8"))
    offline_manifest = json.loads(
        (offline_dir.parent / "manifest.json").read_text(encoding="utf-8")
    )
    token_count = int(capture["token_count"])
    if not (
        capture.get("status") == "completed"
        and capture.get("skill") == skill
        and offline_manifest.get("schema_version") == 3
        and offline_manifest.get("status") == "completed"
        and offline_manifest.get("skill_name") == skill
        and offline_manifest.get("token_count") == token_count
        and offline_manifest.get("token_ids_sha256")
        == capture.get("token_ids_sha256")
    ):
        raise ValueError(f"offline and online records disagree for Skill {skill}")
    if token_count <= calibration_end:
        raise ValueError(
            f"Skill {skill} has {token_count} tokens; no suffix after {calibration_end}"
        )

    shift = int(capture["segment_start"])
    offline_layers = layer_sidecars(offline_dir)
    online_layers = layer_sidecars(online_dir)
    layer_rows: list[dict[str, Any]] = []
    head_rows: list[dict[str, Any]] = []
    fidelity_rows: list[dict[str, Any]] = []
    value_head_rows: list[dict[str, Any]] = []

    for layer in range(40):
        offline_kv, offline_meta = read_layer(offline_layers[layer])
        online_kv, online_meta = read_layer(online_layers[layer])
        if offline_kv.shape != online_kv.shape or offline_kv.shape[1] != token_count:
            raise ValueError(f"KV shape mismatch for {case_id} layer={layer}")
        if (
            offline_meta.get("cached_positions", {}).get("start") != 0
            or online_meta.get("cached_positions", {}).get("start") != shift
        ):
            raise ValueError(f"cached position mismatch for {case_id} layer={layer}")

        offline_key = offline_kv[0].reshape(token_count, 8, 128)
        online_key = online_kv[0].reshape(token_count, 8, 128).to(torch.float32)
        aligned = relocate_neox_rope(offline_key, shift, theta)
        residual = online_key - aligned

        calibrated, alpha, direction, calibration_mean = estimate_calibrated_offset(
            residual, direction_end, calibration_end
        )
        unit_256 = residual[:calibration_end].mean(dim=0)
        direct_tail = aligned[calibration_end:]
        target_tail = online_key[calibration_end:]
        direct_sse = float((direct_tail - target_tail).square().sum())
        unit_sse = float(
            (direct_tail + unit_256.unsqueeze(0) - target_tail).square().sum()
        )
        calibrated_sse = float(
            (direct_tail + calibrated.unsqueeze(0) - target_tail).square().sum()
        )
        layer_rows.append(
            {
                "case_id": case_id,
                "skill": skill,
                "token_count": token_count,
                "layer": layer,
                "direction_start": 0,
                "direction_end": direction_end,
                "calibration_start": direction_end,
                "calibration_end": calibration_end,
                "evaluation_start": calibration_end,
                "direct_sse": direct_sse,
                "unit_256_sse": unit_sse,
                "calibrated_sse": calibrated_sse,
            }
        )
        for head in range(8):
            head_rows.append(
                {
                    "case_id": case_id,
                    "skill": skill,
                    "layer": layer,
                    "head": head,
                    "alpha": float(alpha[head]),
                    "direction_norm": float(torch.linalg.vector_norm(direction[head])),
                    "calibration_mean_norm": float(
                        torch.linalg.vector_norm(calibration_mean[head])
                    ),
                }
            )

        offline_value = offline_kv[1].reshape(token_count, 8, 128).to(torch.float32)
        online_value = online_kv[1].reshape(token_count, 8, 128).to(torch.float32)
        value_residual = online_value - offline_value
        value_calibrated, value_alpha, value_direction, value_calibration_mean = (
            estimate_calibrated_offset(
                value_residual, direction_end, calibration_end
            )
        )
        value_unit_256 = value_residual[:calibration_end].mean(dim=0)

        component_inputs = {
            "K": (aligned, online_key, unit_256, calibrated),
            "V": (
                offline_value,
                online_value,
                value_unit_256,
                value_calibrated,
            ),
        }
        for component, (direct, full, unit_offset, calibrated_offset) in component_inputs.items():
            direct_tail_component = direct[calibration_end:]
            full_tail_component = full[calibration_end:]
            unit_tail_component = direct_tail_component + unit_offset.unsqueeze(0)
            calibrated_tail_component = (
                direct_tail_component + calibrated_offset.unsqueeze(0)
            )
            direct_metrics = fidelity_metrics(
                direct_tail_component, full_tail_component
            )
            unit_metrics = fidelity_metrics(unit_tail_component, full_tail_component)
            calibrated_metrics = fidelity_metrics(
                calibrated_tail_component, full_tail_component
            )
            fidelity_rows.append(
                {
                    "case_id": case_id,
                    "skill": skill,
                    "token_count": token_count,
                    "layer": layer,
                    "component": component,
                    "evaluation_start": calibration_end,
                    "full_sq_norm": direct_metrics["full_sq_norm"],
                    "direct_to_full_sse": direct_metrics["sse"],
                    "unit_to_full_sse": unit_metrics["sse"],
                    "calibrated_to_full_sse": calibrated_metrics["sse"],
                    "direct_to_full_normalized_l2": direct_metrics["normalized_l2"],
                    "unit_to_full_normalized_l2": unit_metrics["normalized_l2"],
                    "calibrated_to_full_normalized_l2": calibrated_metrics["normalized_l2"],
                    "direct_to_full_cosine_mean": direct_metrics["cosine_mean"],
                    "unit_to_full_cosine_mean": unit_metrics["cosine_mean"],
                    "calibrated_to_full_cosine_mean": calibrated_metrics["cosine_mean"],
                    "direct_to_full_cosine_median": direct_metrics["cosine_median"],
                    "unit_to_full_cosine_median": unit_metrics["cosine_median"],
                    "calibrated_to_full_cosine_median": calibrated_metrics["cosine_median"],
                }
            )
        for head in range(8):
            value_head_rows.append(
                {
                    "case_id": case_id,
                    "skill": skill,
                    "layer": layer,
                    "head": head,
                    "alpha": float(value_alpha[head]),
                    "direction_norm": float(
                        torch.linalg.vector_norm(value_direction[head])
                    ),
                    "calibration_mean_norm": float(
                        torch.linalg.vector_norm(value_calibration_mean[head])
                    ),
                }
            )
    return layer_rows, head_rows, fidelity_rows, value_head_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--pool-dir", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--direction-end", type=int, default=128)
    parser.add_argument("--calibration-end", type=int, default=256)
    parser.add_argument("--rope-theta", type=float, default=1_000_000.0)
    args = parser.parse_args()

    cases = json.loads(args.cases.read_text(encoding="utf-8"))["cases"]
    all_layers: list[dict[str, Any]] = []
    all_heads: list[dict[str, Any]] = []
    all_fidelity: list[dict[str, Any]] = []
    all_value_heads: list[dict[str, Any]] = []
    for case in cases:
        case_dir = args.run_dir / case["case_id"]
        layers, heads, fidelity, value_heads = analyze_case(
            case["case_id"],
            case["skill"],
            args.pool_dir / case["skill"] / "kv",
            case_dir / "online_full_kv",
            case_dir / "capture.json",
            args.direction_end,
            args.calibration_end,
            args.rope_theta,
        )
        all_layers.extend(layers)
        all_heads.extend(heads)
        all_fidelity.extend(fidelity)
        all_value_heads.extend(value_heads)
    write_csv(args.output_dir / "calibrated_layer_metrics.csv", all_layers)
    write_csv(args.output_dir / "calibration_heads.csv", all_heads)
    write_csv(args.output_dir / "fidelity_layer_metrics.csv", all_fidelity)
    write_csv(args.output_dir / "value_calibration_heads.csv", all_value_heads)
    print(f"[calibrated] cases={len(cases)} output={args.output_dir}")


if __name__ == "__main__":
    main()
