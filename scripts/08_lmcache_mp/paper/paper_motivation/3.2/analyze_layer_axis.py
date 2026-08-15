#!/usr/bin/env python3
"""Evaluate self-only shallow-layer residual correction for deep Skill KV."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cskcache import SOURCE_ARTIFACT_TYPE

import torch

from analyze_context_free_residual import (
    layer_sidecars,
    read_layer,
    relocate_neox_rope,
    write_csv,
)


NUM_LAYERS = 40
NUM_KV_HEADS = 8
HEAD_DIM = 128


@dataclass(frozen=True)
class CaseRecord:
    case_id: str
    skill: str
    token_count: int
    shift: int
    offline_layers: dict[int, Path]
    online_layers: dict[int, Path]


def validate_case(
    case: dict[str, Any], run_dir: Path, pool_dir: Path
) -> CaseRecord:
    case_id = str(case["case_id"])
    skill = str(case["skill"])
    capture = json.loads(
        (run_dir / case_id / "capture.json").read_text(encoding="utf-8")
    )
    offline_root = pool_dir / skill
    offline_manifest = json.loads(
        (offline_root / "manifest.json").read_text(encoding="utf-8")
    )
    token_count = int(capture["token_count"])
    if not (
        capture.get("status") == "completed"
        and capture.get("skill") == skill
        and offline_manifest.get("artifact_type") == SOURCE_ARTIFACT_TYPE
        and offline_manifest.get("status") == "completed"
        and offline_manifest.get("skill_name") == skill
        and offline_manifest.get("token_count") == token_count
        and offline_manifest.get("token_ids_sha256")
        == capture.get("token_ids_sha256")
    ):
        raise ValueError(f"offline and online records disagree for Skill {skill}")
    return CaseRecord(
        case_id=case_id,
        skill=skill,
        token_count=token_count,
        shift=int(capture["segment_start"]),
        offline_layers=layer_sidecars(offline_root / "kv"),
        online_layers=layer_sidecars(run_dir / case_id / "online_full_kv"),
    )


def load_component(
    case: CaseRecord,
    layer: int,
    component: str,
    theta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return Direct, Recompute and residual in [token, head, dim]."""
    offline_kv, offline_meta = read_layer(case.offline_layers[layer])
    online_kv, online_meta = read_layer(case.online_layers[layer])
    expected_shape = (2, case.token_count, NUM_KV_HEADS * HEAD_DIM)
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
    index = 0 if component == "K" else 1
    direct = offline_kv[index].reshape(
        case.token_count, NUM_KV_HEADS, HEAD_DIM
    )
    if component == "K":
        direct = relocate_neox_rope(direct, case.shift, theta)
    else:
        direct = direct.to(torch.float32)
    recompute = online_kv[index].reshape(
        case.token_count, NUM_KV_HEADS, HEAD_DIM
    ).to(torch.float32)
    return direct, recompute, recompute - direct


def estimate_self_shallow_offset(
    shallow_residuals: torch.Tensor,
) -> torch.Tensor:
    """Average one Skill's observed [layer, token, head, dim] residuals."""
    if shallow_residuals.ndim != 4:
        raise ValueError(
            "shallow residuals must have [layer, token, head, dim] shape"
        )
    if shallow_residuals.shape[0] <= 0 or shallow_residuals.shape[1] <= 0:
        raise ValueError("shallow residuals cannot be empty")
    if tuple(shallow_residuals.shape[2:]) != (NUM_KV_HEADS, HEAD_DIM):
        raise ValueError(
            "shallow residual head shape must be "
            f"[{NUM_KV_HEADS}, {HEAD_DIM}]"
        )
    return shallow_residuals.to(torch.float32).mean(dim=(0, 1))


def fidelity(
    approximation: torch.Tensor, recompute: torch.Tensor
) -> dict[str, float]:
    approximation = approximation.to(torch.float32)
    recompute = recompute.to(torch.float32)
    difference = approximation - recompute
    sse = difference.square().sum()
    recompute_sq_norm = recompute.square().sum()
    dot = (approximation * recompute).sum(dim=-1)
    denominator = torch.sqrt(
        approximation.square().sum(dim=-1)
        * recompute.square().sum(dim=-1)
    ).clamp_min(torch.finfo(torch.float32).eps)
    return {
        "sse": float(sse),
        "recompute_sq_norm": float(recompute_sq_norm),
        "cosine_mean": float((dot / denominator).mean()),
    }


def analyze_component(
    case: CaseRecord,
    component: str,
    cutoffs: list[int],
    theta: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    layer_data = [
        load_component(case, layer, component, theta)
        for layer in range(NUM_LAYERS)
    ]
    rows: list[dict[str, Any]] = []
    offset_rows: list[dict[str, Any]] = []
    for cutoff in cutoffs:
        shallow_residuals = torch.stack(
            [layer_data[layer][2] for layer in range(cutoff)], dim=0
        )
        offset = estimate_self_shallow_offset(shallow_residuals)
        for head in range(NUM_KV_HEADS):
            offset_rows.append(
                {
                    "case_id": case.case_id,
                    "skill": case.skill,
                    "component": component,
                    "cutoff": cutoff,
                    "head": head,
                    "shallow_layer_count": cutoff,
                    "shallow_token_count": case.token_count,
                    "offset_norm": float(torch.linalg.vector_norm(offset[head])),
                    "offset_abs_mean": float(offset[head].abs().mean()),
                }
            )
        for target_layer in range(cutoff, NUM_LAYERS):
            direct, recompute, _ = layer_data[target_layer]
            corrected = direct + offset.unsqueeze(0)
            direct_metrics = fidelity(direct, recompute)
            corrected_metrics = fidelity(corrected, recompute)
            rows.append(
                {
                    "case_id": case.case_id,
                    "skill": case.skill,
                    "token_count": case.token_count,
                    "component": component,
                    "cutoff": cutoff,
                    "target_layer": target_layer,
                    "estimator": "self_shallow_layer_token_mean_per_head",
                    "estimation_skill_count": 1,
                    "shallow_layer_count": cutoff,
                    "shallow_token_count": case.token_count,
                    "direct_to_recompute_sse": direct_metrics["sse"],
                    "corrected_to_recompute_sse": corrected_metrics["sse"],
                    "recompute_sq_norm": direct_metrics["recompute_sq_norm"],
                    "direct_to_recompute_cosine_mean": direct_metrics[
                        "cosine_mean"
                    ],
                    "corrected_to_recompute_cosine_mean": corrected_metrics[
                        "cosine_mean"
                    ],
                }
            )
    return rows, offset_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--pool-dir", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cutoffs", default="4,8")
    parser.add_argument("--rope-theta", type=float, default=1_000_000.0)
    args = parser.parse_args()
    cutoffs = [int(value) for value in args.cutoffs.split(",")]
    if not cutoffs or any(value <= 0 or value >= NUM_LAYERS for value in cutoffs):
        raise ValueError("cutoffs must be inside [1, 39]")
    if cutoffs != sorted(set(cutoffs)):
        raise ValueError("cutoffs must be unique and sorted")

    case_specs = json.loads(args.cases.read_text(encoding="utf-8"))["cases"]
    cases = [validate_case(case, args.run_dir, args.pool_dir) for case in case_specs]
    if not cases:
        raise ValueError("at least one Skill case is required")

    fidelity_rows: list[dict[str, Any]] = []
    offset_rows: list[dict[str, Any]] = []
    for case in cases:
        for component in ("K", "V"):
            component_rows, component_offsets = analyze_component(
                case, component, cutoffs, args.rope_theta
            )
            fidelity_rows.extend(component_rows)
            offset_rows.extend(component_offsets)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "layer_axis_fidelity.csv", fidelity_rows)
    write_csv(args.output_dir / "layer_axis_parameters.csv", offset_rows)
    metadata = {
        "schema_version": 2,
        "method": "per-Skill self-only shallow layer/token mean residual offset",
        "components": ["K", "V"],
        "cutoffs": cutoffs,
        "estimation_skill_count_per_case": 1,
        "shallow_estimator": "mean over this Skill's layers [0, cutoff) and tokens",
        "deep_truth_used_for_estimation": False,
        "other_skills_used_for_estimation": False,
        "num_layers": NUM_LAYERS,
        "num_kv_heads": NUM_KV_HEADS,
        "head_dim": HEAD_DIM,
        "cases": [case.case_id for case in cases],
    }
    (args.output_dir / "layer_axis_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"[self-layer-axis] cases={len(cases)} cutoffs={cutoffs} "
        f"output={args.output_dir}"
    )


if __name__ == "__main__":
    main()
