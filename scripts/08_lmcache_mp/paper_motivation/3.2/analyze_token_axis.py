#!/usr/bin/env python3
"""Evaluate per-Skill, per-layer token-prefix KV correction."""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
CORRECTION_ALPHA = 0.6


@dataclass(frozen=True)
class CaseRecord:
    case_id: str
    skill: str
    token_count: int
    shift: int
    offline_layers: dict[int, Path]
    recompute_layers: dict[int, Path]


def validate_case(
    spec: dict[str, Any], run_dir: Path, pool_dir: Path, evaluation_start: int
) -> CaseRecord:
    case_id = str(spec["case_id"])
    skill = str(spec["skill"])
    capture = json.loads(
        (run_dir / case_id / "capture.json").read_text(encoding="utf-8")
    )
    offline_root = pool_dir / skill
    manifest = json.loads(
        (offline_root / "manifest.json").read_text(encoding="utf-8")
    )
    token_count = int(capture["token_count"])
    if not (
        capture.get("status") == "completed"
        and capture.get("skill") == skill
        and manifest.get("schema_version") in {3, 4}
        and manifest.get("status") == "completed"
        and manifest.get("skill_name") == skill
        and manifest.get("token_count") == token_count
        and manifest.get("token_ids_sha256") == capture.get("token_ids_sha256")
    ):
        raise ValueError(f"offline and Recompute records disagree for Skill {skill}")
    if token_count <= evaluation_start:
        raise ValueError(
            f"Skill {skill} has {token_count} tokens; no suffix after "
            f"{evaluation_start}"
        )
    return CaseRecord(
        case_id=case_id,
        skill=skill,
        token_count=token_count,
        shift=int(capture["segment_start"]),
        offline_layers=layer_sidecars(offline_root / "kv"),
        recompute_layers=layer_sidecars(run_dir / case_id / "online_full_kv"),
    )


def estimate_token_offset(
    residual: torch.Tensor, observation_start: int, observation_end: int
) -> torch.Tensor:
    """Return one offset per KV head from this layer's observed tokens only."""
    if residual.ndim != 3 or tuple(residual.shape[1:]) != (
        NUM_KV_HEADS,
        HEAD_DIM,
    ):
        raise ValueError("residual must have [token, 8, 128] shape")
    if not 0 <= observation_start < observation_end <= residual.shape[0]:
        raise ValueError("invalid token observation window")
    return residual[observation_start:observation_end].to(torch.float32).mean(dim=0)


def apply_token_offset(
    direct_suffix: torch.Tensor, offset: torch.Tensor, alpha: float
) -> torch.Tensor:
    """Apply the fixed-scale, per-head offset to an unobserved token suffix."""
    if direct_suffix.ndim != 3 or tuple(direct_suffix.shape[1:]) != (
        NUM_KV_HEADS,
        HEAD_DIM,
    ):
        raise ValueError("direct_suffix must have [token, 8, 128] shape")
    if tuple(offset.shape) != (NUM_KV_HEADS, HEAD_DIM):
        raise ValueError("offset must have [8, 128] shape")
    if not math.isfinite(alpha) or alpha < 0.0:
        raise ValueError("alpha must be a finite non-negative scalar")
    return direct_suffix + alpha * offset.unsqueeze(0)


def fidelity(
    approximation: torch.Tensor, recompute: torch.Tensor
) -> dict[str, float]:
    approximation = approximation.to(torch.float32)
    recompute = recompute.to(torch.float32)
    difference = approximation - recompute
    sse = difference.square().sum()
    recompute_sq_norm = recompute.square().sum()
    cosine = (approximation * recompute).sum(dim=-1) / torch.sqrt(
        approximation.square().sum(dim=-1)
        * recompute.square().sum(dim=-1)
    ).clamp_min(torch.finfo(torch.float32).eps)
    return {
        "sse": float(sse),
        "recompute_sq_norm": float(recompute_sq_norm),
        "cosine_mean": float(cosine.mean()),
    }


def direction_cosine(first: torch.Tensor, second: torch.Tensor) -> float:
    first = first.to(torch.float64)
    second = second.to(torch.float64)
    denominator = torch.linalg.vector_norm(first) * torch.linalg.vector_norm(second)
    if float(denominator) <= torch.finfo(torch.float64).eps:
        return math.nan
    return float(torch.dot(first, second) / denominator)


def load_layer(
    case: CaseRecord, layer: int, theta: float
) -> tuple[torch.Tensor, torch.Tensor]:
    offline_kv, offline_meta = read_layer(case.offline_layers[layer])
    recompute_kv, recompute_meta = read_layer(case.recompute_layers[layer])
    expected = (2, case.token_count, NUM_KV_HEADS * HEAD_DIM)
    if tuple(offline_kv.shape) != expected or offline_kv.shape != recompute_kv.shape:
        raise ValueError(
            f"KV shape mismatch for {case.case_id} layer={layer}: "
            f"offline={tuple(offline_kv.shape)} "
            f"Recompute={tuple(recompute_kv.shape)}"
        )
    if (
        offline_meta.get("cached_positions", {}).get("start") != 0
        or recompute_meta.get("cached_positions", {}).get("start") != case.shift
    ):
        raise ValueError(f"cached position mismatch for {case.case_id} layer={layer}")
    direct = offline_kv.reshape(2, case.token_count, NUM_KV_HEADS, HEAD_DIM)
    direct = direct.to(torch.float32)
    direct[0] = relocate_neox_rope(direct[0], case.shift, theta)
    recompute = recompute_kv.reshape(
        2, case.token_count, NUM_KV_HEADS, HEAD_DIM
    ).to(torch.float32)
    return direct, recompute


def analyze_case(
    case: CaseRecord,
    observation_start: int,
    observation_end: int,
    evaluation_start: int,
    theta: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    fidelity_rows: list[dict[str, Any]] = []
    commonality_rows: list[dict[str, Any]] = []
    for layer in range(NUM_LAYERS):
        direct_kv, recompute_kv = load_layer(case, layer, theta)
        for component_index, component in enumerate(("K", "V")):
            direct = direct_kv[component_index]
            recompute = recompute_kv[component_index]
            residual = recompute - direct
            offset = estimate_token_offset(
                residual, observation_start, observation_end
            )
            direct_suffix = direct[evaluation_start:]
            recompute_suffix = recompute[evaluation_start:]
            corrected_suffix = apply_token_offset(
                direct_suffix, offset, CORRECTION_ALPHA
            )
            direct_metrics = fidelity(direct_suffix, recompute_suffix)
            corrected_metrics = fidelity(corrected_suffix, recompute_suffix)
            fidelity_rows.append(
                {
                    "case_id": case.case_id,
                    "skill": case.skill,
                    "token_count": case.token_count,
                    "component": component,
                    "layer": layer,
                    "observation_start": observation_start,
                    "observation_end": observation_end,
                    "evaluation_start": evaluation_start,
                    "estimator": "same_skill_same_layer_head_token_mean",
                    "alpha": CORRECTION_ALPHA,
                    "estimation_skill_count": 1,
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

            suffix_residual = residual[evaluation_start:]
            suffix_mean = suffix_residual.mean(dim=0)
            for head in range(NUM_KV_HEADS):
                commonality_rows.append(
                    {
                        "case_id": case.case_id,
                        "skill": case.skill,
                        "token_count": case.token_count,
                        "component": component,
                        "layer": layer,
                        "head": head,
                        "observation_start": observation_start,
                        "observation_end": observation_end,
                        "evaluation_start": evaluation_start,
                        "alpha": CORRECTION_ALPHA,
                        "estimation_skill_count": 1,
                        "prefix_suffix_direction_cosine": direction_cosine(
                            offset[head], suffix_mean[head]
                        ),
                        "prefix_offset_norm": float(
                            torch.linalg.vector_norm(offset[head])
                        ),
                        "applied_offset_norm": float(
                            CORRECTION_ALPHA
                            * torch.linalg.vector_norm(offset[head])
                        ),
                        "suffix_mean_residual_norm": float(
                            torch.linalg.vector_norm(suffix_mean[head])
                        ),
                    }
                )
    return fidelity_rows, commonality_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--pool-dir", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--observation-start", type=int, default=132)
    parser.add_argument("--observation-end", type=int, default=256)
    parser.add_argument("--evaluation-start", type=int, default=256)
    parser.add_argument("--rope-theta", type=float, default=1_000_000.0)
    args = parser.parse_args()
    if not (
        0 <= args.observation_start < args.observation_end <= args.evaluation_start
    ):
        raise ValueError("expected observation_start < observation_end <= evaluation_start")

    specs = json.loads(args.cases.read_text(encoding="utf-8"))["cases"]
    cases = [
        validate_case(spec, args.run_dir, args.pool_dir, args.evaluation_start)
        for spec in specs
    ]
    if not cases:
        raise ValueError("at least one Skill case is required")

    fidelity_rows: list[dict[str, Any]] = []
    commonality_rows: list[dict[str, Any]] = []
    for case in cases:
        case_fidelity, case_commonality = analyze_case(
            case,
            args.observation_start,
            args.observation_end,
            args.evaluation_start,
            args.rope_theta,
        )
        fidelity_rows.extend(case_fidelity)
        commonality_rows.extend(case_commonality)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "token_axis_fidelity.csv", fidelity_rows)
    write_csv(args.output_dir / "token_residual_commonality.csv", commonality_rows)
    metadata = {
        "schema_version": 1,
        "axis": "token",
        "method": "same-Skill same-layer per-KV-head late-prefix mean offset",
        "fixed_alpha": CORRECTION_ALPHA,
        "observation_window": [args.observation_start, args.observation_end],
        "recomputed_prefix": [0, args.evaluation_start],
        "evaluation_suffix": [args.evaluation_start, "S"],
        "components": ["K", "V"],
        "num_layers": NUM_LAYERS,
        "num_kv_heads": NUM_KV_HEADS,
        "head_dim": HEAD_DIM,
        "estimation_skill_count_per_case": 1,
        "other_skills_used_for_estimation": False,
        "suffix_truth_used_for_estimation": False,
        "cross_layer_prediction": False,
        "cases": [case.case_id for case in cases],
    }
    (args.output_dir / "token_axis_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"[token-axis] cases={len(cases)} layers={NUM_LAYERS} "
        f"observation=[{args.observation_start},{args.observation_end}) "
        f"evaluation=[{args.evaluation_start},S) alpha={CORRECTION_ALPHA}"
    )


if __name__ == "__main__":
    main()
