#!/usr/bin/env python3
"""Measure one Skill's self-estimated shallow offset against its deep residual."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from analyze_context_free_residual import write_csv
from analyze_layer_axis import (
    NUM_KV_HEADS,
    NUM_LAYERS,
    CaseRecord,
    estimate_self_shallow_offset,
    load_component,
    validate_case,
)


def direction_cosine(estimate: torch.Tensor, target: torch.Tensor) -> float:
    """Cosine between two same-shaped nonzero direction vectors."""
    if estimate.shape != target.shape or estimate.ndim != 1:
        raise ValueError("direction vectors must have matching one-dimensional shapes")
    estimate = estimate.to(torch.float64)
    target = target.to(torch.float64)
    denominator = torch.linalg.vector_norm(estimate) * torch.linalg.vector_norm(
        target
    )
    if float(denominator) <= 1e-30:
        raise ValueError("cannot measure the direction of a zero residual vector")
    return float(torch.dot(estimate, target) / denominator)


def analyze_component(
    case: CaseRecord,
    component: str,
    cutoff: int,
    theta: float,
) -> list[dict[str, Any]]:
    shallow_residuals = torch.stack(
        [
            load_component(case, layer, component, theta)[2]
            for layer in range(cutoff)
        ],
        dim=0,
    )
    self_offset = estimate_self_shallow_offset(shallow_residuals)
    rows: list[dict[str, Any]] = []
    for target_layer in range(cutoff, NUM_LAYERS):
        target_residual = load_component(
            case, target_layer, component, theta
        )[2]
        target_token_mean = target_residual.mean(dim=0)
        for head in range(NUM_KV_HEADS):
            rows.append(
                {
                    "case_id": case.case_id,
                    "skill": case.skill,
                    "component": component,
                    "cutoff": cutoff,
                    "target_layer": target_layer,
                    "head": head,
                    "estimation_skill_count": 1,
                    "self_shallow_to_deep_cosine": direction_cosine(
                        self_offset[head], target_token_mean[head]
                    ),
                    "self_shallow_offset_norm": float(
                        torch.linalg.vector_norm(self_offset[head])
                    ),
                    "target_token_mean_residual_norm": float(
                        torch.linalg.vector_norm(target_token_mean[head])
                    ),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--pool-dir", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--cutoff", type=int, default=4)
    parser.add_argument("--rope-theta", type=float, default=1_000_000.0)
    args = parser.parse_args()
    if not 0 < args.cutoff < NUM_LAYERS:
        raise ValueError("cutoff must be inside [1, 39]")
    case_specs = json.loads(args.cases.read_text(encoding="utf-8"))["cases"]
    cases = [validate_case(case, args.run_dir, args.pool_dir) for case in case_specs]
    if not cases:
        raise ValueError("at least one Skill case is required")
    rows: list[dict[str, Any]] = []
    for case in cases:
        for component in ("K", "V"):
            rows.extend(
                analyze_component(
                    case, component, args.cutoff, args.rope_theta
                )
            )
    write_csv(args.output_csv, rows)
    print(
        f"[self-shallow-deep-direction] cutoff={args.cutoff} "
        f"cases={len(cases)} rows={len(rows)} output={args.output_csv}"
    )


if __name__ == "__main__":
    main()
