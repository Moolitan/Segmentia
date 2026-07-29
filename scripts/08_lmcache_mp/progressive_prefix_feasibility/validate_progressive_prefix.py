#!/usr/bin/env python3
"""Test whether a recomputed Skill prefix predicts the remaining cached K.

This is an offline feasibility diagnostic.  A decision-side stability signal is
computed only from the prefix that an online system would have recomputed.  The
target-full tail is used strictly after that calculation to score prediction
quality; it never enters the observable signal.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
DIAGNOSIS_DIR = SCRIPT_DIR.parent / "context_residual_diagnosis"
if str(DIAGNOSIS_DIR) not in sys.path:
    sys.path.insert(0, str(DIAGNOSIS_DIR))

from analyze_single_case import (  # noqa: E402
    DEFAULT_MODEL_CONFIG,
    atomic_write_json,
    decode_positions,
    load_sidecar,
    read_raw_kv,
    relocate_neox_rope,
    require_populated_raw_kv,
    validate_case,
    write_csv,
)


CALIBRATION_START = 132
DEFAULT_PREFIX_ENDPOINTS = (160, 192, 224, 256)
COMMON_EVALUATION_START = 256
EARLY_RETENTION_RATIO = 0.80
POOLED_SPEARMAN_GATE = 0.50


def _safe_cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left64 = left.to(torch.float64).reshape(-1)
    right64 = right.to(torch.float64).reshape(-1)
    denominator = math.sqrt(
        max(float(torch.sum(left64.square()) * torch.sum(right64.square())), 1e-30)
    )
    return float(torch.sum(left64 * right64)) / denominator


def _relative_difference(value: torch.Tensor, reference: torch.Tensor) -> float:
    numerator = float(torch.sum((value.to(torch.float64) - reference) ** 2))
    denominator = float(torch.sum(reference.to(torch.float64) ** 2))
    return math.sqrt(numerator / max(denominator, 1e-30))


def _squared_error(prediction: torch.Tensor, target: torch.Tensor) -> float:
    error = prediction.to(torch.float64) - target.to(torch.float64)
    return float(torch.sum(error.square()))


def compute_layer_head_rows(
    *,
    source: torch.Tensor,
    target: torch.Tensor,
    case_id: str,
    split: str,
    layer: int,
    prefix_endpoints: Iterable[int] = DEFAULT_PREFIX_ENDPOINTS,
    calibration_start: int = CALIBRATION_START,
    common_evaluation_start: int = COMMON_EVALUATION_START,
) -> list[dict[str, Any]]:
    """Return prefix-only stability and separately scored tail quality.

    ``source`` and ``target`` are K tensors shaped [tokens, kv_heads, head_dim]
    after the source K has already been relocated to target positions.
    """

    if source.shape != target.shape or source.ndim != 3:
        raise ValueError("source and target must share [tokens, heads, head_dim]")
    tokens, heads, _ = source.shape
    endpoints = tuple(sorted(set(int(value) for value in prefix_endpoints)))
    if not endpoints:
        raise ValueError("prefix_endpoints cannot be empty")
    if not 0 <= calibration_start < endpoints[0]:
        raise ValueError("calibration_start must precede every prefix endpoint")
    if endpoints[-1] > common_evaluation_start:
        raise ValueError("prefix endpoints cannot cross the common evaluation tail")
    if not common_evaluation_start < tokens:
        raise ValueError("case has no tokens after the common evaluation start")

    residual = target.to(torch.float32) - source.to(torch.float32)
    true_tail_offset = residual[common_evaluation_start:].mean(dim=0)
    common_source = source[common_evaluation_start:].to(torch.float32)
    common_target = target[common_evaluation_start:].to(torch.float32)
    previous_offset: torch.Tensor | None = None
    previous_endpoint: int | None = None
    rows: list[dict[str, Any]] = []

    for endpoint in endpoints:
        if not calibration_start < endpoint <= common_evaluation_start:
            raise ValueError(f"invalid prefix endpoint: {endpoint}")
        prefix_offset = residual[calibration_start:endpoint].mean(dim=0)
        policy_source = source[endpoint:].to(torch.float32)
        policy_target = target[endpoint:].to(torch.float32)

        for head in range(heads):
            observable_relative_change: float | None = None
            observable_cosine: float | None = None
            if previous_offset is not None:
                observable_relative_change = _relative_difference(
                    previous_offset[head], prefix_offset[head]
                )
                observable_cosine = _safe_cosine(
                    prefix_offset[head], previous_offset[head]
                )

            common_direct_error = _squared_error(
                common_source[:, head], common_target[:, head]
            )
            common_corrected_error = _squared_error(
                common_source[:, head] + prefix_offset[head],
                common_target[:, head],
            )
            common_oracle_error = _squared_error(
                common_source[:, head] + true_tail_offset[head],
                common_target[:, head],
            )
            policy_direct_error = _squared_error(
                policy_source[:, head], policy_target[:, head]
            )
            policy_corrected_error = _squared_error(
                policy_source[:, head] + prefix_offset[head],
                policy_target[:, head],
            )
            rows.append(
                {
                    "case_id": case_id,
                    "split": split,
                    "layer": layer,
                    "head": head,
                    "tokens": tokens,
                    "calibration_start": calibration_start,
                    "prefix_end": endpoint,
                    "prefix_observation_tokens": endpoint - calibration_start,
                    "previous_prefix_end": previous_endpoint,
                    # The next two fields are the only proposed online signal.
                    "observable_relative_change": observable_relative_change,
                    "observable_cosine": observable_cosine,
                    # Everything below is hidden-tail evaluation only.
                    "tail_offset_relative_error": _relative_difference(
                        prefix_offset[head], true_tail_offset[head]
                    ),
                    "tail_offset_cosine": _safe_cosine(
                        prefix_offset[head], true_tail_offset[head]
                    ),
                    "common_eval_start": common_evaluation_start,
                    "common_eval_tokens": tokens - common_evaluation_start,
                    "common_direct_squared_error": common_direct_error,
                    "common_corrected_squared_error": common_corrected_error,
                    "common_oracle_squared_error": common_oracle_error,
                    "policy_eval_start": endpoint,
                    "policy_eval_tokens": tokens - endpoint,
                    "policy_direct_squared_error": policy_direct_error,
                    "policy_corrected_squared_error": policy_corrected_error,
                }
            )
        previous_offset = prefix_offset
        previous_endpoint = endpoint
    return rows


def _rankdata(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        average_rank = (cursor + 1 + end) / 2.0
        for index in order[cursor:end]:
            ranks[index] = average_rank
        cursor = end
    return ranks


def spearman_correlation(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right):
        raise ValueError("Spearman inputs must have equal lengths")
    if len(left) < 2:
        return None
    left_rank = _rankdata(left)
    right_rank = _rankdata(right)
    left_mean = statistics.fmean(left_rank)
    right_mean = statistics.fmean(right_rank)
    left_centered = [value - left_mean for value in left_rank]
    right_centered = [value - right_mean for value in right_rank]
    numerator = sum(a * b for a, b in zip(left_centered, right_centered, strict=True))
    denominator = math.sqrt(
        sum(value * value for value in left_centered)
        * sum(value * value for value in right_centered)
    )
    if denominator == 0:
        return None
    return numerator / denominator


def summarize_case_budgets(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["case_id"], row["split"], row["prefix_end"])].append(row)
    summaries: list[dict[str, Any]] = []
    for (case_id, split, endpoint), values in sorted(grouped.items()):
        direct = sum(float(row["common_direct_squared_error"]) for row in values)
        corrected = sum(float(row["common_corrected_squared_error"]) for row in values)
        oracle = sum(float(row["common_oracle_squared_error"]) for row in values)
        policy_direct = sum(float(row["policy_direct_squared_error"]) for row in values)
        policy_corrected = sum(
            float(row["policy_corrected_squared_error"]) for row in values
        )
        observable_changes = [
            float(row["observable_relative_change"])
            for row in values
            if row["observable_relative_change"] is not None
        ]
        observable_cosines = [
            float(row["observable_cosine"])
            for row in values
            if row["observable_cosine"] is not None
        ]
        summaries.append(
            {
                "case_id": case_id,
                "split": split,
                "tokens": values[0]["tokens"],
                "prefix_end": endpoint,
                "prefix_observation_tokens": values[0]["prefix_observation_tokens"],
                "common_eval_start": values[0]["common_eval_start"],
                "common_improvement_vs_direct": 1.0 - corrected / max(direct, 1e-30),
                "common_oracle_improvement_vs_direct": 1.0
                - oracle / max(direct, 1e-30),
                "policy_improvement_vs_direct": 1.0
                - policy_corrected / max(policy_direct, 1e-30),
                "median_observable_relative_change": (
                    statistics.median(observable_changes)
                    if observable_changes
                    else None
                ),
                "median_observable_cosine": (
                    statistics.median(observable_cosines)
                    if observable_cosines
                    else None
                ),
                "median_tail_offset_relative_error": statistics.median(
                    float(row["tail_offset_relative_error"]) for row in values
                ),
                "median_tail_offset_cosine": statistics.median(
                    float(row["tail_offset_cosine"]) for row in values
                ),
            }
        )
    return summaries


def evaluate_feasibility_gate(
    rows: list[dict[str, Any]], summaries: list[dict[str, Any]]
) -> dict[str, Any]:
    case_ids = sorted({row["case_id"] for row in rows})
    if len(case_ids) < 3:
        raise ValueError("Feasibility gate requires at least three long-Skill cases")
    summary_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in summaries:
        summary_by_case[row["case_id"]].append(row)

    fixed_positive: list[str] = []
    early_retention: dict[str, int] = {}
    for case_id in case_ids:
        case_rows = sorted(summary_by_case[case_id], key=lambda row: row["prefix_end"])
        fixed = next(
            (row for row in case_rows if row["prefix_end"] == COMMON_EVALUATION_START),
            None,
        )
        if fixed is None:
            raise ValueError(f"Case {case_id} lacks the frozen 256-token endpoint")
        fixed_gain = float(fixed["common_improvement_vs_direct"])
        if fixed_gain > 0:
            fixed_positive.append(case_id)
        for candidate in case_rows:
            gain = float(candidate["common_improvement_vs_direct"])
            if (
                candidate["prefix_end"] < COMMON_EVALUATION_START
                and gain > 0
                and fixed_gain > 0
                and gain >= EARLY_RETENTION_RATIO * fixed_gain
            ):
                early_retention[case_id] = int(candidate["prefix_end"])
                break

    correlations: dict[str, float | None] = {}
    pooled_left: list[float] = []
    pooled_right: list[float] = []
    for case_id in case_ids:
        eligible = [
            row
            for row in rows
            if row["case_id"] == case_id
            and row["observable_relative_change"] is not None
        ]
        left = [float(row["observable_relative_change"]) for row in eligible]
        right = [float(row["tail_offset_relative_error"]) for row in eligible]
        correlations[case_id] = spearman_correlation(left, right)
        pooled_left.extend(left)
        pooled_right.extend(right)
    pooled = spearman_correlation(pooled_left, pooled_right)
    positive_correlation_cases = [
        case_id
        for case_id, value in correlations.items()
        if value is not None and value > 0
    ]

    transfer_go = len(fixed_positive) == len(case_ids)
    observability_go = (
        pooled is not None
        and pooled >= POOLED_SPEARMAN_GATE
        and len(positive_correlation_cases) >= 2
    )
    adaptivity_go = len(early_retention) >= 2
    if transfer_go and observability_go and adaptivity_go:
        status = "go"
    elif transfer_go:
        status = "weak_go"
    else:
        status = "no_go"
    return {
        "status": status,
        "case_count": len(case_ids),
        "transfer": {
            "status": "go" if transfer_go else "no_go",
            "fixed_256_positive_cases": fixed_positive,
            "required": len(case_ids),
        },
        "observability": {
            "status": "go" if observability_go else "no_go",
            "case_spearman": correlations,
            "positive_case_count": len(positive_correlation_cases),
            "pooled_spearman": pooled,
            "pooled_threshold": POOLED_SPEARMAN_GATE,
        },
        "adaptivity_potential": {
            "status": "go" if adaptivity_go else "no_go",
            "early_endpoint_by_case": early_retention,
            "retention_ratio": EARLY_RETENTION_RATIO,
            "required_cases": 2,
        },
        "interpretation": {
            "go": "Prefix transfer and a prefix-only adaptive signal are feasible.",
            "weak_go": "Fixed prefix correction transfers, but adaptive stopping is not supported.",
            "no_go": "The frozen prefix correction does not transfer across all long-Skill cases.",
        }[status],
    }


def _analyze_case(
    case_dir: Path,
    split: str,
    model_config: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = case_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_files, target_files, tokens, kv_heads, _ = validate_case(
        manifest, model_config
    )
    if tokens <= COMMON_EVALUATION_START:
        raise ValueError(f"Case {manifest['case_id']} is too short for this M0 gate")
    head_dim = int(model_config["head_dim"])
    rope_theta = float(model_config["rope_theta"])
    rows: list[dict[str, Any]] = []
    expected_shape = [2, tokens, kv_heads * head_dim]

    for layer, (source_path, target_path) in enumerate(
        zip(source_files, target_files, strict=True)
    ):
        source_meta = load_sidecar(source_path)
        target_meta = load_sidecar(target_path)
        require_populated_raw_kv(source_path)
        require_populated_raw_kv(target_path)
        if source_meta["shape"] != expected_shape or target_meta["shape"] != expected_shape:
            raise ValueError(f"Layer {layer} shape mismatch")
        source_identity = str(source_meta.get("cache_key", "")).rsplit("@", 1)[0]
        target_identity = str(target_meta.get("cache_key", "")).rsplit("@", 1)[0]
        if source_identity != target_identity:
            raise ValueError(f"Layer {layer} source/target cache identities differ")
        source_positions = decode_positions(source_meta["cached_positions"], tokens)
        target_positions = decode_positions(target_meta["cached_positions"], tokens)
        source_raw = read_raw_kv(source_path, expected_shape)
        target_raw = read_raw_kv(target_path, expected_shape)
        aligned_source_k = relocate_neox_rope(
            source_raw[0].reshape(tokens, kv_heads, head_dim),
            source_positions,
            target_positions,
            rope_theta,
        )
        target_k = (
            target_raw[0].reshape(tokens, kv_heads, head_dim).to(torch.float32)
        )
        rows.extend(
            compute_layer_head_rows(
                source=aligned_source_k,
                target=target_k,
                case_id=manifest["case_id"],
                split=split,
                layer=layer,
            )
        )
    return {
        "case_id": manifest["case_id"],
        "skill": manifest["skill"],
        "split": split,
        "tokens": tokens,
        "layers": len(source_files),
        "kv_heads": kv_heads,
        "manifest": str(manifest_path.resolve()),
    }, rows


def _make_plots(
    output_dir: Path,
    rows: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []

    eligible = [row for row in rows if row["observable_relative_change"] is not None]
    fig, axis = plt.subplots(figsize=(6.4, 4.4))
    for case_id in sorted({row["case_id"] for row in eligible}):
        subset = [row for row in eligible if row["case_id"] == case_id]
        axis.scatter(
            [row["observable_relative_change"] for row in subset],
            [row["tail_offset_relative_error"] for row in subset],
            s=8,
            alpha=0.35,
            label=case_id,
        )
    axis.set_xlabel("Prefix-only residual change (lower is more stable)")
    axis.set_ylabel("Hidden common-tail offset error")
    axis.set_title("Can prefix stability predict tail correction quality?")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    path = figures_dir / "stability_vs_tail_error.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path.resolve()))

    fig, axis = plt.subplots(figsize=(6.4, 4.4))
    for case_id in sorted({row["case_id"] for row in summaries}):
        subset = sorted(
            (row for row in summaries if row["case_id"] == case_id),
            key=lambda row: row["prefix_end"],
        )
        axis.plot(
            [row["prefix_end"] for row in subset],
            [100.0 * row["common_improvement_vs_direct"] for row in subset],
            marker="o",
            label=case_id,
        )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xlabel("Locally recomputed Skill prefix endpoint")
    axis.set_ylabel("Common-tail K squared-error improvement (%)")
    axis.set_title("Prefix budget versus correction gain")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    path = figures_dir / "budget_vs_correction_gain.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path.resolve()))
    return paths


def _write_summary_markdown(
    output_dir: Path,
    cases: list[dict[str, Any]],
    gate: dict[str, Any],
) -> None:
    case_lines = "\n".join(
        f"- `{case['case_id']}`: split={case['split']}, tokens={case['tokens']}, "
        f"layers={case['layers']}" for case in cases
    )
    content = f"""# 渐进式 Skill 前缀重计算可行性

## 结论

当前自动 gate：**{gate['status']}**。

{gate['interpretation']}

## 问题与方法

本分析检验：当前上下文下真实重计算的 Skill 前缀 K residual，能否预测同一
Skill 剩余复用区的 K residual。source K 先迁移到 target absolute positions；
在线可观察量只比较相邻前缀预算的 residual estimate。target-full 的 `[256,E)`
只用于事后评分，未进入稳定性计算。

## 数据

{case_lines}

## Gate

- 固定256前缀迁移：{gate['transfer']['status']}；positive cases = {gate['transfer']['fixed_256_positive_cases']}。
- 前缀稳定性可观察性：{gate['observability']['status']}；pooled Spearman = {gate['observability']['pooled_spearman']}。
- 小于256预算的潜力：{gate['adaptivity_potential']['status']}；early endpoints = {gate['adaptivity_potential']['early_endpoint_by_case']}。

## 可靠性边界

- case是实验重复单位；layer/head只描述内部结构，不作为独立样本声称统计显著性。
- `observable_relative_change`只读取prefix；tail指标是离线oracle评价。
- 本轮只分析K，不分析V、logits、Agent action或真实延迟。
- Go只允许进入在线状态机和延迟验证，不等价于端到端质量已经成立。
"""
    (output_dir / "summary.md").write_text(content, encoding="utf-8")


def validate_progressive_prefix(
    *,
    design_case_dir: Path,
    heldout_case_dirs: list[Path],
    model_config_path: Path,
    output_dir: Path,
    plots: bool = True,
) -> dict[str, Any]:
    if len(heldout_case_dirs) < 2:
        raise ValueError("At least two held-out long-Skill cases are required")
    model_config = json.loads(model_config_path.read_text(encoding="utf-8"))
    cases: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for case_dir, split in (
        [(design_case_dir, "design")]
        + [(case_dir, "heldout") for case_dir in heldout_case_dirs]
    ):
        metadata, case_rows = _analyze_case(case_dir, split, model_config)
        cases.append(metadata)
        rows.extend(case_rows)

    summaries = summarize_case_budgets(rows)
    gate = evaluate_feasibility_gate(rows, summaries)
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = output_dir / "tables"
    data_dir = output_dir / "data"
    write_csv(tables_dir / "layer_head_metrics.csv", rows)
    write_csv(tables_dir / "case_budget_summary.csv", summaries)
    atomic_write_json(data_dir / "feasibility_gate.json", gate)
    figures = _make_plots(output_dir, rows, summaries) if plots else []
    _write_summary_markdown(output_dir, cases, gate)

    manifest_rows: list[dict[str, Any]] = []
    for case in cases:
        manifest_rows.append(
            {
                "artifact": case["manifest"],
                "kind": "input_capture_manifest",
                "source": case["case_id"],
                "notes": f"{case['split']} source/target-full raw KV pair",
            }
        )
    for path, kind in (
        (tables_dir / "layer_head_metrics.csv", "table"),
        (tables_dir / "case_budget_summary.csv", "table"),
        (data_dir / "feasibility_gate.json", "data"),
        (output_dir / "summary.md", "summary"),
    ):
        manifest_rows.append(
            {
                "artifact": str(path.resolve()),
                "kind": kind,
                "source": "validate_progressive_prefix.py",
                "notes": "generated from validated raw KV captures",
            }
        )
    for path in figures:
        manifest_rows.append(
            {
                "artifact": path,
                "kind": "figure",
                "source": "validate_progressive_prefix.py",
                "notes": "descriptive visualization; cases are replicates",
            }
        )
    write_csv(output_dir / "source_manifest.csv", manifest_rows)
    result = {
        "schema_version": 1,
        "status": "progressive_prefix_feasibility",
        "cases": cases,
        "configuration": {
            "calibration_start": CALIBRATION_START,
            "prefix_endpoints": list(DEFAULT_PREFIX_ENDPOINTS),
            "common_evaluation_start": COMMON_EVALUATION_START,
            "kv_type": "K_only",
            "source_alignment": "post_rope_source_to_target_absolute_positions",
        },
        "gate": gate,
        "artifacts": {
            "output_dir": str(output_dir.resolve()),
            "figures": figures,
        },
    }
    atomic_write_json(data_dir / "analysis_summary.json", result)
    return result


def preflight_inputs(
    *,
    design_case_dir: Path,
    heldout_case_dirs: list[Path],
    model_config_path: Path,
) -> list[dict[str, Any]]:
    """Validate manifests, layer coverage, sidecars and raw sizes without scanning KV."""

    if len(heldout_case_dirs) < 2:
        raise ValueError("At least two held-out long-Skill cases are required")
    model_config = json.loads(model_config_path.read_text(encoding="utf-8"))
    results: list[dict[str, Any]] = []
    seen_case_ids: set[str] = set()
    for case_dir, split in (
        [(design_case_dir, "design")]
        + [(case_dir, "heldout") for case_dir in heldout_case_dirs]
    ):
        manifest_path = case_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        source_files, target_files, tokens, kv_heads, _ = validate_case(
            manifest, model_config
        )
        case_id = str(manifest["case_id"])
        if case_id in seen_case_ids:
            raise ValueError(f"Duplicate case_id: {case_id}")
        seen_case_ids.add(case_id)
        if tokens <= COMMON_EVALUATION_START:
            raise ValueError(f"Case {case_id} is too short for this M0 gate")
        for path in source_files + target_files:
            load_sidecar(path)
        results.append(
            {
                "case_id": case_id,
                "split": split,
                "tokens": tokens,
                "kv_heads": kv_heads,
                "layers": len(source_files),
                "manifest": str(manifest_path.resolve()),
            }
        )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design-case-dir", type=Path, required=True)
    parser.add_argument("--heldout-case-dir", type=Path, action="append", required=True)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.preflight_only:
        cases = preflight_inputs(
            design_case_dir=args.design_case_dir,
            heldout_case_dirs=args.heldout_case_dir,
            model_config_path=args.model_config,
        )
        for case in cases:
            print(
                f"[preflight] case={case['case_id']} split={case['split']} "
                f"tokens={case['tokens']} layers={case['layers']}"
            )
        print(f"[preflight-complete] cases={len(cases)}")
        return
    result = validate_progressive_prefix(
        design_case_dir=args.design_case_dir,
        heldout_case_dirs=args.heldout_case_dir,
        model_config_path=args.model_config,
        output_dir=args.output_dir,
        plots=not args.no_plots,
    )
    print(
        f"[validated] cases={len(result['cases'])} "
        f"progressive_prefix_gate={result['gate']['status']} "
        f"output={args.output_dir}"
    )


if __name__ == "__main__":
    main()
