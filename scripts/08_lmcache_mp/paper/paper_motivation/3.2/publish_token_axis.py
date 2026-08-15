#!/usr/bin/env python3
"""Publish self-only token-axis CSKCache representation diagnostics."""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import statistics
from collections import defaultdict
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def aggregate_fidelity(
    rows: list[dict[str, str]],
) -> dict[tuple[str, str], dict[str, float]]:
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["component"], row["skill"])].append(row)
    output: dict[tuple[str, str], dict[str, float]] = {}
    for key, group in groups.items():
        norm = sum(float(row["recompute_sq_norm"]) for row in group)
        direct_sse = sum(float(row["direct_to_recompute_sse"]) for row in group)
        corrected_sse = sum(
            float(row["corrected_to_recompute_sse"]) for row in group
        )
        if norm <= 0 or direct_sse <= 0:
            raise ValueError(f"non-positive aggregate norm/error for {key}")
        output[key] = {
            "direct_cosine": statistics.mean(
                float(row["direct_to_recompute_cosine_mean"]) for row in group
            ),
            "corrected_cosine": statistics.mean(
                float(row["corrected_to_recompute_cosine_mean"]) for row in group
            ),
            "direct_l2": math.sqrt(direct_sse / norm),
            "corrected_l2": math.sqrt(corrected_sse / norm),
            "sse_reduction": 1.0 - corrected_sse / direct_sse,
            "cosine_layer_win_rate": statistics.mean(
                float(row["corrected_to_recompute_cosine_mean"])
                > float(row["direct_to_recompute_cosine_mean"])
                for row in group
            ),
            "sse_layer_win_rate": statistics.mean(
                float(row["corrected_to_recompute_sse"])
                < float(row["direct_to_recompute_sse"])
                for row in group
            ),
        }
    return output


def aggregate_commonality(
    rows: list[dict[str, str]],
) -> dict[tuple[str, str], dict[str, float]]:
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["component"], row["skill"])].append(row)
    output: dict[tuple[str, str], dict[str, float]] = {}
    for key, group in groups.items():
        cosines = [
            value
            for row in group
            if math.isfinite(
                value := float(row["prefix_suffix_direction_cosine"])
            )
        ]
        if not cosines:
            raise ValueError(f"no defined direction cosine cells for {key}")
        output[key] = {
            "mean_direction_cosine": statistics.mean(cosines),
            "median_direction_cosine": statistics.median(cosines),
            "positive_direction_rate": statistics.mean(value > 0 for value in cosines),
            "defined_direction_cells": len(cosines),
            "total_cells": len(group),
        }
    return output


def copy_artifacts(run_dir: Path, result_dir: Path) -> dict[str, Path]:
    analysis = run_dir / "analysis"
    figures = run_dir / "figures"
    artifacts = {
        "tables/token_axis_fidelity.csv": analysis / "token_axis_fidelity.csv",
        "tables/token_residual_commonality.csv": analysis
        / "token_residual_commonality.csv",
        "data/token_axis_metadata.json": analysis / "token_axis_metadata.json",
        "figures/direct_recompute_layerwise_cosine.png": figures
        / "direct_recompute_layerwise_cosine.png",
        "figures/direct_recompute_layerwise_cosine.pdf": figures
        / "direct_recompute_layerwise_cosine.pdf",
        "figures/corrected_recompute_layerwise_cosine.png": figures
        / "corrected_recompute_layerwise_cosine.png",
        "figures/corrected_recompute_layerwise_cosine.pdf": figures
        / "corrected_recompute_layerwise_cosine.pdf",
        "figures/direct_recompute_layerwise_normalized_l2.png": figures
        / "direct_recompute_layerwise_normalized_l2.png",
        "figures/direct_recompute_layerwise_normalized_l2.pdf": figures
        / "direct_recompute_layerwise_normalized_l2.pdf",
        "figures/corrected_recompute_layerwise_normalized_l2.png": figures
        / "corrected_recompute_layerwise_normalized_l2.png",
        "figures/corrected_recompute_layerwise_normalized_l2.pdf": figures
        / "corrected_recompute_layerwise_normalized_l2.pdf",
        "figures/token_residual_commonality.png": figures
        / "token_residual_commonality.png",
        "figures/token_residual_commonality.pdf": figures
        / "token_residual_commonality.pdf",
    }
    for name in ("figures", "tables", "data"):
        (result_dir / name).mkdir(parents=True, exist_ok=True)
    for obsolete in (
        "token_axis_layerwise_cosine.png",
        "token_axis_layerwise_cosine.pdf",
        "token_axis_layerwise_normalized_l2.png",
        "token_axis_layerwise_normalized_l2.pdf",
    ):
        (result_dir / "figures" / obsolete).unlink(missing_ok=True)
    for relative, source in artifacts.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, result_dir / relative)
    return artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    analysis = args.run_dir / "analysis"
    metadata = json.loads(
        (analysis / "token_axis_metadata.json").read_text(encoding="utf-8")
    )
    required_flags = {
        "axis": "token",
        "fixed_alpha": 0.6,
        "estimation_skill_count_per_case": 1,
        "other_skills_used_for_estimation": False,
        "suffix_truth_used_for_estimation": False,
        "cross_layer_prediction": False,
    }
    for key, expected in required_flags.items():
        if metadata.get(key) != expected:
            raise ValueError(f"invalid token-axis metadata {key}={metadata.get(key)!r}")

    cases = json.loads(args.cases.read_text(encoding="utf-8"))["cases"]
    fidelity = aggregate_fidelity(read_rows(analysis / "token_axis_fidelity.csv"))
    commonality = aggregate_commonality(
        read_rows(analysis / "token_residual_commonality.csv")
    )
    artifacts = copy_artifacts(args.run_dir, args.result_dir)

    fidelity_lines: list[str] = []
    commonality_lines: list[str] = []
    k_passes: list[bool] = []
    for component in ("K", "V"):
        for case in cases:
            skill = case["skill"]
            item = fidelity[(component, skill)]
            cosine_delta = item["corrected_cosine"] - item["direct_cosine"]
            l2_reduction = 1.0 - item["corrected_l2"] / item["direct_l2"]
            passed = cosine_delta > 0 and l2_reduction > 0
            if component == "K":
                k_passes.append(passed)
            fidelity_lines.append(
                f"| {component} | {skill} | {item['direct_cosine']:.6f} | "
                f"{item['corrected_cosine']:.6f} | {cosine_delta:+.6f} | "
                f"{item['direct_l2']:.6f} | {item['corrected_l2']:.6f} | "
                f"{l2_reduction:+.2%} | "
                f"{item['cosine_layer_win_rate']:.1%} | "
                f"{item['sse_layer_win_rate']:.1%} | "
                f"{'Go' if passed else 'No-Go'} |"
            )
            direction = commonality[(component, skill)]
            commonality_lines.append(
                f"| {component} | {skill} | "
                f"{direction['mean_direction_cosine']:.6f} | "
                f"{direction['median_direction_cosine']:.6f} | "
                f"{direction['positive_direction_rate']:.1%} | "
                f"{int(direction['defined_direction_cells'])}/"
                f"{int(direction['total_cells'])} |"
            )

    observation = metadata["observation_window"]
    evaluation = metadata["evaluation_suffix"]
    summary = f"""# CSKCache self-only token-axis correction

## 结论

本实验在每个 Skill 内独立执行。Skill 的前 256 个 token 被视为当前请求中完整重计算的连续前缀；每一层、每个 KV head 只使用该 Skill 自身 token `[{observation[0]},{observation[1]})` 的平均残差，并以固定系数 `α=0.6` 缩放后纠正同层未观测后缀 `[{evaluation[0]},S)`。其他 Skill、其他层和后缀 Recompute 真值均不参与估计。

K headline gate 要求四个 Skill 的聚合 cosine 均提高且 normalized L2 均降低。本轮结果为 **{'Go' if all(k_passes) else 'No-Go'}（{sum(k_passes)}/{len(k_passes)} Skill）**。V 作为独立诊断，不作为 K-only 候选的必要通过条件。

## KV fidelity

| Component | Skill | Direct cosine | Corrected cosine | Cosine delta | Direct L2 | Corrected L2 | L2 reduction | Cosine layer wins | L2 layer wins | Gate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(fidelity_lines)}

## 同层 token 残差公共偏移

`prefix_suffix_direction_cosine` 比较同一 Skill、同一层、同一 KV head 中，观测窗口平均残差与未观测后缀平均残差的方向。正标量 `α=0.6` 不改变该方向指标。layer、head 和 token 不是独立实验重复。

| Component | Skill | Mean direction cosine | Median direction cosine | Positive direction cells | Defined direction cells |
|---|---|---:|---:|---:|---:|
{chr(10).join(commonality_lines)}

## 实验方法

- 数据来自 `{args.run_dir.resolve()}`；后处理没有重新启动 vLLM 或重新捕获 KV。
- Direct K 是离线位置 0 的 context-free K 经 RoPE 对齐到真实请求位置后的结果；Direct V 原样复用。
- Recompute 是同一 Skill 在真实 OpenHands 请求中完整 Prefill 得到的 KV。
- 重计算范围为 `[0,256)`，offset 估计窗口为 `[{observation[0]},{observation[1]})`，统一评价 `[256,S)`。
- 每个 `Skill × layer × component × KV head` 分别估计一个 128 维 offset，并在本次后处理中统一乘以固定的 `α=0.6`；没有跨 Skill 聚合、跨层预测、训练或 LOSO。
- 后缀 Recompute 只用于 cosine、normalized L2 和方向诊断，不参与 estimator。

## 图表

- `figures/direct_recompute_layerwise_cosine.pdf`：供论文 3.1 使用的 Direct–Recompute 逐层 K/V cosine。
- `figures/direct_recompute_layerwise_normalized_l2.pdf`：供论文 3.1 使用的 Direct–Recompute 逐层 K/V normalized L2。
- `figures/corrected_recompute_layerwise_cosine.pdf`：供论文 3.2 使用的 Corrected (`α=0.6`)–Recompute 逐层 K/V cosine。
- `figures/corrected_recompute_layerwise_normalized_l2.pdf`：供论文 3.2 使用的 Corrected (`α=0.6`)–Recompute 逐层 K/V normalized L2。
- Direct 与 Corrected 的对应子图使用相同纵轴范围，可以直接比较，且不通过自动缩放夸大纠错收益。
- `figures/token_residual_commonality.pdf`：四个 Skill 各自的 K prefix-to-suffix residual direction heatmap。
- `tables/token_axis_fidelity.csv`：逐 Skill、逐层 K/V fidelity。
- `tables/token_residual_commonality.csv`：逐 Skill、逐层、逐 KV head 的 held-out 公共偏移诊断。

## 证据边界

- 这是已有真实 KV 上的表示诊断，不是在线 kernel、Agent 行为或 TTFT 实验。
- commonality heatmap 使用后缀真值做评价，但 estimator 本身严格不读取后缀。
- Skill/case 是实验重复单位；四个 Skill 仅并列汇总，不参与彼此的 offset 估计。
- `α=0.6` 是在同一批四个 case 的既有诊断结果基础上选定的固定候选；当前结果证明该候选在这些 case 上有效，但不构成 held-out Skill 泛化证据。
"""
    (args.result_dir / "summary.md").write_text(summary, encoding="utf-8")

    manifest = [["artifact", "source"], ["summary.md", str(args.run_dir.resolve())]]
    manifest.extend(
        [relative, str(source.resolve())] for relative, source in artifacts.items()
    )
    with (args.result_dir / "source_manifest.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        csv.writer(handle).writerows(manifest)
    print(
        f"[token-axis-published] K={sum(k_passes)}/{len(k_passes)} "
        f"output={args.result_dir}"
    )


if __name__ == "__main__":
    main()
