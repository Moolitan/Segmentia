#!/usr/bin/env python3
"""Publish self-only shallow-to-deep Skill KV results."""
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


def aggregate(
    rows: list[dict[str, str]],
) -> dict[tuple[str, str, int], dict[str, float]]:
    grouped: dict[tuple[str, str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["component"], row["skill"], int(row["cutoff"]))].append(row)
    output: dict[tuple[str, str, int], dict[str, float]] = {}
    for key, group in grouped.items():
        recompute_sq_norm = sum(float(row["recompute_sq_norm"]) for row in group)
        if recompute_sq_norm <= 0:
            raise ValueError(f"non-positive Recompute norm for {key}")
        output[key] = {
            "direct_cosine": statistics.mean(
                float(row["direct_to_recompute_cosine_mean"]) for row in group
            ),
            "corrected_cosine": statistics.mean(
                float(row["corrected_to_recompute_cosine_mean"]) for row in group
            ),
            "direct_normalized_l2": math.sqrt(
                sum(float(row["direct_to_recompute_sse"]) for row in group)
                / recompute_sq_norm
            ),
            "corrected_normalized_l2": math.sqrt(
                sum(float(row["corrected_to_recompute_sse"]) for row in group)
                / recompute_sq_norm
            ),
            "cosine_layer_win_rate": statistics.mean(
                float(row["corrected_to_recompute_cosine_mean"])
                > float(row["direct_to_recompute_cosine_mean"])
                for row in group
            ),
            "l2_layer_win_rate": statistics.mean(
                float(row["corrected_to_recompute_sse"])
                < float(row["direct_to_recompute_sse"])
                for row in group
            ),
        }
    return output


def copy_artifacts(
    run_dir: Path, result_dir: Path
) -> dict[str, Path]:
    analysis = run_dir / "analysis"
    figures = run_dir / "figures"
    artifacts = {
        "tables/layer_axis_fidelity.csv": analysis / "layer_axis_fidelity.csv",
        "tables/direct_full_layerwise_cosine.csv": analysis
        / "direct_full_layerwise_cosine.csv",
        "tables/corrected_recompute_layerwise_cosine.csv": analysis
        / "corrected_recompute_layerwise_cosine.csv",
        "tables/corrected_recompute_8layers_layerwise_cosine.csv": analysis
        / "corrected_recompute_8layers_layerwise_cosine.csv",
        "tables/direct_recompute_layerwise_normalized_l2.csv": analysis
        / "direct_recompute_layerwise_normalized_l2.csv",
        "tables/corrected_recompute_layerwise_normalized_l2.csv": analysis
        / "corrected_recompute_layerwise_normalized_l2.csv",
        "tables/corrected_recompute_8layers_layerwise_normalized_l2.csv": analysis
        / "corrected_recompute_8layers_layerwise_normalized_l2.csv",
        "tables/shallow_deep_residual_direction.csv": analysis
        / "shallow_deep_residual_direction.csv",
        "data/layer_axis_parameters.csv": analysis / "layer_axis_parameters.csv",
        "data/layer_axis_metadata.json": analysis / "layer_axis_metadata.json",
        "figures/direct_full_layerwise_cosine.png": figures
        / "direct_full_layerwise_cosine.png",
        "figures/direct_full_layerwise_cosine.pdf": figures
        / "direct_full_layerwise_cosine.pdf",
        "figures/corrected_recompute_layerwise_cosine.png": figures
        / "corrected_recompute_layerwise_cosine.png",
        "figures/corrected_recompute_layerwise_cosine.pdf": figures
        / "corrected_recompute_layerwise_cosine.pdf",
        "figures/corrected_recompute_8layers_layerwise_cosine.png": figures
        / "corrected_recompute_8layers_layerwise_cosine.png",
        "figures/corrected_recompute_8layers_layerwise_cosine.pdf": figures
        / "corrected_recompute_8layers_layerwise_cosine.pdf",
        "figures/direct_recompute_layerwise_normalized_l2.png": figures
        / "direct_recompute_layerwise_normalized_l2.png",
        "figures/direct_recompute_layerwise_normalized_l2.pdf": figures
        / "direct_recompute_layerwise_normalized_l2.pdf",
        "figures/corrected_recompute_layerwise_normalized_l2.png": figures
        / "corrected_recompute_layerwise_normalized_l2.png",
        "figures/corrected_recompute_layerwise_normalized_l2.pdf": figures
        / "corrected_recompute_layerwise_normalized_l2.pdf",
        "figures/corrected_recompute_8layers_layerwise_normalized_l2.png": figures
        / "corrected_recompute_8layers_layerwise_normalized_l2.png",
        "figures/corrected_recompute_8layers_layerwise_normalized_l2.pdf": figures
        / "corrected_recompute_8layers_layerwise_normalized_l2.pdf",
        "figures/shallow_deep_residual_direction.png": figures
        / "shallow_deep_residual_direction.png",
        "figures/shallow_deep_residual_direction.pdf": figures
        / "shallow_deep_residual_direction.pdf",
    }
    result_dir.mkdir(parents=True, exist_ok=True)
    for name in ("figures", "tables", "data"):
        (result_dir / name).mkdir(exist_ok=True)
    for relative, source in artifacts.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, result_dir / relative)
    return artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    args = parser.parse_args()
    analysis = args.run_dir / "analysis"
    fidelity_rows = read_rows(analysis / "layer_axis_fidelity.csv")
    direction_rows = read_rows(analysis / "shallow_deep_residual_direction.csv")
    metadata = json.loads(
        (analysis / "layer_axis_metadata.json").read_text(encoding="utf-8")
    )
    if metadata.get("other_skills_used_for_estimation") is not False:
        raise ValueError("publisher requires self-only layer-axis metadata")
    cases = json.loads(args.cases.read_text(encoding="utf-8"))["cases"]
    cutoffs = [int(value) for value in metadata["cutoffs"]]
    metrics = aggregate(fidelity_rows)
    artifacts = copy_artifacts(args.run_dir, args.result_dir)

    gate_rows: list[str] = []
    case_rows: list[str] = []
    for component in ("K", "V"):
        for cutoff in cutoffs:
            passes: list[bool] = []
            for case in cases:
                item = metrics[(component, case["skill"], cutoff)]
                cosine_delta = item["corrected_cosine"] - item["direct_cosine"]
                l2_reduction = 1.0 - (
                    item["corrected_normalized_l2"]
                    / item["direct_normalized_l2"]
                )
                case_pass = cosine_delta > 0 and l2_reduction > 0
                passes.append(case_pass)
                case_rows.append(
                    "| {component} | {skill} | {cutoff} | {direct_cos:.6f} | "
                    "{corrected_cos:.6f} | {cos_delta:+.6f} | {direct_l2:.6f} | "
                    "{corrected_l2:.6f} | {l2_reduction:+.2%} | {cos_win:.1%} | "
                    "{l2_win:.1%} | {status} |".format(
                        component=component,
                        skill=case["skill"],
                        cutoff=cutoff,
                        direct_cos=item["direct_cosine"],
                        corrected_cos=item["corrected_cosine"],
                        cos_delta=cosine_delta,
                        direct_l2=item["direct_normalized_l2"],
                        corrected_l2=item["corrected_normalized_l2"],
                        l2_reduction=l2_reduction,
                        cos_win=item["cosine_layer_win_rate"],
                        l2_win=item["l2_layer_win_rate"],
                        status="Go" if case_pass else "No-Go",
                    )
                )
            gate_rows.append(
                f"| {component} | {cutoff} | {sum(passes)}/{len(passes)} | "
                f"{'Go' if all(passes) else 'No-Go'} |"
            )

    direction_summary: list[str] = []
    for component in ("K", "V"):
        component_rows = [
            row for row in direction_rows if row["component"] == component
        ]
        cell_values: dict[tuple[int, int], list[float]] = defaultdict(list)
        for row in component_rows:
            cell_values[(int(row["target_layer"]), int(row["head"]))].append(
                float(row["self_shallow_to_deep_cosine"])
            )
        macro_cells = [statistics.mean(values) for values in cell_values.values()]
        case_means = []
        for case in cases:
            values = [
                float(row["self_shallow_to_deep_cosine"])
                for row in component_rows
                if row["skill"] == case["skill"]
            ]
            case_means.append(statistics.mean(values))
        direction_summary.append(
            f"| {component} | {statistics.mean(case_means):.6f} | "
            f"{min(case_means):.6f} | {max(case_means):.6f} | "
            f"{sum(value > 0 for value in macro_cells)}/{len(macro_cells)} |"
        )

    summary = f"""# Self-only shallow-to-deep Skill KV correction

## 结论

本实验对每个 Skill 独立估计纠错量。对于浅层预算 `c`，只读取当前 Skill 在第 1--`c` 层完整重计算得到的 K/V，并将这些层、全部 Skill token 的残差平均为逐 KV-head 向量。该向量直接应用到同一 Skill 的所有未计算深层。其他 Skill 不参与估计，深层 Recompute 只用于最终评价。

Go 要求同一 component/cutoff 下四个 Skill 的深层聚合 cosine 和 normalized L2 均优于 Direct。

| Component | Shallow layers | Passing Skills | Gate |
|---|---:|---:|---|
{chr(10).join(gate_rows)}

## Case-level fidelity

| Component | Skill | Shallow layers | Direct cosine | Corrected cosine | Cosine delta | Direct L2 | Corrected L2 | L2 reduction | Cosine layer wins | L2 layer wins | Gate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(case_rows)}

## 自身浅层与自身深层的残差方向

方向图固定使用前 4 层。每个 Skill 先独立计算逐 head 的自身浅层平均残差，再与该 Skill 各目标深层的 token-mean 真实残差计算 cosine；表中最后一列统计四 Skill 宏平均后为正的 `target-layer × head` 单元。

| Component | Skill-macro mean | Min Skill mean | Max Skill mean | Positive cells |
|---|---:|---:|---:|---:|
{chr(10).join(direction_summary)}

## 实验方法

- 数据复用自 `{args.run_dir.resolve()}`；没有重新启动 vLLM，也没有重新捕获 Skill KV。
- 离线 KV 来自位置 0、无在线上下文的 Skill；Recompute KV 来自该 Skill 的真实 OpenHands 请求。
- K 先做 RoPE 位置对齐，V 原样复用。残差定义为 `Recompute - Direct`。
- 对 cutoff 4 和 8，纠错量为当前 Skill 自身 `[0, cutoff)` 层和全部 token 的逐 KV-head 平均残差。
- 同一纠错量应用到该 Skill 的所有深层 token；没有拟合、训练、LOSO、Mean-only 或跨 Skill 参数。
- 深层 Recompute 不参与纠错量估计，只用于计算 cosine、SSE 和 normalized L2。

## 图表

- `figures/direct_full_layerwise_cosine.pdf`：Direct 相对 Recompute 的逐层 K/V cosine。
- `figures/corrected_recompute_layerwise_cosine.pdf`：自身前 4 层估计并纠错深层后的 cosine。
- `figures/corrected_recompute_8layers_layerwise_cosine.pdf`：自身前 8 层估计并纠错深层后的 cosine。
- 三个 `*_layerwise_normalized_l2.pdf`：Direct、4-layer 和 8-layer 的逐层归一化 L2。
- `figures/shallow_deep_residual_direction.pdf`：自身前 4 层平均残差与自身目标深层 token-mean 残差的逐层、逐 head cosine。
- `tables/layer_axis_fidelity.csv`：所有 self-only 深层 fidelity 数据。
- `data/layer_axis_parameters.csv`：每个 Skill 自身浅层估计得到的逐 head offset 摘要。

## 证据边界

- 这是已有 KV 上的表示诊断，不是已实现的在线 hybrid forward，也不证明 Prefill 延迟或 Agent 质量改善。
- 纠错图浅层的 cosine=1、L2=0 来自候选策略“这些层完整重计算”的定义；深层数据才是 self-only 纠错评价。
- Skill/case 是实验重复单位；layer、head 和 token 不是独立样本。
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
    print(f"[self-layer-axis-published] output={args.result_dir}")


if __name__ == "__main__":
    main()
