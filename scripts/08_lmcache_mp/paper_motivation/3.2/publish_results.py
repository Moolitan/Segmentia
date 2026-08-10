#!/usr/bin/env python3
"""Publish lightweight context-free residual results after a completed run."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import statistics
import math
from collections import defaultdict
from pathlib import Path


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    args = parser.parse_args()
    analysis = args.run_dir / "analysis"
    figures = args.run_dir / "figures"
    metrics = rows(analysis / "layer_metrics.csv")
    cosines = rows(analysis / "head_cosines.csv")
    calibrated_metrics = rows(analysis / "calibrated_layer_metrics.csv")
    calibration_heads = rows(analysis / "calibration_heads.csv")
    fidelity = rows(analysis / "fidelity_layer_metrics.csv")
    value_calibration_heads = rows(analysis / "value_calibration_heads.csv")
    cases = json.loads(args.cases.read_text(encoding="utf-8"))["cases"]

    args.result_dir.mkdir(parents=True, exist_ok=True)
    for name in ("figures", "tables", "data"):
        (args.result_dir / name).mkdir(exist_ok=True)
    for path in figures.glob("*"):
        if path.suffix in {".png", ".pdf"}:
            shutil.copy2(path, args.result_dir / "figures" / path.name)
    shutil.copy2(analysis / "layer_metrics.csv", args.result_dir / "tables")
    shutil.copy2(analysis / "head_cosines.csv", args.result_dir / "data")
    shutil.copy2(
        analysis / "calibrated_layer_metrics.csv", args.result_dir / "tables"
    )
    shutil.copy2(analysis / "calibration_heads.csv", args.result_dir / "data")
    shutil.copy2(analysis / "fidelity_layer_metrics.csv", args.result_dir / "tables")
    shutil.copy2(
        analysis / "value_calibration_heads.csv", args.result_dir / "data"
    )

    by_case: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in metrics:
        if int(row["budget"]) == 256:
            by_case[row["case_id"]].append(row)
    cosine_by_case: dict[str, list[float]] = defaultdict(list)
    for row in cosines:
        if int(row["budget"]) == 256:
            cosine_by_case[row["case_id"]].append(float(row["offset_cosine"]))

    summaries = []
    for case in cases:
        case_id = case["case_id"]
        case_rows = by_case[case_id]
        direct = sum(float(row["direct_sse"]) for row in case_rows)
        corrected = sum(float(row["corrected_sse"]) for row in case_rows)
        gain = 100.0 * (1.0 - corrected / direct)
        median = statistics.median(cosine_by_case[case_id])
        summaries.append((case_id, case["skill"], gain, median))
    unit_go = all(item[2] > 0 for item in summaries)

    calibrated_by_case: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in calibrated_metrics:
        calibrated_by_case[row["case_id"]].append(row)
    alpha_by_case: dict[str, list[float]] = defaultdict(list)
    for row in calibration_heads:
        alpha_by_case[row["case_id"]].append(float(row["alpha"]))
    calibrated_summaries = []
    for case in cases:
        case_id = case["case_id"]
        case_rows = calibrated_by_case[case_id]
        direct = sum(float(row["direct_sse"]) for row in case_rows)
        unit = sum(float(row["unit_256_sse"]) for row in case_rows)
        calibrated = sum(float(row["calibrated_sse"]) for row in case_rows)
        calibrated_summaries.append(
            (
                case_id,
                case["skill"],
                100.0 * (1.0 - unit / direct),
                100.0 * (1.0 - calibrated / direct),
                calibrated < unit,
                statistics.median(alpha_by_case[case_id]),
            )
        )
    calibrated_go = all(item[3] > 0 for item in calibrated_summaries)
    fidelity_by_case: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in fidelity:
        fidelity_by_case[(row["component"], row["case_id"])].append(row)
    fidelity_summaries: dict[str, list[dict[str, float | str]]] = {
        "K": [],
        "V": [],
    }
    for component in ("K", "V"):
        for case in cases:
            case_rows = fidelity_by_case[(component, case["case_id"])]
            full_sq_norm = sum(float(row["full_sq_norm"]) for row in case_rows)
            item: dict[str, float | str] = {
                "skill": case["skill"],
                "token_count": int(case_rows[0]["token_count"]),
            }
            for method in ("direct", "unit", "calibrated"):
                sse = sum(
                    float(row[f"{method}_to_full_sse"]) for row in case_rows
                )
                layer_cosines = [
                    float(row[f"{method}_to_full_cosine_mean"])
                    for row in case_rows
                ]
                item[f"{method}_normalized_l2"] = math.sqrt(sse / full_sq_norm)
                item[f"{method}_cosine_mean"] = statistics.mean(layer_cosines)
                item[f"{method}_cosine_median"] = statistics.median(layer_cosines)
            fidelity_summaries[component].append(item)

    cosine_gates = {
        component: all(
            float(item["calibrated_cosine_mean"])
            > float(item["direct_cosine_mean"])
            for item in fidelity_summaries[component]
        )
        for component in ("K", "V")
    }
    normalized_l2_gates = {
        component: all(
            float(item["calibrated_normalized_l2"])
            < float(item["direct_normalized_l2"])
            for item in fidelity_summaries[component]
        )
        for component in ("K", "V")
    }

    table = "\n".join(
        f"| {skill} | {unit_gain:.2f}% | {calibrated_gain:.2f}% | {alpha:.3f} |"
        for _, skill, unit_gain, calibrated_gain, _, alpha in calibrated_summaries
    )
    fidelity_tables = {}
    for component in ("K", "V"):
        fidelity_tables[component] = "\n".join(
            "| {skill} | {direct_cos:.6f} | {calibrated_cos:.6f} | {cos_delta:+.6f} | "
            "{direct_l2:.6f} | {calibrated_l2:.6f} | {l2_reduction:+.2f}% |".format(
                skill=item["skill"],
                direct_cos=float(item["direct_cosine_mean"]),
                calibrated_cos=float(item["calibrated_cosine_mean"]),
                cos_delta=float(item["calibrated_cosine_mean"])
                - float(item["direct_cosine_mean"]),
                direct_l2=float(item["direct_normalized_l2"]),
                calibrated_l2=float(item["calibrated_normalized_l2"]),
                l2_reduction=100.0
                * (
                    1.0
                    - float(item["calibrated_normalized_l2"])
                    / float(item["direct_normalized_l2"])
                ),
            )
            for item in fidelity_summaries[component]
        )
    summary = f"""# Context-free Skill KV residual structure

## 结论

在线完整重计算的 Full KV 始终作为 ground truth。Direct、Unit-256 与 Calibrated 的所有误差和相似度均相对于 Full 计算。原始 Unit-256 K squared-error improvement gate 为 **{'Go' if unit_go else 'No-Go'}**；独立幅值校准的 K squared-error improvement gate 为 **{'Go' if calibrated_go else 'No-Go'}**。更直接的 fidelity 结果为：K cosine improvement gate **{'Go' if cosine_gates['K'] else 'No-Go'}**、K normalized-L2 improvement gate **{'Go' if normalized_l2_gates['K'] else 'No-Go'}**；V cosine improvement gate **{'Go' if cosine_gates['V'] else 'No-Go'}**、V normalized-L2 improvement gate **{'Go' if normalized_l2_gates['V'] else 'No-Go'}**。这些 gate 只表示相对 Direct-to-Full baseline 是否改善，不等价于已达到 Full fidelity。

| Skill | Unit-256 K error reduction | Calibrated K error reduction | Median alpha |
|---|---:|---:|---:|
{table}

### K fidelity to Full

| Skill | Direct cosine | Calibrated cosine | Cosine delta | Direct normalized L2 | Calibrated normalized L2 | L2 reduction |
|---|---:|---:|---:|---:|---:|---:|
{fidelity_tables['K']}

### V fidelity to Full

| Skill | Direct cosine | Calibrated cosine | Cosine delta | Direct normalized L2 | Calibrated normalized L2 | L2 reduction |
|---|---:|---:|---:|---:|---:|---:|
{fidelity_tables['V']}

## 实验方法

- 离线 KV 来自 `/mnt/Large_Language_Model_Lab_1/wsh/skill_save_pool/Qwen3-14B/<skill>/kv`，Skill 在位置 0、无在线系统提示词、用户任务或交互历史的条件下计算。
- 在线真值由真实 OpenHands SkillTool 流程产生；第二次 LLM 请求包含完整 `<context_segment>` 后，通过 `lmcache_segmentia_save` 保存 Skill span 的完整重计算 KV。
- 每个 task 使用独立 vLLM 服务，服务重启边界为 `(online_full, task)`，因此不存在跨 task prefix-cache 污染。
- 离线 Key 先由位置 `[0,S)` RoPE 对齐到在线绝对位置，再计算上下文残差。
- 离线 Value 不做 RoPE，直接与在线完整重计算 Value 比较。
- 前缀预算为 32、64、128、256；所有预算均在共同隐藏后缀 `[256,S)` 上评价，避免改变评价 token 集合。
- 校准实验用 `[0,128)` 的平均残差定义方向，用 `[128,256)` 的残差以闭式最小二乘求每层、每 KV head 的缩放系数；方向与幅值估计均不读取 `[256,S)`。
- cosine 在每个layer、suffix token和KV head的128维向量上计算，再先对token/head求均值、最后对40层作case内宏平均。normalized L2由case内全部40层的总squared error除以Full总平方范数后开方。

## 图表

- `figures/prefix_correction_gain.png`：不同连续前缀预算的后缀 Key 误差改善。
- `figures/prefix_tail_offset_cosine.png`：Prefix-256 估计偏移与隐藏后缀真实平均偏移在所有层和 KV head 上的 cosine。
- `figures/calibrated_method_comparison.png`：Direct、Unit-256 与独立幅值校准三种方法的归一化后缀 Key 误差。
- `figures/calibrated_layer_gain.png`：每个 Skill、每层的校准后 Key 误差改善。
- `figures/kv_cosine_to_full.png`：K/V的Direct、Unit-256和Calibrated到Full的平均cosine。
- `figures/kv_normalized_l2_to_full.png`：K/V三种方法到Full的归一化L2差距。
- `tables/layer_metrics.csv`：逐 case、layer、budget 的 Direct 与校正后 squared error。
- `tables/calibrated_layer_metrics.csv`：逐 case、layer 的 Direct、Unit-256 和 calibrated squared error。
- `tables/fidelity_layer_metrics.csv`：逐case、layer、K/V component的to-Full cosine、normalized L2和显式to-Full SSE。
- `data/head_cosines.csv`：逐 case、layer、KV head、budget 的偏移 cosine。
- `data/calibration_heads.csv`：逐 case、layer、KV head 的缩放系数和估计区间向量范数。
- `data/value_calibration_heads.csv`：Value的逐case、layer、KV head缩放系数和估计区间向量范数。

## 证据边界

- 误差和 cosine 是 KV 数值指标，不等价于生成质量、工具调用成功率或端到端延迟。
- case/Skill 是实验重复单位；layer 和 head 只是模型内部观测，不能作为独立样本推断显著性。
- 本轮只验证离线分析中的共享偏移与连续前缀纠错机会，尚未验证在线纠错 kernel。
"""
    (args.result_dir / "summary.md").write_text(summary, encoding="utf-8")

    manifest_rows = [
        ["artifact", "source"],
        ["summary.md", str(args.run_dir.resolve())],
        ["tables/layer_metrics.csv", str((analysis / "layer_metrics.csv").resolve())],
        ["data/head_cosines.csv", str((analysis / "head_cosines.csv").resolve())],
        ["tables/calibrated_layer_metrics.csv", str((analysis / "calibrated_layer_metrics.csv").resolve())],
        ["data/calibration_heads.csv", str((analysis / "calibration_heads.csv").resolve())],
        ["tables/fidelity_layer_metrics.csv", str((analysis / "fidelity_layer_metrics.csv").resolve())],
        ["data/value_calibration_heads.csv", str((analysis / "value_calibration_heads.csv").resolve())],
        ["figures/prefix_correction_gain.png", str((figures / "prefix_correction_gain.png").resolve())],
        ["figures/prefix_tail_offset_cosine.png", str((figures / "prefix_tail_offset_cosine.png").resolve())],
        ["figures/calibrated_method_comparison.png", str((figures / "calibrated_method_comparison.png").resolve())],
        ["figures/calibrated_layer_gain.png", str((figures / "calibrated_layer_gain.png").resolve())],
        ["figures/kv_cosine_to_full.png", str((figures / "kv_cosine_to_full.png").resolve())],
        ["figures/kv_normalized_l2_to_full.png", str((figures / "kv_normalized_l2_to_full.png").resolve())],
    ]
    with (args.result_dir / "source_manifest.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        csv.writer(handle).writerows(manifest_rows)
    print(
        f"[published] k_cosine={'Go' if cosine_gates['K'] else 'No-Go'} "
        f"k_l2={'Go' if normalized_l2_gates['K'] else 'No-Go'} "
        f"v_cosine={'Go' if cosine_gates['V'] else 'No-Go'} "
        f"v_l2={'Go' if normalized_l2_gates['V'] else 'No-Go'} "
        f"output={args.result_dir}"
    )


if __name__ == "__main__":
    main()
