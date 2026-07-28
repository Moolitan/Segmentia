#!/usr/bin/env python3
"""Describe contextual Skill KV residuals for one validated capture pair.

This is an M0 diagnostic, not a learned correction method.  It aligns the
source K cache to the target absolute positions, measures source-to-target
residuals, evaluates boundary-estimated uniform-offset baselines, and reports
oracle token-feature singular spectra for each layer/head.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch


DEFAULT_MODEL_CONFIG = Path(
    "/mnt/Large_Language_Model_Lab_1/llm_models/"
    "Qwen3-14B/Qwen/Qwen3-14B/config.json"
)
DEFAULT_RANKS = (1, 2, 4, 8, 16, 32, 64)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    if any(list(row) != fieldnames for row in rows):
        raise ValueError(f"CSV rows have inconsistent schemas: {path}")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _require_int_list(value: Any, name: str) -> list[int]:
    if not isinstance(value, list) or not value or not all(
        isinstance(item, int) for item in value
    ):
        raise ValueError(f"{name} must be a non-empty integer list")
    return value


def decode_positions(payload: Any, expected_tokens: int) -> torch.Tensor:
    if not isinstance(payload, dict):
        raise ValueError("cached_positions must be an object")
    kind = payload.get("kind")
    if kind == "range":
        start = payload.get("start")
        length = payload.get("length")
        if not isinstance(start, int) or length != expected_tokens:
            raise ValueError(f"Invalid cached position range: {payload}")
        return torch.arange(start, start + length, dtype=torch.int64)
    if kind == "list":
        values = _require_int_list(payload.get("values"), "cached_positions.values")
        if len(values) != expected_tokens:
            raise ValueError("cached_positions length does not match token count")
        return torch.tensor(values, dtype=torch.int64)
    raise ValueError(f"Unsupported cached_positions encoding: {kind!r}")


def load_sidecar(data_path: Path) -> dict[str, Any]:
    sidecar_path = data_path.with_name(f"{data_path.name}.meta.json")
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if payload.get("data_file") != data_path.name:
        raise ValueError(f"Sidecar points at another data file: {sidecar_path}")
    if payload.get("dtype") != "bfloat16":
        raise ValueError(f"Only bfloat16 captures are supported: {sidecar_path}")
    if payload.get("memory_format") != "KV_2TD":
        raise ValueError(f"Expected KV_2TD capture: {sidecar_path}")
    shape = _require_int_list(payload.get("shape"), "shape")
    expected_size = (
        math.prod(shape) * torch.tensor([], dtype=torch.bfloat16).element_size()
    )
    if (
        payload.get("size") != expected_size
        or data_path.stat().st_size != expected_size
    ):
        raise ValueError(f"Raw KV size does not match sidecar: {data_path}")
    return payload


def require_populated_raw_kv(data_path: Path, chunk_bytes: int = 1024 * 1024) -> None:
    with data_path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            if any(chunk):
                return
    raise ValueError(f"Raw KV file contains only zero bytes: {data_path}")


def read_raw_kv(data_path: Path, shape: list[int]) -> torch.Tensor:
    numel = math.prod(shape)
    tensor = torch.from_file(
        str(data_path), shared=False, size=numel, dtype=torch.bfloat16
    )
    return tensor.reshape(shape)


def relocate_neox_rope(
    key: torch.Tensor,
    old_positions: torch.Tensor,
    new_positions: torch.Tensor,
    rope_theta: float,
) -> torch.Tensor:
    """Move post-RoPE K from old positions to new positions in float32."""

    if key.ndim != 3:
        raise ValueError(f"Expected [tokens, heads, head_dim], got {tuple(key.shape)}")
    tokens, _, head_dim = key.shape
    if head_dim % 2:
        raise ValueError("NeoX RoPE requires an even head dimension")
    if old_positions.shape != (tokens,) or new_positions.shape != (tokens,):
        raise ValueError("Position vectors must match the K token dimension")
    key = key.to(torch.float32)
    half = head_dim // 2
    exponent = torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
    inv_freq = torch.pow(torch.tensor(float(rope_theta)), -exponent)
    phase = (new_positions - old_positions).to(torch.float32).unsqueeze(1) * inv_freq
    cosine = phase.cos().unsqueeze(1)
    sine = phase.sin().unsqueeze(1)
    first = key[..., :half]
    second = key[..., half:]
    return torch.cat(
        (first * cosine - second * sine, second * cosine + first * sine),
        dim=-1,
    )


def tensor_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    prediction = prediction.to(torch.float64)
    target = target.to(torch.float64)
    error = prediction - target
    error_sq = float(torch.sum(error * error))
    target_sq = float(torch.sum(target * target))
    pred_sq = float(torch.sum(prediction * prediction))
    dot = float(torch.sum(prediction * target))
    return {
        "squared_error": error_sq,
        "target_squared_norm": target_sq,
        "prediction_squared_norm": pred_sq,
        "dot": dot,
        "relative_l2": math.sqrt(error_sq / max(target_sq, 1e-30)),
        "rmse": math.sqrt(error_sq / error.numel()),
        "cosine": dot / math.sqrt(max(pred_sq * target_sq, 1e-30)),
    }


def offset_predictions(
    source: torch.Tensor,
    target: torch.Tensor,
    boundary_tokens: int,
) -> dict[str, torch.Tensor]:
    """Fit two AgentKVShift-style offsets on B0 and apply them to B1."""

    residual_boundary = target[:boundary_tokens] - source[:boundary_tokens]
    shared_offset = residual_boundary.mean(dim=(0, 1), keepdim=True)
    headwise_offset = residual_boundary.mean(dim=0, keepdim=True)
    suffix = source[boundary_tokens:]
    return {
        "direct": suffix,
        "layer_shared_offset": suffix + shared_offset,
        "headwise_offset": suffix + headwise_offset,
    }


def singular_spectrum_metrics(
    residual: torch.Tensor, ranks: Iterable[int]
) -> dict[str, float]:
    """Return oracle compressibility of one [tokens, head_dim] residual."""

    singular = torch.linalg.svdvals(residual.to(torch.float32))
    energy = singular.square()
    total = float(energy.sum())
    if total <= 0:
        result = {"effective_rank": 0.0, "stable_rank": 0.0}
        for rank in ranks:
            result[f"energy_rank_{rank}"] = 1.0
            result[f"oracle_relative_error_rank_{rank}"] = 0.0
        return result
    probabilities = energy / total
    positive = probabilities[probabilities > 0]
    effective_rank = float(torch.exp(-(positive * positive.log()).sum()))
    stable_rank = total / max(float(energy[0]), 1e-30)
    result = {
        "effective_rank": effective_rank,
        "stable_rank": stable_rank,
    }
    for rank in ranks:
        kept = float(energy[: min(rank, energy.numel())].sum()) / total
        result[f"energy_rank_{rank}"] = kept
        result[f"oracle_relative_error_rank_{rank}"] = math.sqrt(max(0.0, 1.0 - kept))
    return result


def _layer_id(path: Path) -> int:
    try:
        return int(path.stem.rsplit("@", 1)[1])
    except (IndexError, ValueError) as error:
        raise ValueError(f"Cannot parse layer id from {path.name}") from error


def validate_case(
    manifest: dict[str, Any], model_config: dict[str, Any]
) -> tuple[list[Path], list[Path], int, int, int]:
    if manifest.get("schema_version") != 2 or manifest.get("status") != "completed":
        raise ValueError("Expected a completed capture manifest with schema_version=2")
    shape = _require_int_list(manifest.get("kv_shape_per_layer"), "kv_shape_per_layer")
    if len(shape) != 4 or shape[0] != 2:
        raise ValueError(f"Unexpected per-layer KV shape: {shape}")
    _, tokens, kv_heads, head_dim = shape
    if kv_heads != model_config.get("num_key_value_heads"):
        raise ValueError("Capture KV-head count differs from model config")
    if head_dim != model_config.get("head_dim"):
        raise ValueError("Capture head_dim differs from model config")
    if model_config.get("rope_scaling") is not None:
        raise ValueError("This diagnostic does not support scaled RoPE")

    source_files = [Path(path) for path in manifest["shared_storage"]["files"]]
    target_files = [Path(path) for path in manifest["target_full_storage"]["files"]]
    source_files.sort(key=_layer_id)
    target_files.sort(key=_layer_id)
    source_layers = [_layer_id(path) for path in source_files]
    target_layers = [_layer_id(path) for path in target_files]
    if source_layers != target_layers or source_layers != list(
        range(len(source_layers))
    ):
        raise ValueError("Source and target layers are not identical contiguous ranges")

    reuse_event = manifest.get("target", {}).get("reuse_event", {})
    boundary_tokens = reuse_event.get("lookup_cursor", -1) - reuse_event.get(
        "lookup_start", -1
    )
    if not 0 < boundary_tokens < tokens:
        raise ValueError(
            f"Invalid B0 boundary length derived from manifest: {boundary_tokens}"
        )
    return source_files, target_files, tokens, kv_heads, boundary_tokens


def _summarize_baselines(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["kv_type"], row["baseline"])].append(row)
    summary: list[dict[str, Any]] = []
    for (kv_type, baseline), values in sorted(grouped.items()):
        summary.append(
            {
                "kv_type": kv_type,
                "baseline": baseline,
                "layers": len(values),
                "macro_relative_l2": sum(row["relative_l2"] for row in values)
                / len(values),
                "macro_rmse": sum(row["rmse"] for row in values) / len(values),
                "macro_cosine": sum(row["cosine"] for row in values) / len(values),
                "macro_improvement_vs_direct": sum(
                    row["improvement_vs_direct"] for row in values
                )
                / len(values),
            }
        )
    return summary


def make_plots(
    output_dir: Path,
    layer_rows: list[dict[str, Any]],
    head_rows: list[dict[str, Any]],
    spectrum_rows: list[dict[str, Any]],
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    figure_paths: list[str] = []

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    for axis, kv_type in zip(axes, ("K", "V"), strict=True):
        subset = [row for row in layer_rows if row["kv_type"] == kv_type]
        for baseline in ("direct", "layer_shared_offset", "headwise_offset"):
            rows = [row for row in subset if row["baseline"] == baseline]
            axis.plot(
                [row["layer"] for row in rows],
                [row["relative_l2"] for row in rows],
                marker="o",
                markersize=2.5,
                linewidth=1.2,
                label=baseline,
            )
        axis.set_title(f"{kv_type} suffix prediction")
        axis.set_xlabel("Layer")
        axis.set_ylabel("Relative L2 error")
        axis.grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = figures_dir / "baseline_relative_l2.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    figure_paths.append(str(path.resolve()))

    for table_rows, value, filename, title in (
        (
            head_rows,
            "direct_relative_l2",
            "direct_residual_heatmap.png",
            "Direct reuse residual",
        ),
        (
            spectrum_rows,
            "energy_rank_8",
            "rank8_energy_heatmap.png",
            "Oracle rank-8 residual energy",
        ),
    ):
        fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
        for axis, kv_type in zip(axes, ("K", "V"), strict=True):
            rows = [row for row in table_rows if row["kv_type"] == kv_type]
            layers = max(int(row["layer"]) for row in rows) + 1
            heads = max(int(row["head"]) for row in rows) + 1
            matrix = np.full((layers, heads), np.nan)
            for row in rows:
                matrix[int(row["layer"]), int(row["head"])] = float(row[value])
            image = axis.imshow(matrix, aspect="auto", origin="lower", cmap="viridis")
            axis.set_title(kv_type)
            axis.set_xlabel("KV head")
            axis.set_ylabel("Layer")
            fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        fig.suptitle(title)
        fig.tight_layout()
        path = figures_dir / filename
        fig.savefig(path, dpi=180)
        plt.close(fig)
        figure_paths.append(str(path.resolve()))
    return figure_paths


def analyze_case(
    case_dir: Path,
    model_config_path: Path,
    output_dir: Path,
    ranks: tuple[int, ...] = DEFAULT_RANKS,
    plots: bool = True,
) -> dict[str, Any]:
    manifest_path = case_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    model_config = json.loads(model_config_path.read_text(encoding="utf-8"))
    source_files, target_files, tokens, kv_heads, boundary_tokens = validate_case(
        manifest, model_config
    )
    head_dim = int(model_config["head_dim"])
    rope_theta = float(model_config["rope_theta"])

    layer_rows: list[dict[str, Any]] = []
    head_rows: list[dict[str, Any]] = []
    spectrum_rows: list[dict[str, Any]] = []
    token_error_sum = {
        "K": torch.zeros(tokens - boundary_tokens, dtype=torch.float64),
        "V": torch.zeros(tokens - boundary_tokens, dtype=torch.float64),
    }
    token_target_sum = {
        "K": torch.zeros(tokens - boundary_tokens, dtype=torch.float64),
        "V": torch.zeros(tokens - boundary_tokens, dtype=torch.float64),
    }

    for source_path, target_path in zip(source_files, target_files, strict=True):
        layer = _layer_id(source_path)
        source_meta = load_sidecar(source_path)
        target_meta = load_sidecar(target_path)
        require_populated_raw_kv(source_path)
        require_populated_raw_kv(target_path)
        source_shape = _require_int_list(source_meta["shape"], "source shape")
        target_shape = _require_int_list(target_meta["shape"], "target shape")
        expected_shape = [2, tokens, kv_heads * head_dim]
        if source_shape != expected_shape or target_shape != expected_shape:
            raise ValueError(f"Layer {layer} shape mismatch")
        source_key = str(source_meta.get("cache_key", ""))
        target_key = str(target_meta.get("cache_key", ""))
        if source_key.rsplit("@", 1)[0] != target_key.rsplit("@", 1)[0]:
            raise ValueError(f"Layer {layer} source/target cache identities differ")
        source_positions = decode_positions(source_meta["cached_positions"], tokens)
        target_positions = decode_positions(target_meta["cached_positions"], tokens)
        source_raw = read_raw_kv(source_path, source_shape)
        target_raw = read_raw_kv(target_path, target_shape)
        source_k = source_raw[0].reshape(tokens, kv_heads, head_dim)
        source_v = source_raw[1].reshape(tokens, kv_heads, head_dim).to(torch.float32)
        target_k = target_raw[0].reshape(tokens, kv_heads, head_dim).to(torch.float32)
        target_v = target_raw[1].reshape(tokens, kv_heads, head_dim).to(torch.float32)
        aligned_source_k = relocate_neox_rope(
            source_k, source_positions, target_positions, rope_theta
        )

        for kv_type, source, target in (
            ("K", aligned_source_k, target_k),
            ("V", source_v, target_v),
        ):
            suffix_target = target[boundary_tokens:]
            predictions = offset_predictions(source, target, boundary_tokens)
            direct_metrics = tensor_metrics(predictions["direct"], suffix_target)
            for baseline, prediction in predictions.items():
                metrics = tensor_metrics(prediction, suffix_target)
                layer_rows.append(
                    {
                        "layer": layer,
                        "kv_type": kv_type,
                        "baseline": baseline,
                        "boundary_tokens": boundary_tokens,
                        "eval_tokens": tokens - boundary_tokens,
                        "relative_l2": metrics["relative_l2"],
                        "rmse": metrics["rmse"],
                        "cosine": metrics["cosine"],
                        "improvement_vs_direct": 1.0
                        - metrics["squared_error"]
                        / max(direct_metrics["squared_error"], 1e-30),
                    }
                )

            residual = suffix_target - predictions["direct"]
            token_error_sum[kv_type] += residual.to(torch.float64).square().sum(
                dim=(1, 2)
            )
            token_target_sum[kv_type] += suffix_target.to(torch.float64).square().sum(
                dim=(1, 2)
            )
            for head in range(kv_heads):
                head_direct = tensor_metrics(
                    predictions["direct"][:, head], suffix_target[:, head]
                )
                head_rows.append(
                    {
                        "layer": layer,
                        "kv_type": kv_type,
                        "head": head,
                        "direct_relative_l2": head_direct["relative_l2"],
                        "direct_rmse": head_direct["rmse"],
                        "direct_cosine": head_direct["cosine"],
                    }
                )
                spectrum_rows.append(
                    {
                        "layer": layer,
                        "kv_type": kv_type,
                        "head": head,
                        **singular_spectrum_metrics(residual[:, head], ranks),
                    }
                )

    token_rows: list[dict[str, Any]] = []
    for kv_type in ("K", "V"):
        for offset, (error_sq, target_sq) in enumerate(
            zip(token_error_sum[kv_type], token_target_sum[kv_type], strict=True)
        ):
            token_rows.append(
                {
                    "kv_type": kv_type,
                    "skill_token_index": boundary_tokens + offset,
                    "suffix_token_index": offset,
                    "direct_relative_l2": math.sqrt(
                        float(error_sq) / max(float(target_sq), 1e-30)
                    ),
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = output_dir / "tables"
    write_csv(tables_dir / "layer_baselines.csv", layer_rows)
    write_csv(tables_dir / "head_residuals.csv", head_rows)
    write_csv(tables_dir / "oracle_spectra.csv", spectrum_rows)
    write_csv(tables_dir / "token_residuals.csv", token_rows)
    baseline_summary = _summarize_baselines(layer_rows)
    figure_paths = (
        make_plots(output_dir, layer_rows, head_rows, spectrum_rows) if plots else []
    )
    summary = {
        "schema_version": 1,
        "status": "descriptive_single_case",
        "case_id": manifest["case_id"],
        "skill": manifest["skill"],
        "source_task": manifest["source"]["task"],
        "target_task": manifest["target"]["task"],
        "layers": len(source_files),
        "tokens": tokens,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "boundary_tokens": boundary_tokens,
        "evaluated_suffix_tokens": tokens - boundary_tokens,
        "k_alignment": "source_post_rope_relocated_to_target_absolute_positions",
        "rope_theta": rope_theta,
        "ranks": list(ranks),
        "baseline_macro": baseline_summary,
        "interpretation_limits": [
            "One source-target pair cannot establish cross-request generalization.",
            "Singular spectra are oracle compressibility of this observed "
            "residual, not held-out prediction accuracy.",
            "Uniform offsets are estimated only from the actual local B0 "
            "boundary and evaluated on B1.",
        ],
        "artifacts": {
            "manifest": str(manifest_path.resolve()),
            "model_config": str(model_config_path.resolve()),
            "tables": [
                str((tables_dir / name).resolve())
                for name in (
                    "layer_baselines.csv",
                    "head_residuals.csv",
                    "oracle_spectra.csv",
                    "token_residuals.csv",
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
    parser.add_argument("--ranks", type=int, nargs="+", default=list(DEFAULT_RANKS))
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ranks = tuple(sorted(set(args.ranks)))
    if not ranks or ranks[0] <= 0:
        raise ValueError("All requested ranks must be positive")
    output_dir = args.output_dir or args.case_dir / "analysis"
    summary = analyze_case(
        case_dir=args.case_dir,
        model_config_path=args.model_config,
        output_dir=output_dir,
        ranks=ranks,
        plots=not args.no_plots,
    )
    print(
        f"[analyzed] case={summary['case_id']} boundary={summary['boundary_tokens']} "
        f"suffix={summary['evaluated_suffix_tokens']} output={output_dir}"
    )


if __name__ == "__main__":
    main()
