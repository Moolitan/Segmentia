#!/usr/bin/env python3
"""Validate online length-gated continuous-prefix K correction."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import torch

CAPTURE_SCRIPT_DIR = Path(__file__).resolve().parents[1] / "cross_request_kv_capture"
if str(CAPTURE_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(CAPTURE_SCRIPT_DIR))

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
from validate_capture import load_events, load_profile_events, request_event
from validate_distributed_anchor import _validate_reuse_only


PREFIX_TOKENS = 256
CALIBRATION_START = 132
CALIBRATION_END = 256


def _single_prefix_capture(path: Path) -> dict[str, Any]:
    captures = sorted(path.glob("*.pt"))
    if len(captures) != 1:
        raise ValueError(f"Expected one prefix capture in {path}, got {len(captures)}")
    payload = torch.load(captures[0], map_location="cpu", weights_only=True)
    if payload.get("schema_version") != 4:
        raise ValueError("Prefix capture must use schema_version=4")
    if payload.get("correction_mode") != "prefix_k_headwise":
        raise ValueError("Prefix capture correction mode is invalid")
    payload["capture_path"] = str(captures[0].resolve())
    return payload


def _layer_capture(capture: dict[str, Any], layer: int) -> dict[str, torch.Tensor]:
    value = capture["layers"].get(layer)
    if value is None:
        value = capture["layers"].get(str(layer))
    if value is None:
        raise ValueError(f"Prefix capture is missing layer {layer}")
    return value


def analyze_case(
    base_case_dir: Path,
    prefix_case_dir: Path,
    model_config: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    manifest = json.loads((base_case_dir / "manifest.json").read_text(encoding="utf-8"))
    source_files, target_files, tokens, kv_heads, _ = validate_case(
        manifest, model_config
    )
    capture = _single_prefix_capture(
        prefix_case_dir / "target_reuse" / "prefix_correction"
    )
    reuse_validation = _validate_reuse_only(
        prefix_case_dir, manifest, len(source_files)
    )
    request = json.loads(
        (prefix_case_dir / "target_reuse" / "request.json").read_text(
            encoding="utf-8"
        )
    )
    if request.get("correction_mode") != "prefix_k_headwise":
        raise ValueError("Target-reuse request did not enable prefix correction")
    segment_start = int(capture["segment_start"])
    calibration_start = int(capture["calibration_start"])
    calibration_end = int(capture["calibration_end"])
    load_start = int(capture["load_start"])
    if (
        calibration_start - segment_start != CALIBRATION_START
        or calibration_end - segment_start != CALIBRATION_END
        or not calibration_end <= load_start < int(capture["total_tokens"])
    ):
        raise ValueError("Prefix capture does not match the frozen aligned policy")
    relative_load_start = load_start - segment_start
    if relative_load_start < PREFIX_TOKENS or relative_load_start >= tokens:
        raise ValueError("Aligned prefix boundary falls outside the cached Skill")

    head_dim = int(model_config["head_dim"])
    rope_theta = float(model_config["rope_theta"])
    rows: list[dict[str, Any]] = []
    offset_rows: list[dict[str, Any]] = []
    for source_path, target_path in zip(source_files, target_files, strict=True):
        layer = int(source_path.stem.rsplit("@", 1)[1])
        source_meta = load_sidecar(source_path)
        target_meta = load_sidecar(target_path)
        require_populated_raw_kv(source_path)
        require_populated_raw_kv(target_path)
        source_raw = read_raw_kv(source_path, source_meta["shape"])
        target_raw = read_raw_kv(target_path, target_meta["shape"])
        source_k = relocate_neox_rope(
            source_raw[0].reshape(tokens, kv_heads, head_dim),
            decode_positions(source_meta["cached_positions"], tokens),
            decode_positions(target_meta["cached_positions"], tokens),
            rope_theta,
        )
        target_k = target_raw[0].reshape(tokens, kv_heads, head_dim).to(torch.float32)
        direct = source_k[relative_load_start:]
        target = target_k[relative_load_start:]
        direct_error = float(tensor_metrics(direct, target)["squared_error"])

        captured = _layer_capture(capture, layer)
        cached_calibration = captured["cached_calibration_k"].to(torch.float32).reshape(
            CALIBRATION_END - CALIBRATION_START, kv_heads, head_dim
        )
        actual_calibration = captured["actual_calibration_k"].to(torch.float32).reshape(
            CALIBRATION_END - CALIBRATION_START, kv_heads, head_dim
        )
        online_offset = captured["global_offset"].to(torch.float32).reshape(
            kv_heads, head_dim
        )
        derived_offset = (actual_calibration - cached_calibration).mean(dim=0)
        if not torch.allclose(online_offset, derived_offset, rtol=0.0, atol=1e-6):
            raise ValueError(f"Layer {layer} captured prefix offset is inconsistent")
        offline_offset = (
            target_k[CALIBRATION_START:CALIBRATION_END]
            - source_k[CALIBRATION_START:CALIBRATION_END]
        ).mean(dim=0)
        cached_reference = source_k[CALIBRATION_START:CALIBRATION_END]
        actual_reference = target_k[CALIBRATION_START:CALIBRATION_END]
        cached_reference_l2 = float(
            tensor_metrics(cached_calibration, cached_reference)["relative_l2"]
        )
        actual_reference_l2 = float(
            tensor_metrics(actual_calibration, actual_reference)["relative_l2"]
        )
        offset_cosine = float(
            torch.nn.functional.cosine_similarity(
                online_offset.flatten(), offline_offset.flatten(), dim=0
            )
        )
        offset_rows.append(
            {
                "case_id": manifest["case_id"],
                "skill": manifest["skill"],
                "layer": layer,
                "online_offline_offset_cosine": offset_cosine,
                "online_offset_norm": float(online_offset.norm()),
                "offline_offset_norm": float(offline_offset.norm()),
                "cached_reference_relative_l2": cached_reference_l2,
                "actual_reference_relative_l2": actual_reference_l2,
            }
        )
        for variant, offset in (
            ("direct", None),
            ("online_prefix_k", online_offset),
            ("offline_frozen_prefix_k", offline_offset),
        ):
            prediction = direct if offset is None else direct + offset.unsqueeze(0)
            metrics = tensor_metrics(prediction, target)
            error = float(metrics["squared_error"])
            rows.append(
                {
                    "case_id": manifest["case_id"],
                    "skill": manifest["skill"],
                    "tokens": tokens,
                    "aligned_prefix_tokens": relative_load_start,
                    "reused_tokens": tokens - relative_load_start,
                    "layer": layer,
                    "variant": variant,
                    "squared_error": error,
                    "direct_squared_error": direct_error,
                    "improvement_vs_direct": (
                        0.0 if direct_error == 0.0 else 1.0 - error / direct_error
                    ),
                    "relative_l2": metrics["relative_l2"],
                    "rmse": metrics["rmse"],
                    "cosine": metrics["cosine"],
                }
            )

    metadata = {
        "case_id": manifest["case_id"],
        "skill": manifest["skill"],
        "tokens": tokens,
        "layers": len(source_files),
        "aligned_prefix_tokens": relative_load_start,
        "reused_tokens": tokens - relative_load_start,
        "reuse_fraction": (tokens - relative_load_start) / tokens,
        "base_case_dir": str(base_case_dir.resolve()),
        "prefix_case_dir": str(prefix_case_dir.resolve()),
        "capture_path": capture["capture_path"],
        "reuse_validation": reuse_validation,
    }
    return metadata, rows, offset_rows


def validate_short_fallback(short_case_dir: Path) -> dict[str, Any]:
    request_path = short_case_dir / "target_reuse" / "request.json"
    log_path = short_case_dir / "target_reuse" / "vllm.log"
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if request.get("status") != "completed" or request.get("correction_mode") != "prefix_k_headwise":
        raise ValueError("Short-control request is incomplete or has the wrong policy")
    fallback = request_event(log_path, request, "segmentia_prefix_length_fallback")
    if fallback.get("phase") != "full_local" or fallback.get("reusable_tokens", 256) >= 256:
        raise ValueError("Short-control request did not take the length fallback")
    response_id = request["response_id"]
    apply_events = [
        event
        for event in load_events(log_path)
        if event.get("event") == "segmentia_lookup_external_apply"
        and isinstance(event.get("request_id"), str)
        and (
            event["request_id"] == response_id
            or event["request_id"].startswith(f"{response_id}-")
        )
    ]
    reads = [
        event
        for event in load_profile_events(log_path)
        if event.get("event") == "segmentia_storage_read"
        and isinstance(event.get("request_id"), str)
        and (
            event["request_id"] == response_id
            or event["request_id"].startswith(f"{response_id}-")
        )
    ]
    captures = list(
        (short_case_dir / "target_reuse" / "prefix_correction").glob("*.pt")
    )
    if apply_events or reads or captures:
        raise ValueError("Short-control fallback unexpectedly applied external KV")
    return {
        "request_path": str(request_path.resolve()),
        "log_path": str(log_path.resolve()),
        "fallback_event": fallback,
        "external_apply_events": 0,
        "ssd_read_events": 0,
        "captures": 0,
    }


def summarize_pairs(cases: list[dict[str, Any]], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for case in cases:
        case_rows = [row for row in rows if row["case_id"] == case["case_id"]]
        direct_error = sum(
            row["squared_error"] for row in case_rows if row["variant"] == "direct"
        )
        for variant in ("direct", "online_prefix_k", "offline_frozen_prefix_k"):
            selected = [row for row in case_rows if row["variant"] == variant]
            error = sum(row["squared_error"] for row in selected)
            summaries.append(
                {
                    "case_id": case["case_id"],
                    "skill": case["skill"],
                    "tokens": case["tokens"],
                    "aligned_prefix_tokens": case["aligned_prefix_tokens"],
                    "reused_tokens": case["reused_tokens"],
                    "reuse_fraction": case["reuse_fraction"],
                    "variant": variant,
                    "aggregate_squared_error": error,
                    "aggregate_improvement_vs_direct": (
                        0.0 if direct_error == 0.0 else 1.0 - error / direct_error
                    ),
                    "improved_layers": sum(
                        row["improvement_vs_direct"] > 0.0 for row in selected
                    ),
                }
            )
    return summaries


def evaluate_gate(
    pair_summary: list[dict[str, Any]],
    offset_rows: list[dict[str, Any]],
    short_fallback: dict[str, Any] | None,
) -> dict[str, Any]:
    online = [row for row in pair_summary if row["variant"] == "online_prefix_k"]
    thresholds = {
        "required_pairs": 3,
        "minimum_positive_pairs": 3,
        "minimum_median_improvement": 0.15,
        "minimum_worst_improvement": 0.05,
        "minimum_median_improved_layers": 30.0,
        "minimum_median_offset_cosine": 0.99,
        "require_short_full_fallback": True,
    }
    if len(online) != thresholds["required_pairs"]:
        return {
            "status": "insufficient_pairs",
            "passed": False,
            "observed_pairs": len(online),
            "thresholds": thresholds,
        }
    improvements = [row["aggregate_improvement_vs_direct"] for row in online]
    improved_layers = [row["improved_layers"] for row in online]
    offset_cosines = [row["online_offline_offset_cosine"] for row in offset_rows]
    short_passed = short_fallback is not None
    median_improvement = statistics.median(improvements)
    median_layers = statistics.median(improved_layers)
    median_offset_cosine = statistics.median(offset_cosines)
    passed = (
        sum(value > 0.0 for value in improvements)
        >= thresholds["minimum_positive_pairs"]
        and median_improvement >= thresholds["minimum_median_improvement"]
        and min(improvements) >= thresholds["minimum_worst_improvement"]
        and median_layers >= thresholds["minimum_median_improved_layers"]
        and median_offset_cosine >= thresholds["minimum_median_offset_cosine"]
        and short_passed
    )
    return {
        "status": "go" if passed else "no_go",
        "passed": passed,
        "observed_pairs": len(online),
        "positive_pairs": sum(value > 0.0 for value in improvements),
        "median_improvement": median_improvement,
        "worst_improvement": min(improvements),
        "median_improved_layers": median_layers,
        "median_offset_cosine": median_offset_cosine,
        "short_full_fallback_passed": short_passed,
        "thresholds": thresholds,
    }


def make_plot(output_dir: Path, pair_summary: list[dict[str, Any]]) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    variants = ("online_prefix_k", "offline_frozen_prefix_k")
    cases = sorted({row["case_id"] for row in pair_summary})
    values = {
        (row["case_id"], row["variant"]): 100 * row["aggregate_improvement_vs_direct"]
        for row in pair_summary
    }
    x = np.arange(len(cases))
    width = 0.36
    figure, axis = plt.subplots(figsize=(max(9, 2.5 * len(cases)), 4.8))
    for index, variant in enumerate(variants):
        axis.bar(
            x + (index - 0.5) * width,
            [values[(case, variant)] for case in cases],
            width,
            label=variant,
        )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(x, cases, rotation=15, ha="right")
    axis.set_ylabel("Aggregate suffix K error improvement vs Direct (%)")
    axis.set_title("Online continuous-prefix K correction")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    path = figures_dir / "prefix_correction_improvement.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return str(path.resolve())


def validate_prefix_correction(
    base_case_dirs: list[Path],
    prefix_case_dirs: list[Path],
    short_case_dir: Path,
    model_config_path: Path,
    output_dir: Path,
    plots: bool = True,
) -> dict[str, Any]:
    if len(base_case_dirs) != len(prefix_case_dirs):
        raise ValueError("Base and prefix case-dir counts must match")
    model_config = json.loads(model_config_path.read_text(encoding="utf-8"))
    cases: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    offset_rows: list[dict[str, Any]] = []
    for base_case, prefix_case in zip(base_case_dirs, prefix_case_dirs, strict=True):
        metadata, case_rows, case_offsets = analyze_case(
            base_case, prefix_case, model_config
        )
        cases.append(metadata)
        rows.extend(case_rows)
        offset_rows.extend(case_offsets)
    short_fallback = validate_short_fallback(short_case_dir)
    pair_summary = summarize_pairs(cases, rows)
    gate = evaluate_gate(pair_summary, offset_rows, short_fallback)
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = output_dir / "tables"
    write_csv(tables_dir / "prefix_correction_layer_metrics.csv", rows)
    write_csv(tables_dir / "prefix_correction_pair_summary.csv", pair_summary)
    write_csv(tables_dir / "prefix_correction_offset_quality.csv", offset_rows)
    figure = make_plot(output_dir, pair_summary) if plots else None
    summary = {
        "schema_version": 1,
        "status": "prefix_correction_validation",
        "cases": cases,
        "short_fallback": short_fallback,
        "method": {
            "nominal_prefix_tokens": PREFIX_TOKENS,
            "calibration": "Skill-relative [132,256)",
            "reuse": "block-aligned [P,E)",
            "correction": "per-layer/head K mean offset; V Direct",
        },
        "prefix_gate": gate,
        "interpretation_limits": [
            "The KV gate does not establish logits, tool/action, or task correctness.",
            "Capture D2H and file I/O are not part of a clean latency measurement.",
            "Two held-out Skills share the same source/target task pair.",
        ],
        "artifacts": {
            "tables": [
                str((tables_dir / "prefix_correction_layer_metrics.csv").resolve()),
                str((tables_dir / "prefix_correction_pair_summary.csv").resolve()),
                str((tables_dir / "prefix_correction_offset_quality.csv").resolve()),
            ],
            "figure": figure,
        },
    }
    atomic_write_json(output_dir / "prefix_correction_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-case-dir", type=Path, action="append", required=True)
    parser.add_argument("--prefix-case-dir", type=Path, action="append", required=True)
    parser.add_argument("--short-case-dir", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = validate_prefix_correction(
        args.base_case_dir,
        args.prefix_case_dir,
        args.short_case_dir,
        args.model_config,
        args.output_dir,
        plots=not args.no_plots,
    )
    print(
        f"[validated] pairs={len(summary['cases'])} "
        f"prefix_gate={summary['prefix_gate']['status']} output={args.output_dir}"
    )


if __name__ == "__main__":
    main()
