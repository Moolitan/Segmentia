#!/usr/bin/env python3
"""Validate no-copy shared Skill attention with real Prefix-256 offsets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from shared_attention_reference import (
    error_metrics,
    materialized_attention,
    shared_attention,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        action="append",
        nargs=4,
        metavar=("CASE_ID", "CAPTURE", "SOURCE_START", "TARGET_START"),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shared-tokens", type=int, default=257)
    parser.add_argument("--private-tokens", type=int, default=31)
    parser.add_argument("--query-tokens", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def stable_seed(case_id: str, layer_id: int, dtype: str) -> int:
    digest = hashlib.sha256(f"{case_id}:{layer_id}:{dtype}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**63 - 1)


def tensor_metrics(
    case_id: str,
    layer_id: int,
    dtype: torch.dtype,
    offset: torch.Tensor,
    delta: int,
    shared_tokens: int,
    private_tokens: int,
    query_tokens: int,
) -> dict[str, Any]:
    if offset.ndim != 1 or offset.numel() != 8 * 128:
        raise ValueError(
            f"{case_id} layer {layer_id}: expected flattened [8,128] offset, "
            f"got {tuple(offset.shape)}"
        )
    generator = torch.Generator().manual_seed(
        stable_seed(case_id, layer_id, str(dtype))
    )
    q = torch.randn(query_tokens, 40, 128, generator=generator).to(dtype)
    source_k = torch.randn(shared_tokens, 8, 128, generator=generator).to(dtype)
    shared_v = torch.randn(shared_tokens, 8, 128, generator=generator).to(dtype)
    private_k = torch.randn(private_tokens, 8, 128, generator=generator).to(dtype)
    private_v = torch.randn(private_tokens, 8, 128, generator=generator).to(dtype)
    offset = offset.reshape(8, 128)
    materialized = materialized_attention(
        q,
        source_k,
        shared_v,
        offset,
        delta,
        private_key=private_k,
        private_value=private_v,
    )
    shared = shared_attention(
        q,
        source_k,
        shared_v,
        offset,
        delta,
        private_key=private_k,
        private_value=private_v,
    )
    output_error = error_metrics(shared.output, materialized.output)
    lse_error = error_metrics(shared.lse, materialized.lse)
    finite = bool(torch.isfinite(shared.output).all() and torch.isfinite(shared.lse).all())
    if dtype == torch.float32:
        passed = (
            finite
            and output_error["max_abs"] <= 1e-5
            and output_error["relative_l2"] <= 1e-6
            and lse_error["max_abs"] <= 1e-5
        )
    else:
        passed = (
            finite
            and output_error["relative_l2"] <= 1e-2
            and output_error["cosine"] >= 0.9999
        )
    return {
        "case_id": case_id,
        "layer_id": layer_id,
        "dtype": str(dtype).removeprefix("torch."),
        "position_delta": delta,
        "output_max_abs": output_error["max_abs"],
        "output_relative_l2": output_error["relative_l2"],
        "output_cosine": output_error["cosine"],
        "lse_max_abs": lse_error["max_abs"],
        "finite": finite,
        "passed": passed,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"non-empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = args.output_dir / "data"
    tables_dir = args.output_dir / "tables"
    data_dir.mkdir(exist_ok=True)
    tables_dir.mkdir(exist_ok=True)
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for case_id, capture_text, source_text, target_text in sorted(args.case):
        capture = Path(capture_text).resolve()
        payload = torch.load(capture, map_location="cpu", weights_only=True)
        if payload.get("schema_version") != 4:
            raise ValueError(f"{capture}: expected schema_version=4")
        if payload.get("correction_mode") != "prefix_k_headwise":
            raise ValueError(f"{capture}: unexpected correction mode")
        layers = payload.get("layers")
        if not isinstance(layers, dict) or sorted(layers) != list(range(40)):
            raise ValueError(f"{capture}: expected complete layers 0..39")
        delta = int(target_text) - int(source_text)
        sources.append(
            {
                "case_id": case_id,
                "capture": str(capture),
                "source_start": int(source_text),
                "target_start": int(target_text),
                "position_delta": delta,
            }
        )
        for layer_id in sorted(layers):
            layer = layers[layer_id]
            offset = layer.get("global_offset")
            if not isinstance(offset, torch.Tensor):
                raise ValueError(f"{capture}: layer {layer_id} has no global_offset")
            for dtype in (torch.float32, torch.bfloat16):
                rows.append(
                    tensor_metrics(
                        case_id,
                        layer_id,
                        dtype,
                        offset,
                        delta,
                        args.shared_tokens,
                        args.private_tokens,
                        args.query_tokens,
                    )
                )
    rows.sort(key=lambda row: (row["case_id"], row["layer_id"], row["dtype"]))
    write_csv(tables_dir / "equivalence_by_case_layer.csv", rows)
    failed = [row for row in rows if not row["passed"]]
    summary = {
        "schema_version": 1,
        "gate": "go" if not failed else "no_go",
        "cases": len(sources),
        "layers_per_case": 40,
        "rows": len(rows),
        "failed_rows": len(failed),
        "shared_tokens": args.shared_tokens,
        "private_tokens": args.private_tokens,
        "query_tokens": args.query_tokens,
        "max_output_abs_fp32": max(
            row["output_max_abs"] for row in rows if row["dtype"] == "float32"
        ),
        "max_output_relative_l2_fp32": max(
            row["output_relative_l2"]
            for row in rows
            if row["dtype"] == "float32"
        ),
        "max_output_relative_l2_bfloat16": max(
            row["output_relative_l2"]
            for row in rows
            if row["dtype"] == "bfloat16"
        ),
        "min_output_cosine_bfloat16": min(
            row["output_cosine"] for row in rows if row["dtype"] == "bfloat16"
        ),
        "sources": sources,
    }
    (data_dir / "equivalence_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        f"[validated] cases={summary['cases']} rows={summary['rows']} "
        f"shared_attention_gate={summary['gate']} output={args.output_dir}"
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
