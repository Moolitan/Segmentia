#!/usr/bin/env python3
"""Compare offline context-free Skill K with online full-context Skill K."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch


LAYER_SUFFIX = ".pt.meta.json"


def sha256_tokens(token_ids: list[int]) -> str:
    digest = hashlib.sha256()
    for token_id in token_ids:
        digest.update(int(token_id).to_bytes(4, "little", signed=False))
    return digest.hexdigest()


def layer_sidecars(directory: Path) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for path in directory.rglob(f"*{LAYER_SUFFIX}"):
        stem = path.name[: -len(LAYER_SUFFIX)]
        try:
            layer = int(stem.rsplit("@", 1)[1])
        except (IndexError, ValueError) as exc:
            raise ValueError(f"cannot parse layer from {path}") from exc
        if layer in result:
            raise ValueError(f"duplicate layer {layer} under {directory}")
        result[layer] = path
    if set(result) != set(range(40)):
        raise ValueError(f"expected layers 0..39 under {directory}, found {sorted(result)}")
    return result


def read_layer(sidecar: Path) -> tuple[torch.Tensor, dict[str, Any]]:
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    if metadata.get("memory_format") != "KV_2TD" or metadata.get("dtype") != "bfloat16":
        raise ValueError(f"unsupported KV metadata: {sidecar}")
    shape = metadata.get("shape")
    if not isinstance(shape, list) or len(shape) != 3 or shape[0] != 2:
        raise ValueError(f"invalid KV shape in {sidecar}: {shape}")
    data_path = sidecar.with_name(str(metadata["data_file"]))
    count = math.prod(shape)
    tensor = torch.from_file(
        str(data_path), shared=False, size=count, dtype=torch.bfloat16
    ).reshape(shape)
    return tensor, metadata


def relocate_neox_rope(key: torch.Tensor, shift: int, theta: float) -> torch.Tensor:
    key = key.to(torch.float32)
    head_dim = key.shape[-1]
    if head_dim % 2:
        raise ValueError("NeoX RoPE requires an even head dimension")
    half = head_dim // 2
    exponent = torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
    phase = float(shift) * torch.pow(torch.tensor(theta), -exponent)
    cosine = phase.cos().view(1, 1, half)
    sine = phase.sin().view(1, 1, half)
    first, second = key[..., :half], key[..., half:]
    return torch.cat(
        (first * cosine - second * sine, second * cosine + first * sine), dim=-1
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze_case(
    case_id: str,
    skill: str,
    offline_dir: Path,
    online_dir: Path,
    capture_path: Path,
    budgets: list[int],
    evaluation_start: int,
    theta: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    capture = json.loads(capture_path.read_text(encoding="utf-8"))
    offline_manifest_path = offline_dir.parent / "manifest.json"
    offline_manifest = json.loads(offline_manifest_path.read_text(encoding="utf-8"))
    if capture.get("status") != "completed" or capture.get("skill") != skill:
        raise ValueError(f"invalid capture record: {capture_path}")
    token_count = int(capture["token_count"])
    if not (
        offline_manifest.get("schema_version") == 3
        and offline_manifest.get("status") == "completed"
        and offline_manifest.get("skill_name") == skill
        and offline_manifest.get("token_count") == token_count
        and offline_manifest.get("token_ids_sha256")
        == capture.get("token_ids_sha256")
    ):
        raise ValueError(
            f"offline manifest and online capture disagree for Skill {skill}"
        )
    shift = int(capture["segment_start"])
    if token_count <= evaluation_start:
        raise ValueError(
            f"Skill {skill} has {token_count} tokens, not enough for tail [{evaluation_start},S)"
        )
    offline_layers = layer_sidecars(offline_dir)
    online_layers = layer_sidecars(online_dir)
    metric_rows: list[dict[str, Any]] = []
    head_rows: list[dict[str, Any]] = []

    for layer in range(40):
        offline_kv, offline_meta = read_layer(offline_layers[layer])
        online_kv, online_meta = read_layer(online_layers[layer])
        if offline_kv.shape != online_kv.shape or offline_kv.shape[1] != token_count:
            raise ValueError(f"KV shape mismatch for {case_id} layer={layer}")
        offline_pos = offline_meta.get("cached_positions", {})
        online_pos = online_meta.get("cached_positions", {})
        if offline_pos.get("start") != 0 or online_pos.get("start") != shift:
            raise ValueError(
                f"position mismatch for {case_id} layer={layer}: "
                f"offline={offline_pos} online={online_pos} capture_start={shift}"
            )
        offline_key = offline_kv[0].reshape(token_count, 8, 128)
        online_key = online_kv[0].reshape(token_count, 8, 128).to(torch.float32)
        aligned = relocate_neox_rope(offline_key, shift, theta)
        residual = online_key - aligned
        direct_tail = aligned[evaluation_start:]
        target_tail = online_key[evaluation_start:]
        true_tail_offset = residual[evaluation_start:].mean(dim=0)
        direct_sse = float((direct_tail - target_tail).square().sum())

        for budget in budgets:
            if budget > evaluation_start:
                raise ValueError("budget cannot exceed common evaluation start")
            estimate = residual[:budget].mean(dim=0)
            corrected = direct_tail + estimate.unsqueeze(0)
            corrected_sse = float((corrected - target_tail).square().sum())
            metric_rows.append(
                {
                    "case_id": case_id,
                    "skill": skill,
                    "token_count": token_count,
                    "layer": layer,
                    "budget": budget,
                    "evaluation_start": evaluation_start,
                    "direct_sse": direct_sse,
                    "corrected_sse": corrected_sse,
                }
            )
            for head in range(8):
                a = estimate[head].to(torch.float64)
                b = true_tail_offset[head].to(torch.float64)
                cosine = float(torch.dot(a, b) / torch.sqrt(torch.clamp(torch.dot(a, a) * torch.dot(b, b), min=1e-30)))
                head_rows.append(
                    {
                        "case_id": case_id,
                        "skill": skill,
                        "layer": layer,
                        "head": head,
                        "budget": budget,
                        "offset_cosine": cosine,
                    }
                )
    return metric_rows, head_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--pool-dir", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--budgets", default="32,64,128,256")
    parser.add_argument("--evaluation-start", type=int, default=256)
    parser.add_argument("--rope-theta", type=float, default=1000000.0)
    args = parser.parse_args()
    budgets = [int(item) for item in args.budgets.split(",")]
    cases = json.loads(args.cases.read_text(encoding="utf-8"))["cases"]
    all_metrics: list[dict[str, Any]] = []
    all_heads: list[dict[str, Any]] = []
    for case in cases:
        case_dir = args.run_dir / case["case_id"]
        metrics, heads = analyze_case(
            case["case_id"],
            case["skill"],
            args.pool_dir / case["skill"] / "kv",
            case_dir / "online_full_kv",
            case_dir / "capture.json",
            budgets,
            args.evaluation_start,
            args.rope_theta,
        )
        all_metrics.extend(metrics)
        all_heads.extend(heads)
    write_csv(args.output_dir / "layer_metrics.csv", all_metrics)
    write_csv(args.output_dir / "head_cosines.csv", all_heads)
    print(f"[analyzed] cases={len(cases)} output={args.output_dir}")


if __name__ == "__main__":
    main()
