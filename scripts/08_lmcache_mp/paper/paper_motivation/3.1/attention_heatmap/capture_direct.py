#!/usr/bin/env python3
"""Capture direct-reuse attention using the frozen recompute prompt IDs."""
from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np


def post_json(base_url: str, path: str, api_key: str, payload: dict, request_id: str | None = None) -> dict:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    if request_id:
        headers["X-Request-Id"] = request_id
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=json.dumps(payload).encode(),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code} from {path}: {exc.read().decode(errors='replace')}") from exc


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8014")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", default="Qwen3")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--spec-path", type=Path, required=True)
    args = parser.parse_args()

    recompute = json.loads((args.output_dir / "recompute" / "prompt_metadata.json").read_text())
    prompt_ids = [int(token_id) for token_id in recompute["prompt_token_ids"]]
    start = int(recompute["skill_start"])
    end = int(recompute["skill_end"])
    suffix_end = int(recompute["forward_query_end"])
    suffix_count = suffix_end - end

    mode_dir = args.output_dir / "direct"
    spec = {
        key: recompute[key]
        for key in (
            "skill_start",
            "skill_end",
            "cross_key_start",
            "cross_key_end",
            "forward_query_start",
            "forward_query_end",
            "forward_key_start",
            "forward_key_end",
            "prompt_end",
        )
    }
    spec.update(request_id="segmentia-attention-direct", mode="direct")
    write_json(args.spec_path, spec)
    write_json(
        mode_dir / "prompt_metadata.json",
        {**recompute, **spec, "prompt_token_ids": prompt_ids},
    )
    response = post_json(
        args.base_url,
        "/v1/completions",
        args.api_key,
        {
            "model": args.model,
            "prompt": prompt_ids,
            "max_tokens": 1,
            "temperature": 0,
            "seed": 0,
            "kv_transfer_params": {
                "lmcache_segmentia_lookup": {
                    "segment_start": start,
                    "segment_end": end,
                }
            },
        },
        spec["request_id"],
    )
    write_json(mode_dir / "response.json", response)

    files = sorted(mode_dir.glob("direct_layer_*.npz"))
    if len(files) != 40:
        raise RuntimeError(f"expected 40 layers, found {len(files)}")
    for path in files:
        with np.load(path) as layer:
            cross_rows = int(layer["cross"].shape[0])
            if layer["cross"].shape[1:] != (48,) or not 0 <= cross_rows < 16:
                raise RuntimeError(f"unexpected direct cross-attention: {path}")
            if layer["forward"].shape != (suffix_count, end - start):
                raise RuntimeError(f"incomplete attention capture: {path}")
    print(
        f"[captured] direct prompt={len(prompt_ids)} "
        f"cross=alignment-prefix-only K[{start - 48},{start}) "
        f"forward=Q[{end},{suffix_end})xK[{start},{end}) layers=40"
    )


if __name__ == "__main__":
    main()
