from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

import torch

from common import (
    DEFAULT_MODEL_PATH,
    add_common_args,
    base_url,
    default_log_file,
    request_json,
    start_vllm_server,
    stop_server,
)


def _load_tokenizer(model_path: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


def _encode(tokenizer, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def _find_segment(full_ids: list[int], segment_ids: list[int]) -> int:
    for start in range(0, len(full_ids) - len(segment_ids) + 1):
        if full_ids[start : start + len(segment_ids)] == segment_ids:
            return start
    raise RuntimeError("segment_ids not found in full prompt token ids")


def _layer_names(num_layers: int) -> list[str]:
    return [f"model.layers.{i}.self_attn.attn" for i in range(num_layers)]


def _make_fixture(model_path: str, kv_dir: Path) -> tuple[str, dict]:
    tokenizer = _load_tokenizer(model_path)
    prefix = "CSKCache probe-gated smoke prefix.\n"
    skill = (
        "Synthetic reusable skill span for CSKCache. "
        "The content is intentionally simple and deterministic.\n"
    )
    suffix = "Now answer with the single word OK.\n"
    prompt = prefix + skill + suffix

    full_ids = _encode(tokenizer, prompt)
    prefix_ids = _encode(tokenizer, prefix)
    prefix_skill_ids = _encode(tokenizer, prefix + skill)
    segment_ids = full_ids[len(prefix_ids) : len(prefix_skill_ids)]
    if not segment_ids:
        segment_ids = _encode(tokenizer, skill)
    start = _find_segment(full_ids, segment_ids)

    config_path = Path(model_path) / "config.json"
    config = json.loads(config_path.read_text())
    num_layers = int(config["num_hidden_layers"])
    num_heads = int(config["num_attention_heads"])
    num_kv_heads = int(config.get("num_key_value_heads", num_heads))
    hidden_size = int(config["hidden_size"])
    head_dim = int(config.get("head_dim", hidden_size // num_heads))

    # This is a path smoke fixture, not a quality fixture.  Zero KV is enough
    # to exercise probe decision, partial load planning, RoPE no-op, and scatter.
    length = len(segment_ids)
    kv_by_layer = {}
    for layer_name in _layer_names(num_layers):
        key = torch.zeros(length, num_kv_heads, head_dim, dtype=torch.bfloat16)
        value = torch.zeros(length, num_kv_heads, head_dim, dtype=torch.bfloat16)
        kv_by_layer[layer_name] = (key, value)

    payload = {
        "cache_id": "cskcache-smoke-skill",
        "source_start": start,
        "source_end": start + length,
        "token_ids": segment_ids,
        "kv_by_layer": kv_by_layer,
    }
    kv_dir.mkdir(parents=True, exist_ok=True)
    torch.save(payload, kv_dir / "cskcache-smoke-skill.pt")
    return prompt, {
        "start": start,
        "end": start + length,
        "length": length,
        "num_layers": num_layers,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a probe-gated CSKCache end-to-end path smoke."
    )
    add_common_args(parser)
    parser.add_argument("--model-path", default=os.environ.get("VLLM_MODEL_PATH", DEFAULT_MODEL_PATH))
    parser.add_argument("--max-model-len", type=int, default=int(os.environ.get("VLLM_MAX_MODEL_LEN", "4096")))
    parser.add_argument("--gpu-util", type=float, default=float(os.environ.get("VLLM_GPU_UTIL", "0.85")))
    parser.add_argument("--wait-timeout-s", type=int, default=900)
    parser.add_argument("--request-timeout-s", type=float, default=float(os.environ.get("CSKCACHE_TEST04_REQUEST_TIMEOUT_S", "60")))
    parser.add_argument("--keep-server", action="store_true")
    parser.add_argument("--keep-fixture", action="store_true")
    args = parser.parse_args()

    tmp_root = Path(tempfile.mkdtemp(prefix="cskcache_probe_e2e_"))
    kv_dir = tmp_root / "kv"
    log_file = default_log_file(args.port).with_name(f"cskcache_probe_e2e_{args.port}.log")
    prompt, fixture = _make_fixture(args.model_path, kv_dir)
    print("fixture:", fixture)
    print("kv_dir:", kv_dir)

    try:
        start_vllm_server(
            host=args.host,
            port=args.port,
            model_path=args.model_path,
            served_name=args.model,
            api_key=args.api_key,
            log_file=log_file,
            kv_dir=kv_dir,
            probe_enabled=True,
            max_model_len=args.max_model_len,
            gpu_util=args.gpu_util,
            wait_timeout_s=args.wait_timeout_s,
        )
        payload = {
            "model": args.model,
            "prompt": prompt,
            "max_tokens": 1,
            "temperature": 0,
        }
        print("send decode request:", repr(payload), flush=True)
        response = request_json(
            "POST",
            f"{base_url(args.host, args.port)}/v1/completions",
            api_key=args.api_key,
            payload=payload,
            timeout=args.request_timeout_s,
        )
        print("decode response text:", repr(response["choices"][0].get("text", "")))

        log_text = log_file.read_text(errors="replace")
        required = [
            "CSKCacheTRACE",
            "probe state",
        ]
        missing = [item for item in required if item not in log_text]
        if (
            "probe decision" not in log_text
            and "anchor completed" not in log_text
            and "in-process load requested" not in log_text
        ):
            missing.append("probe decision or fallback progress")
        if missing:
            raise RuntimeError(
                "probe-gated path did not leave expected log markers: "
                + ", ".join(missing)
                + f"\nLog: {log_file}"
            )
        print("probe-gated e2e smoke ok")
        print("log:", log_file)
    finally:
        if not args.keep_server:
            stop_server(args.port)
        if not args.keep_fixture:
            shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
