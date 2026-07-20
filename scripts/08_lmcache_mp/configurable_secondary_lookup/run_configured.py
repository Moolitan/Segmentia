#!/usr/bin/env python3
"""Send synthetic secondary-lookup requests described by a JSON file."""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path


FIELDS = (
    "prefix_tokens",
    "gap_tokens",
    "reuse_tokens",
    "suffix_tokens",
)


def token_stream(length: int, base: int, offset: int = 0) -> list[int]:
    # 直接生成指定长度的 token ID 列表，不经过 tokenizer
    return [base + ((offset + index) % 997) for index in range(length)]


def post_json(url: str, api_key: str, payload: dict) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.loads(response.read())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8100")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", default="Qwen3")
    parser.add_argument("--separator-token-id", type=int, default=151663)
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    requests = config["requests"]
    indexes = [item["request_index"] for item in requests]
    if len(indexes) != len(set(indexes)):
        raise ValueError("request_index values must be unique")

    output_dir = args.output_dir
    request_dir = output_dir / "requests"
    response_dir = output_dir / "responses"
    request_dir.mkdir(parents=True, exist_ok=True)
    response_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    timings_path = output_dir / "timings.jsonl"
    previous_prompt: list[int] = []
    with timings_path.open("w") as timings_file:
        for position, item in enumerate(
            sorted(requests, key=lambda row: row["request_index"])
        ):
            request_index = item["request_index"]
            if not isinstance(request_index, int) or request_index < 1:
                raise ValueError("request_index must be a positive integer")
            for field in FIELDS:
                if not isinstance(item[field], int) or item[field] < 0:
                    raise ValueError(f"{field} must be a non-negative integer")

            if position == 0:
                prefix = token_stream(item["prefix_tokens"], 1000)
                inherited_prefix_tokens = 0
            else:
                inherited_prefix_tokens = len(previous_prompt)
                if item["prefix_tokens"] < inherited_prefix_tokens:
                    raise ValueError(
                        f"request {request_index}: prefix_tokens="
                        f"{item['prefix_tokens']} is shorter than the previous "
                        f"prompt ({inherited_prefix_tokens} tokens)"
                    )
                padding_tokens = item["prefix_tokens"] - inherited_prefix_tokens
                prefix = previous_prompt.copy()
                prefix.extend(
                    token_stream(padding_tokens, 12000, request_index * 197)
                )

            gap = token_stream(item["gap_tokens"], 3000, request_index * 131)
            reusable = token_stream(item["reuse_tokens"], 6000)
            suffix = token_stream(item["suffix_tokens"], 9000, request_index * 173)
            separator = [args.separator_token_id]

            segment_start = len(prefix) + len(gap) + len(separator)
            segment_end = segment_start + len(reusable) + len(separator)
            prompt = prefix + gap + separator + reusable + separator + suffix

            payload = {
                "model": args.model,
                "prompt": prompt,
                "max_tokens": 1,
                "temperature": 0,
                "kv_transfer_params": {
                    "lmcache_secondary_lookup": {
                        "segment_start": segment_start,
                        "segment_end": segment_end,
                        "probe_only": False,
                    }
                },
            }
            request_path = request_dir / f"request_{request_index:03d}.json"
            request_path.write_text(json.dumps(payload) + "\n")

            started = time.perf_counter()
            response = post_json(
                f"{args.base_url.rstrip('/')}/v1/completions",
                args.api_key,
                payload,
            )
            elapsed_ms = (time.perf_counter() - started) * 1000

            response_path = response_dir / f"request_{request_index:03d}.json"
            response_path.write_text(json.dumps(response, indent=2) + "\n")
            timing = {
                "request_index": request_index,
                "prompt_tokens": len(prompt),
                "inherited_prefix_tokens": inherited_prefix_tokens,
                "segment_start": segment_start,
                "segment_end": segment_end,
                "elapsed_ms": elapsed_ms,
            }
            timings_file.write(json.dumps(timing) + "\n")
            timings_file.flush()
            print(json.dumps(timing))
            previous_prompt = prompt


if __name__ == "__main__":
    main()
