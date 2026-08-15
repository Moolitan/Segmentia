#!/usr/bin/env python3
"""Prepare a frozen OpenHands prompt and capture full-recompute attention."""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer

REPOSITORY_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPOSITORY_ROOT / "CSKCache"))
from cskcache import (  # noqa: E402
    build_context_segment_token_identity,
    render_context_segment,
)
from attention_heatmap.openhands_prompt_context import (  # noqa: E402
    export_openhands_context,
)

USER_MESSAGE = 'Please explicitly reference and use the "internal-comms" skill for this turn.'
TOOL_CALL_ID = "call_internal_comms"


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


def messages_for(
    system_message: dict,
    skill_text: str,
) -> list[dict]:
    return [
        system_message,
        {"role": "user", "content": USER_MESSAGE},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": TOOL_CALL_ID,
                    "type": "function",
                    "function": {
                        "name": "skill",
                        "arguments": json.dumps(
                            {"name": "internal-comms", "kind": "SkillAction"},
                            separators=(",", ":"),
                        ),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": TOOL_CALL_ID,
            "name": "skill",
            "content": render_context_segment("internal-comms", skill_text),
        },
    ]


def find_once(tokens: list[int], target: list[int]) -> int:
    matches = [i for i in range(len(tokens) - len(target) + 1) if tokens[i:i + len(target)] == target]
    if len(matches) != 1:
        raise RuntimeError(f"expected one Skill span, found {matches}")
    return matches[0]


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8014")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", default="Qwen3")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--skill-path", type=Path, required=True)
    parser.add_argument("--skills-dir", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--spec-path", type=Path, required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=True)
    skill_text = args.skill_path.read_text()
    system_message, tools = export_openhands_context(
        skills_dir=args.skills_dir,
        workspace=args.workspace,
        served_model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
    )
    messages = messages_for(system_message, skill_text)
    token_payload = {
        "model": args.model,
        "messages": messages,
        "tools": tools,
        "add_generation_prompt": True,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    base_prompt_ids = post_json(
        args.base_url, "/tokenize", args.api_key, token_payload
    ).get("tokens")
    if not isinstance(base_prompt_ids, list):
        raise RuntimeError("vLLM /tokenize did not return token IDs")
    base_prompt_ids = [int(token_id) for token_id in base_prompt_ids]
    skill_ids = list(build_context_segment_token_identity(
        tokenizer,
        "internal-comms",
        skill_text,
    ).token_ids)
    start = find_once(base_prompt_ids, skill_ids)
    end = start + len(skill_ids)
    template_suffix_count = len(base_prompt_ids) - end
    if template_suffix_count <= 0:
        raise RuntimeError(
            "Qwen chat template did not append any suffix tokens after the Skill"
        )
    prompt_ids = base_prompt_ids
    suffix_end = len(prompt_ids)

    mode_dir = args.output_dir / "recompute"
    spec = {
        "request_id": "segmentia-attention-recompute",
        "mode": "recompute",
        "skill_start": start,
        "skill_end": end,
        "cross_key_start": start - 48,
        "cross_key_end": start,
        "forward_query_start": end,
        "forward_query_end": suffix_end,
        "forward_key_start": start,
        "forward_key_end": end,
        "prompt_end": len(prompt_ids),
    }
    if spec["cross_key_start"] < 0:
        raise RuntimeError("prompt has fewer than 48 tokens before the Skill")
    write_json(args.spec_path, spec)
    write_json(
        mode_dir / "prompt_metadata.json",
        {
            **spec,
            "user_messages": [USER_MESSAGE],
            "prompt_token_ids": prompt_ids,
            "template_suffix_token_count": template_suffix_count,
            "system_message": system_message,
            "tools": tools,
        },
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
        },
        spec["request_id"],
    )
    write_json(mode_dir / "response.json", response)

    files = sorted(mode_dir.glob("recompute_layer_*.npz"))
    if len(files) != 40:
        raise RuntimeError(f"expected 40 layers, found {len(files)}")
    for path in files:
        with np.load(path) as layer:
            if layer["cross"].shape != (end - start, 48):
                raise RuntimeError(f"incomplete cross-attention capture: {path}")
            if layer["forward"].shape != (template_suffix_count, end - start):
                raise RuntimeError(f"incomplete attention capture: {path}")
    print(
        f"[captured] recompute prompt={len(prompt_ids)} "
        f"cross=Q[{start},{end})xK[{start - 48},{start}) "
        f"forward=Q[{end},{suffix_end})xK[{start},{end}) layers=40"
    )


if __name__ == "__main__":
    main()
