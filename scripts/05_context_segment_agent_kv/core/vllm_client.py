"""HTTP layer for talking to the (modified) vLLM OpenAI-compatible server.

stdlib only — mirrors the real_system script so behaviour stays identical.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any


def post_json(base_url: str, path: str, payload: dict[str, Any], api_key: str) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=720) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {path}: {body}") from exc


def tokenize_chat(
    base_url: str,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict] | None,
    api_key: str,
    *,
    add_generation_prompt: bool,
) -> list[int]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "add_generation_prompt": add_generation_prompt,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    if tools:
        payload["tools"] = tools
    response = post_json(base_url, "/tokenize", payload, api_key)
    return list(response["tokens"])


def chat_completion(
    base_url: str,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict] | None,
    api_key: str,
    *,
    max_tokens: int,
    request_id: str,
    context_segment_cache: dict[str, Any] | None = None,
) -> tuple[dict, float]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "request_id": request_id,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    if tools:
        payload["tools"] = tools
    if context_segment_cache is not None:
        payload["vllm_xargs"] = {
            "context_segment_cache": json.dumps(
                context_segment_cache, ensure_ascii=False, sort_keys=True
            )
        }
    start = time.time()
    response = post_json(base_url, "/v1/chat/completions", payload, api_key)
    return response, time.time() - start
