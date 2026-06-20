"""Small stdlib HTTP client for the local vLLM OpenAI-compatible server."""
from __future__ import annotations

import json
import time
import socket
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
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError, socket.timeout) as exc:
        raise RuntimeError(f"Connection failure from {path}: {exc}") from exc


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
    temperature: float = 0.0,
    top_p: float = 1.0,
    seed: int | None = None,
    logprobs: bool = False,
    top_logprobs: int | None = None,
) -> tuple[dict, float]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "request_id": request_id,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    if seed is not None:
        payload["seed"] = seed
    if logprobs:
        payload["logprobs"] = True
        if top_logprobs is not None:
            payload["top_logprobs"] = top_logprobs
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


def completion_text(response: dict[str, Any]) -> str:
    choice = response.get("choices", [{}])[0]
    message = choice.get("message") or {}
    if message.get("content"):
        return message["content"]
    # Tool-call response: content is null per OpenAI spec; extract from tool_calls
    tool_calls = message.get("tool_calls") or []
    if tool_calls:
        parts = []
        for tc in tool_calls:
            fn = tc.get("function") or {}
            parts.append(f"{fn.get('name', '')}({fn.get('arguments', '')})")
        return "\n".join(parts)
    return choice.get("text") or ""


def extract_response(response: dict[str, Any]) -> dict[str, Any]:
    """Pull out every part of the generated sequence we want to score.

    Qwen3's reasoning parser splits the hidden chain-of-thought into
    ``message.reasoning_content`` and leaves only the visible answer in
    ``message.content``. The earlier harness saved just the visible ``text``,
    so the reasoning stream was never evaluated. We now also return that
    reasoning, the raw visible content, the structured tool calls, and the
    finish reason so downstream metrics can score the full sequence and the
    action-level trajectory rather than visible wording alone.
    """
    choice = (response.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    content = message.get("content") or ""
    tool_calls = message.get("tool_calls") or []
    return {
        "text": completion_text(response),
        "content": content,
        "reasoning": reasoning,
        "tool_calls": tool_calls,
        "finish_reason": choice.get("finish_reason"),
    }
