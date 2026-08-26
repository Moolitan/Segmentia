"""Small OpenAI-compatible HTTP client with streaming TTFT capture."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class CompletionResult:
    response: Mapping[str, Any] | None
    text: str
    output_tokens: int
    client_ttft_ms: float | None
    client_latency_ms: float


def request_json(
    url: str,
    *,
    api_key: str,
    payload: Mapping[str, Any] | None = None,
    method: str = "POST",
    timeout: float = 900,
) -> dict[str, Any]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method=method,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            decoded = response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        decoded = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"HTTP {error.code} from {url}: {decoded[-2000:]}"
        ) from error
    parsed = json.loads(decoded)
    if not isinstance(parsed, dict):
        raise RuntimeError(f"expected JSON object from {url}")
    return parsed


def stream_chat(
    base_url: str,
    *,
    api_key: str,
    payload: Mapping[str, Any],
    timeout: float,
) -> CompletionResult:
    body = dict(payload)
    body["stream"] = True
    body["stream_options"] = {"include_usage": True}
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    started = time.perf_counter_ns()
    first_token_ns = None
    pieces: list[str] = []
    usage_tokens = 0
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                event = json.loads(data)
                usage = event.get("usage") or {}
                usage_tokens = max(
                    usage_tokens, int(usage.get("completion_tokens") or 0)
                )
                choices = event.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                content = delta.get("content")
                reasoning = delta.get("reasoning_content")
                visible = content if content not in (None, "") else reasoning
                if visible not in (None, ""):
                    if first_token_ns is None:
                        first_token_ns = time.perf_counter_ns()
                    pieces.append(str(visible))
                tool_calls = delta.get("tool_calls") or []
                if tool_calls and first_token_ns is None:
                    first_token_ns = time.perf_counter_ns()
    except urllib.error.HTTPError as error:
        decoded = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"stream request failed with HTTP {error.code}: {decoded[-2000:]}"
        ) from error
    ended = time.perf_counter_ns()
    return CompletionResult(
        response=None,
        text="".join(pieces),
        output_tokens=usage_tokens,
        client_ttft_ms=(
            None if first_token_ns is None else (first_token_ns - started) / 1e6
        ),
        client_latency_ms=(ended - started) / 1e6,
    )


def nonstream_chat(
    base_url: str,
    *,
    api_key: str,
    payload: Mapping[str, Any],
    timeout: float,
) -> CompletionResult:
    body = dict(payload)
    body["stream"] = False
    started = time.perf_counter_ns()
    response = request_json(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        api_key=api_key,
        payload=body,
        timeout=timeout,
    )
    ended = time.perf_counter_ns()
    choices = response.get("choices") or []
    message = choices[0].get("message") if choices else {}
    text = str((message or {}).get("content") or "")
    usage = response.get("usage") or {}
    return CompletionResult(
        response=response,
        text=text,
        output_tokens=int(usage.get("completion_tokens") or 0),
        client_ttft_ms=None,
        client_latency_ms=(ended - started) / 1e6,
    )
