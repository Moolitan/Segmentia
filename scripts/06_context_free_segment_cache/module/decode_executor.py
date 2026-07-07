"""Shared decode execution primitives for Segmentia experiments."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from vllm_client import chat_completion, extract_response  # noqa: E402


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def load_existing_keys(output: Path) -> set[tuple[str, str, int, str, int]]:
    existing: set[tuple[str, str, int, str, int]] = set()
    if not output.exists():
        return existing
    with output.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("error"):
                continue
            existing.add(
                (
                    row["task"],
                    row["skill"],
                    int(row["occurrence"]),
                    row["mode"],
                    int(row.get("sample_index", 0)),
                )
            )
    return existing


def run_single_decode(
    *,
    base_url: str,
    model: str,
    api_key: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    seed: int | None,
    request_id: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    context_segment_cache: dict[str, Any],
) -> tuple[dict[str, Any], float, dict[str, Any], str | None]:
    try:
        response, elapsed = chat_completion(
            base_url,
            model,
            messages,
            tools,
            api_key,
            max_tokens=max_tokens,
            request_id=request_id,
            context_segment_cache=context_segment_cache,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
        )
        parts = extract_response(response)
        error = None
    except RuntimeError as exc:
        response = {}
        elapsed = 0.0
        parts = {
            "text": "",
            "content": "",
            "reasoning": "",
            "tool_calls": [],
            "finish_reason": None,
        }
        error = str(exc)
    return response, elapsed, parts, error
