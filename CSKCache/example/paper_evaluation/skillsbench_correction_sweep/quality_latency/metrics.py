"""Thinking extraction and the frozen word-level ROUGE-L Recall metric."""

from __future__ import annotations

import re
from typing import Any, Mapping


WORD_PATTERN = re.compile(r"\w+", flags=re.UNICODE)


def extract_thinking(response: Mapping[str, Any] | None) -> tuple[str, str]:
    if not response:
        return "", "missing_response"
    choices = response.get("choices") or []
    message = choices[0].get("message") if choices else None
    if not isinstance(message, Mapping):
        return "", "missing_message"
    for field in ("reasoning_content", "reasoning"):
        value = message.get(field)
        if isinstance(value, str) and value.strip():
            return value, field
    content = message.get("content")
    if not isinstance(content, str) or "<think>" not in content:
        return "", "missing_thinking"
    thinking = content.split("<think>", 1)[1]
    if "</think>" in thinking:
        thinking = thinking.split("</think>", 1)[0]
    return thinking, "content_think_tag"


def finish_reason(response: Mapping[str, Any] | None) -> str:
    if not response:
        return ""
    choices = response.get("choices") or []
    return str(choices[0].get("finish_reason") or "") if choices else ""


def rouge_tokens(text: str) -> list[str]:
    return WORD_PATTERN.findall(text.casefold())


def rouge_l_recall(reference: str, candidate: str) -> float:
    """Return LCS(reference, candidate) / len(reference) over word tokens."""

    target = rouge_tokens(reference)
    prediction = rouge_tokens(candidate)
    if not target:
        raise ValueError("ROUGE-L reference has no word tokens")
    if not prediction:
        return 0.0
    if len(prediction) > len(target):
        rows, columns = prediction, target
    else:
        rows, columns = target, prediction
    previous = [0] * (len(columns) + 1)
    for row_token in rows:
        current = [0]
        for index, column_token in enumerate(columns, start=1):
            if row_token == column_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1] / len(target)
