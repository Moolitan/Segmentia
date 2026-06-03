"""Locate wrapped skill segments and map their char spans to token spans."""
from __future__ import annotations

import copy

from .vllm_client import tokenize_chat


def find_skill_segments(
    openai_messages: list[dict],
) -> list[tuple[str, int, int, int]]:
    """Locate every wrapped skill segment in tool messages.

    Returns list of (skill_name, msg_idx, char_start, char_end), in message
    order. Occurrence order across the whole conversation = read order.
    """
    found: list[tuple[str, int, int, int]] = []
    for idx, msg in enumerate(openai_messages):
        if msg.get("role") != "tool":
            continue
        text = msg.get("content", "")
        pos = 0
        while True:
            m = text.find('<context_segment id="', pos)
            if m < 0:
                break
            id_start = m + len('<context_segment id="')
            id_end = text.find('"', id_start)
            name = text[id_start:id_end]
            close = text.find("</context_segment>", id_end)
            seg_end = close + len("</context_segment>")
            found.append((name, idx, m, seg_end))
            pos = seg_end
    return found


def span_token_offsets(
    base_url: str,
    model: str,
    messages: list[dict],
    tools: list[dict] | None,
    api_key: str,
    msg_idx: int,
    char_start: int,
    char_end: int,
) -> tuple[int, int]:
    """Convert a char span inside messages[msg_idx].content to a token span.

    Tokenizes the prompt truncated at char_start and at char_end (via vLLM
    /tokenize, so the chat template / enable_thinking match generation) and
    takes the token-length difference as the span.
    """
    msg_text = messages[msg_idx]["content"]

    start_msgs = copy.deepcopy(messages[: msg_idx + 1])
    start_msgs[msg_idx]["content"] = msg_text[:char_start]
    end_msgs = copy.deepcopy(messages[: msg_idx + 1])
    end_msgs[msg_idx]["content"] = msg_text[:char_end]

    start = len(
        tokenize_chat(base_url, model, start_msgs, tools, api_key,
                      add_generation_prompt=False)
    )
    end = len(
        tokenize_chat(base_url, model, end_msgs, tools, api_key,
                      add_generation_prompt=False)
    )
    return start, end
