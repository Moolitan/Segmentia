"""Build the model-native token object cached for one Skill tool response."""
from __future__ import annotations

from typing import Any


CACHE_SCHEMA_VERSION = 2
CACHE_OBJECT_TYPE = "qwen_tool_response"
TOOL_RESPONSE_OPEN = "<tool_response>"
TOOL_RESPONSE_CLOSE = "</tool_response>"


def qwen_tool_response_token_ids(tokenizer: Any, skill_text: str) -> list[int]:
    """Return the complete native Qwen tool-response token span for a Skill."""
    rendered = tokenizer.apply_chat_template(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "skill",
                            "arguments": '{"name":"segmentia-offline-prefill"}',
                        },
                    }
                ],
            },
            {"role": "tool", "name": "skill", "content": skill_text},
        ],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=True,
    )
    token_ids = tokenizer.encode(rendered, add_special_tokens=False)
    open_id = tokenizer.convert_tokens_to_ids(TOOL_RESPONSE_OPEN)
    close_id = tokenizer.convert_tokens_to_ids(TOOL_RESPONSE_CLOSE)
    open_positions = [
        index for index, token_id in enumerate(token_ids) if token_id == open_id
    ]
    close_positions = [
        index for index, token_id in enumerate(token_ids) if token_id == close_id
    ]
    if len(open_positions) != 1 or len(close_positions) != 1:
        raise RuntimeError(
            "Qwen chat template must render exactly one native tool response; "
            f"open_positions={open_positions}, close_positions={close_positions}"
        )
    start = open_positions[0]
    end = close_positions[0] + 1
    if start >= end:
        raise RuntimeError(
            f"invalid Qwen tool-response token span: start={start}, end={end}"
        )
    return token_ids[start:end]
