"""Trace-replay ContextSegmentKV harness, split into functional modules.

Modules:
- config           paths + DEFAULT_TASKS
- vllm_client      HTTP layer (post_json / tokenize_chat / chat_completion)
- trace_loader     load src/traces invocation JSONs
- message_convert  Anthropic typed-blocks -> OpenAI chat/tools, skill wrapping
- segments         locate skill segments + char->token span mapping
- replay           replay one task, assign ContextSegmentKV source/target

The CLI driver lives in replay/replay_trace_context_segment.py, not here.
"""
from __future__ import annotations

from .config import DEFAULT_TASKS, ROOT, TRACES_DIR
from .message_convert import (
    convert_messages,
    convert_tools,
    skill_name_from_read_path,
    wrap_context_segment,
)
from .replay import replay_task
from .segments import find_skill_segments, span_token_offsets
from .trace_loader import load_invocations
from .vllm_client import chat_completion, post_json, tokenize_chat

__all__ = [
    "DEFAULT_TASKS",
    "ROOT",
    "TRACES_DIR",
    "convert_messages",
    "convert_tools",
    "skill_name_from_read_path",
    "wrap_context_segment",
    "replay_task",
    "find_skill_segments",
    "span_token_offsets",
    "load_invocations",
    "chat_completion",
    "post_json",
    "tokenize_chat",
]
