"""Convert Anthropic typed-block traces to OpenAI chat messages / tools.

Skill detection is structural: a tool_result whose originating Read tool_use
targeted .../skills/<name>/SKILL.md is skill <name>; its content is wrapped in
<context_segment id="<name>">…</context_segment> so source and target spans
share identical token boundaries (RoPE reuse needs equal token length).
"""
from __future__ import annotations

import json
from typing import Any


def wrap_context_segment(skill_name: str, text: str) -> str:
    return f'<context_segment id="{skill_name}">\n{text}\n</context_segment>'


def skill_name_from_read_path(path: str) -> str | None:
    """Return <name> if path looks like .../skills/<name>/SKILL.md."""
    p = path.replace("\\", "/")
    if not p.endswith("/SKILL.md"):
        return None
    parts = p.split("/")
    try:
        i = parts.index("skills")
    except ValueError:
        return None
    if i + 1 < len(parts):
        return parts[i + 1]
    return None


def convert_messages(
    anthropic_messages: list[dict],
    system_prompt: str,
) -> tuple[list[dict], dict[str, str]]:
    """Convert Anthropic typed-block messages to OpenAI chat messages.

    Returns (openai_messages, tool_id_to_skill). tool_id_to_skill maps a
    Read tool_use id to the skill it loaded, so the matching tool_result can
    be wrapped as a <context_segment>.
    """
    # First pass: map Read tool_use ids -> skill name.
    tool_id_to_skill: dict[str, str] = {}
    for msg in anthropic_messages:
        if msg["role"] != "assistant":
            continue
        for block in msg["content"]:
            if block.get("type") == "tool_use" and block.get("name") == "Read":
                sk = skill_name_from_read_path(block["input"].get("file_path", ""))
                if sk:
                    tool_id_to_skill[block["id"]] = sk

    out: list[dict] = [{"role": "system", "content": system_prompt}]
    for msg in anthropic_messages:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            # Either plain text blocks or tool_result blocks.
            tool_results = [b for b in content if b.get("type") == "tool_result"]
            texts = [b["text"] for b in content if b.get("type") == "text"]
            if tool_results:
                for b in tool_results:
                    tid = b["tool_use_id"]
                    body = b["content"]
                    if tid in tool_id_to_skill:
                        body = wrap_context_segment(tool_id_to_skill[tid], body)
                    out.append({"role": "tool", "tool_call_id": tid, "content": body})
            if texts:
                out.append({"role": "user", "content": "\n".join(texts)})
        elif role == "assistant":
            texts = [b["text"] for b in content if b.get("type") == "text"]
            tool_uses = [b for b in content if b.get("type") == "tool_use"]
            a_msg: dict[str, Any] = {"role": "assistant", "content": "\n".join(texts)}
            if tool_uses:
                a_msg["tool_calls"] = [
                    {
                        "id": b["id"],
                        "type": "function",
                        "function": {
                            "name": b["name"],
                            "arguments": json.dumps(b["input"], ensure_ascii=False),
                        },
                    }
                    for b in tool_uses
                ]
            out.append(a_msg)
    return out, tool_id_to_skill


def convert_tools(tools_raw: list[dict]) -> list[dict]:
    """Convert Anthropic-style tool defs (_tools.json) to OpenAI tool schema."""
    out = []
    for t in tools_raw:
        out.append(
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {"type": "object"}),
                },
            }
        )
    return out
