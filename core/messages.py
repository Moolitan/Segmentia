from __future__ import annotations

import json


def message_content_to_text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                item_type = item.get("type", "unknown")
                if item_type == "text":
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(json.dumps(item, ensure_ascii=False, sort_keys=True))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=False, sort_keys=True)
    return str(content)


def parse_tool_activity(curr_attempt: dict, next_attempt: dict | None) -> dict:
    """Recover tool calls and tool results caused by one LLM request."""
    if next_attempt is None:
        return {"tool_calls": [], "tool_results": []}

    curr_msgs = curr_attempt.get("messages") or []
    next_msgs = next_attempt.get("messages") or []
    n = len(curr_msgs)
    if len(next_msgs) <= n or next_msgs[:n] != curr_msgs:
        return {"tool_calls": [], "tool_results": []}

    tool_calls: list[dict] = []
    tool_results: list[dict] = []
    for m in next_msgs[n:]:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role == "assistant":
            for tc in m.get("tool_calls") or []:
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function") or {}
                tool_calls.append(
                    {
                        "id": tc.get("id"),
                        "name": fn.get("name") if isinstance(fn, dict) else None,
                        "arguments": (
                            fn.get("arguments") if isinstance(fn, dict) else None
                        ),
                    }
                )
        elif role == "tool":
            tool_results.append(
                {
                    "tool_call_id": m.get("tool_call_id"),
                    "name": m.get("name"),
                    "content": message_content_to_text(m.get("content")),
                }
            )

    return {"tool_calls": tool_calls, "tool_results": tool_results}


def flatten_request_messages(messages) -> str:
    """Deterministically serialize an OpenAI-style messages list."""
    if not isinstance(messages, list):
        return ""
    parts = []
    for msg in messages:
        if not isinstance(msg, dict):
            parts.append(str(msg))
            continue
        role = msg.get("role", "unknown")
        content_text = message_content_to_text(msg.get("content"))
        header = f"[{role}]"
        if msg.get("name"):
            header += f" name={msg['name']}"
        if msg.get("tool_call_id"):
            header += f" tool_call_id={msg['tool_call_id']}"
        tc_text = ""
        if msg.get("tool_calls"):
            tc_text = "\ntool_calls=" + json.dumps(
                msg["tool_calls"], ensure_ascii=False, sort_keys=True, default=str
            )
        parts.append(f"{header}\n{content_text}{tc_text}")
    return "\n\n".join(parts)
