"""从现有实验数据里取出判官的两个输入：任务要求 + agent 产物。

- 任务要求：invocation 里第一条 user message（含"用哪个 skill、先读哪些文件、要写
  什么、增量推进"等要求）。
- agent 产物：raw decode TXT 里 `</think>` 之后的动作（tool call）；解析出工具名和
  arguments（如 Write 的 content），拼成一段可判文本。
"""
from __future__ import annotations

import json
import re
from typing import Any


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return str(content) if content is not None else ""


def task_request_text(invocation: dict[str, Any]) -> str:
    """取**当前这步**的任务要求。

    trace 是多轮的，每个 invocation 是其中一步。当前步要响应的指令是**最后一条有文本
    的 user 消息**（例如 "用 doc-coauthoring skill…写 short spec"），而不是第一条
    （那可能是更早、无关这步的指令）。中间那些无文本的 user 消息是 tool_result（skill
    注入等），跳过。每步指令都会自带 "reference and use the '<skill>' skill" 等约束，
    所以取最后一条文本指令即可自洽。
    """
    for message in reversed(invocation.get("messages", [])):
        if message.get("role") == "user":
            text = _message_text(message).strip()
            if text:
                return text
    return ""


def split_think_and_action(raw_text: str) -> tuple[str, str]:
    """把 raw decode TXT 拆成 (thinking, action)。action = </think> 之后、im_end 之前。"""
    think = ""
    match = re.search(r"<think>\s*(.*?)\s*</think>", raw_text, re.DOTALL)
    if match:
        think = match.group(1).strip()
        action = raw_text[match.end():]
    else:
        action = raw_text
    action = re.sub(r"<\|im_end\|>.*$", "", action, flags=re.DOTALL).strip()
    return think, action


def parse_tool_call(action_text: str) -> dict[str, Any] | None:
    """从 action 文本里解析 <tool_call>{json}</tool_call>，返回 {name, arguments}。"""
    match = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", action_text, re.DOTALL)
    blob = match.group(1) if match else None
    if blob is None:
        # 有些结束符是纯 JSON 无标签，退而求其次
        brace = action_text.find("{")
        if brace == -1:
            return None
        blob = action_text[brace:]
    try:
        obj = json.loads(blob)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or "name" not in obj:
        return None
    return {"name": obj.get("name"), "arguments": obj.get("arguments")}


def agent_output_text(raw_text: str, include_thinking: bool = True) -> str:
    """把 raw decode TXT 整理成判官可读的 AGENT_OUTPUT：思考摘要 + 动作(工具名+参数)。"""
    think, action = split_think_and_action(raw_text)
    call = parse_tool_call(action)
    parts: list[str] = []
    if include_thinking and think:
        parts += ["[reasoning]", think]
    if call is not None:
        args = call.get("arguments")
        args_str = json.dumps(args, ensure_ascii=False, indent=2) if not isinstance(args, str) else args
        parts += ["", f"[action] tool = {call.get('name')}", "[action arguments]", args_str]
    else:
        parts += ["", "[action] (no parseable tool call)", action]
    return "\n".join(parts).strip()
