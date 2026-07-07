"""Trace loading, Anthropic-to-OpenAI conversion, and skill helpers."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from config import TRACES_DIR
from vllm_client import tokenize_chat


def invocation_sort_key(path: Path) -> tuple[int, int]:
    stem = path.stem
    return (
        int(stem.split("turn_")[1].split("_")[0]),
        int(stem.split("inv_")[1]),
    )


def load_invocations(task: str) -> list[dict[str, Any]]:
    files = sorted(
        (TRACES_DIR / task).glob("turn_*_inv_*.json"),
        key=invocation_sort_key,
    )
    return [json.loads(path.read_text(encoding="utf-8")) for path in files]


def load_system_prompt() -> str:
    return (TRACES_DIR / "_system_prompt.txt").read_text(encoding="utf-8")


def load_tools() -> list[dict[str, Any]]:
    raw = json.loads((TRACES_DIR / "_tools.json").read_text(encoding="utf-8"))
    return convert_tools(raw)


def wrap_context_segment(skill_name: str, text: str) -> str:
    return f'<context_segment id="{skill_name}">\n{text}\n</context_segment>'


def skill_name_from_read_path(path: str) -> str | None:
    p = path.replace("\\", "/")
    if not p.endswith("/SKILL.md"):
        return None
    parts = p.split("/")
    try:
        index = parts.index("skills")
    except ValueError:
        return None
    if index + 1 < len(parts):
        return parts[index + 1]
    return None


def convert_messages(
    anthropic_messages: list[dict[str, Any]],
    system_prompt: str,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    tool_id_to_skill: dict[str, str] = {}
    for msg in anthropic_messages:
        if msg["role"] != "assistant":
            continue
        for block in msg["content"]:
            if block.get("type") == "tool_use" and block.get("name") == "Read":
                skill = skill_name_from_read_path(block["input"].get("file_path", ""))
                if skill:
                    tool_id_to_skill[block["id"]] = skill

    out: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    for msg in anthropic_messages:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            tool_results = [b for b in content if b.get("type") == "tool_result"]
            texts = [b["text"] for b in content if b.get("type") == "text"]
            if tool_results:
                for block in tool_results:
                    """
                    HTTP JSON messages
                    → vLLM 校验
                    → Qwen3 chat template 渲染
                    → tokenizer
                    → token IDs
                    → Qwen3 推理
                    → vLLM parser 解析输出并生成 chatcmpl-tool-* ID

                    当前 Qwen3 只渲染:
                        function.name
                        function.arguments
                        tool content
                    不渲染:
                        tool_calls[].id
                        tool_calls[].type
                        tool.tool_call_id
                    见 Qwen3-14B/Qwen/Qwen3-14B/tokenizer_config.json:230 完整模版
                    """
                    tool_id = block["tool_use_id"]
                    body = block["content"]
                    if tool_id in tool_id_to_skill:
                        body = wrap_context_segment(tool_id_to_skill[tool_id], body)
                    out.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_id,
                            "content": body,
                        }
                    )
            if texts:
                out.append({"role": "user", "content": "\n".join(texts)})
        elif role == "assistant":
            texts = [b["text"] for b in content if b.get("type") == "text"]
            tool_uses = [b for b in content if b.get("type") == "tool_use"]
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": "\n".join(texts),
            }
            if tool_uses:
                assistant_msg["tool_calls"] = [
                    {
                        "id": block["id"],
                        "type": "function",
                        "function": {
                            "name": block["name"],
                            "arguments": json.dumps(
                                block["input"], ensure_ascii=False
                            ),
                        },
                    }
                    for block in tool_uses
                ]
            out.append(assistant_msg)
    return out, tool_id_to_skill


def convert_tools(tools_raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {"type": "object"}),
            },
        }
        for tool in tools_raw
    ]


def find_skill_segments(
    openai_messages: list[dict[str, Any]],
) -> list[tuple[str, int, int, int]]:
    found: list[tuple[str, int, int, int]] = []
    for idx, msg in enumerate(openai_messages):
        if msg.get("role") != "tool":
            continue
        text = msg.get("content", "")
        pos = 0
        while True:
            start = text.find('<context_segment id="', pos)
            if start < 0:
                break
            id_start = start + len('<context_segment id="')
            id_end = text.find('"', id_start)
            name = text[id_start:id_end]
            close = text.find("</context_segment>", id_end)
            end = close + len("</context_segment>")
            found.append((name, idx, start, end))
            pos = end
    return found


def span_token_offsets(
    base_url: str,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    api_key: str,
    msg_idx: int,
    char_start: int,
    char_end: int,
) -> tuple[int, int]:
    msg_text = messages[msg_idx]["content"]
    start_msgs = copy.deepcopy(messages[: msg_idx + 1])
    start_msgs[msg_idx]["content"] = msg_text[:char_start]
    end_msgs = copy.deepcopy(messages[: msg_idx + 1])
    end_msgs[msg_idx]["content"] = msg_text[:char_end]
    start = len(
        tokenize_chat(
            base_url,
            model,
            start_msgs,
            tools,
            api_key,
            add_generation_prompt=False,
        )
    )
    end = len(
        tokenize_chat(
            base_url,
            model,
            end_msgs,
            tools,
            api_key,
            add_generation_prompt=False,
        )
    )
    return start, end


def extract_first_skill_text(task: str, skill: str, system_prompt: str) -> str:
    for invocation in load_invocations(task):
        messages, _ = convert_messages(invocation["messages"], system_prompt)
        for name, msg_idx, char_start, char_end in find_skill_segments(messages):
            if name == skill:
                wrapped = messages[msg_idx]["content"][char_start:char_end]
                prefix = f'<context_segment id="{skill}">\n'
                suffix = "\n</context_segment>"
                if not wrapped.startswith(prefix) or not wrapped.endswith(suffix):
                    raise ValueError(f"Unexpected wrapped skill format for {skill}")
                return wrapped[len(prefix) : -len(suffix)]
    raise KeyError(f"Could not find skill={skill!r} in task={task!r}")


def minimal_skill_messages(skill: str, skill_text: str) -> list[dict[str, Any]]:
    tool_call_id = f"call_read_{skill.replace('-', '_')}"
    wrapped = wrap_context_segment(skill, skill_text)
    return [
        {"role": "system", "content": ""},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": "Read",
                        "arguments": json.dumps(
                            {"file_path": f"/tmp/skills/{skill}/SKILL.md"},
                            ensure_ascii=False,
                        ),
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": tool_call_id, "content": wrapped},
    ]


def target_tokens_for_occurrence(
    base_url: str,
    model: str,
    task: str,
    skill: str,
    occurrence: int,
    system_prompt: str,
    tools: list[dict[str, Any]],
    api_key: str,
) -> list[int]:
    invocation_index = None
    skill_record = None
    from config import SKILL_TOKEN_LOCATIONS

    skill_record = SKILL_TOKEN_LOCATIONS[task]["skills"][skill]
    invocation_index = skill_record["invocation_indices"][occurrence - 1]
    invocation = load_invocations(task)[invocation_index - 1]
    messages, _ = convert_messages(invocation["messages"], system_prompt)
    start, end = skill_record["token_spans"][occurrence - 1]
    tokens = tokenize_chat(
        base_url,
        model,
        messages,
        tools,
        api_key,
        add_generation_prompt=True,
    )
    return tokens[start:end]
