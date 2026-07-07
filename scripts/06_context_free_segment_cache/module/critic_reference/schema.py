"""判官结果的数据结构 + 从 LLM 原始输出里稳健解析 JSON。"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CriterionCheck:
    criterion: str
    verdict: str          # met | partial | unmet
    evidence: str = ""


@dataclass
class JudgeResult:
    """一次评判的结果。success_probability 对齐 OpenHands CriticResult.score 的语义
    （0~1 的成功概率），verdict 对齐 Claude Code verify agent 的 PASS/FAIL/PARTIAL。"""

    success_probability: float
    verdict: str                                   # PASS | FAIL | PARTIAL
    derived_criteria: list[str] = field(default_factory=list)
    checks: list[CriterionCheck] = field(default_factory=list)
    issues: list[dict[str, str]] = field(default_factory=list)
    rationale: str = ""
    raw: str = ""                                  # 判官原始输出，便于复核

    @property
    def success(self) -> bool:
        return self.success_probability >= 0.5


def _extract_json_object(text: str) -> str:
    """从可能带 ```json 围栏或前后废话的文本里，取出第一个花括号平衡的 JSON 对象。"""
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        return fence.group(1)
    start = text.find("{")
    if start == -1:
        raise ValueError("judge output contains no JSON object")
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    raise ValueError("judge output has an unbalanced JSON object")


def parse_judge_json(text: str) -> JudgeResult:
    """把判官返回的文本解析成 JudgeResult。容忍围栏/前后废话；字段缺失给安全默认。"""
    payload: dict[str, Any] = json.loads(_extract_json_object(text))

    checks = [
        CriterionCheck(
            criterion=str(c.get("criterion", "")),
            verdict=str(c.get("verdict", "")).lower(),
            evidence=str(c.get("evidence", "")),
        )
        for c in payload.get("checks", [])
        if isinstance(c, dict)
    ]

    try:
        score = float(payload.get("success_probability"))
    except (TypeError, ValueError):
        score = 0.0
    score = max(0.0, min(1.0, score))

    verdict = str(payload.get("verdict", "")).upper()
    if verdict not in ("PASS", "FAIL", "PARTIAL"):
        verdict = "PASS" if score >= 0.5 else "FAIL"

    issues = [
        {"label": str(i.get("label", "")), "note": str(i.get("note", ""))}
        for i in payload.get("issues", [])
        if isinstance(i, dict)
    ]

    return JudgeResult(
        success_probability=score,
        verdict=verdict,
        derived_criteria=[str(x) for x in payload.get("derived_criteria", [])],
        checks=checks,
        issues=issues,
        rationale=str(payload.get("rationale", "")),
        raw=text,
    )
