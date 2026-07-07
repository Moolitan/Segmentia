"""通用判官：读「任务要求 + agent 产物」→ 调一个 LLM → 结构化评判结果。

判官模型走 OpenAI 兼容的 /v1/chat/completions，默认指向本地 vLLM（就是我们在跑的
Qwen3-14B）。想换成 GPT / MiniMax 等，只需改 base_url / model / api_key，无需改代码。

[判官用谁的取舍] 本地 Qwen3-14B：快、离线、可大批量，但和被判的 agent 同族，有轻微
"自己夸自己"偏差；跨模型（GPT-5.4 / MiniMax）更权威、无同族偏差，但外部有成本。
默认本地起步，论文阶段建议再用跨模型对一部分 case 交叉验证。
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Sequence

from prompt import JUDGE_SYSTEM_PROMPT
from schema import JudgeResult, parse_judge_json


class GeneralJudge:
    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1536,
        timeout: float = 300.0,
        enable_thinking: bool = False,
    ) -> None:
        # 默认本地 vLLM；可用环境变量或参数覆盖以指向任意 OpenAI 兼容端点。
        self.base_url = (base_url or os.environ.get("JUDGE_BASE_URL", "http://127.0.0.1:8000")).rstrip("/")
        self.model = model or os.environ.get("JUDGE_MODEL", "Qwen3")
        self.api_key = api_key or os.environ.get("JUDGE_API_KEY", "EMPTY")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.enable_thinking = enable_thinking

    # ------------------------------------------------------------------
    def build_user_message(
        self,
        task_request: str,
        agent_output: str,
        tools: Sequence[dict] | None = None,
    ) -> str:
        """把任务要求 + agent 产物拼成判官的 user 输入。"""
        parts = ["=== TASK ===", task_request.strip(), "", "=== AGENT_OUTPUT ===", agent_output.strip()]
        if tools:
            names = [t.get("function", {}).get("name", t.get("name", "?")) for t in tools]
            parts += ["", "=== TOOLS AVAILABLE ===", ", ".join(str(n) for n in names)]
        return "\n".join(parts)

    def _post_chat(self, messages: list[dict]) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=data,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"judge HTTP {exc.code}: {detail}") from exc
        choice = (body.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        return message.get("content") or ""

    # ------------------------------------------------------------------
    def evaluate(
        self,
        task_request: str,
        agent_output: str,
        tools: Sequence[dict] | None = None,
    ) -> JudgeResult:
        """对一次 agent 产出打分。任意任务通用：判官自己从 task_request 抽标准。"""
        messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": self.build_user_message(task_request, agent_output, tools)},
        ]
        content = self._post_chat(messages)
        return parse_judge_json(content)
