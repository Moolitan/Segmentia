#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
重复相同消息，测量：
1) SDK conversation_stats 的 prompt/completion/cache_read
2) vLLM prefix cache hit rate
3) 每次真实 LLM call 的 messages / usage / 每call命中率
4) 有侵入影响

输出：
- repeated_message_traces.json
- llm_call_prompts.jsonl
- 终端摘要
"""
import types
import argparse
import functools
import inspect
import json
import os
import re
import sys
import threading
import time
import urllib.request
from datetime import datetime, timezone

from openhands.sdk import LLM, Agent, Conversation, Tool, AgentContext, LLMSummarizingCondenser
from openhands.sdk.context.skills import load_skills_from_dir
from openhands.tools.terminal import TerminalTool
from openhands.tools.file_editor import FileEditorTool
from openhands.tools.glob import GlobTool
from openhands.tools.grep import GrepTool
from openhands.tools.apply_patch import ApplyPatchTool

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def to_jsonable(obj):
    """递归转成可被 json.dump 序列化的普通 Python 结构。"""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, list):
        return [to_jsonable(x) for x in obj]

    if isinstance(obj, tuple):
        return [to_jsonable(x) for x in obj]

    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}

    # Pydantic v2
    if hasattr(obj, "model_dump") and callable(obj.model_dump):
        try:
            return to_jsonable(obj.model_dump())
        except Exception:
            pass

    # Pydantic v1
    if hasattr(obj, "dict") and callable(obj.dict):
        try:
            return to_jsonable(obj.dict())
        except Exception:
            pass

    # dataclass / 普通对象
    if hasattr(obj, "__dict__"):
        try:
            return to_jsonable(vars(obj))
        except Exception:
            pass

    # 最后兜底
    return str(obj)

# ---------------------------------------------------------------------------
# vLLM Metrics
# ---------------------------------------------------------------------------

def fetch_vllm_metrics(port: int = 8000) -> dict[str, float]:
    """从 vLLM /metrics 端点解析 Prometheus 格式指标。"""
    url = f"http://localhost:{port}/metrics"
    try:
        resp = urllib.request.urlopen(url, timeout=5).read().decode()
    except Exception:
        return {}

    metrics: dict[str, float] = {}
    for line in resp.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        if len(parts) >= 2:
            name = parts[0].split("{")[0]
            try:
                metrics[name] = float(parts[-1])
            except ValueError:
                pass
    return metrics


# ---------------------------------------------------------------------------
# ANSI 日志清理
# ---------------------------------------------------------------------------

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


class StripAnsiWriter:
    def __init__(self, f):
        self._f = f

    def write(self, s):
        return self._f.write(_ANSI_RE.sub("", s))

    def flush(self):
        self._f.flush()

    def __getattr__(self, name):
        return getattr(self._f, name)


# ---------------------------------------------------------------------------
# Per-call Capture (wrapper-based, 不依赖 telemetry callback)
# ---------------------------------------------------------------------------

class PerCallCapture:
    def __init__(self, vllm_port: int):
        self.vllm_port = vllm_port
        self.calls: list[dict] = []
        self._last_vllm: dict[str, float] = {}
        self._lock = threading.Lock()

    def reset_vllm_baseline(self) -> None:
        """每轮开始前调用，作为本轮 per-call delta 的起点。"""
        self._last_vllm = fetch_vllm_metrics(self.vllm_port)

    def n_calls(self) -> int:
        with self._lock:
            return len(self.calls)

    def get_calls_since(self, n_before: int) -> list[dict]:
        with self._lock:
            return list(self.calls[n_before:])

    def append_call(self, record: dict) -> None:
        with self._lock:
            record["call_index"] = len(self.calls)
            self.calls.append(record)

    def make_record(
        self,
        *,
        method_name: str,
        kwargs: dict,
        response,
        elapsed_seconds: float,
    ) -> dict:
        vllm_after = fetch_vllm_metrics(self.vllm_port)

        with self._lock:
            vllm_before = self._last_vllm
            self._last_vllm = vllm_after

        hits_delta = (
            vllm_after.get("vllm:prefix_cache_hits_total", 0)
            - vllm_before.get("vllm:prefix_cache_hits_total", 0)
        )
        queries_delta = (
            vllm_after.get("vllm:prefix_cache_queries_total", 0)
            - vllm_before.get("vllm:prefix_cache_queries_total", 0)
        )
        hit_rate = hits_delta / queries_delta if queries_delta > 0 else -1.0

        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
        completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
        reasoning_tokens = getattr(usage, "reasoning_tokens", 0) if usage else 0
        cache_read_tokens = getattr(usage, "cache_read_tokens", 0) if usage else 0

        messages = kwargs.get("messages")
        if messages is None:
            messages = kwargs.get("input")
        if messages is None:
            messages = []

        messages_jsonable = to_jsonable(messages)

        messages_summary = []
        if isinstance(messages_jsonable, list):
            for m in messages_jsonable:
                if isinstance(m, dict):
                    messages_summary.append({
                        "role": m.get("role", ""),
                        "chars": len(str(m.get("content", ""))),
                    })

        return {
            "method_name": method_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(elapsed_seconds, 3),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "reasoning_tokens": reasoning_tokens,
            "cache_read_tokens": cache_read_tokens,
            "vllm_prefix_cache_hits_delta": int(hits_delta),
            "vllm_prefix_cache_queries_delta": int(queries_delta),
            "vllm_prefix_cache_hit_rate": round(hit_rate, 6),
            "vllm_gpu_cache_usage": round(
                vllm_after.get("vllm:gpu_cache_usage_perc", -1.0), 6
            ),
            "messages_summary": messages_summary,
            "messages": messages_jsonable,
        }


def install_llm_capture(llm, capture: PerCallCapture):
    """
    给 llm 实例的真实请求方法套 wrapper。
    不依赖 telemetry callback。
    """
    wrapped_any = False

    def wrap_sync_method(method_name: str):
        nonlocal wrapped_any
        if not hasattr(llm, method_name):
            return
        original = getattr(llm, method_name)
        if not callable(original):
            return
        if inspect.iscoroutinefunction(original):
            return

        @functools.wraps(original)
        def wrapper(self, *args, **kwargs):
            t0 = time.time()
            resp = original(*args, **kwargs)
            elapsed = time.time() - t0
            try:
                record = capture.make_record(
                    method_name=method_name,
                    kwargs=kwargs,
                    response=resp,
                    elapsed_seconds=elapsed,
                )
                capture.append_call(record)
            except Exception as e:
                print(f"[capture] failed in {method_name}: {e}", flush=True)
            return resp

        object.__setattr__(llm, method_name, types.MethodType(wrapper, llm))
        wrapped_any = True

    def wrap_async_method(method_name: str):
        nonlocal wrapped_any
        if not hasattr(llm, method_name):
            return
        original = getattr(llm, method_name)
        if not callable(original):
            return
        if not inspect.iscoroutinefunction(original):
            return

        @functools.wraps(original)
        async def wrapper(self, *args, **kwargs):
            t0 = time.time()
            resp = await original(*args, **kwargs)
            elapsed = time.time() - t0
            try:
                record = capture.make_record(
                    method_name=method_name,
                    kwargs=kwargs,
                    response=resp,
                    elapsed_seconds=elapsed,
                )
                capture.append_call(record)
            except Exception as e:
                print(f"[capture] failed in {method_name}: {e}", flush=True)
            return resp

        object.__setattr__(llm, method_name, types.MethodType(wrapper, llm))
        wrapped_any = True

    # 常见候选方法名，哪个存在就包哪个
    for name in ["completion", "chat_completion", "responses_create", "create"]:
        wrap_sync_method(name)
    for name in ["acompletion", "achat_completion", "aresponses_create", "acreate"]:
        wrap_async_method(name)

    if not wrapped_any:
        raise RuntimeError(
            "没有在 llm 实例上找到可包装的请求方法。可用候选属性如下：\n"
            + str([n for n in dir(llm) if "completion" in n or "create" in n or "chat" in n])
        )


# ---------------------------------------------------------------------------
# Agent & LLM 配置
# ---------------------------------------------------------------------------

def build_agent(vllm_port: int, workspace: str, per_call_capture: PerCallCapture):
    """创建 LLM + Agent 实例，并安装真实请求 wrapper。"""
    vllm_api_key = os.environ.get("VLLM_API_KEY", "EMPTY")

    llm = LLM(
        model="openai/Qwen3",
        api_key=vllm_api_key,
        base_url=f"http://localhost:{vllm_port}/v1",
        temperature=0.0,
        max_message_chars=30000,
        stream=False,
        native_tool_calling=True,
        caching_prompt=True,
        prompt_cache_retention="24h",
        drop_params=True,
        modify_params=True,
        reasoning_effort="high",
        extended_thinking_budget=200000,
        litellm_extra_body={},
        log_completions=False,
        disable_vision=True,
        disable_stop_word=False,
        enable_encrypted_reasoning=True,
    )

    install_llm_capture(llm, per_call_capture)

    skills_dir = os.path.join(workspace, ".agents", "skills")
    _, _, agent_skills = load_skills_from_dir(skills_dir)

    agent = Agent(
        llm=llm,
        tools=[
            Tool(name=TerminalTool.name),
            Tool(name=FileEditorTool.name),
            Tool(name=GlobTool.name),
            Tool(name=GrepTool.name),
            Tool(name=ApplyPatchTool.name),
        ],
        include_default_tools=["FinishTool", "ThinkTool"],
        tool_concurrency_limit=1,
        mcp_config={},
        system_prompt_filename="system_prompt.j2",
        system_prompt_kwargs={},
        security_policy_filename="security_policy.j2",
        agent_context=AgentContext(
            skills=list(agent_skills.values()),
            system_message_suffix="始终保持严谨。",
        ),
        condenser=LLMSummarizingCondenser(
            llm=llm,
            max_size=240,
            keep_first=2,
        ),
    )

    return llm, agent, list(agent_skills.keys())


# ---------------------------------------------------------------------------
# 实验执行
# ---------------------------------------------------------------------------

def run_experiment(
    num_turns: int,
    workspace: str,
    vllm_port: int,
    log_path: str,
    calls_output: str,
    message: str = "find a skill for web design. After answering, call the finish tool to end the conversation.",
) -> dict:
    """运行 N 轮相同消息实验，返回结构化结果。"""

    per_call_capture = PerCallCapture(vllm_port)
    llm, agent, skill_names = build_agent(vllm_port, workspace, per_call_capture)

    saved_stdout, saved_stderr = sys.stdout, sys.stderr
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_file = open(log_path, "w", encoding="utf-8")
    writer = StripAnsiWriter(log_file)
    sys.stdout = writer
    sys.stderr = writer

    turn_results = []
    cumulative_prompt_tokens = 0
    experiment_start = time.time()

    os.makedirs(os.path.dirname(calls_output), exist_ok=True)
    calls_jsonl = open(calls_output, "w", encoding="utf-8")

    conversation = None

    try:
        conversation = Conversation(
            agent=agent,
            workspace=workspace,
            max_iteration_per_run=500,
            stuck_detection=True,
            delete_on_close=True,
        )

        for turn_idx in range(num_turns):
            # 1) 轮前快照：SDK metrics
            try:
                metrics_before = conversation.conversation_stats.get_combined_metrics()
                n_usages_before = len(metrics_before.token_usages)
            except Exception:
                metrics_before = None
                n_usages_before = 0

            # 2) 轮前快照：per-call capture 与 vLLM
            n_calls_before = per_call_capture.n_calls()
            per_call_capture.reset_vllm_baseline()
            vllm_before = fetch_vllm_metrics(vllm_port)

            t0 = time.time()

            # 3) 执行
            conversation.send_message(message)
            conversation.run()

            elapsed = time.time() - t0
            vllm_after = fetch_vllm_metrics(vllm_port)

            # 4) 提取本轮 SDK usage 汇总
            prompt_tokens = 0
            completion_tokens = 0
            cache_read_tokens = 0
            cache_write_tokens = 0
            num_llm_calls = 0

            try:
                metrics_after = conversation.conversation_stats.get_combined_metrics()
                new_usages = metrics_after.token_usages[n_usages_before:]

                for u in new_usages:
                    prompt_tokens += getattr(u, "prompt_tokens", 0)
                    completion_tokens += getattr(u, "completion_tokens", 0)
                    cache_read_tokens += getattr(u, "cache_read_tokens", 0)
                    cache_write_tokens += getattr(u, "cache_write_tokens", 0)

                # 恢复旧版稳定口径：本轮新增 usage 数
                num_llm_calls = len(new_usages)
            except Exception:
                pass

            cumulative_prompt_tokens += prompt_tokens

            # 5) 提取本轮 wrapper 抓到的 per-call 记录
            turn_calls = per_call_capture.get_calls_since(n_calls_before)

            # 写完整 JSONL
            for c in turn_calls:
                json.dump(
                    {
                        "turn_number": turn_idx + 1,
                        **c,
                    },
                    calls_jsonl,
                    ensure_ascii=False,
                )
                calls_jsonl.write("\n")

            per_call_summary = [
                {
                    "call_index": c["call_index"],
                    "method_name": c["method_name"],
                    "timestamp": c["timestamp"],
                    "elapsed_seconds": c["elapsed_seconds"],
                    "prompt_tokens": c["prompt_tokens"],
                    "completion_tokens": c["completion_tokens"],
                    "reasoning_tokens": c["reasoning_tokens"],
                    "cache_read_tokens": c["cache_read_tokens"],
                    "vllm_prefix_cache_hits_delta": c["vllm_prefix_cache_hits_delta"],
                    "vllm_prefix_cache_queries_delta": c["vllm_prefix_cache_queries_delta"],
                    "vllm_prefix_cache_hit_rate": c["vllm_prefix_cache_hit_rate"],
                    "vllm_gpu_cache_usage": c["vllm_gpu_cache_usage"],
                    "messages_summary": c["messages_summary"],
                }
                for c in turn_calls
            ]

            # 6) 本轮 vLLM prefix cache 指标（整轮）
            hits_delta = (
                vllm_after.get("vllm:prefix_cache_hits_total", 0)
                - vllm_before.get("vllm:prefix_cache_hits_total", 0)
            )
            queries_delta = (
                vllm_after.get("vllm:prefix_cache_queries_total", 0)
                - vllm_before.get("vllm:prefix_cache_queries_total", 0)
            )
            vllm_cache_hit = hits_delta / queries_delta if queries_delta > 0 else -1.0
            vllm_gpu_usage = vllm_after.get("vllm:gpu_cache_usage_perc", -1.0)

            turn_results.append({
                "turn_number": turn_idx + 1,
                "message": message,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cache_read_tokens": cache_read_tokens,
                "cache_write_tokens": cache_write_tokens,
                "cumulative_prompt_tokens": cumulative_prompt_tokens,
                "num_llm_calls": num_llm_calls,  # 仍以 conversation_stats 为准
                "vllm_prefix_cache_hit_rate": round(vllm_cache_hit, 6),
                "vllm_prefix_cache_hits_tokens": int(hits_delta),
                "vllm_prefix_cache_queries_tokens": int(queries_delta),
                "vllm_gpu_cache_usage": round(vllm_gpu_usage, 6),
                "elapsed_seconds": round(elapsed, 2),
                "llm_calls": per_call_summary,  # 摘要，完整 prompt 在 JSONL
            })

        total_elapsed = time.time() - experiment_start

    finally:
        sys.stdout = saved_stdout
        sys.stderr = saved_stderr
        try:
            log_file.close()
        except Exception:
            pass
        try:
            calls_jsonl.flush()
            calls_jsonl.close()
        except Exception:
            pass
        try:
            if conversation is not None:
                conversation.close()
        except Exception:
            pass

    total_prompt = sum(t["prompt_tokens"] for t in turn_results)
    total_completion = sum(t["completion_tokens"] for t in turn_results)
    total_cache_read = sum(t["cache_read_tokens"] for t in turn_results)

    return {
        "metadata": {
            "experiment": "02_web_repeated_message",
            "num_turns": num_turns,
            "message": message,
            "workspace": workspace,
            "skill_names": skill_names,
            "vllm_port": vllm_port,
            "log_path": os.path.abspath(log_path),
            "calls_output": os.path.abspath(calls_output),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "turns": turn_results,
        "aggregate": {
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_completion,
            "total_cache_read_tokens": total_cache_read,
            "total_elapsed_seconds": round(total_elapsed, 2),
            "total_llm_calls": sum(t["num_llm_calls"] for t in turn_results),
            "cache_read_ratio": round(total_cache_read / max(1, total_prompt), 6),
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--turns", type=int, default=5, help="重复轮数（默认 5）")
    parser.add_argument("--workspace", default=os.path.join(ROOT, "workspace", "01_web"))
    parser.add_argument("--output", default=os.path.join(ROOT, "results", "01_web", "repeated_message_traces.json"))
    parser.add_argument("--calls-output", default=os.path.join(ROOT, "results", "01_web", "llm_call_prompts.jsonl"))
    parser.add_argument("--log", default=os.path.join(ROOT, "log", "01_web_skills_test.log"))
    parser.add_argument("--vllm-port", type=int, default=8000)
    args = parser.parse_args()

    print(f"[实验] {args.turns} 轮相同消息, workspace={args.workspace}")

    result = run_experiment(
        num_turns=args.turns,
        workspace=args.workspace,
        vllm_port=args.vllm_port,
        log_path=args.log,
        calls_output=args.calls_output,
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存: {args.output}")
    print(f"per-call prompts: {args.calls_output}")
    print(f"日志: {args.log}")
    print(f"\n{'轮次':>4}  {'calls':>5}  {'prompt':>8}  {'completion':>10}  {'cache_read':>10}  {'cache_hit%':>10}  {'耗时':>6}")
    print("-" * 72)
    for t in result["turns"]:
        print(
            f"{t['turn_number']:>4}  {t['num_llm_calls']:>5}  "
            f"{t['prompt_tokens']:>8}  {t['completion_tokens']:>10}  "
            f"{t['cache_read_tokens']:>10}  {t['vllm_prefix_cache_hit_rate']:>9.4f}  "
            f"{t['elapsed_seconds']:>5.1f}s"
        )
    print("-" * 72)
    agg = result["aggregate"]
    print(
        f"总计  {agg['total_llm_calls']:>5}  {agg['total_prompt_tokens']:>8}  "
        f"{agg['total_completion_tokens']:>10}  {agg['total_cache_read_tokens']:>10}  "
        f"ratio={agg['cache_read_ratio']:.4f}  {agg['total_elapsed_seconds']:>5.1f}s"
    )


if __name__ == "__main__":
    main()