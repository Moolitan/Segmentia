#!/usr/bin/env python3
"""Run the real slack_launch_pack OpenHands agent bench with ContextSegmentKV.

This reuses scripts/03_14B_anthropic/run_multurn3.py for the actual agent,
Conversation, tools, workspace, and benchmark turns. The only added behavior is:

1. Before the benchmark, offline-prefill the expected context segments
   (SKILL.md files) and save/register their KV in vLLM.
2. Wrap the real LLM transport call. If a later real agent request contains one
   of those exact segment texts outside the system message, compute its token
   span with vLLM /tokenize and attach vllm_xargs.context_segment_cache.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RUN_MULTURN3_PATH = ROOT / "scripts" / "03_14B_anthropic" / "run_multurn3.py"
DEFAULT_BENCH_ROOT = ROOT / "anthropic_skill_benchmark_8_repos_explicit_skills"
DEFAULT_REPO = "slack_launch_pack"
DEFAULT_SKILLS = ["internal-comms", "slack-gif-creator", "brand-guidelines"]


def load_run_multurn3():
    """动态加载真实 benchmark runner。

    这里不重新实现 OpenHands agent 流程，而是复用
    scripts/03_14B_anthropic/run_multurn3.py 里的真实多轮执行逻辑。
    这样后面的实验仍然跑 slack_launch_pack 的真实 agent transcript、
    workspace、tools 和 benchmark turns。
    """
    spec = importlib.util.spec_from_file_location(
        "run_multurn3_real_context_segment", RUN_MULTURN3_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {RUN_MULTURN3_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def post_json(base_url: str, path: str, payload: dict[str, Any], api_key: str) -> dict:
    """
    向 vLLM OpenAI-compatible server 发送 JSON POST 请求

    本脚本只依赖标准库发 HTTP 请求,避免引入额外 client 行为
    用于调用 /tokenize 和 /v1/chat/completions
    """
    data = json.dumps(payload).encode("utf-8")
    # 进入 vLLM 的 OpenAI-compatible chat completion endpoint
    # vllm/vllm/v1/request.py
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=720) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {path}: {body}") from exc


def tokenize_chat(
    base_url: str,
    model: str,
    messages: list[dict[str, Any]],
    api_key: str,
    *,
    add_generation_prompt: bool,
) -> list[int]:
    """用 vLLM 的 chat template 计算 messages 对应的 token ids。

    ContextSegmentKV 注入需要的是 token 级 span，而真实 agent 请求是
    chat messages。因此这里必须走 vLLM /tokenize，保证 tokenization、
    chat template、enable_thinking 等配置和实际 generation 请求一致。
    """
    response = post_json(
        base_url,
        "/tokenize",
        {
            "model": model,
            "messages": messages,
            "add_generation_prompt": add_generation_prompt,
            "chat_template_kwargs": {"enable_thinking": True},
        },
        api_key,
    )
    return list(response["tokens"])


def chat_completion(
    base_url: str,
    model: str,
    messages: list[dict[str, Any]],
    api_key: str,
    *,
    max_tokens: int,
    request_id: str,
    context_segment_cache: dict[str, Any] | None = None,
) -> tuple[dict, float]:
    """向 vLLM 发起一次 chat completion,并可携带 ContextSegmentKV 配置

    离线阶段传 sources,让 vLLM 在完成 prefill 后收集并保存指定 span 的 KV
    在线 agent 阶段传 targets,让 vLLM 在目标 span 位置注入已保存的 KV
    返回原始 response 和耗时,便于 trace 记录
    """
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "request_id": request_id,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    if context_segment_cache is not None:
        payload["vllm_xargs"] = {
            "context_segment_cache": json.dumps(
                context_segment_cache, ensure_ascii=False, sort_keys=True
            )
        }
    start = time.time()
    # POST 到 /v1/chat/completions
    # 从这里开始，请求进入 vLLM
    response = post_json(base_url, "/v1/chat/completions", payload, api_key)
    return response, time.time() - start


def text_from_content(content: Any) -> str:
    """
    把 OpenAI message content 规范化成可搜索的纯文本
    OpenHands/LiteLLM 的 content 可能是字符串，也可能是包含 text part 的list
    为了判断某个 SKILL.md 是否出现在真实请求里，这里先统一转成文本
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            else:
                parts.append(json.dumps(item, ensure_ascii=False, sort_keys=True))
        return "\n".join(parts)
    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=False, sort_keys=True)
    return str(content)


def set_text_content(msg: dict[str, Any], new_text: str) -> dict[str, Any]:
    """复制一个 message，并把其中的文本内容替换成 new_text。

    这个函数用于构造“截断到某个字符位置”的临时 messages，然后调用
    /tokenize 计算该字符位置对应的 token offset。原始 message 不会被修改。
    """
    out = copy.deepcopy(msg)
    content = out.get("content")
    if isinstance(content, str) or content is None:
        out["content"] = new_text
        return out
    if isinstance(content, list):
        remaining = new_text
        new_items = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                original = str(item.get("text", ""))
                take = remaining[: len(original)]
                remaining = remaining[len(original) :]
                new_item = dict(item)
                new_item["text"] = take
                new_items.append(new_item)
                if not remaining:
                    break
            else:
                new_items.append(item)
        out["content"] = new_items
        return out
    out["content"] = new_text
    return out


def locate_segment(
    messages: list[dict[str, Any]], segment_text: str
) -> tuple[int, int, int] | None:
    """
    在真实 agent messages 中查找某段已离线缓存的文本
    返回 (message_index, char_start, char_end)
    messages = [
      {"role": "system", "content": "..."},
      {"role": "user", "content": "hello"},
      {"role": "assistant", "content": "hi"},
      {"role": "user", "content": "SKILL.md text here"},]
    idx: 为列表的索引
    msg: 为列表me e
    """

    for idx, msg in enumerate(messages):
        if msg.get("role") == "system":
            continue
        text = text_from_content(msg.get("content"))
        pos = text.find(segment_text)
        if pos >= 0:
            return idx, pos, pos + len(segment_text)
    return None


def span_token_offsets_for_message_text(
    base_url: str,
    model: str,
    messages: list[dict[str, Any]],
    api_key: str,
    msg_idx: int,
    char_start: int,
    char_end: int,
) -> tuple[int, int]:
    """
    把某个 message 内的字符 span 转换为整个 chat prompt 的 token span。

    做法是分别构造“截至 char_start”和“截至 char_end”的 messages 前缀,
    交给 vLLM /tokenize,两个 token 长度就是该文本片段在完整 prompt 中的
    target_start/source_start 与 target_end/source_end。
    """
    msg_text = text_from_content(messages[msg_idx].get("content"))

    start_messages = copy.deepcopy(messages[: msg_idx + 1])
    start_messages[msg_idx] = set_text_content(
        start_messages[msg_idx], msg_text[:char_start]
    )
    end_messages = copy.deepcopy(messages[: msg_idx + 1])
    end_messages[msg_idx] = set_text_content(end_messages[msg_idx], msg_text[:char_end])

    start = len(
        tokenize_chat(
            base_url, model, start_messages, api_key, add_generation_prompt=False
        )
    )
    end = len(
        tokenize_chat(base_url, model, end_messages, api_key, add_generation_prompt=False)
    )
    return start, end


def offline_prefill_segment(
    base_url: str,
    model: str,
    api_key: str,
    cache_id: str,
    segment_text: str,
) -> dict[str, Any]:
    """
    离线 prefill 一个 context segment,并请求 vLLM 保存它的 KV。

    这里把单个 SKILL.md 当成 user message 发送给 vLLM。先计算该文本在
    chat template 下的 source token span,再通过 vllm_xargs.context_segment_cache
    的 sources 字段告诉 vLLM:这次 prefill 结束后收集这个 span 的 KV,
    用 cache_id 存起来。
    """
    messages = [{"role": "user", "content": segment_text}]
    source_start, source_end = span_token_offsets_for_message_text(
        base_url,
        model,
        messages,
        api_key,
        msg_idx=0,
        char_start=0,
        char_end=len(segment_text),
    )
    cfg = {
        "sources": [
            {
                "cache_id": cache_id,
                "source_start": source_start,
                "source_end": source_end,
            }
        ]
    }
    _, elapsed = chat_completion(
        base_url,
        model,
        messages,
        api_key,
        max_tokens=1,
        request_id=f"ctxseg-offline-{cache_id}",
        context_segment_cache=cfg,
    )
    print("source_start", source_start, "source_end", source_end, "elapsed", elapsed)
    return {
        "cache_id": cache_id,
        "source_start": source_start,
        "source_end": source_end,
        "source_tokens": source_end - source_start,
        "elapsed_seconds": round(elapsed, 6),
    }


def attach_context_segment_injector(
    llm,
    *,
    base_url: str,
    model: str,
    api_key: str,
    segments: dict[str, str],
) -> None:
    if getattr(llm, "_context_segment_kv_patched", False):
        # True 则return
        # 否则赋值 True 并继续往下走
        return
    original = getattr(llm, "_transport_call")
    llm._context_segment_kv_patched = True
    llm._context_segment_kv_events = []
    llm._context_segment_kv_seen_spans = set()

    def wrapped(*args, **kwargs):
        """
        实际替换 llm._transport_call 的包装函数。
        """
        messages = kwargs.get("messages") or []
        injections = []
        seen_spans_to_mark = []
        for cache_id, segment_text in segments.items():
            found = locate_segment(messages, segment_text)
            if found is None:
                continue
            msg_idx, char_start, char_end = found
            span_key = (cache_id, msg_idx, char_start, char_end)
            if span_key in llm._context_segment_kv_seen_spans:
                continue
            target_start, target_end = span_token_offsets_for_message_text(
                base_url,
                model,
                messages,
                api_key,
                msg_idx,
                char_start,
                char_end,
            )
            injections.append(
                {
                    "cache_id": cache_id,
                    "mode": "rope",
                    "target_start": target_start,
                    "target_end": target_end,
                }
            )
            seen_spans_to_mark.append(span_key)

        if injections:
            llm._context_segment_kv_seen_spans.update(seen_spans_to_mark)
            extra_body = dict(kwargs.get("extra_body") or {})
            vllm_xargs = dict(extra_body.get("vllm_xargs") or {})
            vllm_xargs["context_segment_cache"] = json.dumps(
                {"targets": injections}, ensure_ascii=False, sort_keys=True
            )
            extra_body["vllm_xargs"] = vllm_xargs
            kwargs["extra_body"] = extra_body
            llm._context_segment_kv_events.append(
                {
                    "request_index": len(llm._context_segment_kv_events),
                    "injections": injections,
                    "num_messages": len(messages),
                }
            )
        return original(*args, **kwargs)

    object.__setattr__(wrapped, "_context_segment_kv_wrapped", True)
    object.__setattr__(llm, "_transport_call", wrapped)


def prepare_workspace(repo: str, bench_root: Path, workspace: Path, skills_src: Path) -> None:
    """准备 slack_launch_pack 真实 agent 运行目录。

    复制 benchmark seed_files，并把 repo 根目录的 skills 复制到 workspace
    下的 .agents/skills。这样后续 run_multurn3.py 创建的 OpenHands agent
    看到的是一次真实 benchmark run 应该看到的 workspace 布局。
    """
    workspace.mkdir(parents=True, exist_ok=True)
    bench_dir = bench_root / repo
    seed_src = bench_dir / "seed_files"
    if seed_src.is_dir():
        dst = workspace / "seed_files"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(seed_src, dst)

    skills_dst = workspace / ".agents" / "skills"
    if skills_dst.exists():
        shutil.rmtree(skills_dst)
    skills_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(skills_src, skills_dst)


def main() -> None:

    parser = argparse.ArgumentParser(
        description="Run real slack_launch_pack multiturn agent with ContextSegmentKV."
    )
    parser.add_argument("--benchmark-repo", default=DEFAULT_REPO)
    parser.add_argument("--bench-root", default=str(DEFAULT_BENCH_ROOT))
    parser.add_argument("--workspace", default=str(ROOT / "workspace" / "05_context_segment_agent_kv" / DEFAULT_REPO))
    parser.add_argument("--skills-dir", default=str(ROOT / "skills"))
    parser.add_argument("--output", default=str(ROOT / "results" / "05_context_segment_agent_kv" / DEFAULT_REPO / "multiturn_sequence_traces.json"))
    parser.add_argument("--log-dir", default=str(ROOT / "log" / "05_context_segment_agent_kv"))
    parser.add_argument("--vllm-port", type=int, default=int(os.environ.get("VLLM_PORT", "8000")))
    parser.add_argument("--model", default=os.environ.get("VLLM_SERVED_NAME", "Qwen3"))
    parser.add_argument("--max-iteration-per-run", type=int, default=500)
    parser.add_argument("--cache-skill", action="append", default=None, help="Skill directory name to offline-cache. Can be repeated.")
    parser.add_argument(
        "--offline-only",
        action="store_true",
        help="Only prefill and save context segment KV records, then exit.",
    )
    args = parser.parse_args()

    bench_root = Path(args.bench_root)
    workspace = Path(args.workspace)
    skills_src = Path(args.skills_dir)
    output = Path(args.output)
    log_dir = Path(args.log_dir)
    base_url = f"http://127.0.0.1:{args.vllm_port}"
    api_key = os.environ.get("VLLM_API_KEY", "EMPTY")
    skill_names = args.cache_skill or DEFAULT_SKILLS

    prepare_workspace(args.benchmark_repo, bench_root, workspace, skills_src)
    active_skills_dir = workspace / ".agents" / "skills"

    segments: dict[str, str] = {}
    offline_records = []
    for skill_name in skill_names:
        skill_path = active_skills_dir / skill_name / "SKILL.md"
        text = skill_path.read_text(encoding="utf-8")
        cache_id = f"context-segment-{skill_name}-v1"
        segments[cache_id] = text
        offline_records.append(
            offline_prefill_segment(base_url, args.model, api_key, cache_id, text)
        )

    if args.offline_only:
        final = {
            "benchmark_repo": args.benchmark_repo,
            "workspace": str(workspace),
            "skills_dir": str(active_skills_dir),
            "offline_context_segments": offline_records,
            "context_segment_kv_events": [],
            "llm_calls": [],
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(final, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[done] offline-only result: {output}")
        print(f"[done] offline segments: {len(offline_records)}")
        return

    runner = load_run_multurn3()
    template = runner.load_benchmark_sequence(args.benchmark_repo, bench_root=str(bench_root))
    llm, agent = runner.create_agent_and_llm(str(active_skills_dir), args.vllm_port)
    attach_context_segment_injector(
        llm,
        base_url=base_url,
        model=args.model,
        api_key=api_key,
        segments=segments,
    )

    log_dir.mkdir(parents=True, exist_ok=True)
    result = runner.run_sequence(
        template=template,
        agent=agent,
        seq_workspace=str(workspace),
        seq_log_path=str(log_dir / f"{args.benchmark_repo}.log"),
        max_iteration_per_run=args.max_iteration_per_run,
    )

    final = {
        "benchmark_repo": args.benchmark_repo,
        "workspace": str(workspace),
        "skills_dir": str(active_skills_dir),
        "offline_context_segments": offline_records,
        "context_segment_kv_events": getattr(llm, "_context_segment_kv_events", []),
        "llm_calls": result["llm_calls"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(final, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[done] real multiturn result: {output}")
    print(f"[done] log: {log_dir / f'{args.benchmark_repo}.log'}")
    print(f"[done] offline segments: {len(offline_records)}")
    print(f"[done] injected requests: {len(final['context_segment_kv_events'])}")


if __name__ == "__main__":
    main()
