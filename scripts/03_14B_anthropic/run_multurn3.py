
import argparse
import copy
import hashlib
import json
import os
import re
import sys
import traceback
import time
from dataclasses import dataclass, field

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BENCH_ROOT = os.path.join(ROOT, "anthropic_skill_benchmark")  # default; overridden by --bench-root
sys.path.insert(0, ROOT)


# from benchkit.metrics.vllm_prefix_cache import (
#     VllmPrefixCacheSample,
#     compute_vllm_prefix_cache_delta,
# )


@dataclass
class TaskSpec:
    """Per-turn task (built from <repo>/turns/turn_N.txt)."""
    task_id: str
    message: str
    description: str


@dataclass
class SequenceTemplate:
    """A benchmark sequence (all turns under one <repo>/)."""
    template_id: str
    description: str
    turns: list[TaskSpec] = field(default_factory=list)

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



def _message_content_to_text(content) -> str:
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
    """还原本次 LLM 调用实际触发了哪些工具，以及这些工具的执行结果。
    """
    # 没有下一次调用（通常是 turn 最后一次，例如 FinishTool 或纯文本收尾），
    # 无法从后续 messages 中回溯工具活动。
    if next_attempt is None:
        return {"tool_calls": [], "tool_results": []}

    curr_msgs = curr_attempt.get("messages") or []
    next_msgs = next_attempt.get("messages") or []
    n = len(curr_msgs)

    # 健壮性检查：下一次 messages 必须严格在当前基础上追加（前缀一致且更长）。
    # 若 LLMSummarizingCondenser 触发压缩，前缀会被重写，这里直接放弃本次归因。
    if len(next_msgs) <= n or next_msgs[:n] != curr_msgs:
        return {"tool_calls": [], "tool_results": []}

    tool_calls: list[dict] = []
    tool_results: list[dict] = []

    # next_msgs[n:] 即本次 LLM 调用之后新追加的消息：
    # 通常是 1 条 assistant（带规范化后的 tool_calls）+ 若干条 role=tool（执行结果）。
    for m in next_msgs[n:]:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role == "assistant":
            # 模型决定调用的工具：提取 id / 名称 / 参数（JSON 字符串）
            for tc in (m.get("tool_calls") or []):
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function") or {}
                tool_calls.append({
                    "id": tc.get("id"),
                    "name": fn.get("name") if isinstance(fn, dict) else None,
                    "arguments": fn.get("arguments") if isinstance(fn, dict) else None,
                })
        elif role == "tool":
            # 工具执行结果：通过 tool_call_id 与上面的 tool_calls 一一对应
            tool_results.append({
                "tool_call_id": m.get("tool_call_id"),
                "name": m.get("name"),
                "content": _message_content_to_text(m.get("content")),
            })

    return {"tool_calls": tool_calls, "tool_results": tool_results}


def flatten_request_messages(messages) -> str:
    """Deterministically serialize an OpenAI-style messages list.

    Includes ``tool_calls`` (assistant) and ``tool_call_id`` / ``name`` (tool)
    in addition to ``content``. Without these, two messages whose text content
    matches but whose tool-call args differ would flatten identically, causing
    false-positive prefix-hash matches downstream.
    """
    if not isinstance(messages, list):
        return ""
    parts = []
    for msg in messages:
        if not isinstance(msg, dict):
            parts.append(str(msg))
            continue
        role = msg.get("role", "unknown")
        content_text = _message_content_to_text(msg.get("content"))
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


def attach_llm_request_attempt_collector(llm, vllm_port: int) -> None:
    """
    Collect one unified record per actual transport attempt so request text
    and vLLM deltas stay aligned.
    """
    if getattr(llm, "_request_attempt_patched", False):
        return

    llm._request_attempt_patched = True
    llm._request_attempts: list[dict] = []
    llm._request_attempt_vllm_metrics_supported = True

    original = getattr(llm, "_transport_call", None)
    if not callable(original):
        return

    def _sample_vllm():
        if not llm._request_attempt_vllm_metrics_supported:
            return None
        try:
            return VllmPrefixCacheSample.sample(vllm_port=vllm_port)
        except Exception:
            llm._request_attempt_vllm_metrics_supported = False
            return None

    def wrapped(*args, **kwargs):
        messages = kwargs.get("messages")
        before = _sample_vllm()
        started_at = time.time()
        error = None
        try:
            return original(*args, **kwargs)
        except Exception as exc:
            error = exc
            raise
        finally:
            ended_at = time.time()
            after = _sample_vllm() if before is not None else None
            attempt = {
                "messages": copy.deepcopy(messages),
                "transport_started_at": started_at,
                "transport_ended_at": ended_at,
                "transport_elapsed_seconds": round(ended_at - started_at, 6),
                "error_type": type(error).__name__ if error is not None else None,
                "error_message": str(error) if error is not None else None,
                "vllm_prefix_cache_queries_tokens": None,
                "vllm_prefix_cache_hits_tokens": None,
                "vllm_prefix_cache_hit_rate": None,
                "vllm_request_prefill_time_seconds": None,
                "vllm_time_to_first_token_seconds": None,
            }
            if before is not None and after is not None:
                try:
                    attempt.update(compute_vllm_prefix_cache_delta(before, after))
                except Exception:
                    pass
            llm._request_attempts.append(attempt)

    object.__setattr__(wrapped, "_request_attempt_wrapped", True)
    object.__setattr__(llm, "_transport_call", wrapped)


def load_skill_doc(skills_dir: str, skill_name: str) -> tuple[str | None, str | None]:
    skill_path = os.path.join(skills_dir, skill_name, "SKILL.md")
    if not os.path.isfile(skill_path):
        return None, None
    with open(skill_path, encoding="utf-8") as f:
        return skill_path, f.read()



def run_sequence(
    template: SequenceTemplate,
    agent,
    seq_workspace: str,
    seq_log_path: str,
    max_iteration_per_run: int = 500,
) -> dict:
    from openhands.sdk import Conversation

    os.makedirs(seq_workspace, exist_ok=True)

    saved_stdout, saved_stderr = sys.stdout, sys.stderr
    os.makedirs(os.path.dirname(seq_log_path), exist_ok=True)
    seq_log_file = open(seq_log_path, "w", encoding="utf-8")
    seq_writer = StripAnsiWriter(seq_log_file)
    sys.stdout = seq_writer
    sys.stderr = seq_writer

    conversation = Conversation(
        agent=agent,
        workspace=seq_workspace,
        max_iteration_per_run=max_iteration_per_run,
        stuck_detection=True,
        delete_on_close=True,
    )

    all_llm_calls: list[dict] = []
    seq_start = time.time()
    try:
        for i, turn_spec in enumerate(template.turns):
            turn_number = i + 1
            print(
                f"\n{'=' * 60}\n"
                f"[TURN {turn_number}/{len(template.turns)}] {turn_spec.description}\n"
                f"{'=' * 60}\n",
                flush=True,
            )
            seq_log_file.flush()

            message = (
                f"Working directory: {seq_workspace}\n\n"
                + turn_spec.message
            )

            llm = agent.llm
            request_attempts_before = len(getattr(llm, "_request_attempts", []))

            conversation.send_message(message)
            conversation.run()
            seq_log_file.flush()

            turn_attempts = getattr(llm, "_request_attempts", [])[request_attempts_before:]

            for k, attempt in enumerate(turn_attempts):
                next_attempt = turn_attempts[k + 1] if k + 1 < len(turn_attempts) else None
                tool_activity = parse_tool_activity(attempt, next_attempt)
                record = {
                    "call_index_in_turn": k,
                    "turn_number": turn_number,
                    "request_prompt_text": flatten_request_messages(
                        attempt.get("messages", [])
                    ),
                    "tool_calls": tool_activity["tool_calls"],
                    "vllm_prefix_cache_queries_tokens": attempt.get(
                        "vllm_prefix_cache_queries_tokens"
                    ),
                    "vllm_prefix_cache_hits_tokens": attempt.get(
                        "vllm_prefix_cache_hits_tokens"
                    ),
                    "vllm_prefix_cache_hit_rate": attempt.get(
                        "vllm_prefix_cache_hit_rate"
                    ),
                    "vllm_request_prefill_time_seconds": attempt.get(
                        "vllm_request_prefill_time_seconds"
                    ),
                    "vllm_time_to_first_token_seconds": attempt.get(
                        "vllm_time_to_first_token_seconds"
                    ),
                }

                all_llm_calls.append(record)
        _ = time.time() - seq_start

    finally:
        try:
            conversation.close()
        except Exception:
            pass
        sys.stdout = saved_stdout
        sys.stderr = saved_stderr
        seq_log_file.close()

    return {
        "llm_calls": all_llm_calls
    }


def create_agent_and_llm(
    skills_dir: str,
    vllm_port: int = 8000,
):
    """创建 LLM 和 Agent 实例。"""
    from openhands.sdk import LLM, Agent, AgentContext, LLMSummarizingCondenser, Tool
    from openhands.sdk.context.skills import load_skills_from_dir
    from openhands.tools.terminal import TerminalTool
    # from openhands.tools.file_editor import FileEditorTool
    from openhands.tools.glob import GlobTool
    from openhands.tools.grep import GrepTool
    from openhands.tools.apply_patch import ApplyPatchTool
    from openhands.tools.task_tracker import TaskTrackerTool
    from openhands.tools.task import TaskToolSet
    from openhands.tools.browser_use import BrowserToolSet
    from openhands.tools.gemini import ListDirectoryTool, EditTool, ReadFileTool, WriteFileTool
    from openhands.tools.skill import SkillTool


    vllm_api_key = os.environ.get("VLLM_API_KEY", "EMPTY")

    llm = LLM(
        model="openai/Qwen3",
        api_key=vllm_api_key,
        base_url=f"http://localhost:{vllm_port}/v1",
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        stream=False,
        native_tool_calling=True,
        drop_params=True,
        modify_params=True,
        litellm_extra_body={"chat_template_kwargs": {"enable_thinking": True}, "min_p": 0},
        log_completions=False,
        disable_vision=True,
        disable_stop_word=False,
        timeout=720,
    )

    # Capture one unified per-attempt record at the transport boundary so
    # request text and vLLM deltas stay in the same record.
    attach_llm_request_attempt_collector(llm, vllm_port=vllm_port)

    tools = [
        Tool(name=TerminalTool.name),
        # Tool(name=FileEditorTool.name),
        Tool(name=GlobTool.name),
        Tool(name=GrepTool.name),
        Tool(name=ApplyPatchTool.name),
        Tool(name=TaskTrackerTool.name),
        Tool(name=TaskToolSet.name),
        Tool(name=BrowserToolSet.name),
        Tool(name=ListDirectoryTool.name),
        Tool(name=EditTool.name),
        Tool(name=ReadFileTool.name),
        Tool(name=WriteFileTool.name),
    ]

    _, _, agent_skills = load_skills_from_dir(skills_dir)
    agent_skills_list = list(agent_skills.values())
    tools.append(Tool(name=SkillTool.name, params={"skills_dir": skills_dir}))
    print(f"已从 {skills_dir} 加载 {len(agent_skills)} 个 Skills: {sorted(agent_skills.keys())}")

    agent = Agent(
        llm=llm,
        tools=tools,
        include_default_tools=["FinishTool", "ThinkTool"],
        tool_concurrency_limit=1,
        system_prompt_filename="system_prompt.j2",
        agent_context=AgentContext(
            skills=agent_skills_list,
            system_message_suffix=(
                "Always be rigorous. "
                "You do not need to execute any code you write. Your only responsibility is to produce well-structured and complete code files."
            ),
        ),
        condenser=LLMSummarizingCondenser(
            llm=llm,
            max_size=240,
            keep_first=2,
        ),
    )

    return llm, agent


_TURN_RE = re.compile(r"turn_(\d+)\.txt$")


def _turn_index(path: str) -> int:
    m = _TURN_RE.search(path)
    if not m:
        raise ValueError(f"无法解析 turn 序号: {path}")
    return int(m.group(1))


def load_benchmark_sequence(repo_name: str, bench_root: str | None = None) -> "SequenceTemplate":
    """从 <bench_root>/<repo_name>/turns/turn_N.txt 构建 SequenceTemplate。"""
    root = bench_root if bench_root else BENCH_ROOT
    repo_dir = os.path.join(root, repo_name)
    if not os.path.isdir(repo_dir):
        raise ValueError(f"benchmark repo 不存在: {repo_dir}")
    turns_dir = os.path.join(repo_dir, "turns")

    import glob as _glob
    turn_files = sorted(
        _glob.glob(os.path.join(turns_dir, "turn_*.txt")),
        key=_turn_index,
    )
    turns = []
    for turn_file in turn_files:
        i = _turn_index(turn_file)
        with open(turn_file, encoding="utf-8") as f:
            message = f.read().strip()
        turns.append(TaskSpec(
            task_id=f"{repo_name}_t{i}",
            message=message,
            description=f"Turn {i}",
        ))

    if not turns:
        raise ValueError(f"未找到任何 turn_*.txt: {turns_dir}")

    return SequenceTemplate(
        template_id=f"bench_{repo_name}",
        description=f"Benchmark: {repo_name}",
        turns=turns,
    )


def resolve_skills_dir(workspace: str, explicit: str | None) -> str:
    """确定 Skills 目录：显式 --skills-dir 优先,否则 workspace/.agents/skills,再否则仓库 skills/。"""
    if explicit:
        return os.path.abspath(explicit)
    agents_skills = os.path.join(os.path.abspath(workspace), ".agents", "skills")
    if os.path.isdir(agents_skills):
        return agents_skills
    return os.path.join(ROOT, "skills")



def main():
    parser = argparse.ArgumentParser(description="多轮任务序列运行器(单序列执行)")
    parser.add_argument("--benchmark-repo", required=True, metavar="REPO",
                        help="从 <bench-root>/<REPO>/turns/turn_N.txt 加载序列")
    parser.add_argument("--bench-root", default=None, metavar="DIR",
                        help="benchmark 根目录（默认: anthropic_skill_benchmark/）")
    parser.add_argument("--workspace", default=os.path.join(ROOT, "workspace", "03_14B_anthropic"),
                        help="任务工作目录")
    parser.add_argument("--skills-dir", default=None,
                        help="Skills 源目录(默认: <workspace>/.agents/skills,若不存在则用仓库 skills/)")
    parser.add_argument("--output", default=os.path.join(ROOT, "results", "03_14B_anthropic", "multurn_bench",
                                                          "multiturn_sequence_traces.json"),
                        help="结果 JSON 输出路径(bench.sh 传入 $seq_res/multiturn_sequence_traces.json)")
    parser.add_argument("--vllm-port", type=int, default=8000)
    parser.add_argument("--log-dir", default=os.path.join(ROOT, "log", "03_14B_anthropic"),
                        help="日志目录")
    parser.add_argument("--dry-run", action="store_true", help="仅打印序列信息,不实际运行")
    args = parser.parse_args()

    # ── 加载序列 ────────────────────────────────────────────────────────
    template = load_benchmark_sequence(args.benchmark_repo, bench_root=args.bench_root)

    seq_workspace = os.path.abspath(args.workspace)
    seq_log_path = os.path.join(args.log_dir, f"{args.benchmark_repo}.log")
    skills_dir = resolve_skills_dir(args.workspace, args.skills_dir)

    print(f"  序列: {args.benchmark_repo}")
    print(f"  turns:    {len(template.turns)}")
    print(f"  workspace: {seq_workspace}")
    print(f"  skills:    {skills_dir}")
    print(f"  log:       {seq_log_path}")
    print(f"  output:    {args.output}")

    if args.dry_run:
        print("\n[DRY-RUN] 仅预览,未实际运行。")
        return

    os.makedirs(args.log_dir, exist_ok=True)

    print(f"[INFO] 从 {skills_dir} 加载 Skills...")
    _, agent = create_agent_and_llm(
        skills_dir,
        args.vllm_port,
    )

    print(f"\n{'='*60}")
    print(f"开始运行: {args.benchmark_repo}")
    print(f"{'='*60}")

    try:
        result = run_sequence(
            template=template,
            agent=agent,
            seq_workspace=seq_workspace,
            seq_log_path=seq_log_path,
        )

        output = {
            "benchmark_repo": args.benchmark_repo,
            "llm_calls": result["llm_calls"],
        }

        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        n_calls = len(result["llm_calls"])
        print(f"\n完成! 共 {n_calls} 次 LLM 调用")
        print(f"  结果: {args.output}")
        print(f"  日志: {seq_log_path}")

    except Exception as e:
        print(f"\n[ERROR] {args.benchmark_repo}): {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
