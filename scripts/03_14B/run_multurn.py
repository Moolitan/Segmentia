
import argparse
import copy
import json
import os
import re
import sys
import time
import traceback
from datetime import datetime, timezone
import threading
import queue

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts", "03_14B"))

from task_templates import (
    SequenceTemplate,
    get_templates,
)

from benchkit.metrics.vllm_prefix_cache import (
    VllmTimelineSampler,
)


SEQUENCE_MAP: dict[str, tuple[str, str]] = {
    "T2_A": ("T2-A", "admin_panel"),
    "T2_B": ("T2-B", "startup_budget"),
    "T4_B": ("T4-B", "annual_review"),
    "T4_F": ("T4-F", "social_feed"),
    "T4_H": ("T4-H", "customer_satisfaction"),
    "T6_A": ("T6-A", "data_pipeline"),
    "T6_B": ("T6-B", "github_integration"),
    "T6_E": ("T6-E", "project_management"),
    "T6_F": ("T6-F", "devops-automation"),
    "T8_A": ("T8-A", "fitness-tracker"),
    "T8_A_SKILL": ("T8-A-SKILL", "fitness-tracker"),
    "T10_A": ("T10-A", "crm-system"),
}

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


def _safe_json(obj):
    """
    Best-effort JSON serializer, aligned with OpenHands telemetry's behavior:
    prefer Pydantic model_dump when available; otherwise fallback to __dict__/str.
    """
    try:
        model_dump = getattr(obj, "model_dump", None)
        if callable(model_dump):
            return model_dump(mode="json", exclude_none=True)
    except Exception:
        pass
    try:
        return obj.__dict__
    except Exception:
        return str(obj)


class _AsyncRequestInputCollector:
    """
    Offload request input capture to a background thread so _transport_call
    stays lightweight. We only keep `messages` to minimize overhead.
    """

    def __init__(self, llm, max_queue: int = 1024):
        self._llm = llm
        self._q: "queue.Queue[dict | None]" = queue.Queue(maxsize=max_queue)
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        try:
            self._q.put(None, timeout=0.2)
        except Exception:
            pass
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def enqueue(self, raw: dict) -> None:
        # Best-effort: never block the request path.
        try:
            self._q.put(raw, timeout=0.0)
        except Exception:
            pass

    def _run(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                break
            try:
                # Only store messages (process trace). Everything else is excluded by request.
                self._llm._request_messages.append(copy.deepcopy(item.get("messages_raw")))

                # Store timing separately to avoid changing request_input shape.
                self._llm._request_timings.append(
                    {
                        "request_started_at": item.get("request_started_at"),
                        "request_ended_at": item.get("request_ended_at"),
                        "request_latency_seconds": item.get("request_latency_seconds"),
                    }
                )
            except Exception:
                pass


def attach_llm_request_input_collector(llm) -> None:
    """
    Patch OpenHands SDK LLM instance to collect the *actual* request inputs
    right before the transport sends them (messages + final kwargs).

    The sink can be set dynamically via `llm._request_input_sink`, allowing the
    runner to stream data; current runner keeps everything in-memory and writes
    once to the final results JSON.
    """
    if getattr(llm, "_request_input_patched", False):
        return
    llm._request_input_patched = True
    llm._request_messages: list = []
    llm._request_timings: list[dict] = []
    original = getattr(llm, "_transport_call", None)
    if not callable(original):
        return

    collector = _AsyncRequestInputCollector(llm)
    collector.start()
    llm._request_input_collector = collector

    def wrapped(*args, **kwargs):
        # Support both positional and keyword usage; OpenHands uses kwargs.
        messages = kwargs.get("messages")

        t_start = time.time()
        raw = {
            "request_started_at": t_start,
            "messages_raw": messages,
        }
        try:
            return original(*args, **kwargs)
        finally:
            t_end = time.time()
            raw["request_ended_at"] = t_end
            raw["request_latency_seconds"] = round(t_end - t_start, 6)
            try:
                collector.enqueue(raw)
            except Exception:
                pass

    object.__setattr__(wrapped, "_request_input_wrapped", True)
    object.__setattr__(llm, "_transport_call", wrapped)


def run_sequence(
    template: SequenceTemplate,
    theme: str,
    agent,
    seq_workspace: str,
    seq_log_path: str,
    max_iteration_per_run: int = 500,
    vllm_port: int = 8000,
) -> dict:
    from openhands.sdk import Conversation

    seq_id = f"{template.template_id}_{theme}"
    os.makedirs(seq_workspace, exist_ok=True)

    saved_stdout, saved_stderr = sys.stdout, sys.stderr
    os.makedirs(os.path.dirname(seq_log_path), exist_ok=True)
    seq_log_file = open(seq_log_path, "w", encoding="utf-8")
    seq_writer = StripAnsiWriter(seq_log_file)
    sys.stdout = seq_writer
    sys.stderr = seq_writer

    vllm_log_path = os.path.join(ROOT, "log", "vllm.log")
    timeline_sampler = VllmTimelineSampler(
        vllm_port=vllm_port,
        interval_seconds=10.0,
        vllm_log_path=vllm_log_path,
    )

    conversation = Conversation(
        agent=agent,
        workspace=seq_workspace,
        max_iteration_per_run=max_iteration_per_run,
        stuck_detection=True,
        delete_on_close=True,
    )

    all_llm_calls: list[dict] = []   # 全序列 flat list
    global_call_cursor = 0           # 已处理到 token_usages 的位置
    seq_start = time.time()
    vllm_metrics_patched_any = False
    try:
        vllm_metrics_patched_any = bool(getattr(agent.llm, "_vllm_metrics_patched_any", False))
    except Exception:
        pass

    try:
        timeline_sampler.start()

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
                + turn_spec.message.replace("{theme}", theme)
            )
            print(f"    [{turn_number}] {turn_spec.description} (skills: {turn_spec.expected_skills})")

            # 记录本轮开始前的 token_usages 长度
            try:
                metrics_before = conversation.conversation_stats.get_combined_metrics()
                n_usages_before = len(metrics_before.token_usages)
            except Exception:
                n_usages_before = 0

            t0 = time.time()

            conversation.send_message(message)
            conversation.run()
            seq_log_file.flush()

            # 本轮新增的 token_usages(每条对应一次 LLM 调用)
            new_usages = []
            try:
                metrics_after = conversation.conversation_stats.get_combined_metrics()
                new_usages = metrics_after.token_usages[n_usages_before:]
            except Exception:
                pass

            # 仅保留 token + request_input(messages) + request timing
            n_usages = len(new_usages)
            for k in range(n_usages):
                usage = new_usages[k]

                prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
                completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0

                record = {
                    "call_index_in_turn": k,
                    "turn_number": turn_number,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                }

                # Actual request input captured at transport boundary (best-effort).
                try:
                    msgs = getattr(agent.llm, "_request_messages", None)
                    timings = getattr(agent.llm, "_request_timings", None)
                    idx = global_call_cursor + k
                    if isinstance(msgs, list) and 0 <= idx < len(msgs):
                        record["request_input"] = {"messages": msgs[idx]}
                    if isinstance(timings, list) and 0 <= idx < len(timings):
                        record.update(timings[idx])
                except Exception:
                    pass

                all_llm_calls.append(record)

            global_call_cursor += n_usages
        total_elapsed = time.time() - seq_start

    finally:
        timeline_sampler.stop()
        try:
            conversation.close()
        except Exception:
            pass
        try:
            llm = getattr(agent, "llm", None)
            collector = getattr(llm, "_request_input_collector", None) if llm is not None else None
            if collector is not None:
                collector.stop(timeout=2.0)
        except Exception:
            pass
        sys.stdout = saved_stdout
        sys.stderr = saved_stderr
        seq_log_file.close()

    return {
        "sequence_id": seq_id,
        "template_id": template.template_id,
        "theme": theme,
        "description": template.description,
        "num_turns": len(template.turns),
        "total_elapsed_seconds": round(total_elapsed, 2),
        "workspace": seq_workspace,
        "log": seq_log_path,
        "vllm_metrics_supported": bool(getattr(agent.llm, "_vllm_metrics_supported", False)),
        "vllm_metrics_patched_any": vllm_metrics_patched_any,
        "llm_calls": all_llm_calls,
        "vllm_timeline": timeline_sampler.get_timeline(),
    }


def create_agent_and_llm(
    skills_dir: str,
    vllm_port: int = 8000,
    *,
    no_skills: bool = False,
    system_prompt_filename: str = "system_prompt.j2",
):
    """创建 LLM 和 Agent 实例。

    Args:
        skills_dir: Skills 源目录。
        vllm_port: vLLM 端口。
        no_skills: 若为 True，不加载任何 Skills、不注册 SkillTool（对照组）。
        system_prompt_filename: Agent 系统提示 Jinja2 模板文件名（SDK prompts 目录内或绝对路径）。
    """
    from openhands.sdk import LLM, Agent, AgentContext, LLMSummarizingCondenser, Tool
    from openhands.sdk.context.skills import load_skills_from_dir
    from openhands.tools.terminal import TerminalTool
    from openhands.tools.file_editor import FileEditorTool
    from openhands.tools.glob import GlobTool
    from openhands.tools.grep import GrepTool
    from openhands.tools.apply_patch import ApplyPatchTool
    from openhands.tools.skill import SkillTool

    vllm_api_key = os.environ.get("VLLM_API_KEY", "EMPTY")

    llm = LLM(
        model="openai/Qwen3",
        api_key=vllm_api_key,
        base_url=f"http://localhost:{vllm_port}/v1",
        temperature=0.0,           #qwen3-8B
        top_p=0.95,                #qwen3-8B
        top_k=20,                  #qwen3-8B
        # max_message_chars=32768,
        stream=False,
        native_tool_calling=False,
        caching_prompt=True,
        prompt_cache_retention="24h",
        drop_params=True,
        modify_params=True,
        reasoning_effort="high",
        extended_thinking_budget=200000,
        litellm_extra_body={"chat_template_kwargs": {"enable_thinking": True}, "min_p": 0},
        log_completions=False,
        disable_vision=True,
        disable_stop_word=False,
        enable_encrypted_reasoning=True,
        timeout=720,
    )

    # Capture the *actual* request inputs at the transport boundary.
    attach_llm_request_input_collector(llm)

    tools = [
        Tool(name=TerminalTool.name),
        Tool(name=FileEditorTool.name),
        Tool(name=GlobTool.name),
        Tool(name=GrepTool.name),
        Tool(name=ApplyPatchTool.name),
    ]

    if no_skills:
        agent_skills_list = []
        print(f"[NO-SKILLS] 对照组模式：不加载任何 Skills，不注册 SkillTool")
    else:
        _, _, agent_skills = load_skills_from_dir(skills_dir)
        agent_skills_list = list(agent_skills.values())
        tools.append(Tool(name=SkillTool.name, params={"skills_dir": skills_dir}))
        print(f"已从 {skills_dir} 加载 {len(agent_skills)} 个 Skills: {sorted(agent_skills.keys())}")

    agent = Agent(
        llm=llm,
        tools=tools,
        include_default_tools=["FinishTool", "ThinkTool"],
        tool_concurrency_limit=1,
        system_prompt_filename=system_prompt_filename,
        agent_context=AgentContext(
            skills=agent_skills_list,
            system_message_suffix="始终保持严谨。",
        ),
        condenser=LLMSummarizingCondenser(
            llm=llm,
            max_size=240,
            keep_first=2,
        ),
    )

    return llm, agent


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
    parser.add_argument("--sequence", required=True, choices=sorted(SEQUENCE_MAP.keys()),
                        help="要运行的序列 ID,如 T4_B / T6_A")
    parser.add_argument("--workspace", default=os.path.join(ROOT, "workspace", "03_14B"),
                        help="任务工作目录")
    parser.add_argument("--skills-dir", default=None,
                        help="Skills 源目录(默认: <workspace>/.agents/skills,若不存在则用仓库 skills/)")
    parser.add_argument("--output", default=os.path.join(ROOT, "results", "03_14B", "multurn_bench",
                                                          "multiturn_sequence_traces.json"),
                        help="结果 JSON 输出路径(bench.sh 传入 $seq_res/multiturn_sequence_traces.json)")
    parser.add_argument("--vllm-port", type=int, default=8000)
    parser.add_argument("--log-dir", default=os.path.join(ROOT, "log", "03_14B"),
                        help="日志目录")
    parser.add_argument("--no-skills", action="store_true",
                        help="对照组模式：不加载任何 Skills,不注册 SkillTool")
    parser.add_argument(
        "--system-prompt-filename",
        default="system_prompt.j2",
        help="系统提示模板文件名（如 system_prompt.j2 / system_prompt_skill.j2）",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=None,
        help="与 vLLM --max-model-len 对齐的最大上下文长度，仅写入结果 metadata 便于区分实验",
    )
    parser.add_argument("--dry-run", action="store_true", help="仅打印序列信息,不实际运行")
    args = parser.parse_args()

    template_id, theme = SEQUENCE_MAP[args.sequence]
    templates = get_templates(template_id=template_id)
    if not templates:
        print(f"[ERROR] 找不到 template_id={template_id}")
        sys.exit(1)
    template = templates[0]

    seq_id = f"{template.template_id}_{theme}"
    seq_workspace = os.path.abspath(args.workspace)
    seq_log_path = os.path.join(args.log_dir, f"{seq_id}.log")
    skills_dir = resolve_skills_dir(args.workspace, args.skills_dir)

    print(f"序列: {args.sequence} → {seq_id}")
    print(f"  template: {template.template_id} ({template.description})")
    print(f"  theme:    {theme}")
    print(f"  turns:    {len(template.turns)}")
    for i, t in enumerate(template.turns):
        msg = t.message.replace("{theme}", theme)
        print(f"    [{i+1}] {t.description} (skills: {t.expected_skills})")
        print(f"        {msg[:100]}{'...' if len(msg) > 100 else ''}")
    print(f"  workspace: {seq_workspace}")
    print(f"  skills:    {skills_dir}")
    print(f"  log:       {seq_log_path}")
    print(f"  output:    {args.output}")
    print(f"  system_prompt: {args.system_prompt_filename}")
    if args.context_length is not None:
        print(f"  context_length (metadata): {args.context_length}")

    if args.dry_run:
        print("\n[DRY-RUN] 仅预览,未实际运行。")
        return

    os.makedirs(args.log_dir, exist_ok=True)

    if args.no_skills:
        print(f"[INFO] 对照组模式：不加载 Skills")
    else:
        print(f"[INFO] 从 {skills_dir} 加载 Skills...")
    _, agent = create_agent_and_llm(
        skills_dir,
        args.vllm_port,
        no_skills=args.no_skills,
        system_prompt_filename=args.system_prompt_filename,
    )

    print(f"\n{'='*60}")
    print(f"开始运行: {args.sequence} → {seq_id} ({len(template.turns)} 轮)")
    print(f"{'='*60}")

    try:
        result = run_sequence(
            template=template,
            theme=theme,
            agent=agent,
            seq_workspace=seq_workspace,
            seq_log_path=seq_log_path,
            vllm_port=args.vllm_port,
        )

        metadata = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": "Qwen3",
            "vllm_port": args.vllm_port,
            "skills_dir": skills_dir,
            "no_skills": args.no_skills,
            "sequence_key": args.sequence,
            "system_prompt_filename": args.system_prompt_filename,
        }
        if args.context_length is not None:
            metadata["context_length"] = args.context_length
            metadata["vllm_max_model_len"] = args.context_length

        output = {
            "metadata": metadata,
            "sequence": result,
        }

        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        n_calls = len(result["llm_calls"])
        print(f"\n完成! 耗时 {result['total_elapsed_seconds']}s,共 {n_calls} 次 LLM 调用")
        print(f"  结果: {args.output}")
        print(f"  日志: {seq_log_path}")

    except Exception as e:
        print(f"\n[ERROR] {seq_id}: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
