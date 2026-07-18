"""Run one real OpenHands multi-turn benchmark task with LMCache CacheBlend
doing the skill-KV reuse, instead of CSKCache's custom connector.

Independent of scripts/07_cskcache -- no code from that directory is
imported. The agent/tool/benchmark plumbing is deliberately similar in
shape (same OpenHands SDK usage, same benchmark format) since that part
isn't connector-specific; the reuse mechanism itself is written fresh
because LMCache's CacheBlend works completely differently from CSKCache:

- CSKCache: the client computes exact token offsets for each skill
  occurrence and tells the connector "reuse cache_id X at [start, end)"
  via kv_transfer_params on every request.
- LMCache blend: there is no explicit reuse signal at all. LMCache
  recognizes a chunk as "seen before" purely by hashing its *content*
  (ignoring what precedes it), as long as the chunk is delimited by a
  literal separator string in the rendered prompt text
  (LMCACHE_BLEND_SPECIAL_STR). So the only client-side job here is to
  wrap each skill's guide text in that separator wherever it shows up in
  the conversation before the request goes out -- LMCache does the rest
  server-side.

Which messages to wrap is tracked by tool_call_id, not by matching
message content against known skill text. The `on_event` callback below
watches for ObservationEvent -- which still carries its own
`tool_call_id` at that point -- and records that id whenever the guide
text was just loaded. There are two ways that can happen, and the SDK's
own <SKILLS> system-prompt block tells the model to use the *second* one
by default: calling the dedicated `skill` tool, or a plain `read_file`
call whose path happens to be a SKILL.md (skills are documented as
"not tools, read the file instead" -- an assistant that actually follows
that instruction never touches `skill` at all). Both are tracked. By the
time the corresponding tool-role message reaches the transport edge, its
`tool_call_id` field (standard OpenAI schema, survives serialization) is
enough to know it's a skill result, so the whole message content gets
wrapped unconditionally -- immune to the SkillTool appending a "Skill
Resources" listing after the guide text (which would silently defeat a
content-equality match).
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BENCH_ROOT = ROOT / "anthropic_skill_benchmark"
DEFAULT_SKILLS_DIR = ROOT / "skills"
BLEND_SEP = os.environ.get("LMCACHE_BLEND_SPECIAL_STR", "<|fim_pad|><|repo_name|>")

TASK_SUFFIX = (
    "Always be rigorous. You do not need to execute any code you write. "
    "Your only responsibility is to produce well-structured and complete "
    "code files.\n\n"
    "Reading a skill's one-line description in <SKILLS> is not the same as "
    "using the skill. If a skill is relevant to this task, you must "
    "actually open its SKILL.md file (via read_file, or the `skill` tool) "
    "and read the full guidance before you start producing the deliverable "
    "-- do not proceed from the short description alone."
)


@dataclass
class TaskSpec:
    task_id: str
    message: str


@dataclass
class SequenceTemplate:
    template_id: str
    turns: list[TaskSpec] = field(default_factory=list)


_TURN_RE = re.compile(r"turn_(\d+)\.txt$")
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


class StripAnsiWriter:
    def __init__(self, stream: Any) -> None:
        self.stream = stream

    def write(self, value: str) -> int:
        return self.stream.write(_ANSI_RE.sub("", value))

    def flush(self) -> None:
        self.stream.flush()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.stream, name)


def turn_index(path: Path) -> int:
    match = _TURN_RE.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Cannot parse benchmark turn number: {path}")
    return int(match.group(1))


def load_benchmark_sequence(benchmark_repo: str, bench_root: Path) -> SequenceTemplate:
    benchmark_dir = bench_root / benchmark_repo
    if not benchmark_dir.is_dir():
        raise FileNotFoundError(f"Benchmark task does not exist: {benchmark_dir}")
    turn_paths = sorted((benchmark_dir / "turns").glob("turn_*.txt"), key=turn_index)
    turns = [
        TaskSpec(
            task_id=f"{benchmark_repo}_t{turn_index(path)}",
            message=path.read_text(encoding="utf-8").strip(),
        )
        for path in turn_paths
    ]
    if not turns:
        raise ValueError(f"No turn_*.txt files found under {benchmark_dir}/turns")
    return SequenceTemplate(template_id=f"bench_{benchmark_repo}", turns=turns)


def count_skills(skills_dir: Path) -> int:
    count = sum(1 for _ in skills_dir.glob("*/SKILL.md"))
    if not count:
        raise RuntimeError(f"No skills found under {skills_dir}")
    return count


def wrap_text_content(value: Any) -> Any:
    """Prepend/append the blend separator around all text in a message's
    content, unconditionally -- called only once the caller already knows
    (via tool_call_id) that the whole message is a skill result."""

    if isinstance(value, str):
        return f"{BLEND_SEP}{value}{BLEND_SEP}"
    if isinstance(value, list):
        return [wrap_text_content(item) for item in value]
    if isinstance(value, dict):
        if isinstance(value.get("text"), str):
            return {**value, "text": wrap_text_content(value["text"])}
        return value
    return value


def wrap_skill_tool_messages(
    messages: list[dict[str, Any]], skill_tool_call_ids: set[str]
) -> list[dict[str, Any]]:
    normalized = copy.deepcopy(messages)
    for message in normalized:
        if message.get("role") != "tool":
            continue
        if message.get("tool_call_id") not in skill_tool_call_ids:
            continue
        message["content"] = wrap_text_content(message.get("content"))
    return normalized


def serializable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    try:
        return json.loads(json.dumps(value))
    except (TypeError, ValueError):
        return str(value)


def atomic_write_json(path: Path, value: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


class BlendTransportInjector:
    """Wrap skill-guide text with LMCache's blend separator at the transport
    edge, and record every request/response for later inspection."""

    def __init__(self, *, llm: Any, request_dir: Path) -> None:
        self.llm = llm
        self.skill_tool_call_ids: set[str] = set()
        self.request_dir = request_dir
        self.request_dir.mkdir(parents=True, exist_ok=True)
        self.turn = 0
        self.invocation_in_turn = 0
        self.request_index = 0
        self.records: list[dict[str, Any]] = []

    def start_turn(self, turn: int) -> None:
        self.turn = turn
        self.invocation_in_turn = 0

    def record_skill_call(self, tool_call_id: str) -> None:
        self.skill_tool_call_ids.add(tool_call_id)

    def install(self) -> None:
        # 抓住原来的发送函数，_transport_call 是消息真正变成 HTTP 请求、发送给 vLLM 之前的最后一道关卡
        original = getattr(self.llm, "_transport_call", None)
        if not callable(original):
            raise RuntimeError("OpenHands LLM has no callable _transport_call")

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            raw_messages = kwargs.get("messages") or []
            # 把 skill 文字包上分隔符
            messages = wrap_skill_tool_messages(raw_messages, self.skill_tool_call_ids)
            kwargs["messages"] = messages

            self.request_index += 1
            self.invocation_in_turn += 1
            request_id = (
                f"lmcache-blend-turn{self.turn:02d}-"
                f"inv{self.invocation_in_turn:03d}-req{self.request_index:04d}"
            )
            extra_body = dict(kwargs.get("extra_body") or {})
            extra_body["request_id"] = request_id
            kwargs["extra_body"] = extra_body

            record: dict[str, Any] = {
                "schema_version": 1,
                "request_index": self.request_index,
                "turn": self.turn,
                "invocation_in_turn": self.invocation_in_turn,
                "request_id": request_id,
                "messages": messages,
                "status": "running",
            }
            started = time.perf_counter()
            try:
                # 再调用原来的函数真正发送出去
                response = original(*args, **kwargs)
                record.update(
                    status="completed",
                    elapsed_s=round(time.perf_counter() - started, 6),
                    response=serializable(response),
                )
                return response
            except Exception as exc:
                record.update(
                    status="failed",
                    elapsed_s=round(time.perf_counter() - started, 6),
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
                raise
            finally:
                path = self.request_dir / f"turn_{self.turn}_inv_{self.invocation_in_turn}.json"
                atomic_write_json(path, record)
                self.records.append(record)
                print(
                    f"[lmcache-blend request] turn={self.turn} "
                    f"inv={self.invocation_in_turn} status={record['status']} "
                    f"elapsed={record.get('elapsed_s')}",
                    flush=True,
                )

        object.__setattr__(wrapped, "_lmcache_blend_agent_wrapped", True)
        # 用 object.__setattr__ 是绕开 pydantic 的检查、直接在对象上硬改这个属性。
        object.__setattr__(self.llm, "_transport_call", wrapped)


def create_agent_and_llm(*, skills_dir: Path, model: str, vllm_port: int) -> tuple[Any, Any]:
    from openhands.sdk import Agent, AgentContext, LLM, LLMSummarizingCondenser, Tool
    from openhands.sdk.context.skills import load_skills_from_dir
    from openhands.tools.apply_patch import ApplyPatchTool
    from openhands.tools.browser_use import BrowserToolSet
    from openhands.tools.gemini import EditTool, ListDirectoryTool, ReadFileTool, WriteFileTool
    from openhands.tools.glob import GlobTool
    from openhands.tools.grep import GrepTool
    from openhands.tools.skill import SkillTool
    from openhands.tools.task import TaskToolSet
    from openhands.tools.task_tracker import TaskTrackerTool
    from openhands.tools.terminal import TerminalTool

    served_model = model if "/" in model else f"openai/{model}"
    llm = LLM(
        model=served_model,
        api_key=os.environ.get("VLLM_API_KEY", "EMPTY"),
        base_url=f"http://localhost:{vllm_port}/v1",
        temperature=0,
        top_p=0.95,
        top_k=20,
        stream=False,
        native_tool_calling=True,
        force_string_serializer=True,
        drop_params=True,
        modify_params=True,
        litellm_extra_body={
            "chat_template_kwargs": {"enable_thinking": True},
            "min_p": 0,
        },
        log_completions=False,
        disable_vision=True,
        disable_stop_word=False,
        timeout=720,
    )
    tools = [
        Tool(name=TerminalTool.name),
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
    skills_path = str(skills_dir)
    _, _, loaded_skills = load_skills_from_dir(skills_path)
    tools.append(Tool(name=SkillTool.name, params={"skills_dir": skills_path}))
    print(f"Loaded {len(loaded_skills)} skills from {skills_path}: {sorted(loaded_skills)}")

    agent = Agent(
        llm=llm,
        tools=tools,
        include_default_tools=["FinishTool", "ThinkTool"],
        tool_concurrency_limit=1,
        system_prompt_filename="system_prompt.j2",
        agent_context=AgentContext(
            skills=list(loaded_skills.values()),
            system_message_suffix=TASK_SUFFIX,
        ),
        condenser=LLMSummarizingCondenser(llm=llm, max_size=240, keep_first=2),
    )
    return llm, agent


def prepare_task_directory(
    run_dir: Path, benchmark_repo: str, bench_root: Path, skills_dir: Path, overwrite: bool
) -> tuple[Path, Path, Path]:
    task_dir = run_dir / benchmark_repo
    if task_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Task output already exists: {task_dir}")
        shutil.rmtree(task_dir)
    workspace = task_dir / "workspace"
    request_dir = task_dir / "requests"
    workspace.mkdir(parents=True)
    request_dir.mkdir(parents=True)

    benchmark_dir = bench_root / benchmark_repo
    seed_dir = benchmark_dir / "seed_files"
    if seed_dir.is_dir():
        shutil.copytree(seed_dir, workspace / "seed_files")
    active_skills = workspace / ".agents" / "skills"
    active_skills.parent.mkdir(parents=True)
    shutil.copytree(skills_dir, active_skills)
    return task_dir, workspace, active_skills


def extract_answer_text(event: Any) -> str | None:
    from openhands.sdk.llm import content_to_str

    if type(event).__name__ != "MessageEvent":
        return None
    if getattr(event, "source", None) != "agent":
        return None
    content = getattr(event.llm_message, "content", None)
    if not content:
        return None
    return "\n".join(part for part in content_to_str(content) if part)


def run_real_sequence(
    *,
    template: SequenceTemplate,
    agent: Any,
    workspace: Path,
    agent_log: Path,
    transcript_path: Path,
    injector: BlendTransportInjector,
    max_iteration_per_run: int,
) -> None:
    from openhands.sdk import Conversation
    from openhands.tools.gemini import ReadFileTool
    from openhands.tools.skill import SkillTool

    def on_event(event: Any) -> None:
        # A skill's guide text can enter the conversation two ways: the
        # dedicated `skill` tool, or (what the SDK's own <SKILLS> prompt
        # block actually tells the model to do) a plain `read_file` call
        # whose path happens to be a SKILL.md. Track both -- otherwise a
        # model that follows the SDK's own instructions never gets its
        # skill content wrapped at all.
        if type(event).__name__ == "ObservationEvent":
            tool_name = getattr(event, "tool_name", None)
            if tool_name == SkillTool.name:
                injector.record_skill_call(event.tool_call_id)
            elif tool_name == ReadFileTool.name:
                file_path = getattr(event.observation, "file_path", "")
                if file_path.endswith("SKILL.md"):
                    injector.record_skill_call(event.tool_call_id)
        answer = extract_answer_text(event)
        if answer:
            with transcript_path.open("a", encoding="utf-8") as handle:
                handle.write(answer + "\n")

    conversation = Conversation(
        agent=agent,
        workspace=str(workspace),
        callbacks=[on_event],
        max_iteration_per_run=max_iteration_per_run,
        stuck_detection=True,
        delete_on_close=True,
    )
    saved_stdout, saved_stderr = sys.stdout, sys.stderr
    log_handle = agent_log.open("w", encoding="utf-8")
    writer = StripAnsiWriter(log_handle)
    sys.stdout = writer
    sys.stderr = writer
    try:
        for turn, turn_spec in enumerate(template.turns, start=1):
            injector.start_turn(turn)
            message = f"Working directory: {workspace}\n\n{turn_spec.message}"
            with transcript_path.open("a", encoding="utf-8") as handle:
                handle.write(f"\n{'=' * 20} TURN {turn}/{len(template.turns)} {'=' * 20}\n")
                handle.write(f"[user]\n{turn_spec.message}\n\n[assistant]\n")
            print(f"[TURN {turn}/{len(template.turns)}]", flush=True)
            conversation.send_message(message)
            conversation.run()
            log_handle.flush()
    finally:
        conversation.close()
        sys.stdout = saved_stdout
        sys.stderr = saved_stderr
        log_handle.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-repo", required=True)
    parser.add_argument("--bench-root", type=Path, default=DEFAULT_BENCH_ROOT)
    parser.add_argument("--skills-dir", type=Path, default=DEFAULT_SKILLS_DIR)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model", default=os.environ.get("VLLM_SERVED_NAME", "Qwen3"))
    parser.add_argument("--vllm-port", type=int, default=8100)
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument("--max-iteration-per-run", type=int, default=500)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    template = load_benchmark_sequence(args.benchmark_repo, args.bench_root)
    if args.max_turns is not None:
        template.turns = template.turns[: args.max_turns]
    skill_count = count_skills(args.skills_dir)

    if args.dry_run:
        print(
            f"[dry-run] task={args.benchmark_repo} turns={len(template.turns)} "
            f"skills={skill_count} blend_sep={BLEND_SEP!r}"
        )
        return

    task_dir, workspace, active_skills = prepare_task_directory(
        args.run_dir, args.benchmark_repo, args.bench_root, args.skills_dir, args.overwrite
    )
    llm, agent = create_agent_and_llm(
        skills_dir=active_skills, model=args.model, vllm_port=args.vllm_port
    )
    injector = BlendTransportInjector(llm=llm, request_dir=task_dir / "requests")
    injector.install()

    transcript_path = task_dir / "answers.txt"
    transcript_path.write_text(
        f"Benchmark: {args.benchmark_repo}\nModel: {args.model}\n"
        f"Blend separator: {BLEND_SEP!r}\n",
        encoding="utf-8",
    )

    started = time.time()
    status = "completed"
    error: dict[str, str] | None = None
    try:
        run_real_sequence(
            template=template,
            agent=agent,
            workspace=workspace,
            agent_log=task_dir / "agent.log",
            transcript_path=transcript_path,
            injector=injector,
            max_iteration_per_run=args.max_iteration_per_run,
        )
    except Exception as exc:
        status = "failed"
        error = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        summary = {
            "schema_version": 1,
            "benchmark_repo": args.benchmark_repo,
            "status": status,
            "error": error,
            "turns": len(template.turns),
            "requests": len(injector.records),
            "elapsed_s": round(time.time() - started, 4),
            "workspace": str(workspace),
            "request_dir": str(task_dir / "requests"),
            "transcript": str(transcript_path),
        }
        atomic_write_json(task_dir / "_summary.json", summary)


if __name__ == "__main__":
    main()
