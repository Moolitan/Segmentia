"""Run one real OpenHands multi-turn benchmark task with CSKCache reuse."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = Path(
    "/mnt/Large_Language_Model_Lab_1/llm_models/Qwen3-14B/Qwen/Qwen3-14B"
)
DEFAULT_BENCH_ROOT = ROOT / "anthropic_skill_benchmark"
DEFAULT_SKILLS_DIR = ROOT / "skills"
DEFAULT_KV_DIR = Path(
    "/mnt/Large_Language_Model_Lab_1/wsh/Segmentia/output/07_cskcache/"
    "offline_skill_kv"
)


@dataclass
class TaskSpec:
    task_id: str
    message: str
    description: str


@dataclass
class SequenceTemplate:
    template_id: str
    description: str
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


class OpenHandsEventLogger:
    """Persist each real Conversation event as one immediately flushed JSON line."""

    def __init__(self, path: Path) -> None:
        self.handle = path.open("w", encoding="utf-8")
        self.lock = threading.Lock()
        self.turn = 0
        self.event_index = 0

    def start_turn(self, turn: int) -> None:
        with self.lock:
            self.turn = turn

    def on_event(self, event: Any) -> None:
        with self.lock:
            self.event_index += 1
            record = {
                "schema_version": 1,
                "event_index": self.event_index,
                "turn": self.turn,
                "event_type": type(event).__name__,
                "event": event.model_dump(mode="json", exclude_none=True),
            }
            self.handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            self.handle.flush()

    def close(self) -> None:
        self.handle.close()


def turn_index(path: Path) -> int:
    match = _TURN_RE.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Cannot parse benchmark turn number: {path}")
    return int(match.group(1))


def load_benchmark_sequence(
    benchmark_repo: str,
    bench_root: Path,
) -> SequenceTemplate:
    benchmark_dir = bench_root / benchmark_repo
    if not benchmark_dir.is_dir():
        raise FileNotFoundError(f"Benchmark task does not exist: {benchmark_dir}")
    turns_dir = benchmark_dir / "turns"
    turn_paths = sorted(turns_dir.glob("turn_*.txt"), key=turn_index)
    turns = [
        TaskSpec(
            task_id=f"{benchmark_repo}_t{turn_index(path)}",
            message=path.read_text(encoding="utf-8").strip(),
            description=f"Turn {turn_index(path)}",
        )
        for path in turn_paths
    ]
    if not turns:
        raise ValueError(f"No turn_*.txt files found: {turns_dir}")
    return SequenceTemplate(
        template_id=f"bench_{benchmark_repo}",
        description=f"Benchmark: {benchmark_repo}",
        turns=turns,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-repo", required=True)
    parser.add_argument("--bench-root", type=Path, default=DEFAULT_BENCH_ROOT)
    parser.add_argument("--skills-dir", type=Path, default=DEFAULT_SKILLS_DIR)
    parser.add_argument("--kv-dir", type=Path, default=DEFAULT_KV_DIR)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--model", default=os.environ.get("VLLM_SERVED_NAME", "Qwen3"))
    parser.add_argument("--vllm-port", type=int, default=8013)
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument("--max-iteration-per-run", type=int, default=500)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def all_subsequence_starts(tokens: list[int], needle: list[int]) -> list[int]:
    if not needle or len(needle) > len(tokens):
        return []
    first = needle[0]
    width = len(needle)
    return [
        index
        for index, token in enumerate(tokens[: len(tokens) - width + 1])
        if token == first and tokens[index : index + width] == needle
    ]


def common_prefix_length(left: list[int], right: list[int]) -> int:
    size = min(len(left), len(right))
    index = 0
    while index < size and left[index] == right[index]:
        index += 1
    return index


def normalize_tool_content(value: Any, skill_texts: dict[str, str]) -> Any:
    """Normalize real SkillTool output while preserving all visible content."""

    if isinstance(value, str):
        result = value
        resource_marker = "\n\n--- Skill Resources ---\n"
        for text in skill_texts.values():
            with_resources = text + resource_marker
            if result.startswith(with_resources):
                resources = result[len(with_resources) :]
                result = (
                    f"--- Skill Resources ---\n{resources}"
                    f"\n\n--- Skill Guide ---\n{text}"
                )
            if text.endswith("\n"):
                result = result.replace(text, text[:-1])
        return result
    if isinstance(value, list):
        return [normalize_tool_content(item, skill_texts) for item in value]
    if isinstance(value, dict):
        return {
            key: normalize_tool_content(item, skill_texts)
            for key, item in value.items()
        }
    return value


def normalize_skill_tool_messages(
    messages: list[dict[str, Any]],
    skill_texts: dict[str, str],
) -> list[dict[str, Any]]:
    """Avoid duplicate terminal newlines introduced by Qwen's tool template.

    The Conversation state remains untouched. Only the deep-copied HTTP prompt
    is normalized, consistently on every request, so prefix identity is stable.
    """

    normalized = copy.deepcopy(messages)
    for message in normalized:
        if message.get("role") != "tool":
            continue
        message["content"] = normalize_tool_content(
            message.get("content"), skill_texts
        )
    return normalized


def render_prompt_tokens(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
) -> list[int]:
    rendered = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    return [int(token) for token in rendered]


def cache_paths(kv_dir: Path, cache_id: str) -> tuple[Path, Path]:
    digest = hashlib.sha256(cache_id.encode("utf-8")).hexdigest()[:32]
    return kv_dir / f"{digest}.pt", kv_dir / f"{digest}.json"


def load_skill_catalog(
    skills_dir: Path,
    kv_dir: Path,
    tokenizer: Any,
) -> tuple[dict[str, str], dict[str, list[int]]]:
    manifest_path = kv_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing offline manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = {
        str(record["cache_id"]): record
        for record in manifest.get("records", [])
        if isinstance(record, dict) and "cache_id" in record
    }

    texts: dict[str, str] = {}
    tokens: dict[str, list[int]] = {}
    for skill_path in sorted(skills_dir.glob("*/SKILL.md")):
        cache_id = skill_path.parent.name
        text = skill_path.read_text(encoding="utf-8")
        token_ids = [
            int(token)
            for token in tokenizer.encode(text, add_special_tokens=False)
        ]
        payload_path, sidecar_path = cache_paths(kv_dir, cache_id)
        if not payload_path.is_file() or not sidecar_path.is_file():
            raise FileNotFoundError(f"Missing offline entry for skill={cache_id}")
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        record = records.get(cache_id)
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if int(sidecar.get("num_tokens", -1)) != len(token_ids):
            raise ValueError(f"Offline token count is stale for skill={cache_id}")
        if record is None or record.get("text_sha256") != digest:
            raise ValueError(f"Offline manifest text hash is stale for skill={cache_id}")
        texts[cache_id] = text
        tokens[cache_id] = token_ids
    if not texts:
        raise RuntimeError(f"No skills found under {skills_dir}")
    return texts, tokens


def serializable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    try:
        return json.loads(json.dumps(value))
    except (TypeError, ValueError):
        return str(value)


def atomic_write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


class CSKCacheTransportInjector:
    """Add one request-local CSKCache signal at the real LLM transport edge."""

    def __init__(
        self,
        *,
        llm: Any,
        tokenizer: Any,
        skill_texts: dict[str, str],
        skill_tokens: dict[str, list[int]],
        request_dir: Path,
    ) -> None:
        self.llm = llm
        self.tokenizer = tokenizer
        self.skill_texts = skill_texts
        self.skill_tokens = skill_tokens
        self.request_dir = request_dir
        self.request_dir.mkdir(parents=True, exist_ok=True)
        self.previous_prompt_tokens: list[int] = []
        self.turn = 0
        self.invocation_in_turn = 0
        self.request_index = 0
        self.records: list[dict[str, Any]] = []

    def start_turn(self, turn: int) -> None:
        self.turn = turn
        self.invocation_in_turn = 0

    def install(self) -> None:
        original = getattr(self.llm, "_transport_call", None)
        if not callable(original):
            raise RuntimeError("OpenHands LLM has no callable _transport_call")

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            raw_messages = kwargs.get("messages") or []
            tools = kwargs.get("tools")
            messages = normalize_skill_tool_messages(raw_messages, self.skill_texts)
            prompt_tokens = render_prompt_tokens(self.tokenizer, messages, tools)
            lcp = common_prefix_length(self.previous_prompt_tokens, prompt_tokens)

            candidates: list[dict[str, Any]] = []
            for cache_id, needle in self.skill_tokens.items():
                for start in all_subsequence_starts(prompt_tokens, needle):
                    end = start + len(needle)
                    if end > lcp:
                        candidates.append(
                            {
                                "cache_id": cache_id,
                                "target_start": start,
                                "target_end": end,
                            }
                        )
            candidates.sort(
                key=lambda entry: (
                    entry["target_start"],
                    entry["target_end"],
                    entry["cache_id"],
                )
            )
            reuse = (
                {"operation": "reuse", "entries": candidates}
                if candidates
                else None
            )

            self.request_index += 1
            self.invocation_in_turn += 1
            request_id = (
                f"cskcache-agent-turn{self.turn:02d}-"
                f"inv{self.invocation_in_turn:03d}-req{self.request_index:04d}"
            )
            extra_body = dict(kwargs.get("extra_body") or {})
            extra_body["request_id"] = request_id
            if reuse is not None:
                extra_body["kv_transfer_params"] = {"cskcache": reuse}
            else:
                extra_body.pop("kv_transfer_params", None)
            kwargs["extra_body"] = extra_body
            kwargs["messages"] = messages

            record: dict[str, Any] = {
                "schema_version": 1,
                "request_index": self.request_index,
                "turn": self.turn,
                "invocation_in_turn": self.invocation_in_turn,
                "request_id": request_id,
                "prompt_tokens": len(prompt_tokens),
                "stable_prefix_tokens": lcp,
                "messages": messages,
                "tools": tools,
                "cskcache_reuse": reuse,
                "status": "running",
            }
            started = time.perf_counter()
            try:
                response = original(*args, **kwargs)
                record.update(
                    status="completed",
                    elapsed_s=round(time.perf_counter() - started, 6),
                    response=serializable(response),
                )
                # A successful transport request populated vLLM's local prefix
                # cache even if a later SDK validation decides to retry it.
                self.previous_prompt_tokens = prompt_tokens
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
                path = self.request_dir / (
                    f"turn_{self.turn}_inv_{self.invocation_in_turn}.json"
                )
                atomic_write_json(path, record)
                self.records.append(record)
                print(
                    f"[CSKCache request] turn={self.turn} "
                    f"inv={self.invocation_in_turn} prompt={len(prompt_tokens)} "
                    f"reuse={reuse}",
                    flush=True,
                )

        object.__setattr__(wrapped, "_cskcache_real_agent_wrapped", True)
        object.__setattr__(self.llm, "_transport_call", wrapped)


def create_agent_and_llm(
    *,
    skills_dir: Path,
    model: str,
    vllm_port: int,
) -> tuple[Any, Any]:
    from openhands.sdk import (
        Agent,
        AgentContext,
        LLM,
        LLMSummarizingCondenser,
        Tool,
    )
    from openhands.sdk.context.skills import load_skills_from_dir
    from openhands.tools.apply_patch import ApplyPatchTool
    from openhands.tools.browser_use import BrowserToolSet
    from openhands.tools.gemini import (
        EditTool,
        ListDirectoryTool,
        ReadFileTool,
        WriteFileTool,
    )
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
        temperature=0.6,
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
    print(
        f"Loaded {len(loaded_skills)} OpenHands skills from {skills_path}: "
        f"{sorted(loaded_skills)}"
    )
    agent = Agent(
        llm=llm,
        tools=tools,
        include_default_tools=["FinishTool", "ThinkTool"],
        tool_concurrency_limit=1,
        system_prompt_filename="system_prompt.j2",
        agent_context=AgentContext(
            skills=list(loaded_skills.values()),
            system_message_suffix=(
                "Always be rigorous. You do not need to execute any code you "
                "write. Your only responsibility is to produce well-structured "
                "and complete code files."
            ),
        ),
        condenser=LLMSummarizingCondenser(
            llm=llm,
            max_size=240,
            keep_first=2,
        ),
    )
    return llm, agent


def prepare_task_directory(
    run_dir: Path,
    benchmark_repo: str,
    bench_root: Path,
    skills_dir: Path,
    overwrite: bool,
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
    if not benchmark_dir.is_dir():
        raise FileNotFoundError(f"Benchmark task does not exist: {benchmark_dir}")
    seed_dir = benchmark_dir / "seed_files"
    if seed_dir.is_dir():
        shutil.copytree(seed_dir, workspace / "seed_files")
    active_skills = workspace / ".agents" / "skills"
    active_skills.parent.mkdir(parents=True)
    shutil.copytree(skills_dir, active_skills)
    return task_dir, workspace, active_skills


def run_real_sequence(
    *,
    template: SequenceTemplate,
    agent: Any,
    workspace: Path,
    agent_log: Path,
    event_log: Path,
    injector: CSKCacheTransportInjector,
    max_iteration_per_run: int,
) -> None:
    from openhands.sdk import Conversation

    event_logger = OpenHandsEventLogger(event_log)
    try:
        conversation = Conversation(
            agent=agent,
            workspace=str(workspace),
            callbacks=[event_logger.on_event],
            max_iteration_per_run=max_iteration_per_run,
            stuck_detection=True,
            delete_on_close=True,
        )
    except Exception:
        event_logger.close()
        raise
    saved_stdout, saved_stderr = sys.stdout, sys.stderr
    log_handle = agent_log.open("w", encoding="utf-8")
    writer = StripAnsiWriter(log_handle)
    sys.stdout = writer
    sys.stderr = writer
    try:
        for turn, turn_spec in enumerate(template.turns, start=1):
            injector.start_turn(turn)
            event_logger.start_turn(turn)
            message = f"Working directory: {workspace}\n\n{turn_spec.message}"
            print(f"[TURN {turn}/{len(template.turns)}]", flush=True)
            conversation.send_message(message)
            conversation.run()
            log_handle.flush()
    finally:
        try:
            conversation.close()
        finally:
            event_logger.close()
            sys.stdout = saved_stdout
            sys.stderr = saved_stderr
            log_handle.close()


def main() -> None:
    args = parse_args()
    template = load_benchmark_sequence(args.benchmark_repo, args.bench_root)
    if args.max_turns is not None:
        template.turns = template.turns[: args.max_turns]

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    skill_texts, skill_tokens = load_skill_catalog(
        args.skills_dir, args.kv_dir, tokenizer
    )
    if args.dry_run:
        print(
            f"[dry-run] task={args.benchmark_repo} turns={len(template.turns)} "
            f"skills={len(skill_texts)} kv_dir={args.kv_dir}"
        )
        return

    task_dir, workspace, active_skills = prepare_task_directory(
        args.run_dir,
        args.benchmark_repo,
        args.bench_root,
        args.skills_dir,
        args.overwrite,
    )
    llm, agent = create_agent_and_llm(
        skills_dir=active_skills,
        model=args.model,
        vllm_port=args.vllm_port,
    )
    injector = CSKCacheTransportInjector(
        llm=llm,
        tokenizer=tokenizer,
        skill_texts=skill_texts,
        skill_tokens=skill_tokens,
        request_dir=task_dir / "requests",
    )
    injector.install()

    started = time.time()
    status = "completed"
    error: dict[str, str] | None = None
    try:
        run_real_sequence(
            template=template,
            agent=agent,
            workspace=workspace,
            agent_log=task_dir / "agent.log",
            event_log=task_dir / "openhands_events.jsonl",
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
            "reuse_requests": sum(
                record.get("cskcache_reuse") is not None
                for record in injector.records
            ),
            "reuse_entries": sum(
                len((record.get("cskcache_reuse") or {}).get("entries", []))
                for record in injector.records
            ),
            "elapsed_s": round(time.time() - started, 4),
            "workspace": str(workspace),
            "request_dir": str(task_dir / "requests"),
            "openhands_event_log": str(task_dir / "openhands_events.jsonl"),
        }
        atomic_write_json(task_dir / "_summary.json", summary)


if __name__ == "__main__":
    main()
