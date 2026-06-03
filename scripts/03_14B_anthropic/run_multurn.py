
import argparse
import copy
import hashlib
import json
import os
import re
import sys
import traceback
import time
import threading
import queue
from dataclasses import dataclass, field

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BENCH_ROOT = os.path.join(ROOT, "anthropic_skill_benchmark")  # default; overridden by --bench-root
sys.path.insert(0, ROOT)


from core.vllm_metrics import (
    attach_vllm_per_request_metrics,
)


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


class _AsyncRequestInputCollector:
    """
    Offload request input capture to a background thread so _transport_call
    stays lightweight. We only keep `messages`.
    """

    def __init__(self, llm, max_queue: int = 65536):
        self._llm = llm
        self._q: "queue.Queue[dict | None]" = queue.Queue(maxsize=max_queue)
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        try:
            self._q.put(None)
        except Exception:
            pass
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def enqueue(self, raw: dict) -> None:
        # Block if the consumer is behind. Dropping requests desyncs
        # _request_inputs from token_usages / vllm_deltas (which are indexed
        # by call_index_in_turn in run_sequence), so correctness wins over
        # latency here.
        self._q.put(raw)

    def drain(self, timeout: float = 30.0) -> None:
        """Block until every item enqueued so far has been consumed."""
        try:
            self._q.join()
        except Exception:
            pass

    def _run(self) -> None:
        while True:
            item = self._q.get()
            try:
                if item is None:
                    return
                try:
                    self._llm._request_inputs.append(
                        {"messages": copy.deepcopy(item.get("messages_raw"))}
                    )
                except Exception:
                    pass
            finally:
                try:
                    self._q.task_done()
                except Exception:
                    pass


def attach_llm_request_input_collector(llm) -> None:
    """
    Patch OpenHands SDK LLM instance to collect the actual request messages
    right before the transport sends them.
    """
    if getattr(llm, "_request_input_patched", False):
        return
    # 这些字段是运行时动态挂到 llm 实例上的，不是 SDK 类里预先声明的成员。
    # 打上补丁标记，避免同一个 llm 实例被重复包装。
    llm._request_input_patched = True
    # 保存按请求顺序采集到的原始输入，后面会按索引和 usage / delta 对齐。
    llm._request_inputs: list = []
    # 拿到 llm 当前真正负责发请求的底层 transport 函数，后面会包一层 wrapper。
    original = getattr(llm, "_transport_call", None)
    # 如果当前 llm 没有可调用的 transport 层，就无法继续做请求输入采集。
    if not callable(original):
        return

    # 大消息做 deepcopy 可能比较贵，所以交给后台 collector 异步处理，
    # 避免把这部分开销放在请求主路径上。
    collector = _AsyncRequestInputCollector(llm)
    collector.start()
    llm._request_input_collector = collector

    def wrapped(*args, **kwargs):
        # 在 transport 边界抓取实际传入的 messages。
        # 这里比更高层的 conversation 对象更接近真正发出去的请求内容，
        # 因为高层对象在发送前还可能继续被转换。
        messages = kwargs.get("messages")
        raw = {
            "messages_raw": messages,
        }
        try:
            return original(*args, **kwargs)
        finally:
            try:
                # 在 transport 调用后入队：既能按“每次实际请求尝试”记录一条，
                # 也尽量不阻塞主请求路径。
                collector.enqueue(raw)
            except Exception:
                pass

    object.__setattr__(wrapped, "_request_input_wrapped", True)
    # 只替换当前这个 llm 实例上的 transport 函数；
    # 后续请求都会先经过 wrapped()，从而把输入写入 _request_inputs。
    object.__setattr__(llm, "_transport_call", wrapped)


def _hash_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


_TOKENIZER_CACHE: dict = {}


def _get_tokenizer():
    """Lazy-load the HF tokenizer matching the vLLM-served model.

    Returns None if transformers / tokenizer files are unavailable; callers
    must then fall back to a char-ratio approximation.
    """
    if "tok" in _TOKENIZER_CACHE:
        return _TOKENIZER_CACHE["tok"]
    tok = None
    try:
        from transformers import AutoTokenizer
        path = os.environ.get(
            "VLLM_MODEL_PATH",
            "/mnt/Large_Language_Model_Lab_1/llm_models/Qwen3-14B/Qwen/Qwen3-14B",
        )
        tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    except Exception:
        tok = None
    _TOKENIZER_CACHE["tok"] = tok
    return tok


def _count_tokens(text: str, tokenizer) -> int | None:
    if tokenizer is None:
        return None
    try:
        return len(tokenizer.encode(text, add_special_tokens=False))
    except Exception:
        return None


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


def load_skill_doc(skills_dir: str, skill_name: str) -> tuple[str | None, str | None]:
    skill_path = os.path.join(skills_dir, skill_name, "SKILL.md")
    if not os.path.isfile(skill_path):
        return None, None
    with open(skill_path, encoding="utf-8") as f:
        return skill_path, f.read()


def _discover_skills(skills_dir: str) -> dict[str, str]:
    """Return ``{skill_name: SKILL.md text}`` for every skill under skills_dir."""
    out: dict[str, str] = {}
    if not os.path.isdir(skills_dir):
        return out
    for entry in sorted(os.listdir(skills_dir)):
        sub = os.path.join(skills_dir, entry)
        if not os.path.isdir(sub):
            continue
        _, text = load_skill_doc(skills_dir, entry)
        if text:
            out[entry] = text
    return out


def analyze_skill_doc_reuse(
    llm_calls: list[dict],
    skills_dir: str,
) -> dict | None:
    """Per-call analysis of whether each expected skill's KV cache was reused
    by vLLM.

    Two signals are computed per (call, skill):

    * String label — raw-text prefix hash (everything before the skill doc).
      ``first_use`` / ``exact_repeat`` (prefix hash seen before) /
      ``prefix_broken_repeat`` / ``not_present``. Raw text is used so that
      whitespace differences (which affect tokenization) are preserved.

    * ``kv_verdict`` — the ground-truth signal from vLLM.
      We estimate the token position of the skill doc's end in the rendered
      prompt (using the HF tokenizer when available, else a char-ratio
      approximation against ``prompt_tokens``). If
      ``vllm_prefix_cache_hits_tokens >= skill_end_token_pos`` then the
      skill's KV was actually reused.
    """
    skill_docs = _discover_skills(skills_dir)
    if not skill_docs:
        return None

    # Only analyze skills that appear in at least one prompt — a benchmark can
    # have hundreds of skills available but the agent only invokes a handful.
    skill_docs = {
        name: text
        for name, text in skill_docs.items()
        if any(text in (c.get("request_prompt_text") or "") for c in llm_calls)
    }
    if not skill_docs:
        return None

    tokenizer = _get_tokenizer()
    per_skill: dict[str, dict] = {}

    for name, skill_text in skill_docs.items():
        seen_prefix_hashes: set[str] = set()
        summary = {
            "not_present": 0,
            "first_use": 0,
            "exact_repeat": 0,
            "prefix_broken_repeat": 0,
            "kv_reused_by_vllm": 0,
            "kv_not_reused_by_vllm": 0,
            "kv_verdict_unknown": 0,
        }

        for call in llm_calls:
            prompt_text = call.get("request_prompt_text", "") or ""
            idx = prompt_text.find(skill_text)

            entry: dict = {"present": idx >= 0, "label": "not_present"}

            if idx < 0:
                summary["not_present"] += 1
            else:
                skill_end_char = idx + len(skill_text)
                prefix_raw = prompt_text[:idx]
                prefix_hash = _hash_text(prefix_raw)

                # skill_end_token_pos: tokenize prompt[:skill_end_char] if we
                # have a tokenizer, else approximate via char ratio.
                skill_end_token_pos = _count_tokens(prompt_text[:skill_end_char], tokenizer)
                if skill_end_token_pos is None:
                    prompt_tokens = call.get("prompt_tokens") or 0
                    if prompt_tokens and len(prompt_text):
                        skill_end_token_pos = int(
                            prompt_tokens * skill_end_char / len(prompt_text)
                        )

                if not seen_prefix_hashes:
                    label = "first_use"
                elif prefix_hash in seen_prefix_hashes:
                    label = "exact_repeat"
                else:
                    label = "prefix_broken_repeat"
                seen_prefix_hashes.add(prefix_hash)
                summary[label] += 1

                hits = call.get("vllm_prefix_cache_hits_tokens")
                if hits is None or skill_end_token_pos is None:
                    verdict = "unknown"
                    summary["kv_verdict_unknown"] += 1
                elif hits >= skill_end_token_pos:
                    verdict = "reused"
                    summary["kv_reused_by_vllm"] += 1
                else:
                    verdict = "not_reused"
                    summary["kv_not_reused_by_vllm"] += 1

                entry.update({
                    "label": label,
                    "skill_start_char": idx,
                    "skill_end_char": skill_end_char,
                    "skill_end_token_pos": skill_end_token_pos,
                    "prefix_hash": prefix_hash,
                    "kv_verdict": verdict,
                    "vllm_prefix_cache_hits_tokens": hits,
                })

            per_call_skills = call.setdefault("skills", {})
            per_call_skills[name] = entry

        per_skill[name] = {"summary": summary}

    return {
        "skills": per_skill,
        "tokenizer": "hf" if tokenizer is not None else "char_ratio_approx",
    }


def run_sequence(
    template: SequenceTemplate,
    agent,
    skills_dir: str,
    seq_workspace: str,
    seq_log_path: str,
    max_iteration_per_run: int = 500,
    vllm_port: int = 8000,
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
            try:
                metrics_before = (
                    conversation.conversation_stats.get_combined_metrics().deep_copy()
                )
            except Exception:
                metrics_before = None

            request_inputs_before = len(getattr(llm, "_request_inputs", []))
            request_deltas_before = len(getattr(llm, "_vllm_request_deltas", []))

            conversation.send_message(message)
            conversation.run()
            seq_log_file.flush()

            # Wait for the background collector to finish appending everything
            # this turn produced, otherwise _request_inputs may be shorter than
            # token_usages / vllm_deltas and the k-indexed pairing below drops
            # records.
            try:
                collector = getattr(llm, "_request_input_collector", None)
                if collector is not None:
                    collector.drain()
            except Exception:
                pass

            turn_usages = []
            try:
                metrics_after = conversation.conversation_stats.get_combined_metrics()
                if metrics_before is None:
                    turn_usages = metrics_after.token_usages
                else:
                    turn_usages = metrics_after.diff(metrics_before).token_usages
            except Exception:
                pass

            turn_request_inputs = getattr(llm, "_request_inputs", [])[request_inputs_before:]
            turn_request_deltas = getattr(llm, "_vllm_request_deltas", [])[request_deltas_before:]

            if not (len(turn_usages) == len(turn_request_inputs) == len(turn_request_deltas)):
                print(
                    f"[WARN] turn {turn_number} length mismatch: "
                    f"usages={len(turn_usages)} "
                    f"inputs={len(turn_request_inputs)} "
                    f"deltas={len(turn_request_deltas)}",
                    flush=True,
                )

            for k, usage in enumerate(turn_usages):

                prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
                record = {
                    "call_index_in_turn": k,
                    "turn_number": turn_number,
                    "prompt_tokens": prompt_tokens,
                    "request_prompt_text": "",
                    "vllm_prefix_cache_queries_tokens": None,
                    "vllm_prefix_cache_hits_tokens": None,
                    "vllm_prefix_cache_hit_rate": None,
                    "vllm_request_prefill_time_seconds": None,
                    "vllm_time_to_first_token_seconds": None,
                }

                try:
                    if 0 <= k < len(turn_request_inputs):
                        record["request_prompt_text"] = flatten_request_messages(
                            turn_request_inputs[k].get("messages", [])
                        )
                    if 0 <= k < len(turn_request_deltas):
                        delta = turn_request_deltas[k]
                        record["vllm_prefix_cache_queries_tokens"] = delta.get(
                            "vllm_prefix_cache_queries_tokens"
                        )
                        record["vllm_prefix_cache_hits_tokens"] = delta.get(
                            "vllm_prefix_cache_hits_tokens"
                        )
                        record["vllm_prefix_cache_hit_rate"] = delta.get(
                            "vllm_prefix_cache_hit_rate"
                        )
                        record["vllm_request_prefill_time_seconds"] = delta.get(
                            "vllm_request_prefill_time_seconds"
                        )
                        record["vllm_time_to_first_token_seconds"] = delta.get(
                            "vllm_time_to_first_token_seconds"
                        )
                except Exception:
                    pass

                all_llm_calls.append(record)
        _ = time.time() - seq_start

    finally:
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

    skill_doc_analysis = analyze_skill_doc_reuse(
        llm_calls=all_llm_calls,
        skills_dir=skills_dir,
    )

    return {
        "llm_calls": all_llm_calls,
        "skill_doc_analysis": skill_doc_analysis,
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
    attach_vllm_per_request_metrics(llm, vllm_port=vllm_port)

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
                "When initializing a Vite project, you MUST use: CI=true npx --yes create-vite <name> --template react. "
                "Never use interactive scaffold commands without --template. "
                "Never use port 8000; it is reserved by vLLM. Use another port (e.g. 5173, 3000) instead. "
                "Skills are knowledge guides — invoke them with the skill tool directly. "
                "Never pass a skill name as subagent_type in TaskAction; no subagent types are registered."
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
            skills_dir=skills_dir,
            seq_workspace=seq_workspace,
            seq_log_path=seq_log_path,
            vllm_port=args.vllm_port,
        )

        output = {
            "benchmark_repo": args.benchmark_repo,
            "llm_calls": result["llm_calls"],
            "skill_doc_analysis": result["skill_doc_analysis"],
        }

        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        n_calls = len(result["llm_calls"])
        print(f"\n完成! 共 {n_calls} 次 LLM 调用")
        print(f"  结果: {args.output}")
        print(f"  日志: {seq_log_path}")
        if result.get("skill_doc_analysis"):
            analysis = result["skill_doc_analysis"]
            print(f"  skill_doc_analysis (tokenizer={analysis.get('tokenizer')}):")
            for sname, sdata in analysis.get("skills", {}).items():
                print(f"    - {sname}: {sdata['summary']}")

    except Exception as e:
        print(f"\n[ERROR] {args.benchmark_repo}): {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
