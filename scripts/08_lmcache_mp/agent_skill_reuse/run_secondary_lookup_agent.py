"""Run a real OpenHands task with one-shot LMCache secondary lookup.

This keeps the existing agent/benchmark plumbing from ``run_agent.py`` but
changes the transport injector.  Whenever a skill guide enters the
conversation, the immediately following LLM request is rendered locally with
the same Qwen chat template used by vLLM.  The injector locates that guide's
opening LMCache separator in the rendered token IDs and sends its first content
token as ``lmcache_secondary_lookup.segment_start``.

Historical skill messages remain separator-wrapped, but only a newly observed
skill result receives an explicit secondary lookup request.  Later requests
inherit the already-corrected history through vLLM's local prefix cache.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import run_agent as base


DEFAULT_TOKENIZER_PATH = Path(
    os.environ.get(
        "VLLM_MODEL_PATH",
        "/mnt/Large_Language_Model_Lab_1/llm_models/Qwen3-14B/Qwen/Qwen3-14B",
    )
)


class ReuseValidationError(RuntimeError):
    """The agent run completed a step without satisfying reuse invariants."""


def load_tokenizer(path: Path) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path, local_files_only=True)


def repeated_skill_instruction() -> str:
    return (
        "Experimental cache-reuse requirement for this turn: before producing "
        "a substantive response, use read_file to reopen the SKILL.md explicitly "
        "specified by this benchmark turn, even if you read it in an earlier "
        "turn. Do not rely only on skill content already present in conversation "
        "history."
    )


def canonicalize_skill_tool_messages(
    messages: list[dict[str, Any]],
    skill_tool_call_ids: set[str],
    canonical_skill_contents: dict[str, str],
) -> list[dict[str, Any]]:
    normalized = copy.deepcopy(messages)
    for message in normalized:
        tool_call_id = message.get("tool_call_id")
        if (
            message.get("role") != "tool"
            or tool_call_id not in skill_tool_call_ids
        ):
            continue
        content = canonical_skill_contents.get(tool_call_id)
        if content is None:
            raise RuntimeError(
                f"No canonical SKILL.md content for tool call {tool_call_id}"
            )
        message["content"] = f"{base.BLEND_SEP}{content}{base.BLEND_SEP}"
    return normalized


def validate_cold_miss_then_hit(
    *,
    vllm_log: Path,
    injector: "SecondaryLookupTransportInjector",
    repeated_skills: list[str],
) -> dict[str, Any]:
    marker = "SEGMENTIA_SECONDARY_LOOKUP_EVENT "
    events: list[dict[str, Any]] = []
    decoder = json.JSONDecoder()
    for line in vllm_log.read_text(encoding="utf-8", errors="replace").splitlines():
        if marker not in line:
            continue
        payload = line.split(marker, 1)[1].lstrip()
        try:
            event, _ = decoder.raw_decode(payload)
        except json.JSONDecodeError:
            continue
        events.append(event)

    result: dict[str, Any] = {}
    for skill_name in repeated_skills:
        injected = [
            record
            for record in injector.records
            if record.get("status") == "completed"
            and (record.get("target_skill") or {}).get("skill_name") == skill_name
        ]
        if len(injected) < 2:
            raise ReuseValidationError(
                f"Skill {skill_name!r} has fewer than two completed injections"
            )
        first, second = injected[:2]

        def external_apply(record: dict[str, Any]) -> dict[str, Any]:
            engine_request_id = (record.get("response") or {}).get("id")
            if not isinstance(engine_request_id, str) or not engine_request_id:
                raise ReuseValidationError(
                    f"Request {record['request_id']} response has no engine id"
                )
            matches = [
                event
                for event in events
                if event.get("event") == "secondary_lookup_external_apply"
                and (
                    event.get("request_id") == engine_request_id
                    or (
                        isinstance(event.get("request_id"), str)
                        and event["request_id"].startswith(f"{engine_request_id}-")
                    )
                )
            ]
            if len(matches) != 1:
                raise ReuseValidationError(
                    f"Request {record['request_id']} has {len(matches)} "
                    "secondary_lookup_external_apply events"
                )
            return matches[0]

        first_event = external_apply(first)
        second_event = external_apply(second)
        first_is_cold_miss = (
            first_event.get("lookup_start") == first["segment_start"]
            and first_event.get("matched_end") == first["segment_start"]
            and first_event.get("external_tokens_applied") == 0
        )
        second_is_hit = (
            second_event.get("lookup_start") == second["segment_start"]
            and isinstance(second_event.get("matched_end"), int)
            and second_event["matched_end"] > second_event.get("lookup_cursor", -1)
            and isinstance(second_event.get("external_tokens_applied"), int)
            and second_event["external_tokens_applied"] > 0
        )
        result[skill_name] = {
            "first_request_id": first["request_id"],
            "second_request_id": second["request_id"],
            "first_is_cold_miss": first_is_cold_miss,
            "second_is_external_hit": second_is_hit,
            "first_event": first_event,
            "second_event": second_event,
        }
        if not first_is_cold_miss or not second_is_hit:
            raise ReuseValidationError(
                f"Skill {skill_name!r} did not show cold-miss then external-hit: "
                f"first_is_cold_miss={first_is_cold_miss}, "
                f"second_is_external_hit={second_is_hit}"
            )
    return result


def effective_separator_tokens(tokenizer: Any) -> list[int]:
    """Mirror LMCache SegmentTokenDatabase's current separator conversion."""

    encoded = list(tokenizer.encode(base.BLEND_SEP))
    effective = encoded[1:]
    if not effective:
        raise ValueError(
            "LMCache effective separator is empty after tokenizer.encode(separator)[1:]"
        )
    return effective


def _subsequence_starts(values: list[int], needle: list[int]) -> list[int]:
    width = len(needle)
    return [
        index
        for index in range(len(values) - width + 1)
        if values[index : index + width] == needle
    ]


def render_prompt_token_ids(
    *, tokenizer: Any, messages: list[dict[str, Any]], tools: Any
) -> list[int]:
    normalized_tools = base.serializable(tools) if tools is not None else None
    rendered = tokenizer.apply_chat_template(
        messages,
        tools=normalized_tools,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    if hasattr(rendered, "input_ids"):
        rendered = rendered.input_ids
    if not isinstance(rendered, list) or not all(isinstance(token, int) for token in rendered):
        raise TypeError("Tokenizer chat template did not return a flat list of token IDs")
    if not rendered:
        raise ValueError("Tokenizer chat template rendered an empty prompt")
    return rendered


def locate_skill_segment_start(
    *,
    tokenizer: Any,
    messages: list[dict[str, Any]],
    tools: Any,
    skill_tool_call_ids: set[str],
    target_tool_call_id: str,
) -> tuple[int, int, list[int]]:
    """Return ``(segment_start, rendered_length, separator_tokens)``.

    Each wrapped skill tool message contributes exactly an opening and closing
    effective separator.  The target message's rank therefore determines its
    opening occurrence without depending on the skill text itself.
    """

    ordered_skill_ids = [
        str(message.get("tool_call_id"))
        for message in messages
        if message.get("role") == "tool"
        and message.get("tool_call_id") in skill_tool_call_ids
    ]
    if target_tool_call_id not in ordered_skill_ids:
        raise ValueError(
            f"Pending skill result {target_tool_call_id!r} is absent from request messages"
        )
    if len(ordered_skill_ids) != len(set(ordered_skill_ids)):
        raise ValueError("A skill tool_call_id occurs more than once in request messages")

    rendered_ids = render_prompt_token_ids(tokenizer=tokenizer, messages=messages, tools=tools)
    separator = effective_separator_tokens(tokenizer)
    occurrences = _subsequence_starts(rendered_ids, separator)
    expected = 2 * len(ordered_skill_ids)
    if len(occurrences) != expected:
        raise ValueError(
            "Rendered separator count does not match wrapped skill messages: "
            f"found={len(occurrences)} expected={expected}"
        )

    target_rank = ordered_skill_ids.index(target_tool_call_id)
    opening = occurrences[2 * target_rank]
    segment_start = opening + len(separator)
    if not 0 < segment_start < len(rendered_ids):
        raise ValueError(
            f"Invalid segment_start={segment_start} for prompt length={len(rendered_ids)}"
        )
    return segment_start, len(rendered_ids), separator


class SecondaryLookupTransportInjector:
    """Wrap skill results and inject a secondary lookup exactly once per result."""

    def __init__(self, *, llm: Any, tokenizer: Any, request_dir: Path) -> None:
        self.llm = llm
        self.tokenizer = tokenizer
        self.request_dir = request_dir
        self.request_dir.mkdir(parents=True, exist_ok=True)
        self.skill_tool_call_ids: set[str] = set()
        self.pending_skill_tool_call_ids: set[str] = set()
        self.skill_calls: dict[str, dict[str, Any]] = {}
        self.canonical_skill_contents: dict[str, str] = {}
        self.skill_reads_by_name: dict[str, list[dict[str, Any]]] = {}
        self.secondary_injections_by_skill: dict[str, int] = {}
        self.turn = 0
        self.invocation_in_turn = 0
        self.request_index = 0
        self.records: list[dict[str, Any]] = []
        self.secondary_injection_count = 0

    def start_turn(self, turn: int) -> None:
        self.turn = turn
        self.invocation_in_turn = 0

    def record_skill_call(
        self, *, tool_call_id: str, skill_name: str, skill_path: str
    ) -> None:
        if tool_call_id not in self.skill_tool_call_ids:
            path = Path(skill_path)
            if path.name != "SKILL.md" or not path.is_file():
                raise RuntimeError(
                    f"Observed skill path is not a readable SKILL.md: {skill_path}"
                )
            canonical_content = path.read_text(encoding="utf-8")
            if not canonical_content.strip():
                raise RuntimeError(f"Observed SKILL.md is empty: {skill_path}")
            self.skill_tool_call_ids.add(tool_call_id)
            self.pending_skill_tool_call_ids.add(tool_call_id)
            self.canonical_skill_contents[tool_call_id] = canonical_content
            read = {
                "turn": self.turn,
                "tool_call_id": tool_call_id,
                "skill_name": skill_name,
                "skill_path": skill_path,
            }
            self.skill_calls[tool_call_id] = read
            self.skill_reads_by_name.setdefault(skill_name, []).append(read)

    def install(self) -> None:
        original = getattr(self.llm, "_transport_call", None)
        if not callable(original):
            raise RuntimeError("OpenHands LLM has no callable _transport_call")
        if getattr(original, "_lmcache_secondary_lookup_wrapped", False):
            raise RuntimeError("OpenHands LLM transport is already secondary-lookup wrapped")

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            raw_messages = kwargs.get("messages") or []
            if not isinstance(raw_messages, list):
                raise TypeError("LLM transport messages must be a list")
            messages = canonicalize_skill_tool_messages(
                raw_messages,
                self.skill_tool_call_ids,
                self.canonical_skill_contents,
            )
            kwargs["messages"] = messages

            pending_in_request = [
                tool_call_id
                for tool_call_id in self.pending_skill_tool_call_ids
                if any(
                    message.get("role") == "tool"
                    and message.get("tool_call_id") == tool_call_id
                    for message in raw_messages
                )
            ]
            if len(pending_in_request) > 1:
                raise RuntimeError(
                    "Multiple new skill results reached one request; secondary lookup "
                    f"requires one boundary at a time: {sorted(pending_in_request)}"
                )

            self.request_index += 1
            self.invocation_in_turn += 1
            request_id = (
                f"lmcache-secondary-turn{self.turn:02d}-"
                f"inv{self.invocation_in_turn:03d}-req{self.request_index:04d}"
            )
            extra_body = dict(kwargs.get("extra_body") or {})
            extra_body["request_id"] = request_id
            existing_kv_params = extra_body.get("kv_transfer_params")
            if existing_kv_params is not None and not isinstance(
                existing_kv_params, dict
            ):
                raise RuntimeError("extra_body.kv_transfer_params must be a mapping")

            target_tool_call_id: str | None = None
            segment_start: int | None = None
            rendered_token_count: int | None = None
            separator_tokens: list[int] | None = None
            secondary_config: dict[str, Any] | None = None
            if pending_in_request:
                target_tool_call_id = pending_in_request[0]
                segment_start, rendered_token_count, separator_tokens = (
                    locate_skill_segment_start(
                        tokenizer=self.tokenizer,
                        messages=messages,
                        tools=kwargs.get("tools"),
                        skill_tool_call_ids=self.skill_tool_call_ids,
                        target_tool_call_id=target_tool_call_id,
                    )
                )
                if existing_kv_params:
                    raise RuntimeError(
                        "Refusing to overwrite existing extra_body.kv_transfer_params"
                    )
                secondary_config = {
                    "lmcache_secondary_lookup": {
                        "segment_start": segment_start,
                        "probe_only": False,
                    }
                }
                extra_body["kv_transfer_params"] = secondary_config
            else:
                ordinary_params = dict(existing_kv_params or {})
                if ordinary_params.get("lmcache.skip_save") not in (None, True):
                    raise RuntimeError(
                        "Ordinary request has conflicting lmcache.skip_save"
                    )
                ordinary_params["lmcache.skip_save"] = True
                extra_body["kv_transfer_params"] = ordinary_params
            kwargs["extra_body"] = extra_body

            record: dict[str, Any] = {
                "schema_version": 1,
                "request_index": self.request_index,
                "turn": self.turn,
                "invocation_in_turn": self.invocation_in_turn,
                "request_id": request_id,
                "messages": messages,
                "target_skill_tool_call_id": target_tool_call_id,
                "target_skill": (
                    self.skill_calls[target_tool_call_id]
                    if target_tool_call_id is not None
                    else None
                ),
                "rendered_token_count": rendered_token_count,
                "segment_start": segment_start,
                "effective_separator_tokens": separator_tokens,
                "secondary_lookup_config": secondary_config,
                "kv_transfer_params": extra_body["kv_transfer_params"],
                "pending_deferred": bool(
                    self.pending_skill_tool_call_ids and not pending_in_request
                ),
                "status": "running",
            }
            started = time.perf_counter()
            try:
                response = original(*args, **kwargs)
                if target_tool_call_id is not None:
                    self.pending_skill_tool_call_ids.remove(target_tool_call_id)
                    self.secondary_injection_count += 1
                    skill_name = self.skill_calls[target_tool_call_id]["skill_name"]
                    self.secondary_injections_by_skill[skill_name] = (
                        self.secondary_injections_by_skill.get(skill_name, 0) + 1
                    )
                record.update(
                    status="completed",
                    elapsed_s=round(time.perf_counter() - started, 6),
                    response=base.serializable(response),
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
                path = self.request_dir / (
                    f"turn_{self.turn}_inv_{self.invocation_in_turn}.json"
                )
                base.atomic_write_json(path, record)
                self.records.append(record)
                print(
                    f"[secondary request] turn={self.turn} "
                    f"inv={self.invocation_in_turn} status={record['status']} "
                    f"segment_start={segment_start}",
                    flush=True,
                )

        object.__setattr__(wrapped, "_lmcache_secondary_lookup_wrapped", True)
        object.__setattr__(self.llm, "_transport_call", wrapped)


def run_secondary_sequence(
    *,
    template: base.SequenceTemplate,
    agent: Any,
    workspace: Path,
    agent_log: Path,
    transcript_path: Path,
    injector: SecondaryLookupTransportInjector,
    max_iteration_per_run: int,
) -> None:
    """Run turns in order and require a real SKILL.md read in every turn."""

    from openhands.sdk import Conversation
    from openhands.tools.gemini import ReadFileTool
    from openhands.tools.skill import SkillTool

    def on_event(event: Any) -> None:
        if type(event).__name__ == "ObservationEvent":
            tool_name = getattr(event, "tool_name", None)
            observation = event.observation
            if getattr(observation, "is_error", False):
                return
            if tool_name == SkillTool.name:
                skill_name = getattr(observation, "skill_name", "")
                skill_path = getattr(observation, "skill_path", "")
                if (
                    skill_name
                    and skill_name != "list"
                    and Path(skill_path).name == "SKILL.md"
                ):
                    injector.record_skill_call(
                        tool_call_id=event.tool_call_id,
                        skill_name=skill_name,
                        skill_path=skill_path,
                    )
            elif tool_name == ReadFileTool.name:
                skill_path = getattr(observation, "file_path", "")
                path = Path(skill_path)
                if path.name == "SKILL.md" and not getattr(
                    observation, "is_truncated", False
                ):
                    injector.record_skill_call(
                        tool_call_id=event.tool_call_id,
                        skill_name=path.parent.name,
                        skill_path=skill_path,
                    )
        answer = base.extract_answer_text(event)
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
    writer = base.StripAnsiWriter(log_handle)
    sys.stdout = writer
    sys.stderr = writer
    try:
        for turn, turn_spec in enumerate(template.turns, start=1):
            injector.start_turn(turn)
            reads_before = sum(
                len(reads) for reads in injector.skill_reads_by_name.values()
            )
            instruction = repeated_skill_instruction()
            message = (
                f"Working directory: {workspace}\n\n{turn_spec.message}"
                f"\n\n{instruction}"
            )
            with transcript_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    f"\n{'=' * 20} TURN {turn}/{len(template.turns)} {'=' * 20}\n"
                )
                handle.write(f"[user]\n{turn_spec.message}\n\n[assistant]\n")
            print(f"[TURN {turn}/{len(template.turns)}]", flush=True)
            conversation.send_message(message)
            conversation.run()
            unconsumed = sorted(
                tool_call_id
                for tool_call_id in injector.pending_skill_tool_call_ids
                if injector.skill_calls[tool_call_id]["turn"] == turn
            )
            if unconsumed:
                raise ReuseValidationError(
                    f"Turn {turn} ended with unconsumed skill reads: {unconsumed}"
                )
            reads_after = sum(
                len(reads) for reads in injector.skill_reads_by_name.values()
            )
            if reads_after <= reads_before:
                raise ReuseValidationError(
                    f"Turn {turn} did not reopen a complete SKILL.md"
                )
            log_handle.flush()
    finally:
        conversation.close()
        sys.stdout = saved_stdout
        sys.stderr = saved_stderr
        log_handle.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-repo", required=True)
    parser.add_argument("--bench-root", type=Path, default=base.DEFAULT_BENCH_ROOT)
    parser.add_argument("--skills-dir", type=Path, default=base.DEFAULT_SKILLS_DIR)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--vllm-log", type=Path, default=None)
    parser.add_argument("--model", default=os.environ.get("VLLM_SERVED_NAME", "Qwen3"))
    parser.add_argument("--vllm-port", type=int, default=8100)
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument("--max-iteration-per-run", type=int, default=500)
    parser.add_argument("--min-skill-reads", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    template = base.load_benchmark_sequence(args.benchmark_repo, args.bench_root)
    if args.max_turns is not None:
        template.turns = template.turns[: args.max_turns]
    skill_count = base.count_skills(args.skills_dir)
    if args.min_skill_reads < 2:
        raise ValueError("--min-skill-reads must be at least 2")
    if len(template.turns) < args.min_skill_reads:
        raise ValueError(
            f"Task has {len(template.turns)} turns but requires at least "
            f"{args.min_skill_reads} repeated skill reads"
        )
    tokenizer = load_tokenizer(args.tokenizer_path)
    separator_tokens = effective_separator_tokens(tokenizer)

    if args.dry_run:
        print(
            f"[dry-run] task={args.benchmark_repo} turns={len(template.turns)} "
            f"skills={skill_count} tokenizer={args.tokenizer_path} "
            f"min_repeated_reads={args.min_skill_reads} "
            f"blend_sep={base.BLEND_SEP!r} effective_sep={separator_tokens}"
        )
        return
    if args.vllm_log is None:
        raise ValueError("--vllm-log is required for a real run")

    task_dir, workspace, active_skills = base.prepare_task_directory(
        args.run_dir,
        args.benchmark_repo,
        args.bench_root,
        args.skills_dir,
        args.overwrite,
    )
    llm, agent = base.create_agent_and_llm(
        skills_dir=active_skills, model=args.model, vllm_port=args.vllm_port
    )
    injector = SecondaryLookupTransportInjector(
        llm=llm, tokenizer=tokenizer, request_dir=task_dir / "requests"
    )
    injector.install()

    transcript_path = task_dir / "answers.txt"
    transcript_path.write_text(
        f"Benchmark: {args.benchmark_repo}\nModel: {args.model}\n"
        f"Mode: lmcache-secondary-lookup\n"
        f"Minimum repeated reads: {args.min_skill_reads}\n"
        f"Blend separator: {base.BLEND_SEP!r}\n",
        encoding="utf-8",
    )

    started = time.time()
    status = "completed"
    error: dict[str, str] | None = None
    cache_validation: dict[str, Any] | None = None
    repeated_skills: list[str] = []
    try:
        run_secondary_sequence(
            template=template,
            agent=agent,
            workspace=workspace,
            agent_log=task_dir / "agent.log",
            transcript_path=transcript_path,
            injector=injector,
            max_iteration_per_run=args.max_iteration_per_run,
        )
        if injector.pending_skill_tool_call_ids:
            raise ReuseValidationError(
                "Run ended with unconsumed skill reads: "
                f"{sorted(injector.pending_skill_tool_call_ids)}"
            )
        repeated_skills = sorted(
            name
            for name, reads in injector.skill_reads_by_name.items()
            if len(reads) >= args.min_skill_reads
            and injector.secondary_injections_by_skill.get(name, 0)
            >= args.min_skill_reads
        )
        if not repeated_skills:
            raise ReuseValidationError(
                "No observed skill was read and successfully injected at least "
                f"{args.min_skill_reads} times"
            )
        cache_validation = validate_cold_miss_then_hit(
            vllm_log=args.vllm_log,
            injector=injector,
            repeated_skills=repeated_skills,
        )
    except ReuseValidationError as exc:
        status = "no_go"
        error = {"type": type(exc).__name__, "message": str(exc)}
        raise
    except Exception as exc:
        status = "failed"
        error = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        summary = {
            "schema_version": 1,
            "mode": "lmcache-secondary-lookup",
            "benchmark_repo": args.benchmark_repo,
            "status": status,
            "error": error,
            "turns": len(template.turns),
            "min_skill_reads": args.min_skill_reads,
            "repeated_skills": repeated_skills,
            "skill_reads_by_name": injector.skill_reads_by_name,
            "requests": len(injector.records),
            "secondary_injections": injector.secondary_injection_count,
            "secondary_injections_by_skill": injector.secondary_injections_by_skill,
            "cold_miss_then_hit_validation": cache_validation,
            "pending_skill_tool_call_ids": sorted(
                injector.pending_skill_tool_call_ids
            ),
            "elapsed_s": round(time.time() - started, 4),
            "workspace": str(workspace),
            "request_dir": str(task_dir / "requests"),
            "transcript": str(transcript_path),
        }
        base.atomic_write_json(task_dir / "_summary.json", summary)


if __name__ == "__main__":
    main()
