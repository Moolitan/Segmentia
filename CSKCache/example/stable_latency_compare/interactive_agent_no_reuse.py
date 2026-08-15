#!/usr/bin/env python3
"""Run an interactive OpenHands agent without external KV reuse."""
from __future__ import annotations

import argparse
import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import SecretStr
from agent_request_timing import AgentRequestTimingProbe


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_SKILLS_DIR = (
    ROOT / "skills" / "Auto-claude-code-research-in-sleep" / "skills"
)
DEFAULT_EXTRA_SKILLS_DIR = ROOT / "skills"
AUTO_RESEARCH_COLLECTION = "Auto-claude-code-research-in-sleep"
SUPERPOWERS_COLLECTION = "superpowers"
SKILL_COLLECTIONS = (AUTO_RESEARCH_COLLECTION, SUPERPOWERS_COLLECTION)
SCHEDULE_REQUEST_PREFIX = "cskcache-window-"


@dataclass(frozen=True)
class SkillObservationTimestamp:
    """Time at which one normal-Prefill Skill result becomes Agent-visible."""

    skill_name: str
    event_id: str
    event_timestamp: str
    action_id: str
    tool_call_id: str
    callback_unix_ns: int


class NormalPrefillScheduleProbe:
    """Pair pending Skill observations with the following LLM request."""

    def __init__(self) -> None:
        self.session_id = str(uuid.uuid4())
        self._pending: list[SkillObservationTimestamp] = []
        self.events: list[dict[str, Any]] = []
        self.transport_events: list[dict[str, Any]] = []

    def on_event(self, event: Any) -> None:
        if (
            event.__class__.__name__ != "ObservationEvent"
            or getattr(event, "tool_name", None) != "skill"
        ):
            return
        observation = getattr(event, "observation", None)
        skill_name = str(getattr(observation, "skill_name", "")).strip()
        if not skill_name or skill_name == "list":
            return
        self._pending.append(
            SkillObservationTimestamp(
                skill_name=skill_name,
                event_id=str(getattr(event, "id", "")),
                event_timestamp=str(getattr(event, "timestamp", "")),
                action_id=str(getattr(event, "action_id", "")),
                tool_call_id=str(getattr(event, "tool_call_id", "")),
                callback_unix_ns=time.time_ns(),
            )
        )

    def pop_for_next_request(self) -> list[SkillObservationTimestamp]:
        observations = self._pending
        self._pending = []
        return observations


def attach_normal_prefill_schedule_probe(
    llm: Any, probe: NormalPrefillScheduleProbe
) -> None:
    """Tag every request and pair the post-Skill request with pending results."""
    original = getattr(llm, "_transport_call")

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        observations = probe.pop_for_next_request()
        wrapper_enter_unix_ns = time.time_ns()
        request_token = f"{SCHEDULE_REQUEST_PREFIX}{uuid.uuid4().hex}"
        request_id = f"chatcmpl-{request_token}"
        extra_headers = dict(kwargs.get("extra_headers") or {})
        if any(key.lower() == "x-request-id" for key in extra_headers):
            raise RuntimeError("X-Request-Id is already set on the LLM request")
        extra_headers["X-Request-Id"] = request_token
        kwargs["extra_headers"] = extra_headers
        transport_handoff_unix_ns = time.time_ns()

        for observation in observations:
            probe.events.append(
                {
                    "skill": observation.skill_name,
                    "execution_mode": "normal_prefill",
                    "request_id": request_id,
                    "schedule_timing": {
                        "status": "awaiting_scheduler_admission",
                        "session_id": probe.session_id,
                        "observation_event_id": observation.event_id,
                        "observation_event_timestamp": observation.event_timestamp,
                        "action_id": observation.action_id,
                        "tool_call_id": observation.tool_call_id,
                        "observation_callback_unix_ns": observation.callback_unix_ns,
                        "request_wrapper_enter_unix_ns": wrapper_enter_unix_ns,
                        "client_transport_handoff_unix_ns": (
                            transport_handoff_unix_ns
                        ),
                    },
                }
            )
        response = original(*args, **kwargs)
        probe.transport_events.append(
            {
                "request_id": request_id,
                "request_wrapper_enter_unix_ns": wrapper_enter_unix_ns,
                "client_transport_handoff_unix_ns": transport_handoff_unix_ns,
                "client_response_received_unix_ns": time.time_ns(),
                "boundary": "client_transport_response_received",
            }
        )
        return response

    object.__setattr__(llm, "_transport_call", wrapped)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skills-dir", type=Path, default=DEFAULT_SKILLS_DIR)
    parser.add_argument(
        "--extra-skills-dir", type=Path, default=DEFAULT_EXTRA_SKILLS_DIR
    )
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument(
        "--skill",
        action="append",
        help=(
            "Expose one Skill, resolved across all supported sources. Repeat "
            "--skill to expose a controlled multi-Skill catalog."
        ),
    )
    selector.add_argument(
        "--collection",
        choices=SKILL_COLLECTIONS,
        help="Expose every Skill in one multi-Skill collection.",
    )
    parser.add_argument("--served-model", default="Qwen3")
    parser.add_argument("--base-url", default="http://127.0.0.1:8014")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument(
        "--workspace",
        type=Path,
        default=ROOT / "workspace" / "08_lmcache_mp" / "interactive_agent",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=2,
        help=(
            "Agent steps per user message. The default captures exactly the Skill "
            "load step and the first post-Skill completion."
        ),
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        help="Send the complete UTF-8 file as one user message, then exit.",
    )
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def build_skill_catalog(
    skills_dir: Path,
    extra_skills_dir: Path,
    catalog_dir: Path,
    skills: list[str] | None = None,
    collection: str | None = None,
) -> tuple[Path, str, int]:
    groups: dict[str, dict[str, Path]] = {
        AUTO_RESEARCH_COLLECTION: {},
        SUPERPOWERS_COLLECTION: {},
        "standalone": {},
    }
    source_dirs = {
        AUTO_RESEARCH_COLLECTION: skills_dir,
        SUPERPOWERS_COLLECTION: extra_skills_dir / "superpowers" / "skills",
        "standalone": extra_skills_dir,
    }
    all_sources: dict[str, Path] = {}
    for group_name, source_dir in source_dirs.items():
        for skill_md in sorted(source_dir.glob("*/SKILL.md")):
            name = skill_md.parent.name
            resolved_dir = skill_md.parent.resolve()
            previous = all_sources.get(name)
            if previous is not None:
                raise RuntimeError(
                    f"duplicate exposed Skill name {name!r}: "
                    f"{previous} and {resolved_dir}"
                )
            groups[group_name][name] = resolved_dir
            all_sources[name] = resolved_dir

    if skills is not None:
        duplicates = sorted({name for name in skills if skills.count(name) > 1})
        if duplicates:
            raise RuntimeError(
                "duplicate --skill selection: " + ", ".join(duplicates)
            )
        sources = {}
        for name in skills:
            if name in SKILL_COLLECTIONS:
                raise RuntimeError(
                    f"{name!r} is a Skill collection; use "
                    f"--collection {name} instead"
                )
            source = all_sources.get(name)
            if source is None:
                available = ", ".join(sorted(all_sources))
                raise RuntimeError(
                    f"unknown Skill {name!r}; available Skills: {available}"
                )
            sources[name] = source
        selector = (
            f"skill:{skills[0]}"
            if len(skills) == 1
            else "skills:" + ",".join(skills)
        )
    elif collection is not None:
        sources = groups[collection]
        selector = f"collection:{collection}"
    else:
        sources = {
            **groups[AUTO_RESEARCH_COLLECTION],
            **groups["standalone"],
        }
        selector = "default:auto+standalone"
    if not sources:
        raise RuntimeError(f"Skill selector {selector} matched no Skills")

    catalog_dir.mkdir(parents=True, exist_ok=True)
    for entry in catalog_dir.iterdir():
        if not entry.is_symlink():
            raise RuntimeError(f"unexpected non-symlink in Skill catalog: {entry}")
        entry.unlink()
    for name, source_dir in sorted(sources.items()):
        (catalog_dir / name).symlink_to(source_dir, target_is_directory=True)
    return catalog_dir, selector, len(sources)


def build_llm_options(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model": f"openai/{args.served_model}",
        "api_key": SecretStr(args.api_key),
        "base_url": f"{args.base_url.rstrip('/')}/v1",
        "temperature": 0,
        "top_p": 1.0,
        "stream": False,
        "native_tool_calling": True,
        "drop_params": True,
        "modify_params": True,
        "litellm_extra_body": {
            "chat_template_kwargs": {"enable_thinking": True},
            "min_p": 0,
        },
        "log_completions": False,
        "disable_vision": True,
        "timeout": 720,
    }


def create_agent(
    args: argparse.Namespace,
    schedule_probe: NormalPrefillScheduleProbe,
    request_timing_probe: AgentRequestTimingProbe | None = None,
):
    from openhands.sdk import Agent, AgentContext, LLM, Tool
    from openhands.sdk.context.skills import load_skills_from_dir
    from openhands.tools.apply_patch import ApplyPatchTool
    from openhands.tools.glob import GlobTool
    from openhands.tools.grep import GrepTool
    from openhands.tools.skill import SkillTool
    from openhands.tools.task_tracker import TaskTrackerTool
    from openhands.tools.terminal import TerminalTool

    llm = LLM(**build_llm_options(args))
    if request_timing_probe is not None and request_timing_probe.trace_path:
        request_timing_probe.attach(llm)
    else:
        attach_normal_prefill_schedule_probe(llm, schedule_probe)
    _, _, skills = load_skills_from_dir(str(args.skills_dir))
    tools = [
        Tool(name=TerminalTool.name, params={"terminal_type": "subprocess"}),
        Tool(name=GlobTool.name),
        Tool(name=GrepTool.name),
        Tool(name=ApplyPatchTool.name),
        Tool(name=TaskTrackerTool.name),
        Tool(
            name=SkillTool.name,
            params={
                "skills_dir": str(args.skills_dir),
                "context_segment_wrapper": True,
            },
        ),
    ]
    agent = Agent(
        llm=llm,
        tools=tools,
        include_default_tools=["FinishTool", "ThinkTool"],
        tool_concurrency_limit=1,
        system_prompt_filename="system_prompt_skill.j2",
        agent_context=AgentContext(
            skills=list(skills.values()),
            load_public_skills=False,
            load_user_skills=False,
        ),
    )
    return llm, agent


def main() -> None:
    args = parse_args()
    args.skills_dir = args.skills_dir.resolve()
    args.extra_skills_dir = args.extra_skills_dir.resolve()
    args.workspace = args.workspace.resolve()
    if args.prompt_file is not None:
        args.prompt_file = args.prompt_file.resolve()
        if not args.prompt_file.is_file():
            raise FileNotFoundError(f"prompt file does not exist: {args.prompt_file}")
    args.workspace.mkdir(parents=True, exist_ok=True)
    args.skills_dir, selector, exposed_skill_count = build_skill_catalog(
        args.skills_dir,
        args.extra_skills_dir,
        args.workspace / ".cskcache_skills",
        skills=args.skill,
        collection=args.collection,
    )
    schedule_probe = NormalPrefillScheduleProbe()
    request_timing_probe = AgentRequestTimingProbe()
    llm, agent = create_agent(args, schedule_probe, request_timing_probe)
    if args.check:
        print(
            f"[check] no-reuse OpenHands agent config with "
            f"{exposed_skill_count} Skills selector={selector} "
            f"from {args.skills_dir}"
        )
        return

    from openhands.sdk import Conversation

    conversation = Conversation(
        agent=agent,
        workspace=str(args.workspace),
        max_iteration_per_run=args.max_iterations,
        stuck_detection=True,
        delete_on_close=True,
        callbacks=[schedule_probe.on_event, request_timing_probe.on_event],
    )
    print(f"[ready] workspace={args.workspace}")
    print(
        f"[ready] mode=no_reuse; Skills={exposed_skill_count}; "
        f"selector={selector}; "
        f"steps_per_message={args.max_iterations}; enter /exit to quit"
    )
    try:
        if args.prompt_file is not None:
            message = args.prompt_file.read_text(encoding="utf-8").strip()
            if not message:
                raise ValueError(f"prompt file is empty: {args.prompt_file}")
            conversation.send_message(message)
            conversation.run()
        else:
            while True:
                try:
                    message = input("\nYou> ").strip()
                except EOFError:
                    break
                if message in {"/exit", "/quit"}:
                    break
                if not message:
                    continue
                conversation.send_message(message)
                conversation.run()
    finally:
        conversation.close()
        events_path = args.workspace / "normal_prefill_schedule_events.json"
        events_path.write_text(
            json.dumps(schedule_probe.events, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        transport_path = (
            args.workspace / "normal_prefill_transport_events.json"
        )
        transport_path.write_text(
            json.dumps(
                schedule_probe.transport_events,
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"[done] normal-Prefill schedule events: {events_path}")
        print(f"[done] normal-Prefill transport events: {transport_path}")


if __name__ == "__main__":
    main()
