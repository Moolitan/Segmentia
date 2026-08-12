#!/usr/bin/env python3
"""Run an interactive OpenHands agent with on-demand offline Skill KV reuse."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import SecretStr
from transformers import AutoTokenizer

from skill_cache_tokens import (
    CACHE_OBJECT_TYPE,
    CACHE_SCHEMA_VERSION,
    LOCATOR_KIND,
    context_segment_start_marker_text,
    qwen_context_segment_start_marker_token_ids,
    qwen_context_segment_token_ids,
)


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_SKILLS_DIR = (
    ROOT / "skills" / "Auto-claude-code-research-in-sleep" / "skills"
)
DEFAULT_EXTRA_SKILLS_DIR = ROOT / "skills"
AUTO_RESEARCH_COLLECTION = "Auto-claude-code-research-in-sleep"
SUPERPOWERS_COLLECTION = "superpowers"
SKILL_COLLECTIONS = (AUTO_RESEARCH_COLLECTION, SUPERPOWERS_COLLECTION)
DEFAULT_POOL = Path(
    "/mnt/Large_Language_Model_Lab_1/wsh/skill_save_pool/Qwen3-14B"
)
DEFAULT_MODEL = Path(
    "/mnt/Large_Language_Model_Lab_1/llm_models/Qwen3-14B/Qwen/Qwen3-14B"
)
SEGMENTIA_MODES = ("direct_reuse", "prefix_correction")
SCHEDULE_REQUEST_PREFIX = "segmentia-window-"
PREFIX_CORRECTION_POLICY = {
    "correction_mode": "prefix_k_headwise",
    "prefix_tokens": 256,
    "calibration_start": 132,
    "calibration_end": 256,
    "minimum_reuse_tokens": 256,
    "correction_alpha": 0.6,
}
try:
    BOOT_ID = Path("/proc/sys/kernel/random/boot_id").read_text(
        encoding="utf-8"
    ).strip()
except OSError:
    BOOT_ID = ""


@dataclass(frozen=True)
class CachedSkill:
    name: str
    skill_path: Path
    cache_id: str
    text: str
    token_ids: tuple[int, ...]
    token_sha256: str
    start_marker_token_ids: tuple[int, ...]
    start_marker_token_sha256: str


@dataclass(frozen=True)
class SkillObservationTimestamp:
    """The instant when OpenHands makes one Skill result visible to the Agent."""

    event_id: str
    event_timestamp: str
    action_id: str
    tool_call_id: str
    callback_unix_ns: int
    callback_monotonic_ns: int


class SkillScheduleWindowProbe:
    """Pair Skill observations with the next completion transport handoff."""

    def __init__(self) -> None:
        self.session_id = str(uuid.uuid4())
        self.fine_timeline_enabled = os.environ.get(
            "SEGMENTIA_FINE_TIMELINE", "0"
        ) == "1"
        self._pending: dict[str, list[SkillObservationTimestamp]] = {}
        self.transport_events: list[dict[str, Any]] = []
        self.agent_timeline_events: list[dict[str, Any]] = []

    def on_event(self, event: Any) -> None:
        """Retain the source tool-call ID until request B is constructed."""
        if (
            self.fine_timeline_enabled
            and event.__class__.__name__ == "ActionEvent"
            and getattr(event, "tool_name", None) == "skill"
        ):
            self.agent_timeline_events.append(
                {
                    "boundary": "skill_action_event_callback",
                    "action_id": str(getattr(event, "id", "")),
                    "tool_call_id": str(getattr(event, "tool_call_id", "")),
                    "boot_id": BOOT_ID,
                    "monotonic_ns": time.monotonic_ns(),
                    "unix_ns": time.time_ns(),
                    "pid": os.getpid(),
                }
            )
            return
        if (
            event.__class__.__name__ != "ObservationEvent"
            or getattr(event, "tool_name", None) != "skill"
        ):
            return
        observation = getattr(event, "observation", None)
        skill_name = str(getattr(observation, "skill_name", "")).strip()
        if not skill_name or skill_name == "list":
            return
        timestamp = SkillObservationTimestamp(
            event_id=str(getattr(event, "id", "")),
            event_timestamp=str(getattr(event, "timestamp", "")),
            action_id=str(getattr(event, "action_id", "")),
            tool_call_id=str(getattr(event, "tool_call_id", "")),
            callback_unix_ns=time.time_ns(),
            callback_monotonic_ns=time.monotonic_ns(),
        )
        if self.fine_timeline_enabled:
            self.agent_timeline_events.append(
                {
                    "boundary": "skill_observation_event_callback",
                    "skill_name": skill_name,
                    "event_id": timestamp.event_id,
                    "action_id": timestamp.action_id,
                    "tool_call_id": timestamp.tool_call_id,
                    "boot_id": BOOT_ID,
                    "monotonic_ns": timestamp.callback_monotonic_ns,
                    "unix_ns": timestamp.callback_unix_ns,
                    "pid": os.getpid(),
                }
            )
        self._pending.setdefault(skill_name, []).append(timestamp)

    def pop_observation(self, skill_name: str) -> SkillObservationTimestamp | None:
        observations = self._pending.get(skill_name)
        if not observations:
            return None
        timestamp = observations.pop(0)
        if not observations:
            del self._pending[skill_name]
        return timestamp

    def peek_observation(self, skill_name: str) -> SkillObservationTimestamp | None:
        """Return the pending Skill result without consuming it.

        Request B must carry the tool-call ID produced by request A so vLLM can
        join T0 with the later request entirely from its own traces.  The
        observation stays pending until request B succeeds, preserving retry
        behavior.
        """
        observations = self._pending.get(skill_name)
        return observations[0] if observations else None


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
    parser.add_argument("--pool-dir", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--served-model", default="Qwen3")
    parser.add_argument("--base-url", default="http://127.0.0.1:8014")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument(
        "--segmentia-mode",
        choices=SEGMENTIA_MODES,
        default="direct_reuse",
        help="Select direct reuse or the frozen Section 3.2 prefix correction.",
    )
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


def sha256_tokens(token_ids: list[int] | tuple[int, ...]) -> str:
    digest = hashlib.sha256()
    for token_id in token_ids:
        digest.update(int(token_id).to_bytes(4, "little", signed=False))
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


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


def load_cached_skills(
    skills_dir: Path,
    pool_dir: Path,
    tokenizer: Any,
    require_all: bool = False,
) -> dict[str, CachedSkill]:
    manifests: dict[Path, tuple[Path, dict[str, Any]]] = {}
    for manifest_path in sorted(pool_dir.rglob("manifest.json")):
        try:
            record = read_json(manifest_path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        skill_path = record.get("skill_path")
        if not isinstance(skill_path, str):
            continue
        manifests[Path(skill_path).resolve()] = (manifest_path, record)

    cached: dict[str, CachedSkill] = {}
    unavailable: list[str] = []
    for skill_path in sorted(skills_dir.glob("*/SKILL.md")):
        name = skill_path.parent.name
        resolved = skill_path.resolve()
        found = manifests.get(resolved)
        if found is None:
            unavailable.append(f"{name}:missing")
            continue
        manifest_path, record = found
        text = resolved.read_text(encoding="utf-8")
        schema_version = record.get("schema_version")
        if schema_version != CACHE_SCHEMA_VERSION:
            unavailable.append(f"{name}:schema-{schema_version}")
            continue
        if (
            record.get("cache_object") != CACHE_OBJECT_TYPE
            or record.get("skill_name") != name
        ):
            raise RuntimeError(
                f"schema {CACHE_SCHEMA_VERSION} cache metadata is invalid for "
                f"Skill {name}: {manifest_path}"
            )
        if record.get("status") != "completed":
            raise RuntimeError(
                f"schema {CACHE_SCHEMA_VERSION} cache is not completed for "
                f"Skill {name}: {manifest_path}"
            )
        if not manifest_path.with_name("COMPLETED").is_file():
            raise RuntimeError(
                f"schema {CACHE_SCHEMA_VERSION} cache has no COMPLETED marker "
                f"for Skill {name}: {manifest_path}"
            )
        token_ids = qwen_context_segment_token_ids(tokenizer, name, text)
        token_digest = sha256_tokens(token_ids)
        if record.get("token_ids_sha256") != token_digest:
            raise RuntimeError(f"offline token hash is stale for Skill {name}")
        if record.get("token_count") != len(token_ids):
            raise RuntimeError(f"offline token count is stale for Skill {name}")
        locator = record.get("locator")
        if not isinstance(locator, dict) or locator.get("kind") != LOCATOR_KIND:
            raise RuntimeError(f"offline locator is missing for Skill {name}")
        start_marker_token_ids = qwen_context_segment_start_marker_token_ids(
            tokenizer, name
        )
        start_marker_digest = sha256_tokens(start_marker_token_ids)
        if locator.get("start_marker_text") != context_segment_start_marker_text(
            name
        ):
            raise RuntimeError(f"offline locator text is stale for Skill {name}")
        if locator.get("start_marker_token_ids") != start_marker_token_ids:
            raise RuntimeError(f"offline locator tokens are stale for Skill {name}")
        if locator.get("start_marker_token_count") != len(start_marker_token_ids):
            raise RuntimeError(f"offline locator count is stale for Skill {name}")
        if locator.get("start_marker_token_ids_sha256") != start_marker_digest:
            raise RuntimeError(f"offline locator hash is stale for Skill {name}")
        if token_ids[: len(start_marker_token_ids)] != start_marker_token_ids:
            raise RuntimeError(f"offline locator is not a prefix for Skill {name}")
        kv_dir = manifest_path.parent / "kv"
        if len(list(kv_dir.glob("*.pt.meta.json"))) != 40:
            raise RuntimeError(f"offline KV is incomplete for Skill {name}: {kv_dir}")
        cached[name] = CachedSkill(
            name=name,
            skill_path=resolved,
            cache_id=str(record["cache_id"]),
            text=text,
            token_ids=tuple(token_ids),
            token_sha256=token_digest,
            start_marker_token_ids=tuple(start_marker_token_ids),
            start_marker_token_sha256=start_marker_digest,
        )

    if require_all and unavailable:
        details = ", ".join(unavailable)
        raise RuntimeError(
            "explicit Skill selection requires compatible offline KV for every "
            f"exposed Skill; unavailable: {details}"
        )
    if not cached:
        raise RuntimeError(
            f"no compatible schema {CACHE_SCHEMA_VERSION} Skill KV found for "
            f"the exposed catalog in {pool_dir}"
        )
    preview = ", ".join(unavailable[:8])
    suffix = " ..." if len(unavailable) > 8 else ""
    print(
        f"[pool] cached schema{CACHE_SCHEMA_VERSION} Skills={len(cached)}; "
        f"uncached/incompatible Skills={len(unavailable)}"
        + (f" ({preview}{suffix})" if unavailable else ""),
        flush=True,
    )
    return cached


def content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(item.get("text", ""))
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        )
    return ""


def jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return jsonable(value.model_dump(exclude_none=True))
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def attach_skill_kv_injector(
    llm: Any,
    cached_skills: dict[str, CachedSkill],
    schedule_probe: SkillScheduleWindowProbe,
    segmentia_mode: str = "direct_reuse",
) -> None:
    # agent 每次发 LLM 请求时在 LLM 请求里注入 lmcache_segmentia_lookup
    original = getattr(llm, "_transport_call") # 把原始的 _transport_call 方法保存下来，让我们可以在 wrapper 里调用原始方法，相当于掉包
    injected: set[str] = set()
    events: list[dict[str, Any]] = []

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        # These OpenHands timestamps are retained only for legacy diagnostics.
        # The formal T0--T3 experiment records every boundary inside vLLM.
        wrapper_enter_unix_ns = time.time_ns()
        wrapper_enter_monotonic_ns = time.monotonic_ns()
        request_token = f"{SCHEDULE_REQUEST_PREFIX}{uuid.uuid4().hex}"
        request_id = f"chatcmpl-{request_token}"
        extra_headers = dict(kwargs.get("extra_headers") or {})
        if any(key.lower() == "x-request-id" for key in extra_headers):
            raise RuntimeError("X-Request-Id is already set on the LLM request")
        extra_headers["X-Request-Id"] = request_token
        kwargs["extra_headers"] = extra_headers
        messages = jsonable(kwargs.get("messages") or [])
        searchable = "\n".join(
            content_text(message.get("content"))
            for message in messages
            if isinstance(message, dict)
        )
        candidates = [
            skill
            for name, skill in cached_skills.items()
            if name not in injected
            and skill.text in searchable
        ]
        pending: CachedSkill | None = None
        source_observation: SkillObservationTimestamp | None = None
        if candidates:
            if len(candidates) != 1:
                names = [skill.name for skill in candidates]
                raise RuntimeError(
                    "one OpenHands request must introduce exactly one new cached "
                    f"Skill; content matches={names}"
                )
            pending = candidates[0]
            source_observation = schedule_probe.peek_observation(pending.name)
            if source_observation is None:
                raise RuntimeError(
                    f"cached Skill {pending.name!r} has no pending Skill "
                    "Observation for source tool-call correlation"
                )

            lookup = {
                "cache_id": pending.cache_id,
                "skill_name": pending.name,
                "source_tool_call_id": source_observation.tool_call_id,
                "token_count": len(pending.token_ids),
                "token_ids_sha256": pending.token_sha256,
                "locator": {
                    "kind": LOCATOR_KIND,
                    "start_marker_token_ids": list(
                        pending.start_marker_token_ids
                    ),
                    "start_marker_token_count": len(
                        pending.start_marker_token_ids
                    ),
                    "start_marker_token_ids_sha256": (
                        pending.start_marker_token_sha256
                    ),
                },
            }
            if segmentia_mode == "prefix_correction":
                lookup.update(PREFIX_CORRECTION_POLICY)
            elif segmentia_mode != "direct_reuse":
                raise ValueError(f"unsupported Segmentia mode: {segmentia_mode}")

            extra_body = dict(kwargs.get("extra_body") or {})
            extra_body["kv_transfer_params"] = {
                "lmcache_segmentia_lookup": lookup
            }
            kwargs["extra_body"] = extra_body
            print(
                f"[Segmentia] inject Skill={pending.name} "
                f"locator=manifest tokens={len(pending.token_ids)}",
                flush=True,
            )

        dispatch_unix_ns = time.time_ns()
        dispatch_monotonic_ns = time.monotonic_ns()
        response = original(*args, **kwargs)
        response_received_unix_ns = time.time_ns()
        response_received_monotonic_ns = time.monotonic_ns()
        schedule_probe.transport_events.append(
            {
                "request_id": request_id,
                "boot_id": BOOT_ID,
                "request_wrapper_enter_unix_ns": wrapper_enter_unix_ns,
                "request_wrapper_enter_monotonic_ns": wrapper_enter_monotonic_ns,
                "client_transport_handoff_unix_ns": dispatch_unix_ns,
                "client_transport_handoff_monotonic_ns": dispatch_monotonic_ns,
                "client_response_received_unix_ns": response_received_unix_ns,
                "client_response_received_monotonic_ns": (
                    response_received_monotonic_ns
                ),
                "boundary": "client_transport_response_received",
            }
        )
        if pending is not None:
            injected.add(pending.name)
            observation = schedule_probe.pop_observation(pending.name)
            if observation is None:
                schedule_timing: dict[str, Any] = {
                    "status": "missing_skill_observation",
                    "session_id": schedule_probe.session_id,
                }
            else:
                if observation != source_observation:
                    raise RuntimeError(
                        "pending Skill Observation changed while request B was "
                        "in flight"
                    )
                observation_to_wrapper_ns = (
                    wrapper_enter_monotonic_ns - observation.callback_monotonic_ns
                )
                wrapper_prepare_ns = (
                    dispatch_monotonic_ns - wrapper_enter_monotonic_ns
                )
                observation_to_dispatch_ns = (
                    dispatch_monotonic_ns - observation.callback_monotonic_ns
                )
                schedule_timing = {
                    "status": "ok",
                    "session_id": schedule_probe.session_id,
                    "observation_event_id": observation.event_id,
                    "observation_event_timestamp": observation.event_timestamp,
                    "action_id": observation.action_id,
                    "tool_call_id": observation.tool_call_id,
                    "observation_callback_unix_ns": observation.callback_unix_ns,
                    "request_wrapper_enter_unix_ns": wrapper_enter_unix_ns,
                    "completion_transport_handoff_unix_ns": dispatch_unix_ns,
                    "client_transport_handoff_unix_ns": dispatch_unix_ns,
                    "observation_callback_monotonic_ns": (
                        observation.callback_monotonic_ns
                    ),
                    "request_wrapper_enter_monotonic_ns": (
                        wrapper_enter_monotonic_ns
                    ),
                    "completion_transport_handoff_monotonic_ns": (
                        dispatch_monotonic_ns
                    ),
                    "observation_to_wrapper_ms": observation_to_wrapper_ns / 1e6,
                    "wrapper_prepare_ms": wrapper_prepare_ns / 1e6,
                    "observation_to_dispatch_ms": (
                        observation_to_dispatch_ns / 1e6
                    ),
                }
            events.append(
                {
                    "skill": pending.name,
                    "cache_id": pending.cache_id,
                    "request_id": request_id,
                    "segment_start": None,
                    "segment_end": None,
                    "span_owner": "vllm_post_tokenization_locator",
                    "token_count": len(pending.token_ids),
                    "segmentia_mode": segmentia_mode,
                    "schedule_timing": schedule_timing,
                }
            )
        return response

    object.__setattr__(llm, "_transport_call", wrapped)
    object.__setattr__(llm, "_segmentia_skill_events", events)


def create_agent(
    args: argparse.Namespace,
    cached_skills: dict[str, CachedSkill],
    schedule_probe: SkillScheduleWindowProbe,
):
    from openhands.sdk import Agent, AgentContext, LLM, Tool
    from openhands.sdk.context.skills import load_skills_from_dir
    from openhands.tools.apply_patch import ApplyPatchTool
    from openhands.tools.glob import GlobTool
    from openhands.tools.grep import GrepTool
    from openhands.tools.skill import SkillTool
    from openhands.tools.task_tracker import TaskTrackerTool
    from openhands.tools.terminal import TerminalTool

    llm = LLM(
        model=f"openai/{args.served_model}",
        api_key=SecretStr(args.api_key),
        base_url=f"{args.base_url.rstrip('/')}/v1",
        temperature=0,
        top_p=1.0,
        stream=False,
        native_tool_calling=True,
        drop_params=True,
        modify_params=True,
        litellm_extra_body={
            "chat_template_kwargs": {"enable_thinking": True},
            "min_p": 0,
        },
        log_completions=False,
        disable_vision=True,
        timeout=720,
    )
    attach_skill_kv_injector(
        llm,
        cached_skills,
        schedule_probe,
        args.segmentia_mode,
    )

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
    args.pool_dir = args.pool_dir.resolve()
    args.workspace = args.workspace.resolve()
    args.workspace.mkdir(parents=True, exist_ok=True)
    args.skills_dir, selector, exposed_skill_count = build_skill_catalog(
        args.skills_dir,
        args.extra_skills_dir,
        args.workspace / ".segmentia_skills",
        skills=args.skill,
        collection=args.collection,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    cached_skills = load_cached_skills(
        args.skills_dir,
        args.pool_dir,
        tokenizer,
        require_all=args.skill is not None or args.collection is not None,
    )
    if args.check:
        create_agent(args, cached_skills, SkillScheduleWindowProbe())
        print(
            f"[check] OpenHands agent config with "
            f"{len(cached_skills)} cached schema{CACHE_SCHEMA_VERSION} Skills "
            f"and {exposed_skill_count} exposed Skills "
            f"selector={selector} from {args.skills_dir}"
        )
        return
    schedule_probe = SkillScheduleWindowProbe()
    llm, agent = create_agent(args, cached_skills, schedule_probe)

    from openhands.sdk import Conversation

    conversation_options: dict[str, Any] = {
        "agent": agent,
        "workspace": str(args.workspace),
        "max_iteration_per_run": args.max_iterations,
        "stuck_detection": True,
        "delete_on_close": True,
        "callbacks": [schedule_probe.on_event],
    }
    # The fine-grained scheduling experiment measures callback latency.  Rich's
    # default visualizer renders and prints the complete Skill observation before
    # the experiment callback runs, so its terminal cost would be charged to the
    # Agent control path.  Keep normal interactive runs unchanged and disable the
    # visualizer only when the dedicated launcher explicitly requests it.
    if os.environ.get("SEGMENTIA_DISABLE_VISUALIZER", "0") == "1":
        conversation_options["visualizer"] = None

    conversation = Conversation(
        **conversation_options,
    )
    print(f"[ready] workspace={args.workspace}")
    print(
        f"[ready] cached schema{CACHE_SCHEMA_VERSION} Skills={len(cached_skills)}; "
        f"exposed Skills={exposed_skill_count}; selector={selector}; "
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
        events_path = args.workspace / "segmentia_skill_events.json"
        events_path.write_text(
            json.dumps(
                getattr(llm, "_segmentia_skill_events", []),
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        transport_path = args.workspace / "segmentia_transport_events.json"
        transport_path.write_text(
            json.dumps(
                schedule_probe.transport_events,
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        timeline_path = args.workspace / "segmentia_agent_timeline.json"
        timeline_path.write_text(
            json.dumps(
                schedule_probe.agent_timeline_events,
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"[done] Segmentia events: {events_path}")
        print(f"[done] Segmentia transport events: {transport_path}")
        print(f"[done] Segmentia Agent timeline: {timeline_path}")


if __name__ == "__main__":
    main()
