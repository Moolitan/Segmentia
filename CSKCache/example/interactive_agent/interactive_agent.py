#!/usr/bin/env python3
"""Run an interactive OpenHands agent with on-demand offline Skill KV reuse."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from pydantic import SecretStr
from transformers import AutoTokenizer

from cskcache import (
    CacheObjectMetadata,
    MetadataManager,
    build_skill_token_identity,
    fingerprint_full_token_chunks,
    fingerprint_model,
    fingerprint_tokenizer,
)
import config as cfg
from config_loader import backend_root, validate_config

AUTO_RESEARCH_COLLECTION = "Auto-claude-code-research-in-sleep"
SUPERPOWERS_COLLECTION = "superpowers"
SKILL_COLLECTIONS = (AUTO_RESEARCH_COLLECTION, SUPERPOWERS_COLLECTION)
try:
    BOOT_ID = Path("/proc/sys/kernel/random/boot_id").read_text(
        encoding="utf-8"
    ).strip()
except OSError:
    BOOT_ID = ""


class SkillScheduleWindowProbe:
    """Record optional OpenHands callback boundaries for timing experiments."""

    def __init__(self) -> None:
        self.fine_timeline_enabled = os.environ.get(
            "CSKCACHE_FINE_TIMELINE", "0"
        ) == "1"
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
        if self.fine_timeline_enabled:
            callback_unix_ns = time.time_ns()
            callback_monotonic_ns = time.monotonic_ns()
            self.agent_timeline_events.append(
                {
                    "boundary": "skill_observation_event_callback",
                    "skill_name": skill_name,
                    "event_id": str(getattr(event, "id", "")),
                    "action_id": str(getattr(event, "action_id", "")),
                    "tool_call_id": str(getattr(event, "tool_call_id", "")),
                    "boot_id": BOOT_ID,
                    "monotonic_ns": callback_monotonic_ns,
                    "unix_ns": callback_unix_ns,
                    "pid": os.getpid(),
                }
            )


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
    model_path: Path,
    require_all: bool = False,
) -> dict[str, CacheObjectMetadata]:
    """Resolve exact or chunk-authenticated cached candidates for each Skill."""

    catalog_path = pool_dir if pool_dir.is_file() else pool_dir / "catalog.json"
    if not catalog_path.is_file():
        raise FileNotFoundError(f"CSKCache Catalog does not exist: {catalog_path}")
    manager = MetadataManager(catalog_path, expected_layers=40)
    model_digest = fingerprint_model(model_path)
    tokenizer_digest = fingerprint_tokenizer(model_path)

    # Loading the Catalog validates all 40 extents.  Also ensure every referenced
    # sparse/raw container exists before starting a long-lived vLLM process.
    for container in manager.list_containers():
        raw_path = Path(container.raw_file_path)
        if not raw_path.is_file():
            raise FileNotFoundError(
                f"CSKCache raw container does not exist: {raw_path}"
            )
        if raw_path.stat().st_size < container.capacity_bytes:
            raise RuntimeError(
                "CSKCache raw container is shorter than Catalog capacity: "
                f"{raw_path}"
            )

    cached: dict[str, CacheObjectMetadata] = {}
    unavailable: list[str] = []
    for skill_path in sorted(skills_dir.glob("*/SKILL.md")):
        name = skill_path.parent.name
        resolved = skill_path.resolve()
        try:
            cache_object = manager.resolve_object(
                skill_name=name,
                model_fingerprint=model_digest,
                tokenizer_fingerprint=tokenizer_digest,
            )
        except KeyError:
            unavailable.append(f"{name}:missing")
            continue
        text = resolved.read_text(encoding="utf-8")
        token_identity = build_skill_token_identity(
            tokenizer,
            name,
            text,
        )
        skill_version = hashlib.sha256(
            token_identity.cache_text.encode("utf-8")
        ).hexdigest()
        if cache_object.start_marker_token_ids != (
            token_identity.start_marker_token_ids
        ):
            raise RuntimeError(f"offline locator tokens are stale for Skill {name}")
        exact_identity = (
            cache_object.skill_version == skill_version
            and cache_object.token_ids_sha256 == token_identity.token_ids_sha256
            and cache_object.token_count == len(token_identity.token_ids)
        )
        if exact_identity:
            expected_chunk_digests = fingerprint_full_token_chunks(
                token_identity.token_ids,
                cache_object.chunking.chunk_size_tokens,
            )
            if (
                cache_object.chunk_token_ids_sha256
                and cache_object.chunk_token_ids_sha256
                != expected_chunk_digests
            ):
                raise RuntimeError(
                    f"offline chunk authentication is stale for Skill {name}"
                )
        elif not cache_object.chunk_token_ids_sha256:
            unavailable.append(f"{name}:stale_exact_only")
            continue
        cached[name] = cache_object

    if require_all and unavailable:
        details = ", ".join(unavailable)
        raise RuntimeError(
            "explicit Skill selection requires compatible offline KV for every "
            f"exposed Skill; unavailable: {details}"
        )
    if not cached:
        raise RuntimeError(
            "no compatible CSKCache source objects found for "
            f"the exposed catalog in {pool_dir}"
        )
    preview = ", ".join(unavailable[:8])
    suffix = " ..." if len(unavailable) > 8 else ""
    print(
        f"[pool] cached CSKCache Skills={len(cached)}; "
        f"uncached/incompatible Skills={len(unavailable)}"
        + (f" ({preview}{suffix})" if unavailable else ""),
        flush=True,
    )
    return cached


def create_agent(skills_dir: Path):
    from openhands.sdk import Agent, AgentContext, LLM, Tool
    from openhands.sdk.context.skills import load_skills_from_dir
    from openhands.tools.apply_patch import ApplyPatchTool
    from openhands.tools.glob import GlobTool
    from openhands.tools.grep import GrepTool
    from openhands.tools.skill import SkillTool
    from openhands.tools.task_tracker import TaskTrackerTool
    from openhands.tools.terminal import TerminalTool

    llm = LLM(
        model=f"openai/{cfg.SERVED_MODEL}",
        api_key=SecretStr(cfg.API_KEY),
        base_url=f"http://127.0.0.1:{cfg.PORT}/v1",
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
    _, _, skills = load_skills_from_dir(str(skills_dir))
    tools = [
        Tool(name=TerminalTool.name, params={"terminal_type": "subprocess"}),
        Tool(name=GlobTool.name),
        Tool(name=GrepTool.name),
        Tool(name=ApplyPatchTool.name),
        Tool(name=TaskTrackerTool.name),
        Tool(
            name=SkillTool.name,
            params={
                "skills_dir": str(skills_dir),
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
    return agent


def main(*, check: bool = False) -> None:
    validate_config()
    skills_dir = cfg.SKILLS_DIR.resolve()
    extra_skills_dir = cfg.EXTRA_SKILLS_DIR.resolve()
    pool_dir = backend_root().resolve()
    workspace = cfg.WORKSPACE.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    skills_dir, selector, exposed_skill_count = build_skill_catalog(
        skills_dir,
        extra_skills_dir,
        workspace / ".cskcache_skills",
        skills=list(cfg.SKILLS) if cfg.SKILLS else None,
        collection=cfg.COLLECTION,
    )

    cached_skills: dict[str, CacheObjectMetadata] = {}
    if cfg.MODE == "cskcache":
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.MODEL_PATH,
            local_files_only=True,
            trust_remote_code=True,
        )
        cached_skills = load_cached_skills(
            skills_dir,
            pool_dir,
            tokenizer,
            cfg.MODEL_PATH,
            require_all=bool(cfg.SKILLS) or cfg.COLLECTION is not None,
        )
    if check:
        create_agent(skills_dir)
        print(
            f"[check] mode={cfg.MODE} OpenHands agent config with "
            f"{len(cached_skills)} cached CSKCache Skills "
            f"and {exposed_skill_count} exposed Skills "
            f"selector={selector} from {skills_dir}"
        )
        return
    schedule_probe = SkillScheduleWindowProbe()
    agent = create_agent(skills_dir)

    from openhands.sdk import Conversation

    conversation_options: dict[str, Any] = {
        "agent": agent,
        "workspace": str(workspace),
        "max_iteration_per_run": cfg.MAX_ITERATIONS,
        "stuck_detection": True,
        "delete_on_close": True,
        "callbacks": [schedule_probe.on_event],
    }
    # The fine-grained scheduling experiment measures callback latency.  Rich's
    # default visualizer renders and prints the complete Skill observation before
    # the experiment callback runs, so its terminal cost would be charged to the
    # Agent control path.  Keep normal interactive runs unchanged and disable the
    # visualizer only when the dedicated launcher explicitly requests it.
    if os.environ.get("CSKCACHE_DISABLE_VISUALIZER", "0") == "1":
        conversation_options["visualizer"] = None

    conversation = Conversation(
        **conversation_options,
    )
    print(f"[ready] workspace={workspace}")
    print(
        f"[ready] cached CSKCache Skills={len(cached_skills)}; "
        f"exposed Skills={exposed_skill_count}; selector={selector}; "
        f"steps_per_message={cfg.MAX_ITERATIONS}; enter /exit to quit"
    )
    try:
        if cfg.PROMPT_FILE is not None:
            message = cfg.PROMPT_FILE.read_text(encoding="utf-8").strip()
            if not message:
                raise ValueError(f"prompt file is empty: {cfg.PROMPT_FILE}")
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
        timeline_path = workspace / "cskcache_agent_timeline.json"
        timeline_path.write_text(
            json.dumps(
                schedule_probe.agent_timeline_events,
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"[done] CSKCache Agent timeline: {timeline_path}")


if __name__ == "__main__":
    main()
