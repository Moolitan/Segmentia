#!/usr/bin/env python3
"""Run an interactive OpenHands agent with on-demand offline Skill KV reuse."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import SecretStr
from transformers import AutoTokenizer

from skill_cache_tokens import (
    CACHE_OBJECT_TYPE,
    CACHE_SCHEMA_VERSION,
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
PREFIX_CORRECTION_POLICY = {
    "correction_mode": "prefix_k_headwise",
    "prefix_tokens": 256,
    "calibration_start": 132,
    "calibration_end": 256,
    "minimum_reuse_tokens": 256,
    "correction_alpha": 0.6,
}


@dataclass(frozen=True)
class CachedSkill:
    name: str
    skill_path: Path
    cache_id: str
    text: str
    token_ids: tuple[int, ...]
    token_sha256: str


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


def post_json(
    base_url: str,
    path: str,
    api_key: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            value = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {path}: {body}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object from {path}")
    return value


def find_subsequence(haystack: list[int], needle: tuple[int, ...]) -> list[int]:
    if not needle or len(needle) > len(haystack):
        return []
    first = needle[0]
    width = len(needle)
    return [
        index
        for index in range(len(haystack) - width + 1)
        if haystack[index] == first
        and tuple(haystack[index : index + width]) == needle
    ]


def tokenize_openhands_request(
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, Any]],
    tools: Any,
) -> list[int]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "add_generation_prompt": True,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    if tools:
        payload["tools"] = tools
    response = post_json(base_url, "/tokenize", api_key, payload)
    tokens = response.get("tokens")
    if not isinstance(tokens, list) or not all(isinstance(item, int) for item in tokens):
        raise RuntimeError("vLLM /tokenize returned invalid tokens")
    return tokens


def attach_skill_kv_injector(
    llm: Any,
    cached_skills: dict[str, CachedSkill],
    base_url: str,
    api_key: str,
    model: str,
    segmentia_mode: str = "direct_reuse",
) -> None:
    # agent 每次发 LLM 请求时在 LLM 请求里注入 lmcache_segmentia_lookup
    original = getattr(llm, "_transport_call") # 把原始的 _transport_call 方法保存下来，让我们可以在 wrapper 里调用原始方法，相当于掉包
    injected: set[str] = set()
    events: list[dict[str, Any]] = []

    def wrapped(*args: Any, **kwargs: Any) -> Any:
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
        span_start = 0
        span_end = 0
        if candidates:
            prompt_ids = tokenize_openhands_request(
                base_url,
                api_key,
                model,
                messages,
                jsonable(kwargs.get("tools")),
            )
            matches: list[tuple[CachedSkill, int]] = []
            for skill in candidates:
                starts = find_subsequence(prompt_ids, skill.token_ids)
                for start in starts:
                    matches.append((skill, start))

            if len(matches) != 1:
                names = [skill.name for skill, _ in matches]
                raise RuntimeError(
                    "one OpenHands request must introduce exactly one new cached "
                    f"Skill; token matches={names}"
                )
            pending, span_start = matches[0]
            span_end = span_start + len(pending.token_ids)
            if sha256_tokens(prompt_ids[span_start:span_end]) != pending.token_sha256:
                raise RuntimeError(f"online token hash mismatch for Skill {pending.name}")

            lookup = {
                "segment_start": span_start,
                "segment_end": span_end,
            }
            if segmentia_mode == "prefix_correction":
                lookup.update(PREFIX_CORRECTION_POLICY)
                lookup["cache_end"] = span_end
            elif segmentia_mode != "direct_reuse":
                raise ValueError(f"unsupported Segmentia mode: {segmentia_mode}")

            extra_body = dict(kwargs.get("extra_body") or {})
            extra_body["kv_transfer_params"] = {
                "lmcache_segmentia_lookup": lookup
            }
            kwargs["extra_body"] = extra_body
            print(
                f"[Segmentia] inject Skill={pending.name} "
                f"span=[{span_start},{span_end}) tokens={len(pending.token_ids)}",
                flush=True,
            )

        response = original(*args, **kwargs)
        if pending is not None:
            injected.add(pending.name)
            events.append(
                {
                    "skill": pending.name,
                    "cache_id": pending.cache_id,
                    "segment_start": span_start,
                    "segment_end": span_end,
                    "token_count": len(pending.token_ids),
                    "segmentia_mode": segmentia_mode,
                }
            )
        return response

    object.__setattr__(llm, "_transport_call", wrapped)
    object.__setattr__(llm, "_segmentia_skill_events", events)


def create_agent(
    args: argparse.Namespace,
    cached_skills: dict[str, CachedSkill],
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
        args.base_url,
        args.api_key,
        args.served_model,
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
        create_agent(args, cached_skills)
        print(
            f"[check] OpenHands agent config with "
            f"{len(cached_skills)} cached schema{CACHE_SCHEMA_VERSION} Skills "
            f"and {exposed_skill_count} exposed Skills "
            f"selector={selector} from {args.skills_dir}"
        )
        return
    llm, agent = create_agent(args, cached_skills)

    from openhands.sdk import Conversation

    conversation = Conversation(
        agent=agent,
        workspace=str(args.workspace),
        max_iteration_per_run=args.max_iterations,
        stuck_detection=True,
        delete_on_close=True,
    )
    print(f"[ready] workspace={args.workspace}")
    print(
        f"[ready] cached schema{CACHE_SCHEMA_VERSION} Skills={len(cached_skills)}; "
        f"exposed Skills={exposed_skill_count}; selector={selector}; "
        f"steps_per_message={args.max_iterations}; enter /exit to quit"
    )
    try:
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
        print(f"[done] Segmentia events: {events_path}")


if __name__ == "__main__":
    main()
