#!/usr/bin/env python3
"""Capture one Skill's fully recomputed KV inside a real OpenHands request."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[4]
THREE_ONE = Path(__file__).resolve().parent.parent / "3.1"
sys.path.insert(0, str(THREE_ONE))

from interactive_agent import (  # noqa: E402
    DEFAULT_EXTRA_SKILLS_DIR,
    DEFAULT_MODEL,
    DEFAULT_POOL,
    DEFAULT_SKILLS_DIR,
    build_skill_catalog,
    content_text,
    create_agent,
    find_subsequence,
    jsonable,
    load_cached_skills,
    tokenize_openhands_request,
)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def sha256_tokens(token_ids: list[int] | tuple[int, ...]) -> str:
    digest = hashlib.sha256()
    for token_id in token_ids:
        digest.update(int(token_id).to_bytes(4, "little", signed=False))
    return digest.hexdigest()


def attach_online_full_saver(
    llm: Any,
    skill: Any,
    base_url: str,
    api_key: str,
    model: str,
    event_path: Path,
) -> None:
    original = getattr(llm, "_transport_call")
    saved = False

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        nonlocal saved
        messages = jsonable(kwargs.get("messages") or [])
        searchable = "\n".join(
            content_text(message.get("content"))
            for message in messages
            if isinstance(message, dict)
        )
        if saved or skill.text not in searchable:
            # The first OpenHands request only chooses/loads the Skill.  It is
            # not an experimental KV source, so prevent ordinary LMCache prefix
            # storage from contaminating this case's isolated capture directory.
            extra_body = dict(kwargs.get("extra_body") or {})
            kv_params = dict(extra_body.get("kv_transfer_params") or {})
            kv_params["lmcache.skip_save"] = True
            extra_body["kv_transfer_params"] = kv_params
            kwargs["extra_body"] = extra_body
            return original(*args, **kwargs)

        prompt_ids = tokenize_openhands_request(
            base_url,
            api_key,
            model,
            messages,
            jsonable(kwargs.get("tools")),
        )
        starts = find_subsequence(prompt_ids, skill.token_ids)
        if len(starts) != 1:
            raise RuntimeError(
                f"expected one online {skill.name} span, found starts={starts}"
            )
        start = starts[0]
        end = start + len(skill.token_ids)
        online_hash = sha256_tokens(prompt_ids[start:end])
        if online_hash != skill.token_sha256:
            raise RuntimeError(
                f"online/offline token hash mismatch for Skill {skill.name}"
            )

        extra_body = dict(kwargs.get("extra_body") or {})
        kv_params = dict(extra_body.get("kv_transfer_params") or {})
        kv_params.pop("lmcache.skip_save", None)
        kv_params["lmcache_segmentia_save"] = {
            "segment_start": start,
            "segment_end": end,
        }
        extra_body["kv_transfer_params"] = kv_params
        kwargs["extra_body"] = extra_body
        atomic_json(
            event_path,
            {
                "schema_version": 1,
                "status": "sending",
                "skill": skill.name,
                "segment_start": start,
                "segment_end": end,
                "token_count": len(skill.token_ids),
                "token_ids_sha256": online_hash,
                "prompt_token_count": len(prompt_ids),
            },
        )
        response = original(*args, **kwargs)
        saved = True
        atomic_json(
            event_path,
            {
                "schema_version": 1,
                "status": "completed",
                "skill": skill.name,
                "segment_start": start,
                "segment_end": end,
                "token_count": len(skill.token_ids),
                "token_ids_sha256": online_hash,
                "prompt_token_count": len(prompt_ids),
            },
        )
        print(
            f"[online-full] Skill={skill.name} span=[{start},{end}) "
            f"tokens={len(skill.token_ids)}",
            flush=True,
        )
        return response

    object.__setattr__(llm, "_transport_call", wrapped)


def block_actions_after_capture(agent: Any, event_path: Path) -> None:
    """Allow SkillTool once, then stop before executing the sampled next action."""
    from openhands.sdk.conversation.state import ConversationExecutionStatus

    original = getattr(agent, "_execute_actions")

    def guarded(conversation: Any, action_events: list[Any], on_event: Any) -> None:
        if event_path.exists():
            record = json.loads(event_path.read_text(encoding="utf-8"))
            if record.get("status") == "completed":
                conversation.state.execution_status = (
                    ConversationExecutionStatus.FINISHED
                )
                print(
                    "[online-full] blocked post-capture tool execution",
                    flush=True,
                )
                return
        original(conversation, action_events, on_event)

    object.__setattr__(agent, "_execute_actions", guarded)


def wait_for_layer_group(kv_dir: Path, token_count: int, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_count = 0
    while time.monotonic() < deadline:
        sidecars = sorted(kv_dir.rglob("*.pt.meta.json"))
        valid = []
        for path in sidecars:
            metadata = json.loads(path.read_text(encoding="utf-8"))
            shape = metadata.get("shape")
            positions = metadata.get("cached_positions")
            if (
                metadata.get("memory_format") == "KV_2TD"
                and shape == [2, token_count, 1024]
                and isinstance(positions, dict)
                and positions.get("length") == token_count
            ):
                valid.append(path)
        last_count = len(valid)
        if last_count == 40:
            return
        if last_count > 40:
            raise RuntimeError(f"multiple online KV groups found in {kv_dir}")
        time.sleep(1)
    raise RuntimeError(
        f"timed out waiting for 40 online KV layers in {kv_dir}; found {last_count}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skill", required=True)
    parser.add_argument("--task-prompt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kv-dir", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--skills-dir", type=Path, default=DEFAULT_SKILLS_DIR)
    parser.add_argument("--extra-skills-dir", type=Path, default=DEFAULT_EXTRA_SKILLS_DIR)
    parser.add_argument("--pool-dir", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--served-model", default="Qwen3")
    parser.add_argument("--base-url", default="http://127.0.0.1:8015")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--save-timeout", type=float, default=180.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    existing_kv = (
        list(args.kv_dir.rglob("*.pt")) + list(args.kv_dir.rglob("*.pt.meta.json"))
        if args.kv_dir.exists()
        else []
    )
    if args.output.exists() or existing_kv:
        raise FileExistsError("capture output must be a fresh directory")
    args.workspace.mkdir(parents=True, exist_ok=True)
    args.kv_dir.mkdir(parents=True, exist_ok=True)
    catalog, _, _ = build_skill_catalog(
        args.skills_dir.resolve(),
        args.extra_skills_dir.resolve(),
        args.workspace / ".segmentia_skills",
        skills=[args.skill],
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, local_files_only=True, trust_remote_code=True
    )
    cached = load_cached_skills(catalog, args.pool_dir.resolve(), tokenizer, require_all=True)
    skill = cached[args.skill]

    agent_args = argparse.Namespace(
        served_model=args.served_model,
        base_url=args.base_url,
        api_key=args.api_key,
        skills_dir=catalog,
    )
    llm, agent = create_agent(agent_args, {})
    attach_online_full_saver(
        llm, skill, args.base_url, args.api_key, args.served_model, args.output
    )
    block_actions_after_capture(agent, args.output)

    from openhands.sdk import Conversation

    conversation = Conversation(
        agent=agent,
        workspace=str(args.workspace),
        max_iteration_per_run=2,
        stuck_detection=False,
        delete_on_close=True,
    )
    try:
        prompt = args.task_prompt.resolve().read_text(encoding="utf-8").strip()
        conversation.send_message(prompt)
        conversation.run()
    finally:
        conversation.close()

    if not args.output.exists():
        raise RuntimeError(
            f"OpenHands did not load the requested Skill {args.skill}; no KV was saved"
        )
    record = json.loads(args.output.read_text(encoding="utf-8"))
    if record.get("status") != "completed":
        raise RuntimeError(f"online capture did not complete: {record}")
    wait_for_layer_group(args.kv_dir, len(skill.token_ids), args.save_timeout)


if __name__ == "__main__":
    main()
