#!/usr/bin/env python3
"""Run an interactive OpenHands agent without external KV reuse."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from pydantic import SecretStr


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_SKILLS_DIR = (
    ROOT / "skills" / "Auto-claude-code-research-in-sleep" / "skills"
)
DEFAULT_EXTRA_SKILLS_DIR = ROOT / "skills"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skills-dir", type=Path, default=DEFAULT_SKILLS_DIR)
    parser.add_argument(
        "--extra-skills-dir", type=Path, default=DEFAULT_EXTRA_SKILLS_DIR
    )
    parser.add_argument("--served-model", default="Qwen3")
    parser.add_argument("--base-url", default="http://127.0.0.1:8014")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument(
        "--workspace",
        type=Path,
        default=ROOT / "workspace" / "08_lmcache_mp" / "interactive_agent",
    )
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def build_skill_catalog(
    skills_dir: Path,
    extra_skills_dir: Path,
    catalog_dir: Path,
) -> Path:
    sources: dict[str, Path] = {}
    for source_dir in (skills_dir, extra_skills_dir):
        for skill_md in sorted(source_dir.glob("*/SKILL.md")):
            name = skill_md.parent.name
            resolved_dir = skill_md.parent.resolve()
            previous = sources.get(name)
            if previous is not None:
                raise RuntimeError(
                    f"duplicate exposed Skill name {name!r}: "
                    f"{previous} and {resolved_dir}"
                )
            sources[name] = resolved_dir
    if len(sources) != 99:
        raise RuntimeError(f"expected 99 exposed Skills, found {len(sources)}")

    catalog_dir.mkdir(parents=True, exist_ok=True)
    for entry in catalog_dir.iterdir():
        if not entry.is_symlink():
            raise RuntimeError(f"unexpected non-symlink in Skill catalog: {entry}")
        entry.unlink()
    for name, source_dir in sorted(sources.items()):
        (catalog_dir / name).symlink_to(source_dir, target_is_directory=True)
    return catalog_dir


def build_llm_options(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model": f"openai/{args.served_model}",
        "api_key": SecretStr(args.api_key),
        "base_url": f"{args.base_url.rstrip('/')}/v1",
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
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


def create_agent(args: argparse.Namespace):
    from openhands.sdk import Agent, AgentContext, LLM, Tool
    from openhands.sdk.context.skills import load_skills_from_dir
    from openhands.tools.apply_patch import ApplyPatchTool
    from openhands.tools.glob import GlobTool
    from openhands.tools.grep import GrepTool
    from openhands.tools.skill import SkillTool
    from openhands.tools.task_tracker import TaskTrackerTool
    from openhands.tools.terminal import TerminalTool

    llm = LLM(**build_llm_options(args))
    _, _, skills = load_skills_from_dir(str(args.skills_dir))
    tools = [
        Tool(name=TerminalTool.name, params={"terminal_type": "subprocess"}),
        Tool(name=GlobTool.name),
        Tool(name=GrepTool.name),
        Tool(name=ApplyPatchTool.name),
        Tool(name=TaskTrackerTool.name),
        Tool(name=SkillTool.name, params={"skills_dir": str(args.skills_dir)}),
    ]
    agent = Agent(
        llm=llm,
        tools=tools,
        include_default_tools=["FinishTool", "ThinkTool"],
        tool_concurrency_limit=1,
        agent_context=AgentContext(
            skills=list(skills.values()),
            load_public_skills=False,
            load_user_skills=False,
        ),
    )
    return agent


def main() -> None:
    args = parse_args()
    args.skills_dir = args.skills_dir.resolve()
    args.extra_skills_dir = args.extra_skills_dir.resolve()
    args.workspace = args.workspace.resolve()
    args.workspace.mkdir(parents=True, exist_ok=True)
    args.skills_dir = build_skill_catalog(
        args.skills_dir,
        args.extra_skills_dir,
        args.workspace / ".segmentia_skills",
    )
    agent = create_agent(args)
    if args.check:
        print(
            f"[check] no-reuse OpenHands agent config with 99 Skills "
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
    )
    print(f"[ready] workspace={args.workspace}")
    print("[ready] mode=no_reuse; Skills=99; enter /exit to quit")
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
        print("[done] no-reuse conversation closed")


if __name__ == "__main__":
    main()
