"""Deterministic runtime smoke checks for the pinned SDK/Tools image."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import shlex
from pathlib import Path
from types import SimpleNamespace

from openhands.sdk import AgentContext
from openhands.sdk.skills import load_skills_from_dir
from openhands.sdk.skills.execute import render_content_with_commands
from openhands.sdk.tool.builtins.invoke_skill import (
    InvokeSkillAction,
    InvokeSkillExecutor,
)
from openhands.tools.terminal import TerminalAction
from openhands.tools.terminal.impl import TerminalExecutor


EXPECTED_VERSION = "1.43.1"
ARTIFACT = {"producer": "openhands-tools==1.43.1", "status": "ok"}


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skills-dir", type=Path, required=True)
    parser.add_argument("--skill", required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--artifact", default="terminal_smoke_artifact.json")
    args = parser.parse_args()

    versions = {
        package: importlib.metadata.version(package)
        for package in ("openhands-sdk", "openhands-tools")
    }
    if set(versions.values()) != {EXPECTED_VERSION}:
        raise RuntimeError(f"Unexpected OpenHands versions: {versions}")

    repo, knowledge, agent_skills = load_skills_from_dir(args.skills_dir)
    merged = {**repo, **knowledge, **agent_skills}
    skill = merged.get(args.skill)
    if skill is None:
        raise RuntimeError(f"Skill {args.skill!r} was not loaded: {sorted(merged)}")
    context = AgentContext(
        skills=list(merged.values()),
        load_public_skills=False,
        load_user_skills=False,
        load_project_skills=False,
    )
    fake_conversation = SimpleNamespace(
        state=SimpleNamespace(
            agent=SimpleNamespace(agent_context=context),
            workspace=SimpleNamespace(working_dir=str(args.workspace)),
            invoked_skills=[],
        )
    )
    observation = InvokeSkillExecutor()(
        InvokeSkillAction(name=args.skill), fake_conversation
    )
    rendered = render_content_with_commands(skill.content, working_dir=args.workspace)
    expected = InvokeSkillExecutor._append_skill_location_footer(
        rendered, skill.source, args.workspace
    )
    if observation.is_error or observation.text != expected:
        raise RuntimeError("invoke_skill output is incomplete or differs from full body")
    if fake_conversation.state.invoked_skills != [args.skill]:
        raise RuntimeError("invoke_skill did not record the selected Skill")

    args.workspace.mkdir(parents=True, exist_ok=True)
    artifact_path = args.workspace / args.artifact
    payload = json.dumps(ARTIFACT, sort_keys=True) + "\n"
    python_code = (
        "from pathlib import Path; "
        f"Path({str(artifact_path)!r}).write_text({payload!r}, encoding='utf-8')"
    )
    terminal = TerminalExecutor(
        working_dir=str(args.workspace), terminal_type="subprocess"
    )
    try:
        terminal_result = terminal(
            TerminalAction(command=f"python -c {shlex.quote(python_code)}", timeout=30)
        )
    finally:
        terminal.close()
    if terminal_result.is_error or terminal_result.exit_code != 0:
        raise RuntimeError(f"Terminal smoke failed: {terminal_result}")

    print(
        json.dumps(
            {
                "versions": versions,
                "skill": args.skill,
                "skill_body_bytes": len(skill.content.encode("utf-8")),
                "skill_body_sha256": _digest(skill.content),
                "invoke_output_bytes": len(observation.text.encode("utf-8")),
                "invoke_output_sha256": _digest(observation.text),
                "invoke_exact_full_body": True,
                "terminal_exit_code": terminal_result.exit_code,
                "artifact": str(artifact_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
