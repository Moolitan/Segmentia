#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import textwrap
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[2]
SDK_ROOT = ROOT / "software-agent-sdk"
for extra_path in (
    SDK_ROOT / "openhands-sdk",
    SDK_ROOT / "openhands-tools",
    SDK_ROOT / "openhands-workspace",
):
    sys.path.insert(0, str(extra_path))

os.environ.setdefault("OPENHANDS_SUPPRESS_BANNER", "1")

from openhands.sdk import (  # noqa: E402
    Agent,
    AgentContext,
    Conversation,
    LLM,
    LLMSummarizingCondenser,
    Tool,
    load_project_skills,
    register_agent,
)
from openhands.sdk.event import MessageEvent  # noqa: E402
from openhands.sdk.llm import content_to_str  # noqa: E402
from openhands.tools.apply_patch import ApplyPatchTool  # noqa: E402
from openhands.tools.delegate import DelegationVisualizer  # noqa: E402
from openhands.tools.gemini import (  # noqa: E402
    EditTool,
    ListDirectoryTool,
    ReadFileTool,
    WriteFileTool,
)
from openhands.tools.glob import GlobTool  # noqa: E402
from openhands.tools.grep import GrepTool  # noqa: E402
from openhands.tools.task import TaskToolSet  # noqa: E402
from openhands.tools.task_tracker import TaskTrackerTool  # noqa: E402
from openhands.tools.terminal import TerminalTool  # noqa: E402


DEFAULT_TASK = textwrap.dedent(
    """
    Fix a Django bug:

    JSONField values render incorrectly when the field is readonly in Django
    admin. For example, {"foo": "bar"} is shown as {'foo': 'bar'}, which is not
    valid JSON.

    Suspected direction:
    - inspect django.contrib.admin.utils.display_for_field
    - handle JSONField specially
    - prefer the field's prepare_value behavior instead of a raw dict repr
    - add focused regression tests
    """
).strip()


FINDINGS_FILENAME = "MULTI_AGENT_FINDINGS.md"
PLAN_FILENAME = "MULTI_AGENT_PLAN.md"
FINAL_SUMMARY_FILENAME = "MULTI_AGENT_FINAL_SUMMARY.md"
TASK_FILENAME = "MULTI_AGENT_TASK.md"
DEFAULT_WORKSPACE = ROOT / "workspace" / "multi_agents"
WORKSPACE_SKILLS_REL = Path(".agents") / "skills"

# These specs define sub-agent roles only. The actual skills come from
# workspace/.agents/skills as AgentSkills (SKILL.md) and are loaded via
# load_project_skills(), so they keep the progressive-disclosure behavior.
SUBAGENT_SPECS: dict[str, dict[str, str]] = {
    "jsonfield_explorer": {
        "description": "Explores JSONField implementation details and value formatting.",
        "system_suffix": (
            "Stay in research mode only. Read files, search code, run safe shell "
            "inspection commands, and return a concise technical summary."
        ),
        "task_template": textwrap.dedent(
            """
            Explore the repository for JSONField behavior related to this task.

            User task:
            {task}

            Focus areas:
            - Find JSONField implementation and any prepare_value / serialization code.
            - Identify call sites relevant to Django admin display formatting.
            - Note edge cases that could matter for invalid JSON or special wrapper types.

            Output format:
            1. Relevant files
            2. Key findings
            3. Risks / edge cases
            4. Recommendation for the planner
            """
        ).strip(),
    },
    "admin_display_explorer": {
        "description": "Explores django.contrib.admin.utils.display_for_field and related formatting paths.",
        "system_suffix": (
            "Stay in research mode only. Inspect admin formatting code paths and "
            "report exact files, functions, and likely change points."
        ),
        "task_template": textwrap.dedent(
            """
            Explore the repository for Django admin display formatting behavior.

            User task:
            {task}

            Focus areas:
            - Trace django.contrib.admin.utils.display_for_field.
            - Identify how readonly admin values are rendered.
            - Find the exact code path where dict repr could leak into output.

            Output format:
            1. Relevant files
            2. Key findings
            3. Risks / edge cases
            4. Recommendation for the planner
            """
        ).strip(),
    },
    "readonly_render_explorer": {
        "description": "Explores readonly admin field rendering and regression-test locations.",
        "system_suffix": (
            "Stay in research mode only. Find rendering call chains, test files, "
            "and the minimal regression-test surface."
        ),
        "task_template": textwrap.dedent(
            """
            Explore readonly-field rendering and relevant tests.

            User task:
            {task}

            Focus areas:
            - Find readonly admin rendering code paths.
            - Identify the smallest, most relevant regression-test location.
            - Note any existing JSONField admin tests that can be extended.

            Output format:
            1. Relevant files
            2. Key findings
            3. Risks / edge cases
            4. Recommendation for the planner
            """
        ).strip(),
    },
    "fix_planner": {
        "description": "Turns explorer findings into a concrete software-fix plan.",
        "system_suffix": (
            "Work in planning mode only. Produce an implementation plan that is "
            "specific enough for an execution agent to follow without guesswork."
        ),
        "task_template": "",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OpenHands Claude-style main/sub-agent demo."
    )
    parser.add_argument(
        "--mode",
        choices=("orchestrated", "delegated"),
        default="orchestrated",
        help="Execution style. 'orchestrated' is more deterministic.",
    )
    parser.add_argument(
        "--workspace",
        default=str(DEFAULT_WORKSPACE),
        help="Target repository workspace to operate on.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "results" / "multi_agents"),
        help="Directory for persisted conversations and reports.",
    )
    parser.add_argument(
        "--task",
        default=None,
        help="Task string. If omitted, falls back to --task-file or built-in demo task.",
    )
    parser.add_argument(
        "--task-file",
        default=None,
        help="Optional text/markdown file containing the task prompt.",
    )
    parser.add_argument("--vllm-port", type=int, default=8000)
    parser.add_argument(
        "--base-url",
        default=None,
        help="Override the OpenAI-compatible LLM endpoint. Default: http://localhost:<vllm-port>/v1",
    )
    parser.add_argument("--model", default="openai/Qwen3")
    parser.add_argument("--api-key-env", default="VLLM_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--thinking-budget", type=int, default=200000)
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--timeout", type=int, default=720)
    parser.add_argument("--max-iterations", type=int, default=120)
    parser.add_argument(
        "--system-prompt-filename",
        default="system_prompt.j2",
        help="Agent system prompt template filename.",
    )
    return parser.parse_args()


def read_task(args: argparse.Namespace) -> str:
    if args.task:
        return args.task.strip()
    if args.task_file:
        return Path(args.task_file).read_text(encoding="utf-8").strip()
    return DEFAULT_TASK


def build_llm(args: argparse.Namespace, usage_id: str) -> LLM:
    api_key = os.environ.get(args.api_key_env, "EMPTY")
    base_url = args.base_url or f"http://localhost:{args.vllm_port}/v1"
    return LLM(
        model=args.model,
        api_key=api_key,
        base_url=base_url,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        stream=False,
        native_tool_calling=False,
        caching_prompt=True,
        prompt_cache_retention="24h",
        drop_params=True,
        modify_params=True,
        reasoning_effort=args.reasoning_effort,
        extended_thinking_budget=args.thinking_budget,
        litellm_extra_body={
            "chat_template_kwargs": {"enable_thinking": True},
            "min_p": 0,
        },
        log_completions=False,
        disable_vision=True,
        disable_stop_word=False,
        enable_encrypted_reasoning=True,
        timeout=args.timeout,
        usage_id=usage_id,
    )


def clone_llm(parent_llm: LLM, usage_id: str) -> LLM:
    return parent_llm.model_copy(update={"usage_id": usage_id, "stream": False})


def get_workspace_skills_dir(workspace: Path) -> Path:
    return workspace / WORKSPACE_SKILLS_REL


def load_workspace_agent_skills(workspace: Path) -> list[Any]:
    skills = load_project_skills(str(workspace))
    skill_names = sorted(skill.name for skill in skills)
    print(f"[skills] loaded {len(skill_names)} project skills from {workspace}")
    if skill_names:
        print(f"[skills] available: {skill_names}")
    else:
        print(
            f"[skills] warning: no project skills were discovered under "
            f"{get_workspace_skills_dir(workspace)}"
        )
    return skills


def read_only_tools() -> list[Tool]:
    return [
        Tool(name=TerminalTool.name),
        Tool(name=GlobTool.name),
        Tool(name=GrepTool.name),
        Tool(name=ListDirectoryTool.name),
        Tool(name=ReadFileTool.name),
    ]


def execution_tools(include_task_tool: bool) -> list[Tool]:
    tools = [
        Tool(name=TerminalTool.name),
        Tool(name=GlobTool.name),
        Tool(name=GrepTool.name),
        Tool(name=ApplyPatchTool.name),
        Tool(name=TaskTrackerTool.name),
        Tool(name=ListDirectoryTool.name),
        Tool(name=ReadFileTool.name),
        Tool(name=EditTool.name),
        Tool(name=WriteFileTool.name),
    ]
    if include_task_tool:
        tools.append(Tool(name=TaskToolSet.name))
    return tools


def create_explorer_agent(
    llm: LLM,
    *,
    agent_name: str,
    system_suffix: str,
    project_skills: list[Any],
    system_prompt_filename: str,
) -> Agent:
    return Agent(
        llm=llm,
        tools=read_only_tools(),
        include_default_tools=["FinishTool", "ThinkTool"],
        tool_concurrency_limit=3,
        system_prompt_filename=system_prompt_filename,
        agent_context=AgentContext(
            skills=project_skills,
            system_message_suffix=(
                f"You are the specialized sub-agent '{agent_name}'. {system_suffix} "
                "Use the available project AgentSkills proactively when relevant. "
                "Those skills live in the workspace and should be consulted as "
                "knowledge guides, not treated as subagents."
            ),
        ),
        condenser=LLMSummarizingCondenser(llm=llm, max_size=160, keep_first=2),
    )


def create_planner_agent(
    llm: LLM,
    *,
    project_skills: list[Any],
    system_prompt_filename: str,
) -> Agent:
    spec = SUBAGENT_SPECS["fix_planner"]
    return Agent(
        llm=llm,
        tools=read_only_tools(),
        include_default_tools=["FinishTool", "ThinkTool"],
        tool_concurrency_limit=2,
        system_prompt_filename=system_prompt_filename,
        agent_context=AgentContext(
            skills=project_skills,
            system_message_suffix=(
                f"{spec['system_suffix']} Use the available project AgentSkills "
                "proactively when relevant, especially before committing to an "
                "implementation plan."
            ),
        ),
        condenser=LLMSummarizingCondenser(llm=llm, max_size=200, keep_first=2),
    )


def create_execution_agent(
    llm: LLM,
    *,
    include_task_tool: bool,
    project_skills: list[Any],
    system_prompt_filename: str,
) -> Agent:
    return Agent(
        llm=llm,
        tools=execution_tools(include_task_tool=include_task_tool),
        include_default_tools=["FinishTool", "ThinkTool"],
        tool_concurrency_limit=4,
        system_prompt_filename=system_prompt_filename,
        agent_context=AgentContext(
            skills=project_skills,
            system_message_suffix=(
                "You are the main software-fix agent. Be surgical, keep diffs "
                "small, run focused validation, and use the task tracker tool as "
                "your to-do list before editing. Use the available project "
                "AgentSkills proactively when relevant; they are workspace "
                "knowledge guides, not subagent types."
            )
        ),
        condenser=LLMSummarizingCondenser(llm=llm, max_size=240, keep_first=2),
    )


def extract_last_assistant_message(conversation: Conversation) -> str:
    for event in reversed(conversation.state.events):
        if isinstance(event, MessageEvent) and event.llm_message.role == "assistant":
            return "".join(content_to_str(event.llm_message.content)).strip()
    return ""


def run_single_conversation(
    *,
    agent: Agent,
    prompt: str,
    workspace: Path,
    persistence_dir: Path,
    max_iterations: int,
    visualizer: Any | None = None,
) -> dict[str, Any]:
    persistence_dir.mkdir(parents=True, exist_ok=True)
    conversation = Conversation(
        agent=agent,
        workspace=str(workspace),
        persistence_dir=str(persistence_dir),
        max_iteration_per_run=max_iterations,
        stuck_detection=True,
        visualizer=visualizer,
    )
    try:
        conversation.send_message(prompt)
        conversation.run()
        combined_metrics = conversation.conversation_stats.get_combined_metrics()
        cost = combined_metrics.accumulated_cost
        return {
            "reply": extract_last_assistant_message(conversation),
            "accumulated_cost": float(cost) if cost is not None else None,
            "total_events": len(conversation.state.events),
            "persistence_dir": conversation.state.persistence_dir,
        }
    finally:
        conversation.close()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def build_findings_markdown(
    task: str,
    explorer_results: dict[str, dict[str, Any]],
) -> str:
    sections = [f"# Explorer Findings\n\n## User Task\n\n{task}\n"]
    for name in (
        "jsonfield_explorer",
        "admin_display_explorer",
        "readonly_render_explorer",
    ):
        result = explorer_results.get(name)
        if result is None:
            continue
        sections.append(f"## {name}\n\n{result['reply'].strip()}\n")
    return "\n".join(sections).strip()


def build_planner_prompt(task: str, findings_markdown: str) -> str:
    return textwrap.dedent(
        f"""
        Create a concrete implementation plan for this software task.

        User task:
        {task}

        Explorer findings:
        {findings_markdown}

        Requirements for the plan:
        1. Identify the exact files and functions that should change.
        2. Explain the minimal code change needed.
        3. List regression tests to add or update.
        4. Call out edge cases and risks.
        5. Give a short validation checklist.

        Output in markdown with these sections:
        - Summary
        - Proposed Code Changes
        - Test Plan
        - Risks
        - Execution Checklist
        """
    ).strip()


def build_execution_prompt(
    *,
    findings_path: Path,
    plan_path: Path,
    task: str,
) -> str:
    return textwrap.dedent(
        f"""
        Implement this repository task in the current workspace:

        {task}

        Before editing, read these files:
        - {findings_path}
        - {plan_path}

        Required workflow:
        1. Read the findings and plan files first.
        2. Use the task tracker tool to create and maintain a todo list that mirrors
           the execution checklist from the plan.
        3. Make the smallest correct code change.
        4. Add or update focused regression tests.
        5. Run targeted validation commands.
        6. Finish with a concise summary that includes changed files, commands run,
           and whether validation passed.

        Do not ask clarifying questions unless you are blocked by missing context.
        """
    ).strip()


def build_delegated_prompt(task: str) -> str:
    return textwrap.dedent(
        f"""
        You are orchestrating a Claude-Code-style multi-agent software fix.

        User task:
        {task}

        Required workflow:
        1. Use the task tool to launch these three research-only subagents:
           - jsonfield_explorer
           - admin_display_explorer
           - readonly_render_explorer
           Launch them in parallel if the model can do so.
        2. After they finish, use the task tool to launch fix_planner with a prompt
           that includes the explorer findings.
        3. Write the planner result to {PLAN_FILENAME}.
        4. Use the task tracker tool as a todo list.
        5. Implement the fix locally with your own tools.
        6. Add or update focused tests.
        7. Run targeted validation commands.
        8. Finish with a concise execution summary.

        Explorer agents are research-only. You, the main agent, are responsible for
        all code edits and validation.
        """
    ).strip()


def register_specialized_agents(
    args: argparse.Namespace,
    *,
    project_skills: list[Any],
) -> None:
    def make_factory(subagent_name: str) -> Callable[[LLM], Agent]:
        spec = SUBAGENT_SPECS[subagent_name]

        def factory(parent_llm: LLM) -> Agent:
            llm = clone_llm(parent_llm, usage_id=f"subagent:{subagent_name}")
            if subagent_name == "fix_planner":
                return create_planner_agent(
                    llm,
                    project_skills=project_skills,
                    system_prompt_filename=args.system_prompt_filename,
                )
            return create_explorer_agent(
                llm,
                agent_name=subagent_name,
                system_suffix=spec["system_suffix"],
                project_skills=project_skills,
                system_prompt_filename=args.system_prompt_filename,
            )

        return factory

    for subagent_name, spec in SUBAGENT_SPECS.items():
        try:
            register_agent(
                name=subagent_name,
                factory_func=make_factory(subagent_name),
                description=spec["description"],
            )
        except ValueError:
            pass


def run_orchestrated_mode(
    *,
    args: argparse.Namespace,
    task: str,
    workspace: Path,
    output_dir: Path,
    project_skills: list[Any],
) -> dict[str, Any]:
    print("[phase 1/3] running explorer sub-agents in parallel...")
    explorer_results: dict[str, dict[str, Any]] = {}

    def run_explorer(subagent_name: str) -> tuple[str, dict[str, Any]]:
        spec = SUBAGENT_SPECS[subagent_name]
        llm = build_llm(args, usage_id=f"explorer:{subagent_name}")
        agent = create_explorer_agent(
            llm,
            agent_name=subagent_name,
            system_suffix=spec["system_suffix"],
            project_skills=project_skills,
            system_prompt_filename=args.system_prompt_filename,
        )
        result = run_single_conversation(
            agent=agent,
            prompt=spec["task_template"].format(task=task),
            workspace=workspace,
            persistence_dir=output_dir / "conversations" / subagent_name,
            max_iterations=args.max_iterations,
        )
        return subagent_name, result

    explorer_names = [
        "jsonfield_explorer",
        "admin_display_explorer",
        "readonly_render_explorer",
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(run_explorer, name) for name in explorer_names]
        for future in concurrent.futures.as_completed(futures):
            name, result = future.result()
            explorer_results[name] = result
            print(f"  - {name} finished")

    findings_markdown = build_findings_markdown(task, explorer_results)
    findings_path = workspace / FINDINGS_FILENAME
    write_text(findings_path, findings_markdown)

    print("[phase 2/3] running planner sub-agent...")
    planner_llm = build_llm(args, usage_id="planner:fix_planner")
    planner_agent = create_planner_agent(
        planner_llm,
        project_skills=project_skills,
        system_prompt_filename=args.system_prompt_filename,
    )
    planner_result = run_single_conversation(
        agent=planner_agent,
        prompt=build_planner_prompt(task, findings_markdown),
        workspace=workspace,
        persistence_dir=output_dir / "conversations" / "fix_planner",
        max_iterations=args.max_iterations,
    )
    plan_path = workspace / PLAN_FILENAME
    write_text(plan_path, planner_result["reply"])

    print("[phase 3/3] running main execution agent...")
    executor_llm = build_llm(args, usage_id="main:executor")
    executor_agent = create_execution_agent(
        executor_llm,
        include_task_tool=False,
        project_skills=project_skills,
        system_prompt_filename=args.system_prompt_filename,
    )
    execution_result = run_single_conversation(
        agent=executor_agent,
        prompt=build_execution_prompt(
            findings_path=findings_path,
            plan_path=plan_path,
            task=task,
        ),
        workspace=workspace,
        persistence_dir=output_dir / "conversations" / "main_executor",
        max_iterations=args.max_iterations,
    )

    summary_path = workspace / FINAL_SUMMARY_FILENAME
    write_text(summary_path, execution_result["reply"])

    return {
        "mode": "orchestrated",
        "task": task,
        "workspace": str(workspace),
        "workspace_skills_dir": str(get_workspace_skills_dir(workspace)),
        "findings_file": str(findings_path),
        "plan_file": str(plan_path),
        "summary_file": str(summary_path),
        "loaded_skill_names": sorted(skill.name for skill in project_skills),
        "explorers": explorer_results,
        "planner": planner_result,
        "executor": execution_result,
    }


def run_delegated_mode(
    *,
    args: argparse.Namespace,
    task: str,
    workspace: Path,
    output_dir: Path,
    project_skills: list[Any],
) -> dict[str, Any]:
    print("[delegated mode] registering specialized sub-agents...")
    register_specialized_agents(args, project_skills=project_skills)

    task_path = workspace / TASK_FILENAME
    write_text(task_path, task)

    print("[delegated mode] running main agent with TaskToolSet...")
    llm = build_llm(args, usage_id="main:delegated")
    main_agent = create_execution_agent(
        llm,
        include_task_tool=True,
        project_skills=project_skills,
        system_prompt_filename=args.system_prompt_filename,
    )
    result = run_single_conversation(
        agent=main_agent,
        prompt=build_delegated_prompt(task),
        workspace=workspace,
        persistence_dir=output_dir / "conversations" / "delegated_main",
        max_iterations=args.max_iterations,
        visualizer=DelegationVisualizer(name="MainAgent"),
    )
    summary_path = workspace / FINAL_SUMMARY_FILENAME
    write_text(summary_path, result["reply"])
    return {
        "mode": "delegated",
        "task": task,
        "workspace": str(workspace),
        "workspace_skills_dir": str(get_workspace_skills_dir(workspace)),
        "task_file": str(task_path),
        "summary_file": str(summary_path),
        "loaded_skill_names": sorted(skill.name for skill in project_skills),
        "main_agent": result,
    }


def main() -> None:
    args = parse_args()
    workspace = Path(args.workspace).resolve()
    output_dir = Path(args.output_dir).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    task = read_task(args)
    print(f"mode: {args.mode}")
    print(f"workspace: {workspace}")
    print(f"output_dir: {output_dir}")
    print(f"model: {args.model}")
    print(f"base_url: {args.base_url or f'http://localhost:{args.vllm_port}/v1'}")
    print(f"task preview: {task[:120]}{'...' if len(task) > 120 else ''}")

    project_skills = load_workspace_agent_skills(workspace)

    if args.mode == "orchestrated":
        result = run_orchestrated_mode(
            args=args,
            task=task,
            workspace=workspace,
            output_dir=output_dir,
            project_skills=project_skills,
        )
    else:
        result = run_delegated_mode(
            args=args,
            task=task,
            workspace=workspace,
            output_dir=output_dir,
            project_skills=project_skills,
        )

    result_path = output_dir / f"{args.mode}_result.json"
    write_text(result_path, json.dumps(result, indent=2, ensure_ascii=False))

    print("\nrun complete")
    print(f"result json: {result_path}")
    if "plan_file" in result:
        print(f"plan file: {result['plan_file']}")
    print(f"summary file: {result['summary_file']}")


if __name__ == "__main__":
    main()
