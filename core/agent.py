from __future__ import annotations

import os

from core.request_metrics import attach_llm_request_attempt_collector


def create_agent_and_llm(
    skills_dir: str,
    vllm_port: int = 8000,
):
    """Create the OpenHands LLM and Agent used by benchmark runs."""
    from openhands.sdk import Agent, AgentContext, LLM, LLMSummarizingCondenser, Tool
    from openhands.sdk.context.skills import load_skills_from_dir
    from openhands.tools.apply_patch import ApplyPatchTool
    from openhands.tools.browser_use import BrowserToolSet
    from openhands.tools.gemini import (
        EditTool,
        ListDirectoryTool,
        ReadFileTool,
        WriteFileTool,
    )
    from openhands.tools.glob import GlobTool
    from openhands.tools.grep import GrepTool
    from openhands.tools.skill import SkillTool
    from openhands.tools.task import TaskToolSet
    from openhands.tools.task_tracker import TaskTrackerTool
    from openhands.tools.terminal import TerminalTool

    vllm_api_key = os.environ.get("VLLM_API_KEY", "EMPTY")

    llm = LLM(
        model="openai/Qwen3",
        api_key=vllm_api_key,
        base_url=f"http://localhost:{vllm_port}/v1",
        temperature=0.6,
        top_p=0.95,
        top_k=20,
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
        disable_stop_word=False,
        timeout=720,
    )
    attach_llm_request_attempt_collector(llm, vllm_port=vllm_port)

    tools = [
        Tool(name=TerminalTool.name),
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
    print(
        f"已从 {skills_dir} 加载 {len(agent_skills)} 个 Skills: "
        f"{sorted(agent_skills.keys())}"
    )

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
                "You do not need to execute any code you write. "
                "Your only responsibility is to produce well-structured and "
                "complete code files."
            ),
        ),
        condenser=LLMSummarizingCondenser(
            llm=llm,
            max_size=240,
            keep_first=2,
        ),
    )

    return llm, agent
