"""Export the exact first-request OpenHands system message and tool schemas."""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
from pathlib import Path
from typing import Any


PARENT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PARENT_DIR))
os.environ.setdefault("OPENHANDS_SUPPRESS_BANNER", "1")
from interactive_agent import create_agent  # noqa: E402


def _json_value(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json", exclude_none=True)
    return json.loads(json.dumps(value))


def export_openhands_context(
    *,
    skills_dir: Path,
    workspace: Path,
    served_model: str,
    base_url: str,
    api_key: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build the same Agent as interactive_agent.py and export its first event."""
    from openhands.sdk import Conversation
    from openhands.sdk.event import SystemPromptEvent

    args = argparse.Namespace(
        served_model=served_model,
        base_url=base_url,
        api_key=api_key,
        skills_dir=skills_dir,
    )
    # SDK initialization logs the complete AgentContext (including every Skill
    # body). Suppress that diagnostic dump; the exported prompt itself is saved
    # by the caller in prompt_metadata.json.
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
        io.StringIO()
    ):
        llm, agent = create_agent(args, {})
        conversation = Conversation(
            agent=agent,
            workspace=str(workspace),
            max_iteration_per_run=1,
            stuck_detection=False,
            visualizer=None,
            delete_on_close=False,
        )
        try:
            # Conversation initializes lazily. No user message or LLM call is sent.
            conversation._ensure_agent_ready()
            events = list(conversation.state.events)
            system_events = [
                event for event in events if isinstance(event, SystemPromptEvent)
            ]
            if len(system_events) != 1:
                raise RuntimeError(
                    "expected one OpenHands SystemPromptEvent, "
                    f"found {len(system_events)}"
                )
            event = system_events[0]
            message = llm.format_messages_for_llm([event.to_llm_message()])[0]
            tools = [_json_value(tool.to_openai_tool()) for tool in event.tools]
            return _json_value(message), tools
        finally:
            conversation.close()
