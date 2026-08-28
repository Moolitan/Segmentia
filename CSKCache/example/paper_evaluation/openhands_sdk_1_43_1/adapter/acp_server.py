"""ACP adapter for an OpenHands SDK 1.43.1 local Agent.

This module deliberately depends on ``openhands-sdk`` and ``openhands-tools``
only.  It does not import or install OpenHands CLI.  BenchFlow launches this
process over stdio and remains responsible for trajectory capture, usage
accounting, sandbox lifecycle, and verifier execution.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import uuid
from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ACP owns stdout. These must be set before importing OpenHands/LiteLLM so
# import-time informational output cannot corrupt the JSON-RPC transport.
os.environ.setdefault("OPENHANDS_SUPPRESS_BANNER", "1")
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

from acp import (
    Agent as ACPAgent,
    Client,
    InitializeResponse,
    NewSessionResponse,
    PromptResponse,
    RequestError,
    start_tool_call,
    stdio_streams,
    text_block,
    tool_content,
)
from acp.core import AgentSideConnection
from acp.schema import (
    AgentCapabilities,
    AgentMessageChunk,
    AgentThoughtChunk,
    AuthenticateResponse,
    CloseSessionResponse,
    ForkSessionResponse,
    Implementation,
    ListSessionsResponse,
    LoadSessionResponse,
    PromptCapabilities,
    ResumeSessionResponse,
    SetSessionConfigOptionResponse,
    SetSessionModeResponse,
    SetSessionModelResponse,
    TextContentBlock,
    ToolCallProgress,
    Usage,
)
from openhands.sdk import (
    Agent,
    AgentContext,
    Conversation,
    LLM,
    Message,
    TextContent,
    Tool,
)
from openhands.sdk.event import (
    ActionEvent,
    AgentErrorEvent,
    ConversationStateUpdateEvent,
    Event,
    MessageEvent,
    ObservationEvent,
    SystemPromptEvent,
    UserRejectObservation,
)
from openhands.sdk.skills import load_skills_from_dir
from openhands.sdk.tool.builtins.finish import FinishAction, FinishObservation
from openhands.sdk.tool.builtins.think import ThinkAction, ThinkObservation
from openhands.tools.file_editor import FileEditorAction, FileEditorTool
from openhands.tools.task_tracker import TaskTrackerTool
from openhands.tools.terminal import TerminalAction, TerminalTool
from pydantic import SecretStr


SDK_VERSION = "1.43.1"
SKILLS_DIR_ENV = "CSKCACHE_SKILLS_DIR"
DEFAULT_SKILLS_DIR = "/skills"
MAX_ITERATIONS_ENV = "CSKCACHE_MAX_ITERATIONS"

logging.basicConfig(
    level=os.getenv("CSKCACHE_ADAPTER_LOG_LEVEL", "INFO"),
    format="[openhands-sdk-adapter] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def _plain(event: Event) -> str:
    return str(event.visualize.plain)


def _tool_kind(event: ActionEvent) -> str:
    action = event.action
    if isinstance(action, FileEditorAction):
        return "read" if action.command == "view" else "edit"
    if isinstance(action, TerminalAction):
        return "execute"
    if event.tool_name == "think":
        return "think"
    return "other"


def _tool_title(event: ActionEvent) -> str:
    action = event.action
    if event.tool_name in {"invoke_skill", "activate_skill"}:
        return event.tool_name
    if isinstance(action, TerminalAction):
        command = action.command.strip().replace("\n", " ")
        return f"$ {command}" if command else "terminal"
    if isinstance(action, FileEditorAction):
        operation = "Read" if action.command == "view" else "Edit"
        return f"{operation} {action.path or ''}".strip()
    summary = str(event.summary).strip() if event.summary else ""
    return summary or event.tool_name


def _content(text: str) -> list[Any] | None:
    if not text.strip():
        return None
    return [tool_content(block=text_block(text=text))]


def _usage(conversation: Conversation) -> Usage | None:
    stats = conversation.conversation_stats
    metrics = stats.get_combined_metrics() if stats else None
    token_usage = metrics.accumulated_token_usage if metrics else None
    if token_usage is None:
        return None
    prompt = int(token_usage.prompt_tokens or 0)
    completion = int(token_usage.completion_tokens or 0)
    return Usage(
        input_tokens=prompt,
        output_tokens=completion,
        total_tokens=prompt + completion,
        cached_read_tokens=int(token_usage.cache_read_tokens or 0),
        cached_write_tokens=int(token_usage.cache_write_tokens or 0),
        thought_tokens=int(token_usage.reasoning_tokens or 0),
    )


class EventBridge:
    """Translate SDK events to ordered ACP session updates."""

    def __init__(self, session_id: str, connection: Client, loop: asyncio.AbstractEventLoop):
        self.session_id = session_id
        self.connection = connection
        self.loop = loop
        self.conversation: Conversation | None = None
        self._pending: list[Future[Any]] = []
        self._lock = threading.Lock()

    def callback(self, event: Event) -> None:
        future = asyncio.run_coroutine_threadsafe(self._handle(event), self.loop)
        with self._lock:
            self._pending.append(future)

    async def drain(self) -> None:
        while True:
            with self._lock:
                pending, self._pending = self._pending, []
            if not pending:
                return
            for future in pending:
                await asyncio.wrap_future(future)

    async def _update(self, update: Any) -> None:
        await self.connection.session_update(
            session_id=self.session_id,
            update=update,
        )

    async def _handle(self, event: Event) -> None:
        try:
            if isinstance(event, ConversationStateUpdateEvent):
                return
            if isinstance(event, ActionEvent):
                await self._handle_action(event)
                return
            if isinstance(event, (AgentErrorEvent, UserRejectObservation)):
                await self._update(
                    ToolCallProgress(
                        session_update="tool_call_update",
                        tool_call_id=event.tool_call_id,
                        status="failed",
                        content=_content(_plain(event)),
                        raw_output=event.model_dump(mode="json"),
                    )
                )
                return
            if isinstance(event, ObservationEvent):
                if isinstance(event.observation, (ThinkObservation, FinishObservation)):
                    return
                await self._update(
                    ToolCallProgress(
                        session_update="tool_call_update",
                        tool_call_id=event.tool_call_id,
                        status="completed",
                        content=_content(_plain(event)),
                        raw_output=event.model_dump(mode="json"),
                    )
                )
                return
            if isinstance(event, MessageEvent):
                if event.llm_message.role != "user":
                    text = _plain(event)
                    if text.strip():
                        await self._update(
                            AgentMessageChunk(
                                session_update="agent_message_chunk",
                                content=TextContentBlock(type="text", text=text),
                            )
                        )
                return
            if isinstance(event, SystemPromptEvent):
                text = _plain(event)
                if text.strip():
                    await self._update(
                        AgentThoughtChunk(
                            session_update="agent_thought_chunk",
                            content=TextContentBlock(type="text", text=text),
                        )
                    )
        except Exception:
            logger.exception("Failed to translate SDK event %s", type(event).__name__)
            raise

    async def _handle_action(self, event: ActionEvent) -> None:
        if event.reasoning_content and event.reasoning_content.strip():
            await self._update(
                AgentThoughtChunk(
                    session_update="agent_thought_chunk",
                    content=TextContentBlock(
                        type="text", text=event.reasoning_content.strip() + "\n"
                    ),
                )
            )
        thought = " ".join(item.text for item in event.thought).strip()
        if thought:
            await self._update(
                AgentThoughtChunk(
                    session_update="agent_thought_chunk",
                    content=TextContentBlock(type="text", text=thought + "\n"),
                )
            )
        if isinstance(event.action, ThinkAction):
            text = _plain(event)
            if text.strip():
                await self._update(
                    AgentThoughtChunk(
                        session_update="agent_thought_chunk",
                        content=TextContentBlock(type="text", text=text),
                    )
                )
            return
        if isinstance(event.action, FinishAction):
            text = _plain(event)
            if text.strip():
                await self._update(
                    AgentMessageChunk(
                        session_update="agent_message_chunk",
                        content=TextContentBlock(type="text", text=text),
                    )
                )
            return
        await self._update(
            start_tool_call(
                tool_call_id=event.tool_call_id,
                title=_tool_title(event),
                kind=_tool_kind(event),
                status="in_progress",
                content=_content(_plain(event)),
                raw_input=(
                    event.action.model_dump(mode="json") if event.action else None
                ),
            )
        )


@dataclass
class SessionState:
    conversation: Conversation
    bridge: EventBridge
    running_task: asyncio.Task[Any] | None = field(default=None)


def _model_name() -> str:
    model = os.environ.get("LLM_MODEL") or os.environ.get(
        "BENCHFLOW_PROVIDER_MODEL", ""
    )
    if not model:
        raise RuntimeError("LLM_MODEL or BENCHFLOW_PROVIDER_MODEL is required")
    if "/" not in model and os.environ.get("LLM_BASE_URL"):
        model = f"openai/{model}"
    return model


def _load_task_skills() -> list[Any]:
    skills_dir = Path(os.environ.get(SKILLS_DIR_ENV, DEFAULT_SKILLS_DIR))
    if not skills_dir.is_dir():
        raise FileNotFoundError(f"Skills directory does not exist: {skills_dir}")
    repo, knowledge, agent_skills = load_skills_from_dir(skills_dir)
    merged: dict[str, Any] = {}
    for category in (repo, knowledge, agent_skills):
        for name, skill in category.items():
            if name in merged:
                raise ValueError(f"Duplicate loaded Skill name: {name}")
            merged[name] = skill
    if not merged:
        raise RuntimeError(f"No Skills loaded from {skills_dir}")
    return list(merged.values())


class OpenHandsSDKAgent(ACPAgent):
    def __init__(self, connection: Client):
        self.connection = connection
        self.sessions: dict[str, SessionState] = {}

    def on_connect(self, conn: Client) -> None:
        self.connection = conn

    async def initialize(
        self,
        protocol_version: int,
        client_capabilities: Any | None = None,
        client_info: Any | None = None,
        **_: Any,
    ) -> InitializeResponse:
        return InitializeResponse(
            protocol_version=protocol_version,
            auth_methods=[],
            agent_capabilities=AgentCapabilities(
                load_session=False,
                prompt_capabilities=PromptCapabilities(
                    audio=False,
                    embedded_context=False,
                    image=False,
                ),
            ),
            agent_info=Implementation(
                name="openhands-sdk-benchflow-adapter",
                title="OpenHands SDK BenchFlow Adapter",
                version=SDK_VERSION,
            ),
        )

    async def new_session(
        self,
        cwd: str,
        additional_directories: list[str] | None = None,
        mcp_servers: list[Any] | None = None,
        **_: Any,
    ) -> NewSessionResponse:
        if additional_directories:
            raise RequestError.invalid_params(
                {"reason": "additional_directories are not supported"}
            )
        if mcp_servers:
            raise RequestError.invalid_params({"reason": "MCP is not enabled"})
        workspace = Path(cwd).resolve()
        workspace.mkdir(parents=True, exist_ok=True)
        session_id = str(uuid.uuid4())
        loop = asyncio.get_running_loop()
        bridge = EventBridge(session_id, self.connection, loop)
        api_key = os.environ.get("LLM_API_KEY") or os.environ.get(
            "BENCHFLOW_PROVIDER_API_KEY"
        )
        if not api_key:
            raise RequestError.internal_error({"reason": "LLM_API_KEY is missing"})
        llm = LLM(
            usage_id="benchflow",
            model=_model_name(),
            api_key=SecretStr(api_key),
            base_url=os.environ.get("LLM_BASE_URL")
            or os.environ.get("BENCHFLOW_PROVIDER_BASE_URL"),
        )
        context = AgentContext(
            skills=_load_task_skills(),
            load_public_skills=False,
            load_user_skills=False,
            load_project_skills=False,
            load_memory=False,
        )
        agent = Agent(
            llm=llm,
            tools=[
                Tool(name=TerminalTool.name),
                Tool(name=FileEditorTool.name),
                Tool(name=TaskTrackerTool.name),
            ],
            agent_context=context,
        )
        conversation = Conversation(
            agent=agent,
            workspace=workspace,
            callbacks=[bridge.callback],
            max_iteration_per_run=int(os.getenv(MAX_ITERATIONS_ENV, "500")),
            visualizer=None,
        )
        bridge.conversation = conversation
        self.sessions[session_id] = SessionState(conversation, bridge)
        logger.info(
            "Created session %s in %s with %d task Skills",
            session_id,
            workspace,
            len(context.skills),
        )
        return NewSessionResponse(session_id=session_id)

    async def prompt(
        self,
        prompt: list[Any],
        session_id: str,
        message_id: str | None = None,
        **_: Any,
    ) -> PromptResponse:
        state = self.sessions.get(session_id)
        if state is None:
            raise RequestError.invalid_params({"reason": "Unknown session"})
        unsupported = [block for block in prompt if not isinstance(block, TextContentBlock)]
        if unsupported:
            raise RequestError.invalid_params(
                {"reason": "This benchmark adapter accepts text prompt blocks only"}
            )
        text = "\n".join(block.text for block in prompt).strip()
        if not text:
            return PromptResponse(
                stop_reason="end_turn",
                usage=_usage(state.conversation),
                user_message_id=message_id,
            )
        state.conversation.send_message(
            Message(role="user", content=[TextContent(text=text)])
        )
        run_task = asyncio.create_task(asyncio.to_thread(state.conversation.run))
        state.running_task = run_task
        try:
            await run_task
            await state.bridge.drain()
        except Exception as exc:
            logger.exception("SDK conversation failed")
            await state.bridge.drain()
            await self.connection.session_update(
                session_id=session_id,
                update=AgentMessageChunk(
                    session_update="agent_message_chunk",
                    content=TextContentBlock(type="text", text=f"Error: {exc}"),
                ),
            )
            raise RequestError.internal_error(
                {"reason": "OpenHands SDK conversation failed", "details": str(exc)}
            ) from exc
        finally:
            state.running_task = None
        return PromptResponse(
            stop_reason="end_turn",
            usage=_usage(state.conversation),
            user_message_id=message_id,
        )

    async def cancel(self, session_id: str, **_: Any) -> None:
        state = self.sessions.get(session_id)
        if state is None:
            return
        state.conversation.pause()
        if state.running_task and not state.running_task.done():
            try:
                await asyncio.wait_for(state.running_task, timeout=10)
            except TimeoutError:
                state.running_task.cancel()

    async def close_session(
        self, session_id: str, **_: Any
    ) -> CloseSessionResponse | None:
        state = self.sessions.pop(session_id, None)
        if state is not None:
            state.conversation.close()
        return CloseSessionResponse()

    async def authenticate(
        self, method_id: str, **_: Any
    ) -> AuthenticateResponse | None:
        raise RequestError.method_not_found("authenticate")

    async def list_sessions(self, **_: Any) -> ListSessionsResponse:
        return ListSessionsResponse(sessions=[])

    async def load_session(self, **_: Any) -> LoadSessionResponse | None:
        raise RequestError.method_not_found("session/load")

    async def fork_session(self, **_: Any) -> ForkSessionResponse:
        raise RequestError.method_not_found("session/fork")

    async def resume_session(self, **_: Any) -> ResumeSessionResponse:
        raise RequestError.method_not_found("session/resume")

    async def set_session_mode(self, **_: Any) -> SetSessionModeResponse | None:
        return SetSessionModeResponse()

    async def set_session_model(self, **_: Any) -> SetSessionModelResponse | None:
        return SetSessionModelResponse()

    async def set_config_option(
        self, **_: Any
    ) -> SetSessionConfigOptionResponse | None:
        return SetSessionConfigOptionResponse(config_options=[])

    async def ext_method(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        raise RequestError.method_not_found(method)

    async def ext_notification(self, method: str, params: dict[str, Any]) -> None:
        logger.debug("Ignoring extension notification %s", method)


async def main() -> None:
    reader, writer = await stdio_streams()

    def create_agent(connection: Client) -> OpenHandsSDKAgent:
        return OpenHandsSDKAgent(connection)

    AgentSideConnection(create_agent, writer, reader)
    await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())
