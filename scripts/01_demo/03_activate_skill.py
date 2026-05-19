import json
import os
import time

from pydantic import SecretStr

from openhands.sdk import (
    LLM,
    Agent,
    AgentContext,
    Conversation,
    Event,
    LLMConvertibleEvent,
    get_logger,
)
from openhands.sdk.context import (
    KeywordTrigger,
    Skill,
)
from openhands.sdk.tool import Tool
from openhands.tools.file_editor import FileEditorTool
from openhands.tools.terminal import TerminalTool


logger = get_logger(__name__)

# Configure LLM
# scripts/vllm_start.sh serves the local checkpoint as Qwen3:
# /mnt/Large_Language_Model_Lab_1/llm_models/Qwen3-14B/Qwen/Qwen3-14B
vllm_port = os.getenv("VLLM_PORT", "8000")
api_key = os.getenv("LLM_API_KEY") or os.getenv("VLLM_API_KEY", "EMPTY")
model = os.getenv("LLM_MODEL", "hosted_vllm/Qwen3")
base_url = os.getenv("LLM_BASE_URL", f"http://localhost:{vllm_port}/v1")
llm = LLM(
    usage_id="agent",
    model=model,
    base_url=base_url,
    api_key=SecretStr(api_key),
    log_completions=True,
    log_completions_folder=os.path.join(os.getcwd(), "log", "03_activate_skill"),
)

prompt_log_path = os.path.join(os.getcwd(), "log", "03_activate_skill_prompts.jsonl")
pretty_prompt_log_path = os.path.join(
    os.getcwd(), "log", "03_activate_skill_prompts.pretty.json"
)
os.makedirs(os.path.dirname(prompt_log_path), exist_ok=True)
_original_transport_call = llm._transport_call
_request_counter = {"value": 0}
_request_records = []


def _dump_vllm_request(*, messages, enable_streaming=False, on_token=None, **kwargs):
    """Capture the exact SDK -> LiteLLM request payload before it is sent."""
    _request_counter["value"] += 1
    request_record = {
        "request_index": _request_counter["value"],
        "timestamp": time.time(),
        "model": llm.model,
        "base_url": llm.base_url,
        "enable_streaming": enable_streaming,
        "messages": messages,
        "kwargs": kwargs,
    }
    with open(prompt_log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(request_record, ensure_ascii=False, default=str) + "\n")
    _request_records.append(request_record)
    with open(pretty_prompt_log_path, "w", encoding="utf-8") as f:
        json.dump(_request_records, f, ensure_ascii=False, indent=2, default=str)

    print("=" * 100)
    print(f"VLLM REQUEST {request_record['request_index']} -> {llm.base_url}")
    print(json.dumps(request_record, ensure_ascii=False, indent=2, default=str))

    return _original_transport_call(
        messages=messages,
        enable_streaming=enable_streaming,
        on_token=on_token,
        **kwargs,
    )


llm._transport_call = _dump_vllm_request

# Tools
cwd = os.getcwd()
tools = [
    Tool(
        name=TerminalTool.name,
        params={"terminal_type": "subprocess"},
    ),
    Tool(name=FileEditorTool.name),
]

# AgentContext provides flexible ways to customize prompts:
# 1. Skills: Inject instructions (always-active or keyword-triggered)
# 2. system_message_suffix: Append text to the system prompt
# 3. user_message_suffix: Append text to each user message
#
# For complete control over the system prompt, you can also use Agent's
# system_prompt_filename parameter to provide a custom Jinja2 template:
#
#   agent = Agent(
#       llm=llm,
#       tools=tools,
#       system_prompt_filename="/path/to/custom_prompt.j2",
#       system_prompt_kwargs={"cli_mode": True, "repo": "my-project"},
#   )
#
# See: https://docs.openhands.dev/sdk/guides/skill#customizing-system-prompts
agent_context = AgentContext(
    skills=[
        Skill(
            name="repo.md",
            content="When you see this message, you should reply like "
            "you are a grumpy cat forced to use the internet.",
            # source is optional - identifies where the skill came from
            # You can set it to be the path of a file that contains the skill content
            source=None,
            # trigger determines when the skill is active
            # trigger=None means always active (repo skill)
            trigger=None,
        ),
        Skill(
            name="flarglebargle",
            content=(
                'IMPORTANT! The user has said the magic word "flarglebargle". '
                "You must only respond with a message telling them how smart they are"
            ),
            source=None,
            # KeywordTrigger = activated when keywords appear in user messages
            trigger=KeywordTrigger(keywords=["flarglebargle"]),
        ),
    ],
    # system_message_suffix is appended to the system prompt (always active)
    system_message_suffix="Always finish your response with the word 'yay!'",
    # user_message_suffix is appended to each user message
    user_message_suffix="The first character of your response should be 'I'",
    # You can also enable automatic load skills from
    # public registry at https://github.com/OpenHands/extensions
    load_public_skills=True,
)

# Agent
agent = Agent(llm=llm, tools=tools, agent_context=agent_context)

llm_messages = []  # collect raw LLM messages


def conversation_callback(event: Event):
    if isinstance(event, LLMConvertibleEvent):
        llm_messages.append(event.to_llm_message())


conversation = Conversation(
    agent=agent, callbacks=[conversation_callback], workspace=cwd
)

print("=" * 100)
print("Checking if the repo skill is activated.")
conversation.send_message("Hey are you a grumpy cat?")
conversation.run()

print("=" * 100)
print("Now sending flarglebargle to trigger the knowledge skill!")
conversation.send_message("flarglebargle!")
conversation.run()

print("=" * 100)
print("Now triggering public skill 'github'")
conversation.send_message(
    "About GitHub - tell me what additional info I've just provided?"
)
conversation.run()

print("=" * 100)
print("Conversation finished. Got the following LLM messages:")
for i, message in enumerate(llm_messages):
    print(f"Message {i}: {str(message)[:200]}")

# Report cost
cost = llm.metrics.accumulated_cost
print(f"EXAMPLE_COST: {cost}")
