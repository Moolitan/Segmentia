"""BenchFlow CLI entrypoint with the isolated SDK 1.43.1 adapter registered."""

from __future__ import annotations

from benchflow.agents.registry import register_agent


AGENT_NAME = "openhands-sdk-1.43.1"
ADAPTER = "/opt/cskcache/openhands_sdk_1_43_1/acp_server.py"

register_agent(
    name=AGENT_NAME,
    description="OpenHands Agent built directly with SDK/Tools 1.43.1",
    install_cmd=(
        "python -c \"import importlib.metadata as m; "
        "assert m.version('openhands-sdk') == '1.43.1'; "
        "assert m.version('openhands-tools') == '1.43.1'\" "
        f"&& test -r {ADAPTER}"
    ),
    launch_cmd=f"python -u {ADAPTER}",
    protocol="acp",
    requires_env=["LLM_API_KEY"],
    api_protocol="",
    env_mapping={
        "BENCHFLOW_PROVIDER_BASE_URL": "LLM_BASE_URL",
        "BENCHFLOW_PROVIDER_API_KEY": "LLM_API_KEY",
        "BENCHFLOW_PROVIDER_MODEL": "LLM_MODEL",
    },
    skill_paths=["$HOME/.agents/skills", "$WORKSPACE/.agents/skills"],
    supports_acp_set_model=False,
)


if __name__ == "__main__":
    from benchflow.cli.main import app

    app()
