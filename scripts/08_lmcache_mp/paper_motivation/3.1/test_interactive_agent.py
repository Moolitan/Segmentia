from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
MODULE_PATH = SCRIPT_DIR / "interactive_agent.py"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


def load_module():
    module_name = "segmentia_interactive_agent"
    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class FakeLLM:
    def __init__(self) -> None:
        self.calls = []

    def _transport_call(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return "ok"


def test_reuse_prompt_does_not_install_skill_boundaries() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    launcher = (SCRIPT_DIR / "run_interactive_agent.sh").read_text(
        encoding="utf-8"
    )
    prefill_launcher = (SCRIPT_DIR / "run.sh").read_text(encoding="utf-8")

    assert "install_skill_boundary" not in source
    assert "separator-config" not in source
    assert "<|repo_name|>" not in source
    assert "<|repo_name|>" not in launcher
    assert "<|repo_name|>" not in prefill_launcher


def test_injector_sends_the_exact_cached_object_span(monkeypatch, tmp_path) -> None:
    module = load_module()
    skill_tokens = (10, 11, 12)
    skill = module.CachedSkill(
        name="doc-coauthoring",
        skill_path=tmp_path / "SKILL.md",
        cache_id="cache-id",
        text="# Doc coauthoring\nUse the workflow.",
        token_ids=skill_tokens,
        token_sha256=module.sha256_tokens(skill_tokens),
    )
    prompt_ids = [1, 2, *skill_tokens, 20, 21]
    monkeypatch.setattr(
        module,
        "tokenize_openhands_request",
        lambda *_args, **_kwargs: prompt_ids,
    )
    llm = FakeLLM()
    module.attach_skill_kv_injector(
        llm,
        {skill.name: skill},
        "http://127.0.0.1:8014",
        "EMPTY",
        "Qwen3",
    )
    messages = [{"role": "tool", "content": skill.text}]

    result = llm._transport_call(messages=messages, tools=[])

    assert result == "ok"
    assert messages[0]["content"] == skill.text
    request_kwargs = llm.calls[0][1]
    assert request_kwargs["extra_body"]["kv_transfer_params"] == {
        "lmcache_segmentia_lookup": {
            "segment_start": 2,
            "segment_end": 5,
        }
    }
    assert llm._segmentia_skill_events == [
        {
            "skill": skill.name,
            "cache_id": skill.cache_id,
            "segment_start": 2,
            "segment_end": 5,
            "token_count": len(skill_tokens),
        }
    ]


def test_qwen_cache_object_includes_native_tool_response_boundaries() -> None:
    from skill_cache_tokens import qwen_tool_response_token_ids

    class FakeTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert messages[-1] == {
                "role": "tool",
                "name": "skill",
                "content": "skill body\n",
            }
            assert kwargs == {
                "tokenize": False,
                "add_generation_prompt": False,
                "enable_thinking": True,
            }
            return "rendered"

        def encode(self, rendered, add_special_tokens):
            assert rendered == "rendered"
            assert add_special_tokens is False
            return [7, 100, 10, 11, 101, 8]

        def convert_tokens_to_ids(self, token):
            return {"<tool_response>": 100, "</tool_response>": 101}[token]

    assert qwen_tool_response_token_ids(FakeTokenizer(), "skill body\n") == [
        100,
        10,
        11,
        101,
    ]
