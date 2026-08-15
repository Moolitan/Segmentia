from __future__ import annotations

from cskcache import (
    build_context_segment_token_identity,
    parse_context_segment,
    render_context_segment,
)


class ByteTokenizer:
    def encode(self, text: str, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return list(text.encode("utf-8"))


def test_render_and_parse_round_trip_with_tool_resource_suffix() -> None:
    rendered = render_context_segment('internal&"comms', "skill body")
    observation = rendered + "\n--- Skill Resources ---\n  examples/: guide.md"

    parsed = parse_context_segment(observation)

    assert parsed is not None
    assert parsed.skill_name == 'internal&"comms'
    assert parsed.skill_text == "skill body\n"
    assert parsed.trailing_text == "\n--- Skill Resources ---\n  examples/: guide.md"


def test_parser_rejects_missing_or_ambiguous_segment() -> None:
    assert parse_context_segment("plain Skill text") is None
    duplicated = (
        render_context_segment("one", "body")
        + render_context_segment("two", "body")
    )
    assert parse_context_segment(duplicated) is None


def test_token_identity_includes_only_the_tool_boundary_newline() -> None:
    identity = build_context_segment_token_identity(
        ByteTokenizer(),
        "internal-comms",
        "skill body\n",
    )

    assert identity.observation_text == (
        '<context_segment skill_name="internal-comms">\n'
        "skill body\n"
        "</context_segment>"
    )
    assert identity.cache_text == identity.observation_text + "\n"
    assert bytes(identity.token_ids).decode("utf-8") == identity.cache_text
    assert bytes(identity.start_marker_token_ids).decode("utf-8") == (
        '<context_segment skill_name="internal-comms">\n'
    )
