"""Skill-aware KV cache reuse planning utilities.

This package is a lightweight research prototype for the SkillBlend paper
framework. It intentionally avoids importing LMCache or vLLM so the policy can
be tested on existing experiment summaries before being wired into serving code.
"""

from .policy import (
    ReuseMode,
    SkillBlendConfig,
    SkillCachePolicy,
    SkillRequestContext,
    SkillReusePlan,
    SkillSegment,
)
from .prompt_segments import (
    SkillMarkerConfig,
    extract_marked_skill_segments,
    mark_skill_text,
    stable_text_hash,
    stable_token_hash,
)
from .selection import (
    SelectionStrategy,
    kv_deviation_scores,
    select_first_last_tokens,
    select_high_kv_deviation_tokens,
    select_random_tokens,
    token_budget,
)

__all__ = [
    "ReuseMode",
    "SkillBlendConfig",
    "SkillCachePolicy",
    "SkillMarkerConfig",
    "SkillRequestContext",
    "SkillReusePlan",
    "SkillSegment",
    "SelectionStrategy",
    "extract_marked_skill_segments",
    "kv_deviation_scores",
    "mark_skill_text",
    "select_first_last_tokens",
    "select_high_kv_deviation_tokens",
    "select_random_tokens",
    "stable_text_hash",
    "stable_token_hash",
    "token_budget",
]
