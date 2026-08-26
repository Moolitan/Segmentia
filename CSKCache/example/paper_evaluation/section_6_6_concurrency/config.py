"""Fixed concurrent-arrival matrix; edit this file, not the launcher."""

from paper_evaluation.config import ACTIVE_PLATFORMS, ROOT, TASK_PROMPT_ROOT
from common.driver import SystemVariant


PLATFORM_IDS = ACTIVE_PLATFORMS
SKILL_NAME = "paper-writing"
SKILL_PATH = (
    ROOT / "skills/Auto-claude-code-research-in-sleep/skills/paper-writing/SKILL.md"
)
TASK_PROMPT_PATH = TASK_PROMPT_ROOT / "paper-writing-segmentia-paper-plan.txt"
SYSTEMS = (
    SystemVariant("Full", "full"),
    SystemVariant("CacheBlend-15%", "cacheblend", cacheblend_ratio=0.15),
    SystemVariant(
        "CSKCache", "cskcache", correction_strategy="fixed_prefix",
        calibration_tokens=256,
    ),
)
CONCURRENCIES = (1, 2, 4, 8)
CHUNK_TOKENS = 256
MAX_TOKENS = 1
REPLICAS = 3
WARMUP_BATCHES = 1
MEASURED_BATCHES = 3
CORRECTION_ALPHA = 0.6
MINIMUM_FULL_RECOMPUTE_TOKENS = 32
MINIMUM_REUSE_TOKENS = 256
