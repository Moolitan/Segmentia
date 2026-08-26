"""Fixed matrix for cross-model and cross-platform TTFT."""

from paper_evaluation.config import ACTIVE_PLATFORMS, ROOT, TASK_PROMPT_ROOT
from common.driver import SystemVariant


PLATFORM_IDS = ACTIVE_PLATFORMS
WORKLOADS = (
    (
        "internal-comms",
        ROOT / "skills/internal-comms/SKILL.md",
        TASK_PROMPT_ROOT / "internal-comms-incident-update.txt",
    ),
    (
        "doc-coauthoring",
        ROOT / "skills/doc-coauthoring/SKILL.md",
        TASK_PROMPT_ROOT / "doc-coauthoring-retry-design-doc.txt",
    ),
    (
        "docx",
        ROOT / "skills/docx/SKILL.md",
        TASK_PROMPT_ROOT / "docx-convert-pdf-images.txt",
    ),
    (
        "paper-writing",
        ROOT
        / "skills/Auto-claude-code-research-in-sleep/skills/paper-writing/SKILL.md",
        TASK_PROMPT_ROOT / "paper-writing-segmentia-paper-plan.txt",
    ),
)
SYSTEMS = (
    SystemVariant("Full", "full"),
    SystemVariant("CacheBlend-15%", "cacheblend", cacheblend_ratio=0.15),
    SystemVariant(
        "Ratio-15%", "cskcache", correction_strategy="ratio_prefix",
        calibration_ratio=0.15,
    ),
    SystemVariant(
        "CSKCache", "cskcache", correction_strategy="fixed_prefix",
        calibration_tokens=256,
    ),
)
CHUNK_TOKENS = 256
MAX_TOKENS = 1
REPLICAS = 3
WARMUPS = 1
REPETITIONS = 5
CORRECTION_ALPHA = 0.6
MINIMUM_FULL_RECOMPUTE_TOKENS = 32
MINIMUM_REUSE_TOKENS = 256
