"""Logical Chunk granularity matrix."""

from paper_evaluation.config import ROOT, TASK_PROMPT_ROOT


PLATFORM_ID = "a6000_qwen3_14b"
WORKLOADS = (
    (
        "internal-comms", ROOT / "skills/internal-comms/SKILL.md",
        TASK_PROMPT_ROOT / "internal-comms-incident-update.txt",
    ),
    (
        "doc-coauthoring", ROOT / "skills/doc-coauthoring/SKILL.md",
        TASK_PROMPT_ROOT / "doc-coauthoring-retry-design-doc.txt",
    ),
    (
        "docx", ROOT / "skills/docx/SKILL.md",
        TASK_PROMPT_ROOT / "docx-convert-pdf-images.txt",
    ),
    (
        "paper-writing",
        ROOT / "skills/Auto-claude-code-research-in-sleep/skills/paper-writing/SKILL.md",
        TASK_PROMPT_ROOT / "paper-writing-segmentia-paper-plan.txt",
    ),
)
CHUNK_TOKENS = (64, 128, 256, 512)
MUTATIONS = (
    ("exact", 0.0),
    ("replace", 0.25),
    ("replace", 0.50),
    ("replace", 0.75),
    ("append", 1.0),
)
REPETITIONS = 3
MAX_TOKENS = 1
CALIBRATION_TOKENS = 256
CORRECTION_ALPHA = 0.6
MINIMUM_FULL_RECOMPUTE_TOKENS = 32
MINIMUM_REUSE_TOKENS = 256
INCLUDE_NATIVE_CACHEBLEND = True
