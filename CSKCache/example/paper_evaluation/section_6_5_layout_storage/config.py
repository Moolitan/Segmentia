"""Physical SSD layout ingestion and SSD-prefetch hierarchy ablation."""

from paper_evaluation.config import EXISTING_LAYOUT_IO_RUN, ROOT, TASK_PROMPT_ROOT
from common.driver import SystemVariant


SSD_LAYOUT_RUN = EXISTING_LAYOUT_IO_RUN
SKILL_TOKENS = 12518
SSD_LAYOUTS = (
    "chunk_all_layers",
    "chunk_single_layer",
    "packed_chunks_single_layer",
    "packed_chunks_all_layers",
)

PLATFORM_ID = "a6000_qwen3_14b"
SKILL_NAME = "doc-coauthoring"
SKILL_PATH = ROOT / "skills/doc-coauthoring/SKILL.md"
TASK_PROMPT_PATH = TASK_PROMPT_ROOT / "doc-coauthoring-retry-design-doc.txt"
SYSTEM = SystemVariant(
    "CSKCache", "cskcache", correction_strategy="fixed_prefix",
    calibration_tokens=256,
)
HIERARCHY_MODES = (
    ("Blocking SSD", False),
    ("Prefetched SSD", True),
)
CHUNK_TOKENS = 256
MAX_TOKENS = 1
REPLICAS = 3
WARMUPS = 1
REPETITIONS = 5
CORRECTION_ALPHA = 0.6
MINIMUM_FULL_RECOMPUTE_TOKENS = 32
MINIMUM_REUSE_TOKENS = 256
