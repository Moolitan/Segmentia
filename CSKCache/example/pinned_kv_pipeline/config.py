"""Configuration for the single-request pinned-KV pipeline example."""

import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
_CASE_CONFIG_PATH = os.environ.get("CSKCACHE_PINNED_CASE_CONFIG")
_CASE_CONFIG = (
    json.loads(Path(_CASE_CONFIG_PATH).read_text(encoding="utf-8"))
    if _CASE_CONFIG_PATH
    else {}
)

MODEL_PATH = Path(
    "/mnt/Large_Language_Model_Lab_1/llm_models/Qwen3-14B/Qwen/Qwen3-14B"
)
OUTPUT_ROOT = Path(
    "/mnt/Large_Language_Model_Lab_1/wsh/CSKCache/output/"
    "pinned_kv_pipeline"
)
SKILL_NAME = "synthetic-pipeline"
SKILL_TOKENS = int(_CASE_CONFIG.get("skill_tokens", 8000))
PREFIX_TOKENS = 256
TAIL_TOKENS = 32
MINIMUM_FULL_RECOMPUTE_TOKENS = 32
CALIBRATION_TOKENS = int(_CASE_CONFIG.get("calibration_tokens", 32))
CHUNK_SIZE_TOKENS = int(_CASE_CONFIG.get("chunk_size_tokens", 256))
STORAGE_LAYOUT = str(
    _CASE_CONFIG.get("storage_layout", "packed_chunks_single_layer")
)
HOST_LAYOUT = str(
    _CASE_CONFIG.get("host_layout", "packed_chunks_single_layer")
)
EXECUTION_ORDER = str(_CASE_CONFIG.get("execution_order", "h2d_first"))
WARMUP_REQUESTS = int(_CASE_CONFIG.get("warmup_requests", 0))
MINIMUM_REUSE_TOKENS = 256
CORRECTION_ALPHA = 0.6

PROMPT_FILL_TOKEN_ID = 100
KV_FILL_VALUE = 0.0
MAX_MODEL_LEN = 32000
GPU_MEMORY_UTILIZATION = 0.9
MAX_TOKENS = 1
RUN_DIR_OVERRIDE = (
    Path(_CASE_CONFIG["run_dir"]) if "run_dir" in _CASE_CONFIG else None
)
