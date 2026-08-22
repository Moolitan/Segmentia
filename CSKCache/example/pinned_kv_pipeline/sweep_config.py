"""Configuration for the warmed critical pinned-KV sweep."""

from config import OUTPUT_ROOT


RUN_NAME = "pipeline_warmed_critical_v2"

EXECUTION_ORDERS = ("h2d_first", "compute_first")
SKILL_TOKEN_VALUES = (1000, 3000, 5000, 8000)
CALIBRATION_BY_SKILL = {
    1000: 16,
    3000: 16,
    5000: 32,
    8000: 64,
}

# The critical fragmentation point and the one-buffer-per-layer reference.
CHUNK_SIZE_TOKENS = 256
WARMUP_REQUESTS = 1
REPETITIONS = 3

CASE_RETRIES = 1
MAX_CONSECUTIVE_FAILURES = 3

# Pair adaptation(layer) with H2D(layer + 1), excluding boundary layers.
STABLE_LAYER_START = 5
STABLE_LAYER_STOP = 35

SWEEP_OUTPUT_ROOT = OUTPUT_ROOT / "sweeps" / RUN_NAME
