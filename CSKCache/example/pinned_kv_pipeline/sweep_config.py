"""Configuration for the calibration-ratio and chunk-size sweep."""

from config import OUTPUT_ROOT


RUN_NAME = "pipeline_calibration_ratio_chunk_sweep_v1"

SKILL_TOKENS = 8192
CALIBRATION_RATIOS = (0.05, 0.10, 0.15)
CALIBRATION_TOKEN_ALIGNMENT = 16
CHUNK_SIZE_VALUES = (64, 128, 256, 512, 1024, SKILL_TOKENS)
HOST_LAYOUTS = (
    "chunk_single_layer",
    "packed_chunks_single_layer",
)
EXECUTION_ORDERS = ("h2d_first", "compute_first")
WARMUP_REQUESTS = 1
REPETITIONS = 1

CASE_RETRIES = 1
MAX_CONSECUTIVE_FAILURES = 3

# Pair adaptation(layer) with H2D(layer + 1), excluding boundary layers.
STABLE_LAYER_START = 5
STABLE_LAYER_STOP = 35

SWEEP_OUTPUT_ROOT = OUTPUT_ROOT / "sweeps" / RUN_NAME
