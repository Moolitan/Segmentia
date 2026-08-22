"""Configuration for the packed-layout stage crossover sweep."""

from config import OUTPUT_ROOT


RUN_NAME = "pipeline_stage_crossover_ratio_v1"

SKILL_TOKEN_VALUES = (8192, 12000)
CALIBRATION_RATIOS = (0.01, 0.015, 0.02, 0.025, 0.03)
CALIBRATION_TOKEN_ALIGNMENT = 16
CHUNK_SIZE_TOKENS = 256
HOST_LAYOUTS = ("packed_chunks_single_layer",)
EXECUTION_ORDERS = ("compute_first",)
WARMUP_REQUESTS = 1
REPETITIONS = 3

CASE_RETRIES = 1
MAX_CONSECUTIVE_FAILURES = 3

# Pair adaptation(layer) with H2D(layer + 1), excluding boundary layers.
STABLE_LAYER_START = 5
STABLE_LAYER_STOP = 35

SWEEP_OUTPUT_ROOT = OUTPUT_ROOT / "sweeps" / RUN_NAME
