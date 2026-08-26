"""Configuration for the Host-layout, compute, and H2D latency sweep."""

from config import OUTPUT_ROOT


RUN_NAME = "pipeline_layout_compute_latency_v1"

SKILL_TOKEN_VALUES = (512, 1024, 3000, 5000, 8192, 10000)
CALIBRATION_TOKENS = 32
CHUNK_SIZE_TOKENS = 256
HOST_LAYOUTS = (
    "chunk_single_layer",
    "packed_chunks_single_layer",
)
EXECUTION_ORDERS = ("compute_first",)
WARMUP_REQUESTS = 1
REPETITIONS = 1

CASE_RETRIES = 1
MAX_CONSECUTIVE_FAILURES = 3

# Pair adaptation(layer) with H2D(layer + 1), excluding boundary layers.
STABLE_LAYER_START = 5
STABLE_LAYER_STOP = 35

SWEEP_OUTPUT_ROOT = OUTPUT_ROOT / "sweeps" / RUN_NAME
