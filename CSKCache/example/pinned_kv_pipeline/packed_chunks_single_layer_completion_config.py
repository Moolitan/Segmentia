"""Configuration for the packed-chunks-single-layer completion sweep."""

from config import OUTPUT_ROOT


RUN_NAME = "packed_chunks_single_layer_completion_without_final_layer_v1"

SKILL_TOKEN_VALUES = (1024, 2048, 4096, 8192)
CHUNK_SIZE_TOKEN_VALUES = (64, 128, 256, 512)
CALIBRATION_TOKENS = 32
STORAGE_LAYOUT = "packed_chunks_single_layer"
HOST_LAYOUT = "packed_chunks_single_layer"
EXECUTION_ORDER = "compute_first"

WARMUP_REQUESTS = 1
REPETITIONS = 3
CASE_RETRIES = 1
MAX_CONSECUTIVE_FAILURES = 3

# Qwen3-14B has layers 0--39.  The request still restores all layers; the
# primary completion metric uses layers 0--38 to match the chunkwise sweep.
EXPECTED_LAYERS = 40
COMPLETION_LAYER_STOP = 39

SWEEP_OUTPUT_ROOT = OUTPUT_ROOT / "sweeps" / RUN_NAME
