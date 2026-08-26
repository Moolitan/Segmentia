"""Fixed configuration for the first CE/SM contention go/no-go test."""

from pathlib import Path

from config import OUTPUT_ROOT


RUN_NAME = "reuse_induced_bandwidth_inversion_smoke_v1"
RUN_ROOT = OUTPUT_ROOT / "contention" / RUN_NAME

SKILL_TOKENS = 3072
CALIBRATION_TOKENS = 32
CHUNK_SIZE_TOKENS = 256
STORAGE_LAYOUT = "packed_chunks_single_layer"
HOST_LAYOUT = "packed_chunks_single_layer"
EXECUTION_ORDER = "h2d_first"

ARMS = ("h2d_only", "calibration_only", "concurrent", "full")
REPETITIONS = 3
WARMUP_REQUESTS = 1
STABLE_LAYER_START = 5
STABLE_LAYER_STOP = 35

CASE_CONFIG_ENV = "CSKCACHE_PINNED_CASE_CONFIG"
CASE_SPECS = RUN_ROOT / "case_specs"
LOG_DIR = RUN_ROOT / "logs"
RUNNER = Path(__file__).resolve().parent / "run.py"
