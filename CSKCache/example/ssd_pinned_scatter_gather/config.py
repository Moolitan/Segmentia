"""Editable configuration for the SSD-to-pinned scatter/gather benchmark."""

from pathlib import Path


DATA_ROOT = Path("/mnt/990_pro/cskcache_layout_io/12518_tokens")
OUTPUT_ROOT = Path(
    "/mnt/Large_Language_Model_Lab_1/wsh/CSKCache/output/"
    "ssd_pinned_scatter_gather"
)
DATA_MOUNT = Path("/mnt/990_pro")
EXPECTED_DEVICE = "/dev/nvme1n1"

TOKEN_COUNT = 12518
CHUNK_SIZE_TOKENS = 256
NUM_LAYERS = 40
KV_HEAD_DIM = 1024
DTYPE_BYTES = 2
DATASET_SEED = 20260824
ALIGNMENT_BYTES = 4096

SOURCE_LAYOUT = "packed_chunks_all_layers"
MAX_IOVECS_PER_CALL = 1024
IO_URING_QUEUE_DEPTH = 32
WARMUPS = 2
REPETITIONS = 20
CASE_ORDER_SEED = 20260824
