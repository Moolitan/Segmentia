"""Editable configuration for the SSD-to-pinned layout I/O benchmark."""

from pathlib import Path


DATA_ROOT = Path("/mnt/990_pro/cskcache_layout_io/12518_tokens")
OUTPUT_ROOT = Path(
    "/mnt/Large_Language_Model_Lab_1/wsh/CSKCache/output/ssd_pinned_layout_io"
)

# Fail closed when DATA_ROOT is not backed by this writable mount/device.
DATA_MOUNT = Path("/mnt/990_pro")
EXPECTED_DEVICE = "/dev/nvme1n1"

# Synthetic Qwen3-14B-shaped KV. Content does not affect raw read bandwidth;
# deterministic bytes make every transfer verifiable.
TOKEN_COUNT = 12518
CHUNK_SIZE_TOKENS = 256
NUM_LAYERS = 40
KV_HEAD_DIM = 1024
DTYPE_BYTES = 2
DATASET_SEED = 20260824

ALIGNMENT_BYTES = 4096
HEADER_BYTES = 4096
METADATA_BYTES = 64 * 1024**2
IO_URING_QUEUE_DEPTH = 32

WARMUPS = 2
REPETITIONS = 20
CASE_ORDER_SEED = 20260824
