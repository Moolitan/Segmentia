"""Persistent container-format vocabulary."""

from enum import Enum


class StorageFormat(str, Enum):
    """Implemented encodings of layout regions."""

    TORCH_PT = "torch_pt"
    RAW_CONTAINER = "raw_container"
