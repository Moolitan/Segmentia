"""Offset validation for regions stored in one raw container."""

from __future__ import annotations


def validate_region_extent(
    *,
    offset_bytes: int,
    length_bytes: int,
    alignment_bytes: int,
    capacity_bytes: int,
    minimum_offset_bytes: int = 0,
) -> None:
    if offset_bytes < minimum_offset_bytes or length_bytes <= 0:
        raise ValueError("raw region extent must be non-empty")
    if offset_bytes % alignment_bytes != 0:
        raise ValueError("raw region offset is not alignment compliant")
    if offset_bytes + length_bytes > capacity_bytes:
        raise ValueError("raw region exceeds container capacity")
