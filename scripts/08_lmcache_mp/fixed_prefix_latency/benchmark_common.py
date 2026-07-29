#!/usr/bin/env python3
"""Shared deterministic workload helpers for fixed-prefix latency tests."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


ARMS = ("full", "direct", "prefix_no_correction", "prefix_256")
REUSE_ARMS = frozenset(ARMS) - {"full"}
PREFIX_ARMS = frozenset({"prefix_no_correction", "prefix_256"})
DEFAULT_LENGTHS = (512, 640, 768, 1024, 1280, 1536, 1792, 2048, 2560, 3301)
SEPARATOR_TOKEN_ID = 151663
PREFIX_TOKENS = 4096
SUFFIX_TOKENS = 64


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def parse_lengths(value: str) -> list[int]:
    lengths = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not lengths or any(length <= 0 for length in lengths):
        raise ValueError("lengths must be a non-empty list of positive integers")
    if len(lengths) != len(set(lengths)):
        raise ValueError("lengths must not contain duplicates")
    return sorted(lengths)


def token_stream(length: int, *, namespace: int, nonce: int = 0) -> list[int]:
    """Return valid-looking deterministic token ids without tokenization.

    The namespace and nonce are mixed into every position.  Repeated target
    requests therefore have equal lengths but unrelated leading block hashes,
    while a Skill of a given length is identical in source and target prompts.
    """

    if length < 0:
        raise ValueError("token length must be non-negative")
    base = 1000 + (namespace * 7919 + nonce * 104729) % 120000
    tokens = [1000 + ((base + index * 1543 + (index * index) % 997) % 140000) for index in range(length)]
    return [token + 1 if token == SEPARATOR_TOKEN_ID else token for token in tokens]


def skill_tokens(skill_length: int) -> list[int]:
    return token_stream(skill_length, namespace=100 + skill_length)


def build_prompt(
    *, skill_length: int, phase: str, arm: str, replica: int, nonce: int
) -> tuple[list[int], int, int, int]:
    if arm not in ARMS and phase != "source":
        raise ValueError(f"unsupported arm: {arm}")
    # All target arms must see byte-identical token prompts for the same
    # (replica, nonce, Skill length).  Arm changes execution only, never input.
    phase_namespace = 1 if phase == "source" else 2
    prefix = token_stream(
        PREFIX_TOKENS,
        namespace=phase_namespace + replica * 17,
        nonce=nonce,
    )
    reusable = skill_tokens(skill_length)
    suffix = token_stream(
        SUFFIX_TOKENS,
        namespace=300 + phase_namespace + replica * 17,
        nonce=nonce,
    )
    segment_start = len(prefix) + 1
    segment_end = segment_start + len(reusable) + 1
    prompt = prefix + [SEPARATOR_TOKEN_ID] + reusable + [SEPARATOR_TOKEN_ID] + suffix
    return prompt, segment_start, segment_end, segment_end - 1


def lookup_params(arm: str, segment_start: int, segment_end: int, cache_end: int) -> dict[str, Any] | None:
    if arm == "full":
        return None
    lookup: dict[str, Any] = {
        "segment_start": segment_start,
        "segment_end": segment_end,
    }
    if arm in PREFIX_ARMS:
        lookup.update(
            {
                "correction_mode": "prefix_k_headwise",
                "cache_end": cache_end,
                "prefix_tokens": 256,
                "calibration_start": 132,
                "calibration_end": 256,
                "minimum_reuse_tokens": 256,
            }
        )
    return lookup


def token_sha256(tokens: list[int]) -> str:
    digest = hashlib.sha256()
    for token in tokens:
        digest.update(token.to_bytes(4, "little", signed=False))
    return digest.hexdigest()


def request_payload(
    *, model: str, arm: str, prompt: list[int], segment_start: int,
    segment_end: int, cache_end: int, request_id: str, skip_save: bool
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 1,
        "temperature": 0,
        "request_id": request_id,
    }
    lookup = lookup_params(arm, segment_start, segment_end, cache_end)
    if lookup is not None:
        transfer: dict[str, Any] = {"lmcache_segmentia_lookup": lookup}
        if skip_save:
            transfer["lmcache.skip_save"] = True
        payload["kv_transfer_params"] = transfer
    return payload
