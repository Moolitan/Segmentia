"""Stable model and tokenizer fingerprints shared by packing and serving."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence
import hashlib


def fingerprint_model(model_path: str | Path) -> str:
    """Fingerprint one local model without rereading all weight payloads."""

    root = Path(model_path)
    digest = hashlib.sha256()
    identity_files = [
        root / name
        for name in (
            "config.json",
            "generation_config.json",
            "model.safetensors.index.json",
            "pytorch_model.bin.index.json",
        )
        if (root / name).is_file()
    ]
    weight_files = sorted(root.glob("*.safetensors")) + sorted(
        root.glob("pytorch_model*.bin")
    )
    if not identity_files or not weight_files:
        raise RuntimeError(f"model identity files are incomplete: {root}")
    for path in sorted(identity_files):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    for path in weight_files:
        digest.update(path.name.encode())
        digest.update(str(path.stat().st_size).encode())
    return digest.hexdigest()


def fingerprint_tokenizer(tokenizer_path: str | Path) -> str:
    """Fingerprint the tokenizer and chat-template files used for Prefill."""

    root = Path(tokenizer_path)
    names = {
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "chat_template.jinja",
    }
    files = sorted(path for path in root.iterdir() if path.name in names)
    if not files:
        raise RuntimeError(f"tokenizer identity files are missing: {root}")
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def fingerprint_token_ids(token_ids: Sequence[int]) -> str:
    """Hash the exact token sequence used as one CSKCache object identity."""

    digest = hashlib.sha256()
    for token_id in token_ids:
        value = int(token_id)
        if value < 0 or value >= 1 << 32:
            raise ValueError(f"token ID is outside uint32: {value}")
        digest.update(value.to_bytes(4, "little", signed=False))
    return digest.hexdigest()


def fingerprint_full_token_chunks(
    token_ids: Sequence[int],
    chunk_size_tokens: int,
) -> tuple[str, ...]:
    """Fingerprint complete fixed-size chunks for prefix authentication.

    The final short tail is deliberately omitted.  A partial-version hit may
    end only at a complete logical-chunk boundary; an exact full-object hit is
    still authenticated by ``token_ids_sha256``.
    """

    if chunk_size_tokens <= 0:
        raise ValueError("chunk_size_tokens must be > 0")
    complete_tokens = len(token_ids) - (len(token_ids) % chunk_size_tokens)
    return tuple(
        fingerprint_token_ids(token_ids[start : start + chunk_size_tokens])
        for start in range(0, complete_tokens, chunk_size_tokens)
    )
