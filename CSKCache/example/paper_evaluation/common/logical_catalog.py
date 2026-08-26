"""Derive logical-Chunk Catalog variants without copying packed KV payloads."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping


def derive_catalog(
    *,
    source_catalog: Path,
    output_catalog: Path,
    tokenizer_path: Path,
    skill_paths: Mapping[str, Path],
    chunk_tokens: int,
) -> Path:
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive")
    if output_catalog.is_file():
        return output_catalog
    from transformers import AutoTokenizer
    from cskcache import build_skill_token_identity, fingerprint_full_token_chunks

    payload = json.loads(source_catalog.read_text(encoding="utf-8"))
    objects = [
        cache_object
        for cache_object in (payload.get("objects") or [])
        if str(cache_object.get("skill_name", "")) in skill_paths
    ]
    if not objects:
        raise ValueError("source Catalog has no configured cache objects")
    payload["objects"] = objects
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, local_files_only=True
    )
    for cache_object in objects:
        skill_name = str(cache_object["skill_name"])
        skill_path = skill_paths[skill_name]
        identity = build_skill_token_identity(
            tokenizer,
            skill_name,
            skill_path.read_text(encoding="utf-8"),
        )
        if len(identity.token_ids) != int(cache_object["token_count"]):
            raise ValueError(
                f"Catalog/tokenizer token count changed for {skill_name}: "
                f"catalog={cache_object['token_count']} current={len(identity.token_ids)}"
            )
        if identity.token_ids_sha256 != cache_object["token_ids_sha256"]:
            raise ValueError(f"Catalog token identity changed for {skill_name}")
        cache_object["chunking"] = {"chunk_size_tokens": chunk_tokens}
        cache_object["chunk_token_ids_sha256"] = list(
            fingerprint_full_token_chunks(identity.token_ids, chunk_tokens)
        )
    output_catalog.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_catalog.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_catalog)
    return output_catalog
