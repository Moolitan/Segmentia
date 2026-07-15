#!/usr/bin/env python3
"""One-time migration: add the `length` field to existing LocalDiskBackend
JSON sidecars.

CSKCache's metadata-only scheduler lookup (see cache_engine.py's
_reuse_from_signal -> StorageManager.get_metadata()) needs each sidecar to
carry `length` (= source_end - source_start). Sidecars written before that
change only have `cache_id`/`nbytes`/`num_tokens`, so get_metadata() falls
back to a full torch.load for every one of them -- silently reproducing the
exact cost the optimization was meant to remove.

This script pays that full-read cost once per entry, offline, and patches
only the small .json sidecar in place. It never rewrites the .pt payload, so
multi-hundred-MB files are left untouched (same bytes, same mtime).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--disk-dir", type=Path, required=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would change without writing any sidecar.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sidecars = sorted(args.disk_dir.glob("*.json"))
    migrated, skipped, errors = 0, 0, 0
    for sidecar_path in sidecars:
        try:
            meta = json.loads(sidecar_path.read_text())
        except Exception as exc:
            print(f"[error] {sidecar_path.name}: unreadable sidecar ({exc})")
            errors += 1
            continue
        if "cache_id" not in meta:
            # Not a per-entry sidecar (e.g. an index manifest); leave it alone.
            print(f"[skip] {sidecar_path.name}: no cache_id field, not an entry sidecar")
            skipped += 1
            continue
        if "length" in meta:
            print(f"[skip] {sidecar_path.name}: cache_id={meta['cache_id']!r} already has length")
            skipped += 1
            continue
        payload_path = sidecar_path.with_suffix(".pt")
        if not payload_path.exists():
            print(f"[error] {sidecar_path.name}: cache_id={meta['cache_id']!r} missing {payload_path.name}")
            errors += 1
            continue
        payload = torch.load(payload_path, map_location="cpu")
        length = int(payload["source_end"]) - int(payload["source_start"])
        print(
            f"[migrate] {sidecar_path.name}: cache_id={meta['cache_id']!r} "
            f"num_tokens={meta.get('num_tokens')} -> length={length}"
        )
        if not args.dry_run:
            meta["length"] = length
            sidecar_path.write_text(json.dumps(meta))
        migrated += 1
    print(
        f"[done] migrated={migrated} skipped={skipped} errors={errors} "
        f"dry_run={args.dry_run}"
    )
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
