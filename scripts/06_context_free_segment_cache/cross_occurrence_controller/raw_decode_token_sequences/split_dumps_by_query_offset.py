"""Split combined attention-probe dumps into per-query-offset directories.

The probe captures several query positions in a single decode pass and writes
every position into one combined dump directory, tagging each row with its
absolute query position (``query_abs_position``). This script routes each row
into ``<out-base>/offset_<NNN>/dumps/<original-file>`` using the per-request
base position recorded in the manifest written by
``run_attention_divergence_probe.py``:

    offset = query_abs_position - base_query_position[request_id]

Only offsets listed in ``--offsets`` produce output directories; rows whose
offset is not requested are dropped (they should not occur in normal runs).

Usage:
  python split_dumps_by_query_offset.py \\
    --combined-dump-dir .../attention_divergence/temp0.6/without_occ12/_combined/dumps \\
    --manifest .../attention_divergence/temp0.6/without_occ12/_combined/query_position_manifest.jsonl \\
    --out-base .../attention_divergence/temp0.6/without_occ12 \\
    --offsets 0,30,60,90,120
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

STEM_PREFIXES = ("attention_region_mass_", "attention_local_windows_")


def load_base_by_request(manifest: Path) -> dict[str, int]:
    base: dict[str, int] = {}
    with manifest.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            base[str(rec["request_id"])] = int(rec["base_query_position"])
    if not base:
        raise ValueError(f"Manifest {manifest} contained no records")
    return base


def offset_dir(out_base: Path, offset: int) -> Path:
    return out_base / f"offset_{offset:03d}" / "dumps"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--combined-dump-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-base", type=Path, required=True)
    parser.add_argument("--offsets", required=True,
                        help="Comma-separated offsets to emit, e.g. 0,30,60,90,120")
    args = parser.parse_args()

    wanted = {int(o) for o in str(args.offsets).split(",") if o.strip() != ""}
    base_by_request = load_base_by_request(args.manifest)

    # Prepare clean output dirs so re-runs do not append onto stale rows.
    for off in sorted(wanted):
        d = offset_dir(args.out_base, off)
        d.mkdir(parents=True, exist_ok=True)

    total_in = 0
    total_out = 0
    dropped = defaultdict(int)
    # Accumulate lines per (offset, filename) then write once.
    buffers: dict[tuple[int, str], list[str]] = defaultdict(list)

    for stem in STEM_PREFIXES:
        for path in sorted(args.combined_dump_dir.glob(f"{stem}*.jsonl")):
            # Clear any prior split of this same source file.
            for off in wanted:
                target = offset_dir(args.out_base, off) / path.name
                if target.exists():
                    target.unlink()
            with path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    total_in += 1
                    row: dict[str, Any] = json.loads(line)
                    req = str(row["request_id"])
                    base = base_by_request.get(req)
                    if base is None:
                        dropped["no_base"] += 1
                        continue
                    offset = int(row["query_abs_position"]) - base
                    if offset not in wanted:
                        dropped[f"offset_{offset}"] += 1
                        continue
                    buffers[(offset, path.name)].append(json.dumps(row, ensure_ascii=False))
                    total_out += 1

    for (offset, fname), lines in sorted(buffers.items()):
        target = offset_dir(args.out_base, offset) / fname
        with target.open("w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    print(f"[split] read {total_in} rows, wrote {total_out} rows into "
          f"{len(wanted)} offset dirs under {args.out_base}")
    for off in sorted(wanted):
        d = offset_dir(args.out_base, off)
        n_files = len(list(d.glob('*.jsonl')))
        print(f"  offset {off:>3}: {n_files} files -> {d}")
    if dropped:
        print(f"[split] dropped rows by reason: {dict(dropped)}")


if __name__ == "__main__":
    main()
