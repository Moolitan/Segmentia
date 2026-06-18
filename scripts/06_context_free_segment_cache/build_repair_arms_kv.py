"""Build per-case mixed KV files for the value-side repair 2x2 ablation.

This is the offline preparation step for experiment (d): isolating how much of
the reuse behavior gap is caused by a stale KEY (position mismatch) versus a
stale VALUE (representational drift from the no-context skill prefill).

We never modify vLLM. Instead we splice two existing KV sources into new
per-case `.pt` files that the existing injection path can consume:

  - skill KV   : results/06.../offline_skill_kv/context-free-skill-<skill>.pt
                 (key/value prefilled with NO surrounding context; source_start
                 is the position inside the minimal wrapper)
  - recompute  : results/06.../cksim_kv/cf-recompute-<task>-<skill>-occ<n>.pt
                 (the in-context ground-truth key/value dumped at the real
                 target position; this is the "oracle")

From these we emit three arms per case:

  vrep   = skill key      + recompute value   -> inject with mode=rope
  krep   = recompute key  + skill value       -> inject with mode=direct
  oracle = recompute key  + recompute value   -> inject with mode=direct

vrep keeps the skill key's offline source_start so the worker re-rotates it to
the target exactly like the plain `rope` arm; the only change versus `rope` is
that the value is the oracle value. krep/oracle carry the oracle key, which was
dumped already sitting at the target position, so they need no rotation and use
mode=direct.

Comparisons this enables (behavior fidelity vs recompute):
  rope   vs vrep    -> marginal effect of repairing the VALUE
  rope   vs krep    -> marginal effect of repairing the KEY
  oracle vs recompute -> sanity check that the splice path itself is faithful

Run this AFTER a recompute pass dumped the oracle KV, e.g.:
  python run_decode_compare.py --modes recompute --dump-kv-for-cksim ...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from config import (  # noqa: E402
    DEFAULT_CKSIM_KV_DIR,
    DEFAULT_KV_DIR,
    DEFAULT_REPAIR_KV_DIR,
    cache_id_for_skill,
    get_skill_token_span,
    parse_tasks,
    SKILL_TOKEN_LOCATIONS,
)


def load_pt(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def build_entry(
    cache_id: str,
    source_start: int,
    length: int,
    key_src: dict[str, Any],
    value_src: dict[str, Any],
    token_ids: list[int] | None,
) -> dict[str, Any]:
    layers = sorted(set(key_src["kv_by_layer"]) & set(value_src["kv_by_layer"]))
    if not layers:
        raise ValueError("no overlapping layers between key and value sources")
    kv_by_layer: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for layer in layers:
        k = key_src["kv_by_layer"][layer][0]
        v = value_src["kv_by_layer"][layer][1]
        if k.shape[0] < length or v.shape[0] < length:
            raise ValueError(
                f"layer {layer}: source too short for length {length} "
                f"(k={tuple(k.shape)}, v={tuple(v.shape)})"
            )
        kv_by_layer[layer] = (
            k[:length].contiguous().clone(),
            v[:length].contiguous().clone(),
        )
    return {
        "cache_id": cache_id,
        "source_start": int(source_start),
        "source_end": int(source_start + length),
        "token_ids": token_ids[:length] if token_ids else None,
        "kv_by_layer": kv_by_layer,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--occurrences", default="2,3")
    parser.add_argument("--kv-dir", default=str(DEFAULT_KV_DIR))
    parser.add_argument("--cksim-kv-dir", default=str(DEFAULT_CKSIM_KV_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_REPAIR_KV_DIR))
    parser.add_argument(
        "--arms",
        default="vrep,krep,oracle",
        help="Subset of {vrep,krep,oracle} to build.",
    )
    args = parser.parse_args()

    tasks = parse_tasks(args.tasks)
    occurrences = [int(x) for x in args.occurrences.split(",") if x.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = sorted(set(arms) - {"vrep", "krep", "oracle"})
    if unknown:
        raise ValueError(f"Unknown arms: {unknown}")

    kv_dir = Path(args.kv_dir)
    cksim_dir = Path(args.cksim_kv_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for task in tasks:
        for skill in SKILL_TOKEN_LOCATIONS[task]["skills"]:
            skill_path = kv_dir / f"{cache_id_for_skill(skill)}.pt"
            if not skill_path.exists():
                skipped.append({"task": task, "skill": skill, "reason": "missing skill KV"})
                continue
            skill_entry = load_pt(skill_path)
            for occurrence in occurrences:
                if occurrence == 1:
                    continue
                start, end = get_skill_token_span(task, skill, occurrence)
                length = end - start
                rec_path = cksim_dir / f"cf-recompute-{task}-{skill}-occ{occurrence}.pt"
                if not rec_path.exists():
                    skipped.append(
                        {
                            "task": task,
                            "skill": skill,
                            "occurrence": occurrence,
                            "reason": f"missing recompute dump {rec_path.name}",
                        }
                    )
                    continue
                rec_entry = load_pt(rec_path)
                skill_len = skill_entry["source_end"] - skill_entry["source_start"]
                if skill_len < length:
                    raise ValueError(
                        f"{task}/{skill}: skill KV len {skill_len} < target span {length}"
                    )

                # The live worker validates entry.token_ids against the prompt
                # tokens at the target span. The skill KV token_ids are the
                # proven match (the direct/rope arms pass this check), and the
                # recompute dump's first L tokens are byte-identical to them, so
                # all arms carry the skill token_ids regardless of key source.
                target_token_ids = skill_entry.get("token_ids")
                specs = {
                    "vrep": dict(
                        source_start=skill_entry["source_start"],
                        key_src=skill_entry,
                        value_src=rec_entry,
                        token_ids=target_token_ids,
                    ),
                    "krep": dict(
                        source_start=start,
                        key_src=rec_entry,
                        value_src=skill_entry,
                        token_ids=target_token_ids,
                    ),
                    "oracle": dict(
                        source_start=start,
                        key_src=rec_entry,
                        value_src=rec_entry,
                        token_ids=target_token_ids,
                    ),
                }
                for arm in arms:
                    spec = specs[arm]
                    cache_id = f"cf-{arm}-{task}-{skill}-occ{occurrence}"
                    entry = build_entry(
                        cache_id,
                        spec["source_start"],
                        length,
                        spec["key_src"],
                        spec["value_src"],
                        spec["token_ids"],
                    )
                    # Scope each case under out_dir/<task>/ rather than a flat
                    # directory: vLLM eager-loads every .pt file in whatever
                    # directory VLLM_CONTEXT_SEGMENT_KV_DIR points at, on every
                    # worker restart. A flat dir holding all tasks' arms makes
                    # every restart pay for loading everyone else's KV too
                    # (~16GB total vs ~1-2GB needed by a single task).
                    task_dir = out_dir / task
                    task_dir.mkdir(parents=True, exist_ok=True)
                    out_path = task_dir / f"{cache_id}.pt"
                    torch.save(entry, out_path)
                    records.append(
                        {
                            "arm": arm,
                            "task": task,
                            "skill": skill,
                            "occurrence": occurrence,
                            "cache_id": cache_id,
                            "source_start": entry["source_start"],
                            "source_end": entry["source_end"],
                            "length": length,
                            "layers": len(entry["kv_by_layer"]),
                        }
                    )
                    print(
                        f"[build] {arm:6s} {task} {skill} occ{occurrence} "
                        f"len={length} -> {out_path.name}",
                        flush=True,
                    )

    manifest = {
        "kv_dir": str(kv_dir),
        "cksim_kv_dir": str(cksim_dir),
        "out_dir": str(out_dir),
        "arms": arms,
        "built": records,
        "skipped": skipped,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] built {len(records)} entries; skipped {len(skipped)}")
    print(f"[done] manifest: {manifest_path}")


if __name__ == "__main__":
    main()
