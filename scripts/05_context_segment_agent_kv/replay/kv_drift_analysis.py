"""Characterize WHERE the reuse error lives: Key vs Value, shallow vs deep.

This is the analysis behind Step 2 of the position/content decomposition. It does
NOT touch vLLM; it only reads the .pt KV dumps that trace_reuse_cksim.py already
produced (recompute vs reuse pairs) and measures, per layer:

  * Kcos / Vcos   - per-token cosine similarity (recompute vs reuse) for K and V.
                    Same CKSim convention as trace_reuse_cksim.py.
  * dK / dV       - relative drift norm ||recompute - reuse|| / ||recompute||.
                    ~0 means identical, ~1.0 means roughly orthogonal (unusable).
  * K_top / V_top - fraction of total energy held by the single most energetic
                    feature dim (head_dim axis). High => "massive activation":
                    a few stable dims dominate the vector.
  * Kcos_noTop    - K cosine after dropping the top ~2% most energetic dims, to
                    test whether K's high similarity is merely propped up by
                    massive activations (it is not - it only drops partway).

Key finding it reproduces: deep-layer VALUE drifts far more than KEY (V ~0.5 vs
K ~0.9), consistently across very different skill sizes. K behaves like stable
"addressing" info (massive activations, context-robust); V is context-dependent
"payload" and is near-unusable deep. This contradicts CacheSlide's claim that V
similarity tracks K similarity.

Usage:
    python kv_drift_analysis.py                      # all cases in the summary
    python kv_drift_analysis.py --task internal_comms_incident_update
    python kv_drift_analysis.py --kv-dir <dir> --summary <summary.json>
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

PKG_ROOT = Path(__file__).resolve().parent.parent
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from core.config import ROOT  # noqa: E402

OUTPUT_ROOT = ROOT / "results" / "05_context_segment_agent_kv" / "CKSim"
DEFAULT_KV_DIR = OUTPUT_ROOT / "kv_cache_trace"
DEFAULT_SUMMARY = OUTPUT_ROOT / "trace_reuse_cksim_summary.json"
DEFAULT_CSV = OUTPUT_ROOT / "kv_drift_analysis.csv"


def layer_index(name: str) -> int:
    return int(name.split("layers.")[1].split(".")[0])


def as_heads(x: torch.Tensor, tokens: int) -> torch.Tensor:
    # stored [tokens, kv_heads, head_dim] -> [kv_heads, tokens, head_dim], fp32
    return x[:tokens].permute(1, 0, 2).contiguous().float()


def cos_per_token(a: torch.Tensor, b: torch.Tensor) -> float:
    # cosine over head_dim, averaged across heads and tokens
    return float(F.cosine_similarity(a, b, dim=2).mean().item())


def top_dim_energy(x: torch.Tensor) -> float:
    # fraction of total energy in the single most energetic head_dim
    e = (x ** 2).sum(dim=(0, 1))
    return float((e.max() / e.sum()).item())


def cos_drop_top(a: torch.Tensor, b: torch.Tensor, frac: float = 0.02) -> float:
    # cosine after removing the top-energy ~frac of dims (massive-activation test)
    e = (a ** 2).sum(dim=(0, 1))
    k = max(1, int(a.shape[2] * frac))
    drop = e.topk(k).indices
    keep = torch.ones(a.shape[2], dtype=torch.bool)
    keep[drop] = False
    return float(F.cosine_similarity(a[:, :, keep], b[:, :, keep], dim=2).mean().item())


def analyze_case(kv_dir: Path, recompute_id: str, reuse_id: str, tokens: int) -> list[dict]:
    rc = torch.load(kv_dir / f"{recompute_id}.pt", map_location="cpu", weights_only=False)
    ru = torch.load(kv_dir / f"{reuse_id}.pt", map_location="cpu", weights_only=False)
    layers = sorted(set(rc["kv_by_layer"]) & set(ru["kv_by_layer"]), key=layer_index)
    rows = []
    for name in layers:
        rk, rv = rc["kv_by_layer"][name]
        uk, uv = ru["kv_by_layer"][name]
        rk, rv, uk, uv = (as_heads(t, tokens) for t in (rk, rv, uk, uv))
        rows.append(
            {
                "layer": layer_index(name),
                "key_cksim": cos_per_token(rk, uk),
                "value_cksim": cos_per_token(rv, uv),
                "key_drift_norm": float(((rk - uk).norm() / rk.norm()).item()),
                "value_drift_norm": float(((rv - uv).norm() / rv.norm()).item()),
                "key_topdim_energy": top_dim_energy(rk),
                "value_topdim_energy": top_dim_energy(rv),
                "key_cksim_drop_top2pct": cos_drop_top(rk, uk),
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="K-vs-V / shallow-vs-deep reuse drift analysis.")
    ap.add_argument("--kv-dir", default=str(DEFAULT_KV_DIR))
    ap.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    ap.add_argument("--task", default=None, help="filter to a single task name")
    ap.add_argument("--csv", default=str(DEFAULT_CSV))
    args = ap.parse_args()

    kv_dir = Path(args.kv_dir)
    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    cases = summary["cases"]
    if args.task:
        cases = [c for c in cases if c["task"] == args.task]
    if not cases:
        raise SystemExit("no matching cases in summary")

    def reuse_id_of(c: dict) -> str:
        # new summary format nests the reuse cache id under comparisons; fall back
        # to the old flat key.
        if "reuse_cache_id" in c:
            return c["reuse_cache_id"]
        return c["comparisons"]["recompute_vs_reuse"]["cache_id"]

    all_rows = []
    show = lambda li: li in (0, 1, 5, 10, 15) or (li >= 20 and li % 4 == 0) or li >= 38
    for c in cases:
        rows = analyze_case(kv_dir, c["recompute_cache_id"], reuse_id_of(c), c["skill_tokens"])
        # if a no-rope reuse variant exists, report its case-level means: this
        # isolates the positional half of the KEY drift (re-rotation lifts K, V
        # is rotation-invariant).
        cmp = c.get("comparisons", {})
        if "recompute_vs_reuse_no_rope" in cmp:
            r = cmp["recompute_vs_reuse"]
            nr = cmp["recompute_vs_reuse_no_rope"]
            print(f"\n[rope vs no-rope] {c['task']} | {c['skill_name']} occ{c['occurrence']}: "
                  f"key {nr['mean_key_cksim']:.3f}->{r['mean_key_cksim']:.3f} "
                  f"value {nr['mean_value_cksim']:.3f}->{r['mean_value_cksim']:.3f} "
                  f"(re-rotation lifts KEY only; VALUE unchanged)")
        for r in rows:
            r.update(task=c["task"], skill_name=c["skill_name"],
                     occurrence=c["occurrence"], skill_tokens=c["skill_tokens"])
        all_rows.extend(rows)
        print(f"\n===== {c['task']} | {c['skill_name']} occ{c['occurrence']} (T={c['skill_tokens']}) =====")
        print(f"{'L':>3} {'Kcos':>7} {'Vcos':>7} {'dK':>6} {'dV':>6} "
              f"{'Ktop%':>6} {'Vtop%':>6} {'Kcos_noTop':>10}")
        for r in rows:
            if not show(r["layer"]):
                continue
            print(f"{r['layer']:>3} {r['key_cksim']:>7.4f} {r['value_cksim']:>7.4f} "
                  f"{r['key_drift_norm']:>6.2f} {r['value_drift_norm']:>6.2f} "
                  f"{r['key_topdim_energy']*100:>5.1f}% {r['value_topdim_energy']*100:>5.1f}% "
                  f"{r['key_cksim_drop_top2pct']:>10.4f}")

    fields = ["task", "skill_name", "occurrence", "skill_tokens", "layer",
              "key_cksim", "value_cksim", "key_drift_norm", "value_drift_norm",
              "key_topdim_energy", "value_topdim_energy", "key_cksim_drop_top2pct"]
    out = Path(args.csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r[k] for k in fields})
    print(f"\n[done] {len(all_rows)} rows -> {out}")


if __name__ == "__main__":
    main()
