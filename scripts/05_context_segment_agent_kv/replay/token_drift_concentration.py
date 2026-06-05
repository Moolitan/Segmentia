"""Step 3 / option B, first question: is the deep-V drift concentrated in a few
skill tokens, or spread evenly across all of them?

This decides whether "recompute only the high-drift tokens" (token selection) can
help at all in our setting:
  * concentrated  -> recomputing a few tokens fixes most of the drift -> the 2D
                     (token x depth) idea is worth building.
  * spread evenly -> every token must be recomputed -> token selection is useless
                     here and depth is all we have.

It reuses the real 14B setup from residual_checkpoint_probe_14b.py: rebuild the
exact token sequence, run occ1 (the cached source) and occ2 (the ground truth),
and compare the skill span's VALUE vectors PER TOKEN at deep layers. It does NOT
use vLLM.

Output:
  * per-token VALUE cosine distribution (reuse vs truth) at deep layers;
  * which tokens drift worst / least (decoded);
  * an idealized "recompute the worst X% tokens" recovery curve (upper bound that
    assumes a recomputed token becomes exact) to gauge how concentrated the drift
    is. Real recompute is not exact, so this is only a feasibility gauge.

Usage:
    python token_drift_concentration.py
    python token_drift_concentration.py --task internal_comms_incident_update \
        --skill internal-comms --pair 1-2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

PKG_ROOT = Path(__file__).resolve().parent.parent
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from core.config import SKILL_TOKEN_LOCATIONS  # noqa: E402
from core.hf_probe import (  # noqa: E402
    DEFAULT_MODEL, build_full_ids, skill_hidden, value_of,
)


def per_token_cos(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # a,b: [span, hidden] -> cosine per token (length = span)
    return F.cosine_similarity(a.float(), b.float(), dim=-1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--task", default="internal_comms_incident_update")
    ap.add_argument("--skill", default="internal-comms")
    ap.add_argument("--pair", default="1-2", help="src-dst occurrence pair")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    spans = SKILL_TOKEN_LOCATIONS[args.task]["skills"][args.skill]["token_spans"]
    full = build_full_ids(tok, args.task)
    src, dst = (int(x) for x in args.pair.split("-"))
    s, e = spans[dst - 1]
    span_len = e - s
    print(f"task={args.task} skill={args.skill} pair=occ{src}->occ{dst} "
          f"skill_tokens={span_len}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(device).eval()
    L = len(model.model.layers)
    full = full.to(device)
    deep = [L // 2, 3 * L // 4, L - 1]

    s_src, e_src = spans[src - 1]
    src_h = skill_hidden(model, full[:, :e_src], s_src, e_src)   # cached source
    truth_h = skill_hidden(model, full[:, :e], s, e)             # ground truth

    skill_ids = full[0, s:e].tolist()

    for l in deep:
        rv = value_of(model, l, src_h[l].to(device))
        tv = value_of(model, l, truth_h[l].to(device))
        cos = per_token_cos(rv, tv).cpu()  # [span_len]
        order = torch.argsort(cos)         # ascending: worst first

        print(f"\n===== layer {l} : per-token VALUE cosine (reuse vs truth) =====")
        print(f"  mean={cos.mean():.3f}  min={cos.min():.3f}  max={cos.max():.3f}  "
              f"std={cos.std():.3f}")
        frac_bad = (cos < 0.7).float().mean().item()
        frac_ok = (cos > 0.9).float().mean().item()
        print(f"  fraction of tokens cos<0.70 (badly drifted) = {frac_bad*100:.0f}%")
        print(f"  fraction of tokens cos>0.90 (basically fine) = {frac_ok*100:.0f}%")

        def show(idx_list, label):
            print(f"  {label}:")
            for i in idx_list:
                t = tok.decode([skill_ids[i]]).replace("\n", "\\n")
                print(f"     tok#{i:>3} cos={cos[i]:.3f}  {t!r}")
        show(order[:5].tolist(), "worst 5 tokens")
        show(order[-5:].tolist(), "best 5 tokens")

        # idealized "recompute worst X% tokens -> they become exact (cos=1)" curve
        sorted_cos, _ = torch.sort(cos)  # ascending
        print("  idealized recompute-worst-X%% -> resulting AVERAGE cosine:")
        for pct in (0, 10, 20, 30, 50):
            k = int(span_len * pct / 100)
            kept = sorted_cos[k:]  # the ones we did NOT recompute keep their cos
            avg = (kept.sum() + k * 1.0) / span_len
            print(f"     recompute worst {pct:>3d}% -> avg cos {avg:.3f}")

    print("\nReading: if a small fraction of tokens holds most of the drift (low "
          "'fraction fine', and the recompute-worst-X% curve jumps up fast), then "
          "token selection helps and the 2D (token x depth) idea is worth building. "
          "If drift is spread evenly (most tokens already bad, curve rises slowly), "
          "token selection cannot help and depth is the only lever.")


if __name__ == "__main__":
    main()
