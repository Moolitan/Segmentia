"""Quality-vs-cost Pareto for KV-reuse repair strategies, on one fair axis.

Earlier notes compared depth-partial recompute (measured) against an IDEALIZED
token-selective curve (which assumed recomputed tokens become exact) -- not a fair
comparison. This script puts every strategy on the SAME measured axis using one
unified mechanism: a pin mask over the skill's (token x layer) grid.

For each skill cell (token i, layer l):
  * REUSED  -> the cell's activation is pinned to occ1's cached value (drifted);
  * FREE    -> the cell is recomputed (flows normally, attending the real occ2
               context and its neighbors, exactly like real partial recompute).

Strategies are just different masks:
  * full-reuse     : all reused                       (cost 0)
  * depth@N        : reuse layers 0..N, recompute N+1..L-1 (residual checkpoint)
  * token@X%       : recompute the worst X% tokens at ALL layers, reuse the rest
                     (CacheBlend / CacheSlide-WCA style; worst chosen by a cheap
                     shallow-layer deviation -- see also the predictor below)
  * 2D@(X%,N)      : recompute only (worst X% tokens AND layer>N)
  * full-recompute : none reused                      (cost 1)

cost    = fraction of (token x layer) cells recomputed (= FLOP fraction).
quality = mean skill VALUE cosine vs full-recompute truth, at deep layers.

Token selection uses layer-`sel-layer` per-token deviation ||occ1 - truth|| as the
ranking signal -- the same kind of cheap shallow signal CacheBlend selects on, and
a candidate "impact predictor" (we also report how well it ranks the truly-worst
deep tokens).

It does NOT use vLLM.

Usage:
    python pareto_recompute_14b.py --task internal_comms_incident_update \
        --skill internal-comms --dst-occ 2
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
from residual_checkpoint_probe_14b import DEFAULT_MODEL, build_full_ids, value_of  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--task", default="internal_comms_incident_update")
    ap.add_argument("--skill", default="internal-comms")
    ap.add_argument("--src-occ", type=int, default=1)
    ap.add_argument("--dst-occ", type=int, default=2)
    ap.add_argument("--sel-layer", type=int, default=2, help="shallow layer whose per-token deviation ranks tokens")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    spans = SKILL_TOKEN_LOCATIONS[args.task]["skills"][args.skill]["token_spans"]
    full = build_full_ids(tok, args.task)
    s_src, e_src = spans[args.src_occ - 1]
    s, e = spans[args.dst_occ - 1]
    span = e - s
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(device).eval()
    L = len(model.model.layers)
    full = full.to(device)
    deep = [L // 2, 3 * L // 4, L - 1]
    print(f"task={args.task} skill={args.skill} occ{args.src_occ}->occ{args.dst_occ} "
          f"span={span} layers={L}")

    @torch.no_grad()
    def run_capture(ids, start, end, pin_per_layer=None):
        """Forward; capture skill-span layer inputs (post-pin). Returns list[L+1]."""
        cap = {}
        handles = []

        def mk(li):
            def hook(module, args_in):
                hs = args_in[0]
                if hs.shape[1] <= 1:
                    return None
                if pin_per_layer is not None and li in pin_per_layer:
                    rows, src = pin_per_layer[li]
                    hs = hs.clone()
                    hs[0, start:end][rows] = src.to(hs.dtype)
                    cap[li] = hs[0, start:end].clone()
                    return (hs,) + tuple(args_in[1:])
                cap[li] = hs[0, start:end].clone()
                return None
            return hook
        for li in range(L):
            handles.append(model.model.layers[li].register_forward_pre_hook(mk(li)))
        try:
            model(ids, use_cache=False, logits_to_keep=1)
        finally:
            for h in handles:
                h.remove()
        return cap  # {layer_idx: [span, hidden]}

    # truth (occ2 fresh) and source (occ1) skill activations per layer
    truth = run_capture(full[:, :e], s, e)
    src = run_capture(full[:, :e_src], s_src, e_src)
    truth_V = {l: value_of(model, l, truth[l].to(device)) for l in deep}

    def vq(cap):
        return sum(F.cosine_similarity(value_of(model, l, cap[l].to(device)).float(),
                                       truth_V[l].float(), dim=-1).mean().item()
                   for l in deep) / len(deep)

    # cheap ranking signal: per-token deviation at a shallow layer
    dev = (src[args.sel_layer].to(device) - truth[args.sel_layer].to(device)).norm(dim=-1)
    worst_order = torch.argsort(dev, descending=True)  # worst tokens first

    # truth per-token deep-V drift (to score the predictor): mean over deep layers
    deep_drift = torch.zeros(span)
    for l in deep:
        c = F.cosine_similarity(value_of(model, l, src[l].to(device)).float(),
                                truth_V[l].float(), dim=-1).cpu()
        deep_drift += (1 - c)
    true_worst = torch.argsort(deep_drift, descending=True)

    def pin_all(rows):  # reuse given rows at every layer
        r = torch.tensor(rows, dtype=torch.long)
        return {li: (r, src[li][r].to(device)) for li in range(L)}

    def build(strategy):
        if strategy == "full-reuse":
            return pin_all(list(range(span))), 0
        if strategy == "full-recompute":
            return {}, span * L
        if strategy.startswith("depth@"):
            N = int(strategy.split("@")[1])
            r = torch.arange(span)
            pin = {li: (r, src[li][r].to(device)) for li in range(0, N + 1)}
            free = span * (L - (N + 1))
            return pin, free
        if strategy.startswith("token@") or strategy.startswith("tokenOracle@"):
            X = int(strategy.split("@")[1].rstrip("%"))
            k = round(span * X / 100)
            ranking = true_worst if strategy.startswith("tokenOracle@") else worst_order
            free_tok = set(ranking[:k].tolist())
            reuse_rows = torch.tensor([i for i in range(span) if i not in free_tok], dtype=torch.long)
            pin = {li: (reuse_rows, src[li][reuse_rows].to(device)) for li in range(L)}
            return pin, k * L
        if strategy.startswith("2D@"):
            X, N = strategy.split("@")[1].split(",")
            k = round(span * int(X.rstrip("%")) / 100); N = int(N)
            free_tok = set(worst_order[:k].tolist())
            pin = {}
            for li in range(L):
                if li <= N:
                    rows = torch.arange(span)               # reuse all shallow
                else:
                    rows = torch.tensor([i for i in range(span) if i not in free_tok], dtype=torch.long)
                pin[li] = (rows, src[li][rows].to(device))
            free = k * (L - (N + 1))
            return pin, free
        raise ValueError(strategy)

    strategies = (["full-reuse"]
                  + [f"depth@{n}" for n in (4, 8, 12, 16, 20)]
                  + [f"token@{x}%" for x in (10, 20, 30, 50)]
                  + [f"tokenOracle@{x}%" for x in (10, 20, 30, 50)]
                  + ["2D@30%,8", "2D@50%,8"]
                  + ["full-recompute"])

    print(f"\n{'strategy':>16} {'cost':>6} {'deepV_cos':>10}")
    rows_out = []
    for st in strategies:
        pin, free = build(st)
        cap = run_capture(full[:, :e], s, e, pin_per_layer=pin if pin else None)
        q = 1.0 if st == "full-recompute" else vq(cap)
        cost = free / (span * L)
        family = ("anchor" if st in ("full-reuse", "full-recompute")
                  else st.split("@")[0])
        rows_out.append((st, family, cost, q))
        print(f"{st:>16} {cost*100:>5.0f}% {q:>10.4f}")

    import csv
    out_dir = Path("results/05_context_segment_agent_kv/Pareto")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"pareto_{args.task}_{args.skill}_occ{args.src_occ}-{args.dst_occ}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["strategy", "family", "cost", "deepV_cos"])
        w.writerows(rows_out)
    print(f"[done] csv -> {csv_path}")

    # ---- predictor check: does the cheap shallow signal pick the right tokens? ----
    for k_pct in (10, 20, 30):
        k = round(span * k_pct / 100)
        pred = set(worst_order[:k].tolist())
        true = set(true_worst[:k].tolist())
        overlap = len(pred & true) / max(1, k)
        print(f"[predictor] shallow(layer{args.sel_layer}) dev picks worst {k_pct}% tokens: "
              f"overlap with truly-worst deep tokens = {overlap*100:.0f}%")

    print("\nReading: compare strategies at equal cost -- whichever gives higher "
          "deepV_cos is the better repair axis for this setting. The predictor line "
          "says whether a cheap shallow signal (no truth needed) can pick which "
          "tokens actually matter deep.")


if __name__ == "__main__":
    main()
