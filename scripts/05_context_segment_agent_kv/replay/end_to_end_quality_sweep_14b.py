"""Generalize the end-to-end output-quality test across ALL tasks/skills.

end_to_end_quality_14b.py answered "does the deep-V drift change the OUTPUT?" for
one skill (internal-comms): teacher-forced top-1 agreement was 96%, mean KL 0.067,
i.e. reuse barely changes behavior. This script checks whether that holds across
every task in the trace set, reusing (drifting) ALL repeated skill occurrences in
each task -- the faithful full-reuse scenario.

Faithful reuse simulation (no vLLM): for every skill, occ1 is the fresh source and
occ2..occN reuse it. We pin the occ2.. skill rows' activation at every layer to
occ1's, during prefill only (RoPE then re-applies at the new positions, so K is
re-rotated and V copied -- exactly what vLLM rope.py does).

Butterfly-free metric: greedy-decode a truth continuation, then teacher-force that
same token path through both the truth model and the reuse model and compare the
next-token distributions position by position (mean KL, top-1 agreement). Feeding
identical tokens removes greedy divergence, so this measures behavior, not wording.

Memory: uses logits_to_keep so the big (~25k-token) prompts don't materialize a
full [seq x vocab] logits tensor.

Usage:
    python end_to_end_quality_sweep_14b.py
    python end_to_end_quality_sweep_14b.py --max-new 128 --max-prompt 26000
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
from residual_checkpoint_probe_14b import DEFAULT_MODEL, build_full_ids  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--max-prompt", type=int, default=26000,
                    help="skip tasks whose full prompt exceeds this (memory guard)")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(device).eval()
    L = len(model.model.layers)

    rows = []
    for task in SKILL_TOKEN_LOCATIONS:
        full = build_full_ids(tok, task).to(device)
        T = full.shape[1]
        if T > args.max_prompt:
            print(f"\n## SKIP {task} (prompt {T} > {args.max_prompt})")
            continue
        skills = SKILL_TOKEN_LOCATIONS[task]["skills"]
        # occ1 source spans and the reuse (occ>=2) target spans, across all skills
        src_spans = {sk: rec["token_spans"][0] for sk, rec in skills.items()}
        tgt_spans = [s for rec in skills.values() for s in rec["token_spans"][1:]]
        print(f"\n## {task}  prompt={T}  skills={list(skills)}  reused_copies={len(tgt_spans)}")

        # ---- capture occ1 source activation per layer (read-only hooks) ----
        store: dict[int, torch.Tensor] = {}
        cap_handles = []

        def make_cap(li):
            def hook(module, args_in):
                hs = args_in[0]
                if hs.shape[1] <= 1:
                    return None
                store[li] = torch.cat([hs[0, s:e, :] for (s, e) in src_spans.values()], dim=0).clone()
                return None
            return hook
        for li in range(L):
            cap_handles.append(model.model.layers[li].register_forward_pre_hook(make_cap(li)))
        with torch.no_grad():
            model(full, use_cache=False, logits_to_keep=1)
        for h in cap_handles:
            h.remove()

        # map: within the concatenated occ1 store, which rows belong to which skill
        src_lens = [e - s for (s, e) in src_spans.values()]
        offs, acc = {}, 0
        for sk, ln in zip(src_spans, src_lens):
            offs[sk] = (acc, acc + ln); acc += ln
        # per reuse target, the source rows to pin it to (match by skill)
        pin_plan = []  # (tgt_start, tgt_end, src_off_start, src_off_end)
        for sk, rec in skills.items():
            a, b = offs[sk]
            for (s, e) in rec["token_spans"][1:]:
                pin_plan.append((s, e, a, b))

        # ---- pin hooks (reuse): overwrite occ>=2 rows with occ1 source, prefill only ----
        pin_handles = []

        def make_pin(li):
            src = store[li]

            def hook(module, args_in):
                hs = args_in[0]
                if hs.shape[1] <= 1:
                    return None
                hs = hs.clone()
                for (ts, te, sa, sb) in pin_plan:
                    hs[0, ts:te, :] = src[sa:sb, :].to(hs.dtype)
                return (hs,) + tuple(args_in[1:])
            return hook

        def add_pins():
            for li in range(L):
                pin_handles.append(model.model.layers[li].register_forward_pre_hook(make_pin(li)))

        def clear_pins():
            while pin_handles:
                pin_handles.pop().remove()

        # ---- truth continuation (greedy, no pins) ----
        with torch.no_grad():
            gen = model.generate(full, max_new_tokens=args.max_new, do_sample=False,
                                 use_cache=True, pad_token_id=tok.eos_token_id)
        cont = gen[0, T:]
        G = len(cont)

        # ---- teacher-force the truth path through both models ----
        seq = torch.cat([full, cont.unsqueeze(0)], dim=1)

        @torch.no_grad()
        def kept_logits(pins: bool):
            if pins:
                add_pins()
            try:
                out = model(seq, use_cache=False, logits_to_keep=G + 1)
            finally:
                if pins:
                    clear_pins()
            return out.logits[0, :G, :].float()  # predicts cont[0..G-1]

        tl = kept_logits(False)
        rl = kept_logits(True)
        kl = F.kl_div(F.log_softmax(rl, -1), F.log_softmax(tl, -1),
                      log_target=True, reduction="none").sum(-1)
        top1 = (tl.argmax(-1) == rl.argmax(-1)).float().mean().item()
        print(f"   teacher-forced over {G} tokens: top-1 agreement={top1*100:.0f}%  "
              f"mean KL={kl.mean():.4f}  maxKL={kl.max():.2f}")
        rows.append((task, T, len(tgt_spans), top1, kl.mean().item()))
        del store
        torch.cuda.empty_cache()

    print("\n================ SUMMARY ================")
    print(f"{'task':32s} {'prompt':>7} {'reused':>7} {'top1%':>6} {'meanKL':>7}")
    for task, T, nt, top1, kl in rows:
        print(f"{task:32s} {T:>7} {nt:>7} {top1*100:>5.0f}% {kl:>7.4f}")
    if rows:
        import statistics as st
        print(f"\nacross {len(rows)} tasks: top-1 agreement "
              f"min={min(r[3] for r in rows)*100:.0f}% mean={st.mean(r[3] for r in rows)*100:.0f}%")
    print("\nReading: high top-1 agreement + low mean KL across tasks => reuse "
          "barely changes the OUTPUT broadly (not just for one skill) => 'simple "
          "training-free reuse is good enough end-to-end' holds.")


if __name__ == "__main__":
    main()
