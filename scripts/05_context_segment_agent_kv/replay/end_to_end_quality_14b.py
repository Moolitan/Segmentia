"""Step 3 / option B follow-up: does the deep-V drift actually change the OUTPUT?

Everything so far measured INTERNAL similarity (K/V/activation cosine) -- an
indirect proxy. This script measures the DIRECT thing: what the model actually
generates, with KV reuse vs without. If the generated answer barely changes
despite deep-V cosine ~0.5, then reuse is "good enough" and SegKV's story is
"reuse works, here's the quality evidence". If the answer changes a lot, the
drift genuinely hurts and the (broad + deep) repair problem is real.

How "reuse" is simulated faithfully (no vLLM needed):
  A skill recurs at occ1, occ2, occ3 in one real conversation. In SegKV occ1 is
  computed fresh and occ2/occ3 REUSE occ1's cached KV. Reuse = the skill tokens
  are NOT recomputed; later tokens attend to KV derived from occ1's activations.
  We reproduce that by pinning the occ2/occ3 skill rows' activation at EVERY
  layer to occ1's cached activation during prefill (RoPE is then applied at the
  occ2/occ3 positions automatically, so K is re-rotated and V is copied -- exactly
  what vLLM's rope.py does). Then we let the model generate and compare to the
  truth run (no pinning).

Metrics:
  * greedy-decoded continuation: truth vs reuse, side by side;
  * first token position where they diverge, and overall token agreement;
  * KL divergence + top-1 agreement of the next-token distribution at the first
    generation step.

Usage:
    python end_to_end_quality_14b.py
    python end_to_end_quality_14b.py --task internal_comms_incident_update \
        --skill internal-comms --reuse-occ 2,3 --max-new 64
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
from residual_checkpoint_probe_14b import (  # noqa: E402
    DEFAULT_MODEL, build_full_ids, skill_hidden,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--task", default="internal_comms_incident_update")
    ap.add_argument("--skill", default="internal-comms")
    ap.add_argument("--src-occ", type=int, default=1, help="occurrence used as the cached source")
    ap.add_argument("--reuse-occ", default="2,3", help="occurrences whose KV is reused (drifted)")
    ap.add_argument("--max-new", type=int, default=64)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    spans = SKILL_TOKEN_LOCATIONS[args.task]["skills"][args.skill]["token_spans"]
    full = build_full_ids(tok, args.task)
    T = full.shape[1]
    src = args.src_occ
    reuse_occ = [int(x) for x in args.reuse_occ.split(",") if x.strip()]
    s_src, e_src = spans[src - 1]
    print(f"task={args.task} skill={args.skill} prompt_tokens={T}")
    print(f"source occ{src} span=[{s_src},{e_src}); reuse (drifted) occs={reuse_occ}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(device).eval()
    L = len(model.model.layers)
    full = full.to(device)

    # cached source: occ1 skill activation at every layer (input to each layer)
    src_h = skill_hidden(model, full[:, :e_src], s_src, e_src)  # len L+1, on CPU

    # pin occ2/occ3 skill rows to occ1's activation, but ONLY during prefill
    target_spans = [spans[o - 1] for o in reuse_occ]
    handles = []

    def make_hook(layer_idx: int):
        cached = src_h[layer_idx].to(device)

        def hook(module, args_in):
            hs = args_in[0]
            if hs.shape[1] <= 1:            # generation step: do nothing
                return None
            hs = hs.clone()
            for (s, e) in target_spans:     # pin reused skill copies to occ1
                hs[0, s:e, :] = cached.to(hs.dtype)
            return (hs,) + tuple(args_in[1:])
        return hook

    def add_hooks():
        for li in range(L):
            handles.append(model.model.layers[li].register_forward_pre_hook(make_hook(li)))

    def clear_hooks():
        while handles:
            handles.pop().remove()

    @torch.no_grad()
    def generate():
        out = model.generate(full, max_new_tokens=args.max_new, do_sample=False,
                             use_cache=True, pad_token_id=tok.eos_token_id)
        return out[0, T:]  # only the newly generated tokens

    @torch.no_grad()
    def first_step_logits():
        return model(full, use_cache=False).logits[0, -1, :].float()

    # ---- truth ----
    truth_logits = first_step_logits()
    truth_gen = generate()

    # ---- reuse ----
    add_hooks()
    try:
        reuse_logits = first_step_logits()
        reuse_gen = generate()
    finally:
        clear_hooks()

    # ---- compare next-token distribution at first generation step ----
    pt = F.log_softmax(truth_logits, -1)
    pr = F.log_softmax(reuse_logits, -1)
    kl = F.kl_div(pr, pt, log_target=True, reduction="sum").item()  # KL(truth||reuse)
    top1_same = int(truth_logits.argmax() == reuse_logits.argmax())
    tk_t = set(truth_logits.topk(5).indices.tolist())
    tk_r = set(reuse_logits.topk(5).indices.tolist())
    print("\n[first generation step] next-token distribution: truth vs reuse")
    print(f"  KL(truth||reuse) = {kl:.4f}   top-1 same = {bool(top1_same)}   "
          f"top-5 overlap = {len(tk_t & tk_r)}/5")
    print(f"  truth top-1 = {tok.decode([truth_logits.argmax()])!r}   "
          f"reuse top-1 = {tok.decode([reuse_logits.argmax()])!r}")

    # ---- butterfly-free metric: teacher-force the TRUTH continuation through
    #      both models and compare next-token distributions position by position.
    #      Feeding identical tokens to both removes greedy divergence, so this
    #      measures how differently the model BEHAVES, not just wording. ----
    @torch.no_grad()
    def forced_logits(use_hooks: bool):
        seq = torch.cat([full, truth_gen.unsqueeze(0)], dim=1)
        if use_hooks:
            add_hooks()
        try:
            lg = model(seq, use_cache=False).logits[0, T - 1: T - 1 + len(truth_gen), :].float()
        finally:
            if use_hooks:
                clear_hooks()
        return lg  # [gen_len, vocab], distribution that predicts each truth token

    tl = forced_logits(False)
    rl = forced_logits(True)
    kl_tf = F.kl_div(F.log_softmax(rl, -1), F.log_softmax(tl, -1),
                     log_target=True, reduction="none").sum(-1)  # per position
    top1_tf = (tl.argmax(-1) == rl.argmax(-1)).float()
    print(f"\n[teacher-forced along truth path] {len(truth_gen)} positions")
    print(f"  mean KL(truth||reuse) per token = {kl_tf.mean():.4f}  (max {kl_tf.max():.3f})")
    print(f"  same top-1 next token           = {top1_tf.mean()*100:.0f}%")

    # ---- compare greedy continuation ----
    n = min(len(truth_gen), len(reuse_gen))
    same = (truth_gen[:n] == reuse_gen[:n])
    agree = same.float().mean().item()
    diverge = int((~same).nonzero()[0].item()) if (~same).any() else n
    print(f"\n[greedy continuation] {n} tokens, token-level agreement = {agree*100:.0f}%, "
          f"first divergence at token #{diverge}")
    print("\n--- TRUTH ---")
    print(tok.decode(truth_gen, skip_special_tokens=True))
    print("\n--- REUSE ---")
    print(tok.decode(reuse_gen, skip_special_tokens=True))

    print("\nReading: high agreement / low KL => deep-V drift barely changes the "
          "OUTPUT (reuse is good enough; SegKV = 'reuse works' story). Low "
          "agreement / high KL / early divergence => the drift genuinely changes "
          "what the model says (the broad+deep repair problem is real).")


if __name__ == "__main__":
    main()
