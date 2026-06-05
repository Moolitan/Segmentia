"""Step 3 probe: does "resume from a cached shallow activation" recover deep KV?

Background (Step 2 conclusion): K and V are both linear projections of the same
residual-stream activation h_l, so "recompute V only" saves no compute. The only
route that *could* save compute is depth-partial recompute with a residual
checkpoint:

    reuse shallow KV (near-lossless) + cache the skill's activation h_N at a
    boundary layer N, then RESUME the forward pass from layer N so the deep
    layers attend to the REAL preceding context, recomputing only layers N..end.

This route hinges on ONE fact that cannot be read from the saved .pt KV (those
only store K/V, not the residual stream h_l): if we inject a shallow checkpoint
h_N that was computed under a DIFFERENT prefix (P1) and let the deep layers run
while attending to the true prefix (P2), do the deep activations snap back to
the true (full-recompute-under-P2) activations?

This script answers that directly on a small Qwen3 (mechanism is architecture-
general; model size is irrelevant to whether the mechanism works). It does NOT
use vLLM. It compares, per layer, the skill span's activation under three paths:

    truth   : full forward of  P2 + skill              (ground truth)
    reuse   : full forward of  P1 + skill              (what plain reuse gives)
    resume@N: forward of P2 + skill, but the skill rows' INPUT to layer N is
              overwritten with the cached h_N from the P1 run, then layers
              N..end run normally (attending to the real P2)

We report cosine(path, truth) for the skill span at deep layers. If resume@N is
close to truth while reuse is far, the residual-checkpoint route is viable; the
largest N that still recovers tells us how much compute can be skipped.

Usage:
    python residual_checkpoint_probe.py
    python residual_checkpoint_probe.py --model /path/to/Qwen3-1.7B --boundaries 2,4,6,8,12
"""
from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "/mnt/Large_Language_Model_Lab_1/llm_models/Qwen3-1.7B"

# A reusable "skill"-like block (kept identical across both contexts).
SKILL = (
    "When drafting an internal incident update, lead with severity and customer "
    "impact, then give a one-line status, the current owner, the next checkpoint "
    "time, and a short list of mitigations already in flight. Keep it factual, "
    "avoid speculation, and never bury the user-facing impact below internal "
    "detail. End with a single explicit ask or 'no action needed'."
)
# Two DIFFERENT preceding contexts (different topic AND different length) so the
# skill lands at a different absolute position with different prior content.
PREFIX_1 = "User: hi.\nAssistant: Hello! How can I help today?\n"
PREFIX_2 = (
    "User: Our checkout service started returning 503s about twenty minutes ago "
    "and the on-call paging just fired. Marketing also has a launch email going "
    "out in an hour and is asking whether to hold it. Finance pinged about the "
    "revenue dashboard looking flat. Can you help me write the update?\n"
    "Assistant: Yes. Let me pull the current status and the runbook before we "
    "draft anything for the broader channel.\n"
)


def skill_span(tok, prefix: str, skill: str) -> tuple[torch.Tensor, int, int]:
    p_ids = tok(prefix, return_tensors="pt", add_special_tokens=True).input_ids
    full_ids = tok(prefix + skill, return_tensors="pt", add_special_tokens=True).input_ids
    start = p_ids.shape[1]
    end = full_ids.shape[1]
    return full_ids, start, end


@torch.no_grad()
def hidden_states(model, ids: torch.Tensor) -> tuple[torch.Tensor, ...]:
    # tuple length L+1; hs[l] = INPUT to layer l (hs[0] = embeddings)
    out = model(ids, output_hidden_states=True, use_cache=False)
    return out.hidden_states


def cos_span(a: torch.Tensor, b: torch.Tensor) -> float:
    # a,b: [span, hidden]; cosine per token over hidden, averaged
    return float(F.cosine_similarity(a.float(), b.float(), dim=-1).mean().item())


@torch.no_grad()
def value_proj(model, layer_idx: int, h_rows: torch.Tensor) -> torch.Tensor:
    """The actual VALUE vectors for these rows at this layer.

    Qwen3 value path = v_proj(input_layernorm(h)); V carries no RoPE and no norm,
    so it is fully determined by the layer-input activation. This is exactly the
    'V' whose deep-layer drift Step 2 found to be the real problem (~0.5 cosine).
    """
    layer = model.model.layers[layer_idx]
    return layer.self_attn.v_proj(layer.input_layernorm(h_rows))


def main() -> None:
    ap = argparse.ArgumentParser(description="Residual-checkpoint resume viability probe.")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--boundaries", default="2,4,6,8,12,16",
                    help="comma list of layer N to resume from")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device).eval()
    layers = model.model.layers
    L = len(layers)
    print(f"model={args.model} layers={L} device={device}")

    ids1, s1, e1 = skill_span(tok, PREFIX_1, SKILL)
    ids2, s2, e2 = skill_span(tok, PREFIX_2, SKILL)
    ids1, ids2 = ids1.to(device), ids2.to(device)
    assert e1 - s1 == e2 - s2, "skill tokenization must match across prefixes"
    span_len = e1 - s1
    print(f"skill tokens={span_len}  P1 pos=[{s1},{e1})  P2 pos=[{s2},{e2})  "
          f"position shift={s2 - s1}")

    hs1 = hidden_states(model, ids1)              # under P1 (the cached run)
    hs_truth = hidden_states(model, ids2)         # under P2 (ground truth)

    def skill1(l):  # cached skill activation at layer-l input, from P1 run
        return hs1[l][0, s1:e1, :]

    def skill_truth(l): 
        return hs_truth[l][0, s2:e2, :]

    # deep layers at which we judge VALUE recovery (the metric that matters)
    deep = [l for l in (L // 2, 3 * L // 4, L - 1)]

    def v_cos(hs_path, l):
        # VALUE cosine of skill span: path vs truth, at layer l
        vt = value_proj(model, l, hs_truth[l][0, s2:e2, :])
        vp = value_proj(model, l, hs_path[l][0, (s2 if hs_path is not hs1 else s1):(e2 if hs_path is not hs1 else e1), :])
        return cos_span(vp, vt)

    # ---- baseline: plain reuse (use P1 activations as if they were P2's) ----
    print("\n[reuse vs truth] skill-span similarity, P1-run vs P2-truth")
    print(f"{'layer':>5} {'h_cos':>7} {'V_cos':>7}")
    for l in range(0, L, max(1, L // 8)):
        print(f"{l:>5} {cos_span(skill1(l), skill_truth(l)):>7.4f} {v_cos(hs1, l):>7.4f}")
    final_reuse = cos_span(skill1(L), skill_truth(L))
    # reuse VALUE cosine at the deep layers (this is the ~0.5 problem from Step 2)
    reuse_vdeep = {l: v_cos(hs1, l) for l in deep}

    # ---- resume@N: inject cached h_N into the P2 run at layer N's input ----
    boundaries = [int(x) for x in args.boundaries.split(",") if x.strip()]
    print("\n[resume@N] inject cached h_N (from P1) into the P2 forward at layer N,")
    print("           let layers N..end recompute while attending to real P2.")
    dcols = " ".join(f"{'Vcos@'+str(l):>9}" for l in deep)
    print(f"{'N':>4} {'saved':>6} {dcols} {'meanVrecov':>11}")
    print(f"{'reuse':>4} {'0%':>6} " +
          " ".join(f"{reuse_vdeep[l]:>9.4f}" for l in deep) + f" {'-':>11}")
    cached_hN = None

    def pre_hook(module, args_in):
        # overwrite the skill-span INPUT rows with the cached checkpoint
        hs = args_in[0].clone()
        hs[0, s2:e2, :] = cached_hN.to(hs.dtype) # 把skill位置的激活替换成P1缓存
        return (hs,) + tuple(args_in[1:])

    for N in boundaries:
        if N >= L:
            continue
        cached_hN = skill1(N)  # the checkpoint we'd have stored from occ1
        h = layers[N].register_forward_pre_hook(pre_hook)
        try:
            hs_resume = hidden_states(model, ids2)
        finally:
            h.remove()
        saved = N / L  # we skip layers 0..N-1 for the skill tokens
        vcos = {l: v_cos(hs_resume, l) for l in deep}
        recov = [(vcos[l] - reuse_vdeep[l]) / (1.0 - reuse_vdeep[l] + 1e-9) for l in deep]
        meanrec = sum(recov) / len(recov)
        print(f"{N:>4} {saved*100:>5.0f}% " +
              " ".join(f"{vcos[l]:>9.4f}" for l in deep) + f" {meanrec*100:>10.0f}%")

    print("\nReading: VALUE cosine (skill span) vs full-recompute truth, at deep "
          "layers. 'reuse' row = plain KV reuse (Step 2's ~0.5 problem). 'saved' = "
          "fraction of per-skill-token layers skipped. 'meanVrecov' = how much of "
          "the reuse->truth VALUE gap the resume closes. High recovery at large N "
          "=> depth+checkpoint fixes the VALUE drift AND saves real compute.")


if __name__ == "__main__":
    main()
