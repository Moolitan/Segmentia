"""Step 3 / option A: re-run the residual-checkpoint probe on the REAL 14B setting.

The small-model probe (residual_checkpoint_probe.py) showed the mechanism works,
but on a mild example where plain reuse already had V cosine ~0.85-0.95. The real
14B trace had deep VALUE cosine ~0.5 (a much larger gap). This script repeats the
exact same test on Qwen3-14B with the REAL agent contexts, so we learn whether
"resume from a cached shallow activation" still recovers the deep VALUE when the
reuse gap is actually large.

Setup, in plain terms:
  * One skill recurs several times in one task's full conversation, at increasing
    positions (occ1, occ2, ...), each preceded by DIFFERENT real conversation.
  * occ1 run   : forward the conversation up to and including occ1's skill copy;
                 cache the skill rows' activation h_l at every layer.
  * occ2 truth : forward up to and including occ2's skill copy; this is the
                 ground-truth deep V for the skill under its real (occ2) context.
  * reuse      : use occ1's activations as if they were occ2's (today's SegKV).
  * resume@N   : redo the occ2 forward but overwrite the skill rows' INPUT to
                 layer N with occ1's cached h_N, then let layers N..end run while
                 attending to the real occ2 context. Measure deep V recovery.

This reconstructs the exact token sequence the CKSim harness used (same chat
template, tools, enable_thinking, empty system prefix), so the skill spans in
core.config line up. It does NOT use vLLM.

Usage:
    python residual_checkpoint_probe_14b.py
    python residual_checkpoint_probe_14b.py --task internal_comms_incident_update \
        --skill internal-comms --pairs 1-2,1-3 --boundaries 4,8,12,16,20,24
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

PKG_ROOT = Path(__file__).resolve().parent.parent
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from core.config import SKILL_TOKEN_LOCATIONS, TRACES_DIR  # noqa: E402
from core.message_convert import convert_messages, convert_tools  # noqa: E402

DEFAULT_MODEL = "/mnt/Large_Language_Model_Lab_1/llm_models/Qwen3-14B/Qwen/Qwen3-14B"


def last_invocation_path(task: str) -> Path:
    files = sorted(
        (TRACES_DIR / task).glob("turn_*_inv_*.json"),
        key=lambda p: (int(p.stem.split("turn_")[1].split("_")[0]),
                       int(p.stem.split("inv_")[1])),
    )
    if not files:
        raise FileNotFoundError(f"no invocation files for {task}")
    return files[-1]


def build_full_ids(tok, task: str):
    system_prompt = (TRACES_DIR / "_system_prompt.txt").read_text(encoding="utf-8")
    tools = convert_tools(json.loads((TRACES_DIR / "_tools.json").read_text(encoding="utf-8")))
    inv = json.loads(last_invocation_path(task).read_text(encoding="utf-8"))
    msgs, _ = convert_messages(inv["messages"], system_prompt)
    ids = tok.apply_chat_template(
        msgs, tools=tools, add_generation_prompt=False,
        tokenize=True, return_tensors="pt",
        chat_template_kwargs={"enable_thinking": True},
    )
    return ids  # [1, T]


def cos_rows(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.float(), b.float(), dim=-1).mean().item())


@torch.no_grad()
def value_of(model, layer_idx: int, h_rows: torch.Tensor) -> torch.Tensor:
    layer = model.model.layers[layer_idx]
    return layer.self_attn.v_proj(layer.input_layernorm(h_rows))


@torch.no_grad()
def skill_hidden(model, ids: torch.Tensor, start: int, end: int) -> list[torch.Tensor]:
    """Return per-layer skill-span activation (input to each layer), on CPU."""
    out = model(ids, output_hidden_states=True, use_cache=False)
    hs = [h[0, start:end, :].to("cpu") for h in out.hidden_states]  # len L+1
    del out
    torch.cuda.empty_cache()
    return hs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--task", default="internal_comms_incident_update")
    ap.add_argument("--skill", default="internal-comms")
    ap.add_argument("--pairs", default="1-2,1-3", help="src-dst occurrence pairs")
    ap.add_argument("--boundaries", default="4,8,12,16,20,24")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    spans = SKILL_TOKEN_LOCATIONS[args.task]["skills"][args.skill]["token_spans"]
    full = build_full_ids(tok, args.task)
    T = full.shape[1]
    print(f"task={args.task} skill={args.skill} full_prompt_tokens={T}")
    print(f"config spans (occ1..): {spans}")

    # ---- alignment check: the token ids at each occurrence span must be equal ----
    seg0 = full[0, spans[0][0]:spans[0][1]]
    for i, (s, e) in enumerate(spans, 1):
        seg = full[0, s:e]
        ok = (seg.shape == seg0.shape) and bool((seg == seg0).all())
        print(f"  occ{i} span=[{s},{e}) len={e-s} matches_occ1={ok}")
        if not ok:
            raise SystemExit("span misalignment: rebuilt tokenization != config spans")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16).to(device).eval()
    L = len(model.model.layers)
    full = full.to(device)
    boundaries = [int(x) for x in args.boundaries.split(",") if x.strip()]
    deep = [L // 2, 3 * L // 4, L - 1]

    # cache per occurrence: skill activations for every layer
    occ_hidden: dict[int, list[torch.Tensor]] = {}
    for occ, (s, e) in enumerate(spans, 1):
        occ_hidden[occ] = skill_hidden(model, full[:, :e], s, e)
        print(f"[forward] occ{occ} done (context up to {e} tokens)")

    for pair in args.pairs.split(","):  # 默认 "1-2,1-3"
        # 1-2：用 occ1 的缓存去替代 occ2（位置偏移中等）
        # 1-3：用 occ1 的缓存去替代 occ3（位置偏移更大，更难）
        src, dst = (int(x) for x in pair.split("-"))
        s_dst, e_dst = spans[dst - 1]
        truth = occ_hidden[dst]
        cached = occ_hidden[src]

        # reuse baseline: src activations vs dst truth, VALUE cosine at deep layers
        reuse_v = {l: cos_rows(value_of(model, l, cached[l].to(device)),
                               value_of(model, l, truth[l].to(device))) for l in deep}
        print(f"\n===== pair occ{src}->occ{dst}  (skill at [{s_dst},{e_dst})) =====")
        print(f"{'N':>5} {'saved':>6} " + " ".join(f"{'Vcos@'+str(l):>9}" for l in deep)
              + f" {'meanRecov':>10}")
        print(f"{'reuse':>5} {'0%':>6} " + " ".join(f"{reuse_v[l]:>9.4f}" for l in deep)
              + f" {'-':>10}")

        for N in boundaries:
            if N >= L:
                continue
            cached_hN = cached[N].to(device)

            def pre_hook(module, args_in, _h=cached_hN):
                hs = args_in[0].clone()
                hs[0, s_dst:e_dst, :] = _h.to(hs.dtype)
                return (hs,) + tuple(args_in[1:])

            handle = model.model.layers[N].register_forward_pre_hook(pre_hook)
            try:
                res = skill_hidden(model, full[:, :e_dst], s_dst, e_dst)
            finally:
                handle.remove()
            # only layers strictly deeper than the injection point are actually
            # "resumed"; output_hidden_states[l] for l <= N is captured before the
            # injection (== truth), so exclude those from recovery (mark n/a).
            vcos = {l: (cos_rows(value_of(model, l, res[l].to(device)),
                                 value_of(model, l, truth[l].to(device))) if l > N else None)
                    for l in deep}
            valid = [l for l in deep if l > N]
            recov = (sum((vcos[l] - reuse_v[l]) / (1 - reuse_v[l] + 1e-9) for l in valid)
                     / len(valid)) if valid else float("nan")
            cells = " ".join((f"{vcos[l]:>9.4f}" if vcos[l] is not None else f"{'n/a':>9}")
                             for l in deep)
            print(f"{N:>5} {N/L*100:>5.0f}% " + cells + f" {recov*100:>9.0f}%")

    print("\nReading: VALUE cosine of the skill span vs full-recompute truth, at "
          "deep layers. 'reuse' = plain KV reuse (the large ~0.5 gap on 14B). "
          "'saved' = fraction of per-skill-token layers skipped. 'meanRecov' = "
          "how much of the reuse->truth VALUE gap the resume closes.")


if __name__ == "__main__":
    main()
