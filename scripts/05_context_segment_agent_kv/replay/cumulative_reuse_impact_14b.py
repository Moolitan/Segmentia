"""Cumulative reuse impact — does cross-position KV-reuse error ACCUMULATE?

doc (四)'s perturn_impact experiment is OPEN-LOOP and occurrence-ISOLATED: each
occ_k is measured on the clean recorded history, in isolation, single-shot. It
cannot see the real system's CUMULATIVE behaviour, where occ2's reused decode
pollutes the history that occ3 (and every later turn) conditions on.

This script measures the "history-pollution" channel of that accumulation,
agent-free, by teacher-forcing the trajectory (recorded tokens, no live tools,
no branching) and isolating the effect of UPSTREAM reuses:

  For each skill occurrence occ_k (k>=2) in its real request (invocation_k):
    * fresh      — no KV injection anywhere (reference).
    * isolated   — inject canonical occ1 KV only at occ_k's span (= doc 四).
    * cumulative — inject canonical occ1 KV at occ2..occ_k (every upstream reuse
                   the real system would have done by this turn).
  All injected K is re-rotated to its own occurrence's position (the SegKV
  position fix); V is reused as-is.

  At occ_k's generation frontier we teacher-force the FRESH greedy continuation
  (a fixed yardstick, so the trajectory is held constant — channel 2 excluded)
  and measure prediction divergence vs fresh for isolated and cumulative:
    mean KL + top-1 disagreement.
  accum = cumulative - isolated = the extra divergence contributed purely by
  UPSTREAM reuses. If accum grows with the number of upstream reuses, the error
  accumulates; if accum ~ 0, the agent self-regrounds each turn.

NOTE: the traces here have <=3 occurrences per skill, so the upstream count maxes
at 1 (occ3 has occ2 upstream). This gives the first accumulation data point, not
a long growth curve — skills recurring 4+ times are needed for the full curve.

No vLLM. Outputs a CSV.

Usage:
    python cumulative_reuse_impact_14b.py --task slack_launch_pack
    python cumulative_reuse_impact_14b.py            # all tasks
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

PKG_ROOT = Path(__file__).resolve().parent.parent
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from core.config import SKILL_TOKEN_LOCATIONS, TRACES_DIR  # noqa: E402
from core.hf_probe import DEFAULT_MODEL  # noqa: E402
from core.message_convert import convert_messages, convert_tools  # noqa: E402
from core.segments import find_skill_segments  # noqa: E402

OUT = Path("results/05_context_segment_agent_kv/DP3/cumulative_reuse_impact.csv")


def invocation_sort_key(path: Path) -> tuple[int, int]:
    stem = path.stem
    return (int(stem.split("turn_")[1].split("_")[0]), int(stem.split("inv_")[1]))


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


class SkillCapturingCache(DynamicCache):
    """Capture post-RoPE K/V for one span [s, e) at every layer (occ1 source)."""

    def __init__(self, s: int, e: int) -> None:
        super().__init__()
        self.s, self.e, self.skill_kv = s, e, {}

    def update(self, k, v, layer_idx, cache_kwargs=None):
        if k.shape[-2] > 1:
            self.skill_kv[layer_idx] = (k[:, :, self.s:self.e, :].clone(),
                                        v[:, :, self.s:self.e, :].clone())
        return super().update(k, v, layer_idx, cache_kwargs)


class MultiSpanInjectingCache(DynamicCache):
    """Inject pre-built (already re-rotated) K/V at several spans during prefill.

    spans: list of (s, e, kv_by_layer) where kv_by_layer[layer_idx] = (K, V),
    each K already re-rotated to that span's own position.
    """

    def __init__(self, spans) -> None:
        super().__init__()
        self.spans = spans

    def update(self, k, v, layer_idx, cache_kwargs=None):
        if k.shape[-2] > 1 and self.spans:
            k = k.clone()
            v = v.clone()
            for s, e, kvl in self.spans:
                sk, sv = kvl[layer_idx]
                k[:, :, s:e, :] = sk.to(k.dtype)
                v[:, :, s:e, :] = sv.to(v.dtype)
        return super().update(k, v, layer_idx, cache_kwargs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--task", default=None)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--max-prompt", type=int, default=26000)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(device).eval()
    system_prompt = (TRACES_DIR / "_system_prompt.txt").read_text(encoding="utf-8")
    tools = convert_tools(json.loads((TRACES_DIR / "_tools.json").read_text(encoding="utf-8")))

    def prompt_ids(msgs, add_gen):
        return tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=add_gen,
                                       tokenize=True, return_tensors="pt",
                                       chat_template_kwargs={"enable_thinking": True})

    def span_of(msgs, midx, cs, ce):
        def upto(char):
            m = copy.deepcopy(msgs[: midx + 1])
            m[midx]["content"] = m[midx]["content"][:char]
            return len(tok.apply_chat_template(m, tools=tools, add_generation_prompt=False,
                                               tokenize=True, chat_template_kwargs={"enable_thinking": True}))
        return upto(cs), upto(ce)

    @torch.no_grad()
    def capture_occ1(full_ids, s, e):
        cache = SkillCapturingCache(s, e)
        model(full_ids, past_key_values=cache, use_cache=True)
        return cache.skill_kv

    @torch.no_grad()
    def rerotate_k(skill_kv, delta):
        """Re-rotate occ1 K by position offset delta; V unchanged. delta==0 no-op."""
        if delta == 0:
            return skill_kv
        ref = next(iter(skill_kv.values()))[0]
        pos = torch.tensor([[delta]], device=device, dtype=torch.long)
        cos, sin = model.model.rotary_emb(ref, pos)
        cos, sin = cos.unsqueeze(1).float(), sin.unsqueeze(1).float()
        return {li: ((k.float() * cos + rotate_half(k.float()) * sin).to(k.dtype), v)
                for li, (k, v) in skill_kv.items()}

    @torch.no_grad()
    def fresh_continuation(full_ids):
        T = full_ids.shape[1]
        gen = model.generate(full_ids, max_new_tokens=args.max_new, do_sample=False,
                             use_cache=True, pad_token_id=tok.eos_token_id)
        return gen[0, T:]

    @torch.no_grad()
    def forced_logits(seq, spans, G):
        cache = MultiSpanInjectingCache(spans) if spans else DynamicCache()
        out = model(seq, past_key_values=cache, use_cache=True, logits_to_keep=G + 1)
        return out.logits[0, :G, :].float()

    def divergence(inj_logits, fresh_logits):
        kl = F.kl_div(F.log_softmax(inj_logits, -1), F.log_softmax(fresh_logits, -1),
                      log_target=True, reduction="none").sum(-1).mean().item()
        dis = (inj_logits.argmax(-1) != fresh_logits.argmax(-1)).float().mean().item()
        return kl, dis

    tasks = [args.task] if args.task else list(SKILL_TOKEN_LOCATIONS)
    rows = []

    for task in tasks:
        print(f"\n## {task}")
        all_files = sorted((TRACES_DIR / task).glob("turn_*_inv_*.json"), key=invocation_sort_key)
        for skill, cfg in SKILL_TOKEN_LOCATIONS[task]["skills"].items():
            inv_indices = cfg.get("invocation_indices", [])
            if len(inv_indices) < 2:
                continue

            # occ1 canonical source
            f1 = all_files[inv_indices[0] - 1]
            msgs1, _ = convert_messages(json.loads(f1.read_text(encoding="utf-8"))["messages"], system_prompt)
            seg1 = [s for s in find_skill_segments(msgs1) if s[0] == skill][-1]
            ids1 = prompt_ids(msgs1, add_gen=True).to(device)
            s1, e1 = span_of(msgs1, seg1[1], seg1[2], seg1[3])
            canon = capture_occ1(ids1, s1, e1)
            canon_len = e1 - s1
            print(f"   {skill:22s} occ1 SOURCE @ {f1.stem}  span_len={canon_len}")

            for k, inv_idx in enumerate(inv_indices[1:], start=2):
                f = all_files[inv_idx - 1]
                msgs, _ = convert_messages(json.loads(f.read_text(encoding="utf-8"))["messages"], system_prompt)
                occ_segs = [s for s in find_skill_segments(msgs) if s[0] == skill]
                if len(occ_segs) < k:
                    print(f"   {skill} occ{k}: only {len(occ_segs)} occ in {f.stem}, skip")
                    continue

                full_ids = prompt_ids(msgs, add_gen=True).to(device)
                T = full_ids.shape[1]
                if T > args.max_prompt:
                    print(f"   {skill} occ{k}: SKIP (prompt {T})")
                    continue

                # token spans of occ1..occ_k in THIS request
                spans_tok = [span_of(msgs, sg[1], sg[2], sg[3]) for sg in occ_segs[:k]]
                if any((b - a) != canon_len for a, b in spans_tok):
                    print(f"   {skill} occ{k}: span len mismatch, skip")
                    continue

                # build injected (re-rotated) canonical KV for each occurrence span
                injected = [(a, b, rerotate_k(canon, a - s1)) for a, b in spans_tok]
                span_isolated = [injected[-1]]            # only occ_k
                span_cumulative = injected[1:]            # occ2..occ_k (occ1=identity)

                # fixed reference continuation (fresh greedy), trajectory held constant
                cont = fresh_continuation(full_ids)
                G = len(cont)
                seq = torch.cat([full_ids, cont.unsqueeze(0)], dim=1)

                lg_fresh = forced_logits(seq, [], G)
                lg_iso = forced_logits(seq, span_isolated, G)
                lg_cum = forced_logits(seq, span_cumulative, G)

                kl_iso, dis_iso = divergence(lg_iso, lg_fresh)
                kl_cum, dis_cum = divergence(lg_cum, lg_fresh)
                n_up = k - 2  # upstream reuses injected beyond occ_k itself (occ2..occ_{k-1})

                print(f"   {skill:22s} occ{k} @ {f.stem}  n_upstream={n_up}  "
                      f"iso[KL={kl_iso:.4f} dis={dis_iso*100:.0f}%]  "
                      f"cum[KL={kl_cum:.4f} dis={dis_cum*100:.0f}%]  "
                      f"accumKL={kl_cum-kl_iso:+.4f}")
                rows.append({
                    "task": task, "skill": skill, "occ": k, "n_upstream": n_up,
                    "invocation": f.stem, "prompt_tokens": T, "skill_len": canon_len,
                    "iso_kl": round(kl_iso, 5), "iso_disagree": round(dis_iso, 5),
                    "cum_kl": round(kl_cum, 5), "cum_disagree": round(dis_cum, 5),
                    "accum_kl": round(kl_cum - kl_iso, 5),
                    "accum_disagree": round(dis_cum - dis_iso, 5),
                })
                torch.cuda.empty_cache()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[done] {len(rows)} rows -> {OUT}")


if __name__ == "__main__":
    main()
