"""Per-use reuse impact — direct KV cache injection via DynamicCache subclass.

For each skill in each task, config.py's invocation_indices lists exactly which
invocation JSON files contain a new copy of that skill.

  occ1  (invocation_indices[0]): prefill only.  A SkillCapturingCache intercepts
        DynamicCache.update() — which fires after RoPE — and saves the real K/V
        tensors for the skill span [s, e) at every layer.

  occ_k (invocation_indices[k-1], k>=2): two full greedy-decode runs:
          Run A (truth) — standard generate, no intervention.
          Run B (reuse) — SkillInjectingCache replaces K and V at [s, e) with
                          occ1's saved tensors inside update(), so the attention
                          and all subsequent decode steps use occ1's KV as-is.
        Each generation runs long enough (--max-new) to pass </think> into the
        actual deliverable, and is split into the thinking block vs the deliverable.
        BLEU-4, ROUGE-L, and embedding cosine (mean-pooled last hidden layer) are
        reported separately for the full output, the think block, and the
        deliverable — the deliverable is what actually matters for reuse quality.

No vLLM required. Outputs a CSV (metrics) and a JSONL (split text pairs).

Usage:
    python perturn_impact_from_invocations_14b.py --task internal_comms_incident_update
    python perturn_impact_from_invocations_14b.py            # all tasks
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
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

PKG_ROOT = Path(__file__).resolve().parent.parent
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from core.config import SKILL_TOKEN_LOCATIONS, TRACES_DIR  # noqa: E402
from core.hf_probe import DEFAULT_MODEL  # noqa: E402
from core.message_convert import convert_messages, convert_tools  # noqa: E402
from core.segments import find_skill_segments  # noqa: E402

OUT = Path("results/05_context_segment_agent_kv/DP3/perturn_invocation_impact.csv")
TEXT_OUT = Path("results/05_context_segment_agent_kv/DP3/perturn_invocation_texts.jsonl")

_rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
_smooth = SmoothingFunction().method1


def invocation_sort_key(path: Path) -> tuple[int, int]:
    stem = path.stem
    return (int(stem.split("turn_")[1].split("_")[0]), int(stem.split("inv_")[1]))


def rotate_half(x):
    """Standard RoPE rotate_half (matches transformers' apply_rotary_pos_emb)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


class SkillCapturingCache(DynamicCache):
    """DynamicCache that saves K/V for token span [s, e) at every layer.

    update() is called inside self_attn.forward() after RoPE is applied, so
    the captured tensors are the genuine post-RoPE K/V — exactly what would
    be stored in a real KV cache.
    """

    def __init__(self, s: int, e: int) -> None:
        super().__init__()
        self.s, self.e = s, e
        self.skill_kv: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if key_states.shape[-2] > 1:  # prefill pass (seq_len > 1)
            self.skill_kv[layer_idx] = (
                key_states[:, :, self.s:self.e, :].clone(),
                value_states[:, :, self.s:self.e, :].clone(),
            )
        return super().update(key_states, value_states, layer_idx, cache_kwargs)


class SkillInjectingCache(DynamicCache):
    """DynamicCache that replaces K/V at span [s, e) with occ1's saved tensors.

    During prefill the attention computation — and the KV stored for decode —
    both see occ1's K/V at the skill positions, faithfully simulating KV reuse.
    Decode steps (seq_len == 1) are not modified.

    The injected K must already be re-rotated to occ_k's position (matching the
    real SegKV system's rope.py re-rotation); V carries no position and is
    injected as captured. This isolates the *content* error from the *position*
    error, which re-rotation eliminates exactly.
    """

    def __init__(
        self,
        s: int,
        e: int,
        skill_kv: dict[int, tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        super().__init__()
        self.s, self.e = s, e
        self.skill_kv = skill_kv

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if key_states.shape[-2] > 1 and layer_idx in self.skill_kv:
            sk, sv = self.skill_kv[layer_idx]
            key_states = key_states.clone()
            value_states = value_states.clone()
            key_states[:, :, self.s:self.e, :] = sk.to(key_states.dtype)
            value_states[:, :, self.s:self.e, :] = sv.to(value_states.dtype)
        return super().update(key_states, value_states, layer_idx, cache_kwargs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--task", default=None)
    ap.add_argument("--max-new", type=int, default=2048)
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
        """Prefill occ1 and capture per-layer K/V at skill span [s, e) after RoPE."""
        cache = SkillCapturingCache(s, e)
        model(full_ids, past_key_values=cache, use_cache=True)
        return cache.skill_kv

    @torch.no_grad()
    def rerotate_k(skill_kv, delta):
        """Re-rotate occ1's captured (post-RoPE) K by a constant position offset
        delta = occ_k_start - occ1_start, so the injected K lands at occ_k's
        position. RoPE composes additively, so R(delta)·R(occ1) = R(occ_k).
        V is returned unchanged (it carries no position). delta==0 is a no-op.
        """
        if delta == 0:
            return skill_kv
        ref = next(iter(skill_kv.values()))[0]
        pos = torch.tensor([[delta]], device=device, dtype=torch.long)
        cos, sin = model.model.rotary_emb(ref, pos)   # [1, 1, head_dim]
        cos = cos.unsqueeze(1).float()                # [1, 1, 1, head_dim]
        sin = sin.unsqueeze(1).float()
        out = {}
        for li, (k, v) in skill_kv.items():
            kf = k.float()
            k_rot = kf * cos + rotate_half(kf) * sin
            out[li] = (k_rot.to(k.dtype), v)
        return out

    @torch.no_grad()
    def generate_truth(full_ids):
        """Standard greedy decode, no KV intervention."""
        T = full_ids.shape[1]
        gen = model.generate(full_ids, max_new_tokens=args.max_new, do_sample=False,
                              use_cache=True, pad_token_id=tok.eos_token_id)
        return gen[0, T:]

    @torch.no_grad()
    def generate_reuse(full_ids, s, e, skill_kv):
        """Greedy decode with occ1's K/V injected at skill span during prefill."""
        T = full_ids.shape[1]
        cache = SkillInjectingCache(s, e, skill_kv)
        gen = model.generate(full_ids, max_new_tokens=args.max_new, do_sample=False,
                              past_key_values=cache, use_cache=True,
                              pad_token_id=tok.eos_token_id)
        return gen[0, T:]

    @torch.no_grad()
    def embed_mean(text: str) -> torch.Tensor:
        ids = tok(text, return_tensors="pt").input_ids.to(device)
        out = model(ids, output_hidden_states=True, use_cache=False)
        return out.hidden_states[-1][0].mean(0).float()

    def split_think(text: str) -> tuple[str, str]:
        """Split a generation into (think, deliverable) at the </think> marker.

        If </think> is absent (generation ran out of budget mid-thinking), the
        whole text is the think part and the deliverable is empty.
        """
        marker = "</think>"
        idx = text.find(marker)
        if idx == -1:
            return text.strip(), ""
        return text[:idx].strip(), text[idx + len(marker):].strip()

    def compute_metrics(text_a: str, text_b: str):
        """BLEU-4 / ROUGE-L / embed cosine for a text pair. Returns None on empty."""
        if not text_a.strip() or not text_b.strip():
            return None, None, None
        ref_tok, hyp_tok = text_a.split(), text_b.split()
        bleu4 = sentence_bleu([ref_tok], hyp_tok, weights=(0.25,) * 4, smoothing_function=_smooth)
        rouge_l = _rouge.score(text_a, text_b)["rougeL"].fmeasure
        ea, eb = embed_mean(text_a), embed_mean(text_b)
        cos_sim = F.cosine_similarity(ea.unsqueeze(0), eb.unsqueeze(0)).item()
        return bleu4, rouge_l, cos_sim

    def _round(x):
        return round(x, 5) if x is not None else None

    tasks = [args.task] if args.task else list(SKILL_TOKEN_LOCATIONS)
    rows = []
    text_rows = []

    for task in tasks:
        print(f"\n## {task}")
        all_files = sorted((TRACES_DIR / task).glob("turn_*_inv_*.json"), key=invocation_sort_key)
        skill_configs = SKILL_TOKEN_LOCATIONS[task]["skills"]

        for skill, skill_cfg in skill_configs.items():
            inv_indices = skill_cfg.get("invocation_indices", [])
            if not inv_indices:
                print(f"   {skill}: no invocation_indices in config, skip")
                continue

            skill_kv_occ1 = None
            s_occ1 = None  # occ1's skill span start position (for K re-rotation)

            for occ, inv_idx in enumerate(inv_indices, start=1):
                f = all_files[inv_idx - 1]  # invocation_indices are 1-based
                inv = json.loads(f.read_text(encoding="utf-8"))
                msgs, _ = convert_messages(inv["messages"], system_prompt)
                segs = find_skill_segments(msgs)

                # The occ-th copy is the last occurrence in the cumulative history.
                skill_segs = [seg for seg in segs if seg[0] == skill]
                if not skill_segs:
                    print(f"   {skill} occ{occ}: not found in {f.stem}, skip")
                    continue
                _, midx, cs, ce = skill_segs[-1]

                full_ids = prompt_ids(msgs, add_gen=True).to(device)
                T = full_ids.shape[1]
                if T > args.max_prompt:
                    print(f"   {skill} occ{occ}: SKIP (prompt {T})")
                    continue

                s, e = span_of(msgs, midx, cs, ce)
                dist = T - e  # tokens from skill end to generation start

                if occ == 1:
                    skill_kv_occ1 = capture_occ1(full_ids, s, e)
                    s_occ1 = s
                    print(f"   {skill:22s} occ1 SOURCE @ {f.stem}  span_len={e-s} dist_to_gen={dist}")
                    continue

                if skill_kv_occ1 is None or s_occ1 is None:
                    print(f"   {skill} occ{occ}: no source captured, skip")
                    continue

                span_len_occ1 = next(iter(skill_kv_occ1.values()))[0].shape[-2]
                if span_len_occ1 != (e - s):
                    print(f"   {skill} occ{occ}: span len mismatch ({span_len_occ1} vs {e-s}), skip")
                    continue

                # Run A: truth decode (long enough to pass </think> into the deliverable).
                cont_a = generate_truth(full_ids)
                text_a = tok.decode(cont_a, skip_special_tokens=True)

                # Re-rotate occ1's K to occ_k's position (the SegKV position fix);
                # V is reused as-is. delta = occ_k start - occ1 start.
                delta = s - s_occ1
                skill_kv_reuse = rerotate_k(skill_kv_occ1, delta)

                # Run B: reuse decode — re-rotated occ1 K + occ1 V injected at skill span.
                cont_b = generate_reuse(full_ids, s, e, skill_kv_reuse)
                text_b = tok.decode(cont_b, skip_special_tokens=True)

                # Split each generation into thinking vs the actual deliverable.
                think_a, deliv_a = split_think(text_a)
                think_b, deliv_b = split_think(text_b)
                reached_a = bool(deliv_a)  # whether truth run actually produced a deliverable

                # Metrics on the full output, the think block, and the deliverable.
                f_bleu, f_rouge, f_cos = compute_metrics(text_a, text_b)
                t_bleu, t_rouge, t_cos = compute_metrics(think_a, think_b)
                d_bleu, d_rouge, d_cos = compute_metrics(deliv_a, deliv_b)

                deliv_str = (f"deliv[BLEU4={d_bleu:.3f} cos={d_cos:.4f}]"
                             if d_bleu is not None else "deliv[--none reached--]")
                print(f"   {skill:22s} occ{occ} REUSE @ {f.stem}  dist={dist}  "
                      f"full[BLEU4={f_bleu:.3f} cos={f_cos:.4f}] {deliv_str}  (prompt={T})")
                meta = {
                    "task": task, "skill": skill, "occ": occ,
                    "invocation": f.stem, "prompt_tokens": T,
                    "skill_len": e - s, "dist_to_gen": dist,
                    "reached_deliverable": int(reached_a),
                    "full_bleu4": _round(f_bleu), "full_rouge_l": _round(f_rouge), "full_cos": _round(f_cos),
                    "think_bleu4": _round(t_bleu), "think_rouge_l": _round(t_rouge), "think_cos": _round(t_cos),
                    "deliv_bleu4": _round(d_bleu), "deliv_rouge_l": _round(d_rouge), "deliv_cos": _round(d_cos),
                }
                rows.append(meta)
                # Keep the actual decoded text pair (split) for manual inspection of divergence.
                text_rows.append({
                    **meta,
                    "think_truth": think_a, "think_reuse": think_b,
                    "deliv_truth": deliv_a, "deliv_reuse": deliv_b,
                })
                torch.cuda.empty_cache()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[done] {len(rows)} reuse rows -> {OUT}")

    with TEXT_OUT.open("w", encoding="utf-8") as fp:
        for tr in text_rows:
            fp.write(json.dumps(tr, ensure_ascii=False) + "\n")
    print(f"[done] {len(text_rows)} text pairs -> {TEXT_OUT}")


if __name__ == "__main__":
    main()
