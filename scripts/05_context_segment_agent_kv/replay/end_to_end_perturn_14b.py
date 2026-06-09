"""CORRECTED end-to-end reuse impact: measure each reuse at ITS OWN turn.

Why this supersedes end_to_end_quality(_sweep)_14b.py and dp3_impact_labels_14b.py:
those used only the FINAL cumulative prompt and generated once at its very end, so
occ2/occ3 sat at fixed mid-positions and their measured impact depended on an
artificial distance to that single end. In a real multi-turn agent, each skill
copy is APPENDED and the model generates RIGHT AFTER it -- so when a skill is
actually used, it is near the generation frontier.

This script reconstructs, for each reused occurrence occ_k (k>=2), the turn where
it was just loaded: the prompt = conversation truncated at occ_k's tool message +
the generation prompt. occ_k then sits at the frontier (small dist_to_gen). We
measure: generate with occ_k computed fresh (truth) vs occ_k's KV reused from occ1
(drift its rows to occ1's activation at every layer), and compare the output along
the truth continuation (teacher-forced: mean KL + top-1 disagreement). occ1's
source activation is captured from the same prompt (occ1 appears earlier in it).

The internal KV-drift findings (CKSim notes) are unaffected by the earlier bug
(skill KV at occ_k depends only on its prefix, identical in cumulative vs per-turn);
only the OUTPUT-impact measurement is corrected here.

No vLLM. Outputs a CSV.

Usage:
    python end_to_end_perturn_14b.py --task internal_comms_incident_update
    python end_to_end_perturn_14b.py            # all tasks
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
from transformers import AutoModelForCausalLM, AutoTokenizer

PKG_ROOT = Path(__file__).resolve().parent.parent
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from core.config import SKILL_TOKEN_LOCATIONS, TRACES_DIR  # noqa: E402
from core.hf_probe import DEFAULT_MODEL, last_invocation_path, value_of  # noqa: E402
from core.message_convert import convert_messages, convert_tools  # noqa: E402
from core.segments import find_skill_segments  # noqa: E402

OUT = Path("results/05_context_segment_agent_kv/DP3/perturn_impact_labels.csv")


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
    L = len(model.model.layers)

    system_prompt = (TRACES_DIR / "_system_prompt.txt").read_text(encoding="utf-8")
    tools = convert_tools(json.loads((TRACES_DIR / "_tools.json").read_text(encoding="utf-8")))

    def tok_upto(msgs, midx, char_end, add_gen):
        m = copy.deepcopy(msgs[: midx + 1])
        m[midx]["content"] = m[midx]["content"][:char_end]
        return tok.apply_chat_template(m, tools=tools, add_generation_prompt=add_gen,
                                       tokenize=True, chat_template_kwargs={"enable_thinking": True})

    tasks = [args.task] if args.task else list(SKILL_TOKEN_LOCATIONS)
    rows = []
    for task in tasks:
        inv = json.loads(last_invocation_path(task).read_text(encoding="utf-8"))
        msgs, _ = convert_messages(inv["messages"], system_prompt)
        segs = find_skill_segments(msgs)  # (name, midx, cstart, cend) in order
        # group occurrences per skill
        per_skill: dict[str, list] = {}
        for name, midx, cs, ce in segs:
            per_skill.setdefault(name, []).append((midx, cs, ce))
        print(f"\n## {task}")

        for skill, occs in per_skill.items():
            if len(occs) < 2:
                continue
            o1_mid, o1_cs, o1_ce = occs[0]
            for k in range(1, len(occs)):  # occ_k, k index 1.. => occurrence k+1
                ok_mid, ok_cs, ok_ce = occs[k]
                # build occ_k's use-turn prompt (truncate conversation at occ_k's msg)
                full_ids = tok.apply_chat_template(
                    msgs[: ok_mid + 1], tools=tools, add_generation_prompt=True,
                    tokenize=True, return_tensors="pt",
                    chat_template_kwargs={"enable_thinking": True}).to(device)
                T = full_ids.shape[1]
                if T > args.max_prompt:
                    print(f"   {skill} occ{k+1}: SKIP (prompt {T})")
                    continue
                # spans within this prompt (add_gen=False positions are valid prefix positions)
                s1 = len(tok_upto(msgs, o1_mid, o1_cs, False)); e1 = len(tok_upto(msgs, o1_mid, o1_ce, False))
                sk = len(tok_upto(msgs, ok_mid, ok_cs, False)); ek = len(tok_upto(msgs, ok_mid, ok_ce, False))
                if (e1 - s1) != (ek - sk):
                    print(f"   {skill} occ{k+1}: SKIP (span len mismatch {e1-s1} vs {ek-sk})")
                    continue
                dist_to_gen = T - ek

                # capture occ1 source activation per layer (light: pre-hooks grab rows)
                store: dict[int, torch.Tensor] = {}
                caps = []

                def mk_cap(li):
                    def hook(m, a):
                        hs = a[0]
                        if hs.shape[1] > 1:
                            store[li] = hs[0, s1:e1, :].clone()
                        return None
                    return hook
                for li in range(L):
                    caps.append(model.model.layers[li].register_forward_pre_hook(mk_cap(li)))
                with torch.no_grad():
                    model(full_ids, use_cache=False, logits_to_keep=1)
                for h in caps:
                    h.remove()

                # truth continuation
                with torch.no_grad():
                    gen = model.generate(full_ids, max_new_tokens=args.max_new, do_sample=False,
                                         use_cache=True, pad_token_id=tok.eos_token_id)
                cont = gen[0, T:]
                G = len(cont)
                seq = torch.cat([full_ids, cont.unsqueeze(0)], dim=1)

                def pins_on():
                    hs_ = []

                    def mk(li):
                        def hook(m, a):
                            hs = a[0]
                            if hs.shape[1] <= 1:
                                return None
                            hs = hs.clone()
                            hs[0, sk:ek, :] = store[li].to(hs.dtype)
                            return (hs,) + tuple(a[1:])
                        return hook
                    for li in range(L):
                        hs_.append(model.model.layers[li].register_forward_pre_hook(mk(li)))
                    return hs_

                @torch.no_grad()
                def forced(pin):
                    hh = pins_on() if pin else []
                    try:
                        out = model(seq, use_cache=False, logits_to_keep=G + 1)
                    finally:
                        for h in hh:
                            h.remove()
                    return out.logits[0, :G, :].float()

                tl = forced(False)
                rl = forced(True)
                kl = F.kl_div(F.log_softmax(rl, -1), F.log_softmax(tl, -1),
                              log_target=True, reduction="none").sum(-1).mean().item()
                dis = (tl.argmax(-1) != rl.argmax(-1)).float().mean().item()
                print(f"   {skill:22s} occ{k+1}: dist_to_gen={dist_to_gen:>4}  "
                      f"KL={kl:.4f}  disagree={dis*100:.0f}%  (prompt={T})")
                rows.append({"task": task, "skill": skill, "occ": k + 1,
                             "prompt_tokens": T, "skill_len": ek - sk,
                             "dist_to_gen": dist_to_gen,
                             "impact_meanKL": round(kl, 5),
                             "impact_top1_disagree": round(dis, 5)})
                del store
                torch.cuda.empty_cache()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[done] {len(rows)} rows -> {OUT}")
    print("Reading: each reuse measured AT ITS OWN turn (skill near the frontier, "
          "small dist_to_gen). This is the faithful per-use output impact.")


if __name__ == "__main__":
    main()
