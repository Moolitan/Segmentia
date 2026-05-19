"""
Shared utilities for the natural reference-block KV reuse experiments.

The runner implements the same three-way comparison used by the original
single-prompt scripts:
  A  = normal prefill of the repeated reference block
  B1 = naive reuse of the first block's KV cache
  B2 = reuse after RoPE re-rotation of K only
"""

from __future__ import annotations

import math
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache, DynamicLayer


@dataclass(frozen=True)
class PromptSpec:
    name: str
    reference_text: str
    user_context: str
    middle_history: str
    query_text: str
    note: str = ""

    @property
    def full_prompt(self) -> str:
        return (
            self.user_context
            + self.reference_text
            + self.middle_history
            + self.reference_text
            + self.query_text
        )


def _ngrams(tokens: list[int], n: int) -> list[tuple[int, ...]]:
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def bleu4(ref: list[int], hyp: list[int]) -> float:
    if not hyp:
        return 0.0
    bp = min(1.0, len(hyp) / max(len(ref), 1))
    log_sum = 0.0
    for n in range(1, 5):
        ref_ng = Counter(_ngrams(ref, n))
        hyp_ng = Counter(_ngrams(hyp, n))
        clipped = sum(min(cnt, ref_ng[g]) for g, cnt in hyp_ng.items())
        total = max(sum(hyp_ng.values()), 1)
        log_sum += math.log(clipped / total + 1e-9)
    return bp * math.exp(log_sum / 4)


def _lcs_len(a: list[int], b: list[int]) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = [0] * (len(b) + 1)
    for x in a:
        curr = [0] * (len(b) + 1)
        for j, y in enumerate(b, 1):
            curr[j] = prev[j - 1] + 1 if x == y else max(prev[j], curr[j - 1])
        prev = curr
    return prev[len(b)]


def rouge_l(ref: list[int], hyp: list[int]) -> float:
    if not ref or not hyp:
        return 0.0
    lcs = _lcs_len(ref, hyp)
    p = lcs / len(hyp)
    r = lcs / len(ref)
    return 2 * p * r / (p + r + 1e-9)


def _make_layer(k: torch.Tensor, v: torch.Tensor) -> DynamicLayer:
    layer = DynamicLayer()
    layer.dtype = k.dtype
    layer.device = k.device
    layer.keys = k
    layer.values = v
    layer.is_initialized = True
    return layer


def clone_cache(cache: DynamicCache) -> DynamicCache:
    new_cache = DynamicCache()
    for layer in cache.layers:
        new_cache.layers.append(_make_layer(layer.keys.clone(), layer.values.clone()))
    return new_cache


def slice_cache(cache: DynamicCache, tok_start: int, tok_end: int) -> DynamicCache:
    new_cache = DynamicCache()
    for layer in cache.layers:
        k = layer.keys[:, :, tok_start:tok_end, :].clone()
        v = layer.values[:, :, tok_start:tok_end, :].clone()
        new_cache.layers.append(_make_layer(k, v))
    return new_cache


def concat_cache(cache1: DynamicCache, cache2: DynamicCache) -> DynamicCache:
    assert len(cache1.layers) == len(cache2.layers)
    new_cache = DynamicCache()
    for l1, l2 in zip(cache1.layers, cache2.layers):
        k = torch.cat([l1.keys, l2.keys], dim=-2).contiguous()
        v = torch.cat([l1.values, l2.values], dim=-2).contiguous()
        new_cache.layers.append(_make_layer(k, v))
    return new_cache


def strip_per_step(metrics: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in metrics.items() if not k.endswith("_per_step")}


def resolve_model_path(model_path: str) -> str:
    path = Path(model_path).expanduser()
    if path.is_absolute() and not path.exists():
        raise FileNotFoundError(
            "Local model path does not exist: "
            f"{path}\n"
            "Pass the mounted checkpoint with --model-path, or use a valid "
            "Hugging Face repo id such as Qwen/Qwen3-14B."
        )
    return str(path) if path.exists() else model_path


class KVReuseRunner:
    def __init__(
        self,
        model_path: str,
        *,
        t_steps: int = 128,
        free_max_tokens: int = 256,
        dtype: torch.dtype = torch.bfloat16,
        device: str | None = None,
    ) -> None:
        self.model_path = model_path
        self.t_steps = t_steps
        self.free_max_tokens = free_max_tokens
        self.dtype = dtype
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        resolved_model_path = resolve_model_path(model_path)

        print(f"Loading tokenizer: {resolved_model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(resolved_model_path)
        if not self.tokenizer.is_fast:
            raise RuntimeError("A fast tokenizer is required for offset_mapping.")

        print(f"Loading model on {self.device} dtype={dtype} ...")
        t0 = time.time()
        self.model = AutoModelForCausalLM.from_pretrained(
            resolved_model_path,
            attn_implementation="eager",
            dtype=dtype,
            device_map=str(self.device),
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        self.rotary_emb = self.model.model.rotary_emb
        self.head_dim = self.model.config.hidden_size // self.model.config.num_attention_heads
        print(f"Model loaded in {time.time() - t0:.1f}s")

    def token_len(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def locate_reference_tokens(
        self,
        prompt_text: str,
        reference_text: str,
        occurrence: str,
    ) -> tuple[list[int], int, int]:
        if occurrence == "first":
            char_start = prompt_text.find(reference_text)
        elif occurrence == "last":
            char_start = prompt_text.rfind(reference_text)
        else:
            raise ValueError(occurrence)
        if char_start == -1:
            raise ValueError(f"Reference text not found: {occurrence}")
        char_end = char_start + len(reference_text)

        enc = self.tokenizer(prompt_text, add_special_tokens=False, return_offsets_mapping=True)
        ids = enc["input_ids"]
        offsets = enc["offset_mapping"]

        tok_start = tok_end = None
        for i, (s, e) in enumerate(offsets):
            if s >= char_start and tok_start is None:
                tok_start = i
            if e <= char_end:
                tok_end = i + 1
        if tok_start is None or tok_end is None or tok_end <= tok_start:
            raise RuntimeError(
                f"Failed to locate reference tokens: start={tok_start}, end={tok_end}"
            )
        return ids, tok_start, tok_end

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def _apply_rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        return x * cos + self._rotate_half(x) * sin

    def get_cos_sin(self, pos_start: int, length: int) -> tuple[torch.Tensor, torch.Tensor]:
        position_ids = torch.arange(
            pos_start, pos_start + length, device=self.device
        ).unsqueeze(0)
        dummy_x = torch.zeros(
            1, 1, length, self.head_dim, device=self.device, dtype=self.dtype
        )
        with torch.inference_mode():
            cos, sin = self.rotary_emb(dummy_x, position_ids)
        return cos.to(self.dtype), sin.to(self.dtype)

    def rerotate_k(
        self,
        k: torch.Tensor,
        pos_from: int,
        pos_to: int,
        length: int,
    ) -> torch.Tensor:
        orig_dtype = k.dtype
        k = k.float()
        cos_from, sin_from = self.get_cos_sin(pos_from, length)
        cos_to, sin_to = self.get_cos_sin(pos_to, length)
        k_unrot = self._apply_rope(k, cos_from.float(), -sin_from.float())
        k_rerot = self._apply_rope(k_unrot, cos_to.float(), sin_to.float())
        return k_rerot.to(orig_dtype)

    def make_b2_reference_cache(
        self,
        reference_v1_cache: DynamicCache,
        pos_from: int,
        pos_to: int,
        length: int,
    ) -> DynamicCache:
        new_cache = DynamicCache()
        for layer in reference_v1_cache.layers:
            k_new = self.rerotate_k(layer.keys.clone(), pos_from, pos_to, length)
            new_cache.layers.append(_make_layer(k_new.contiguous(), layer.values.clone()))
        return new_cache

    def model_forward(
        self,
        token_ids: list[int] | torch.Tensor,
        cache: DynamicCache,
    ) -> tuple[torch.Tensor, DynamicCache]:
        if isinstance(token_ids, list):
            input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        else:
            input_ids = token_ids if token_ids.ndim == 2 else token_ids.unsqueeze(0)
            input_ids = input_ids.to(self.device)

        past_len = cache.get_seq_length()
        seq_len = input_ids.shape[1]
        position_ids = torch.arange(
            past_len, past_len + seq_len, dtype=torch.long, device=self.device
        ).unsqueeze(0)

        with torch.inference_mode():
            out = self.model(
                input_ids=input_ids,
                position_ids=position_ids,
                past_key_values=cache,
                use_cache=True,
                output_attentions=False,
                output_hidden_states=False,
            )
        return out.logits[0, -1, :], out.past_key_values

    def _step_metrics(self, logits_a: torch.Tensor, logits_b: torch.Tensor) -> dict[str, Any]:
        p_a = F.softmax(logits_a.float(), dim=-1)
        p_b = F.softmax(logits_b.float(), dim=-1)
        kl = float(F.kl_div(p_b.clamp(min=1e-9).log(), p_a, reduction="sum").item())
        tvd = float(0.5 * (p_a - p_b).abs().sum().item())
        argmax_match = int(p_a.argmax().item()) == int(p_b.argmax().item())
        top5_a = set(p_a.topk(5).indices.tolist())
        top5_b = set(p_b.topk(5).indices.tolist())
        return {
            "kl": max(kl, 0.0),
            "tvd": tvd,
            "argmax_match": argmax_match,
            "top5_jaccard": len(top5_a & top5_b) / len(top5_a | top5_b),
        }

    def _finalize_distribution_metrics(
        self,
        label: str,
        per_step: list[dict[str, Any]],
    ) -> dict[str, Any]:
        kl = np.array([x["kl"] for x in per_step], dtype=float)
        tvd = np.array([x["tvd"] for x in per_step], dtype=float)
        match = np.array([x["argmax_match"] for x in per_step], dtype=float)
        jac = np.array([x["top5_jaccard"] for x in per_step], dtype=float)
        return {
            "label": label,
            "T": int(len(per_step)),
            "KL_first": float(kl[0]),
            "KL_mean": float(kl.mean()),
            "KL_max": float(kl.max()),
            "TVD_first": float(tvd[0]),
            "TVD_mean": float(tvd.mean()),
            "argmax_match_rate": float(match.mean()),
            "argmax_match_count": int(match.sum()),
            "top5_overlap_mean": float(jac.mean()),
            "kl_per_step": kl.tolist(),
            "tvd_per_step": tvd.tolist(),
            "argmax_match_per_step": match.tolist(),
            "top5_jaccard_per_step": jac.tolist(),
        }

    def teacher_forcing_metrics(
        self,
        logits_a_first: torch.Tensor,
        logits_b1_first: torch.Tensor,
        logits_b2_first: torch.Tensor,
        cache_a: DynamicCache,
        cache_b1: DynamicCache,
        cache_b2: DynamicCache,
    ) -> tuple[list[int], dict[str, Any], dict[str, Any]]:
        y_a = [int(logits_a_first.argmax().item())]
        b1_steps = [self._step_metrics(logits_a_first, logits_b1_first)]
        b2_steps = [self._step_metrics(logits_a_first, logits_b2_first)]

        dec_cache_a = clone_cache(cache_a)
        dec_cache_b1 = clone_cache(cache_b1)
        dec_cache_b2 = clone_cache(cache_b2)

        for _ in range(1, self.t_steps):
            teacher = torch.tensor([[y_a[-1]]], dtype=torch.long, device=self.device)
            logits_a, dec_cache_a = self.model_forward(teacher, dec_cache_a)
            logits_b1, dec_cache_b1 = self.model_forward(teacher, dec_cache_b1)
            logits_b2, dec_cache_b2 = self.model_forward(teacher, dec_cache_b2)
            b1_steps.append(self._step_metrics(logits_a, logits_b1))
            b2_steps.append(self._step_metrics(logits_a, logits_b2))
            y_a.append(int(logits_a.argmax().item()))

        return (
            y_a,
            self._finalize_distribution_metrics("B1_naive_reuse", b1_steps),
            self._finalize_distribution_metrics("B2_rope_corrected", b2_steps),
        )

    def free_run_decode(
        self,
        start_cache: DynamicCache,
        first_logits: torch.Tensor,
    ) -> list[int]:
        tokens: list[int] = []
        curr_cache = clone_cache(start_cache)
        next_logits = first_logits

        for _ in range(self.free_max_tokens):
            next_tok = int(next_logits.argmax().item())
            tokens.append(next_tok)
            if next_tok == self.tokenizer.eos_token_id:
                break
            inp = torch.tensor([[next_tok]], dtype=torch.long, device=self.device)
            next_logits, curr_cache = self.model_forward(inp, curr_cache)
        return tokens

    def embed_cosine(self, tokens_a: list[int], tokens_b: list[int]) -> float:
        if not tokens_a or not tokens_b:
            return 0.0
        with torch.inference_mode():
            ea = self.model.model.embed_tokens(
                torch.tensor(tokens_a, device=self.device)
            ).mean(dim=0)
            eb = self.model.model.embed_tokens(
                torch.tensor(tokens_b, device=self.device)
            ).mean(dim=0)
        return float(F.cosine_similarity(ea.unsqueeze(0), eb.unsqueeze(0)).item())

    def text_metrics(self, ref_tok: list[int], hyp_tok: list[int], label: str) -> dict[str, Any]:
        return {
            "label": label,
            "ref_len": len(ref_tok),
            "hyp_len": len(hyp_tok),
            "bleu4": bleu4(ref_tok, hyp_tok),
            "rouge_l": rouge_l(ref_tok, hyp_tok),
            "embed_cosine": self.embed_cosine(ref_tok, hyp_tok),
        }

    def run_prompt(self, spec: PromptSpec, *, scenario: str) -> dict[str, Any]:
        prompt = spec.full_prompt
        full_ids, ref1_start, ref1_end = self.locate_reference_tokens(
            prompt, spec.reference_text, "first"
        )
        _, ref2_start, ref2_end = self.locate_reference_tokens(
            prompt, spec.reference_text, "last"
        )

        n_full = len(full_ids)
        l_ref = ref1_end - ref1_start
        if l_ref != ref2_end - ref2_start:
            raise RuntimeError(f"{spec.name}: reference token lengths differ.")
        if ref1_end > ref2_start:
            raise RuntimeError(f"{spec.name}: repeated reference blocks overlap.")
        if ref2_end >= n_full:
            raise RuntimeError(f"{spec.name}: query/assistant prefix after ref2 is required.")

        context_end = ref1_start
        middle_start = ref1_end
        middle_end = ref2_start
        phase1_ids = full_ids[:middle_end]
        ref2_ids = full_ids[middle_end:ref2_end]
        query_ids = full_ids[ref2_end:]

        print(
            f"\n[{scenario}/{spec.name}] N={n_full} L={l_ref} "
            f"middle={middle_end - middle_start} "
            f"start_gap={ref2_start - ref1_start} query={len(query_ids)}"
        )

        cache_phase1 = DynamicCache()
        _, cache_phase1 = self.model_forward(phase1_ids, cache_phase1)
        ref_v1_cache = slice_cache(cache_phase1, ref1_start, ref1_end)

        cache_a = clone_cache(cache_phase1)
        _, cache_a = self.model_forward(ref2_ids, cache_a)
        logits_a_first, cache_a = self.model_forward(query_ids, cache_a)

        cache_b1_pre = concat_cache(cache_phase1, ref_v1_cache)
        if cache_b1_pre.get_seq_length() != ref2_end:
            raise RuntimeError(f"{spec.name}: B1 cache length mismatch.")
        logits_b1_first, cache_b1 = self.model_forward(query_ids, cache_b1_pre)

        ref_v1_rerot = self.make_b2_reference_cache(
            ref_v1_cache, ref1_start, ref2_start, l_ref
        )
        cache_b2_pre = concat_cache(cache_phase1, ref_v1_rerot)
        logits_b2_first, cache_b2 = self.model_forward(query_ids, cache_b2_pre)

        if not (
            cache_a.get_seq_length()
            == cache_b1.get_seq_length()
            == cache_b2.get_seq_length()
            == n_full
        ):
            raise RuntimeError(f"{spec.name}: cache lengths are not aligned.")

        y_a, metrics_b1, metrics_b2 = self.teacher_forcing_metrics(
            logits_a_first,
            logits_b1_first,
            logits_b2_first,
            cache_a,
            cache_b1,
            cache_b2,
        )

        free_run: dict[str, str] = {}
        text_metrics: dict[str, Any] = {}
        if self.free_max_tokens > 0:
            free_a = self.free_run_decode(cache_a, logits_a_first)
            free_b1 = self.free_run_decode(cache_b1, logits_b1_first)
            free_b2 = self.free_run_decode(cache_b2, logits_b2_first)
            free_run = {
                "A_text": self.tokenizer.decode(free_a, skip_special_tokens=True),
                "B1_text": self.tokenizer.decode(free_b1, skip_special_tokens=True),
                "B2_text": self.tokenizer.decode(free_b2, skip_special_tokens=True),
            }
            text_metrics = {
                "A_vs_B1": self.text_metrics(free_a, free_b1, "A_vs_B1"),
                "A_vs_B2": self.text_metrics(free_a, free_b2, "A_vs_B2"),
            }

        print(
            f"  B1 KL_mean={metrics_b1['KL_mean']:.5f} "
            f"argmax={metrics_b1['argmax_match_rate']:.3f}; "
            f"B2 KL_mean={metrics_b2['KL_mean']:.5f} "
            f"argmax={metrics_b2['argmax_match_rate']:.3f}"
        )

        return {
            "scenario": scenario,
            "case_name": spec.name,
            "model": self.model_path,
            "T_steps": self.t_steps,
            "free_max_tokens": self.free_max_tokens,
            "sequence_info": {
                "N_full": n_full,
                "context": [0, context_end],
                "reference_v1": [ref1_start, ref1_end],
                "middle": [middle_start, middle_end],
                "reference_v2": [ref2_start, ref2_end],
                "L_reference": l_ref,
                "middle_tokens_between_references": middle_end - middle_start,
                "reference_start_position_gap": ref2_start - ref1_start,
                "query_tokens_after_reference_v2": len(query_ids),
                "note": spec.note,
            },
            "y_A": y_a,
            "y_A_text": self.tokenizer.decode(y_a),
            "free_run": free_run,
            "text_metrics": text_metrics,
            "B1": strip_per_step(metrics_b1),
            "B2": strip_per_step(metrics_b2),
            "B1_per_step": {
                k: metrics_b1[k]
                for k in (
                    "kl_per_step",
                    "tvd_per_step",
                    "argmax_match_per_step",
                    "top5_jaccard_per_step",
                )
            },
            "B2_per_step": {
                k: metrics_b2[k]
                for k in (
                    "kl_per_step",
                    "tvd_per_step",
                    "argmax_match_per_step",
                    "top5_jaccard_per_step",
                )
            },
        }


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def mean_std(values: list[float]) -> dict[str, float]:
    arr = np.array(values, dtype=float)
    if len(arr) == 0:
        return {"mean": 0.0, "std": 0.0}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
    }
