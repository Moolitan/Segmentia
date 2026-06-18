"""Evaluate decode outputs and optional CKSim KV dumps.

This scorer now treats three things as first-class, on top of the original
text-similarity metrics:

  (a) Reasoning fidelity. Qwen3 emits a hidden chain-of-thought in
      ``reasoning_content``. The harness saves it as ``reasoning``; we score the
      reasoning stream, the visible deliverable, and the full sequence
      (reasoning + deliverable) separately, so "the text looks similar" can no
      longer hide a divergent hidden plan.

  (b) Action-level / trajectory consistency. We extract the ordered tuple of
      tool-call names from each response and compare modality (tool vs text),
      tool set, and exact trajectory against recompute. These are reported as
      primary rates, not as anecdotes.

  (c) Stable effect vs sampling noise. With multiple samples per (case, mode)
      we measure each mode's own action self-consistency (how often it repeats
      its majority action). The recompute self-consistency is the sampling
      noise floor: a reuse divergence only counts as systematic if it is larger
      than that floor.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from config import (  # noqa: E402
    DEFAULT_CKSIM_KV_DIR,
    DEFAULT_METRICS_CSV,
    DEFAULT_METRICS_JSON,
    DEFAULT_MODEL_PATH,
    DEFAULT_OUTPUT_JSONL,
    DEFAULT_STABILITY_CSV,
)

REUSE_MODES = ("direct", "rope", "vrep", "krep", "oracle")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def pair_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["task"],
        row["skill"],
        row["occurrence"],
        row["invocation_index"],
    )


def sample_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (*pair_key(row), row["mode"], int(row.get("sample_index", 0)))


def keep_latest_sample_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        latest[sample_key(row)] = row
    return list(latest.values())


def split_think(text: str) -> tuple[str, str]:
    marker = "</think>"
    idx = text.find(marker)
    if idx == -1:
        return text.strip(), ""
    return text[:idx].strip(), text[idx + len(marker) :].strip()


def get_streams(row: dict[str, Any]) -> tuple[str, str, str]:
    """Return (reasoning, deliverable, full) for one response row.

    New rows carry ``reasoning`` (hidden CoT) and ``content`` (visible) fields.
    Legacy rows only have ``text``, possibly with an inline ``<think>`` block,
    which we split for backward compatibility.
    """
    text = row.get("text", "") or ""
    reasoning = row.get("reasoning")
    if reasoning is None:
        think, deliv = split_think(text)
        return think, deliv, text.strip()
    deliverable = (row.get("content") or text) or ""
    full = (reasoning + "\n" + deliverable).strip()
    return reasoning.strip(), deliverable.strip(), full


def extract_action(row: dict[str, Any]) -> dict[str, Any]:
    """Canonical action signature of a response: the ordered tool-call names."""
    tool_calls = row.get("tool_calls") or []
    names: list[str] = []
    for tc in tool_calls:
        fn = (tc or {}).get("function") or {}
        name = fn.get("name")
        if name:
            names.append(name)
    if names:
        return {"modality": "tool", "trajectory": tuple(names)}
    return {"modality": "text", "trajectory": ()}


def action_label(row: dict[str, Any]) -> str:
    act = extract_action(row)
    if act["modality"] == "text":
        return "text"
    return "tool:" + "+".join(act["trajectory"])


def action_match(ref: dict[str, Any], hyp: dict[str, Any]) -> dict[str, bool]:
    a, b = extract_action(ref), extract_action(hyp)
    return {
        "modality_match": a["modality"] == b["modality"],
        "tool_set_match": set(a["trajectory"]) == set(b["trajectory"]),
        "trajectory_match": a["trajectory"] == b["trajectory"],
    }


def text_metrics(ref: str, hyp: str, embedder) -> dict[str, float | None]:
    if not ref.strip() or not hyp.strip():
        return {"bleu4": None, "rouge_l": None, "embedding_cos": None}
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    from rouge_score import rouge_scorer

    ref_tok = ref.split()
    hyp_tok = hyp.split()
    bleu4 = sentence_bleu(
        [ref_tok],
        hyp_tok,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=SmoothingFunction().method1,
    )
    rouge_l = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False).score(
        ref, hyp
    )["rougeL"].fmeasure
    emb_cos = embedder.cosine(ref, hyp) if embedder is not None else None
    return {
        "bleu4": round(float(bleu4), 6),
        "rouge_l": round(float(rouge_l), 6),
        "embedding_cos": round(float(emb_cos), 6) if emb_cos is not None else None,
    }


class MeanHiddenEmbedder:
    def __init__(self, model_path: str, device: str | None = None) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = (
            AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
            .to(self.device)
            .eval()
        )

    def embed(self, text: str):
        ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
        with self.torch.no_grad():
            out = self.model(ids, output_hidden_states=True, use_cache=False)
        return out.hidden_states[-1][0].mean(0).float()

    def cosine(self, text_a: str, text_b: str) -> float:
        import torch.nn.functional as F

        emb_a = self.embed(text_a)
        emb_b = self.embed(text_b)
        return float(F.cosine_similarity(emb_a.unsqueeze(0), emb_b.unsqueeze(0)).item())


def load_entry(kv_dir: Path, cache_id: str) -> dict[str, Any]:
    import torch

    path = kv_dir / f"{cache_id}.pt"
    return torch.load(path, map_location="cpu", weights_only=False)


def as_heads(tensor):
    if tensor.dim() != 3:
        raise ValueError(f"expected [tokens, heads, dim], got {tuple(tensor.shape)}")
    return tensor.permute(1, 0, 2).contiguous()


def cksim_pair(a, b, tokens: int) -> tuple[float, float]:
    import torch.nn.functional as F

    ah = as_heads(a[:tokens]).float()
    bh = as_heads(b[:tokens]).float()
    if ah.shape != bh.shape:
        raise ValueError(f"KV shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
    head_scores = F.cosine_similarity(ah.flatten(1), bh.flatten(1), dim=1)
    token_scores = F.cosine_similarity(ah, bh, dim=2)
    return float(head_scores.mean().item()), float(token_scores.mean().item())


def compute_cksim(kv_dir: Path, ref_cache_id: str, other_cache_id: str, tokens: int):
    if not ref_cache_id or not other_cache_id:
        return None
    try:
        ref = load_entry(kv_dir, ref_cache_id)
        other = load_entry(kv_dir, other_cache_id)
    except FileNotFoundError:
        return None
    rows = []
    layers = sorted(set(ref["kv_by_layer"]) & set(other["kv_by_layer"]))
    for layer in layers:
        ref_k, ref_v = ref["kv_by_layer"][layer]
        other_k, other_v = other["kv_by_layer"][layer]
        key_head, key_token = cksim_pair(ref_k, other_k, tokens)
        value_head, value_token = cksim_pair(ref_v, other_v, tokens)
        rows.append(
            {
                "layer": layer,
                "key_cksim": key_head,
                "value_cksim": value_head,
                "key_token_mean": key_token,
                "value_token_mean": value_token,
            }
        )
    if not rows:
        return None
    return {
        "layers": len(rows),
        "mean_key_cksim": round(sum(r["key_cksim"] for r in rows) / len(rows), 6),
        "mean_value_cksim": round(sum(r["value_cksim"] for r in rows) / len(rows), 6),
    }


def first_cksim_id(samples: list[dict[str, Any]]) -> str | None:
    for s in samples:
        cid = s.get("cksim_cache_id")
        if cid:
            return cid
    return None


def by_sample_index(samples: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(s.get("sample_index", 0)): s for s in samples}


def mean_or_none(values: list[float | None]) -> float | None:
    present = [float(v) for v in values if v is not None]
    if not present:
        return None
    return round(sum(present) / len(present), 6)


def rate(flags: list[bool]) -> float | None:
    if not flags:
        return None
    return round(sum(1 for f in flags if f) / len(flags), 6)


def self_consistency(samples: list[dict[str, Any]]) -> tuple[str | None, float | None, dict[str, int]]:
    labels = [action_label(s) for s in samples if not s.get("error")]
    if not labels:
        return None, None, {}
    counts = Counter(labels)
    majority, top = counts.most_common(1)[0]
    return majority, round(top / len(labels), 6), dict(counts)


def stability_rows(grouped: dict[tuple[Any, ...], dict[str, list[dict[str, Any]]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, by_mode in grouped.items():
        for mode, samples in by_mode.items():
            majority, consistency, dist = self_consistency(samples)
            rows.append(
                {
                    "task": key[0],
                    "skill": key[1],
                    "occurrence": key[2],
                    "invocation_index": key[3],
                    "mode": mode,
                    "n_samples": len(samples),
                    "n_errors": sum(1 for s in samples if s.get("error")),
                    "majority_action": majority,
                    "action_self_consistency": consistency,
                    "action_distribution": json.dumps(dist, ensure_ascii=False),
                }
            )
    return rows


def evaluate_pair(
    ref_samples: list[dict[str, Any]],
    hyp_samples: list[dict[str, Any]],
    embedder,
) -> dict[str, Any]:
    """Aggregate one (case, reuse-mode) against recompute across samples."""
    ref_by_idx = by_sample_index(ref_samples)
    ref_default = next((s for s in ref_samples if not s.get("error")), None)

    full_cos: list[float | None] = []
    full_bleu: list[float | None] = []
    full_rouge: list[float | None] = []
    deliv_cos: list[float | None] = []
    deliv_rouge: list[float | None] = []
    reason_cos: list[float | None] = []
    reason_rouge: list[float | None] = []
    modality_flags: list[bool] = []
    toolset_flags: list[bool] = []
    traj_flags: list[bool] = []
    hyp_errors = 0

    for hyp in hyp_samples:
        if hyp.get("error"):
            hyp_errors += 1
            continue
        idx = int(hyp.get("sample_index", 0))
        ref = ref_by_idx.get(idx) or ref_default
        if ref is None or ref.get("error"):
            continue
        ref_reason, ref_deliv, ref_full = get_streams(ref)
        hyp_reason, hyp_deliv, hyp_full = get_streams(hyp)

        fm = text_metrics(ref_full, hyp_full, embedder)
        dm = text_metrics(ref_deliv, hyp_deliv, embedder)
        rm = text_metrics(ref_reason, hyp_reason, embedder)
        full_cos.append(fm["embedding_cos"])
        full_bleu.append(fm["bleu4"])
        full_rouge.append(fm["rouge_l"])
        deliv_cos.append(dm["embedding_cos"])
        deliv_rouge.append(dm["rouge_l"])
        reason_cos.append(rm["embedding_cos"])
        reason_rouge.append(rm["rouge_l"])

        am = action_match(ref, hyp)
        modality_flags.append(am["modality_match"])
        toolset_flags.append(am["tool_set_match"])
        traj_flags.append(am["trajectory_match"])

    return {
        "n_samples": len(hyp_samples),
        "n_hyp_errors": hyp_errors,
        "full_bleu4": mean_or_none(full_bleu),
        "full_rouge_l": mean_or_none(full_rouge),
        "full_embedding_cos": mean_or_none(full_cos),
        "deliverable_rouge_l": mean_or_none(deliv_rouge),
        "deliverable_embedding_cos": mean_or_none(deliv_cos),
        "reasoning_rouge_l": mean_or_none(reason_rouge),
        "reasoning_embedding_cos": mean_or_none(reason_cos),
        "modality_match_rate": rate(modality_flags),
        "tool_set_match_rate": rate(toolset_flags),
        "trajectory_match_rate": rate(traj_flags),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_OUTPUT_JSONL))
    parser.add_argument("--metrics-json", default=str(DEFAULT_METRICS_JSON))
    parser.add_argument("--metrics-csv", default=str(DEFAULT_METRICS_CSV))
    parser.add_argument("--stability-csv", default=str(DEFAULT_STABILITY_CSV))
    parser.add_argument("--cksim-kv-dir", default=str(DEFAULT_CKSIM_KV_DIR))
    parser.add_argument("--embedding-model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--skip-embedding", action="store_true")
    args = parser.parse_args()

    rows = keep_latest_sample_rows(load_jsonl(Path(args.input)))
    # grouped[case][mode] -> list of per-sample rows
    grouped: dict[tuple[Any, ...], dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        grouped[pair_key(row)][row["mode"]].append(row)

    embedder = None if args.skip_embedding else MeanHiddenEmbedder(args.embedding_model)
    metric_rows: list[dict[str, Any]] = []
    for key, by_mode in grouped.items():
        ref_samples = by_mode.get("recompute") or []
        if not any(not s.get("error") for s in ref_samples):
            continue
        token_count = int(ref_samples[0]["target_end"]) - int(ref_samples[0]["target_start"])
        ref_cksim_id = first_cksim_id(ref_samples)
        for mode in REUSE_MODES:
            hyp_samples = by_mode.get(mode) or []
            if not hyp_samples:
                continue
            agg = evaluate_pair(ref_samples, hyp_samples, embedder)
            cksim = compute_cksim(
                Path(args.cksim_kv_dir),
                ref_cksim_id,
                first_cksim_id(hyp_samples),
                token_count,
            )
            _, hyp_consistency, _ = self_consistency(hyp_samples)
            metric_rows.append(
                {
                    "task": key[0],
                    "skill": key[1],
                    "occurrence": key[2],
                    "invocation_index": key[3],
                    "mode": mode,
                    **agg,
                    "action_self_consistency": hyp_consistency,
                    "cksim_mean_key": cksim["mean_key_cksim"] if cksim else None,
                    "cksim_mean_value": cksim["mean_value_cksim"] if cksim else None,
                    "cksim_layers": cksim["layers"] if cksim else None,
                }
            )

    stab_rows = stability_rows(grouped)

    summary: dict[str, Any] = {
        "input": args.input,
        "rows": len(metric_rows),
        "by_mode": {},
        "stability": {},
    }
    for mode in sorted({row["mode"] for row in metric_rows}):
        subset = [row for row in metric_rows if row["mode"] == mode]
        summary["by_mode"][mode] = {
            "rows": len(subset),
            "mean_full_bleu4": mean_present(subset, "full_bleu4"),
            "mean_full_rouge_l": mean_present(subset, "full_rouge_l"),
            "mean_full_embedding_cos": mean_present(subset, "full_embedding_cos"),
            "mean_reasoning_embedding_cos": mean_present(subset, "reasoning_embedding_cos"),
            "mean_trajectory_match_rate": mean_present(subset, "trajectory_match_rate"),
            "mean_modality_match_rate": mean_present(subset, "modality_match_rate"),
            "mean_cksim_key": mean_present(subset, "cksim_mean_key"),
            "mean_cksim_value": mean_present(subset, "cksim_mean_value"),
        }
    # Sampling-noise floor: how self-consistent is each mode across its own samples.
    for mode in sorted({row["mode"] for row in stab_rows}):
        subset = [row for row in stab_rows if row["mode"] == mode]
        summary["stability"][mode] = {
            "cases": len(subset),
            "mean_action_self_consistency": mean_present(subset, "action_self_consistency"),
            "mean_samples": mean_present(subset, "n_samples"),
        }

    metrics_json = Path(args.metrics_json)
    metrics_csv = Path(args.metrics_csv)
    stability_csv = Path(args.stability_csv)
    metrics_json.parent.mkdir(parents=True, exist_ok=True)
    metrics_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if metric_rows:
        with metrics_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0]))
            writer.writeheader()
            writer.writerows(metric_rows)
    if stab_rows:
        with stability_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(stab_rows[0]))
            writer.writeheader()
            writer.writerows(stab_rows)
    print(f"[done] metrics json: {metrics_json}")
    print(f"[done] metrics csv:  {metrics_csv}")
    print(f"[done] stability csv: {stability_csv}")


def mean_present(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [float(row[key]) for row in rows if row.get(key) is not None]
    if not vals:
        return None
    return round(sum(vals) / len(vals), 6)


if __name__ == "__main__":
    main()
