"""Analyze Segmentia thinking-to-action diagnostic outputs.

The analysis compares recompute vs rope for each free-generation case and
writes pair-level metrics:

  lexical similarity + optional embedding cosine + task grounding/intent
  + action boundary margin + A/B/C/D category

It is offline-only: it reads JSONL outputs and never calls vLLM.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from config import RESULTS_DIR  # noqa: E402

DEFAULT_ROOT = RESULTS_DIR / "thinking_to_action_divergence"
DEFAULT_FREE_JSONL = DEFAULT_ROOT / "data" / "free_generation_rows.jsonl"
DEFAULT_PAIR_CSV = DEFAULT_ROOT / "tables" / "thinking_pair_summary.csv"
DEFAULT_EMBEDDING_MODEL = Path(
    "/mnt/Large_Language_Model_Lab_1/模型/rag_models/BAAI-bge-base-en-v1.5"
)

WORD_RE = re.compile(r"[A-Za-z0-9_./-]+|[\u4e00-\u9fff]")
FILE_RE = re.compile(
    r"(?:/[^\s'\"(){}<>]+|[A-Za-z0-9_.-]+\."
    r"(?:md|py|json|html|css|js|ts|tsx|csv|txt|svg|gif|png))"
)
TOOL_NAMES = ("Write", "Edit", "Read", "Bash")
STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "of",
    "in",
    "for",
    "with",
    "on",
    "this",
    "that",
    "it",
    "is",
    "are",
    "be",
    "i",
    "me",
    "user",
    "need",
    "needs",
    "should",
    "will",
    "let",
    "okay",
}

INTENT_KEYWORDS = {
    "read_check_review": (
        "read",
        "check",
        "review",
        "inspect",
        "look",
        "verify",
        "open",
        "see",
    ),
    "write_create": (
        "write",
        "create",
        "draft",
        "generate",
        "build",
        "make",
        "compose",
        "produce",
    ),
    "edit_update": (
        "edit",
        "update",
        "modify",
        "revise",
        "tighten",
        "refine",
        "change",
        "replace",
    ),
    "finalize_text": (
        "final",
        "finalize",
        "answer",
        "respond",
        "summary",
        "summarize",
        "provide",
        "return",
    ),
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def norm_text(text: str) -> str:
    return " ".join((text or "").strip().split())


def word_tokens(text: str) -> list[str]:
    return [tok.lower() for tok in WORD_RE.findall(text or "")]


def ngrams(items: list[str], n: int) -> Counter[tuple[str, ...]]:
    if len(items) < n:
        return Counter()
    return Counter(tuple(items[i : i + n]) for i in range(len(items) - n + 1))


def sentence_bleu(reference: list[str], candidate: list[str], max_n: int = 4) -> float:
    if not reference or not candidate:
        return 0.0
    precisions = []
    for n in range(1, max_n + 1):
        ref_counts = ngrams(reference, n)
        cand_counts = ngrams(candidate, n)
        if not cand_counts:
            precisions.append(1e-9)
            continue
        overlap = sum(min(count, ref_counts[gram]) for gram, count in cand_counts.items())
        # Add-one smoothing keeps short reasoning snippets from collapsing to zero.
        precisions.append((overlap + 1.0) / (sum(cand_counts.values()) + 1.0))
    geo_mean = math.exp(sum(math.log(p) for p in precisions) / max_n)
    brevity = 1.0 if len(candidate) > len(reference) else math.exp(1.0 - len(reference) / len(candidate))
    return float(brevity * geo_mean)


def lcs_len(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, start=1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def rouge_l_f1(reference: list[str], candidate: list[str]) -> float:
    if not reference or not candidate:
        return 0.0
    lcs = lcs_len(reference, candidate)
    precision = lcs / len(candidate)
    recall = lcs / len(reference)
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def char_ngrams(text: str, n: int) -> Counter[str]:
    compact = re.sub(r"\s+", " ", text or "").strip()
    if len(compact) < n:
        return Counter()
    return Counter(compact[i : i + n] for i in range(len(compact) - n + 1))


def chrf_score(reference: str, candidate: str, max_n: int = 6, beta: float = 2.0) -> float:
    scores = []
    beta2 = beta * beta
    for n in range(1, max_n + 1):
        ref_counts = char_ngrams(reference, n)
        cand_counts = char_ngrams(candidate, n)
        if not ref_counts or not cand_counts:
            scores.append(0.0)
            continue
        overlap = sum(min(count, ref_counts[gram]) for gram, count in cand_counts.items())
        precision = overlap / sum(cand_counts.values())
        recall = overlap / sum(ref_counts.values())
        if precision + recall == 0:
            scores.append(0.0)
        else:
            scores.append((1 + beta2) * precision * recall / (beta2 * precision + recall))
    return float(sum(scores) / len(scores)) if scores else 0.0


def token_jaccard(a: list[str], b: list[str]) -> float:
    sa = set(a)
    sb = set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return float(len(sa & sb) / len(sa | sb))


def extract_tools(text: str) -> set[str]:
    lower = (text or "").lower()
    found = {name for name in TOOL_NAMES if name.lower() in lower}
    for label in ("write", "edit", "read", "bash"):
        if re.search(rf"\b{label}\b", lower):
            found.add(label.capitalize())
    return found


def extract_files(text: str) -> set[str]:
    return {match.strip(".,;:") for match in FILE_RE.findall(text or "")}


def keyword_set(text: str) -> set[str]:
    return {tok for tok in word_tokens(text) if len(tok) > 2 and tok not in STOPWORDS}


def overlap_ratio(a: set[str], b: set[str]) -> float | None:
    if not a and not b:
        return None
    if not a or not b:
        return 0.0
    return float(len(a & b) / len(a | b))


def classify_intent(text: str) -> str:
    lower = (text or "").lower()
    hits = []
    for label, words in INTENT_KEYWORDS.items():
        if any(re.search(rf"\b{re.escape(word)}\b", lower) for word in words):
            hits.append(label)
    unique = sorted(set(hits))
    if not unique:
        return "unknown"
    if len(unique) > 1:
        return "multi_step"
    return unique[0]


class EmbeddingScorer:
    def __init__(self, model_path: Path | None):
        self.status = "disabled"
        self.tokenizer = None
        self.model = None
        self.torch = None
        if model_path is None:
            return
        try:
            import torch
            import torch.nn.functional as F
            from transformers import AutoModel, AutoTokenizer

            self.torch = torch
            self.functional = F
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(model_path), local_files_only=True
            )
            self.model = AutoModel.from_pretrained(
                str(model_path), local_files_only=True
            )
            self.model.eval()
            self.status = "available"
        except Exception as exc:  # noqa: BLE001 - status is written for audit.
            self.status = f"unavailable: {type(exc).__name__}: {exc}"
            self.tokenizer = None
            self.model = None
            self.torch = None

    def encode(self, text: str):
        if self.status != "available" or self.tokenizer is None or self.model is None:
            return None
        encoded = self.tokenizer(
            text or "",
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        with self.torch.no_grad():
            output = self.model(**encoded)
        hidden = output.last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        pooled = self.functional.normalize(pooled, p=2, dim=1)
        return pooled[0]

    def cosine(self, a: str, b: str) -> float | None:
        va = self.encode(a)
        vb = self.encode(b)
        if va is None or vb is None:
            return None
        return float((va * vb).sum().item())


def action_diverged(recompute: dict[str, Any], rope: dict[str, Any]) -> bool | None:
    if recompute.get("error") or rope.get("error"):
        return None
    return str(recompute.get("action_label")) != str(rope.get("action_label"))


def lexical_similar(row: dict[str, Any], *, rouge_threshold: float, chrf_threshold: float, jaccard_threshold: float) -> bool:
    return (
        float(row["rouge_l_f1"]) >= rouge_threshold
        or float(row["chrf"]) >= chrf_threshold
        or float(row["token_jaccard"]) >= jaccard_threshold
    )


def thinking_similar(
    row: dict[str, Any],
    *,
    embedding_threshold: float,
    rouge_threshold: float,
    chrf_threshold: float,
    jaccard_threshold: float,
) -> bool:
    emb = row.get("embedding_cosine")
    if emb not in (None, ""):
        semantic_ok = float(emb) >= embedding_threshold
    else:
        semantic_ok = lexical_similar(
            row,
            rouge_threshold=rouge_threshold,
            chrf_threshold=chrf_threshold,
            jaccard_threshold=jaccard_threshold,
        )
    return bool(
        semantic_ok
        and bool(row["intent_match"])
        and not bool(row["grounding_conflict"])
    )


def category(thinking_same: bool, diverged: bool | None) -> str:
    if diverged is None:
        return "missing"
    if thinking_same and not diverged:
        return "A_thinking_similar_action_same"
    if thinking_same and diverged:
        return "B_thinking_similar_action_diverged"
    if not thinking_same and diverged:
        return "C_thinking_different_action_diverged"
    return "D_thinking_different_action_same"


def boundary_top1_changed(recompute: dict[str, Any], rope: dict[str, Any]) -> bool | None:
    rec = recompute.get("action_boundary_top1")
    rep = rope.get("action_boundary_top1")
    if rec is None or rep is None:
        return None
    return str(rec) != str(rep)


def make_pair_rows(
    rows: list[dict[str, Any]],
    scorer: EmbeddingScorer,
    *,
    embedding_threshold: float,
    rouge_threshold: float,
    chrf_threshold: float,
    jaccard_threshold: float,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int, int], dict[str, dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["task"]),
            str(row["skill"]),
            int(row["occurrence"]),
            int(row["invocation_index"]),
        )
        grouped.setdefault(key, {})[str(row["mode"])] = row

    out = []
    for key, by_mode in sorted(grouped.items()):
        if "recompute" not in by_mode or "rope" not in by_mode:
            continue
        rec = by_mode["recompute"]
        rope = by_mode["rope"]
        rec_reasoning = norm_text(rec.get("reasoning") or "")
        rope_reasoning = norm_text(rope.get("reasoning") or "")
        rec_tokens = word_tokens(rec_reasoning)
        rope_tokens = word_tokens(rope_reasoning)

        rec_tools = extract_tools(rec_reasoning)
        rope_tools = extract_tools(rope_reasoning)
        rec_files = extract_files(rec_reasoning)
        rope_files = extract_files(rope_reasoning)
        rec_keywords = keyword_set(rec_reasoning)
        rope_keywords = keyword_set(rope_reasoning)
        rec_intent = classify_intent(rec_reasoning)
        rope_intent = classify_intent(rope_reasoning)
        tool_overlap = overlap_ratio(rec_tools, rope_tools)
        file_overlap = overlap_ratio(rec_files, rope_files)
        keyword_overlap = overlap_ratio(rec_keywords, rope_keywords)
        grounding_conflict = (
            (tool_overlap == 0.0 and bool(rec_tools) and bool(rope_tools))
            or (file_overlap == 0.0 and bool(rec_files) and bool(rope_files))
        )

        emb = scorer.cosine(rec_reasoning, rope_reasoning)
        diverged = action_diverged(rec, rope)
        pair = {
            "task": key[0],
            "skill": key[1],
            "occurrence": key[2],
            "invocation_index": key[3],
            "recompute_action": rec.get("action_label"),
            "rope_action": rope.get("action_label"),
            "action_diverged": diverged,
            "bleu": sentence_bleu(rec_tokens, rope_tokens),
            "rouge_l_f1": rouge_l_f1(rec_tokens, rope_tokens),
            "chrf": chrf_score(rec_reasoning, rope_reasoning),
            "token_jaccard": token_jaccard(rec_tokens, rope_tokens),
            "embedding_cosine": emb,
            "embedding_metric_status": scorer.status,
            "recompute_intent": rec_intent,
            "rope_intent": rope_intent,
            "intent_match": rec_intent == rope_intent,
            "recompute_tools": ";".join(sorted(rec_tools)),
            "rope_tools": ";".join(sorted(rope_tools)),
            "tool_overlap": tool_overlap,
            "recompute_files": ";".join(sorted(rec_files)),
            "rope_files": ";".join(sorted(rope_files)),
            "file_overlap": file_overlap,
            "keyword_overlap": keyword_overlap,
            "grounding_conflict": grounding_conflict,
            "recompute_boundary_status": rec.get("boundary_status"),
            "rope_boundary_status": rope.get("boundary_status"),
            "recompute_boundary_type": rec.get("action_boundary_type"),
            "rope_boundary_type": rope.get("action_boundary_type"),
            "recompute_boundary_margin": rec.get("action_boundary_margin"),
            "rope_boundary_margin": rope.get("action_boundary_margin"),
            "boundary_margin_delta_rope_minus_recompute": None,
            "boundary_top1_changed": boundary_top1_changed(rec, rope),
        }
        if (
            pair["recompute_boundary_margin"] is not None
            and pair["rope_boundary_margin"] is not None
        ):
            pair["boundary_margin_delta_rope_minus_recompute"] = (
                float(pair["rope_boundary_margin"])
                - float(pair["recompute_boundary_margin"])
            )
        same = thinking_similar(
            pair,
            embedding_threshold=embedding_threshold,
            rouge_threshold=rouge_threshold,
            chrf_threshold=chrf_threshold,
            jaccard_threshold=jaccard_threshold,
        )
        pair["thinking_similar"] = same
        pair["category"] = category(same, diverged)
        out.append(pair)
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--free-jsonl", default=str(DEFAULT_FREE_JSONL))
    parser.add_argument("--pair-csv", default=str(DEFAULT_PAIR_CSV))
    parser.add_argument("--embedding-model", default=str(DEFAULT_EMBEDDING_MODEL))
    parser.add_argument("--disable-embedding", action="store_true")
    parser.add_argument("--embedding-threshold", type=float, default=0.80)
    parser.add_argument("--rouge-threshold", type=float, default=0.35)
    parser.add_argument("--chrf-threshold", type=float, default=0.45)
    parser.add_argument("--jaccard-threshold", type=float, default=0.25)
    args = parser.parse_args()

    embedding_model = None if args.disable_embedding else Path(args.embedding_model)
    scorer = EmbeddingScorer(embedding_model)
    rows = load_jsonl(Path(args.free_jsonl))
    pair_rows = make_pair_rows(
        rows,
        scorer,
        embedding_threshold=args.embedding_threshold,
        rouge_threshold=args.rouge_threshold,
        chrf_threshold=args.chrf_threshold,
        jaccard_threshold=args.jaccard_threshold,
    )
    write_csv(Path(args.pair_csv), pair_rows)
    print(f"[done] pair summary: {args.pair_csv}")
    print(f"[done] pairs: {len(pair_rows)}")
    print(f"[done] embedding_metric_status: {scorer.status}")


if __name__ == "__main__":
    main()
