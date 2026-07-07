import argparse
import os
import re
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


def extract_thinking(filepath) -> str:
    text = Path(filepath).read_text(encoding="utf-8", errors="replace")
    m = re.search(r"<think>\s*(.*?)\s*</think>", text, re.DOTALL)
    return m.group(1).strip() if m else ""


def extract_action(filepath) -> str:
    text = Path(filepath).read_text(encoding="utf-8", errors="replace")
    m = re.search(r"<think>\s*.*?\s*</think>", text, re.DOTALL)
    action = text[m.end():].strip() if m else text.strip()
    action = re.sub(r"<\|im_end\|>.*$", "", action, flags=re.DOTALL).strip()
    return action


def load_embedding_model(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return tokenizer, model


def encode_texts(texts: list[str], tokenizer, model, max_length: int = 512) -> np.ndarray:
    """Encode a list of texts into normalized embeddings (CLS pooling)."""
    all_embs = []
    batch_size = 8
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        if torch.cuda.is_available():
            encoded = {k: v.cuda() for k, v in encoded.items()}
        with torch.no_grad():
            outputs = model(**encoded)
        embs = outputs.last_hidden_state[:, 0, :]  # CLS token
        embs = torch.nn.functional.normalize(embs, p=2, dim=1)
        all_embs.append(embs.cpu().numpy())
    return np.concatenate(all_embs, axis=0)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def compute_bertscore_batch(
    references: list[str],
    candidates: list[str],
    tokenizer,
    model,
    max_length: int = 512,
) -> list[dict]:
    """Compute BERTScore P/R/F1 for each (ref, cand) pair."""
    results = []
    for ref, cand in zip(references, candidates):
        ref_tok = tokenizer(
            ref, truncation=True, max_length=max_length, return_tensors="pt"
        )
        cand_tok = tokenizer(
            cand, truncation=True, max_length=max_length, return_tensors="pt"
        )
        if torch.cuda.is_available():
            ref_tok = {k: v.cuda() for k, v in ref_tok.items()}
            cand_tok = {k: v.cuda() for k, v in cand_tok.items()}

        with torch.no_grad():
            ref_emb = model(**ref_tok).last_hidden_state[0]   # (seq_r, dim)
            cand_emb = model(**cand_tok).last_hidden_state[0] # (seq_c, dim)

        ref_emb = torch.nn.functional.normalize(ref_emb, p=2, dim=-1)
        cand_emb = torch.nn.functional.normalize(cand_emb, p=2, dim=-1)

        # cosine similarity matrix: (seq_c, seq_r)
        sim = torch.mm(cand_emb, ref_emb.t())

        # skip [CLS] and [SEP] tokens (first and last)
        sim = sim[1:-1, 1:-1]

        if sim.numel() == 0:
            results.append({"precision": 0.0, "recall": 0.0, "f1": 0.0})
            continue

        precision = sim.max(dim=1).values.mean().item()  # each cand token -> best ref
        recall = sim.max(dim=0).values.mean().item()     # each ref token -> best cand
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        results.append({"precision": precision, "recall": recall, "f1": f1})

    return results



results_dir = "results/problem_exploration/raw_decode_token_sequences/sequences_without_occ12"
# results_dir = "results/problem_exploration/raw_decode_token_sequences/sequences_without_occ1"
# results_dir = "results/problem_exploration/raw_decode_token_sequences/sequences"
model_path = "/mnt/Large_Language_Model_Lab_1/llm_models/BAAI/bge-large-en-v1.5/BAAI/bge-large-en-v1.5"

def main():

    recompute_dir = results_dir +  "/recompute"
    rope_dir = results_dir + "/rope"

    # Collect pairs
    rc_files = sorted(os.listdir(recompute_dir))
    rp_files = sorted(os.listdir(rope_dir))
    common = sorted(set(rc_files) & set(rp_files))
    print(f"Found {len(common)} pairs\n")

    rc_thinks, rp_thinks = [], []
    rc_actions, rp_actions = [], []
    for fname in common:
        rc_thinks.append(extract_thinking(recompute_dir + "/" + fname))
        rp_thinks.append(extract_thinking(rope_dir + "/" + fname))
        rc_actions.append(extract_action(recompute_dir + "/" + fname))
        rp_actions.append(extract_action(rope_dir + "/" + fname))

    # Load model
    print(f"Loading model from {model_path} ...")
    tokenizer, model = load_embedding_model(model_path)
    print("Model loaded.\n")

    # ── Metric 1: Embedding cosine similarity ──
    print("Computing embedding cosine similarity ...")
    rc_think_embs = encode_texts(rc_thinks, tokenizer, model)
    rp_think_embs = encode_texts(rp_thinks, tokenizer, model)
    rc_action_embs = encode_texts(rc_actions, tokenizer, model)
    rp_action_embs = encode_texts(rp_actions, tokenizer, model)

    think_cosines = [
        cosine_similarity(rc_think_embs[i], rp_think_embs[i])
        for i in range(len(common))
    ]
    action_cosines = [
        cosine_similarity(rc_action_embs[i], rp_action_embs[i])
        for i in range(len(common))
    ]

    # ── Metric 2: BERTScore ──
    print("Computing BERTScore (thinking) ...")
    think_bertscores = compute_bertscore_batch(rc_thinks, rp_thinks, tokenizer, model)
    print("Computing BERTScore (action) ...")
    action_bertscores = compute_bertscore_batch(rc_actions, rp_actions, tokenizer, model)

    # ── Print results ──
    header = (
        f"{'filename':<62} | "
        f"{'think_cos':>9} {'think_P':>8} {'think_R':>8} {'think_F1':>9} | "
        f"{'act_cos':>8} {'act_P':>8} {'act_R':>8} {'act_F1':>8}"
    )
    sep = "─" * len(header)

    print(f"\n{sep}")
    print(header)
    print(sep)

    for i, fname in enumerate(common):
        name = fname.replace(".txt", "")
        tc = think_cosines[i]
        tb = think_bertscores[i]
        ac = action_cosines[i]
        ab = action_bertscores[i]
        print(
            f"{name:<62} | "
            f"{tc:>9.4f} {tb['precision']:>8.4f} {tb['recall']:>8.4f} {tb['f1']:>9.4f} | "
            f"{ac:>8.4f} {ab['precision']:>8.4f} {ab['recall']:>8.4f} {ab['f1']:>8.4f}"
        )

    # ── Summary statistics ──
    print(f"\n{sep}")
    print("SUMMARY (mean / median / min / max)")
    print(sep)

    def stats(vals, label):
        arr = np.array(vals)
        print(
            f"  {label:<20}: "
            f"mean={arr.mean():.4f}  median={np.median(arr):.4f}  "
            f"min={arr.min():.4f}  max={arr.max():.4f}"
        )

    stats(think_cosines, "think cosine")
    stats([b["f1"] for b in think_bertscores], "think BERTScore-F1")
    stats(action_cosines, "action cosine")
    stats([b["f1"] for b in action_bertscores], "action BERTScore-F1")


if __name__ == "__main__":
    main()
