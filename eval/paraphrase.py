"""Back-translate the exact queries (English -> German -> English) (P3 brief, step 2).

    python eval/paraphrase.py

Reads eval/data/queries_exact.jsonl, writes eval/data/queries_paraphrased.jsonl.
Back-translation changes the wording but keeps the meaning, which recreates
vocabulary mismatch between the query and the abstract.
"""

from __future__ import annotations

import re
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_queries import print_histogram  # noqa: E402
from common import (QUERY_FILES, load_corpus_for, max_overlap, read_jsonl,  # noqa: E402
                    write_jsonl)

EN_DE = "Helsinki-NLP/opus-mt-en-de"
DE_EN = "Helsinki-NLP/opus-mt-de-en"
BATCH_SIZE = 32
NUM_BEAMS = 4
MAX_NEW_TOKENS = 256


def translate(texts: list[str], model_id: str, device: str) -> list[str]:
    import torch
    from transformers import MarianMTModel, MarianTokenizer

    tokenizer = MarianTokenizer.from_pretrained(model_id)
    model = MarianMTModel.from_pretrained(model_id).to(device).eval()

    # Sort by length so batches have little padding, then restore the original order.
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    out: list[str] = [""] * len(texts)
    for start in range(0, len(order), BATCH_SIZE):
        idx = order[start:start + BATCH_SIZE]
        batch = tokenizer([texts[i] for i in idx], return_tensors="pt", padding=True,
                          truncation=True, max_length=512).to(device)
        with torch.no_grad():
            generated = model.generate(**batch, num_beams=NUM_BEAMS, max_new_tokens=MAX_NEW_TOKENS)
        for i, text in zip(idx, tokenizer.batch_decode(generated, skip_special_tokens=True)):
            out[i] = text.strip()
        if (start // BATCH_SIZE) % 10 == 0:
            print(f"  {model_id}: {min(start + BATCH_SIZE, len(order))}/{len(order)}", flush=True)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


def normalize(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


def main() -> None:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    exact = read_jsonl(QUERY_FILES["exact"])
    masked = load_corpus_for("masked")
    originals = [q["query"] for q in exact]
    print(f"Device: {device}; paraphrasing {len(originals)} queries")

    t0 = time.perf_counter()
    german = translate(originals, EN_DE, device)
    english = translate(german, DE_EN, device)
    print(f"Back-translation took {time.perf_counter() - t0:.0f}s")

    rows: list[dict] = []
    dropped = 0
    for q, para in zip(exact, english):
        if not para:
            dropped += 1
            continue
        rows.append({
            "qid": f"para-{len(rows) + 1:04d}",
            "query": para,
            "aspect": q["aspect"],
            "relevant_docs": q["relevant_docs"],
            "corpus": "masked",
            "overlap": round(max_overlap(para, [masked[d] for d in q["relevant_docs"]]), 4),
            "source_qid": q["qid"],
            "original": q["query"],
        })
    write_jsonl(QUERY_FILES["paraphrased"], rows)
    print(f"Wrote {QUERY_FILES['paraphrased']} ({len(rows)} queries; {dropped} empty outputs dropped)")

    identical = sum(r["query"] == r["original"] for r in rows)
    identical_norm = sum(normalize(r["query"]) == normalize(r["original"]) for r in rows)
    print(f"Paraphrase identical to the original: {identical} ({100 * identical / len(rows):.1f}%); "
          f"identical ignoring case/punctuation: {identical_norm} ({100 * identical_norm / len(rows):.1f}%)")

    old = [q["overlap"] for q in exact]
    new = [r["overlap"] for r in rows]
    print(f"Mean overlap: exact {statistics.mean(old):.3f} -> paraphrased {statistics.mean(new):.3f}")
    print("Paraphrased overlap histogram:")
    print_histogram(new)

    print("\n5 examples (original -> paraphrase):")
    for r in rows[:: max(1, len(rows) // 5)][:5]:
        print(f"  {r['original']!r}\n    -> {r['query']!r}  (overlap {r['overlap']})")


if __name__ == "__main__":
    main()
