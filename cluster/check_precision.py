"""Does half precision (fp16) change retrieval quality? Run once on any GPU laptop.

    python cluster/check_precision.py

The cluster embeds the corpus in fp16 (about 3x faster on the GPU). This re-runs P3's
exact and paraphrased queries on the 727-paper masked corpus twice per model, once with
fp32 document embeddings (what the searchers normally use) and once with fp16 ones,
and compares MRR@10. Writes results/cluster/precision_check.json.
"""

from __future__ import annotations

import re
import statistics
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))

import numpy as np  # noqa: E402

from cluster.common import RESULTS_CLUSTER, write_json  # noqa: E402
from common import QUERY_FILES, load_corpus_for, read_jsonl  # noqa: E402  (eval/common.py)
from run_eval import first_relevant_rank, query_metrics  # noqa: E402
from searchers.registry import build_searcher  # noqa: E402

TOLERANCE = 0.005   # |MRR@10 difference| we accept as "no change"


def fp16_embeddings(model_id: str, texts: list[str], normalize: bool) -> np.ndarray:
    import torch
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_id, device="cuda" if torch.cuda.is_available() else "cpu").half()
    emb = model.encode(texts, batch_size=64, normalize_embeddings=normalize, convert_to_numpy=True)
    return emb.astype(np.float32)


def evaluate(searcher, queries: list[dict]) -> tuple[float, list]:
    hits = []
    for i in range(0, len(queries), 256):
        hits += searcher.search_batch([q["query"] for q in queries[i:i + 256]], k=100)
    ranks = [first_relevant_rank(h, set(q["relevant_docs"])) for h, q in zip(hits, queries)]
    mrr = statistics.mean(query_metrics(r, [d for d, _ in h[:10]], q["relevant_docs"])[0]
                          for r, h, q in zip(ranks, hits, queries))
    return mrr, ranks


def main() -> None:
    from searchers.models import BGE_MODEL, SPECTER_MODEL

    corpus = load_corpus_for("masked")
    sets = {name: read_jsonl(QUERY_FILES[name]) for name in ("exact", "paraphrased")}
    clean = lambda t: re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", t)).strip()  # noqa: E731  (BERTSearcher's cleaning)

    rows = []
    for key, model_id, texts, normalize in [("bge", BGE_MODEL, corpus, True),
                                            ("specter", SPECTER_MODEL, [clean(t) for t in corpus], False)]:
        searcher = build_searcher(key)
        searcher.fit(corpus)
        fp32_corpus = searcher._corpus                     # the searchers keep normalised embeddings here
        emb16 = fp16_embeddings(model_id, texts, normalize)
        fp16_corpus = emb16 / np.linalg.norm(emb16, axis=1, keepdims=True)
        for name, queries in sets.items():
            searcher._corpus = fp32_corpus
            mrr32, r32 = evaluate(searcher, queries)
            searcher._corpus = fp16_corpus
            mrr16, r16 = evaluate(searcher, queries)
            same = sum(a == b for a, b in zip(r32, r16)) / len(queries)
            rows.append({"method": key, "query_set": name, "mrr@10_fp32": round(mrr32, 4),
                         "mrr@10_fp16": round(mrr16, 4), "difference": round(mrr16 - mrr32, 4),
                         "identical_ranks": round(same, 4)})
            print(f"{key:8s} {name:12s} MRR@10 fp32 {mrr32:.4f}  fp16 {mrr16:.4f}  "
                  f"diff {mrr16 - mrr32:+.4f}  identical ranks {same:.1%}", flush=True)

    ok = all(abs(r["difference"]) <= TOLERANCE for r in rows)
    write_json(RESULTS_CLUSTER / "precision_check.json", {
        "rows": rows, "tolerance": TOLERANCE, "fp16_ok": ok,
        "created": datetime.now().isoformat(timespec="seconds"),
    })
    print(f"\nfp16 {'is fine' if ok else 'CHANGES the results: build with --precision fp32'} "
          f"(tolerance ±{TOLERANCE} MRR@10). Saved results/cluster/precision_check.json")


if __name__ == "__main__":
    main()
