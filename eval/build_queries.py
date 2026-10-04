"""Build the masked corpus and the exact-span query set (P3 brief, step 1).

    python eval/build_queries.py

Writes eval/data/corpus_masked.json and eval/data/queries_exact.jsonl.

Why a masked corpus: every task/problem span is copied word for word from its own
abstract, so searching the original abstracts would let keyword methods win by
matching the copied text. The masked corpus removes all task/problem spans first.
"""

from __future__ import annotations

import json
import re
import statistics
import sys
from collections import Counter, defaultdict

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from common import (CORPUS_MASKED, OVERLAP_BINS, QUERY_FILES, overlap, overlap_bin,  # noqa: E402
                    write_jsonl)

from data import load_corpus, load_spans  # noqa: E402

QUERY_ASPECTS = ("task", "problem")
MIN_QUERY_WORDS = 4
MIN_MASKED_WORDS = 10  # a paper with less text than this left can't be found by any method


def mask_abstract(abstract: str, ranges: list[tuple[int, int]]) -> str:
    """Remove the character ranges, one space per removed run, then collapse whitespace."""
    removed = [False] * len(abstract)
    for start, end in ranges:
        for i in range(max(0, start), min(end, len(abstract))):
            removed[i] = True
    out: list[str] = []
    prev_removed = False
    for ch, gone in zip(abstract, removed):
        if gone:
            if not prev_removed:
                out.append(" ")
        else:
            out.append(ch)
        prev_removed = gone
    return re.sub(r"\s+", " ", "".join(out)).strip()


def histogram(values: list[float]) -> list[int]:
    counts = [0] * (len(OVERLAP_BINS) - 1)
    for v in values:
        counts[overlap_bin(v)] += 1
    return counts


def print_histogram(values: list[float], indent: str = "  ") -> None:
    counts = histogram(values)
    for i, c in enumerate(counts):
        lo, hi = OVERLAP_BINS[i], OVERLAP_BINS[i + 1]
        close = "]" if i == len(counts) - 1 else ")"
        print(f"{indent}[{lo:.1f}, {hi:.1f}{close}  {c:>5}  {'#' * round(40 * c / max(1, max(counts)))}")


def main() -> None:
    papers = load_corpus()
    spans = [s for s in load_spans(MIN_QUERY_WORDS) if s.aspect in QUERY_ASPECTS]

    # --- masked corpus ----------------------------------------------------
    ranges: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for s in spans:
        ranges[s.doc_index].append((s.raw_start, s.raw_end))
    masked = [mask_abstract(p.abstract, ranges[p.doc_index]) for p in papers]

    CORPUS_MASKED.parent.mkdir(parents=True, exist_ok=True)
    with open(CORPUS_MASKED, "w", encoding="utf-8") as f:
        json.dump(masked, f, ensure_ascii=False, indent=0)
    print(f"Wrote {CORPUS_MASKED} ({len(masked)} abstracts)")

    lengths = [len(m.split()) for m in masked]
    shortest = min(range(len(masked)), key=lambda i: lengths[i])
    print(f"Words per masked abstract: min {min(lengths)}, median {statistics.median(lengths):.0f}, "
          f"max {max(lengths)}; empty: {sum(n == 0 for n in lengths)}")
    orig_words = [len(p.abstract.split()) for p in papers]
    print(f"Words removed per abstract (mean): {statistics.mean(o - m for o, m in zip(orig_words, lengths)):.1f}")
    print(f"Papers with no task/problem span removed: {sum(1 for p in papers if p.doc_index not in ranges)}")
    print(f"Shortest masked abstract (doc {shortest}, {lengths[shortest]} words, "
          f"original {orig_words[shortest]}):\n  {masked[shortest]!r}")

    # --- exact queries ----------------------------------------------------
    docs_of_text: dict[str, set[int]] = defaultdict(set)
    for s in spans:
        docs_of_text[s.text].add(s.doc_index)
    ambiguous = {t for t, d in docs_of_text.items() if len(d) > 1}

    rows: list[dict] = []
    seen_text: set[str] = set()
    unanswerable = 0
    for s in spans:
        if s.text in ambiguous or s.text in seen_text:
            continue
        if lengths[s.doc_index] < MIN_MASKED_WORDS:
            unanswerable += 1
            continue
        seen_text.add(s.text)
        rows.append({
            "qid": f"exact-{len(rows) + 1:04d}",
            "query": s.text,
            "aspect": s.aspect,
            "relevant_docs": [s.doc_index],
            "corpus": "masked",
            "overlap": round(overlap(s.text, masked[s.doc_index]), 4),
        })
    write_jsonl(QUERY_FILES["exact"], rows)

    print(f"\nWrote {QUERY_FILES['exact']} ({len(rows)} queries)")
    print(f"Candidate spans (task/problem, >= {MIN_QUERY_WORDS} words): {len(spans)}; "
          f"dropped {len(ambiguous)} ambiguous duplicate texts "
          f"({sum(1 for s in spans if s.text in ambiguous)} spans)")
    print(f"Dropped {unanswerable} queries whose masked paper has < {MIN_MASKED_WORDS} words left "
          f"(docs: {[i for i, n in enumerate(lengths) if n < MIN_MASKED_WORDS]})")
    print("Queries by aspect:", dict(Counter(r["aspect"] for r in rows)))
    ov = [r["overlap"] for r in rows]
    print(f"Overlap with the masked relevant doc: mean {statistics.mean(ov):.3f}, "
          f"median {statistics.median(ov):.3f}; queries with 0 content words/overlap: {sum(v == 0 for v in ov)}")
    print("Overlap histogram:")
    print_histogram(ov)
    wc = [len(r["query"].split()) for r in rows]
    print(f"Query length in words: min {min(wc)}, median {statistics.median(wc):.0f}, max {max(wc)}")


if __name__ == "__main__":
    main()
