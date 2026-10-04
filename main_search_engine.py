"""
ScholarSense — Main Search Engine
===================================
Compares retrieval methods side-by-side. By default, the original three:
  1. TF-IDF       (lexical / keyword matching)
  2. Word2Vec     (static semantic — inline PyTorch implementation, no gensim needed)
  3. SPECTER      (contextual semantic)

Any method in the registry (searchers/registry.py) can be picked instead.

Usage:
    python main_search_engine.py
    python main_search_engine.py --methods bm25,bge,hybrid
"""

import sys, io
# Force UTF-8 output so box-drawing and other Unicode chars work on all
# Windows terminals regardless of the system code-page (cp1252 etc.)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import argparse

from data import arxiv_url, load_corpus
from searchers.models import get_device
from searchers.registry import METHOD_ORDER, available_methods, build_searcher

DEFAULT_METHODS = ["tfidf", "word2vec", "specter"]


# =============================================================================
# DISPLAY
# =============================================================================

def display_top_k(searcher, query, papers, k=3):
    """Prints one method's top-k results with arXiv ID, link and abstract snippet."""
    hits = searcher.search(query, k)

    header = f"[ {searcher.label} ({searcher.family}) | Top {k} Results ]"
    print("=" * 70)
    print(header)
    print("=" * 70)
    if not hits:
        print("  (no results — none of the query's words are known to this method)\n")
    for rank, (idx, score) in enumerate(hits, 1):
        paper   = papers[idx]
        snippet = paper.abstract.replace("\n", " ").strip()[:300] + "..."
        link    = arxiv_url(paper.arxiv_id) or "(no arXiv ID for this paper)"

        print(f"  #{rank}  Score: {score:.4f}   arXiv: {paper.arxiv_id or '-'}")
        print(f"       Link   : {link}")
        print(f"       Snippet: {snippet}")
        print()
    print("-" * 70 + "\n")


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="ScholarSense interactive search")
    parser.add_argument(
        "--methods", default=",".join(DEFAULT_METHODS),
        help=f"comma-separated method keys (default: {','.join(DEFAULT_METHODS)}). "
             f"Known: {','.join(METHOD_ORDER)}",
    )
    args = parser.parse_args()
    keys = [k.strip() for k in args.methods.split(",") if k.strip()]
    unknown = [k for k in keys if k not in METHOD_ORDER]
    if unknown:
        parser.error(f"unknown method(s): {', '.join(unknown)}")
    return keys


def main():
    keys = parse_args()

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("Loading dataset...")
    papers    = load_corpus()
    abstracts = [p.abstract for p in papers]
    print(f"Loaded {len(papers)} unique research papers!\n")

    # ── 2. Fit the chosen methods ─────────────────────────────────────────────
    available = available_methods()
    missing = [k for k in keys if k not in available]
    if missing:
        print(f"Skipping unavailable method(s): {', '.join(missing)}\n")
    keys = [k for k in keys if k in available]
    if not keys:
        print("No methods available — nothing to search with.")
        return

    print("Initializing models (the first run trains/embeds; later runs use .cache/)...\n")
    searchers = []
    for key in keys:
        searcher = build_searcher(key)
        print(f"> Setting up {searcher.label}...")
        searcher.fit(abstracts)
        searchers.append(searcher)
    print()

    # ── 3. Interactive loop ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SCHOLARSENSE — INTERACTIVE SEMANTIC SEARCH ENGINE")
    print("=" * 70)
    print(f"  Corpus  : {len(papers)} research papers (Scholar Inbox dataset)")
    print(f"  Methods : {'  |  '.join(s.label for s in searchers)}")
    print(f"  Device  : {get_device().upper()}")
    print("=" * 70)
    print("  Type your query and press Enter.  Type 'exit' to quit.")
    print("=" * 70 + "\n")

    while True:
        try:
            query = input(">> Search Query: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            print("  (empty — please type a query)\n")
            continue
        if query.lower() == "exit":
            print("Goodbye!")
            break

        print()
        for searcher in searchers:
            display_top_k(searcher, query, papers, k=3)


if __name__ == "__main__":
    main()
