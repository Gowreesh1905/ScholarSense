"""Quick check that methods fit and return sensible results.

    python -m searchers.smoke                       # every available method
    python -m searchers.smoke bm25 bge              # only these
    python -m searchers.smoke tfidf --query "..."   # your own query (repeatable)
"""

from __future__ import annotations

import argparse
import sys
import time

from data import load_corpus
from searchers.registry import METHOD_ORDER, available_methods, build_searcher

DEFAULT_QUERIES = [
    "making blurry photos sharp again",
    "self-driving cars sensing surroundings with laser scanners",
    "teaching robots to grasp unfamiliar objects",
]


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("keys", nargs="*", help=f"method keys (default: all available). Known: {' '.join(METHOD_ORDER)}")
    parser.add_argument("--query", action="append", help="query to run (repeatable; default: 3 built-in queries)")
    parser.add_argument("-k", type=int, default=3, help="results per query (default 3)")
    args = parser.parse_args(argv)

    unknown = [k for k in args.keys if k not in METHOD_ORDER]
    if unknown:
        parser.error(f"unknown method(s): {', '.join(unknown)}")

    available = available_methods()
    keys = args.keys or available
    missing = [k for k in keys if k not in available]
    if missing:
        print(f"Not available (module missing or failed to import): {', '.join(missing)}")
    keys = [k for k in keys if k in available]
    queries = args.query or DEFAULT_QUERIES

    papers = load_corpus()
    abstracts = [p.abstract for p in papers]
    print(f"Corpus: {len(abstracts)} abstracts. Methods: {', '.join(keys) or '(none)'}\n")

    failed = []
    for key in keys:
        try:
            searcher = build_searcher(key)
            t0 = time.perf_counter()
            searcher.fit(abstracts)
            fit_s = time.perf_counter() - t0
        except Exception as exc:
            print(f"=== {key}: FIT FAILED: {exc!r}\n")
            failed.append(key)
            continue

        print(f"=== {key} ({searcher.label}, {searcher.score_type})  fit {fit_s:.2f}s")
        for query in queries:
            try:
                t0 = time.perf_counter()
                hits = searcher.search(query, args.k)
                query_ms = (time.perf_counter() - t0) * 1000
            except Exception as exc:
                print(f"  Q: {query}\n    SEARCH FAILED: {exc!r}")
                failed.append(key)
                continue
            print(f"  Q: {query}   ({query_ms:.1f} ms)")
            if not hits:
                print("    (no results)")
            for doc_index, score in hits:
                snippet = abstracts[doc_index][:80].replace("\n", " ")
                print(f"    {doc_index:>4}  {score:8.4f}  {snippet}")
        print()

    if failed:
        print(f"FAILED: {', '.join(dict.fromkeys(failed))}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
