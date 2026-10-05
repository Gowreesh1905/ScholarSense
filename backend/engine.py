"""
ScholarSense search engine wrapper for the web API.

Loads the corpus once and fits every method the registry reports as
available (searchers/registry.py), plus aspect search when
searchers/aspect.py exists. Caching of the expensive artifacts (Word2Vec
model, embeddings) lives in the searchers themselves (searchers/cache.py).
"""

import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # so the root-level data.py / searchers/ package import

from data import Paper, arxiv_url, load_corpus, load_spans          # noqa: E402
from searchers.base import BaseSearcher                            # noqa: E402
from searchers.models import get_device                            # noqa: E402
from searchers.registry import (                                   # noqa: E402
    DEFAULT_UI_METHODS, METHOD_COLORS, available_methods, build_searcher,
)

SNIPPET_CHARS = 300

ASPECT_DESCRIPTIONS = {
    "task": "BGE similarity against only the task each paper sets out to do.",
    "problem": "BGE similarity against only the problem each paper says it addresses.",
    "method": "BGE similarity against only the method or idea each paper proposes.",
    "result": "BGE similarity against only the results each paper reports.",
}


def _snippet(abstract: str) -> str:
    text = abstract.replace("\n", " ").strip()
    return text[:SNIPPET_CHARS].rstrip() + "…" if len(text) > SNIPPET_CHARS else text


class SearchEngine:
    """Holds every fitted searcher and answers ranked top-k queries."""

    def __init__(self, verbose: bool = True):
        self._log = print if verbose else (lambda *a, **k: None)
        self.device = get_device()

        self.papers: list[Paper] = load_corpus()
        abstracts = [p.abstract for p in self.papers]
        self._log(f"[engine] Loaded {len(self.papers)} papers. Device: {self.device}")

        self.searchers: dict[str, BaseSearcher] = {}
        for key in available_methods():
            start = time.perf_counter()
            try:
                searcher = build_searcher(key)
                searcher.fit(abstracts)
            except Exception:
                # One broken method shouldn't take the whole API down.
                self._log(f"[engine] {key}: fit FAILED, skipping\n{traceback.format_exc()}")
                continue
            self.searchers[key] = searcher
            self._log(f"[engine] {key}: fitted in {time.perf_counter() - start:.2f}s")

        self.aspect_searcher = None
        self.aspect_names: list[str] = []
        self._build_aspect_searcher()

        defaults = [k for k in DEFAULT_UI_METHODS if k in self.searchers]
        self.default_methods = defaults or list(self.searchers)[:4]

        # corpus key -> {"label", "papers", "searchers"}; "727" is always there,
        # "scale" only once the laptop cluster has built the index (cluster/build_index.py).
        self.corpora: dict[str, dict] = {
            "727": {"label": f"{len(self.papers)} papers", "papers": self.papers, "searchers": self.searchers},
        }
        self._build_scale_corpus()
        self._log(f"[engine] Ready: {', '.join(self.searchers)}"
                  f"{' + aspect' if self.aspect_searcher else ''}"
                  f"{' + scale corpus' if 'scale' in self.corpora else ''}")

    def _build_scale_corpus(self):
        try:
            from cluster.common import SCALE_METHODS, scale_papers
            from cluster.index import ready_index
        except ImportError as exc:
            self._log(f"[engine] Scale corpus unavailable: {exc}")
            return
        manifest = ready_index(("full",))
        if not manifest:
            self._log("[engine] No cluster-built index (run cluster/build_index.py); serving 727 papers only.")
            return
        papers = scale_papers(manifest["n_distractors"])
        abstracts = [p.abstract for p in papers]
        searchers: dict[str, BaseSearcher] = {}
        for key in SCALE_METHODS:
            if key not in self.searchers:
                continue
            start = time.perf_counter()
            try:
                searcher = build_searcher(key)
                searcher.fit(abstracts)   # reads the cluster-built embeddings/tokens from .cache/
            except Exception:
                self._log(f"[engine] scale/{key}: fit FAILED, skipping\n{traceback.format_exc()}")
                continue
            searchers[key] = searcher
            self._log(f"[engine] scale/{key}: fitted on {len(papers):,} papers in {time.perf_counter() - start:.2f}s")
        if searchers:
            self.corpora["scale"] = {"label": f"{len(papers):,} papers", "papers": papers, "searchers": searchers}

    def _build_aspect_searcher(self):
        try:
            from searchers.aspect import ASPECTS, AspectSearcher
        except ImportError as exc:
            self._log(f"[engine] Aspect search unavailable: {exc}")
            return
        start = time.perf_counter()
        try:
            searcher = AspectSearcher()
            searcher.fit(self.papers, load_spans())
        except Exception:
            self._log(f"[engine] aspect: fit FAILED, skipping\n{traceback.format_exc()}")
            return
        self.aspect_searcher = searcher
        self.aspect_names = list(ASPECTS)
        self._log(f"[engine] aspect: fitted in {time.perf_counter() - start:.2f}s")

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def aspects(self) -> list[str]:
        return ["all", *self.aspect_names]

    def method_info(self, key: str) -> dict:
        s = self.searchers[key]
        return {
            "key": key,
            "label": s.label,
            "family": s.family,
            "description": s.description,
            "score_type": s.score_type,
            "color": METHOD_COLORS.get(key, "#6b7280"),
        }

    def status(self) -> dict:
        return {
            "status": "ready",
            "corpus_size": len(self.papers),
            "device": self.device,
            "methods": [self.method_info(k) for k in self.searchers],
            "default_methods": self.default_methods,
            "aspects": self.aspects,
            "aspect_coverage": self.aspect_searcher.coverage() if self.aspect_searcher else {},
            "corpora": [
                {"key": key, "label": c["label"], "size": len(c["papers"]), "methods": list(c["searchers"]),
                 "default_methods": self.corpus_defaults(key), "aspects": key == "727"}
                for key, c in self.corpora.items()
            ],
        }

    def corpus_defaults(self, corpus: str) -> list[str]:
        available = self.corpora[corpus]["searchers"]
        return [k for k in self.default_methods if k in available] or list(available)[:4]

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def _hit(self, papers: list, rank: int, doc_index: int, score: float, matched_span: dict | None = None) -> dict:
        paper = papers[doc_index]
        return {
            "rank": rank,
            "doc_index": doc_index,
            "score": float(score),
            "arxiv_id": paper.arxiv_id,
            "url": arxiv_url(paper.arxiv_id),
            "snippet": _snippet(paper.abstract),
            "abstract": paper.abstract,
            "title": getattr(paper, "title", None),
            "matched_span": matched_span,
        }

    def search(self, query: str, k: int = 5, methods: list[str] | None = None, aspect: str = "all",
               corpus: str = "727") -> dict:
        """Callers validate `methods`, `aspect` and `corpus` first (see backend/app.py)."""
        query = query.strip()
        papers, searchers = self.corpora[corpus]["papers"], self.corpora[corpus]["searchers"]
        methods = methods or self.corpus_defaults(corpus)
        total_start = time.perf_counter()

        method_results = []
        for key in methods:
            start = time.perf_counter()
            hits = searchers[key].search(query, k)
            took_ms = (time.perf_counter() - start) * 1000
            method_results.append({
                **self.method_info(key),
                "took_ms": round(took_ms, 1),
                "results": [self._hit(papers, r, d, s) for r, (d, s) in enumerate(hits, 1)],
            })

        if aspect != "all":
            start = time.perf_counter()
            hits = self.aspect_searcher.search(query, aspect, k)
            took_ms = (time.perf_counter() - start) * 1000
            method_results.append({
                "key": "aspect",
                "label": f"Aspect: {aspect.capitalize()}",
                "family": "contextual",
                "description": ASPECT_DESCRIPTIONS.get(aspect, f"Similarity against only the {aspect} spans."),
                "score_type": "cosine",
                "color": METHOD_COLORS["aspect"],
                "took_ms": round(took_ms, 1),
                "results": [
                    self._hit(papers, r, h.doc_index, h.score,
                              {"start": h.span_start, "end": h.span_end, "text": h.span_text})
                    for r, h in enumerate(hits, 1)
                ],
            })

        return {
            "query": query,
            "aspect": aspect,
            "corpus": corpus,
            "took_ms": round((time.perf_counter() - total_start) * 1000, 1),
            "methods": method_results,
        }
