"""Method registry (team/CONTRACT.md §4).

Each entry imports its module lazily, so a method whose module isn't merged
yet (or whose dependency isn't installed) is skipped with a warning instead
of breaking everything else.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any

from searchers.base import BaseSearcher

# key -> (module, class name, constructor kwargs)
_ENTRIES: dict[str, tuple[str, str, dict[str, Any]]] = {
    "tfidf":      ("searchers.legacy",   "TFIDFAdapter",    {}),
    "bm25":       ("searchers.bm25",     "BM25Searcher",    {}),
    "word2vec":   ("searchers.legacy",   "Word2VecAdapter", {}),
    "glove":      ("searchers.glove",    "GloVeSearcher",   {}),
    "specter":    ("searchers.legacy",   "SpecterAdapter",  {}),
    "bge":        ("searchers.bge",      "BGESearcher",     {}),
    "hybrid_rrf": ("searchers.hybrid",   "HybridSearcher",  {"rerank": False}),
    "hybrid":     ("searchers.hybrid",   "HybridSearcher",  {"rerank": True}),
    "bm25_prf":   ("searchers.bm25_prf", "BM25PRFSearcher", {}),
}

METHOD_ORDER: list[str] = list(_ENTRIES)
DEFAULT_UI_METHODS = ["tfidf", "bm25", "bge", "hybrid"]

# Shared by the UI and the charts so colors match everywhere.
METHOD_COLORS = {
    "tfidf": "#d97706",
    "bm25": "#dc2626",
    "word2vec": "#0891b2",
    "glove": "#059669",
    "specter": "#9333ea",
    "bge": "#2563eb",
    "hybrid_rrf": "#64748b",
    "hybrid": "#db2777",
    "bm25_prf": "#ea580c",
    "aspect": "#0f766e",
}

_warned: set[str] = set()


def _load_class(key: str) -> type[BaseSearcher]:
    if key not in _ENTRIES:
        raise KeyError(f"Unknown method {key!r}. Known methods: {', '.join(METHOD_ORDER)}")
    module_name, class_name, _ = _ENTRIES[key]
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def build_searcher(key: str) -> BaseSearcher:
    """New, unfitted instance of the method `key`."""
    cls = _load_class(key)
    return cls(**_ENTRIES[key][2])


def available_methods() -> list[str]:
    """Keys in METHOD_ORDER whose module imports OK."""
    available = []
    for key in METHOD_ORDER:
        try:
            _load_class(key)
        except (ImportError, AttributeError) as exc:
            if key not in _warned:
                _warned.add(key)
                print(f"[registry] Skipping {key!r}: {exc}", file=sys.stderr)
            continue
        available.append(key)
    return available
