"""Which files a deployed ScholarSense needs, besides the code.

The website only *loads* precomputed artifacts; the deployed server never trains or embeds
a corpus. Two groups, all from .cache/ (searchers/cache.py) and cluster/:

  727-paper corpus: BGE / SPECTER / GloVe embeddings, BM25 tokens, the trained Word2Vec
                    (so the server serves exactly the models that were evaluated)
  100,727 papers:   the cluster-built BGE / SPECTER embeddings and BM25 tokens, the
                    distractor corpus and the index manifest (cluster/build_index.py)

Paths in the index repo: cache/<file> -> .cache/<file>; cluster/... -> cluster/...
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def index_files(include_scale: bool = True) -> list[tuple[Path, str]]:
    """(local path, path in the index repo) for every artifact the backend loads."""
    from data import load_corpus
    from searchers.cache import cache_path, text_hash
    from searchers.legacy import Word2VecAdapter
    from searchers.models import BGE_MODEL, GLOVE_MODEL, SPECTER_MODEL

    abstracts = [p.abstract for p in load_corpus()]
    names = [f"emb_{text_hash(abstracts, model, '')}.npy" for model in (BGE_MODEL, SPECTER_MODEL, GLOVE_MODEL)]
    names.append(f"bm25_tok_{text_hash(abstracts, 'bm25')}.pkl")
    names.append(f"word2vec_{text_hash(abstracts, sorted(Word2VecAdapter.PARAMS.items()))}.pkl")
    files = [(cache_path(n), f"cache/{n}") for n in names]

    if include_scale:
        from cluster.common import DISTRACTORS_FILE, DISTRACTORS_MANIFEST, INDEX_MANIFEST
        from cluster.index import ready_index

        manifest = ready_index(("full",))
        if manifest:
            from cluster.index import cache_files
            files += [(p, f"cache/{p.name}") for p in cache_files(manifest["n_distractors"], ("full",)).values()]
            files += [(DISTRACTORS_FILE, "cluster/data/distractors.jsonl"),
                      (DISTRACTORS_MANIFEST, "cluster/data/distractors_manifest.json"),
                      (INDEX_MANIFEST, "cluster/output/index_manifest.json")]
        else:
            print("No cluster-built index here: deploying the 727-paper corpus only.")
    return files


def local_path_for(repo_path: str) -> Path:
    """Where a file from the index repo belongs in the app."""
    if repo_path.startswith("cache/"):
        return ROOT / ".cache" / repo_path.removeprefix("cache/")
    return ROOT / repo_path


# The code the Space needs to run (paths relative to the repo root). Everything the
# backend imports, the evaluation results the website shows, and the website's source.
SPACE_FILES = [
    "data.py", "tfidf_lexical_search.py", "word2vec_pytorch_search.py", "bert_specter_contextual_search.py",
    "abstract_sentences.csv",
    "backend/*.py", "backend/requirements.txt",
    "searchers/*.py",
    "cluster/__init__.py", "cluster/common.py", "cluster/index.py",
    "results/*.json", "results/*.md", "results/*.png",
    "results/cluster/*.json", "results/cluster/*.png", "results/cluster/runs/*.json",
    "results/scale/*.json", "results/scale/*.md", "results/scale/*.png",
    "frontend/package.json", "frontend/package-lock.json", "frontend/index.html", "frontend/vite.config.ts",
    "frontend/tsconfig.json", "frontend/tsconfig.app.json", "frontend/tsconfig.node.json",
    "frontend/src/**/*", "frontend/public/**/*",
    "deploy/__init__.py", "deploy/bundle.py", "deploy/fetch_assets.py",
]


def space_files() -> list[tuple[Path, str]]:
    out = []
    for pattern in SPACE_FILES:
        for p in sorted(ROOT.glob(pattern)):
            if p.is_file() and "__pycache__" not in p.parts:
                out.append((p, p.relative_to(ROOT).as_posix()))
    # Hugging Face reads the Space's README (front matter) and Dockerfile from the repo root.
    out.append((ROOT / "deploy" / "Dockerfile", "Dockerfile"))
    out.append((ROOT / "deploy" / "SPACE_README.md", "README.md"))
    return out
