"""Download arXiv metadata and build the distractor corpus (run on laptop A only).

    python cluster/prepare_corpus.py                 # 100,000 distractors
    python cluster/prepare_corpus.py --n 50000

Source: the Hugging Face dataset `bluuebunny/arxiv_metadata_by_year` (a copy of the
Cornell arXiv metadata snapshot, one parquet file per year; no login needed). We take
2021 and 2022, the same years as most of the 727 papers, keep papers with at least one
computer-science category, drop the 727 papers themselves (by arXiv ID and by text),
and shuffle with a fixed seed so every run picks the same distractors.

Writes cluster/data/distractors.jsonl and distractors_manifest.json (gitignored).
Only laptop A needs these: the cluster sends each worker the texts it embeds.
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cluster.common import DATA_DIR, DISTRACTORS_FILE, DISTRACTORS_MANIFEST, sha1_texts, write_json  # noqa: E402

REPO_ID = "bluuebunny/arxiv_metadata_by_year"
YEARS = ["21", "22"]
MIN_WORDS = 50          # very short "abstracts" are mostly errata/withdrawn notices
SEED = 0

_CS = re.compile(r"(^|\s)cs\.")
_VERSION = re.compile(r"v\d+$")


def normalize_id(arxiv_id: str) -> str:
    return _VERSION.sub("", str(arxiv_id).strip())


def normalize_text(text: str) -> str:
    return " ".join(str(text).split()).lower()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=100_000, help="number of distractors (default 100000)")
    args = ap.parse_args()

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    from data import load_corpus

    originals = load_corpus()
    original_ids = {normalize_id(p.arxiv_id) for p in originals if p.arxiv_id}
    original_texts = {normalize_text(p.abstract) for p in originals}

    pool: dict[str, dict] = {}
    for year in YEARS:
        print(f"Downloading data/{year}.parquet (cached after the first time)...", flush=True)
        path = hf_hub_download(REPO_ID, f"data/{year}.parquet", repo_type="dataset")
        table = pq.read_table(path, columns=["id", "title", "categories", "abstract"])
        rows = table.to_pylist()
        kept = 0
        for r in rows:
            if not r["abstract"] or not r["categories"] or not _CS.search(r["categories"]):
                continue
            aid = normalize_id(r["id"])
            abstract = " ".join(r["abstract"].split())   # arXiv abstracts have hard line breaks
            if len(abstract.split()) < MIN_WORDS or aid in pool:
                continue
            if aid in original_ids or abstract.lower() in original_texts:
                continue
            pool[aid] = {
                "arxiv_id": aid,
                "title": " ".join((r["title"] or "").split()),
                "abstract": abstract,
                "categories": r["categories"],
                "year": f"20{year}",
            }
            kept += 1
        print(f"  20{year}: {len(rows):,} papers, {kept:,} computer-science papers kept", flush=True)

    ids = sorted(pool)
    random.Random(SEED).shuffle(ids)
    if args.n > len(ids):
        sys.exit(f"Only {len(ids):,} candidate papers, asked for {args.n:,}.")
    chosen = [pool[i] for i in ids[: args.n]]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = DISTRACTORS_FILE.with_suffix(".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for row in chosen:
            f.write(__import__("json").dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(DISTRACTORS_FILE)

    digest = sha1_texts([r["abstract"] for r in chosen])
    write_json(DISTRACTORS_MANIFEST, {
        "count": len(chosen),
        "sha1": digest,
        "source": f"huggingface:{REPO_ID} data/{{{','.join(YEARS)}}}.parquet",
        "filter": f"at least one cs.* category, >= {MIN_WORDS} words, the 727 papers removed",
        "seed": SEED,
        "created": datetime.now().isoformat(timespec="seconds"),
    })
    words = sorted(len(r["abstract"].split()) for r in chosen)
    print(f"\nWrote {len(chosen):,} distractors to {DISTRACTORS_FILE}")
    print(f"  pool of candidates: {len(ids):,}; median abstract length {words[len(words) // 2]} words")
    print(f"  sha1 {digest[:16]}")


if __name__ == "__main__":
    main()
