"""Download everything the server loads: the search index and the models.

Runs once while the Docker image is built (see deploy/Dockerfile), so the running Space
starts without downloading anything.

    python deploy/fetch_assets.py                       # index repo from deploy/index_repo.txt
    python deploy/fetch_assets.py --from-dir <folder>   # a local copy (for testing)
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deploy.bundle import ROOT, local_path_for  # noqa: E402


def place(src_root: Path) -> int:
    count = 0
    for src in sorted(src_root.rglob("*")):
        if not src.is_file():
            continue
        repo_path = src.relative_to(src_root).as_posix()
        if not (repo_path.startswith("cache/") or repo_path.startswith("cluster/")):
            continue
        dest = local_path_for(repo_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
        count += 1
    return count


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-dir", type=Path, help="use a local copy of the index repo instead of downloading")
    args = ap.parse_args()

    local_bundle = ROOT / "deploy" / "index_bundle"   # only for a local `docker build` test
    if args.from_dir or local_bundle.is_dir():
        src = args.from_dir or local_bundle
        print(f"Index: copying from {src}", flush=True)
        print(f"  {place(src)} files placed")
        if not args.from_dir:
            shutil.rmtree(local_bundle, ignore_errors=True)
    else:
        repo_file = ROOT / "deploy" / "index_repo.txt"
        repo = os.environ.get("INDEX_REPO") or (repo_file.read_text().strip() if repo_file.exists() else "")
        if not repo:
            sys.exit("No index repo: set INDEX_REPO or write it to deploy/index_repo.txt (publish.py does this).")
        from huggingface_hub import snapshot_download
        print(f"Index: downloading the dataset {repo} ...", flush=True)
        tmp = ROOT / "deploy" / "_index"
        snapshot_download(repo_id=repo, repo_type="dataset", local_dir=tmp)
        print(f"  {place(tmp)} files placed")
        shutil.rmtree(tmp, ignore_errors=True)

    print("Models: downloading (cached in the image) ...", flush=True)
    from sentence_transformers import CrossEncoder, SentenceTransformer

    from searchers.models import BGE_MODEL, CROSS_ENCODER_MODEL, GLOVE_MODEL, SPECTER_MODEL
    for model_id in (SPECTER_MODEL, BGE_MODEL, GLOVE_MODEL):
        SentenceTransformer(model_id, device="cpu")
        print(f"  {model_id}", flush=True)
    CrossEncoder(CROSS_ENCODER_MODEL, device="cpu")
    print(f"  {CROSS_ENCODER_MODEL}", flush=True)

    print("NLTK data ...", flush=True)
    from data import content_words
    content_words("download the tokenizer data")
    print("Done.")


if __name__ == "__main__":
    main()
