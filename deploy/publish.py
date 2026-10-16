"""Publish ScholarSense to Hugging Face (free): an index dataset + a Docker Space.

    pip install -U huggingface_hub
    hf auth login                      # paste a token with WRITE access (huggingface.co/settings/tokens)
    python deploy/publish.py           # first time: uploads ~1 GB of index, then the code
    python deploy/publish.py --code-only     # later: only the code changed

    python deploy/publish.py --dry-run       # list what would be uploaded, upload nothing

Creates (or updates) two public repos under your account:
  <you>/scholarsense-index   dataset: the precomputed embeddings / tokens / corpus
  <you>/scholarsense         Space:   the code; Hugging Face builds deploy/Dockerfile
Then the site is at https://<you>-scholarsense.hf.space (building takes ~10-15 minutes).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deploy.bundle import index_files, space_files  # noqa: E402


def mb(paths) -> float:
    return sum(p.stat().st_size for p, _ in paths) / 1e6


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--space", default="scholarsense", help="Space name (default scholarsense)")
    ap.add_argument("--dataset", default="scholarsense-index", help="index dataset name")
    ap.add_argument("--code-only", action="store_true", help="skip the index upload")
    ap.add_argument("--no-scale", action="store_true", help="deploy only the 727-paper corpus (smaller, faster)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    index = [] if args.code_only else index_files(include_scale=not args.no_scale)
    missing = [str(p) for p, _ in index if not p.exists()]
    if missing:
        sys.exit("Missing index files (run the backend once, and cluster/build_index.py for the 100k corpus):\n  "
                 + "\n  ".join(missing))
    code = space_files()
    print(f"Index: {len(index)} files, {mb(index):.0f} MB")
    print(f"Code:  {len(code)} files, {mb(code):.1f} MB")
    if args.dry_run:
        for p, r in index + code:
            print(f"  {r:<70} {p.stat().st_size / 1e6:8.2f} MB")
        return

    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi()
    user = api.whoami()["name"]
    dataset_id, space_id = f"{user}/{args.dataset}", f"{user}/{args.space}"

    if index:
        print(f"\nUploading the index to the dataset {dataset_id} (unchanged files are skipped) ...", flush=True)
        api.create_repo(dataset_id, repo_type="dataset", exist_ok=True)
        api.create_commit(dataset_id, repo_type="dataset", commit_message="ScholarSense search index",
                          operations=[CommitOperationAdd(path_in_repo=r, path_or_fileobj=str(p)) for p, r in index])

    print(f"\nUploading the code to the Space {space_id} ...", flush=True)
    api.create_repo(space_id, repo_type="space", space_sdk="docker", exist_ok=True)
    ops = [CommitOperationAdd(path_in_repo=r, path_or_fileobj=str(p)) for p, r in code]
    ops.append(CommitOperationAdd(path_in_repo="deploy/index_repo.txt", path_or_fileobj=dataset_id.encode()))
    api.create_commit(space_id, repo_type="space", operations=ops, commit_message="Deploy ScholarSense")

    print(f"\nDone. Hugging Face is now building the Space (~10-15 minutes the first time):")
    print(f"  build logs: https://huggingface.co/spaces/{space_id}  (the 'Logs' button)")
    print(f"  the site:   https://{user}-{args.space}.hf.space")


if __name__ == "__main__":
    main()
