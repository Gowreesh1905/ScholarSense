"""One-time setup on EVERY laptop (A, B and C), before joining the cluster.

    pip install -r cluster/requirements.txt
    python cluster/setup_laptop.py

Downloads the models and NLTK data the workers need (so the job doesn't download
~1 GB per laptop over the hotspot), checks that PyTorch sees the GPU, and prints this
laptop's versions and IP addresses. Compare the "fingerprint" line across laptops:
it must be identical, or build_index.py will refuse to start.
"""

from __future__ import annotations

import importlib.metadata as md
import platform
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def version(pkg: str) -> str:
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return "MISSING"


def main() -> None:
    ok = True
    print("1. Python packages")
    pkgs = ["dask", "distributed", "bokeh", "torch", "sentence-transformers", "transformers", "nltk", "scikit-learn"]
    for p in pkgs:
        v = version(p)
        ok &= v != "MISSING"
        print(f"   {p:<22} {v}")

    print("\n2. GPU")
    import torch
    if torch.cuda.is_available():
        print(f"   {torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda})")
    else:
        ok = False
        print("   NO CUDA. Install the GPU build of PyTorch:\n"
              "   pip install torch --index-url https://download.pytorch.org/whl/cu126")

    print("\n3. Models (downloaded once, ~1 GB)")
    from sentence_transformers import SentenceTransformer

    from searchers.models import BGE_MODEL, SPECTER_MODEL
    for model_id in (BGE_MODEL, SPECTER_MODEL):
        SentenceTransformer(model_id, device="cpu")
        print(f"   {model_id}: ready")

    print("\n4. NLTK data (for the BM25 tokenizer)")
    from data import content_words
    print(f"   tokenizer works: {content_words('Neural networks were learning faster')}")

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    ips = sorted({a[4][0] for a in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)
                  if not a[4][0].startswith(("127.", "169.254."))})
    print("\n5. This laptop")
    print(f"   name {socket.gethostname()}, IP address(es): {', '.join(ips) or 'none found'}")
    print(f"\nfingerprint: python={platform.python_version()} dask={version('dask')} "
          f"distributed={version('distributed')} torch={version('torch')} "
          f"sentence-transformers={version('sentence-transformers')} commit={commit}")
    print("\nAll good." if ok else "\nFix the problems above before joining the cluster.")


if __name__ == "__main__":
    main()
