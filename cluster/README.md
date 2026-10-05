# Laptop cluster: building the 100k-paper index on 3 laptops

**What this does.** The 727-paper corpus is too small to need a cluster. We grow it to
**~100,700 papers** (the 727 + 100,000 arXiv computer-science abstracts from 2021–2022)
and ask: *do our findings still hold when the right paper is hidden among 100× more?*

Searching 100k papers semantically first needs every abstract **embedded** with BGE and
SPECTER (GPU work) and **tokenized** for BM25 (CPU work). That index build is the job we
split across the laptops with [Dask](https://distributed.dask.org):

```
                         laptop A (scheduler + client + workers)
                   ┌──────────────────────────────────────────┐
  abstracts ──────►│ build_index.py: 250-abstract chunks       │
                   │ scheduler: hands each chunk to a free     │
                   │ worker; GPU tasks → GPU workers,          │
                   │ tokenizing tasks → CPU workers            │
                   └──────┬───────────────┬───────────────┬────┘
                hotspot   │               │               │
                   laptop A workers   laptop B workers   laptop C workers
                   1 GPU + 4 CPU      1 GPU + 4 CPU      1 GPU + 4 CPU
                          └─────── embeddings + tokens back to A ──────┘
                                         │
                       assembled into .cache/ → backend + eval_scale.py
```

- A faster laptop simply finishes more chunks (nothing to configure).
- If a laptop disappears, Dask re-runs its unfinished chunks on the others.
- Finished chunks are saved as they arrive, so a crashed build resumes where it stopped.

Measured on one RTX 4060 laptop: ~100–125 abstracts/s for both models (fp16), so the
full build takes **about 15 minutes on one laptop** and roughly a third of that on three.

---

## 1. Before the day (each laptop, ~20 min, needs internet)

On **every** laptop (A, B and C):

```powershell
git fetch
git switch cluster            # the same commit on every laptop
git pull

pip install -r cluster/requirements.txt
# NVIDIA GPU build of PyTorch, if `python -c "import torch; print(torch.cuda.is_available())"` says False:
pip install torch --index-url https://download.pytorch.org/whl/cu126

python cluster/setup_laptop.py   # downloads the models (~1 GB), checks the GPU
```

`setup_laptop.py` ends with a **fingerprint** line. Compare it across the three laptops:
Python, dask, distributed, torch, sentence-transformers and the commit must match.

Then, in an **Administrator** PowerShell on every laptop (one time):

```powershell
powershell -ExecutionPolicy Bypass -File cluster\open_firewall.ps1
```

This allows ports 8786–8787 and 9000–9360 in, on Private networks. If your Wi-Fi adapter
isn't called "Wi-Fi", pass `-InterfaceAlias "Wi-Fi 2"` (see `Get-NetAdapter`).

On **laptop A only** (the one that will run the scheduler and the backend):

```powershell
python cluster/prepare_corpus.py     # downloads 2 parquet files (~330 MB), writes 100,000 distractors
```

Only laptop A needs the corpus: the cluster sends each worker the abstracts it embeds.

## 2. Connect the laptops (on the day)

1. Turn on one phone's **hotspot** and connect all three laptops to it. College Wi-Fi
   usually blocks laptop-to-laptop traffic. When Windows asks, choose **Private network**,
   or re-run `open_firewall.ps1` after connecting (it marks the network Private).
2. **Laptop A**, in a terminal it keeps open:

   ```powershell
   powershell -ExecutionPolicy Bypass -File cluster\start_scheduler.ps1
   ```

   It prints laptop A's hotspot IP, e.g. `192.168.43.10`. Everyone needs it.
   Open the dashboard on laptop A: **http://localhost:8787/status**.
3. **Every laptop (A too)** joins with its own name:

   ```powershell
   powershell -ExecutionPolicy Bypass -File cluster\start_workers.ps1 -Scheduler 192.168.43.10 -Name laptopA
   powershell -ExecutionPolicy Bypass -File cluster\start_workers.ps1 -Scheduler 192.168.43.10 -Name laptopB
   powershell -ExecutionPolicy Bypass -File cluster\start_workers.ps1 -Scheduler 192.168.43.10 -Name laptopC
   ```

   Each opens two windows: one GPU worker and the CPU workers (4 tokenizer processes; change
   with `-CpuWorkers`). **Closing those windows = unplugging that laptop.**
4. **Laptop A** checks that everyone is there and matches:

   ```powershell
   python cluster/build_index.py --scheduler 192.168.43.10 --check
   ```

   Expected: 3 laptops, each with a GPU name and 4 CPU workers, then `Environment OK.`

## 3. Run the experiments (all commands on laptop A)

Replace `192.168.43.10` with laptop A's IP. Rough times are for three RTX 40-series laptops.

| Step | Command | Time | What you get |
|---|---|---|---|
| a. fp16 check (once) | `python cluster/check_precision.py` | 1 min | proof that half precision doesn't change MRR (`precision_check.json`) |
| b. Scaling experiment | `python cluster/bench_scaling.py --scheduler 192.168.43.10 --laptops laptopA laptopB laptopC` | ~20 min | 5 runs of 30,000 abstracts: each laptop alone, then A+B, then A+B+C |
| c. Full index build | `python cluster/build_index.py --scheduler 192.168.43.10` | ~6 min | the 100k index, assembled into `.cache/` |
| d. Fault tolerance | `python cluster/build_index.py --scheduler 192.168.43.10 --limit 30000 --run-name demo --discard`, then **close laptop B's two worker windows** about 20 s in | ~3 min | the job still finishes; the report records when B left |
| e. Charts | `python cluster/charts.py --main full --fault demo` | seconds | PNGs + `summary.json` for the website |
| f. Evaluation at scale | `python cluster/eval_scale.py` | ~5 min | MRR@10/Recall@10 at 727 vs. 100k for BM25, SPECTER, BGE and both hybrids |
| g. Website | restart `python backend/app.py` | ~1 min | a **727 / 100,727 papers** switch on the Search page; the Results page gets "At 100,727 papers" and "Laptop cluster" sections |

Notes:
- **b** puts the first laptop you list as the baseline. `--no-solo` skips B and C alone
  (saves ~9 min) but then the table can't compute the "ideal" for your mix of GPUs.
  `--limit 20000` makes every run shorter.
- **c** resumes if interrupted: just run the same command again.
- **d** with `--discard` doesn't touch the real index. For the live demo use
  `--limit 30000` (~2–3 min). To bring laptop B back: run its `start_workers.ps1` again.
- **f** needs the build from **c** to have finished and assembled.

## 4. Where the outputs go

| What | Where | Commit it? |
|---|---|---|
| Distractor corpus | `cluster/data/distractors.jsonl` (+ `_manifest.json`) | no (gitignored, 120 MB; rebuild with prepare_corpus.py) |
| Embedding/token chunks | `cluster/output/runs/<run>/chunks/` | no (gitignored, ~400 MB) |
| The assembled index | `.cache/emb_*.npy`, `.cache/bm25_tok_*.pkl`, `cluster/output/index_manifest.json` | no (gitignored) |
| Timing report per run | `results/cluster/runs/<run>.json` | **yes** |
| Scaling experiment list | `results/cluster/scaling.json` | **yes** |
| fp16 check | `results/cluster/precision_check.json` | **yes** |
| Cluster charts + summary | `results/cluster/*.png`, `results/cluster/summary.json` | **yes** |
| Evaluation at scale | `results/scale/summary.json`, `summary.md`, `mrr_727_vs_scale.png`, `raw/` | **yes** |

The website reads the committed files through `/api/results/files/...`, so after you commit
them anyone can see the results; only laptop A needs the index to *search* 100k papers.

## 5. Live demo (≈2 minutes of the 6)

1. Laptop A on the projector with the **dashboard** open (http://localhost:8787/status):
   the task stream shows every worker on every laptop lighting up.
2. Start step **d** (`--limit 30000 --run-name demo --discard`).
3. ~20 s in, close **laptop B's** two worker windows. The dashboard shows B's workers
   vanish; within seconds, B's unfinished chunks reappear on A and C.
4. The run finishes; the terminal prints `laptopB-gpu left` and the per-laptop counts.
5. Switch to the website's Results page: the scaling chart and the fault-tolerance timeline.

Record a backup video of this the day before, in case the hotspot misbehaves.

## 6. What to say about the results (honestly)

- **Speedup won't be 3×.** Amdahl's law: some work can't be split, such as handing out
  chunks, sending ~4 KB of embeddings per abstract back to laptop A over the hotspot, and
  waiting for the last chunk. The table's "serial share" (Karp–Flatt) estimates how much.
- **The GPUs aren't identical**, so the fair yardstick is "ideal for these GPUs" (the laptops'
  solo speeds added up), not 3×. "Cluster efficiency" is measured ÷ that ideal.
- **The scheduler is a single point of failure.** Closing laptop A stops the job; closing B
  or C doesn't. A real system would replicate the scheduler.
- **fp16** is ~3× faster on the GPU and changes MRR@10 by at most ~0.0003 (step a).
- **Which methods are at 100k:** BM25, SPECTER, BGE and both hybrids. TF-IDF is left out
  (BM25 covers the keyword family and scored within 0.005 of it at 727); Word2Vec and GloVe
  are left out (far behind at 727, and retraining Word2Vec on 100k adds nothing).

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| `Can't reach the scheduler at …:8786` | Same hotspot? `start_scheduler.ps1` running on A? `open_firewall.ps1` run on A (as admin)? Test with `Test-NetConnection 192.168.43.10 -Port 8786`. |
| Workers connect, but `--check` hangs or tasks never start on B/C | Laptop A can't reach B's/C's worker ports: run `open_firewall.ps1` on B and C too, and make sure the network is Private. |
| `Cluster environment mismatch: … dask … vs …` | Different package versions: `pip install -r cluster/requirements.txt` on that laptop. |
| `git commit … vs … here` | That laptop isn't on the same commit: `git pull`. (`--allow-commit-mismatch` overrides, at your own risk.) |
| `GPU worker but PyTorch has no CUDA` | Install the CUDA build of torch (see step 1). |
| `…'s workers are already running` | `powershell -ExecutionPolicy Bypass -File cluster\stop_workers.ps1 -Name laptopB`, then start again. |
| Scripts won't run ("running scripts is disabled") | Always run them as `powershell -ExecutionPolicy Bypass -File …` as shown. |
| A worker dies with `WinError 10055` | Windows ran short of network buffers (seen once under heavy load). The job carries on without it; restart that laptop's workers if you want it back. |
| Need to change the corpus size | `python cluster/prepare_corpus.py --n 50000`, then rebuild (`build_index.py`). |

## Files

| File | Runs on | Purpose |
|---|---|---|
| `prepare_corpus.py` | A | download + filter + sample the distractors |
| `setup_laptop.py` | every laptop | download models, check GPU and versions |
| `open_firewall.ps1` | every laptop (admin, once) | firewall + Private network |
| `start_scheduler.ps1` | A | the Dask scheduler + dashboard |
| `start_workers.ps1` / `stop_workers.ps1` | every laptop | join / leave the cluster |
| `build_index.py` | A | the distributed index build (also `--check`, demo runs) |
| `bench_scaling.py` | A | the 1 → 2 → 3 laptop experiment |
| `check_precision.py` | any GPU laptop | fp16 vs fp32 on the 727-paper evaluation |
| `eval_scale.py` | A | P3's evaluation on the scaled corpus |
| `charts.py` | A | charts + `results/cluster/summary.json` |
| `job.py`, `tasks.py`, `index.py`, `common.py` | — | the job, the worker functions, saving/assembling, shared paths |
