# ScholarSense: demo script (about 6 minutes)

**Status.** Every query below was run through the real backend (`/api/search`) on the P5 branch, and the first two
parts were also run in P4's frontend (branch `p4-frontend`, commit `9cf58c4`) with the P5 branch merged in a throwaway
copy. All behaviours described here were seen. **TBD-FINAL:** run the checklist and the whole script once on the final
merged `main` before presenting, because the interface may have changed.

Expected results are written as arXiv IDs (what you see on screen) with the `doc_index` in brackets.

---

## Before you start (5 minutes ahead)

1. **Check port 8000 is free.** `python backend/app.py` and the frontend proxy both use port 8000. On my laptop another
   program was already listening there. Check with `netstat -ano | findstr :8000` and close whatever it is.
2. **Terminal 1:** `python backend/app.py`. Wait for the line `[engine] Ready: tfidf, bm25, word2vec, glove, specter, bge, hybrid_rrf, hybrid + aspect`.
   This takes about 1 minute when the `.cache/` folder is warm, and 3 to 4 minutes on a fresh clone (Word2Vec trains once).
3. **Terminal 2:** `cd frontend` then `npm run dev`, then open http://localhost:5173.
4. **Warm up (important).** The very first search that includes *Hybrid + re-rank* took 5 to 6 seconds in my tests
   because the cross-encoder loads lazily. Run any query once with the default methods, and run one aspect search, so the
   real demo is fast afterwards (about 150 to 250 ms for the hybrid).
5. The header should say `727 papers · 8 methods · CUDA`. Open the **Results** tab once to check the four charts load.
6. Light or dark mode: pick one and keep it for the whole demo.

## Script

### Part 1 (0:00 to 1:45). Side by side: the vocabulary gap
Search page. The four default methods are already selected: **TF-IDF, BM25, BGE-base, Hybrid + re-rank**. Aspect: *Whole abstract*.

**Query A** (type it exactly):
`figuring out how far away everything is in a picture taken by a camera with a 360-degree view`

| Column | What you should see | What to say |
|---|---|---|
| TF-IDF, BM25 | Rank 1 is arXiv:2203.04450 (an out-of-distribution detection paper), then unrelated papers. None of the three right papers. | "They match the words *far away* and *picture*, not the idea." |
| BGE-base | Rank 1 arXiv:2112.14931 (dense depth for multiview 360° images) [197], rank 2 arXiv:2202.08010 (depth from omnidirectional images) [475]. | "A semantic model understands that this is 360° depth estimation." |
| Hybrid + re-rank | arXiv:2112.14931 [197] at rank 2. | "Fusion keeps it, but note it did not put the best paper first." |

Point at the small *also in …* tags and the per-column timing (lexical a few ms, BGE tens of ms, hybrid about 150 ms).

**Query B:** `working out where a robot should hold an object by splitting it into simple shapes`
Expected: TF-IDF and BM25 miss (top results are unrelated); **BGE and Hybrid both rank arXiv:2201.00956 [292] first.**

### Part 2 (1:45 to 3:00). Results page: the evidence
Click **Results**.

1. **Table.** Say: "BM25 is as good as or better than BGE on exact queries (0.523 against 0.498); the best single
   system is BM25 + BGE by rank fusion (0.560 exact, 0.507 paraphrased)."
2. **Overlap chart** (`overlap_vs_mrr.png`). "As the query shares fewer words with the paper, BM25 falls from 0.83 to 0.14;
   BGE only from 0.63 to 0.30. This is the vocabulary-mismatch effect."
3. **Speed against accuracy** (one sentence): "Re-ranking costs 150 ms against 17 ms for no gain."
4. **TBD-HW:** once the hand-written column exists, point at it and read the BGE and BM25 numbers.

### Part 3 (3:00 to 4:00). The ablation, live
On the Search page, deselect TF-IDF and BM25 and select **BGE-base**, **BM25 + BGE (RRF)** and **Hybrid + re-rank**.

**Query C:** `filling in the missing or removed parts of a damaged picture`

| Column | Expected top 5 | Meaning |
|---|---|---|
| BGE-base | 2201.10753 [347] at rank 2 and 2201.09865 [328] at rank 4 are both inpainting papers | finds the topic |
| BM25 + BGE (RRF) | keeps only 2201.09865 [328]; 2201.10753 [347] drops out of the top 5 | fusing with a weak BM25 list pushed one right paper out |
| Hybrid + re-rank | none of the five inpainting papers in the top 5 | the re-ranker made it worse |

Say: "On average fusion helps and re-ranking does not. Here you can see the re-ranker losing papers BGE found. We report
that, we do not hide it." Then show `results/ablation.png` if you want the averages (0.498 → 0.560 → 0.548).

### Part 4 (4:00 to 5:15). Aspect search
Select all four default methods again (or just **BGE-base**, to keep the screen readable). Click the **Problem** control.
A fifth column, **Aspect: Problem**, appears with the matched sentence highlighted in teal.

**Query D (Problem):** `results look unrealistic or contain visible artifacts`
- *Aspect: Problem* column: arXiv:2203.06457 [535], 2110.14373 [459], 2112.09061 [22], 2112.08867 [21], 2110.08985 [98], each with its problem sentence highlighted, for example "naively optimizing the latent space leads to artifacts and poor novel view rendering".
- *BGE-base* column (whole abstract): an off-target mix (augmented-reality display synthesis, material style transfer, a metric paper).
- Say: "Searching only the *problem* sentences finds papers that say this is *their* problem."

**Query E (Problem):** `views of the same object are not consistent with each other in 3D`
Aspect: 2204.06307 [615], 2106.09051 [44], 2112.12484 [162], 2203.06457 [535], 2112.15399 [193]. BGE returns general view-synthesis papers.

**Query F (Result):** click **Result**, then `runs in real time at tens of frames per second`
Aspect: 2203.17261 [571], 2201.05989 [296], 2112.10703 [38] report big speedups (also 2112.14683 [191] and 2105.14391 [439], weaker). BGE returns video papers matched on the word *frames*.

Say honestly: "We chose these by looking at the output. Aspect search helps when the query reads like a complaint or a
result; for plain topic queries, whole-abstract search is as good or better. We have no accuracy number for it."

### Part 5 (5:15 to 6:00). Honest findings and close
Optional live failure, **Query G:** `a robot following written directions to find its way through a building`.
The right paper is arXiv:2201.10788 [342]. BGE's top results are other robot and navigation papers (2112.14084, 2105.10396) but not that one
(BGE ranks it 57th; the hybrid reaches 14th). "Semantic search is not magic: here everything misses."

Close with the three points from slide 10: (1) keywords win when words overlap, (2) semantic methods help most at low
overlap and fusion beats both, (3) re-ranking added cost but no measured gain.

---

## If something goes wrong

| Problem | Fix |
|---|---|
| Backend slow on the first hybrid query | Expected (cross-encoder cold start, 5 to 6 s). Warm up beforehand (step 4). |
| Backend will not start or the page shows errors | Use the screenshots in `report/img/` and the static charts in `results/*.png`. For the Results part you do not need the backend. |
| Port 8000 busy | `netstat -ano \| findstr :8000`, close that program, restart the backend. |
| Frontend down but backend up | `python -m searchers.smoke bm25 bge hybrid --query "figuring out how far away everything is in a picture taken by a camera with a 360-degree view"` shows the same rankings in the terminal. `python -m searchers.aspect` shows aspect hits. |
| Charts do not load on the Results page | Open the PNGs from `results/` directly. |

## Screenshots to capture (manual, on the final build)

Save as PNG in `report/img/` (the folder is not created yet). Names used by `SLIDES.md`:

| File | What to capture |
|---|---|
| `01-mismatch.png` | Search page, Query A, four default methods |
| `02-results-page.png` | Results tab: table and overlap chart |
| `03-ablation-live.png` | Query C with BGE, RRF and Hybrid |
| `04-aspect.png` | Query D with the **Aspect: Problem** column highlighted |
| `05-aspect-result.png` | Query F with the **Result** aspect |
