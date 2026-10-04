# P4 — Frontend

Read `team/README.md` and `team/CONTRACT.md` first (especially §6 HTTP API and §8 `summary.json`).

**Your job:** turn the existing 3-column React UI into a registry-driven one: a model picker (1–4 methods), an aspect selector with highlighted matched spans, and a Results page showing the evaluation.

**You own:** everything in `frontend/`.
**Don't edit:** anything outside `frontend/`.

Stack (already set up): React 19, Vite 8, TypeScript 6, Tailwind v4 (`@tailwindcss/vite`), lint with oxlint. Run with `cd frontend && npm install && npm run dev` → http://localhost:5173; `/api` is proxied to `http://127.0.0.1:8000`.

---

## Current state (what to change)

- `src/lib/api.ts`: `MethodKey = "tfidf" | "word2vec" | "bert"` is a fixed union; `runSearch(query, k)` sends no methods/aspect.
- `src/App.tsx`: `METHOD_ORDER` hardcoded to 3 keys; fixed `md:grid-cols-3` grid; overlap badges keyed by `arxiv_id`; heading "Search 727 papers, three ways."
- `src/lib/methodStyles.ts`: Tailwind classes per method (`bg-tfidf`, …) backed by `--color-tfidf/word2vec/bert` in `src/index.css`.
- `src/components/ScoreBar.tsx`: shows `score * 100` as a %, which is wrong for non-cosine scores (BM25, RRF, cross-encoder).
- `src/components/ExampleQueries.tsx`: two examples ("detecting fake news…", "compressing large language models") match **0** papers in the corpus.
- `src/components/EmptyState.tsx`: says "Three retrieval methods".
- `ResultCard`: always renders an arXiv link; 18 papers have `arxiv_id: null`.

## Phase 1 (H0.5 → H4): build against a mock API

Before the backend is ready, add `src/lib/mock.ts` returning responses that match CONTRACT §6 **exactly** (8 methods in health; 1–4 methods + an `aspect` column in search; some results with `arxiv_id: null`; aspect results with `matched_span`). Enable it with `VITE_MOCK=1 npm run dev`; the default is the real API.

1. **Types (`api.ts`):** `MethodKey = string`; add `family`, `score_type`, `color`, `took_ms` to the method types; `doc_index`, nullable `arxiv_id`/`url`, and `matched_span: {start, end, text} | null` to hits; `default_methods`, `aspects`, `aspect_coverage` to health. `runSearch(query, {k, methods, aspect})`. Add `fetchResults()` for `GET /api/results` (404 → "no results yet" state).
2. **Colors:** Tailwind v4 only generates classes it can see as complete strings in the source, so **don't build class names dynamically** (`bg-${key}` won't work). Use the `color` from the API (or a fallback map in `methodStyles.ts` with the CONTRACT colors) via inline styles: dot/bar `style={{ backgroundColor: color }}`, text `style={{ color }}`, soft background `style={{ backgroundColor: color + "1a" }}`. Remove the per-method Tailwind classes and `--color-tfidf/word2vec/bert` once nothing uses them. Keep `METHOD_TAGLINE` as a fallback map, but prefer `description` from the API.
3. **Model picker:** below the search bar, chips for every method from `/api/health`, grouped by family (Lexical / Static / Contextual / Hybrid), each with its color dot. Select 1–4 (disable further chips at 4, with a hint). Default = `default_methods`. Changing the selection re-runs the current query. Add a clear **"Hybrid + re-rank"** switch that adds/removes `hybrid` (the demo uses it).
4. **Aspect selector:** a segmented control `Whole abstract | Task | Problem | Method | Result` from `health.aspects`, showing coverage counts as small text (e.g. "Problem · 480 papers"). Hide it if `aspects` is only `["all"]`.
5. **Results grid:** columns follow the order the response returns them. Use a static class map for the grid: `{1: "md:grid-cols-1", 2: "md:grid-cols-2", 3: "md:grid-cols-3", 4: "md:grid-cols-2 xl:grid-cols-4", 5: "md:grid-cols-3 xl:grid-cols-5"}` (5 = 4 methods + the aspect column). Show each method's `took_ms` in its column header.
6. **ResultCard:**
   - Missing `arxiv_id` → show "No arXiv ID" in muted text, no link.
   - React `key` and the "also in X" overlap index use **`doc_index`**, not `arxiv_id` (several papers have a null ID).
   - If `matched_span` is set: show the abstract with the span wrapped in `<mark>` (styled with the aspect color at low opacity), by slicing `abstract` at `start`/`end`. Collapsed, show ~150 chars around the span, not the first 300 chars; "read more" shows the full abstract with the highlight.
7. **ScoreBar:** if `score_type === "cosine"`, keep the 0–100% bar. Otherwise draw the bar relative to the top score in that column (`score / maxScoreInColumn`) and print the raw value (`12.31`, `0.0325`, `-3.2`) instead of a %. For cross-encoder logits, which can be negative, use min–max within the column.
8. **Example queries:** replace the list with these (topics verified to exist in the corpus):
   - `making blurry photos sharp again`
   - `turning a single photo into a 3D model`
   - `self-driving cars sensing surroundings with laser scanners`
   - `teaching robots to grasp unfamiliar objects`
   - `learning image features without human labels`
   After merge #2, run each against the real API and swap any that give poor results for all methods.
9. **Copy:** update the heading and EmptyState text to describe several methods instead of "three ways"; mention aspect search.

## Phase 2 (H4 → H7): real API + Results page

1. Switch to the real backend (no `VITE_MOCK`) and fix every mismatch. If the backend doesn't match CONTRACT §6, tell P1; don't work around it silently.
2. **Navigation:** tabs in the Header, **Search | Results**, kept in the URL hash (`#search`, `#results`) so a page is linkable and survives reload. No router library needed.
3. **Results page** (data from `fetchResults()`):
   - Header line: device/GPU, corpus size, query set sizes (`query_sets[*].n`) with their descriptions.
   - **Results table:** rows = methods (sorted by `METHOD_ORDER` / the order in `rows`), color dot + label + family tag. Columns: for each query set, MRR@10 and Recall@10; then latency (mean ms). Bold the best value in each column. Group the ablation rows (`summary.ablation`) under a sub-heading "Hybrid ablation".
   - **Charts:** for each `charts[i]`, show `<img src="/api/results/files/{file}">` with the title and caption. The PNGs have white backgrounds, so put them in a white rounded card that also looks fine in dark mode.
   - **Examples:** a card per `examples[i]`: the query, query set, a chip per method showing its rank (or "not in top 100"), and the explanation.
   - If `/api/results` returns 404: an empty state saying "Results not generated yet. Run `python eval/run_eval.py` and `python eval/charts.py`."
4. Check both light and dark themes, and the layout at phone width (one column, no horizontal scroll).

## Done when
- [ ] `npm run build` and `npm run lint` pass with no errors.
- [ ] With the real backend: any 1–4 methods can be picked; the hybrid switch works; per-method timing shows.
- [ ] An aspect search shows the extra column with highlighted spans.
- [ ] Papers without an arXiv ID show "No arXiv ID" and no broken link.
- [ ] Non-cosine scores display sensibly.
- [ ] The Results page renders P3's `summary.json` and all charts; the 404 state works.
- [ ] Nothing in the UI says "three methods" anymore.
