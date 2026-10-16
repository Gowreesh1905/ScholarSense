# Deploying ScholarSense (free)

The whole app (website + API) runs as **one server**: the backend serves the built website
and the `/api` routes on the same address. Two free ways to put it online:

| | Hugging Face Spaces (main) | Cloudflare Tunnel (backup for review day) |
|---|---|---|
| Cost | free, no credit card | free, no account |
| Runs on | Hugging Face's free CPU (2 vCPU, 16 GB RAM) | your laptop (GPU, so Hybrid + re-rank is fast) |
| Online | always (sleeps after ~48 h without visitors, wakes in a few minutes) | only while your laptop and the command are running |
| Link | `https://<you>-scholarsense.hf.space` | a random `https://….trycloudflare.com` |

Nothing is recomputed on the server: it loads the precomputed embeddings, tokens and the
100k index (built by the laptop cluster) from a Hugging Face **dataset**. The cluster code
itself isn't deployed; its results are shown on the Results page.

---

## A. Hugging Face Spaces (do this once; ~15 min of your time + ~15 min of building)

Do this on **laptop A** (the one with the built index: `cluster/output/index_manifest.json`).

**1. Make a free account** at https://huggingface.co/join (no card needed).

**2. Create a write token:** https://huggingface.co/settings/tokens → *Create new token* →
type **Write** → copy it.

**3. Log in from the project folder** (PowerShell):
```
cd D:\ScholarSense
pip install -U huggingface_hub
hf auth login
```
Paste the token when asked. Answer **n** to "add token as git credential".

**4. Check what will be uploaded, then publish:**
```
python deploy\publish.py --dry-run
python deploy\publish.py
```
The first publish uploads ~900 MB of index (a few minutes on a good connection), then ~4 MB
of code. It prints two links:

- the Space page, `https://huggingface.co/spaces/<you>/scholarsense`: click **Logs** to
  watch the build (~10–15 minutes the first time: website, Python packages, models);
- the site, `https://<you>-scholarsense.hf.space`: it shows "Starting" for ~2–4 minutes
  while the models load, then the app.

**Updating later** (you changed code or results): run `python deploy\publish.py --code-only`.
The Space rebuilds by itself. Re-run without `--code-only` only if you rebuilt the index.

**If the free tier is too slow or you don't need 100k papers:** `python deploy\publish.py --no-scale`
deploys only the 727-paper corpus (smaller, starts faster).

### On review day
Open the site **10 minutes early**: after ~48 h without visitors the free Space sleeps,
and waking it (loading the models) takes a few minutes. Hybrid + re-rank takes a few
seconds per query on the free CPU; every other method answers in well under a second.

---

## B. Cloudflare Tunnel (backup: a public link to your laptop)

No account needed. Uses your laptop's GPU.

**1. Install once:**
```
winget install --id Cloudflare.cloudflared
```
(Close and reopen PowerShell afterwards.)

**2. Build the website once** (and after frontend changes):
```
cd D:\ScholarSense\frontend
npm run build
```

**3. Start the app** (window 1; the backend now also serves the website):
```
cd D:\ScholarSense
python backend\app.py
```
Wait for `Application startup complete`, then check http://localhost:8000 in your browser.

**4. Open the tunnel** (window 2):
```
cloudflared tunnel --url http://localhost:8000
```
It prints a link like `https://random-words.trycloudflare.com`. Anyone can open it while
both windows stay open. The link changes every time you start the tunnel.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `publish.py`: "Missing index files" | Run `python backend\app.py` once on this laptop (creates the 727-paper caches) and make sure `cluster\build_index.py` finished. |
| `hf auth login`: command not found | `pip install -U huggingface_hub`, then reopen PowerShell. |
| Space build fails | Open **Logs** on the Space page and send the last lines. |
| Space says "Runtime error" / out of memory | Republish with `--no-scale` (the 100k corpus needs ~2 GB more). |
| Site works but search says "Backend offline" | It's still starting: wait for the Logs to show `Application startup complete`. |
| Local test of the container | `docker build -f deploy/Dockerfile -t scholarsense .` then `docker run -p 7860:7860 scholarsense` and open http://localhost:7860. To build without Hugging Face, first copy the index into `deploy/index_bundle/` (see `deploy/bundle.py`). |
