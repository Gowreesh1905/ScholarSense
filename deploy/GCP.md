# Deploying ScholarSense on Google Cloud (Compute Engine)

About 45 minutes, most of it waiting for the build. You only need a browser. The server
downloads the code from GitHub and the precomputed search index (~900 MB) from Hugging Face
(`Gowreesh05/scholarsense-index`).

**Cost:** an `e2-standard-2` (2 CPUs, 8 GB RAM) is roughly **$0.07–0.08/hour, ~$2/day** while
running, plus ~$1–2/month for the 30 GB disk. New Google Cloud accounts get a **free trial with
$300 of credit for 90 days**. During the trial **you are not charged** unless you click
"Activate full account" yourself; when the credit or the 90 days run out, the servers just stop.

---

## Part 1: Account and project (~10 min)

**1. Free trial.** Go to https://console.cloud.google.com and sign in with a Google account.
Click **Start free** / **Try for free**, fill in the form. Google asks for a card **to verify
you're a real person**; the trial doesn't charge it. **Don't** click "Activate full account" later.

**2. Project.** Top bar → project picker → **New project** → name `scholarsense` → **Create**,
then select it in the project picker.

**3. Budget alert.** Search **Budgets & alerts** → **Create budget** → name `scholarsense` →
**Next** → amount **50** (USD) → **Next** → keep the 50% / 90% / 100% alerts → **Finish**.
(Alerts only email you; they don't stop anything. The trial itself is the hard limit.)

---

## Part 2: Create the server (~5 min)

**4.** Search **Compute Engine** → **VM instances**. If asked, click **Enable** (Compute Engine
API) and wait a minute. Then **Create instance** and set:

| Section | Field | Value |
|---|---|---|
| Machine configuration | Name | `scholarsense` |
| | Region / Zone | **asia-south1 (Mumbai)**, any zone |
| | Machine type | **E2** → **e2-standard-2** (2 vCPU, **8 GB** memory) |
| OS and storage | Boot disk → **Change** | **Ubuntu**, version **Ubuntu 24.04 LTS (x86/64)**, size **30 GB** → **Select** |
| Networking | Firewall | ✅ **Allow HTTP traffic** |

Click **Create**. Wait for the green tick next to the VM (~1 min).

**5. Open the terminal.** In the VM list, click **SSH** on the `scholarsense` row → **Authorize**.
A terminal window opens in the browser.

---

## Part 3: Install and start the app (in that terminal, ~25 min)

Paste each block, press Enter, wait for it to finish.

**6. Install Docker and git** (~2 min):
```
curl -fsSL https://get.docker.com | sudo sh
sudo apt-get install -y git
```

**7. Download the code and name the index:**
```
git clone -b deploy https://github.com/Gowreesh1905/ScholarSense.git
cd ScholarSense
echo "Gowreesh05/scholarsense-index" > deploy/index_repo.txt
```

**8. Build** (~15–25 min: website, Python packages, ~2.5 GB of models, the index). It ends with
`naming to docker.io/library/scholarsense`:
```
sudo docker build -f deploy/Dockerfile -t scholarsense .
```

**9. Start it** (it restarts by itself if the VM reboots):
```
sudo docker run -d --name scholarsense --restart unless-stopped -p 80:7860 scholarsense
sudo docker logs -f scholarsense
```
Wait ~2–4 minutes for **`Application startup complete`**, then press **Ctrl+C** (that only stops
watching the logs; the app keeps running).

**10. Check:**
```
curl -s localhost/api/health | head -c 150; echo
```
It should start with `{"status":"ready","corpus_size":727,...`.

---

## Part 4: The link

**11.** In **VM instances**, copy the **External IP** (e.g. `34.93.12.34`). The site is
**http://34.93.12.34**. Type `http://` yourself; "Not secure" is expected for plain HTTP.
Anyone with the link can use it.

---

## Day-to-day

| Task | How |
|---|---|
| **Stop paying for the server** | VM instances → tick `scholarsense` → **Stop** (only the disk keeps costing, ~$1–2/month) |
| Start it again | **Start / Resume**; the app starts by itself in ~3 min. **The External IP may change**; to keep one fixed address, VPC network → **IP addresses** → reserve the VM's IP as **static**. |
| See the app's logs | **SSH**, then `sudo docker logs --tail 50 scholarsense` |
| Update to the latest code | **SSH**, then:<br>`cd ScholarSense && git pull`<br>`sudo docker build -f deploy/Dockerfile -t scholarsense .`<br>`sudo docker rm -f scholarsense`<br>`sudo docker run -d --name scholarsense --restart unless-stopped -p 80:7860 scholarsense` |
| Delete everything when the project is over | VM instances → tick → **Delete** (also deletes its disk). Release any reserved static IP. |

## Troubleshooting

| Problem | Fix |
|---|---|
| Site doesn't open | Use `http://` not `https://`. VM → Edit → Networking → **Allow HTTP traffic** must be ticked. Check step 10 works. |
| Page opens but says "Backend offline" | Still starting: `sudo docker logs --tail 20 scholarsense`, wait for `Application startup complete`. |
| `e2-standard-2` unavailable in the zone | Pick another zone in the same region, or region `asia-south2` / `us-central1`. |
| Build: `no space left on device` | Boot disk too small: VM → Edit → boot disk → 30 GB, or recreate with 30 GB. |
| Container exits, logs end in `Killed` | Out of memory: the VM needs 8 GB (`e2-standard-2`), not `e2-micro`/`e2-small`/`e2-medium`. |
| `permission denied ... docker.sock` | Put `sudo` before `docker`. |
| Hybrid + re-rank takes a few seconds | Normal on CPU; the other methods answer in well under a second. |
