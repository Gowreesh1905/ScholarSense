# Deploying ScholarSense on AWS (EC2)

About 45 minutes, most of it waiting for the build. You need an AWS account and a browser;
nothing has to be installed on your own computer. The server downloads the code from GitHub
and the precomputed search index (~900 MB) from Hugging Face (`Gowreesh05/scholarsense-index`).

**Cost:** an `m7i-flex.large` (2 CPUs, 8 GB RAM) is about **$0.10/hour, ~$2.30/day** if left
running, plus ~$3/month for disk and IP. **Stop the server when it isn't needed**: a stopped
server only costs the disk.

---

## Part 1: Launch the server (AWS website, ~10 min)

**1. Region.** Sign in at https://console.aws.amazon.com. Top-right, choose
**Asia Pacific (Mumbai) ap-south-1** (or the region closest to you). Keep the same region throughout.

**2. Billing alert (do this first).** Search **Budgets** → **Create budget** → **Use a template**
→ **Monthly cost budget** → amount **20** (USD) → your email → **Create budget**.

**3. Launch the server.** Search **EC2** → **Launch instance**, and fill in:

| Field | Value |
|---|---|
| Name | `scholarsense` |
| Application and OS Images | **Ubuntu** → **Ubuntu Server 24.04 LTS**, architecture **64-bit (x86)** |
| Instance type | **m7i-flex.large** (if not offered: **t3.large**). It must have **8 GiB** memory. |
| Key pair | **Proceed without a key pair** (we'll use the browser terminal) |
| Network settings | tick **Allow SSH traffic from Anywhere** and **Allow HTTP traffic from the internet** |
| Configure storage | change 8 to **30** GiB (gp3) |

Click **Launch instance**, then **View all instances**. Wait until **Instance state = Running**
and **Status check = 3/3 checks passed** (~2 min).

**4. Open the terminal.** Tick the instance → **Connect** (top) → tab **EC2 Instance Connect** →
**Connect**. A black terminal opens in the browser.

---

## Part 2: Install and start the app (browser terminal, ~25 min)

Paste each block, press Enter, wait for it to finish.

**5. Install Docker and git** (~2 min):
```
curl -fsSL https://get.docker.com | sudo sh
sudo apt-get install -y git
```

**6. Download the code and name the index:**
```
git clone -b deploy https://github.com/Gowreesh1905/ScholarSense.git
cd ScholarSense
echo "Gowreesh05/scholarsense-index" > deploy/index_repo.txt
```

**7. Build the app** (~15–25 min: website, Python packages, ~2.5 GB of models, the index):
```
sudo docker build -f deploy/Dockerfile -t scholarsense .
```
It ends with `naming to docker.io/library/scholarsense` / `Successfully tagged`. If it stops with
an error, copy the last ~20 lines and send them.

**8. Start it** (it restarts by itself if the server reboots):
```
sudo docker run -d --name scholarsense --restart unless-stopped -p 80:7860 scholarsense
sudo docker logs -f scholarsense
```
Wait ~2–4 minutes for **`Application startup complete`**, then press **Ctrl+C** (this only stops
watching the logs; the app keeps running).

**9. Check it locally on the server:**
```
curl -s localhost/api/health | head -c 150; echo
```
It should start with `{"status":"ready","corpus_size":727,...`.

---

## Part 3: The link

**10.** In the EC2 instance list, copy the instance's **Public IPv4 address** (e.g. `13.233.45.67`).
The site is **http://13.233.45.67**. Type `http://` yourself; the browser shows "Not secure"
because it's plain HTTP, which is fine. Anyone with the link can use it.

---

## Day-to-day

| Task | How |
|---|---|
| **Stop paying for the server** | EC2 → tick instance → **Instance state → Stop instance** |
| Start it again | **Instance state → Start instance**; the app starts by itself in ~3 min. **The public IP changes** after a stop/start, so share the new one (or attach an **Elastic IP** to keep a fixed address; it costs about the same as the normal IP). |
| See the app's logs | Connect (step 4), then `sudo docker logs --tail 50 scholarsense` |
| Update to the latest code | Connect, then:<br>`cd ScholarSense && git pull`<br>`sudo docker build -f deploy/Dockerfile -t scholarsense .`<br>`sudo docker rm -f scholarsense`<br>`sudo docker run -d --name scholarsense --restart unless-stopped -p 80:7860 scholarsense` |
| Delete everything when the project is over | EC2 → **Instance state → Terminate instance** (deletes the server and its disk; no more charges) |

## Troubleshooting

| Problem | Fix |
|---|---|
| Site doesn't open | Use `http://` not `https://`. Check **Security** tab → security group allows **HTTP port 80 from 0.0.0.0/0**. Check step 9 works. |
| Page opens but says "Backend offline" | It's still starting: `sudo docker logs --tail 20 scholarsense` and wait for `Application startup complete`. |
| `m7i-flex.large` not in the list | Use `t3.large` (also 8 GiB). Smaller types (1–4 GiB) are too small. |
| Build: `no space left on device` | Storage too small: Stop instance → Volumes → Modify → 30 GiB, or relaunch with 30 GiB. |
| Container stops, logs end in `Killed`, or `docker ps` shows it exited (137) | Out of memory: the instance needs 8 GiB. |
| `permission denied ... docker.sock` | Put `sudo` before `docker`. |
| Hybrid + re-rank is slow (a few seconds) | Normal on CPU; the other methods answer in well under a second. |
