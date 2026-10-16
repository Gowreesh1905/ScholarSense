# ScholarSense on Hugging Face Spaces (Docker SDK, free CPU tier: 2 vCPU, 16 GB RAM).
# publish.py uploads this file as the Space's Dockerfile. Local test (from the repo root):
#   docker build -f deploy/Dockerfile -t scholarsense .  &&  docker run -p 7860:7860 scholarsense

# --- 1. build the website ----------------------------------------------------
FROM node:24-slim AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# --- 2. the server -----------------------------------------------------------
FROM python:3.13-slim

# Hugging Face runs the container as user 1000; the app folder must belong to it, or the
# server can't create .cache/ inside (WORKDIR alone would create it owned by root).
RUN useradd -m -u 1000 user && mkdir -p /home/user/app && chown -R user:user /home/user
ENV HOME=/home/user \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOST=0.0.0.0 \
    PORT=7860
WORKDIR /home/user/app

# CPU-only PyTorch (~200 MB instead of ~2.5 GB for the CUDA build), then the rest.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY backend/requirements.txt backend/requirements.txt
RUN pip install -r backend/requirements.txt scipy

COPY --chown=user . .
COPY --from=web --chown=user /web/dist frontend/dist

USER user
# The search index (from the Hugging Face dataset named in deploy/index_repo.txt), the models
# and the NLTK data are baked into the image, so the Space starts without downloading.
RUN python deploy/fetch_assets.py

EXPOSE 7860
CMD ["python", "backend/app.py"]
