"""
ScholarSense API server.

    python backend/app.py

Serves (see team/CONTRACT.md §6):
    GET  /api/health              -> engine status (corpus, device, methods, aspects)
    POST /api/search              -> {query, k, methods, aspect} -> ranked results per method
    GET  /api/results             -> results/summary.json (404 until evaluation has run)
    GET  /api/results/files/<f>   -> chart images etc. from results/
"""

import json
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from engine import ROOT, SearchEngine

RESULTS_DIR = ROOT / "results"
MAX_METHODS = 4

engine: Optional[SearchEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    engine = SearchEngine()
    yield


app = FastAPI(title="ScholarSense API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    k: int = Field(default=5, ge=1, le=20)
    methods: Optional[list[str]] = None
    aspect: str = "all"


def _ready_engine() -> SearchEngine:
    if engine is None:
        raise HTTPException(503, "Engine still starting up")
    return engine


@app.get("/api/health")
def health():
    return _ready_engine().status()


@app.post("/api/search")
def search(req: SearchRequest):
    eng = _ready_engine()

    query = req.query.strip()
    if not query:
        raise HTTPException(400, "Query must not be empty")

    methods = None
    if req.methods is not None:
        methods = list(dict.fromkeys(req.methods))  # drop duplicates, keep order
        if not 1 <= len(methods) <= MAX_METHODS:
            raise HTTPException(400, f"Choose between 1 and {MAX_METHODS} methods (got {len(methods)})")
        unknown = [m for m in methods if m not in eng.searchers]
        if unknown:
            raise HTTPException(
                400,
                f"Unknown or unavailable method(s): {', '.join(unknown)}. "
                f"Available: {', '.join(eng.searchers)}",
            )

    if req.aspect not in eng.aspects:
        raise HTTPException(400, f"Unknown aspect {req.aspect!r}. Available: {', '.join(eng.aspects)}")

    return eng.search(query, k=req.k, methods=methods, aspect=req.aspect)


@app.get("/api/results")
def results():
    path = RESULTS_DIR / "summary.json"
    if not path.exists():
        raise HTTPException(404, "No results yet")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(500, f"results/summary.json is not valid JSON: {exc}")


# check_dir=False: the server starts even before the evaluation creates results/.
app.mount("/api/results/files", StaticFiles(directory=RESULTS_DIR, check_dir=False), name="results-files")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
