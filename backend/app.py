"""
ScholarSense API server.

    python backend/app.py

Serves:
    GET  /api/health   -> engine status (corpus size, device, methods)
    POST /api/search   -> {query, k} -> ranked results from all 3 methods
"""

from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from engine import SearchEngine

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


@app.get("/api/health")
def health():
    if engine is None:
        raise HTTPException(503, "Engine still starting up")
    return engine.status()


@app.post("/api/search")
def search(req: SearchRequest):
    if engine is None:
        raise HTTPException(503, "Engine still starting up")

    query = req.query.strip()
    if not query:
        raise HTTPException(400, "Query must not be empty")

    result = engine.search(query, k=req.k)
    result["methods"] = [
        {**m, "results": [asdict(h) for h in m["results"]]} for m in result["methods"]
    ]
    return result


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
