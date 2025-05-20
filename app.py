#!/usr/bin/env python
# app.py ──────────────────────────────────────────────────────────────
from pathlib import Path
from typing import List, Optional, Generator

import joblib
import pandas as pd
from annoy import AnnoyIndex
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# ── Paths ────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
ANNOY_PATH = BASE_DIR / "annoy_index.ann"
META_PATH = BASE_DIR / "metadata.joblib"
STATIC_DIR = BASE_DIR / "static"

# ── FastAPI app with lifespan loader ─────────────────────────────────


def lifespan(app: FastAPI) -> Generator[None, None, None]:
    """Load index + metadata once per worker; rebuild search DataFrame."""
    # ---------- Metadata ----------
    if not META_PATH.exists():
        raise RuntimeError(f"{META_PATH} missing in slug")
    meta = joblib.load(META_PATH)

    track_ids = meta["track_ids"]
    track_names = meta["track_names"]
    artist_names = meta["artist_names"]
    id2idx = meta["id2idx"]
    f_dim = meta["f_dim"]
    years = meta.get("years")  # optional

    # ---------- Annoy Index ----------
    ann = AnnoyIndex(f_dim, "angular")
    if not ann.load(str(ANNOY_PATH)):
        raise RuntimeError("Could not load Annoy index")

    # ---------- Build search DataFrame ----------
    data = {
        "track_id":    track_ids,
        "track_name":  track_names,
        "artist_name": artist_names,
    }
    if years is not None:
        data["year"] = years

    df = pd.DataFrame(data)
    df["track_name_lc"] = df["track_name"].str.lower()  # helper col

    # ---------- Expose to app.state ----------
    app.state.df = df
    app.state.ann_index = ann
    app.state.track_ids = track_ids
    app.state.track_names = track_names
    app.state.artist_names = artist_names
    app.state.id2idx = id2idx

    yield  # ---- server starts here ----
    # (optional) cleanup on shutdown


app = FastAPI(title="Spotify Recommender (Annoy)", lifespan=lifespan)

# ── CORS & static files ─────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def root():
    return FileResponse(STATIC_DIR / "index.html")

# ── Pydantic models ─────────────────────────────────────────────────


class TrackOption(BaseModel):
    track_id: str
    track_name: str
    artist_name: str
    year: Optional[int] = None


class RecResponse(BaseModel):
    track_id: str
    track_name: str
    artist_name: str
    score: float  # similarity 1→identical, 0→different

# ── Endpoints ───────────────────────────────────────────────────────


@app.get("/search", response_model=List[TrackOption])
def search_tracks(
    query: str = Query(..., description="Song title (case-insensitive)"),
    limit: int = Query(20, ge=1, le=100),
):
    df = app.state.df
    hits = (
        df[df["track_name_lc"].str.contains(query.lower())]
        .head(limit)
        .drop(columns="track_name_lc")
        .to_dict(orient="records")
    )
    if not hits:
        raise HTTPException(404, "No matching tracks found")
    return hits


@app.get("/recommend", response_model=List[RecResponse])
def recommend(
    track_id: str = Query(..., description="Seed Spotify track_id"),
    k: int = Query(10, ge=3, le=50),
):
    ann = app.state.ann_index
    id2idx = app.state.id2idx
    track_ids = app.state.track_ids
    track_names = app.state.track_names
    artist_names = app.state.artist_names

    if track_id not in id2idx:
        raise HTTPException(404, "track_id not found")

    seed_idx = id2idx[track_id]
    seed_artist = artist_names[seed_idx]

    # 1. primary neighbours
    idxs, dists = ann.get_nns_by_item(seed_idx, k + 1, include_distances=True)
    neigh_idxs = idxs[1:]
    neigh_dists = dists[1:]

    # 2. two closest tracks by same artist
    sa_idxs = [i for i, a in enumerate(
        artist_names) if a == seed_artist and i != seed_idx]
    sa_dists = [ann.get_distance(seed_idx, i) for i in sa_idxs]
    top_two = sorted(zip(sa_idxs, sa_dists), key=lambda x: x[1])[:2]

    # merge + dedupe
    combined = {i: d for i, d in zip(neigh_idxs, neigh_dists)}
    for i, d in top_two:
        combined.setdefault(i, d)

    # best k
    items = sorted(combined.items(), key=lambda x: x[1])[:k]
    def to_sim(d): return round((2 - d) / 2, 4)

    return [
        RecResponse(
            track_id=track_ids[i],
            track_name=track_names[i],
            artist_name=artist_names[i],
            score=to_sim(d),
        )
        for i, d in items
    ]


# ── Local dev entrypoint ────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
