#!/usr/bin/env python
# app.py  ─────────────────────────────────────────────────────────────
from pathlib import Path
from typing import List, Optional

import joblib
import pandas as pd
from annoy import AnnoyIndex
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── Paths ────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
CSV_PATH = BASE_DIR / "spotify_data.csv"
ANNOY_PATH = BASE_DIR / "annoy_index.ann"
META_PATH = BASE_DIR / "metadata.joblib"
STATIC_DIR = BASE_DIR / "static"

# ── Load metadata & Annoy index ──────────────────────────────────────
meta = joblib.load(META_PATH)
track_ids = meta["track_ids"]
track_names = meta["track_names"]
artist_names = meta["artist_names"]
id2idx = meta["id2idx"]
F_DIM = meta["f_dim"]

ann_index = AnnoyIndex(F_DIM, "angular")
if not ann_index.load(str(ANNOY_PATH)):
    raise RuntimeError("Could not load Annoy index")

# DataFrame only for /search
df = (
    pd.read_csv(CSV_PATH)
      .loc[:, ["track_id", "track_name", "artist_name", "year"]]
      .dropna(subset=["track_id", "track_name"])
)

# ── FastAPI setup ────────────────────────────────────────────────────
app = FastAPI(title="Spotify Recommender (Annoy)")

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

# ── Schemas ──────────────────────────────────────────────────────────


class TrackOption(BaseModel):
    track_id: str
    track_name: str
    artist_name: str
    year: Optional[int] = None


class RecResponse(BaseModel):
    track_id: str
    track_name: str
    artist_name: str
    score: float       # 1 = identical, 0 = very different


# ── Endpoints ─────────────────────────────────────────────────────────
@app.get("/search", response_model=List[TrackOption])
def search_tracks(
    query: str = Query(...,
                       description="Full or partial song title (case-insensitive)"),
    limit: int = Query(20, ge=1, le=100),
):
    hits = (
        df[df["track_name"].str.contains(query, case=False, na=False)]
        .head(limit)
        .to_dict(orient="records")
    )
    if not hits:
        raise HTTPException(status_code=404, detail="No matching tracks found")
    return hits


@app.get("/recommend", response_model=List[RecResponse])
def recommend(
    track_id: str = Query(...,
                          description="Seed Spotify track_id (from /search)"),
    k: int = Query(10, ge=3, le=50),  # ≥3 to fit same-artist guarantee
):
    if track_id not in id2idx:
        raise HTTPException(status_code=404, detail="track_id not found")

    seed_idx = id2idx[track_id]
    seed_artist = artist_names[seed_idx]

    # 1️⃣  Primary neighbours
    idxs, dists = ann_index.get_nns_by_item(
        seed_idx, k + 1, include_distances=True
    )
    neigh_idxs, neigh_dists = idxs[1:], dists[1:]   # drop seed

    # 2️⃣  Two closest tracks by same artist
    same_artist_idxs = [
        i for i, a in enumerate(artist_names)
        if a == seed_artist and i != seed_idx
    ]
    same_artist_dists = [
        ann_index.get_distance(seed_idx, i) for i in same_artist_idxs
    ]
    top_two_sa = sorted(
        zip(same_artist_idxs, same_artist_dists), key=lambda x: x[1]
    )[:2]

    # Merge & de-duplicate
    combined = list(zip(neigh_idxs, neigh_dists))
    for idx, dist in top_two_sa:
        if idx not in neigh_idxs:
            combined.append((idx, dist))

    # Sort by distance & trim to k
    combined = sorted(combined, key=lambda x: x[1])[:k]

    # Angular distance [0,2]  →  similarity [1,0]
    def dist_to_sim(d: float) -> float:
        return round((2 - d) / 2, 4)

    return [
        RecResponse(
            track_id=track_ids[i],
            track_name=track_names[i],
            artist_name=artist_names[i],
            score=dist_to_sim(dist),   # ← renamed field
        )
        for i, dist in combined
    ]


# ── Dev entry-point ──────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
