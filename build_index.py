#!/usr/bin/env python
# build_index_annoy.py
# ----------------------------------------------------------
# Build an Annoy index from spotify_data.csv
#  • Scales numeric columns
#  • 1-hot encodes the 'genre' column
#  • L2-normalises each track vector

from pathlib import Path
import numpy as np
import pandas as pd
from annoy import AnnoyIndex
import joblib

# ── Config ─────────────────────────────────────────────────
CSV_PATH = Path(__file__).with_name("spotify_data.csv")
ANNOY_PATH = Path(__file__).with_name("annoy_index.ann")
META_PATH = Path(__file__).with_name("metadata.joblib")
METRIC = "angular"
N_TREES = 50
META_COLS = {"track_id", "track_name", "artist_name", "genre"}

# ── 1 · Load data ─────────────────────────────────────────
df = (
    pd.read_csv(CSV_PATH)
      .loc[:, lambda d: ~d.columns.str.match(r"Unnamed")]
      .dropna(subset=["track_id", "track_name", "artist_name"])
      .reset_index(drop=True)
)

# Ensure genre is a string and fill NAs
df["genre"] = df["genre"].fillna("unknown").astype(str)

# ── 2 · 1-hot encode genre ────────────────────────────────
genre_ohe = pd.get_dummies(df["genre"], prefix="genre", dtype=np.float32)
df = pd.concat([df.drop(columns=["genre"]), genre_ohe], axis=1)

# ── 3 · Build the feature matrix ──────────────────────────
num_cols = df.select_dtypes(include=[np.number]).columns
feature_cols = [c for c in num_cols if c not in META_COLS]  # includes OHE
X = df[feature_cols].to_numpy(dtype=np.float32)

# Column-wise min-max scaling → [0, 1]
mn, mx = X.min(axis=0), X.max(axis=0)
rng = np.where(mx - mn == 0, 1.0, mx - mn)
X = (X - mn) / rng

# L2 normalise each vector (needed for angular metric)
norms = np.linalg.norm(X, axis=1, keepdims=True)
X = np.divide(X, norms, out=np.zeros_like(X), where=norms != 0)

# ── 4 · Build Annoy index ─────────────────────────────────
f_dim = X.shape[1]
ann = AnnoyIndex(f_dim, METRIC)
for i, vec in enumerate(X):
    ann.add_item(i, vec)
ann.build(N_TREES)
ann.save(str(ANNOY_PATH))

# ── 5 · Persist metadata ─────────────────────────────────
joblib.dump(
    dict(
        track_ids=df["track_id"].tolist(),
        track_names=df["track_name"].tolist(),
        artist_names=df["artist_name"].tolist(),
        id2idx={tid: i for i, tid in enumerate(df["track_id"])},
        feature_cols=feature_cols,
        f_dim=f_dim,
    ),
    META_PATH,
)

print(f"✅ Annoy index built — {len(df):,} tracks, {f_dim}-D  →  {ANNOY_PATH}")
