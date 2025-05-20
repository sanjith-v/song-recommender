#!/usr/bin/env python
# shrink_annoy.py ────────────────────────────────────────────────────
import joblib
import numpy as np
from sklearn.decomposition import PCA
from annoy import AnnoyIndex
from pathlib import Path

# ── Configuration ───────────────────────────────────────────────────
META_PATH = Path("metadata.joblib")
ANN_IN = Path("annoy_index.ann")          # your big index
ANN_OUT = Path("annoy_index_small.ann")    # output
TARGET_DIM = 50    # e.g. reduce from 128 → 50 dims
N_TREES = 10    # fewer trees → smaller index

# ── 1) Load metadata (to get f_dim & item count) ─────────────────
meta = joblib.load(META_PATH)
track_ids = meta["track_ids"]
f_dim = meta["f_dim"]
n_items = len(track_ids)

# ── 2) Load original Annoy index ──────────────────────────────────
orig = AnnoyIndex(f_dim, "angular")
if not orig.load(str(ANN_IN)):
    raise RuntimeError(f"Failed to load {ANN_IN}")

# ── 3) Extract all feature vectors ────────────────────────────────
print(f"Extracting {n_items} vectors of dim {f_dim} …")
X = np.zeros((n_items, f_dim), dtype=np.float32)
for i in range(n_items):
    X[i] = orig.get_item_vector(i)

# ── 4) PCA (optional) ─────────────────────────────────────────────
if f_dim > TARGET_DIM:
    print(f"Applying PCA: {f_dim} → {TARGET_DIM} dimensions")
    pca = PCA(n_components=TARGET_DIM)
    X = pca.fit_transform(X).astype(np.float32)

# ── 5) Build a smaller Annoy index ────────────────────────────────
new_dim = X.shape[1]
print(f"Building new Annoy index: dim={new_dim}, trees={N_TREES}")
small = AnnoyIndex(new_dim, "angular")
for i, vec in enumerate(X):
    small.add_item(i, vec)
small.build(N_TREES)

# ── 6) Save & report ───────────────────────────────────────────────
small.save(str(ANN_OUT))
size_mb = ANN_OUT.stat().st_size / 1_000_000
print(f"Saved {ANN_OUT} ({size_mb:.1f} MB) — ready to upload to S3")
