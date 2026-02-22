"""
advisor/scripts/ingest_qdrant.py
=================================
Voice-QL Advisor Edition - Qdrant Index Builder
Author: Mohamed Attia (Voice-QI) / DataOverflow Team
Version: 2.0 (Phase 2 Productization)

Loads train.csv, runs Phase 1 preprocess() to get engineered features,
and upserts all rows into a file-based Qdrant collection.

Design note: We use file-based Qdrant (not in-memory) to avoid OOM
under the 1GB RAM constraint while the backend handles Gemini audio streams.

Run once before starting the backend:
    python advisor/scripts/ingest_qdrant.py
"""

import sys
import os
import math

# Resolve project root so we can import solution.py from the top level
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    PayloadSchemaType,
)

# ---------------------------------------------
# Config
# ---------------------------------------------
TRAIN_CSV      = os.path.join(ROOT, "train.csv")
QDRANT_PATH    = os.path.join(ROOT, "advisor", "qdrant_data")
COLLECTION     = "insurance_clients"
BATCH_SIZE     = 10000 # Rows per upsert batch - keeps peak RAM low

# ---------------------------------------------
# 1. Load & preprocess
# ---------------------------------------------
print("Loading train.csv...")
df_raw = pd.read_csv(TRAIN_CSV)
print(f"   -> {len(df_raw):,} rows loaded.")

# Import after sys.path is patched
from solution import preprocess, _load_artifact  # noqa: E402

print("Running Phase 1 preprocess()...")
df_proc = preprocess(df_raw.drop(columns=["Purchased_Coverage_Bundle"], errors="ignore"))

# We only keep numeric feature columns for the vector (same cols the model uses)
art = _load_artifact()
feat_cols = art["feature_cols"]

# Ensure feature cols exist (preprocess may have dropped some)
feat_cols = [c for c in feat_cols if c in df_proc.columns]

# Build the feature matrix - all numeric after preprocess()
X = df_proc[feat_cols].to_numpy(dtype="float32")

# Replace any NaN / inf with 0 to keep Qdrant happy
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

VECTOR_DIM = X.shape[1]
print(f"   -> Feature matrix: {X.shape}  (dim={VECTOR_DIM})")

# ---------------------------------------------
# 2. Build payload columns
# ---------------------------------------------
# Attach key human-readable fields for the LookalikeCard UI
payload_cols = [
    "User_ID",
    "Purchased_Coverage_Bundle",
    "Estimated_Annual_Income",
    "Employment_Status",
    "Region_Code",
    "Adult_Dependents",
    "Child_Dependents",
    "Infant_Dependents",
    "Previous_Claims_Filed",
    "Years_Without_Claims",
    "Deductible_Tier",
    "Acquisition_Channel",
    "Payment_Schedule",
]

# Optimized payload building: select cols, handle NaNs, and convert to dicts
df_payload = df_raw[payload_cols].copy()

# Fill NaNs with None (Qdrant handles None better than NaN)
df_payload = df_payload.replace({np.nan: None})

# Convert to list of dicts
payloads = df_payload.to_dict("records")

# ---------------------------------------------
# 3. Init Qdrant (file-based)
# ---------------------------------------------
os.makedirs(QDRANT_PATH, exist_ok=True)
print(f"Connecting to file-based Qdrant at: {QDRANT_PATH}")
client = QdrantClient(path=QDRANT_PATH)

# Drop existing collection for a clean rebuild
existing = [c.name for c in client.get_collections().collections]
if COLLECTION in existing:
    print(f"   -> Dropping existing '{COLLECTION}' collection...")
    client.delete_collection(COLLECTION)

client.create_collection(
    collection_name=COLLECTION,
    vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
)
print(f"   -> Collection '{COLLECTION}' created (dim={VECTOR_DIM}, cosine).")

# ---------------------------------------------
# 4. Batch upsert
# ---------------------------------------------
total = len(X)
num_batches = math.ceil(total / BATCH_SIZE)
print(f"Uploading {total:,} vectors in {num_batches} batches...")

for batch_idx in range(num_batches):
    start = batch_idx * BATCH_SIZE
    end   = min(start + BATCH_SIZE, total)

    points = [
        PointStruct(
            id=int(i),
            vector=X[i].tolist(),
            payload=payloads[i],
        )
        for i in range(start, end)
    ]

    client.upsert(collection_name=COLLECTION, points=points)
    pct = (end / total) * 100
    print(f"   [{pct:5.1f}%] Batch {batch_idx + 1}/{num_batches} -- rows {start}-{end}")

# ---------------------------------------------
# 5. Quick validation
# ---------------------------------------------
info = client.get_collection(COLLECTION)
count = info.points_count
print(f"\nDone! {count:,} vectors indexed in '{COLLECTION}'.")
print(f"   Path: {QDRANT_PATH}")
print("   Run the backend now: uvicorn advisor.backend.main:app --port 8002 --reload")
