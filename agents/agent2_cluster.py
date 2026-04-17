"""
Agent 2 – Speaker Clustering (D1 → D2)

Extracts speaker embeddings and clusters chunks into per-speaker directories.

Embedding method:
  - 40-dim MFCCs + deltas + delta-deltas (120 features total) via librosa
  - Mean and std pooled over time → 240-dim speaker vector
  - No external model downloads required (fully offline)
  - Comparable to i-vectors for short-to-medium utterances

Output layout:
    data/D2_clustered/
        speaker_00/
            chunk_xxxx_xx.wav
        speaker_01/
            ...
        cluster_map.json
"""

import os
import json
import shutil
import logging
import numpy as np
from tqdm import tqdm
import librosa

import config
import utils

logger = logging.getLogger("Agent2-Cluster")


# ─── sklearn helpers ─────────────────────────────────────────────────────────

def _get_sklearn():
    from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import normalize
    return AgglomerativeClustering, KMeans, SpectralClustering, silhouette_score, normalize


# ─── Embedding extraction (MFCC-based, offline) ──────────────────────────────

def extract_embedding(_, wav_path: str) -> np.ndarray:
    """
    Extract a 240-dim speaker embedding:
      40 MFCCs + 40 delta + 40 delta-delta  →  mean & std over time  →  240-dim.
    The first argument (_) is unused (kept for API compatibility).
    """
    wav, sr = utils.load_wav(wav_path)

    # Compute MFCCs
    mfcc = librosa.feature.mfcc(y=wav, sr=sr, n_mfcc=40)          # (40, T)
    delta  = librosa.feature.delta(mfcc)                            # (40, T)
    delta2 = librosa.feature.delta(mfcc, order=2)                  # (40, T)

    features = np.concatenate([mfcc, delta, delta2], axis=0)       # (120, T)

    # Mean + std pooling over time → 240-dim embedding
    emb = np.concatenate([features.mean(axis=1), features.std(axis=1)])
    return emb.astype(np.float32)


def _get_model():
    """No-op — embedding is self-contained. Returns None as placeholder."""
    logger.info("Using offline MFCC-based speaker embedding (no model download needed).")
    return None




# ─── Cluster-count selection ──────────────────────────────────────────────────

def optimal_n_clusters(X: np.ndarray, max_k: int) -> int:
    """
    Find the best number of clusters using silhouette score over K-Means.
    Returns the k with the highest silhouette.
    """
    _, _, _, silhouette_score, normalize = _get_sklearn()
    X_norm = normalize(X)
    from sklearn.cluster import KMeans

    best_k, best_score = 2, -1.0
    for k in range(2, min(max_k + 1, len(X))):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X_norm)
        score = silhouette_score(X_norm, labels)
        if score > best_score:
            best_score, best_k = score, k
    logger.info(f"Auto-selected k={best_k} (silhouette={best_score:.3f})")
    return best_k


# ─── Clustering ──────────────────────────────────────────────────────────────

def cluster_embeddings(embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
    """Run the configured clustering algorithm; return integer label array."""
    AgglomerativeClustering, KMeans, SpectralClustering, _, normalize = _get_sklearn()
    X = normalize(embeddings)

    method = config.CLUSTER_METHOD
    if method == "agglomerative":
        clf = AgglomerativeClustering(n_clusters=n_clusters, metric="cosine", linkage="average")
    elif method == "spectral":
        clf = SpectralClustering(n_clusters=n_clusters, affinity="cosine", random_state=42)
    else:  # kmeans
        clf = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)

    labels = clf.fit_predict(X)
    return labels


# ─── Main run ─────────────────────────────────────────────────────────────────

def run(src_dir: str = config.D1_DIR, dst_dir: str = config.D2_DIR):
    logger.info("=== Agent 2: Speaker Clustering started ===")

    # Collect all chunks from D1
    chunk_paths = []
    for dirpath, _, filenames in os.walk(src_dir):
        for f in sorted(filenames):
            if f.endswith(".wav") and not f.startswith("_"):
                chunk_paths.append(os.path.join(dirpath, f))

    if not chunk_paths:
        logger.warning(f"No WAV chunks found in {src_dir}")
        return

    logger.info(f"Found {len(chunk_paths)} chunks. Extracting embeddings...")
    model = _get_model()

    embeddings = []
    valid_paths = []
    for path in tqdm(chunk_paths, desc="Embeddings"):
        try:
            emb = extract_embedding(model, path)
            embeddings.append(emb)
            valid_paths.append(path)
        except Exception as e:
            logger.warning(f"Embedding failed for {path}: {e}")

    if len(valid_paths) < 2:
        logger.error("Not enough valid chunks for clustering.")
        return

    X = np.stack(embeddings)

    # Determine number of clusters
    n_clusters = config.N_CLUSTERS
    if n_clusters is None:
        n_clusters = optimal_n_clusters(X, config.MAX_CLUSTERS)

    logger.info(f"Clustering {len(X)} embeddings into {n_clusters} speakers...")
    labels = cluster_embeddings(X, n_clusters)

    # ── Copy chunks into per-speaker directories ──────────────────────────────
    os.makedirs(dst_dir, exist_ok=True)
    cluster_map = {}   # path → speaker_id
    speaker_counts = {}

    for path, label in zip(valid_paths, labels):
        spk_id = f"speaker_{label:02d}"
        spk_dir = os.path.join(dst_dir, spk_id)
        os.makedirs(spk_dir, exist_ok=True)

        filename = os.path.basename(path)
        dst_path = os.path.join(spk_dir, filename)
        # Handle duplicate names across recordings
        if os.path.exists(dst_path):
            base, ext = os.path.splitext(filename)
            dst_path = os.path.join(spk_dir, f"{base}_{np.random.randint(1000)}{ext}")

        shutil.copy2(path, dst_path)
        cluster_map[path] = spk_id
        speaker_counts[spk_id] = speaker_counts.get(spk_id, 0) + 1

    # ── Filter sparse speakers ────────────────────────────────────────────────
    sparse = [s for s, c in speaker_counts.items() if c < config.MIN_CHUNKS_PER_SPK]
    if sparse:
        logger.info(f"Removing {len(sparse)} sparse speaker(s): {sparse}")
        for s in sparse:
            shutil.rmtree(os.path.join(dst_dir, s), ignore_errors=True)

    # Save cluster map
    map_path = os.path.join(dst_dir, "cluster_map.json")
    with open(map_path, "w") as f:
        json.dump(
            {
                "n_clusters": n_clusters,
                "method": config.CLUSTER_METHOD,
                "speaker_counts": speaker_counts,
                "chunk_assignments": cluster_map,
            },
            f,
            indent=2,
        )

    logger.info(f"=== Agent 2 complete: {n_clusters} speakers ===")
    logger.info(f"Cluster map → {map_path}")
    return cluster_map


if __name__ == "__main__":
    run()
