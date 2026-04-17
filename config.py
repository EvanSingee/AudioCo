"""
AudioCo - Data-Centric Speech Separation Pipeline
Shared configuration across all agents.
"""

import os

# ─── Directory Layout ────────────────────────────────────────────────────────

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")

RAW_DIR    = os.path.join(DATA_DIR, "D0_raw")          # raw recordings
D1_DIR     = os.path.join(DATA_DIR, "D1_segmented")    # VAD chunks
D2_DIR     = os.path.join(DATA_DIR, "D2_clustered")    # speaker clusters
D3_DIR     = os.path.join(DATA_DIR, "D3_structured")   # mix/s1/s2 pairs

CLEAN_DIR  = os.path.join(DATA_DIR, "clean_reference") # optional clean set
LOGS_DIR   = os.path.join(BASE_DIR, "logs")
MODELS_DIR = os.path.join(BASE_DIR, "models")

for d in [RAW_DIR, D1_DIR, D2_DIR, D3_DIR, CLEAN_DIR, LOGS_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

# ─── Audio Settings ──────────────────────────────────────────────────────────

SAMPLE_RATE     = 16000   # Hz – standard for speech models
CHANNELS        = 1       # mono
BIT_DEPTH       = 16      # bits
MAX_DURATION_S  = 30      # max segment length (seconds)
MIN_DURATION_S  = 1.0     # min usable segment (seconds)

# ─── Agent 1 – VAD Segmentation ──────────────────────────────────────────────

VAD_AGGRESSIVENESS  = 3       # webrtcvad: 0-3 (3 = most aggressive)
VAD_FRAME_MS        = 30      # frame size in ms (10, 20, or 30)
VAD_PADDING_CHUNKS  = 10      # silence padding chunks around speech
VAD_MIN_SPEECH_MS   = 500     # minimum speech duration to keep

# ─── Agent 2 – Speaker Clustering ────────────────────────────────────────────

EMBEDDING_MODEL     = "speechbrain/spkrec-ecapa-voxceleb"
N_CLUSTERS          = None    # None = auto-detect via silhouette score
MAX_CLUSTERS        = 10
CLUSTER_METHOD      = "agglomerative"   # "kmeans" | "agglomerative" | "spectral"
MIN_CHUNKS_PER_SPK  = 2       # discard speakers with fewer than this many chunks

# ─── Agent 3 – Quality Filter ────────────────────────────────────────────────

SNR_THRESHOLD_DB    = 5.0     # minimum SNR to keep a chunk
RMS_FLOOR           = 1e-4    # silence floor
OVERLAP_RATIO_MAX   = 1.5      # spectral flatness CV threshold; raise to avoid
                               # false positives on AM-modulated or tonal signals

# ─── Agent 4 – Dataset Builder ───────────────────────────────────────────────

MIX_SNR_RANGE       = (-5, 5)     # dB range for mixing s1 and s2
TRAIN_RATIO         = 0.8
VAL_RATIO           = 0.1
TEST_RATIO          = 0.1
PAIRS_PER_SPEAKER   = 20          # synthetic mix pairs per speaker combination

# ─── Evaluation ──────────────────────────────────────────────────────────────

CONV_TASNET_MODEL   = "mpariente/ConvTasNet_WHAM!_sepclean"  # pretrained HF id
EVAL_METRICS        = ["sdr", "si_sdr", "stoi"]
