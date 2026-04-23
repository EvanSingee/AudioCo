"""Central configuration for the AudioCo speech-separation pipeline."""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "data")

# ── Audio ────────────────────────────────────────────────────────────────────
SAMPLE_RATE = 16_000
CHANNELS    = 1
BIT_DEPTH   = 16

# ── VAD ─────────────────────────────────────────────────────────────────────
VAD_AGGRESSIVENESS    = 2      # 0-3  (3 = most aggressive)
VAD_FRAME_MS          = 30     # must be 10 / 20 / 30
VAD_PADDING_MS        = 300
MIN_SPEECH_DURATION_S = 1.0

# ── Data-quality thresholds (Agent 2) ────────────────────────────────────────
MIN_DURATION_S   = 1.0
MIN_RMS          = 0.004
MIN_SNR_DB       = 4.0

# ── Model ────────────────────────────────────────────────────────────────────
MODEL_NAME = "mpariente/ConvTasNet_WHAM!_sepclean"

# ── Evaluation ───────────────────────────────────────────────────────────────
EVAL_PAIRS = 3           # number of (mix, s1, s2) pairs used per stage

# ── Data directories (created automatically) ─────────────────────────────────
D0_DIR      = os.path.join(DATA_DIR, "D0_raw")
D1_DIR      = os.path.join(DATA_DIR, "D1_segments")
D2_DIR      = os.path.join(DATA_DIR, "D2_clean")
D3_MIX_DIR  = os.path.join(DATA_DIR, "D3", "mix")
D3_S1_DIR   = os.path.join(DATA_DIR, "D3", "s1")
D3_S2_DIR   = os.path.join(DATA_DIR, "D3", "s2")
RESULTS_DIR = os.path.join(DATA_DIR, "results")

for _d in [D0_DIR, D1_DIR, D2_DIR, D3_MIX_DIR, D3_S1_DIR, D3_S2_DIR, RESULTS_DIR]:
    os.makedirs(_d, exist_ok=True)
