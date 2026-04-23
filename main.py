"""
main.py – CLI pipeline runner

Usage
-----
# Full pipeline (with Drive download):
python main.py --drive "https://drive.google.com/drive/folders/..."

# Skip download (data already in data/D0_raw/):
python main.py

# Run only specific stages:
python main.py --from 2       # start from Agent 2
python main.py --only 1       # run Agent 1 only
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

import config
from utils.audio import load_audio, save_audio, make_mixture
from utils.metrics import mean_sdr
from model.separator import separate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger("Pipeline")


# ─── Evaluation helper ────────────────────────────────────────────────────────

def evaluate_stage(stage_label: str, audio_dir: str, n_pairs: int = None) -> float | None:
    """
    Build synthetic (mix, s1, s2) from up to *n_pairs* random WAV pairs in
    *audio_dir*, run Conv-TasNet, and return mean SDR.
    """
    n_pairs = n_pairs or config.EVAL_PAIRS
    wavs = sorted(Path(audio_dir).glob("*.wav"))
    if len(wavs) < 2:
        logger.warning(f"{stage_label}: need ≥2 WAV files, found {len(wavs)}")
        return None

    import random
    random.seed(0)
    random.shuffle(wavs)
    pairs = list(zip(wavs[::2], wavs[1::2]))[:n_pairs]

    sdrs = []
    for p1, p2 in pairs:
        try:
            s1_ref, sr = load_audio(str(p1))
            s2_ref, _  = load_audio(str(p2))
            mix, s1_ref, s2_ref = make_mixture(s1_ref, s2_ref)
            s1_est, s2_est = separate(mix, sr)

            refs = np.stack([s1_ref, s2_ref])
            ests = np.stack([s1_est, s2_est])
            sdrs.append(mean_sdr(refs, ests))
        except Exception as e:
            logger.warning(f"Eval pair failed: {e}")

    if not sdrs:
        return None
    val = float(np.mean(sdrs))
    logger.info(f"[{stage_label}] mean SDR = {val:.2f} dB  (over {len(sdrs)} pairs)")
    return val


def evaluate_d3() -> float | None:
    """Evaluate on structured D3 pairs (ground-truth mix/s1/s2)."""
    mix_files = sorted(Path(config.D3_MIX_DIR).glob("*.wav"))
    s1_files  = sorted(Path(config.D3_S1_DIR).glob("*.wav"))
    s2_files  = sorted(Path(config.D3_S2_DIR).glob("*.wav"))

    n = min(len(mix_files), len(s1_files), len(s2_files), config.EVAL_PAIRS)
    if n == 0:
        logger.warning("D3: no matching triples found")
        return None

    sdrs = []
    for mf, s1f, s2f in zip(mix_files[:n], s1_files[:n], s2_files[:n]):
        try:
            mix, sr = load_audio(str(mf))
            s1_ref, _ = load_audio(str(s1f))
            s2_ref, _ = load_audio(str(s2f))
            s1_est, s2_est = separate(mix, sr)

            t = min(len(s1_ref), len(s1_est))
            refs = np.stack([s1_ref[:t], s2_ref[:t]])
            ests = np.stack([s1_est[:t], s2_est[:t]])
            sdrs.append(mean_sdr(refs, ests))
        except Exception as e:
            logger.warning(f"D3 eval failed: {e}")

    if not sdrs:
        return None
    val = float(np.mean(sdrs))
    logger.info(f"[D3] mean SDR = {val:.2f} dB  (over {len(sdrs)} pairs)")
    return val


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AudioCo pipeline")
    parser.add_argument("--drive", default=None,
                        help="Google Drive folder URL (skip to use existing D0_raw/)")
    parser.add_argument("--from", dest="from_stage", type=int, default=1,
                        choices=[1, 2, 3], help="Start from agent N")
    parser.add_argument("--only", type=int, default=None,
                        choices=[1, 2, 3], help="Run only agent N")
    parser.add_argument("--eval", action="store_true",
                        help="Run evaluation after pipeline")
    args = parser.parse_args()

    from_stage = args.from_stage
    to_stage   = args.only if args.only else 3

    results = {}

    # ── Agent 1 ───────────────────────────────────────────────────────────────
    if from_stage <= 1 <= to_stage:
        from agents.agent1_ingest import run as run_a1
        results["agent1"] = run_a1(drive_url=args.drive)

    # ── Agent 2 ───────────────────────────────────────────────────────────────
    if from_stage <= 2 <= to_stage:
        from agents.agent2_clean import run as run_a2
        results["agent2"] = run_a2()

    # ── Agent 3 ───────────────────────────────────────────────────────────────
    if from_stage <= 3 <= to_stage:
        from agents.agent3_structure import run as run_a3
        results["agent3"] = run_a3()

    # ── Evaluation ────────────────────────────────────────────────────────────
    if args.eval or to_stage == 3:
        logger.info("\n=== Running Evaluation ===")
        sdr_results = {
            "D0_raw":       evaluate_stage("D0", config.D0_DIR),
            "D1_segments":  evaluate_stage("D1", config.D1_DIR),
            "D2_clean":     evaluate_stage("D2", config.D2_DIR),
            "D3_structured": evaluate_d3(),
        }
        results["sdr"] = sdr_results

        print("\n┌─────────────────────────────┐")
        print("│   SDR Comparison (dB)       │")
        print("├──────────────┬──────────────┤")
        for k, v in sdr_results.items():
            val = f"{v:+.2f}" if v is not None else "  N/A  "
            print(f"│ {k:<12} │  {val:>8}    │")
        print("└──────────────┴──────────────┘")

        out_path = os.path.join(config.RESULTS_DIR, "sdr_results.json")
        with open(out_path, "w") as f:
            json.dump(sdr_results, f, indent=2)
        logger.info(f"Results saved → {out_path}")

    print("\nDone ✅")


if __name__ == "__main__":
    main()
