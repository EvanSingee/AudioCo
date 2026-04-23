"""
Agent 2 – Data Cleaning
  • Remove clips shorter than MIN_DURATION_S
  • Remove low-energy / noisy clips
  • Normalize amplitude
  • Output: data/D2_clean/
"""
import os
import logging
from pathlib import Path

import config
from utils.audio import load_audio, save_audio, rms, estimate_snr, normalize

logger = logging.getLogger("Agent2-Clean")


def run(progress_cb=None) -> dict:
    """
    Returns
    -------
    dict with keys: n_input, n_kept, n_removed, d2_dir
    """
    def _prog(frac, msg):
        logger.info(msg)
        if progress_cb:
            progress_cb(frac, msg)

    files = sorted(Path(config.D1_DIR).glob("*.wav"))
    if not files:
        raise RuntimeError(f"No WAV files found in {config.D1_DIR}. Run Agent 1 first.")

    n_kept = n_removed = 0

    for i, fpath in enumerate(files):
        try:
            audio, sr = load_audio(str(fpath))
            duration  = len(audio) / sr

            # ── Filters ──────────────────────────────────────────────────────
            if duration < config.MIN_DURATION_S:
                logger.debug(f"SKIP {fpath.name}: too short ({duration:.2f}s)")
                n_removed += 1
                continue

            if rms(audio) < config.MIN_RMS:
                logger.debug(f"SKIP {fpath.name}: low energy")
                n_removed += 1
                continue

            snr = estimate_snr(audio, sr)
            if snr < config.MIN_SNR_DB:
                logger.debug(f"SKIP {fpath.name}: low SNR ({snr:.1f} dB)")
                n_removed += 1
                continue

            # ── Normalize & save ─────────────────────────────────────────────
            audio = normalize(audio)
            dst   = os.path.join(config.D2_DIR, fpath.name)
            save_audio(audio, dst, sr)
            n_kept += 1

        except Exception as e:
            logger.warning(f"Error processing {fpath.name}: {e}")
            n_removed += 1

        _prog((i + 1) / len(files),
              f"Cleaned {i+1}/{len(files)} — kept {n_kept}, removed {n_removed}")

    _prog(1.0, f"✅ Agent 2 done — {n_kept} clean clips in D2_clean/")
    return {
        "n_input": len(files),
        "n_kept": n_kept,
        "n_removed": n_removed,
        "d2_dir": config.D2_DIR,
    }
