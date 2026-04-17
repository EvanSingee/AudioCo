"""
Agent 3 – Data Quality Filter (D2 in-place or → filtered subset)

Inspects each speaker cluster and removes:
  • Silent / near-silent chunks  (RMS below floor)
  • Low-SNR chunks                (SNR < threshold)
  • Chunks with speaker overlap   (detected via energy variance heuristic)
  • Too-short / too-long chunks   (outside MIN/MAX_DURATION_S)

Produces a quality_report.json summarising per-file decisions.
"""

import os
import json
import shutil
import logging
import numpy as np
from tqdm import tqdm

import config
import utils

logger = logging.getLogger("Agent3-Quality")


# ─── Overlap detection ───────────────────────────────────────────────────────

def _spectral_flatness_variance(wav: np.ndarray, sr: int) -> float:
    """
    Proxy for overlap: highly varied spectral flatness across short frames
    suggests multiple simultaneous speakers.  Returns the coefficient of
    variation of per-frame spectral flatness.
    """
    import librosa

    frame_len = int(sr * 0.025)
    hop_len   = int(sr * 0.010)
    S = np.abs(librosa.stft(wav, n_fft=frame_len, hop_length=hop_len))
    flatness = librosa.feature.spectral_flatness(S=S)[0]  # shape (T,)

    mu = flatness.mean()
    if mu < 1e-10:
        return 0.0
    return float(flatness.std() / mu)


def has_overlap(wav: np.ndarray, sr: int, threshold: float = config.OVERLAP_RATIO_MAX) -> bool:
    """
    Heuristic: if spectral flatness variance coefficient > threshold,
    the chunk likely contains overlapping speech.
    """
    cv = _spectral_flatness_variance(wav, sr)
    return cv > threshold


# ─── Per-file quality check ───────────────────────────────────────────────────

def assess_chunk(wav_path: str) -> dict:
    """
    Return a dict with quality flags for a single chunk.
    'keep' = True if the chunk passes all filters.
    """
    try:
        wav, sr = utils.load_wav(wav_path)
    except Exception as e:
        return {"path": wav_path, "keep": False, "reason": f"load_error: {e}"}

    duration = len(wav) / sr

    result = {"path": wav_path, "keep": True, "reason": "ok", "duration_s": round(duration, 2)}

    # Duration check
    if duration < config.MIN_DURATION_S:
        return {**result, "keep": False, "reason": "too_short"}
    if duration > config.MAX_DURATION_S:
        return {**result, "keep": False, "reason": "too_long"}

    # RMS / silence check
    rms = utils.compute_rms(wav)
    result["rms"] = round(float(rms), 6)
    if rms < config.RMS_FLOOR:
        return {**result, "keep": False, "reason": "silent"}

    # SNR check
    snr = utils.compute_snr(wav)
    result["snr_db"] = round(float(snr), 2)
    if snr < config.SNR_THRESHOLD_DB:
        return {**result, "keep": False, "reason": f"low_snr({snr:.1f}dB)"}

    # Overlap check
    if has_overlap(wav, sr):
        return {**result, "keep": False, "reason": "overlap_detected"}

    return result


# ─── Main run ─────────────────────────────────────────────────────────────────

def run(data_dir: str = config.D2_DIR, remove_bad: bool = True):
    """
    Walk through all speaker directories in data_dir, assess each chunk,
    optionally remove failing ones, and write quality_report.json.
    """
    logger.info("=== Agent 3: Data Quality Filter started ===")

    report = {"kept": [], "removed": []}
    kept_count = removed_count = 0

    wav_files = []
    for dirpath, _, filenames in os.walk(data_dir):
        for f in sorted(filenames):
            if f.endswith(".wav"):
                wav_files.append(os.path.join(dirpath, f))

    if not wav_files:
        logger.warning(f"No WAV files found in {data_dir}")
        return

    for wav_path in tqdm(wav_files, desc="Quality check"):
        assessment = assess_chunk(wav_path)

        if assessment["keep"]:
            report["kept"].append(assessment)
            kept_count += 1
        else:
            report["removed"].append(assessment)
            removed_count += 1
            if remove_bad:
                try:
                    os.remove(wav_path)
                except OSError:
                    pass

    # Remove now-empty speaker directories
    if remove_bad:
        for dirpath, dirnames, filenames in os.walk(data_dir, topdown=False):
            if not os.listdir(dirpath) and dirpath != data_dir:
                os.rmdir(dirpath)

    # Write report
    report_path = os.path.join(data_dir, "quality_report.json")
    with open(report_path, "w") as f:
        json.dump(
            {
                "summary": {
                    "total":   kept_count + removed_count,
                    "kept":    kept_count,
                    "removed": removed_count,
                    "keep_rate": round(kept_count / max(1, kept_count + removed_count), 3),
                },
                "details": report,
            },
            f,
            indent=2,
        )

    logger.info(
        f"=== Agent 3 complete: {kept_count} kept, {removed_count} removed "
        f"({removed_count / max(1, kept_count + removed_count) * 100:.1f}% filtered) ==="
    )
    logger.info(f"Report → {report_path}")
    return report


if __name__ == "__main__":
    run()
