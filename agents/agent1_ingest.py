"""
Agent 1 – Ingest & Segment
  • Download from Google Drive
  • Convert .mp4 → mono 16 kHz WAV
  • Apply VAD → split into speech chunks
  • Output: data/D0_raw/  (converted wavs)
           data/D1_segments/  (speech chunks)
"""
import os
import logging
from pathlib import Path

import config
from utils.audio import load_audio, save_audio, convert_to_wav, vad_segment
from utils.drive import download_folder

logger = logging.getLogger("Agent1-Ingest")

SUPPORTED = {".mp4", ".m4a", ".mpeg", ".mp3", ".wav", ".flac", ".ogg"}


def run(drive_url: str = None, progress_cb=None) -> dict:
    """
    Parameters
    ----------
    drive_url   : Google Drive folder URL (optional; skip download if None)
    progress_cb : callable(float, str) for UI progress updates

    Returns
    -------
    dict with keys: n_files, n_chunks, d1_dir
    """
    def _prog(frac, msg):
        logger.info(msg)
        if progress_cb:
            progress_cb(frac, msg)

    # ── 1. Download ──────────────────────────────────────────────────────────
    if drive_url:
        _prog(0.05, "⬇️  Downloading from Google Drive…")
        downloaded = download_folder(drive_url, config.D0_DIR)
        _prog(0.25, f"Downloaded {len(downloaded)} files to D0_raw/")
    else:
        downloaded = [
            str(p) for p in Path(config.D0_DIR).rglob("*")
            if p.suffix.lower() in SUPPORTED
        ]
        _prog(0.10, f"Found {len(downloaded)} files in D0_raw/")

    # Keep only supported audio files
    audio_files = [f for f in downloaded if Path(f).suffix.lower() in SUPPORTED]
    if not audio_files:
        raise RuntimeError(
            f"No supported audio files found in {config.D0_DIR}. "
            f"Supported: {SUPPORTED}"
        )

    # ── 2. Convert to WAV ────────────────────────────────────────────────────
    wav_files = []
    for i, src in enumerate(audio_files):
        stem = Path(src).stem
        dst  = os.path.join(config.D0_DIR, stem + ".wav")
        if not dst.endswith(src):          # avoid overwriting in-place
            try:
                convert_to_wav(src, dst)
                wav_files.append(dst)
            except Exception as e:
                logger.warning(f"Convert failed for {src}: {e}")
        else:
            wav_files.append(src)

        _prog(0.25 + 0.25 * (i + 1) / len(audio_files),
              f"Converted {i+1}/{len(audio_files)}: {Path(src).name}")

    # ── 3. VAD Segmentation ──────────────────────────────────────────────────
    n_chunks = 0
    for i, wav in enumerate(wav_files):
        try:
            audio, sr = load_audio(wav)
            chunks = vad_segment(
                audio, sr,
                aggressiveness=config.VAD_AGGRESSIVENESS,
                frame_ms=config.VAD_FRAME_MS,
                padding_ms=config.VAD_PADDING_MS,
                min_duration_s=config.MIN_SPEECH_DURATION_S,
            )
            stem = Path(wav).stem
            for j, chunk in enumerate(chunks):
                out = os.path.join(config.D1_DIR, f"{stem}_chunk{j:03d}.wav")
                save_audio(chunk, out, sr)
                n_chunks += 1
        except Exception as e:
            logger.warning(f"VAD failed for {wav}: {e}")

        _prog(0.50 + 0.50 * (i + 1) / len(wav_files),
              f"Segmented {i+1}/{len(wav_files)}: {n_chunks} chunks so far")

    _prog(1.0, f"✅ Agent 1 done — {n_chunks} chunks in D1_segments/")
    return {"n_files": len(wav_files), "n_chunks": n_chunks, "d1_dir": config.D1_DIR}
