"""
Agent 1 – VAD Segmentation (D0 → D1)

Uses WebRTC VAD to strip silence and split recordings into speech-only
chunks.  Falls back to librosa energy-based VAD when webrtcvad is not
available.

Output layout:
    data/D1_segmented/
        <recording_stem>/
            chunk_0001.wav
            chunk_0002.wav
            ...
"""

import os
import json
import logging
import struct
import numpy as np
from tqdm import tqdm

try:
    import webrtcvad
    HAS_WEBRTCVAD = True
except ImportError:
    HAS_WEBRTCVAD = False
    logging.warning("webrtcvad not installed – falling back to librosa VAD")

import librosa

import config
import utils

logger = logging.getLogger("Agent1-VAD")


# ─── WebRTC VAD helpers ───────────────────────────────────────────────────────

def _frame_generator(frame_ms: int, audio_bytes: bytes, sample_rate: int):
    """Yield audio frames of exactly frame_ms milliseconds."""
    n = int(sample_rate * frame_ms / 1000) * 2  # 16-bit = 2 bytes/sample
    offset = 0
    while offset + n <= len(audio_bytes):
        yield audio_bytes[offset : offset + n]
        offset += n


def _vad_collector(sample_rate, frame_ms, padding_chunks, vad, frames):
    """
    Smooth VAD decisions with a ring-buffer and return speech segments
    as lists of (start_sample, end_sample) pairs.
    """
    num_padding = padding_chunks
    ring_buffer = []
    triggered = False
    voiced_frames = []
    segments = []
    current_start = 0

    samples_per_frame = int(sample_rate * frame_ms / 1000)

    for i, frame in enumerate(frames):
        is_speech = vad.is_speech(frame, sample_rate)
        if not triggered:
            ring_buffer.append((i, is_speech))
            num_voiced = sum(1 for _, s in ring_buffer if s)
            if num_voiced > 0.9 * num_padding:
                triggered = True
                current_start = (ring_buffer[0][0]) * samples_per_frame
                voiced_frames = []  # start collecting
                ring_buffer = []
        else:
            voiced_frames.append(frame)
            ring_buffer.append((i, is_speech))
            if len(ring_buffer) > num_padding:
                ring_buffer.pop(0)
            num_unvoiced = sum(1 for _, s in ring_buffer if not s)
            if num_unvoiced > 0.9 * num_padding:
                triggered = False
                end_sample = i * samples_per_frame
                segments.append((current_start, end_sample))
                ring_buffer = []
                voiced_frames = []

    if triggered and voiced_frames:
        segments.append((current_start, len(frames) * samples_per_frame))

    return segments


def webrtcvad_segments(wav: np.ndarray, sr: int) -> list[tuple[int, int]]:
    """Return list of (start_sample, end_sample) speech segments via WebRTC VAD."""
    vad = webrtcvad.Vad(config.VAD_AGGRESSIVENESS)
    # Convert float32 → int16 PCM bytes
    pcm_int16 = (wav * 32767).astype(np.int16)
    audio_bytes = pcm_int16.tobytes()

    frames = list(_frame_generator(config.VAD_FRAME_MS, audio_bytes, sr))
    segments = _vad_collector(
        sr,
        config.VAD_FRAME_MS,
        config.VAD_PADDING_CHUNKS,
        vad,
        frames,
    )
    # Filter too-short segments
    min_samples = int(config.VAD_MIN_SPEECH_MS * sr / 1000)
    segments = [(s, e) for s, e in segments if (e - s) >= min_samples]
    return segments


# ─── Librosa fallback VAD ─────────────────────────────────────────────────────

def librosa_vad_segments(wav: np.ndarray, sr: int) -> list[tuple[int, int]]:
    """
    Energy-based VAD using librosa.effects.split.
    Returns list of (start_sample, end_sample).
    """
    intervals = librosa.effects.split(wav, top_db=30)
    min_samples = int(config.VAD_MIN_SPEECH_MS * sr / 1000)
    return [(s, e) for s, e in intervals if (e - s) >= min_samples]


# ─── Main segmentation logic ──────────────────────────────────────────────────

def segment_file(wav_path: str, out_dir: str) -> list[str]:
    """
    Segment a single WAV file into speech chunks.
    Returns list of output chunk paths.
    """
    wav, sr = utils.load_wav(wav_path)

    if HAS_WEBRTCVAD:
        segments = webrtcvad_segments(wav, sr)
        # If WebRTC finds nothing (e.g. synthetic / tonal audio), fall back to librosa
        if not segments:
            segments = librosa_vad_segments(wav, sr)
    else:
        segments = librosa_vad_segments(wav, sr)

    # Last-resort: treat whole file as one segment
    if not segments:
        segments = [(0, len(wav))]

    os.makedirs(out_dir, exist_ok=True)
    out_paths = []

    max_samples = int(config.MAX_DURATION_S * sr)

    for idx, (start, end) in enumerate(segments):
        chunk = wav[start:end]
        # Split long chunks further
        sub_chunks = [
            chunk[i : i + max_samples]
            for i in range(0, len(chunk), max_samples)
        ]
        for sub_idx, sub in enumerate(sub_chunks):
            if len(sub) < int(config.MIN_DURATION_S * sr):
                continue
            name = f"chunk_{idx:04d}_{sub_idx:02d}.wav"
            out_path = os.path.join(out_dir, name)
            utils.save_wav(sub, out_path, sr)
            out_paths.append(out_path)

    return out_paths


def run(src_dir: str = config.RAW_DIR, dst_dir: str = config.D1_DIR):
    """
    Run Agent 1 across all recordings in src_dir.
    Converts non-WAV files first, then segments.
    """
    logger.info("=== Agent 1: VAD Segmentation started ===")
    audio_files = utils.find_audio_files(src_dir)

    if not audio_files:
        logger.warning(f"No audio files found in {src_dir}")
        return

    manifest = {}  # recording → list of chunk paths

    for file_path in tqdm(audio_files, desc="Segmenting"):
        stem = os.path.splitext(os.path.basename(file_path))[0]
        ext = os.path.splitext(file_path)[1].lower()

        wav_path = file_path
        if ext != ".wav":
            wav_path = os.path.join(dst_dir, "_converted", f"{stem}.wav")
            try:
                utils.convert_to_wav(file_path, wav_path)
            except Exception as e:
                logger.error(f"Conversion failed for {file_path}: {e}")
                continue

        out_dir = os.path.join(dst_dir, stem)
        try:
            chunks = segment_file(wav_path, out_dir)
            manifest[stem] = chunks
            logger.info(f"  {stem}: {len(chunks)} chunks")
        except Exception as e:
            logger.error(f"Segmentation failed for {wav_path}: {e}")

    # Save manifest
    manifest_path = os.path.join(dst_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    total_chunks = sum(len(v) for v in manifest.values())
    logger.info(f"=== Agent 1 complete: {total_chunks} chunks from {len(manifest)} recordings ===")
    logger.info(f"Manifest saved → {manifest_path}")
    return manifest


if __name__ == "__main__":
    run()
