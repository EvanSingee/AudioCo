"""Core audio processing utilities."""
import os
import struct
import logging
import warnings

import numpy as np
import soundfile as sf
import librosa
import config

logger = logging.getLogger(__name__)

SUPPORTED_EXT = {".wav", ".mp3", ".mp4", ".m4a", ".mpeg", ".flac", ".ogg"}


# ─── I/O ─────────────────────────────────────────────────────────────────────

def load_audio(path: str, sr: int = None) -> tuple[np.ndarray, int]:
    """Load any audio/video file as mono float32 at *sr* Hz."""
    sr = sr or config.SAMPLE_RATE
    audio, _ = librosa.load(path, sr=sr, mono=True)
    return audio.astype(np.float32), sr


def save_audio(audio: np.ndarray, path: str, sr: int = None):
    """Write mono float32 array as 16-bit WAV."""
    sr = sr or config.SAMPLE_RATE
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    sf.write(path, audio.astype(np.float32), sr, subtype="PCM_16")


def convert_to_wav(src: str, dst: str, sr: int = None) -> str:
    """Convert any audio file to mono 16 kHz WAV. Returns *dst*."""
    audio, sample_rate = load_audio(src, sr)
    save_audio(audio, dst, sample_rate)
    return dst


# ─── VAD ─────────────────────────────────────────────────────────────────────

def vad_segment(
    audio: np.ndarray,
    sr: int = 16_000,
    aggressiveness: int = 2,
    frame_ms: int = 30,
    padding_ms: int = 300,
    min_duration_s: float = 1.0,
) -> list[np.ndarray]:
    """
    Split *audio* into speech-only chunks using webrtcvad.
    Falls back to energy-based segmentation on any error.
    """
    try:
        import webrtcvad
        return _vad_webrtc(audio, sr, aggressiveness, frame_ms,
                           padding_ms, min_duration_s)
    except Exception as e:
        logger.warning(f"webrtcvad unavailable ({e}), using energy VAD")
        return _vad_energy(audio, sr, min_duration_s)


def _vad_webrtc(audio, sr, aggressiveness, frame_ms,
                padding_ms, min_duration_s):
    import webrtcvad
    assert sr in (8000, 16000, 32000, 48000)

    vad = webrtcvad.Vad(aggressiveness)
    frame_len   = int(sr * frame_ms / 1000)
    pad_frames  = int(padding_ms / frame_ms)

    pcm = (audio * 32768).clip(-32768, 32767).astype(np.int16)

    # Build (sample_offset, is_speech) list
    voiced = []
    for start in range(0, len(pcm) - frame_len + 1, frame_len):
        frame  = pcm[start : start + frame_len]
        fbytes = struct.pack(f"{frame_len}h", *frame)
        voiced.append((start, vad.is_speech(fbytes, sr)))

    if not voiced:
        return [audio] if len(audio) / sr >= min_duration_s else []

    # Sliding-window smoothing → collect speech regions
    segments, ring, triggered, seg_start = [], [], False, 0
    for i, (start, is_speech) in enumerate(voiced):
        if not triggered:
            ring.append((start, is_speech))
            if len(ring) > pad_frames:
                ring.pop(0)
            if sum(s for _, s in ring) > 0.8 * pad_frames:
                triggered = True
                seg_start = ring[0][0]
                ring = []
        else:
            ring.append((start, is_speech))
            if len(ring) > pad_frames:
                ring.pop(0)
            if sum(1 - s for _, s in ring) > 0.8 * pad_frames:
                triggered = False
                end = start + frame_len
                chunk = audio[seg_start:end]
                if len(chunk) / sr >= min_duration_s:
                    segments.append(chunk)
                ring = []

    if triggered:
        chunk = audio[seg_start:]
        if len(chunk) / sr >= min_duration_s:
            segments.append(chunk)

    return segments if segments else ([audio] if len(audio) / sr >= min_duration_s else [])


def _vad_energy(audio, sr, min_duration_s, hop=512, ratio=0.08):
    """Simple energy-threshold VAD fallback."""
    rms = librosa.feature.rms(y=audio, hop_length=hop)[0]
    thresh = ratio * np.max(rms) if np.max(rms) > 0 else 1e-4
    samples = librosa.frames_to_samples(np.arange(len(rms)), hop_length=hop)

    in_speech, seg_start, segments = False, 0, []
    for t, r in zip(samples, rms):
        if not in_speech and r > thresh:
            in_speech, seg_start = True, max(0, t - hop * 2)
        elif in_speech and r <= thresh:
            in_speech = False
            chunk = audio[seg_start : min(len(audio), t + hop * 2)]
            if len(chunk) / sr >= min_duration_s:
                segments.append(chunk)
    if in_speech:
        chunk = audio[seg_start:]
        if len(chunk) / sr >= min_duration_s:
            segments.append(chunk)
    return segments if segments else ([audio] if len(audio) / sr >= min_duration_s else [])


# ─── Quality helpers ─────────────────────────────────────────────────────────

def rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(audio ** 2)))


def estimate_snr(audio: np.ndarray, sr: int = 16_000) -> float:
    """Estimate SNR via speech/non-speech frame energy ratio."""
    try:
        import webrtcvad
        vad = webrtcvad.Vad(2)
        flen = int(sr * 0.03)
        pcm  = (audio * 32768).clip(-32768, 32767).astype(np.int16)
        sp, no = [], []
        for i in range(0, len(pcm) - flen + 1, flen):
            frame  = pcm[i:i+flen]
            fbytes = struct.pack(f"{flen}h", *frame)
            e = float(np.mean(frame.astype(np.float32) ** 2))
            (sp if vad.is_speech(fbytes, sr) else no).append(e)
        if sp and no and np.mean(no) > 0:
            return float(10 * np.log10(np.mean(sp) / np.mean(no)))
    except Exception:
        pass
    return float(20 * np.log10(rms(audio) + 1e-8))


def normalize(audio: np.ndarray, target_rms: float = 0.1) -> np.ndarray:
    r = rms(audio)
    if r < 1e-8:
        return audio
    return (audio * (target_rms / r)).clip(-1.0, 1.0).astype(np.float32)


# ─── Mixture builder ─────────────────────────────────────────────────────────

def make_mixture(s1: np.ndarray, s2: np.ndarray,
                 snr_db: float = 0.0
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (mix, s1_scaled, s2_scaled) of equal length."""
    n = min(len(s1), len(s2))
    s1, s2 = s1[:n].copy(), s2[:n].copy()

    r1 = rms(s1)
    if r1 > 1e-8:
        s1 /= r1
    r2 = rms(s2)
    if r2 > 1e-8:
        s2 /= r2
        s2 *= 10 ** (-snr_db / 20.0)

    mix = (s1 + s2) * 0.5
    return mix.astype(np.float32), (s1 * 0.5).astype(np.float32), (s2 * 0.5).astype(np.float32)
