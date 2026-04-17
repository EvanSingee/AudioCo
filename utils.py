"""
utils.py – Shared audio utilities used by all agents.
"""

import os
import logging
import numpy as np
import soundfile as sf
import librosa
import config

# ─── Configure pydub to use bundled ffmpeg (imageio-ffmpeg) ─────────────────
# Must happen BEFORE pydub is imported so it can locate ffmpeg.
try:
    import imageio_ffmpeg as _iio
    _ffmpeg_bin = _iio.get_ffmpeg_exe()
    _ffmpeg_dir = os.path.dirname(_ffmpeg_bin)
    # Prepend ffmpeg dir to PATH so pydub's which() call finds it
    os.environ["PATH"] = _ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
    # Also create a symlink named 'ffmpeg' in the dir if binary has a longer name
    _symlink = os.path.join(_ffmpeg_dir, "ffmpeg")
    if not os.path.exists(_symlink):
        try:
            os.symlink(_ffmpeg_bin, _symlink)
        except OSError:
            pass
except Exception:
    _ffmpeg_bin = "ffmpeg"

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from pydub import AudioSegment
    try:
        AudioSegment.converter = _ffmpeg_bin
        AudioSegment.ffmpeg    = _ffmpeg_bin
    except Exception:
        pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)


# ─── Format Conversion ───────────────────────────────────────────────────────

SUPPORTED_FORMATS = {".wav", ".mp4", ".m4a", ".mpeg", ".mp3", ".flac", ".ogg"}


def convert_to_wav(src_path: str, dst_path: str) -> str:
    """
    Convert any audio/video file to mono 16-bit 16 kHz WAV.
    Returns the destination path.
    """
    ext = os.path.splitext(src_path)[1].lower()
    if ext not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported format: {ext}")

    audio = AudioSegment.from_file(src_path)
    audio = (
        audio.set_frame_rate(config.SAMPLE_RATE)
             .set_channels(config.CHANNELS)
             .set_sample_width(config.BIT_DEPTH // 8)
    )
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    audio.export(dst_path, format="wav")
    return dst_path


def load_wav(path: str) -> tuple[np.ndarray, int]:
    """Load WAV → (waveform float32, sample_rate)."""
    wav, sr = librosa.load(path, sr=config.SAMPLE_RATE, mono=True)
    return wav.astype(np.float32), sr


def save_wav(wav: np.ndarray, path: str, sr: int = config.SAMPLE_RATE):
    """Save numpy waveform as 16-bit WAV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Normalise to prevent clipping
    peak = np.abs(wav).max()
    if peak > 1.0:
        wav = wav / peak
    sf.write(path, wav, sr, subtype="PCM_16")


# ─── Audio Analysis ──────────────────────────────────────────────────────────

def compute_rms(wav: np.ndarray) -> float:
    return float(np.sqrt(np.mean(wav ** 2)))


def compute_snr(signal: np.ndarray, noise_floor: float = config.RMS_FLOOR) -> float:
    """Rough SNR estimate assuming noise = minimum-energy 10% of frames."""
    frame_len = int(config.SAMPLE_RATE * 0.02)  # 20 ms frames
    frames = [
        signal[i : i + frame_len]
        for i in range(0, len(signal) - frame_len, frame_len)
    ]
    energies = [np.mean(f ** 2) for f in frames]
    energies.sort()
    noise_energy = max(np.mean(energies[: max(1, len(energies) // 10)]), noise_floor)
    signal_energy = max(np.mean(energies), noise_floor)
    return 10 * np.log10(signal_energy / noise_energy)


def normalize_loudness(wav: np.ndarray, target_rms: float = 0.1) -> np.ndarray:
    rms = compute_rms(wav)
    if rms < config.RMS_FLOOR:
        return wav
    return wav * (target_rms / rms)


def mix_signals(s1: np.ndarray, s2: np.ndarray, snr_db: float = 0.0) -> np.ndarray:
    """
    Mix two signals at a given SNR (s1 is reference).
    The shorter signal is zero-padded to match the longer one.
    """
    n = max(len(s1), len(s2))
    s1 = np.pad(s1, (0, n - len(s1)))
    s2 = np.pad(s2, (0, n - len(s2)))

    s1 = normalize_loudness(s1)
    rms_s1 = compute_rms(s1)
    target_rms_s2 = rms_s1 * (10 ** (-snr_db / 20))
    s2 = normalize_loudness(s2, target_rms=target_rms_s2)

    mix = s1 + s2
    # Prevent clipping
    peak = np.abs(mix).max()
    if peak > 0.99:
        mix /= peak
    return mix


# ─── File Utilities ──────────────────────────────────────────────────────────

def find_audio_files(root: str) -> list[str]:
    paths = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if os.path.splitext(f)[1].lower() in SUPPORTED_FORMATS:
                paths.append(os.path.join(dirpath, f))
    return sorted(paths)


def duration_seconds(path: str) -> float:
    info = sf.info(path)
    return info.duration
