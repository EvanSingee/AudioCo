"""
scripts/generate_demo_audio.py

Generates synthetic multi-speaker audio for testing the pipeline
without real recordings. Creates 50 WAV files in data/D0_raw/:
  - 5 pseudo-speakers, each with a different fundamental frequency
  - Each file: 5-15 seconds of "speech-like" tone with AM modulation
  - Some files contain 2 mixed speakers (simulating overlap)
"""

import os
import sys
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

SR = config.SAMPLE_RATE
OUT = config.RAW_DIR
os.makedirs(OUT, exist_ok=True)

rng = np.random.default_rng(42)

# Fundamental frequencies for 5 pseudo-speakers (Hz)
SPEAKER_F0 = {
    "spkA": 120.0,
    "spkB": 165.0,
    "spkC": 210.0,
    "spkD": 145.0,
    "spkE": 195.0,
}

def speech_like_signal(duration_s: float, f0: float, sr: int = SR) -> np.ndarray:
    """
    Generate a speech-like signal: harmonics of f0 with AM modulation
    mimicking syllabic rate (~4 Hz) plus some noise.
    """
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    # Harmonics
    sig = sum(
        (1.0 / (k + 1)) * np.sin(2 * np.pi * f0 * (k + 1) * t + rng.uniform(0, 2 * np.pi))
        for k in range(6)
    )
    # Syllabic AM envelope (4 Hz)
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 4.0 * t)
    sig = sig * envelope
    # Add shaped noise
    noise = rng.normal(0, 0.05, len(t))
    sig = sig + noise
    # Add silence gaps randomly
    n_gaps = rng.integers(2, 6)
    for _ in range(n_gaps):
        start = rng.integers(0, max(1, len(t) - sr // 2))
        length = rng.integers(sr // 8, sr // 2)
        sig[start : start + length] = 0.0
    # Normalise
    peak = np.abs(sig).max()
    return (sig / peak * 0.7).astype(np.float32)


idx = 0

# Single-speaker files (40 files, 8 per speaker)
for spk_name, f0 in SPEAKER_F0.items():
    for _ in range(8):
        duration = rng.uniform(5, 12)
        wav = speech_like_signal(duration, f0)
        path = os.path.join(OUT, f"rec_{idx:04d}_{spk_name}.wav")
        sf.write(path, wav, SR, subtype="PCM_16")
        idx += 1
        print(f"  Created: {os.path.basename(path)}  ({duration:.1f}s)")

# Mixed 2-speaker files (10 files)
speakers = list(SPEAKER_F0.items())
for i in range(10):
    spkA_name, f0A = speakers[i % len(speakers)]
    spkB_name, f0B = speakers[(i + 2) % len(speakers)]
    dur = rng.uniform(8, 15)
    wavA = speech_like_signal(dur, f0A)
    wavB = speech_like_signal(dur, f0B)
    # Mix at random gain
    gain = rng.uniform(0.4, 0.9)
    mix = wavA + gain * wavB
    peak = np.abs(mix).max()
    mix = (mix / peak * 0.7).astype(np.float32)
    path = os.path.join(OUT, f"rec_{idx:04d}_{spkA_name}+{spkB_name}_mixed.wav")
    sf.write(path, mix, SR, subtype="PCM_16")
    idx += 1
    print(f"  Created: {os.path.basename(path)}  ({dur:.1f}s) [MIXED]")

print(f"\nTotal: {idx} files → {OUT}")
