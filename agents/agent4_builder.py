"""
Agent 4 – Dataset Builder (D2 → D3)  ★ CRITICAL ★

Converts quality-filtered, per-speaker chunks into a supervised
speech-separation dataset of the form:

    data/D3_structured/
        train/
            mix/  0001.wav  ...   # s1 + s2 mixed
            s1/   0001.wav  ...   # clean source 1
            s2/   0001.wav  ...   # clean source 2
        val/
            mix/ s1/ s2/
        test/
            mix/ s1/ s2/
        metadata.json

Algorithm:
  1. For each pair of distinct speakers (A, B), randomly pick segments.
  2. Mix them at a random SNR in config.MIX_SNR_RANGE.
  3. Trim/pad both sources to the same length as the mix.
  4. Save mix, s1, s2 to the appropriate split directory.
"""

import os
import json
import random
import logging
import numpy as np
from itertools import combinations
from tqdm import tqdm

import config
import utils

logger = logging.getLogger("Agent4-Builder")

SPLITS = {
    "train": config.TRAIN_RATIO,
    "val":   config.VAL_RATIO,
    "test":  config.TEST_RATIO,
}


# ─── helpers ─────────────────────────────────────────────────────────────────

def _get_speaker_chunks(data_dir: str) -> dict[str, list[str]]:
    """Return {speaker_id: [list of .wav paths]} from D2 directory."""
    speaker_map = {}
    for name in sorted(os.listdir(data_dir)):
        spk_dir = os.path.join(data_dir, name)
        if not os.path.isdir(spk_dir) or name.startswith("_"):
            continue
        wavs = sorted(
            os.path.join(spk_dir, f)
            for f in os.listdir(spk_dir)
            if f.endswith(".wav")
        )
        if wavs:
            speaker_map[name] = wavs
    return speaker_map


def _make_pair(
    s1_path: str,
    s2_path: str,
    mix_snr_db: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load both sources, mix them, align lengths.
    Returns (mix, s1, s2) as float32 numpy arrays.
    """
    s1, sr = utils.load_wav(s1_path)
    s2, _  = utils.load_wav(s2_path)

    # Align to the same length
    n = min(len(s1), len(s2), int(config.MAX_DURATION_S * sr))
    s1, s2 = s1[:n], s2[:n]

    # Normalise each source independently
    s1 = utils.normalize_loudness(s1)
    s2 = utils.normalize_loudness(s2)

    mix = utils.mix_signals(s1, s2, snr_db=mix_snr_db)
    return mix, s1, s2


def _split_assignment(idx: int, total: int) -> str:
    """Assign pair index to a split by ratio."""
    r = idx / total
    if r < config.TRAIN_RATIO:
        return "train"
    elif r < config.TRAIN_RATIO + config.VAL_RATIO:
        return "val"
    else:
        return "test"


# ─── Main build ──────────────────────────────────────────────────────────────

def run(src_dir: str = config.D2_DIR, dst_dir: str = config.D3_DIR):
    logger.info("=== Agent 4: Dataset Builder started ===")

    speaker_chunks = _get_speaker_chunks(src_dir)
    if len(speaker_chunks) < 2:
        logger.error("Need at least 2 speakers to build mixture dataset.")
        return

    logger.info(f"Found {len(speaker_chunks)} speakers: {list(speaker_chunks.keys())}")

    # Create directory tree
    for split in SPLITS:
        for subdir in ("mix", "s1", "s2"):
            os.makedirs(os.path.join(dst_dir, split, subdir), exist_ok=True)

    speaker_pairs = list(combinations(speaker_chunks.keys(), 2))
    rng = random.Random(42)
    metadata = []

    global_idx = 0
    total_pairs_estimate = len(speaker_pairs) * config.PAIRS_PER_SPEAKER

    for spk_a, spk_b in tqdm(speaker_pairs, desc="Speaker pairs"):
        chunks_a = speaker_chunks[spk_a]
        chunks_b = speaker_chunks[spk_b]

        n_pairs = min(config.PAIRS_PER_SPEAKER, len(chunks_a), len(chunks_b))
        # Shuffle independently so pairs are diverse
        sample_a = rng.choices(chunks_a, k=n_pairs)
        sample_b = rng.choices(chunks_b, k=n_pairs)

        for s1_path, s2_path in zip(sample_a, sample_b):
            snr_db = rng.uniform(*config.MIX_SNR_RANGE)
            try:
                mix, s1, s2 = _make_pair(s1_path, s2_path, snr_db)
            except Exception as e:
                logger.warning(f"Pair failed ({s1_path}, {s2_path}): {e}")
                continue

            split = _split_assignment(global_idx, total_pairs_estimate)
            fname = f"{global_idx:06d}.wav"

            utils.save_wav(mix, os.path.join(dst_dir, split, "mix", fname))
            utils.save_wav(s1,  os.path.join(dst_dir, split, "s1",  fname))
            utils.save_wav(s2,  os.path.join(dst_dir, split, "s2",  fname))

            metadata.append({
                "id":      global_idx,
                "split":   split,
                "filename": fname,
                "speaker_1": spk_a,
                "speaker_2": spk_b,
                "s1_src":  s1_path,
                "s2_src":  s2_path,
                "mix_snr_db": round(snr_db, 2),
            })
            global_idx += 1

    # Save metadata
    meta_path = os.path.join(dst_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(
            {
                "total_pairs": global_idx,
                "split_counts": {
                    s: sum(1 for m in metadata if m["split"] == s)
                    for s in SPLITS
                },
                "speakers": list(speaker_chunks.keys()),
                "samples": metadata,
            },
            f,
            indent=2,
        )

    logger.info(f"=== Agent 4 complete: {global_idx} mixture pairs created ===")
    logger.info(f"Metadata → {meta_path}")
    return metadata


if __name__ == "__main__":
    run()
