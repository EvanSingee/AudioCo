"""
Agent 3 – Dataset Structuring
  Reads clean clips from D2_clean/ and creates D3/:
    D3/mix/   – mixed (2-speaker) audio
    D3/s1/    – speaker-1 source
    D3/s2/    – speaker-2 source

  Strategy
  --------
  1. Try filename-based detection: files with "mix" / "mixed" / "combined"
     in their name are used as mix; everything else as single-speaker.
  2. If fewer than 2 single-speaker clips exist, create synthetic pairs
     (mix = s1 + s2, perfectly aligned) from all available clips.
  3. Creates EVAL_PAIRS (mix, s1, s2) triples – used later for SDR.
"""
import os
import random
import logging
from pathlib import Path

import numpy as np

import config
from utils.audio import load_audio, save_audio, make_mixture

logger = logging.getLogger("Agent3-Structure")

random.seed(42)


def _is_mixed(name: str) -> bool:
    lname = name.lower()
    return any(kw in lname for kw in ("mix", "mixed", "combined", "both", "overlap"))


def run(progress_cb=None) -> dict:
    """
    Returns
    -------
    dict with keys: n_pairs, d3_mix, d3_s1, d3_s2
    """
    def _prog(frac, msg):
        logger.info(msg)
        if progress_cb:
            progress_cb(frac, msg)

    files = sorted(Path(config.D2_DIR).glob("*.wav"))
    if not files:
        raise RuntimeError(f"No files in {config.D2_DIR}. Run Agent 2 first.")

    # ── Classify ──────────────────────────────────────────────────────────────
    mix_files    = [f for f in files if _is_mixed(f.name)]
    single_files = [f for f in files if not _is_mixed(f.name)]

    logger.info(f"Detected {len(mix_files)} mixed, {len(single_files)} single-speaker files")

    # ── Decide pairing strategy ───────────────────────────────────────────────
    pairs_created = 0

    if len(single_files) >= 2:
        # Pair up single-speaker clips → synthetic mix + real sources
        random.shuffle(single_files)
        # Make pairs
        pair_list = list(zip(single_files[::2], single_files[1::2]))
        n = min(len(pair_list), max(config.EVAL_PAIRS, 10))
        pair_list = pair_list[:n]

        for i, (p1, p2) in enumerate(pair_list):
            try:
                s1_audio, sr = load_audio(str(p1))
                s2_audio, _  = load_audio(str(p2))
                mix, s1, s2  = make_mixture(s1_audio, s2_audio)

                tag = f"pair{i:03d}"
                save_audio(mix, os.path.join(config.D3_MIX_DIR, f"{tag}.wav"), sr)
                save_audio(s1,  os.path.join(config.D3_S1_DIR,  f"{tag}.wav"), sr)
                save_audio(s2,  os.path.join(config.D3_S2_DIR,  f"{tag}.wav"), sr)
                pairs_created += 1
            except Exception as e:
                logger.warning(f"Pair {i} failed: {e}")

            _prog(0.5 + 0.5 * (i + 1) / len(pair_list),
                  f"Built {pairs_created} pairs…")

    # ── Also pair mix files with two random single speakers ───────────────────
    if mix_files and len(single_files) >= 2:
        random.shuffle(mix_files)
        for i, mf in enumerate(mix_files[:5]):
            try:
                # Use raw mix as the mix track, synthesise references
                # (real mix ≈ s1+s2, so we still need synthetic s1/s2)
                s1_path = random.choice(single_files)
                remaining = [f for f in single_files if f != s1_path]
                if not remaining:
                    continue
                s2_path = random.choice(remaining)

                mix_audio, sr = load_audio(str(mf))
                s1_audio, _   = load_audio(str(s1_path))
                s2_audio, _   = load_audio(str(s2_path))
                _, s1, s2     = make_mixture(s1_audio, s2_audio)

                # Trim mix to match source length
                n   = min(len(mix_audio), len(s1))
                mix = mix_audio[:n]
                s1  = s1[:n]
                s2  = s2[:n]

                tag = f"real_mix{i:03d}"
                save_audio(mix, os.path.join(config.D3_MIX_DIR, f"{tag}.wav"), sr)
                save_audio(s1,  os.path.join(config.D3_S1_DIR,  f"{tag}.wav"), sr)
                save_audio(s2,  os.path.join(config.D3_S2_DIR,  f"{tag}.wav"), sr)
                pairs_created += 1
            except Exception as e:
                logger.warning(f"Real-mix pair {i} failed: {e}")

    # ── Fallback: all-synthetic from any files ────────────────────────────────
    if pairs_created == 0:
        logger.warning("Falling back: creating synthetic pairs from all available files")
        all_files = list(files)
        random.shuffle(all_files)
        for i in range(0, len(all_files) - 1, 2):
            try:
                s1_audio, sr = load_audio(str(all_files[i]))
                s2_audio, _  = load_audio(str(all_files[i + 1]))
                mix, s1, s2  = make_mixture(s1_audio, s2_audio)
                tag = f"synth{i//2:03d}"
                save_audio(mix, os.path.join(config.D3_MIX_DIR, f"{tag}.wav"), sr)
                save_audio(s1,  os.path.join(config.D3_S1_DIR,  f"{tag}.wav"), sr)
                save_audio(s2,  os.path.join(config.D3_S2_DIR,  f"{tag}.wav"), sr)
                pairs_created += 1
            except Exception as e:
                logger.warning(f"Fallback pair failed: {e}")
            if pairs_created >= config.EVAL_PAIRS + 5:
                break

    _prog(1.0, f"✅ Agent 3 done — {pairs_created} (mix, s1, s2) pairs in D3/")
    return {
        "n_pairs":  pairs_created,
        "d3_mix":   config.D3_MIX_DIR,
        "d3_s1":    config.D3_S1_DIR,
        "d3_s2":    config.D3_S2_DIR,
    }
