"""
scripts/ablation.py

Runs the ablation study by selectively disabling agents and comparing SDR.

Ablation conditions:
  A) Full pipeline                (D0 → Agent1 → Agent2 → Agent3 → Agent4)
  B) Without Agent 1 (no VAD)    (D0 raw chunks → Agent2 → Agent3 → Agent4)
  C) Without Agent 2 (no cluster)(D1 → random assignment → Agent3 → Agent4)
  D) Without Agent 3 (no filter) (D1 → Agent2 → skip → Agent4)

Results saved to logs/ablation_results.json
"""

import os
import sys
import json
import shutil
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Ablation")


def snapshot_dir(src, tag):
    """Copy a dataset version directory to a tagged backup."""
    dst = src.rstrip("/") + f"_ablation_{tag}"
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    return dst


def run_ablation():
    from agents import agent1_vad, agent2_cluster, agent3_quality, agent4_builder
    import evaluate as ev

    results = {}

    # ── A: Full pipeline ───────────────────────────────────────────────────────
    logger.info("=== A: Full pipeline ===")
    agent1_vad.run()
    agent2_cluster.run()
    agent3_quality.run()
    agent4_builder.run()
    results["A_full"] = ev.evaluate_d3("test")
    snapshot_dir(config.D3_DIR, "A")

    # ── B: Without Agent 1 ────────────────────────────────────────────────────
    logger.info("=== B: Without Agent 1 (no VAD) ===")
    # Use entire raw files as "chunks" – copy raw WAVs directly to D1
    shutil.rmtree(config.D1_DIR, ignore_errors=True)
    import os
    os.makedirs(config.D1_DIR, exist_ok=True)
    import utils
    raw_wavs = utils.find_audio_files(config.RAW_DIR)
    for i, p in enumerate(raw_wavs):
        dst = os.path.join(config.D1_DIR, f"raw_{i:04d}", f"chunk_0000_00.wav")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if p.endswith(".wav"):
            shutil.copy2(p, dst)
        else:
            utils.convert_to_wav(p, dst)

    shutil.rmtree(config.D2_DIR, ignore_errors=True)
    shutil.rmtree(config.D3_DIR, ignore_errors=True)
    agent2_cluster.run()
    agent3_quality.run()
    agent4_builder.run()
    results["B_no_vad"] = ev.evaluate_d3("test")
    snapshot_dir(config.D3_DIR, "B")

    # ── C: Without Agent 2 (random cluster assignment) ────────────────────────
    logger.info("=== C: Without Agent 2 (no clustering) ===")
    agent1_vad.run()   # re-segment properly

    # Random speaker assignment
    import random
    random.seed(0)
    import glob
    chunks = glob.glob(os.path.join(config.D1_DIR, "**", "*.wav"), recursive=True)
    shutil.rmtree(config.D2_DIR, ignore_errors=True)
    os.makedirs(config.D2_DIR, exist_ok=True)
    n_random_spk = 5
    for path in chunks:
        spk = f"speaker_{random.randint(0, n_random_spk-1):02d}"
        dst = os.path.join(config.D2_DIR, spk, os.path.basename(path))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(path, dst)

    shutil.rmtree(config.D3_DIR, ignore_errors=True)
    agent3_quality.run()
    agent4_builder.run()
    results["C_no_cluster"] = ev.evaluate_d3("test")
    snapshot_dir(config.D3_DIR, "C")

    # ── D: Without Agent 3 (no quality filter) ────────────────────────────────
    logger.info("=== D: Without Agent 3 (no quality filter) ===")
    agent1_vad.run()
    agent2_cluster.run()
    # SKIP agent3_quality
    shutil.rmtree(config.D3_DIR, ignore_errors=True)
    agent4_builder.run()
    results["D_no_filter"] = ev.evaluate_d3("test")
    snapshot_dir(config.D3_DIR, "D")

    # Save
    out = os.path.join(config.LOGS_DIR, "ablation_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Ablation results → {out}")

    print("\n=== Ablation Study Results ===")
    for condition, res in results.items():
        sdr = res.get("sdr_avg_mean", "–")
        print(f"  {condition:<25} SDR = {sdr} dB")

    return results


if __name__ == "__main__":
    run_ablation()
