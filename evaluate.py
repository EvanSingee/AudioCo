"""
evaluate.py – Conv-TasNet Evaluation across dataset versions.

Usage:
    python evaluate.py --dataset D3   # evaluates on data/D3_structured/test/
    python evaluate.py --all          # evaluates D0 → D3 sequentially
    python evaluate.py --ablation     # runs ablation study

The evaluation runs a pretrained Conv-TasNet (Asteroid) on mixture files
and computes SDR, SI-SDR, and optionally STOI.

Results are saved to logs/eval_results.json
"""

import os
import json
import argparse
import logging
import warnings
import numpy as np
import torch
from tqdm import tqdm

import config
import utils

warnings.filterwarnings("ignore")
logger = logging.getLogger("Evaluate")


# ─── Model loading ────────────────────────────────────────────────────────────

_model_cache = {}


def load_model():
    """Load the pretrained Conv-TasNet from Asteroid / HuggingFace."""
    if "model" in _model_cache:
        return _model_cache["model"]

    try:
        from asteroid.models import ConvTasNet
        model = ConvTasNet.from_pretrained(config.CONV_TASNET_MODEL)
        model.eval()
        _model_cache["model"] = model
        logger.info(f"Loaded Conv-TasNet: {config.CONV_TASNET_MODEL}")
        return model
    except Exception as e:
        logger.error(f"Could not load Conv-TasNet: {e}")
        raise


# ─── Metric helpers ───────────────────────────────────────────────────────────

def compute_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Signal-to-Distortion Ratio (mir_eval)."""
    try:
        from mir_eval.separation import bss_eval_sources
        sdr, _, _, _ = bss_eval_sources(
            reference[np.newaxis, :],
            estimate[np.newaxis, :],
        )
        return float(sdr[0])
    except Exception:
        return float("nan")


def compute_si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Scale-Invariant SDR."""
    eps = 1e-8
    ref = reference - reference.mean()
    est = estimate  - estimate.mean()
    alpha = (est @ ref) / (ref @ ref + eps)
    proj = alpha * ref
    noise = est - proj
    si_sdr = 10 * np.log10((proj @ proj + eps) / (noise @ noise + eps))
    return float(si_sdr)


def compute_stoi(reference: np.ndarray, estimate: np.ndarray, sr: int) -> float:
    """Short-Time Objective Intelligibility."""
    try:
        from pystoi import stoi
        return float(stoi(reference, estimate, sr, extended=False))
    except Exception:
        return float("nan")


# ─── Inference + metrics for one sample ──────────────────────────────────────

@torch.no_grad()
def evaluate_sample(
    model,
    mix_path: str,
    s1_path: str = None,
    s2_path: str = None,
    sr: int = config.SAMPLE_RATE,
) -> dict:
    """
    Run model on one mix file.  If reference sources are provided,
    compute full metrics.  Returns a dict of metric values.
    """
    mix, _ = utils.load_wav(mix_path)
    mix_t = torch.tensor(mix).unsqueeze(0).unsqueeze(0)  # (1, 1, T)

    # Forward pass
    estimates = model(mix_t)  # (1, n_src, T)
    est_np = estimates.squeeze(0).numpy()  # (n_src, T)

    result = {"mix_path": mix_path}

    if s1_path and s2_path:
        ref1, _ = utils.load_wav(s1_path)
        ref2, _ = utils.load_wav(s2_path)
        n = min(len(ref1), len(ref2), est_np.shape[1])
        ref1, ref2 = ref1[:n], ref2[:n]
        e1, e2 = est_np[0, :n], est_np[1, :n]

        # Permutation-invariant evaluation (pick best assignment)
        sdr_11 = compute_sdr(ref1, e1)
        sdr_22 = compute_sdr(ref2, e2)
        sdr_12 = compute_sdr(ref1, e2)
        sdr_21 = compute_sdr(ref2, e1)

        if (sdr_11 + sdr_22) >= (sdr_12 + sdr_21):
            best1, best2 = e1, e2
        else:
            best1, best2 = e2, e1

        result.update({
            "sdr_s1":    compute_sdr(ref1, best1),
            "sdr_s2":    compute_sdr(ref2, best2),
            "sdr_avg":   (compute_sdr(ref1, best1) + compute_sdr(ref2, best2)) / 2,
            "si_sdr_s1": compute_si_sdr(ref1, best1),
            "si_sdr_s2": compute_si_sdr(ref2, best2),
            "si_sdr_avg":(compute_si_sdr(ref1, best1) + compute_si_sdr(ref2, best2)) / 2,
            "stoi_s1":   compute_stoi(ref1, best1, sr),
            "stoi_s2":   compute_stoi(ref2, best2, sr),
        })

    return result


# ─── Dataset version evaluators ──────────────────────────────────────────────

def evaluate_d3(split: str = "test") -> dict:
    """Evaluate on structured D3 dataset."""
    model = load_model()
    mix_dir = os.path.join(config.D3_DIR, split, "mix")
    s1_dir  = os.path.join(config.D3_DIR, split, "s1")
    s2_dir  = os.path.join(config.D3_DIR, split, "s2")

    mix_files = sorted(f for f in os.listdir(mix_dir) if f.endswith(".wav"))
    results = []
    for fname in tqdm(mix_files, desc=f"Eval D3/{split}"):
        r = evaluate_sample(
            model,
            os.path.join(mix_dir, fname),
            os.path.join(s1_dir, fname),
            os.path.join(s2_dir, fname),
        )
        results.append(r)
    return _summarise(results, label=f"D3/{split}")


def evaluate_raw(n_samples: int = 20) -> dict:
    """
    Blind evaluation on raw D0 audio – no reference, report only qualitative
    stats (RMS, SNR of model output vs input).
    """
    model = load_model()
    raw_files = utils.find_audio_files(config.RAW_DIR)[:n_samples]
    results = []
    for f in tqdm(raw_files, desc="Eval D0 (blind)"):
        try:
            # Convert if needed
            if not f.endswith(".wav"):
                tmp = f.replace(config.RAW_DIR, os.path.join(config.LOGS_DIR, "_tmp")) + ".wav"
                utils.convert_to_wav(f, tmp)
                f = tmp
            r = evaluate_sample(model, f)
            results.append(r)
        except Exception as e:
            logger.warning(f"Skipping {f}: {e}")
    return _summarise(results, label="D0/blind")


def _summarise(results: list, label: str) -> dict:
    keys = ["sdr_avg", "si_sdr_avg", "stoi_s1"]
    summary = {"label": label, "n_samples": len(results)}
    for k in keys:
        vals = [r[k] for r in results if k in r and not np.isnan(r[k])]
        if vals:
            summary[k + "_mean"] = round(float(np.mean(vals)), 3)
            summary[k + "_std"]  = round(float(np.std(vals)), 3)
    return summary


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Conv-TasNet Evaluation")
    parser.add_argument("--dataset", choices=["D0", "D3"], default="D3")
    parser.add_argument("--split",   default="test")
    parser.add_argument("--all",     action="store_true", help="Evaluate all dataset versions")
    parser.add_argument("--ablation",action="store_true", help="Run ablation study")
    args = parser.parse_args()

    all_results = []

    if args.all or args.ablation:
        logger.info("Running full comparative evaluation D0 → D3...")
        all_results.append(evaluate_raw())
        all_results.append(evaluate_d3("test"))
    else:
        if args.dataset == "D0":
            all_results.append(evaluate_raw())
        else:
            all_results.append(evaluate_d3(args.split))

    # Persist results
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    out_path = os.path.join(config.LOGS_DIR, "eval_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n=== Evaluation Results ===")
    for r in all_results:
        print(json.dumps(r, indent=2))
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    main()
