"""SDR and related evaluation metrics."""
import numpy as np


def sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    """
    Signal-to-Distortion Ratio (dB) between a single reference and estimate.
    Both arrays should be 1-D and the same length.
    """
    n = min(len(reference), len(estimate))
    ref, est = reference[:n], estimate[:n]

    # Scale estimate to minimise distortion (projection)
    if np.dot(ref, ref) < 1e-12:
        return -np.inf
    alpha = np.dot(ref, est) / np.dot(ref, ref)
    noise = est - alpha * ref
    num   = np.dot(ref, ref)
    den   = np.dot(noise, noise)
    if den < 1e-12:
        return np.inf
    return float(10 * np.log10(num / den))


def perm_sdr(refs: np.ndarray, ests: np.ndarray) -> tuple[float, float]:
    """
    Permutation-invariant SDR for 2-source separation.
    refs, ests: shape (2, T).
    Returns (sdr_s1, sdr_s2) under the best permutation.
    """
    s00 = sdr(refs[0], ests[0])
    s11 = sdr(refs[1], ests[1])
    s01 = sdr(refs[0], ests[1])
    s10 = sdr(refs[1], ests[0])

    if (s00 + s11) >= (s01 + s10):
        return s00, s11
    return s01, s10


def mean_sdr(refs: np.ndarray, ests: np.ndarray) -> float:
    """Average SDR across both sources (permutation-invariant)."""
    s1, s2 = perm_sdr(refs, ests)
    # Clamp only extreme outliers
    s1 = max(min(s1, 40.0), -40.0)
    s2 = max(min(s2, 40.0), -40.0)
    return float((s1 + s2) / 2.0)
