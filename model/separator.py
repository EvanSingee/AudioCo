"""
Conv-TasNet wrapper for speech separation inference (CPU only).
Uses a pretrained model from the Asteroid library.
"""
import logging
import numpy as np
import torch

import config

logger = logging.getLogger("Separator")

_model = None   # singleton


def _load_model():
    global _model
    if _model is not None:
        return _model

    logger.info(f"Loading Conv-TasNet: {config.MODEL_NAME}")
    from asteroid.models import ConvTasNet

    _model = ConvTasNet.from_pretrained(config.MODEL_NAME)
    _model.eval()
    # Force CPU
    _model = _model.cpu()
    logger.info("Model ready (CPU)")
    return _model


def separate(audio: np.ndarray, sr: int = 16_000) -> tuple[np.ndarray, np.ndarray]:
    """
    Run Conv-TasNet on a mixed 1-D float32 array.

    Parameters
    ----------
    audio : 1-D numpy array, float32, at *sr* Hz
    sr    : sample rate (must match model; default 16 kHz)

    Returns
    -------
    (s1, s2) – separated sources as 1-D float32 arrays
    """
    model = _load_model()

    # Model expects (batch, time)
    x = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)   # (1, T)

    with torch.no_grad():
        out = model(x)   # (1, n_src, T)

    s1 = out[0, 0].numpy().astype(np.float32)
    s2 = out[0, 1].numpy().astype(np.float32)
    return s1, s2
