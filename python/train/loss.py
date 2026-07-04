from __future__ import annotations

import numpy as np


def cross_entropy_loss(logits: np.ndarray, labels: np.ndarray) -> float:
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    probs = np.exp(shifted)
    probs = probs / probs.sum(axis=-1, keepdims=True)
    safe_probs = np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1.0)
    return float(-np.log(safe_probs).mean())
