"""Calibration metrics for binary concept predictions.

Every function takes predicted probabilities P(c = 1) and binary labels, either as
vectors of shape (n,) or as matrices of shape (n, k) for k concepts. "Pooled" metrics
treat every (sample, concept) pair as one prediction; "per-concept" metrics are
computed column by column.
"""

import numpy as np
from sklearn.metrics import roc_auc_score

EPS = 1e-7


def _bin_edges(probs: np.ndarray, n_bins: int, strategy: str) -> np.ndarray:
    if strategy == "uniform":
        return np.linspace(0.0, 1.0, n_bins + 1)
    if strategy == "quantile":
        # Equal-mass bins: each bin holds roughly the same number of predictions.
        edges = np.quantile(probs, np.linspace(0.0, 1.0, n_bins + 1))
        edges[0], edges[-1] = 0.0, 1.0
        return np.unique(edges)
    raise ValueError(f"unknown binning strategy: {strategy}")


def reliability_curve(probs, labels, n_bins: int = 15, strategy: str = "uniform") -> dict:
    """Per-bin mean confidence, observed frequency of c = 1, and counts."""
    probs = np.asarray(probs, dtype=float).ravel()
    labels = np.asarray(labels, dtype=float).ravel()
    edges = _bin_edges(probs, n_bins, strategy)
    n = len(edges) - 1
    idx = np.clip(np.searchsorted(edges, probs, side="right") - 1, 0, n - 1)
    counts = np.bincount(idx, minlength=n)
    conf_sum = np.bincount(idx, weights=probs, minlength=n)
    freq_sum = np.bincount(idx, weights=labels, minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        confidence = conf_sum / counts
        frequency = freq_sum / counts
    return {"edges": edges, "counts": counts, "confidence": confidence, "frequency": frequency}


def ece(probs, labels, n_bins: int = 15, strategy: str = "uniform") -> float:
    """Expected calibration error over all predictions (pooled)."""
    curve = reliability_curve(probs, labels, n_bins, strategy)
    counts = curve["counts"]
    mask = counts > 0
    gaps = np.abs(curve["frequency"][mask] - curve["confidence"][mask])
    return float(np.sum(counts[mask] * gaps) / counts.sum())


def per_concept_ece(probs, labels, n_bins: int = 15, strategy: str = "uniform") -> np.ndarray:
    """ECE of each concept (column) separately, shape (k,)."""
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    return np.array([ece(probs[:, j], labels[:, j], n_bins, strategy) for j in range(probs.shape[1])])


def brier(probs, labels) -> float:
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    return float(np.mean((probs - labels) ** 2))


def nll(probs, labels) -> float:
    probs = np.clip(np.asarray(probs, dtype=float), EPS, 1 - EPS)
    labels = np.asarray(labels, dtype=float)
    return float(-np.mean(labels * np.log(probs) + (1 - labels) * np.log(1 - probs)))


def macro_auroc(probs, labels) -> float:
    """Mean AUROC over concepts that have both classes present."""
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels)
    scores = [
        roc_auc_score(labels[:, j], probs[:, j])
        for j in range(probs.shape[1])
        if 0 < labels[:, j].sum() < len(labels)
    ]
    return float(np.mean(scores)) if scores else float("nan")


def concept_summary(probs, labels, n_bins: int = 15) -> dict:
    """All Step 1 metrics for a (n, k) matrix of concept probabilities."""
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    return {
        "ece_pooled": ece(probs, labels, n_bins, "uniform"),
        "ece_pooled_quantile": ece(probs, labels, n_bins, "quantile"),
        "ece_per_concept_mean": float(per_concept_ece(probs, labels, n_bins).mean()),
        "brier": brier(probs, labels),
        "nll": nll(probs, labels),
        "accuracy": float(np.mean((probs >= 0.5) == labels)),
        "auroc": macro_auroc(probs, labels),
    }
