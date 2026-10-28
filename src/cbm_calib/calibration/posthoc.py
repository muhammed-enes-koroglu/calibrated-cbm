"""Post-hoc calibration of concept logits.

Each 1-D calibrator maps concept logits to probabilities and is fitted on a held-out
calibration split. `GroupedCalibrator` applies one calibrator per group of concepts:
one group for all concepts ("global"), one per concept group ("group"), or one per
concept ("concept").
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.isotonic import IsotonicRegression


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _fit_affine(z, y, fit_bias: bool) -> tuple[float, float]:
    """Fit sigmoid(a * z + b) to labels y by minimising BCE with L-BFGS."""
    z = torch.as_tensor(z, dtype=torch.float64)
    y = torch.as_tensor(y, dtype=torch.float64)
    a = torch.ones((), dtype=torch.float64, requires_grad=True)
    b = torch.zeros((), dtype=torch.float64, requires_grad=fit_bias)
    params = [a, b] if fit_bias else [a]
    opt = torch.optim.LBFGS(params, lr=0.5, max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(a * z + b, y)
        loss.backward()
        return loss

    opt.step(closure)
    return a.item(), b.item()


class TemperatureScaling:
    """p = sigmoid(z / T)."""

    def fit(self, z, y):
        scale, _ = _fit_affine(z, y, fit_bias=False)
        self.temperature = 1.0 / scale
        return self

    def predict(self, z):
        return _sigmoid(np.asarray(z) / self.temperature)


class PlattScaling:
    """p = sigmoid(a * z + b)."""

    def fit(self, z, y):
        self.a, self.b = _fit_affine(z, y, fit_bias=True)
        return self

    def predict(self, z):
        return _sigmoid(self.a * np.asarray(z) + self.b)


class HistogramBinning:
    """Replace each probability by the observed frequency of its bin."""

    def __init__(self, n_bins: int = 15):
        self.n_bins = n_bins

    def fit(self, z, y):
        p = _sigmoid(np.asarray(z))
        y = np.asarray(y, dtype=float)
        self.edges = np.linspace(0.0, 1.0, self.n_bins + 1)
        idx = self._bin(p)
        counts = np.bincount(idx, minlength=self.n_bins)
        sums = np.bincount(idx, weights=y, minlength=self.n_bins)
        centres = (self.edges[:-1] + self.edges[1:]) / 2
        # Empty bins keep their centre, i.e. leave those predictions unchanged.
        self.values = np.where(counts > 0, sums / np.maximum(counts, 1), centres)
        return self

    def _bin(self, p):
        return np.clip(np.searchsorted(self.edges, p, side="right") - 1, 0, self.n_bins - 1)

    def predict(self, z):
        return self.values[self._bin(_sigmoid(np.asarray(z)))]


class IsotonicCalibration:
    """Monotone, piecewise-constant mapping from probability to frequency."""

    def fit(self, z, y):
        self.iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        self.iso.fit(_sigmoid(np.asarray(z)), np.asarray(y, dtype=float))
        return self

    def predict(self, z):
        return self.iso.predict(_sigmoid(np.asarray(z)))


METHODS = {
    "temperature": TemperatureScaling,
    "platt": PlattScaling,
    "histogram": HistogramBinning,
    "isotonic": IsotonicCalibration,
}
GRANULARITIES = ("global", "group", "concept")


class GroupedCalibrator:
    """Fit one calibrator per group of concepts on (n, k) logits and labels."""

    def __init__(self, method: str, granularity: str = "global", groups=None, **method_kwargs):
        if granularity not in GRANULARITIES:
            raise ValueError(f"granularity must be one of {GRANULARITIES}")
        if granularity == "group" and groups is None:
            raise ValueError("granularity='group' needs a concept-to-group array")
        self.method = method
        self.granularity = granularity
        self.groups = None if groups is None else np.asarray(groups)
        self.method_kwargs = method_kwargs

    def _group_ids(self, k: int) -> np.ndarray:
        if self.granularity == "global":
            return np.zeros(k, dtype=int)
        if self.granularity == "concept":
            return np.arange(k)
        return self.groups

    def fit(self, logits, labels):
        logits = np.asarray(logits, dtype=float)
        labels = np.asarray(labels, dtype=float)
        self.group_ids = self._group_ids(logits.shape[1])
        self.calibrators = {}
        for g in np.unique(self.group_ids):
            cols = self.group_ids == g
            cal = METHODS[self.method](**self.method_kwargs)
            self.calibrators[g] = cal.fit(logits[:, cols].ravel(), labels[:, cols].ravel())
        return self

    def predict(self, logits) -> np.ndarray:
        logits = np.asarray(logits, dtype=float)
        probs = np.empty_like(logits)
        for g, cal in self.calibrators.items():
            cols = self.group_ids == g
            probs[:, cols] = cal.predict(logits[:, cols].ravel()).reshape(len(logits), -1)
        return probs
