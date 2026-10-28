import numpy as np
import pytest

from cbm_calib.calibration import GRANULARITIES, METHODS, GroupedCalibrator
from cbm_calib.metrics import brier, concept_summary, ece, nll, per_concept_ece


def calibrated_sample(n=50_000, k=4, seed=0):
    """Labels drawn from the predicted probabilities, so predictions are calibrated."""
    rng = np.random.default_rng(seed)
    logits = rng.normal(0, 2, size=(n, k))
    probs = 1 / (1 + np.exp(-logits))
    labels = (rng.random((n, k)) < probs).astype(float)
    return logits, probs, labels


def test_ece_near_zero_when_calibrated():
    _, probs, labels = calibrated_sample()
    assert ece(probs, labels) < 0.01
    assert ece(probs, labels, strategy="quantile") < 0.01
    assert per_concept_ece(probs, labels).max() < 0.02


def test_ece_detects_overconfidence():
    logits, _, labels = calibrated_sample()
    overconfident = 1 / (1 + np.exp(-3 * logits))
    assert ece(overconfident, labels) > 0.05


def test_ece_known_value():
    # One bin at 0.8 confidence with 50% positives -> ECE = 0.3.
    probs = np.full(10, 0.8)
    labels = np.array([1, 0] * 5)
    assert ece(probs, labels) == pytest.approx(0.3)


def test_scoring_rules():
    probs = np.array([[1.0, 0.0], [0.5, 0.5]])
    labels = np.array([[1, 0], [1, 0]])
    assert brier(probs, labels) == pytest.approx(0.125)
    assert nll(probs, labels) == pytest.approx(np.log(2) / 2, rel=1e-4)
    summary = concept_summary(probs, labels)
    assert set(summary) >= {"ece_pooled", "ece_per_concept_mean", "brier", "nll", "accuracy", "auroc"}


def test_temperature_recovers_scale():
    logits, _, labels = calibrated_sample()
    cal = GroupedCalibrator("temperature", "global").fit(3 * logits, labels)
    assert cal.calibrators[0].temperature == pytest.approx(3.0, rel=0.05)


@pytest.mark.parametrize("method", list(METHODS))
@pytest.mark.parametrize("granularity", GRANULARITIES)
def test_every_calibrator_fixes_overconfidence(method, granularity):
    logits, _, labels = calibrated_sample()
    fit_half, test_half = slice(0, 25_000), slice(25_000, None)
    groups = np.array([0, 0, 1, 1])
    cal = GroupedCalibrator(method, granularity, groups).fit(3 * logits[fit_half], labels[fit_half])
    probs = cal.predict(3 * logits[test_half])
    assert probs.shape == logits[test_half].shape
    assert ((probs >= 0) & (probs <= 1)).all()
    before = ece(1 / (1 + np.exp(-3 * logits[test_half])), labels[test_half])
    assert ece(probs, labels[test_half]) < before / 3


def test_per_concept_beats_global_when_concepts_differ():
    logits, _, labels = calibrated_sample()
    scaled = logits * np.array([0.5, 1.0, 2.0, 4.0])  # each concept miscalibrated differently
    glob = GroupedCalibrator("temperature", "global").fit(scaled, labels)
    per = GroupedCalibrator("temperature", "concept").fit(scaled, labels)
    assert ece(per.predict(scaled), labels) < ece(glob.predict(scaled), labels)
