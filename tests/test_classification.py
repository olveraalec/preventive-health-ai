"""Analytic binary fixtures; no patient records, model fitting, or inference."""

import json

import numpy as np
import pytest
from pydantic import ValidationError

from healthcare_ai_lab.evaluation.classification import (
    CLASSIFICATION_METRICS,
    binary_losses,
    calibration_bins,
    evaluate_classification,
)
from healthcare_ai_lab.evaluation.contracts import EvaluationConfig


def test_analytic_log_loss_brier_and_directions():
    p = np.array([0.25, 0.75])
    original = p.copy()
    report = evaluate_classification([0, 1], p)
    assert report.metrics["bernoulli_log_loss"].value == pytest.approx(-np.log(0.75))
    assert report.metrics["brier_score"].value == 0.0625
    assert report.metrics["bernoulli_log_loss"].metric.probability_clip == 1e-10
    assert report.metrics["bernoulli_log_loss"].metric.role == "objective"
    assert report.metrics["auprc"].metric.direction == "maximize"
    assert report.metrics["auprc"].metric.role == "monitoring"
    assert report.metrics["mean_predicted_probability"].metric.direction == "none"
    np.testing.assert_array_equal(p, original)
    assert all(
        spec.definition.metric_id == name
        for name, spec in CLASSIFICATION_METRICS.items()
    )


def test_score_only_clipping_at_endpoints():
    p = np.array([0.0, 1.0, 1.0, 0.0])
    result = evaluate_classification([0, 1, 0, 1], p)
    assert np.isfinite(result.metrics["bernoulli_log_loss"].value)
    assert result.metrics["brier_score"].value == 0.5
    np.testing.assert_array_equal(p, [0, 1, 1, 0])
    assert (
        evaluate_classification([0, 1], [0, 1]).metrics["predictive_entropy_mean"].value
        == 0
    )
    other = evaluate_classification(
        [0, 1], [1, 0], config=EvaluationConfig(epsilon=1e-5)
    )
    assert other.metrics["bernoulli_log_loss"].metric.probability_clip == 1e-5


@pytest.mark.parametrize(
    "y,p",
    [
        ([], []),
        ([0], [0, 1]),
        ([[0]], [[0.1]]),
        ([np.nan], [0.1]),
        ([2], [0.1]),
        ([-1], [0.1]),
        ([0], [np.nan]),
        ([0], [np.inf]),
        ([0], [-0.1]),
        ([1], [1.1]),
    ],
)
def test_invalid_arrays_fail(y, p):
    with pytest.raises(ValueError):
        evaluate_classification(y, p)


@pytest.mark.parametrize("label", [0, 1])
def test_single_class_discrimination_explicitly_undefined(label):
    result = evaluate_classification([label, label], [0.1, 0.9])
    for name in ("auprc", "auroc"):
        assert result.metrics[name].value is None
        assert result.metrics[name].reason == "single_class_cohort"
        assert result.metrics[name].status == "undefined"
    assert result.metrics["observed_prevalence"].value == label
    json.dumps(result.model_dump(), allow_nan=False)


def test_average_precision_not_trapezoidal_and_constant_predictions():
    metrics = evaluate_classification([0, 1, 0, 1], [0.1, 0.4, 0.35, 0.8]).metrics
    assert metrics["auprc"].value == 1
    assert metrics["auroc"].value == 1
    tied = evaluate_classification([0, 1, 0, 1], [0.5] * 4).metrics
    assert tied["auprc"].value == 0.5
    assert tied["auroc"].value == 0.5
    other = evaluate_classification([1, 0, 1], [0.9, 0.8, 0.7]).metrics
    assert other["auprc"].value == pytest.approx(5 / 6)
    assert other["auroc"].value == 0.5


def test_fixed_width_calibration_edges_empty_bins_and_ece():
    config = EvaluationConfig(calibration_bins=2)
    report = evaluate_classification([0, 1], [0.25, 0.75], config=config)
    assert report.metrics["ece_binned"].value == 0.25
    bins = calibration_bins([0, 1, 1], [0, 0.5, 1], 2)
    assert [b.count for b in bins] == [1, 2]
    assert bins[-1].upper_inclusive
    empty = calibration_bins([0], [0], 2)[1]
    assert (
        empty.count == 0
        and empty.mean_probability is None
        and empty.observed_fraction is None
    )
    with pytest.raises(ValueError):
        calibration_bins([0], [0], 0)
    with pytest.raises(ValueError):
        binary_losses([0], [0.5], epsilon=0)
    with pytest.raises(ValidationError):
        EvaluationConfig(calibration_bins=True)
    edges = [index / 10 for index in range(11)]
    assert [bin.count for bin in calibration_bins([0] * 11, edges, 10)] == [1] * 9 + [2]


def test_brier_gain_identity():
    rng = np.random.default_rng(12345)
    y, p0, delta = (
        rng.integers(0, 2, 50),
        rng.uniform(0.25, 0.75, 50),
        rng.uniform(-0.2, 0.2, 50),
    )
    for a in (0, 0.25, 1):
        observed = np.mean((y - p0) ** 2 - (y - (p0 + a * delta)) ** 2)
        expected = 2 * a * np.mean((y - p0) * delta) - a**2 * np.mean(delta**2)
        assert observed == pytest.approx(expected, abs=1e-15)


def test_unrepresentable_score_clip_is_rejected():
    with pytest.raises(ValidationError, match="representable"):
        EvaluationConfig(epsilon=1e-300)
    with pytest.raises(ValueError, match="clipping"):
        binary_losses([0], [1], epsilon=1e-300)
