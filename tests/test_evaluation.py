import numpy as np
import pytest

from healthcare_ai_lab.evaluation.metrics import (
    rmse,
    gaussian_nll,
    gaussian_crps,
    interval_coverage,
    mean_decision_regret,
)


def test_perfect_prediction_has_zero_rmse():
    values = [1, 2, 3]
    assert rmse(values, values) == 0


def test_rmse_known_example():
    assert rmse([0, 0], [3, 4]) == pytest.approx(
        np.sqrt(12.5)
    )


def test_nll_finite():
    assert np.isfinite(
        gaussian_nll([0, 1], [0, 0], [1, 1])
    )


def test_crps_finite_and_nonnegative():
    score = gaussian_crps(
        [0, 1], [0, 0], [1, 1]
    )
    assert np.isfinite(score)
    assert score >= 0


def test_invalid_sigma_rejected():
    with pytest.raises(ValueError):
        gaussian_nll([0], [0], [0])


def test_interval_coverage():
    assert interval_coverage(
        [1, 5], [0, 0], [2, 4]
    ) == 0.5


def test_decision_regret():
    assert mean_decision_regret(
        [2, 4], [1, 3]
    ) == pytest.approx(1.0)
