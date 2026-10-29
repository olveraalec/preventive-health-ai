"""Analytic regression guards for the unchanged continuous-response registry."""

import numpy as np
import pytest

from healthcare_ai_lab.evaluation.registry import METRICS, evaluate


def test_existing_gaussian_values_and_directions():
    scores = evaluate([0, 0], [0, 2], sigma=1)
    assert scores["rmse"] == pytest.approx(np.sqrt(2))
    assert scores["mae"] == 1
    assert scores["bias"] == -1  # original target-minus-prediction convention
    assert scores["p90_absolute_error"] == pytest.approx(1.8)
    assert scores["nll"] == pytest.approx(0.5 * np.log(2 * np.pi) + 1)
    assert scores["coverage_90"] == 0.5
    assert scores["width_90"] == pytest.approx(2 * 1.6448536269514722)
    assert evaluate([0], [0], sigma=1)["crps"] == pytest.approx(
        (np.sqrt(2) - 1) / np.sqrt(np.pi)
    )
    assert {name: spec.direction for name, spec in METRICS.items()} == {
        "rmse": "minimize",
        "mae": "minimize",
        "bias": "target",
        "p90_absolute_error": "minimize",
        "nll": "minimize",
        "crps": "minimize",
        "coverage_90": "target",
        "width_90": "minimize",
    }


def test_sigma_omission_broadcasting_and_validation_unchanged():
    assert set(evaluate([0, 1], [0, 1])) == {
        "rmse",
        "mae",
        "bias",
        "p90_absolute_error",
    }
    assert evaluate([0, 1], [0, 1], sigma=2) == evaluate([0, 1], [0, 1], sigma=[2, 2])
    for sigma in (0, -1, np.nan, np.inf, [1]):
        with pytest.raises(ValueError):
            evaluate([0, 1], [0, 1], sigma=sigma)
