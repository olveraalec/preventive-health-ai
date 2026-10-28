import numpy as np
import pandas as pd
import pytest
from healthcare_ai_lab.models.sepsis_inference import (
    predict_engineered_row,
    predict_engineered_trajectory,
)


class ProbModel:
    def __init__(self, p):
        self.p = p

    def predict_proba(self, x):
        return np.tile([1 - self.p, self.p], (len(x), 1))


class GainModel:
    def predict(self, x):
        return np.zeros(len(x))


@pytest.fixture
def bundle():
    return {
        "static": {"model": ProbModel(0.1), "features": ["HR_current"]},
        "temporal": {"model": ProbModel(0.3), "features": ["HR_current", "HR_last"]},
        "controller": GainModel(),
        "meta_features": ["ICULOS", "HR_observed", "entropy_static"],
        "fallback_q": 0.0,
        "temperature": 0.01,
    }


def test_bounded_and_no_feedback(bundle):
    output = predict_engineered_row(
        bundle,
        {
            "HR_current": 90.0,
            "HR_last": 88.0,
            "HR_observed": 1,
            "ICULOS": 4,
            "SepsisLabel": 1,
        },
    )
    assert output["controller"]["temporal_weight"] == pytest.approx(0.125)
    assert output["prediction"]["final_probability"] == pytest.approx(0.125)
    assert output["metadata"]["clinical_use"] is False


def test_missing_features(bundle):
    with pytest.raises(ValueError, match="Missing engineered"):
        predict_engineered_row(bundle, {"HR_current": 90, "ICULOS": 1})


def test_history_and_order(bundle):
    rows = pd.DataFrame(
        [
            {"ICULOS": 2, "HR_current": 92, "HR_last": 90, "HR_observed": 1},
            {"ICULOS": 1, "HR_current": 90, "HR_last": 90, "HR_observed": 1},
        ]
    )
    output = predict_engineered_trajectory(bundle, rows)
    assert [r["icu_hour"] for r in output] == [1, 2]
    assert all(0 <= r["controller"]["temporal_weight"] <= 0.25 for r in output)
