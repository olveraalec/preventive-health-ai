"""Allocation summaries are label-free and never join different patients' steps."""

import numpy as np
import pytest

from healthcare_ai_lab.evaluation.controller_diagnostics import controller_diagnostics


def test_analytic_dynamics_and_distinct_fallback_meanings():
    result = controller_diagnostics(
        [0, 0.25, 0.75, 0],
        emergency_fallback=[False, False, False, True],
        static_probability=[0.2] * 4,
        temporal_probability=[0.6, 0.6, 0.6, None],
    )
    assert result["q_mean"].value == 0.25
    assert result["q_variance"].value == pytest.approx(0.09375)
    assert result["mean_absolute_change"].value == 0.5
    assert result["mean_squared_change"].value == pytest.approx(0.875 / 3)
    assert result["expert_switching_frequency"].value == pytest.approx(2 / 3)
    assert result["allocation_change_frequency"].value == 1
    assert result["q_total_variation"].value == 1.5
    assert result["fallback_frequency"].value == 0.25
    assert result["static_mode_fraction"].value == 0.5
    assert result["expert_disagreement_mean_squared"].value == pytest.approx(0.16)
    assert result["expert_disagreement_mean_squared"].n_observations == 3
    assert result["available_expert_pair_fraction"].value == 0.75


def test_single_hour_has_no_transition_denominator():
    result = controller_diagnostics([0])
    assert result["q_variance"].value == 0
    assert result["q_total_variation"].value == 0
    assert result["allocation_step_count"].value == 0
    assert result["mean_absolute_change"].value is None
    assert result["mean_absolute_change"].reason == "no_within_patient_transitions"
    assert result["fallback_frequency"].value is None
    assert result["expert_disagreement_mean_squared"].value is None


@pytest.mark.parametrize("weights", [[], [np.nan], [np.inf], [-0.1], [1.1], [[0]]])
def test_invalid_weights(weights):
    with pytest.raises(ValueError):
        controller_diagnostics(weights)


def test_invalid_flags_and_experts():
    with pytest.raises(ValueError):
        controller_diagnostics([0.5], emergency_fallback=[True])
    with pytest.raises(ValueError):
        controller_diagnostics([0], emergency_fallback=[1])
    with pytest.raises(ValueError):
        controller_diagnostics(
            [0], static_probability=[0.1], temporal_probability=[np.inf]
        )
