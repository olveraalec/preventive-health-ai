import pytest

from healthcare_ai_lab.models.state_estimator import (
    BayesianStateEstimator,
    StateEstimate,
)
from healthcare_ai_lab.memory.policies import (
    MemoryCandidate,
    FixedLagPolicy,
    RecencyPolicy,
    AdaptiveRegimePolicy,
)
from healthcare_ai_lab.memory.controller import MemoryController


def make_state(probability=0.8):
    return StateEstimate(
        timestamp=100,
        probability_regime_1=probability,
        uncertainty_nats=0.5,
        feedback_used=False,
    )


def make_candidates():
    return [
        MemoryCandidate("lag12", 88, 12),
        MemoryCandidate("lag30", 70, 30),
    ]


def test_estimator_returns_valid_probability():
    estimator = BayesianStateEstimator()
    state = estimator.update(100, proxy=0.5)

    assert 0 < state.probability_regime_1 < 1
    assert 0 <= state.uncertainty_nats <= 0.694


def test_large_past_error_increases_high_variance_belief():
    plain = BayesianStateEstimator()
    feedback = BayesianStateEstimator()

    plain.update(100, 0.0)
    feedback.update(100, 0.0)

    without_error = plain.update(101, 0.0)
    with_error = feedback.update(
        101, 0.0, previous_residual=5.0
    )

    assert (
        with_error.probability_regime_1
        > without_error.probability_regime_1
    )


def test_estimator_rejects_repeated_timestamp():
    estimator = BayesianStateEstimator()
    estimator.update(100, 0.0)

    with pytest.raises(ValueError):
        estimator.update(100, 0.0)


def test_estimator_rejects_feedback_without_prior_state():
    estimator = BayesianStateEstimator()

    with pytest.raises(ValueError):
        estimator.update(100, 0.0, previous_residual=2.0)


def test_adaptive_policy_selects_lag12():
    controller = MemoryController(
        AdaptiveRegimePolicy(), budget=1
    )

    decision = controller.select(
        make_state(0.8), make_candidates()
    )

    assert decision.selected_memory_ids == ("lag12",)


def test_adaptive_policy_selects_lag30():
    controller = MemoryController(
        AdaptiveRegimePolicy(), budget=1
    )

    decision = controller.select(
        make_state(0.2), make_candidates()
    )

    assert decision.selected_memory_ids == ("lag30",)


def test_fixed_and_recency_baselines():
    state = make_state()
    candidates = make_candidates()

    fixed = MemoryController(
        FixedLagPolicy(30)
    ).select(state, candidates)

    recent = MemoryController(
        RecencyPolicy()
    ).select(state, candidates)

    assert fixed.selected_memory_ids == ("lag30",)
    assert recent.selected_memory_ids == ("lag12",)


def test_memory_budget_enforced():
    decision = MemoryController(
        AdaptiveRegimePolicy(),
        budget=1,
    ).select(make_state(), make_candidates())

    assert len(decision.selected_memory_ids) == 1


def test_future_memory_rejected():
    candidates = [
        MemoryCandidate("future", 105, 1)
    ]

    with pytest.raises(ValueError):
        MemoryController(
            AdaptiveRegimePolicy()
        ).select(make_state(), candidates)


def test_duplicate_memory_ids_rejected():
    candidates = [
        MemoryCandidate("same", 88, 12),
        MemoryCandidate("same", 70, 30),
    ]

    with pytest.raises(ValueError):
        MemoryController(
            AdaptiveRegimePolicy()
        ).select(make_state(), candidates)
