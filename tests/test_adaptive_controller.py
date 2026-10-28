import pytest

from healthcare_ai_lab.controllers.adaptive import (
    ControllerConfig,
    adaptive_allocation,
    blend_probabilities,
)


def test_no_feedback_ignores_reactive_expert():
    config = ControllerConfig(fallback_weight=0.0, max_deviation=1.0, max_step=1.0)

    q = adaptive_allocation(
        q_anticipatory=0.8,
        q_reactive=0.1,
        reliability=1.0,
        anticipatory_share=0.2,
        feedback_available=False,
        config=config,
    )
    assert q == pytest.approx(0.8)


def test_unreliable_signal_uses_fallback():
    q = adaptive_allocation(
        q_anticipatory=0.9,
        q_reactive=0.9,
        reliability=0.0,
        anticipatory_share=0.5,
        feedback_available=True,
    )
    assert q == pytest.approx(0.0)


def test_weight_deviation_is_bounded():
    q = adaptive_allocation(
        q_anticipatory=1.0,
        q_reactive=1.0,
        reliability=1.0,
        anticipatory_share=0.5,
        feedback_available=True,
    )
    assert q == pytest.approx(0.25)


def test_probability_blending():
    p = blend_probabilities(0.1, 0.3, 0.25)
    assert p == pytest.approx(0.15)
