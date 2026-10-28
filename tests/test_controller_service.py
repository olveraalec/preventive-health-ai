import pytest

from healthcare_ai_lab.controller.contracts import (
    ControllerInput,
)
from healthcare_ai_lab.controller.service import (
    ControllerService,
)


def test_controller_produces_valid_allocation():
    service = ControllerService()

    decision = service.decide(
        ControllerInput(
            episode_id="demo_001",
            step=0,
            variance_1=4.0,
            variance_2=1.0,
            covariance_12=0.2,
        )
    )

    assert decision.weight_1 >= 0
    assert decision.weight_2 >= 0
    assert decision.weight_1 + decision.weight_2 == pytest.approx(1)

    assert decision.estimated_risk >= 0
    assert decision.estimated_objective >= 0


def test_higher_quality_second_path_preferred():
    service = ControllerService()

    decision = service.decide(
        ControllerInput(
            episode_id="demo_002",
            step=0,
            variance_1=4.0,
            variance_2=1.0,
            covariance_12=0.0,
        )
    )

    assert decision.weight_2 > decision.weight_1


def test_invalid_variance_rejected():
    with pytest.raises(ValueError):
        ControllerInput(
            episode_id="demo_003",
            step=0,
            variance_1=-1,
            variance_2=1,
            covariance_12=0,
        )


def test_optional_information_preserved():
    service = ControllerService()

    decision = service.decide(
        ControllerInput(
            episode_id="demo_004",
            step=3,
            variance_1=2,
            variance_2=1,
            covariance_12=0.5,
            state_entropy=0.6,
            information_value=0.2,
        )
    )

    assert decision.state_entropy == pytest.approx(0.6)
    assert decision.information_value == pytest.approx(0.2)
