"""Reusable controller service.

Keeps controller mathematics independent of LangGraph.
"""

from healthcare_ai_lab.controller.contracts import (
    ControllerInput,
    ControllerOutput,
)
from healthcare_ai_lab.controller.optimizer import (
    ControllerConfig,
    optimize_weight,
)


class ControllerService:
    def __init__(self, config=None):
        self.config = config or ControllerConfig()

    def decide(self, request: ControllerInput):
        result = optimize_weight(
            moments=(
                request.variance_1,
                request.variance_2,
                request.covariance_12,
            ),
            previous_weight=request.previous_weight,
            config=self.config,
            memory_cost=request.resource_cost,
        )

        return ControllerOutput(
            episode_id=request.episode_id,
            step=request.step,
            weight_1=1.0 - result.weight,
            weight_2=result.weight,
            estimated_risk=result.estimated_risk,
            estimated_objective=result.estimated_objective,
            state_entropy=request.state_entropy,
            information_value=request.information_value,
            used_fallback=result.used_fallback,
            policy_name="regularized_psd",
        )
