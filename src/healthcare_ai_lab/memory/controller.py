"""Bounded, auditable memory decisions."""

from dataclasses import dataclass
import math

from healthcare_ai_lab.memory.policies import MemoryCandidate


@dataclass(frozen=True)
class MemoryDecision:
    timestamp: int
    policy: str
    budget: int
    selected_memory_ids: tuple[str, ...]
    relevance_scores: dict[str, float]
    probability_regime_1: float


class MemoryController:
    def __init__(self, policy, budget=1):
        if not isinstance(budget, int) or budget < 1:
            raise ValueError("Memory budget must be positive")

        self.policy = policy
        self.budget = budget

    def select(self, state, candidates):
        candidates = list(candidates)

        ids = [item.memory_id for item in candidates]

        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate memory identifiers")

        for item in candidates:
            if not isinstance(item, MemoryCandidate):
                raise TypeError("Invalid memory candidate")

            if item.lag < 1:
                raise ValueError("Memory lag must be positive")

            if item.observation_timestamp != state.timestamp - item.lag:
                raise ValueError("Inconsistent memory timestamp or lag")

        scores = self.policy.score(state, candidates)

        if set(scores) != set(ids):
            raise ValueError("Policy must score every candidate")

        if any(
            not math.isfinite(value) or value < 0
            for value in scores.values()
        ):
            raise ValueError("Invalid relevance scores")

        ordered = sorted(
            candidates,
            key=lambda item: (
                -scores[item.memory_id],
                item.memory_id
            ),
        )

        selected = tuple(
            item.memory_id
            for item in ordered[:self.budget]
        )

        return MemoryDecision(
            timestamp=state.timestamp,
            policy=self.policy.name,
            budget=self.budget,
            selected_memory_ids=selected,
            relevance_scores=scores,
            probability_regime_1=state.probability_regime_1,
        )
