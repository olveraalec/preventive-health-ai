"""Interchangeable historical-memory selection policies."""

from dataclasses import dataclass


@dataclass(frozen=True)
class MemoryCandidate:
    memory_id: str
    observation_timestamp: int
    lag: int


class FixedLagPolicy:
    name = "fixed_lag"

    def __init__(self, preferred_lag=12):
        self.preferred_lag = preferred_lag

    def score(self, state, candidates):
        return {
            item.memory_id: float(
                item.lag == self.preferred_lag
            )
            for item in candidates
        }


class RecencyPolicy:
    name = "recency"

    def score(self, state, candidates):
        return {
            item.memory_id: 1.0 / (1.0 + item.lag)
            for item in candidates
        }


class AdaptiveRegimePolicy:
    name = "adaptive_regime"

    def score(self, state, candidates):
        p = state.probability_regime_1

        return {
            item.memory_id: (
                p if item.lag == 12
                else 1 - p if item.lag == 30
                else 0.0
            )
            for item in candidates
        }
