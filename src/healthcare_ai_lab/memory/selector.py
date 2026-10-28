"""Budgeted selection of scored memory candidates.

Two pathways receive continuous weights from the controller.
Candidates are ranked by weighted utility, then selected under
a discrete item-count budget.

Scores are assumed to be calibrated and comparable across
pathways. That assumption must be tested in later experiments.
"""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class MemoryCandidate:
    memory_id: str
    pathway: int
    information_score: float


@dataclass(frozen=True)
class SelectionResult:
    selected_ids: tuple[str, ...]
    budget: int
    used: int
    total_weighted_utility: float


def select_memories(
    candidates: list[MemoryCandidate],
    weight_1: float,
    weight_2: float,
    budget: int,
) -> SelectionResult:
    if budget < 0:
        raise ValueError("Budget cannot be negative")

    if not all(math.isfinite(w) and 0 <= w <= 1 for w in (weight_1, weight_2)):
        raise ValueError("Invalid pathway weights")

    if not math.isclose(
        weight_1 + weight_2,
        1.0,
        abs_tol=1e-8,
    ):
        raise ValueError("Pathway weights must sum to one")

    weights = {1: weight_1, 2: weight_2}
    scored = []
    seen = set()

    for item in candidates:
        if not item.memory_id or item.memory_id in seen:
            raise ValueError("Invalid or duplicate memory ID")

        seen.add(item.memory_id)

        if item.pathway not in weights:
            raise ValueError("Unknown pathway")

        if not math.isfinite(item.information_score) or item.information_score < 0:
            raise ValueError("Invalid information score")

        utility = weights[item.pathway] * item.information_score

        # Ignore items with no positive estimated utility.
        if utility > 0:
            scored.append((utility, item.memory_id))

    # Stable tie-breaking for reproducible experiments.
    scored.sort(key=lambda x: (-x[0], x[1]))
    selected = scored[:budget]

    return SelectionResult(
        selected_ids=tuple(memory_id for _, memory_id in selected),
        budget=budget,
        used=len(selected),
        total_weighted_utility=float(sum(utility for utility, _ in selected)),
    )
