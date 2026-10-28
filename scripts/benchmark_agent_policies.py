"""Controlled benchmark of agent memory-selection policies.

This is a simulation with known realized utility, not evidence of
clinical benefit. All policies receive identical candidates and budgets.
"""

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from healthcare_ai_lab.agent.workflow import agent_graph
from healthcare_ai_lab.evaluation.suite import compare_paired
from healthcare_ai_lab.memory.selector import (
    MemoryCandidate,
    select_memories,
)

SEED = 12345
EPISODES = 60
BUDGET = 2
REDUNDANCY_PENALTY = 0.15

POLICIES = [
    "recent_only",
    "top_score",
    "static_equal",
    "continuous",
]


def realized_utility(selected, candidates, true_values):
    """Compute realized value minus within-pathway redundancy."""
    lookup = {c.memory_id: c for c in candidates}
    utility = sum(true_values[mid] for mid in selected)

    # Two selections from the same pathway can be redundant.
    for a, b in combinations(selected, 2):
        if lookup[a].pathway == lookup[b].pathway:
            utility -= REDUNDANCY_PENALTY

    return float(utility)


def generate_episode(rng, episode):
    """Generate one reproducible, deliberately imperfect value model."""
    regime = int(rng.integers(1, 3))

    candidates = []
    true_values = {}
    recency = {}

    for pathway in (1, 2):
        for position in range(3):
            memory_id = f"path{pathway}_memory{position}"
            score = float(rng.uniform(0.2, 1.0))

            candidates.append(
                MemoryCandidate(
                    memory_id=memory_id,
                    pathway=pathway,
                    information_score=score,
                )
            )

            # Scores are not the same as realized information value.
            alignment = 0.25 if pathway == regime else -0.15
            noise = float(rng.normal(0, 0.12))

            true_values[memory_id] = score * 0.7 + alignment + noise

            # Position 0 represents the newest memory.
            recency[memory_id] = position

    # Positive-semidefinite, varying two-pathway covariance.
    variances = rng.uniform(0.5, 4.0, size=2)
    correlation = float(rng.uniform(-0.3, 0.7))
    covariance = correlation * float(np.sqrt(np.prod(variances)))

    state = {
        "episode_id": f"simulation_{episode:03d}",
        "step": 0,
        "variance_1": float(variances[0]),
        "variance_2": float(variances[1]),
        "covariance_12": covariance,
        "previous_weight": 0.5,
        "memory_budget": BUDGET,
        "memory_candidates": [
            {
                "memory_id": c.memory_id,
                "pathway": c.pathway,
                "information_score": c.information_score,
            }
            for c in candidates
        ],
    }

    return candidates, true_values, recency, state


def choose_policies(candidates, recency, state):
    """Run existing implementations and simple baselines."""
    recent = sorted(
        candidates,
        key=lambda c: (
            recency[c.memory_id],
            c.pathway,
            c.memory_id,
        ),
    )[:BUDGET]

    top = sorted(
        candidates,
        key=lambda c: (-c.information_score, c.memory_id),
    )[:BUDGET]

    static = select_memories(
        candidates=candidates,
        weight_1=0.5,
        weight_2=0.5,
        budget=BUDGET,
    )

    result = agent_graph.invoke(state)

    if not result["verified"]:
        raise RuntimeError("Agent verification failed")

    continuous = result["execution"]["selected_memory_ids"]

    return {
        "recent_only": tuple(c.memory_id for c in recent),
        "top_score": tuple(c.memory_id for c in top),
        "static_equal": static.selected_ids,
        "continuous": tuple(continuous),
    }, result


def evaluate_episode(episode, candidates, true_values, choices, agent):
    """Evaluate each policy against the same feasible oracle."""
    ids = [c.memory_id for c in candidates]

    feasible = [subset for k in range(BUDGET + 1) for subset in combinations(ids, k)]

    optimal = max(realized_utility(s, candidates, true_values) for s in feasible)

    rows = []
    lookup = {c.memory_id: c for c in candidates}

    for policy, selected in choices.items():
        utility = realized_utility(selected, candidates, true_values)

        score_sum = sum(lookup[mid].information_score for mid in selected)

        same_path_pairs = sum(
            lookup[a].pathway == lookup[b].pathway for a, b in combinations(selected, 2)
        )

        rows.append(
            {
                "recording_id": f"simulation_{episode:03d}",
                "policy": policy,
                "realized_utility": utility,
                "regret": max(0.0, optimal - utility),
                "information_score_sum": score_sum,
                "budget_used": len(selected),
                "redundant_pairs": same_path_pairs,
                "selected_ids": "|".join(selected),
                "controller_weight_1": (
                    agent["decision"]["weight_1"] if policy == "continuous" else np.nan
                ),
                "controller_fallback": (
                    bool(agent["decision"]["used_fallback"])
                    if policy == "continuous"
                    else False
                ),
            }
        )

    return rows


def main():
    rng = np.random.default_rng(SEED)
    rows = []

    for episode in range(EPISODES):
        candidates, true_values, recency, state = generate_episode(rng, episode)

        choices, agent = choose_policies(candidates, recency, state)

        rows.extend(evaluate_episode(episode, candidates, true_values, choices, agent))

    results = pd.DataFrame(rows)

    output = Path("results/agent_policy_benchmark.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False)

    print("\nPOLICY PERFORMANCE")
    columns = [
        "realized_utility",
        "regret",
        "information_score_sum",
        "budget_used",
        "redundant_pairs",
    ]

    print(results.groupby("policy")[columns].agg(["mean", "std"]).round(4).to_string())

    print("\nPAIRED REGRET IMPROVEMENT VS STATIC EQUAL")
    for policy in POLICIES:
        if policy == "static_equal":
            continue

        comparison = compare_paired(
            results,
            reference="static_equal",
            candidate=policy,
            metric="regret",
            n_bootstrap=2000,
            seed=SEED,
        )
        print(policy, comparison)

    print("\nCONTINUOUS CONTROLLER DIAGNOSTICS")
    dynamic = results[results.policy == "continuous"]

    print("Mean weight 1:", dynamic.controller_weight_1.mean())
    print("Weight std:", dynamic.controller_weight_1.std())
    print("Fallback rate:", dynamic.controller_fallback.mean())

    print("\nSaved:", output)


if __name__ == "__main__":
    main()
