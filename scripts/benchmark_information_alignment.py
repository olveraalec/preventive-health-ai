"""Test how covariance-information alignment affects agent utility.

Uses the existing LangGraph, memory selector and paired evaluation.
All outcomes are simulated. The controller never sees true utility.
"""

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark_agent_policies import (
    BUDGET,
    choose_policies,
    realized_utility,
)
from healthcare_ai_lab.memory.selector import MemoryCandidate
from healthcare_ai_lab.evaluation.suite import compare_paired

SEED = 12345
EPISODES = 180

SCENARIOS = (
    "aligned",
    "uninformative",
    "misleading",
)


def make_episode(rng, episode, scenario, latent):
    """Create a paired experiment with shared candidate outcomes."""
    useful_pathway = latent["useful_pathway"]
    candidates = latent["candidates"]
    true_values = latent["true_values"]
    recency = latent["recency"]

    # The covariance signal changes; true utility does not.
    if scenario == "aligned":
        low_risk_pathway = useful_pathway
    elif scenario == "misleading":
        low_risk_pathway = 3 - useful_pathway
    else:
        low_risk_pathway = None

    variances = [2.0, 2.0]

    if low_risk_pathway is not None:
        variances[low_risk_pathway - 1] = 0.6
        variances[2 - low_risk_pathway] = 3.0

    state = {
        "episode_id": f"{scenario}_{episode}",
        "step": 0,
        "variance_1": variances[0],
        "variance_2": variances[1],
        "covariance_12": 0.2,
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


def generate_latent_episode(rng):
    """Create the same hidden values for all scenarios."""
    useful_pathway = int(rng.integers(1, 3))
    candidates = []
    true_values = {}
    recency = {}

    for pathway in (1, 2):
        for position in range(3):
            mid = f"path{pathway}_memory{position}"
            score = float(rng.uniform(0.2, 1.0))

            candidates.append(
                MemoryCandidate(
                    memory_id=mid,
                    pathway=pathway,
                    information_score=score,
                )
            )

            pathway_effect = 0.35 if pathway == useful_pathway else -0.10

            true_values[mid] = 0.7 * score + pathway_effect + float(rng.normal(0, 0.12))

            recency[mid] = position

    return {
        "useful_pathway": useful_pathway,
        "candidates": candidates,
        "true_values": true_values,
        "recency": recency,
    }


def main():
    rng = np.random.default_rng(SEED)
    rows = []

    for episode in range(EPISODES):
        # Common random numbers make scenario comparisons paired.
        latent = generate_latent_episode(rng)

        for scenario in SCENARIOS:
            candidates, truth, recency, state = make_episode(
                rng, episode, scenario, latent
            )

            choices, agent = choose_policies(candidates, recency, state)

            ids = [c.memory_id for c in candidates]

            feasible = (
                subset
                for size in range(BUDGET + 1)
                for subset in combinations(ids, size)
            )

            optimal = max(realized_utility(s, candidates, truth) for s in feasible)

            for policy, selected in choices.items():
                utility = realized_utility(selected, candidates, truth)

                rows.append(
                    {
                        "scenario": scenario,
                        "recording_id": f"episode_{episode:03d}",
                        "policy": policy,
                        "realized_utility": utility,
                        "regret": max(0.0, optimal - utility),
                        "budget_used": len(selected),
                        "redundant_pairs": sum(
                            a.startswith("path1") == b.startswith("path1")
                            for a, b in combinations(selected, 2)
                        ),
                        "weight_1": (
                            agent["decision"]["weight_1"]
                            if policy == "continuous"
                            else np.nan
                        ),
                    }
                )

    results = pd.DataFrame(rows)

    output = Path("results/information_alignment.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False)

    columns = [
        "realized_utility",
        "regret",
        "redundant_pairs",
    ]

    print("\nSCENARIO PERFORMANCE")
    print(
        results.groupby(["scenario", "policy"])[columns]
        .agg(["mean", "std"])
        .round(4)
        .to_string()
    )

    print("\nPAIRED REGRET COMPARISONS")

    for scenario in SCENARIOS:
        part = results[results.scenario == scenario]

        comparison = compare_paired(
            part,
            reference="static_equal",
            candidate="continuous",
            metric="regret",
            n_bootstrap=5000,
            seed=SEED,
        )

        print(f"\n{scenario.upper()}")
        for key, value in comparison.items():
            print(f"{key}: {value}")

    print("\nSaved:", output)


if __name__ == "__main__":
    main()
