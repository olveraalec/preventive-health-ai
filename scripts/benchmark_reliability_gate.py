"""Causal reliability gate under changing information quality.

Uses historical paired regret from a synthetic full-information oracle.
Only previous episodes update the gate before the next decision.
"""

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 12345
BETA = 0.10
TEMPERATURE = 0.10
THRESHOLD = 0.02
INITIAL_ADVANTAGE = -0.20
BLOCK_SIZE = 30

SOURCE = Path("results/information_alignment.csv")
OUTPUT = Path("results/reliability_gate.csv")

SCENARIOS = ["aligned", "misleading", "uninformative"]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def load_sequence():
    df = pd.read_csv(SOURCE)

    paired = df.pivot(
        index=["scenario", "recording_id"],
        columns="policy",
        values="regret",
    ).reset_index()

    required = ["static_equal", "continuous"]
    if paired[required].isna().any().any():
        raise ValueError("Missing paired policy outcomes.")

    # Reuse the SAME episode outcomes across policies.
    # Reorder scenario blocks to simulate distribution changes.
    grouped = {
        name: paired[paired["scenario"] == name]
        .sort_values("recording_id")
        .reset_index(drop=True)
        for name in SCENARIOS
    }

    lengths = {len(v) for v in grouped.values()}
    if len(lengths) != 1:
        raise ValueError("Scenario lengths must match.")

    total = len(grouped[SCENARIOS[0]])
    if total % BLOCK_SIZE:
        raise ValueError("Episode count must divide into blocks.")

    blocks = []

    for start in range(0, total, BLOCK_SIZE):
        for scenario in SCENARIOS:
            block = grouped[scenario].iloc[start : start + BLOCK_SIZE].copy()
            blocks.append(block)

    return pd.concat(blocks, ignore_index=True)


def main():
    rng = np.random.default_rng(SEED)
    sequence = load_sequence()

    estimate = INITIAL_ADVANTAGE
    previous_continuous_regret = []
    rows = []

    for t, row in enumerate(sequence.itertuples(index=False)):
        static_regret = float(row.static_equal)
        continuous_regret = float(row.continuous)

        # Decision computed BEFORE observing current outcome.
        estimate_before = estimate

        q = float(sigmoid((estimate_before - THRESHOLD) / TEMPERATURE))

        # Randomized policy choice with historical exposure q.
        use_continuous = bool(rng.random() < q)

        adaptive_regret = continuous_regret if use_continuous else static_regret

        rows.append(
            {
                "time": t,
                "scenario": row.scenario,
                "recording_id": row.recording_id,
                "static_regret": static_regret,
                "continuous_regret": continuous_regret,
                "adaptive_regret": adaptive_regret,
                "estimated_advantage_before": estimate_before,
                "covariance_exposure": q,
                "used_continuous": use_continuous,
            }
        )

        # Full-information feedback is available in simulation.
        # This update affects only FUTURE decisions.
        realized_advantage = static_regret - continuous_regret

        estimate = (1.0 - BETA) * estimate + BETA * realized_advantage

    results = pd.DataFrame(rows)

    print("\nPERFORMANCE BY SCENARIO")
    print(
        results.groupby("scenario")[
            [
                "static_regret",
                "continuous_regret",
                "adaptive_regret",
                "covariance_exposure",
            ]
        ]
        .mean()
        .round(4)
        .to_string()
    )

    print("\nOVERALL PERFORMANCE")
    print(
        results[
            [
                "static_regret",
                "continuous_regret",
                "adaptive_regret",
            ]
        ]
        .mean()
        .round(4)
        .to_string()
    )

    print("\nTRANSITION DIAGNOSTICS")

    transitions = results[results["scenario"] != results["scenario"].shift()]

    print(
        transitions[
            [
                "time",
                "scenario",
                "estimated_advantage_before",
                "covariance_exposure",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )

    print("\nADAPTIVE CONTROLLER")
    print(
        "Continuous selection frequency:",
        round(results.used_continuous.mean(), 4),
    )

    print(
        "Mean regret improvement vs static:",
        round(
            (results.static_regret - results.adaptive_regret).mean(),
            4,
        ),
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT, index=False)

    print("\nSaved:", OUTPUT)


if __name__ == "__main__":
    main()
