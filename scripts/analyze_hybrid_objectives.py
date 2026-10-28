"""Phase A.12 — Hybrid multi-objective diagnostic.

Evaluates existing controller results across prediction quality,
probabilistic quality, downside risk, and weight stability.

Exploratory evaluation only: no selected test-set winner.
"""

from pathlib import Path

import numpy as np
import pandas as pd

SOURCE = Path("results/bidmc_hybrid_controller.csv")
OUTPUT = Path("results/bidmc_hybrid_objectives.csv")

df = pd.read_csv(SOURCE)

LOWER_BETTER = [
    "rmse",
    "mae",
    "nll",
    "crps",
    "p90_absolute_error",
    "weight_variation",
    "mean_deviation_from_equal",
]

# Coverage is a constraint, not simply a loss to minimize.
COVERAGE_TARGET = 0.90

# Predeclared exploratory objective weights.
OBJECTIVES = {
    "prediction": {
        "rmse": 0.60,
        "mae": 0.40,
    },
    "probabilistic": {
        "nll": 0.50,
        "crps": 0.50,
    },
    "risk_aware": {
        "rmse": 0.40,
        "p90_absolute_error": 0.40,
        "nll": 0.20,
    },
    "balanced": {
        "rmse": 0.25,
        "nll": 0.20,
        "crps": 0.20,
        "p90_absolute_error": 0.20,
        "weight_variation": 0.15,
    },
}


def normalize_relative(data, metrics):
    """Scale metrics using fixed equal-blend references.

    Descriptive normalization, not a deployable fitted scaler.
    """
    out = data.copy()
    reference = out[out.policy == "equal_blend"]

    for metric in metrics:
        scale = max(
            float(reference[metric].mean()),
            1e-6,
        )
        out[f"norm_{metric}"] = out[metric] / scale

    return out


def pareto_flags(frame, metrics):
    """Mark nondominated policies under lower-is-better metrics."""
    values = frame[metrics].to_numpy(dtype=float)
    result = np.ones(len(frame), dtype=bool)

    for i in range(len(frame)):
        dominated = np.all(values <= values[i], axis=1) & np.any(
            values < values[i], axis=1
        )
        result[i] = not dominated.any()

    return result


def main():
    required = set(
        LOWER_BETTER
        + [
            "policy",
            "run",
            "recording_id",
            "coverage_90",
            "width_90",
        ]
    )

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    data = normalize_relative(df, LOWER_BETTER)

    # Evaluate the predeclared objectives.
    for objective, weights in OBJECTIVES.items():
        data[objective] = sum(
            weight * data[f"norm_{metric}"] for metric, weight in weights.items()
        )

    # Coverage deviation is reported independently.
    data["coverage_gap"] = (data["coverage_90"] - COVERAGE_TARGET).abs()

    metrics = list(OBJECTIVES) + [
        "rmse",
        "mae",
        "nll",
        "crps",
        "p90_absolute_error",
        "coverage_gap",
        "width_90",
        "weight_variation",
    ]

    summary = data.groupby("policy")[metrics].mean()

    print("\nMULTI-OBJECTIVE COMPARISON")
    print(summary.round(5).to_string())

    print("\nPARETO ANALYSIS")
    pareto_metrics = [
        "rmse",
        "nll",
        "crps",
        "p90_absolute_error",
        "weight_variation",
        "coverage_gap",
    ]

    pareto = summary.copy()
    pareto["pareto_efficient"] = pareto_flags(pareto, pareto_metrics)

    print(pareto[pareto_metrics + ["pareto_efficient"]].round(5).to_string())

    print("\nPAIRED OBJECTIVE GAINS VS EQUAL BLEND")

    for objective in OBJECTIVES:
        paired = data.pivot(
            index=["run", "recording_id"],
            columns="policy",
            values=objective,
        )

        reference = paired["equal_blend"]

        print(f"\n{objective.upper()}")

        for policy in paired.columns:
            if policy == "equal_blend":
                continue

            gains = reference - paired[policy]

            print(
                f"{policy}: "
                f"mean={gains.mean():.5f}, "
                f"median={gains.median():.5f}, "
                f"win={((gains > 0).mean()):.3f}, "
                f"p10={gains.quantile(0.10):.5f}"
            )

    # Patient-level comparisons: repeated runs involving
    # the same recording are not independent observations.
    patient_scores = (
        data.groupby(["recording_id", "policy"])[list(OBJECTIVES)].mean().reset_index()
    )

    print("\nPATIENT-LEVEL BALANCED OBJECTIVE")
    print(
        patient_scores.pivot(
            index="recording_id",
            columns="policy",
            values="balanced",
        )
        .round(4)
        .to_string()
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUTPUT, index=False)

    print("\nSaved:", OUTPUT)


if __name__ == "__main__":
    main()
