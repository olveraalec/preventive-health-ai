"""Phase A.13 — Corrected exploratory hybrid objective.

All metric scales and objective weights are predeclared.
Reports per-patient results and paired downside risk.
Does not perform test-set model selection.
"""

from pathlib import Path
import numpy as np
import pandas as pd

SOURCE = Path("results/bidmc_hybrid_controller.csv")
OUTPUT = Path("results/bidmc_hybrid_corrected.csv")

# Fixed reference scales; never divide by a zero baseline.
SCALES = {
    "rmse": 1.2,
    "nll": 1.7,
    "crps": 0.7,
    "p90_absolute_error": 1.8,
    "weight_variation": 0.02,
}

WEIGHTS = {
    "rmse": 0.25,
    "nll": 0.20,
    "crps": 0.20,
    "p90_absolute_error": 0.20,
    "weight_variation": 0.15,
}

COVERAGE_TARGET = 0.90
COVERAGE_TOLERANCE = 0.03


def main():
    df = pd.read_csv(SOURCE)

    required = set(SCALES) | {
        "policy",
        "run",
        "recording_id",
        "coverage_90",
        "width_90",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    for name, scale in SCALES.items():
        df[f"scaled_{name}"] = df[name] / scale

    df["hybrid_objective"] = sum(
        weight * df[f"scaled_{name}"] for name, weight in WEIGHTS.items()
    )

    df["coverage_gap"] = (df["coverage_90"] - COVERAGE_TARGET).abs()

    df["coverage_feasible"] = df["coverage_gap"] <= COVERAGE_TOLERANCE

    metrics = [
        "hybrid_objective",
        "rmse",
        "mae",
        "bias",
        "nll",
        "crps",
        "p90_absolute_error",
        "coverage_90",
        "coverage_gap",
        "width_90",
        "weight_variation",
    ]

    print("\nCORRECTED FULL METRIC SUITE")
    print(df.groupby("policy")[metrics].agg(["mean", "std"]).round(5).to_string())

    print("\nCOVERAGE FEASIBILITY RATE")
    print(df.groupby("policy")["coverage_feasible"].mean().round(3).to_string())

    # Pair policies within the same run and recording.
    paired = df.pivot(
        index=["run", "recording_id"],
        columns="policy",
        values="hybrid_objective",
    )

    print("\nPAIRED HYBRID OBJECTIVE VS EQUAL BLEND")
    for policy in paired.columns:
        if policy == "equal_blend":
            continue

        gain = paired["equal_blend"] - paired[policy]

        print(
            f"{policy}: "
            f"mean={gain.mean():.5f}, "
            f"median={gain.median():.5f}, "
            f"win={(gain > 0).mean():.3f}, "
            f"p10={gain.quantile(0.10):.5f}"
        )

    # Aggregate repeated appearances of each patient.
    # There are only ten distinct recording groups.
    patient = (
        df.groupby(["recording_id", "policy"])[
            ["hybrid_objective", "rmse", "nll", "crps"]
        ]
        .mean()
        .reset_index()
    )

    print("\nPATIENT-LEVEL HYBRID OBJECTIVE")
    print(
        patient.pivot(
            index="recording_id",
            columns="policy",
            values="hybrid_objective",
        )
        .round(5)
        .to_string()
    )

    print("\nPATIENT-LEVEL GAINS VS EQUAL BLEND")
    patient_paired = patient.pivot(
        index="recording_id",
        columns="policy",
        values="hybrid_objective",
    )

    for policy in patient_paired.columns:
        if policy == "equal_blend":
            continue

        gains = patient_paired["equal_blend"] - patient_paired[policy]
        print(
            f"{policy}: "
            f"mean={gains.mean():.5f}, "
            f"win={(gains > 0).mean():.3f}, "
            f"worst={gains.min():.5f}"
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False)
    print("\nSaved:", OUTPUT)


if __name__ == "__main__":
    main()
