"""Phase A.4 — State-conditional predictive value of memory.

Uses training-only calibration and state thresholds.
Reports the shared metrics suite on patient-held-out validation.
Exploratory only: recordings are reused across runs.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from healthcare_ai_lab.evaluation.registry import evaluate

DATA = Path("data/processed/bidmc_batch.parquet")
OUTPUT = Path("results/bidmc_state_conditional_value.csv")

TARGET = "target_hr_10s"
STATE = "heart_rate_std_15"
RUNS = 10
SEED = 12345

BASE = [
    "heart_rate",
    "pulse_rate",
    "respiratory_rate",
    "spo2",
]

MEMORY = [
    "pulse_rate_mean_30",
    "heart_rate_std_15",
    "pulse_rate_std_30",
    "pulse_rate_missing_fraction_15",
]


def make_model():
    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
        Ridge(alpha=10.0),
    )


def fit_calibrated(train, columns):
    residuals = []

    # Out-of-fold residuals from held-out training patients.
    for pid in train.recording_id.unique():
        inner = train[train.recording_id != pid]
        held = train[train.recording_id == pid]

        model = make_model()
        model.fit(inner[columns], inner[TARGET])

        residuals.extend(held[TARGET].to_numpy() - model.predict(held[columns]))

    sigma = max(
        float(np.sqrt(np.mean(np.square(residuals)))),
        1e-6,
    )

    model = make_model()
    model.fit(train[columns], train[TARGET])

    return model, sigma


def main():
    df = pd.read_parquet(DATA)
    df = df.sort_values(["recording_id", "time_seconds"]).dropna(subset=[TARGET]).copy()

    flags = [f"{c}_missing" for c in BASE if f"{c}_missing" in df.columns]
    base = BASE + flags

    policies = {
        "current": base,
        "targeted_1": base + MEMORY[:1],
        "targeted_4": base + MEMORY,
        "without_pulse_mean": base + MEMORY[1:],
    }

    missing = set(sum(policies.values(), []) + [STATE]) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    ids = np.array(sorted(df.recording_id.unique()))
    rng = np.random.default_rng(SEED)
    rows = []

    if len(ids) != 10:
        raise ValueError("Expected ten patient recordings.")

    for run in range(RUNS):
        shuffled = rng.permutation(ids)
        train = df[df.recording_id.isin(shuffled[:6])]
        validation = df[df.recording_id.isin(shuffled[6:8])].copy()

        # Thresholds learned from training patients only.
        thresholds = train[STATE].dropna().quantile([1 / 3, 2 / 3]).to_numpy()

        if len(thresholds) != 2 or not np.isfinite(thresholds).all():
            raise ValueError("Invalid training-state thresholds.")

        validation["state_group"] = np.select(
            [
                validation[STATE] <= thresholds[0],
                validation[STATE] <= thresholds[1],
            ],
            ["low", "medium"],
            default="high",
        )

        for name, columns in policies.items():
            model, sigma = fit_calibrated(train, columns)
            validation["prediction"] = model.predict(validation[columns])

            # Evaluate every patient/state combination separately.
            for (pid, state), part in validation.groupby(
                ["recording_id", "state_group"]
            ):
                if len(part) < 5:
                    continue

                scores = evaluate(
                    part[TARGET].to_numpy(),
                    part["prediction"].to_numpy(),
                    sigma=np.full(len(part), sigma),
                )

                rows.append(
                    {
                        "run": run,
                        "recording_id": str(pid),
                        "state_group": state,
                        "policy": name,
                        "observations": len(part),
                        "memory_budget": len(columns) - len(base),
                        **scores,
                    }
                )

    results = pd.DataFrame(rows)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT, index=False)

    metrics = [
        "rmse",
        "mae",
        "bias",
        "p90_absolute_error",
        "nll",
        "crps",
        "coverage_90",
        "width_90",
    ]
    metrics = [m for m in metrics if m in results]

    print("\nFULL STATE-CONDITIONAL METRIC SUITE")
    print(
        results.groupby(["state_group", "policy"])[metrics]
        .agg(["mean", "std"])
        .round(4)
        .to_string()
    )

    print("\nPAIRED STATE-CONDITIONAL IMPROVEMENTS")

    for metric in ["rmse", "nll", "crps"]:
        if metric not in results:
            continue

        paired = results.pivot_table(
            index=["run", "recording_id", "state_group"],
            columns="policy",
            values=metric,
        )

        print(f"\nMETRIC: {metric}")

        for state in ["low", "medium", "high"]:
            try:
                subset = paired.xs(state, level="state_group")
            except KeyError:
                continue

            for policy in [
                "targeted_1",
                "targeted_4",
                "without_pulse_mean",
            ]:
                gains = (subset["current"] - subset[policy]).dropna()

                if gains.empty:
                    continue

                print(
                    f"{state:7s} {policy:19s} "
                    f"gain={gains.mean():+.4f} "
                    f"median={gains.median():+.4f} "
                    f"win={((gains > 0).mean()):.3f} "
                    f"groups={len(gains)}"
                )

    print("\nSaved:", OUTPUT)


if __name__ == "__main__":
    main()
