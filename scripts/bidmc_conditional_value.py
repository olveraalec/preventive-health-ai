"""Phase A.2: Conditional value and patient-level robustness.

All policies evaluated on identical patient partitions.
Probabilistic scale estimated with training-only group CV.
Previously inspected patient recordings make this exploratory,
not a final untouched test.
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from healthcare_ai_lab.evaluation.registry import evaluate


DATA = Path("data/processed/bidmc_batch.parquet")
OUTPUT = Path("results/bidmc_conditional_value.csv")

TARGET = "target_hr_10s"
SEED = 12345
RUNS = 10

BASE = [
    "heart_rate",
    "pulse_rate",
    "respiratory_rate",
    "spo2",
]

TARGETED = [
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


def fit_and_calibrate(train, features):
    """Fit a predictor; calibrate sigma using held-out patients."""
    residuals = []

    for patient_id in train.recording_id.unique():
        inner_train = train[train.recording_id != patient_id]
        inner_valid = train[train.recording_id == patient_id]

        estimator = make_model()
        estimator.fit(
            inner_train[features],
            inner_train[TARGET],
        )

        predicted = estimator.predict(inner_valid[features])

        residuals.extend((inner_valid[TARGET].to_numpy() - predicted).tolist())

    # One Gaussian scale per policy, from training CV only.
    sigma = max(
        float(np.sqrt(np.mean(np.square(residuals)))),
        1e-6,
    )

    final_model = make_model()
    final_model.fit(train[features], train[TARGET])

    return final_model, sigma


def get_policies(df):
    flags = [f"{col}_missing" for col in BASE if f"{col}_missing" in df.columns]
    current = BASE + flags

    pattern = re.compile(r"_(?:mean|std|missing_fraction)_(?:5|15|30)$")

    history = [col for col in df.columns if pattern.search(col) and col not in current]

    missing = set(TARGETED) - set(history)
    if missing:
        raise ValueError(f"Missing targeted features: {sorted(missing)}")

    recent = [col for col in history if col.endswith("_5")][:4]

    return {
        "current": current,
        "targeted_1": current + TARGETED[:1],
        "targeted_2": current + TARGETED[:2],
        "targeted_4": current + TARGETED,
        "recent_4": current + recent,
        "full_history": current + history,
    }, len(current)


def main():
    df = pd.read_parquet(DATA)

    df = df.sort_values(["recording_id", "time_seconds"]).dropna(subset=[TARGET]).copy()

    policies, base_size = get_policies(df)
    ids = np.array(sorted(df.recording_id.unique()))

    if len(ids) != 10:
        raise ValueError("This exploratory design expects 10 recordings.")

    rng = np.random.default_rng(SEED)
    rows = []

    for run in range(RUNS):
        permutation = rng.permutation(ids)

        train_ids = permutation[:6]
        val_ids = permutation[6:8]
        test_ids = permutation[8:]

        train = df[df.recording_id.isin(train_ids)]

        for name, features in policies.items():
            estimator, sigma = fit_and_calibrate(train, features)

            for split_name, selected_ids in [
                ("validation", val_ids),
                ("test_exploratory", test_ids),
            ]:
                for patient_id in selected_ids:
                    patient = df[df.recording_id == patient_id]

                    truth = patient[TARGET].to_numpy()
                    pred = estimator.predict(patient[features])

                    scores = evaluate(
                        truth,
                        pred,
                        sigma=np.full(len(truth), sigma),
                    )

                    rows.append(
                        {
                            "run": run,
                            "split": split_name,
                            "recording_id": str(patient_id),
                            "policy": name,
                            "memory_budget": len(features) - base_size,
                            "sigma": sigma,
                            **scores,
                        }
                    )

    results = pd.DataFrame(rows)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT, index=False)

    metric_cols = [
        name
        for name in [
            "rmse",
            "mae",
            "bias",
            "p90_absolute_error",
            "nll",
            "crps",
            "coverage_90",
            "width_90",
        ]
        if name in results.columns
    ]

    print("\nFULL METRIC SUITE")
    summary = (
        results.groupby(["split", "policy"])[metric_cols].agg(["mean", "std"]).round(4)
    )
    print(summary.to_string())

    print("\nPAIRED PATIENT-LEVEL RMSE GAINS")
    paired = results.pivot(
        index=["run", "split", "recording_id"],
        columns="policy",
        values="rmse",
    )

    for split_name in [
        "validation",
        "test_exploratory",
    ]:
        print(f"\n{split_name.upper()}")

        part = paired.xs(split_name, level="split")

        for policy in policies:
            if policy == "current":
                continue

            gains = part["current"] - part[policy]

            print(
                f"{policy:16s} "
                f"mean_gain={gains.mean():.4f} "
                f"median={gains.median():.4f} "
                f"win_rate={(gains > 0).mean():.3f}"
            )

    print("\nWORST-RECORDING RMSE")
    worst = results.groupby(["split", "policy"])["rmse"].max()
    print(worst.round(4).to_string())

    print("\nINCREMENTAL TARGETED FEATURE VALUE")
    for split_name in [
        "validation",
        "test_exploratory",
    ]:
        part = paired.xs(split_name, level="split")

        for previous, current in [
            ("current", "targeted_1"),
            ("targeted_1", "targeted_2"),
            ("targeted_2", "targeted_4"),
        ]:
            gains = part[previous] - part[current]
            print(f"{split_name}: {previous} -> {current}: {gains.mean():+.5f}")

    print("\nSaved:", OUTPUT)


if __name__ == "__main__":
    main()
