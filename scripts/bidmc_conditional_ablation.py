"""Phase A.3: Conditional memory value and regularization.

Patient-grouped evaluation, with the full shared metric registry.
No exploratory test patients are used for policy selection.
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
OUTPUT = Path("results/bidmc_conditional_ablation.csv")
TARGET = "target_hr_10s"

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

ALPHAS = [1.0, 10.0, 100.0]
RUNS = 10


def make_model(alpha):
    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
        Ridge(alpha=alpha),
    )


def fit_calibrated(train, features, alpha):
    """Training-only patient cross-fitting for predictive scale."""
    errors = []

    for pid in train["recording_id"].unique():
        inner = train[train.recording_id != pid]
        held = train[train.recording_id == pid]

        model = make_model(alpha)
        model.fit(inner[features], inner[TARGET])

        errors.extend(
            (held[TARGET].to_numpy() - model.predict(held[features])).tolist()
        )

    sigma = max(
        float(np.sqrt(np.mean(np.square(errors)))),
        1e-6,
    )

    model = make_model(alpha)
    model.fit(train[features], train[TARGET])

    return model, sigma


def main():
    df = pd.read_parquet(DATA)
    df = df.dropna(subset=[TARGET]).copy()

    flags = [f"{x}_missing" for x in BASE if f"{x}_missing" in df.columns]
    base = BASE + flags

    missing = set(base + MEMORY) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    ids = np.array(sorted(df.recording_id.unique()))
    if len(ids) != 10:
        raise ValueError("Expected exactly 10 recordings.")

    policies = {
        "current": base,
        "targeted_4": base + MEMORY,
    }

    for feature in MEMORY:
        policies[f"without_{feature}"] = base + [x for x in MEMORY if x != feature]

    rng = np.random.default_rng(12345)
    rows = []

    for run in range(RUNS):
        permutation = rng.permutation(ids)

        train_ids = permutation[:6]
        val_ids = permutation[6:8]

        train = df[df.recording_id.isin(train_ids)]
        validation = df[df.recording_id.isin(val_ids)]

        for alpha in ALPHAS:
            for name, features in policies.items():
                model, sigma = fit_calibrated(train, features, alpha)

                for pid, patient in validation.groupby("recording_id"):
                    y = patient[TARGET].to_numpy()
                    pred = model.predict(patient[features])

                    metrics = evaluate(
                        y,
                        pred,
                        sigma=np.full(len(y), sigma),
                    )

                    rows.append(
                        {
                            "run": run,
                            "recording_id": str(pid),
                            "policy": name,
                            "alpha": alpha,
                            "memory_budget": len(features) - len(base),
                            "sigma": sigma,
                            **metrics,
                        }
                    )

    results = pd.DataFrame(rows)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT, index=False)

    metric_names = [
        "rmse",
        "mae",
        "bias",
        "p90_absolute_error",
        "nll",
        "crps",
        "coverage_90",
        "width_90",
    ]

    available = [m for m in metric_names if m in results.columns]

    print("\nFULL VALIDATION METRIC SUITE")
    print(results.groupby(["alpha", "policy"])[available].mean().round(4).to_string())

    print("\nCONDITIONAL FEATURE CONTRIBUTIONS")
    for alpha in ALPHAS:
        subset = results[results.alpha == alpha]

        for metric in ["rmse", "nll", "crps"]:
            if metric not in available:
                continue

            paired = subset.pivot(
                index=["run", "recording_id"],
                columns="policy",
                values=metric,
            )

            print(f"\nalpha={alpha}, metric={metric}")

            for feature in MEMORY:
                without = f"without_{feature}"

                gain = paired[without] - paired["targeted_4"]

                print(
                    f"{feature:38s} "
                    f"gain={gain.mean():+.5f} "
                    f"win_fraction={(gain > 0).mean():.3f}"
                )

    print("\nPATIENT-LEVEL RMSE VARIABILITY")
    print(
        results.groupby(["alpha", "policy"])["rmse"]
        .agg(["mean", "std", "max"])
        .round(4)
        .to_string()
    )

    # Training-only association diagnostic.
    train_example = df[df.recording_id.isin(ids[:6])]
    print("\nMEMORY FEATURE CORRELATION")
    print(train_example[MEMORY].corr().round(3).to_string())

    print("\nSaved:", OUTPUT)


if __name__ == "__main__":
    main()
