"""Learned error-feedback ablation with chronological fitting.

Synthetic regime labels are TRAINING-ONLY privileged information.
Validation hidden labels and conditional means are evaluation-only.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from healthcare_ai_lab.simulation.generator import (
    SimulationConfig,
    generate_telemetry,
)
from healthcare_ai_lab.models.state_estimator import (
    BayesianStateEstimator,
)


BASE = ["x1", "x2", "regime_proxy"]


def ridge(train, features):
    model = make_pipeline(
        StandardScaler(),
        Ridge(alpha=10.0)
    )
    model.fit(train[features], train["y"])
    return model


def evaluate_seed(seed, observations=5000):
    df = generate_telemetry(
        SimulationConfig(
            seed=seed,
            n_observations=observations,
        )
    ).copy()

    n = len(df)

    reference_end = int(n * 0.30)
    policy_end = int(n * 0.60)
    validation_end = int(n * 0.80)

    reference_train = df.iloc[:reference_end]
    policy_train = df.iloc[reference_end:policy_end]

    # Reference predictions are out-of-sample for
    # policy training and subsequent validation.
    reference = ridge(reference_train, BASE)

    df["reference_residual"] = np.nan

    df.loc[
        df.index[reference_end:],
        "reference_residual"
    ] = (
        df["y"].iloc[reference_end:].to_numpy()
        - reference.predict(df.iloc[reference_end:][BASE])
    )

    # At decision t, use feedback through t-1.
    previous_error = df["reference_residual"].shift(1)

    df["previous_abs_error"] = previous_error.abs()
    df["previous_squared_error"] = previous_error.pow(2)

    df["rolling_error_energy"] = (
        previous_error.pow(2)
        .rolling(12, min_periods=12)
        .mean()
    )

    df["rolling_abs_error"] = (
        previous_error.abs()
        .rolling(12, min_periods=12)
        .mean()
    )

    # Fair telemetry-history controls.
    df["proxy_history_mean"] = (
        df["regime_proxy"]
        .shift(1)
        .rolling(12, min_periods=12)
        .mean()
    )

    df["proxy_history_std"] = (
        df["regime_proxy"]
        .shift(1)
        .rolling(12, min_periods=12)
        .std()
    )

    proxy_features = [
        "regime_proxy",
        "proxy_history_mean",
        "proxy_history_std",
    ]

    feedback_features = proxy_features + [
        "previous_abs_error",
        "previous_squared_error",
        "rolling_error_energy",
        "rolling_abs_error",
    ]

    # Start only after full out-of-sample
    # historical error windows are available.
    fit_rows = df.iloc[reference_end + 12:policy_end]
    val = df.iloc[policy_end:validation_end]

    classifiers = {}

    for name, features in {
        "learned_proxy_history": proxy_features,
        "learned_error_feedback": feedback_features,
    }.items():

        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0,
                max_iter=3000
            )
        )

        clf.fit(
            fit_rows[features],
            fit_rows["true_regime"]
        )

        classifiers[name] = (clf, features)

    # Specialists use only the earlier 60% and
    # are evaluated on the later validation period.
    specialist_train = df.iloc[:policy_end]

    predictions = {}

    for lag in (12, 30):
        features = BASE + [f"phi{lag}"]

        model = ridge(specialist_train, features)
        predictions[lag] = model.predict(val[features])

    # Bayesian proxy baseline.
    estimator = BayesianStateEstimator(
        use_error_feedback=False
    )

    probabilities = {
        "bayesian_proxy": []
    }

    for _, row in val.iterrows():
        state = estimator.update(
            timestamp=int(row["time"]),
            proxy=float(row["regime_proxy"])
        )
        probabilities["bayesian_proxy"].append(
            state.probability_regime_1
        )

    for name, (model, features) in classifiers.items():
        probabilities[name] = model.predict_proba(
            val[features]
        )[:, 1]

    y = val["y"].to_numpy()
    z = val["true_regime"].to_numpy()
    true_mean = val["true_mean"].to_numpy()

    oracle_loss = np.minimum(
        (true_mean - predictions[12]) ** 2,
        (true_mean - predictions[30]) ** 2
    )

    records = []

    for name, raw_probability in probabilities.items():
        p = np.asarray(raw_probability)

        chosen_lag = np.where(p >= 0.5, 12, 30)

        selected_prediction = np.where(
            chosen_lag == 12,
            predictions[12],
            predictions[30]
        )

        structural_loss = (
            true_mean - selected_prediction
        ) ** 2

        records.append({
            "seed": seed,
            "policy": name,
            "rmse": float(np.sqrt(
                np.mean((y - selected_prediction) ** 2)
            )),
            "regret": float(np.mean(
                structural_loss - oracle_loss
            )),
            "regime_accuracy": float(np.mean(
                (p >= 0.5) == z
            )),
            "belief_brier": float(np.mean(
                (p - z) ** 2
            )),
            "switch_rate": float(np.mean(
                chosen_lag[1:] != chosen_lag[:-1]
            )),
        })

    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument(
        "--output",
        default="results/learned_feedback.csv"
    )
    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs must be positive")

    results = pd.DataFrame([
        row
        for i in range(args.runs)
        for row in evaluate_seed(50000 + i)
    ])

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False)

    print("\nLEARNED FEEDBACK EXPERIMENT")
    print(
        results.groupby("policy")[
            [
                "rmse", "regret", "regime_accuracy",
                "belief_brier", "switch_rate"
            ]
        ].agg(["mean", "std"]).round(4).to_string()
    )

    print(f"\nSaved to {output}")


if __name__ == "__main__":
    main()
