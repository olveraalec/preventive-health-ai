"""Phase A.7: Out-of-sample information-value generalization."""

from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from healthcare_ai_lab.evaluation.registry import evaluate

DATA = Path("data/processed/bidmc_batch.parquet")
OUTPUT = Path("results/bidmc_value_generalization.csv")

TARGET = "target_hr_10s"
RUNS = 10

BASE = ["heart_rate", "pulse_rate", "respiratory_rate", "spo2"]
MEMORY = [
    "pulse_rate_mean_30",
    "heart_rate_std_15",
    "pulse_rate_std_30",
    "pulse_rate_missing_fraction_15",
]
GATE = [
    "heart_rate_std_15",
    "pulse_rate_std_30",
    "pulse_rate_missing_fraction_15",
    "heart_rate",
    "respiratory_rate",
]


def model(alpha):
    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
        Ridge(alpha=alpha),
    )


def crossfit(train, policies):
    result = train[["recording_id", TARGET] + GATE].copy()

    for name in policies:
        result[name] = np.nan

    for pid in train.recording_id.unique():
        fit = train[train.recording_id != pid]
        held = train[train.recording_id == pid]

        for name, columns in policies.items():
            expert = model(10.0)
            expert.fit(fit[columns], fit[TARGET])
            result.loc[held.index, name] = expert.predict(held[columns])

    return result


def safe_spearman(a, b):
    if np.std(a) < 1e-10 or np.std(b) < 1e-10:
        return np.nan
    return float(spearmanr(a, b).statistic)


def main():
    df = pd.read_parquet(DATA)
    df = df.dropna(subset=[TARGET]).copy()

    flags = [f"{c}_missing" for c in BASE if f"{c}_missing" in df.columns]

    current = BASE + flags

    policies = {
        "one": current + MEMORY[:1],
        "four": current + MEMORY,
    }

    needed = set(current + MEMORY + GATE + [TARGET])
    if needed - set(df.columns):
        raise ValueError(f"Missing: {sorted(needed - set(df.columns))}")

    ids = np.array(sorted(df.recording_id.unique()))
    if len(ids) != 10:
        raise ValueError("Expected 10 recordings")

    rng = np.random.default_rng(12345)

    scores = []
    diagnostics = []

    for run in range(RUNS):
        shuffled = rng.permutation(ids)
        train = df[df.recording_id.isin(shuffled[:6])]
        validation = df[df.recording_id.isin(shuffled[6:8])]

        # Cross-fitted training predictions.
        oof = crossfit(train, policies)
        y_train = oof[TARGET].to_numpy()

        observed_gain_train = (y_train - oof["one"].to_numpy()) ** 2 - (
            y_train - oof["four"].to_numpy()
        ) ** 2

        scale = max(float(np.std(observed_gain_train)), 1e-6)

        value_estimator = model(100.0)
        value_estimator.fit(
            oof[GATE],
            np.clip(observed_gain_train / scale, -5, 5),
        )

        expert_models = {}
        for name, columns in policies.items():
            expert_models[name] = model(10.0)
            expert_models[name].fit(train[columns], train[TARGET])

        # Calibrate predictive scales using training OOF.
        train_gate_weight = 1 / (
            1 + np.exp(-np.clip(value_estimator.predict(oof[GATE]), -30, 30))
        )

        train_preds = {
            "fixed_one": oof["one"].to_numpy(),
            "fixed_four": oof["four"].to_numpy(),
        }
        train_preds["equal_blend"] = (
            train_preds["fixed_one"] + train_preds["fixed_four"]
        ) / 2
        train_preds["value_gate"] = train_preds["fixed_one"] + train_gate_weight * (
            train_preds["fixed_four"] - train_preds["fixed_one"]
        )

        sigma = {
            k: max(
                float(np.sqrt(np.mean((y_train - pred) ** 2))),
                1e-6,
            )
            for k, pred in train_preds.items()
        }

        for pid, patient in validation.groupby("recording_id"):
            y = patient[TARGET].to_numpy()

            p1 = expert_models["one"].predict(patient[policies["one"]])
            p4 = expert_models["four"].predict(patient[policies["four"]])

            predicted_value = value_estimator.predict(patient[GATE])
            realized_value = ((y - p1) ** 2 - (y - p4) ** 2) / scale

            weights = 1 / (1 + np.exp(-np.clip(predicted_value, -30, 30)))

            predictions = {
                "fixed_one": p1,
                "fixed_four": p4,
                "equal_blend": 0.5 * (p1 + p4),
                "value_gate": p1 + weights * (p4 - p1),
            }

            diagnostics.append(
                {
                    "run": run,
                    "recording_id": str(pid),
                    "spearman": safe_spearman(predicted_value, realized_value),
                    "sign_accuracy": float(
                        np.mean((predicted_value > 0) == (realized_value > 0))
                    ),
                    "mean_predicted_value": float(np.mean(predicted_value)),
                    "mean_realized_value": float(np.mean(realized_value)),
                    "mean_weight": float(np.mean(weights)),
                }
            )

            for name, prediction in predictions.items():
                metric_values = evaluate(
                    y,
                    prediction,
                    sigma=np.full(len(y), sigma[name]),
                )

                scores.append(
                    {
                        "run": run,
                        "recording_id": str(pid),
                        "policy": name,
                        **metric_values,
                    }
                )

    result = pd.DataFrame(scores)
    diag = pd.DataFrame(diagnostics)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT, index=False)
    diag.to_csv(
        OUTPUT.with_name("bidmc_value_diagnostics.csv"),
        index=False,
    )

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

    print("\nFULL METRIC REGISTRY")
    print(result.groupby("policy")[metrics].agg(["mean", "std"]).round(4).to_string())

    print("\nOUT-OF-SAMPLE INFORMATION-VALUE DIAGNOSTICS")
    print(diag.describe().round(4).to_string())

    print("\nPAIRED GAIN VS FIXED FOUR")
    for metric in ["rmse", "nll", "crps"]:
        paired = result.pivot(
            index=["run", "recording_id"],
            columns="policy",
            values=metric,
        )
        gain = paired["fixed_four"] - paired["value_gate"]
        print(
            metric,
            "mean:",
            round(gain.mean(), 5),
            "median:",
            round(gain.median(), 5),
            "win fraction:",
            round((gain > 0).mean(), 3),
        )

    print("\nSaved:", OUTPUT)


if __name__ == "__main__":
    main()
