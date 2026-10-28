"""Phase A.6: Predict the conditional value of richer memory.

Patient-grouped cross-fitting. Validation only.
Held-out test recordings are not used.
All controller comparisons use the shared metric registry.
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
OUTPUT = Path("results/bidmc_value_gate.csv")

TARGET = "target_hr_10s"
RUNS = 10

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

# All gate inputs must be available at prediction time.
GATE_FEATURES = [
    "heart_rate_std_15",
    "pulse_rate_std_30",
    "pulse_rate_missing_fraction_15",
    "heart_rate",
    "respiratory_rate",
]


def make_model(alpha=10.0):
    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
        Ridge(alpha=alpha),
    )


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -30, 30)))


def crossfit(train, policies):
    """Generate predictions without training on the evaluated patient."""
    result = train[["recording_id", TARGET] + GATE_FEATURES].copy()

    for name in policies:
        result[name] = np.nan

    for pid in train.recording_id.unique():
        fit = train[train.recording_id != pid]
        held = train[train.recording_id == pid]

        for name, features in policies.items():
            model = make_model()
            model.fit(fit[features], fit[TARGET])
            result.loc[held.index, name] = model.predict(held[features])

    return result


def main():
    df = pd.read_parquet(DATA)
    df = df.dropna(subset=[TARGET]).copy()

    flags = [f"{c}_missing" for c in BASE if f"{c}_missing" in df.columns]

    current = BASE + flags

    policies = {
        "one": current + MEMORY[:1],
        "four": current + MEMORY,
    }

    required = set(current + MEMORY + GATE_FEATURES + [TARGET])
    if required - set(df.columns):
        raise ValueError(f"Missing features: {sorted(required - set(df.columns))}")

    ids = np.array(sorted(df.recording_id.unique()))
    if len(ids) != 10:
        raise ValueError("Expected 10 recordings.")

    rng = np.random.default_rng(12345)

    records = []
    diagnostics = []

    for run in range(RUNS):
        shuffled = rng.permutation(ids)

        train = df[df.recording_id.isin(shuffled[:6])]
        validation = df[df.recording_id.isin(shuffled[6:8])]
        # shuffled[8:] remains completely unused.

        oof = crossfit(train, policies)

        y_oof = oof[TARGET].to_numpy()
        p1_oof = oof["one"].to_numpy()
        p4_oof = oof["four"].to_numpy()

        gain = (y_oof - p1_oof) ** 2 - (y_oof - p4_oof) ** 2

        # Scale the target to avoid unstable large regression targets.
        gain_scale = max(float(np.std(gain)), 1e-6)
        normalized_gain = np.clip(gain / gain_scale, -5, 5)

        # Regularized value estimator.
        gate = make_model(alpha=100.0)
        gate.fit(oof[GATE_FEATURES], normalized_gain)

        predicted_oof_gain = gate.predict(oof[GATE_FEATURES])

        # Fixed predeclared temperature; no validation tuning.
        temperature = 1.0

        def value_weights(frame):
            score = gate.predict(frame[GATE_FEATURES])
            return sigmoid(score / temperature)

        w_oof = value_weights(oof)

        train_predictions = {
            "fixed_one": p1_oof,
            "fixed_four": p4_oof,
            "equal_blend": 0.5 * (p1_oof + p4_oof),
            "value_gate": (p1_oof + w_oof * (p4_oof - p1_oof)),
        }

        # Shared out-of-fold scale calibration.
        sigmas = {
            name: max(
                float(np.sqrt(np.mean((y_oof - pred) ** 2))),
                1e-6,
            )
            for name, pred in train_predictions.items()
        }

        expert_models = {}
        for name, features in policies.items():
            expert_models[name] = make_model()
            expert_models[name].fit(train[features], train[TARGET])

        diagnostics.append(
            {
                "run": run,
                "mean_predicted_gain": float(np.mean(predicted_oof_gain)),
                "mean_oof_weight": float(np.mean(w_oof)),
                "oof_gain_correlation": (
                    float(np.corrcoef(normalized_gain, predicted_oof_gain)[0, 1])
                    if np.std(predicted_oof_gain) > 1e-10
                    else np.nan
                ),
            }
        )

        for pid, patient in validation.groupby("recording_id"):
            y = patient[TARGET].to_numpy()

            p1 = expert_models["one"].predict(patient[policies["one"]])
            p4 = expert_models["four"].predict(patient[policies["four"]])

            w = value_weights(patient)

            predictions = {
                "fixed_one": p1,
                "fixed_four": p4,
                "equal_blend": 0.5 * (p1 + p4),
                "value_gate": p1 + w * (p4 - p1),
            }

            for name, prediction in predictions.items():
                scores = evaluate(
                    y,
                    prediction,
                    sigma=np.full(len(y), sigmas[name]),
                )

                records.append(
                    {
                        "run": run,
                        "recording_id": str(pid),
                        "policy": name,
                        "memory_weight": (
                            float(np.mean(w))
                            if name == "value_gate"
                            else {
                                "fixed_one": 0.0,
                                "fixed_four": 1.0,
                                "equal_blend": 0.5,
                            }[name]
                        ),
                        "weight_variation": (
                            float(np.mean(np.abs(np.diff(w))))
                            if name == "value_gate" and len(w) > 1
                            else 0.0
                        ),
                        **scores,
                    }
                )

    results = pd.DataFrame(records)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT, index=False)

    columns = [
        "rmse",
        "mae",
        "bias",
        "p90_absolute_error",
        "nll",
        "crps",
        "coverage_90",
        "width_90",
        "memory_weight",
        "weight_variation",
    ]

    columns = [c for c in columns if c in results.columns]

    print("\nFULL VALIDATION METRIC SUITE")
    print(results.groupby("policy")[columns].agg(["mean", "std"]).round(4).to_string())

    print("\nPAIRED IMPROVEMENTS VS FIXED FOUR")

    for metric in ["rmse", "nll", "crps"]:
        paired = results.pivot(
            index=["run", "recording_id"],
            columns="policy",
            values=metric,
        )

        print(f"\n{metric}")

        for candidate in ["fixed_one", "equal_blend", "value_gate"]:
            gain = paired["fixed_four"] - paired[candidate]
            print(
                candidate,
                "mean:",
                round(gain.mean(), 5),
                "median:",
                round(gain.median(), 5),
                "win:",
                round((gain > 0).mean(), 3),
            )

    print("\nVALUE ESTIMATOR DIAGNOSTICS")
    print(pd.DataFrame(diagnostics).round(4).to_string(index=False))

    print("\nWORST RECORDING RMSE")
    print(results.groupby("policy")["rmse"].max().round(4).to_string())

    print("\nSaved:", OUTPUT)


if __name__ == "__main__":
    main()
