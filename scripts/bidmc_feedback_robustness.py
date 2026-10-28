"""Repeated patient-level generalization of residual feedback."""

import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 12345
RUNS = 30
HORIZON = 10
TARGET = "target_hr_10s"

df = pd.read_parquet("data/processed/bidmc_batch.parquet")
df = df.sort_values(["recording_id", "time_seconds"]).reset_index(drop=True)

ids = np.array(sorted(df.recording_id.unique()))

BASE = [
    "heart_rate",
    "pulse_rate",
    "respiratory_rate",
    "spo2",
]

BASE += [f"{col}_missing" for col in BASE if f"{col}_missing" in df.columns]

HISTORY = [
    "error_lag_10",
    "error_lag_12",
    "error_lag_20",
    "error_lag_30",
    "energy_12",
    "mean_error_12",
]

INTERACTIONS = [f"state_x_{c}" for c in HISTORY]


POLICIES = {
    "baseline": ([], 0.0),
    "feedback_010": (HISTORY, 0.10),
    "feedback_025": (HISTORY, 0.25),
    "feedback_050": (HISTORY, 0.50),
    "feedback_075": (HISTORY, 0.75),
    "feedback_100": (HISTORY, 1.00),
    "interactions_025": (HISTORY + INTERACTIONS, 0.25),
}


def make_model(alpha):
    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
        Ridge(alpha=alpha),
    )


def continuous_gate(correction, scale, alpha0=0.50, kappa=1.0):
    """Smoothly suppress unusually large feedback corrections."""
    correction = np.asarray(correction, dtype=float)
    scale = max(float(scale), 1e-6)

    alpha_t = alpha0 / (1.0 + kappa * np.abs(correction) / scale)

    return alpha_t * correction, alpha_t


def bounded_continuous_gate(
    correction,
    scale,
    alpha0=0.35,
    kappa=1.0,
    alpha_min=0.05,
    alpha_max=0.35,
):
    """Continuous feedback with explicit correction-weight bounds."""

    correction = np.asarray(correction, dtype=float)
    scale = max(float(scale), 1e-6)

    if not (0 <= alpha_min <= alpha_max <= 1):
        raise ValueError("Invalid controller weight bounds")

    raw_alpha = alpha0 / (1.0 + kappa * np.abs(correction) / scale)

    alpha_t = np.clip(raw_alpha, alpha_min, alpha_max)

    return alpha_t * correction, alpha_t


def prepare(part, base_model, state_center):
    part = part.copy()
    part["baseline"] = base_model.predict(part[BASE])
    part["error"] = part[TARGET] - part["baseline"]

    group = part.groupby("recording_id", sort=False)

    for lag in [10, 12, 20, 30]:
        part[f"error_lag_{lag}"] = group["error"].shift(lag)

    part["energy_12"] = group["error"].transform(
        lambda s: s.shift(HORIZON).pow(2).rolling(12, min_periods=6).mean()
    )

    part["mean_error_12"] = group["error"].transform(
        lambda s: s.shift(HORIZON).rolling(12, min_periods=6).mean()
    )

    state = part["heart_rate"] - state_center

    for col in HISTORY:
        part[f"state_x_{col}"] = state * part[col]

    return part.dropna(subset=[TARGET] + HISTORY + INTERACTIONS)


rows = []
rng = np.random.default_rng(SEED)

for run in range(RUNS):
    permutation = rng.permutation(ids)

    base_ids = permutation[:4]
    feedback_ids = permutation[4:6]
    validation_ids = permutation[6:8]
    test_ids = permutation[8:10]

    base_train = df[df.recording_id.isin(base_ids)]
    base_train = base_train.dropna(subset=[TARGET])

    base_model = make_model(alpha=10)
    base_model.fit(base_train[BASE], base_train[TARGET])

    state_center = df[df.recording_id.isin(feedback_ids)]["heart_rate"].mean()

    feedback = prepare(df[df.recording_id.isin(feedback_ids)], base_model, state_center)

    validation = prepare(
        df[df.recording_id.isin(validation_ids)], base_model, state_center
    )

    test = prepare(df[df.recording_id.isin(test_ids)], base_model, state_center)

    # Fit controllers using only the feedback-training recordings.
    controllers = {}

    for name, (features, shrinkage) in POLICIES.items():
        if name == "baseline":
            continue

        model = make_model(alpha=20)
        model.fit(
            feedback[features],
            feedback["error"],
        )
        controllers[name] = model

    # Evaluate on validation and held-out test recordings.
    for split, data in [
        ("validation", validation),
        ("test", test),
    ]:
        for name, (features, shrinkage) in POLICIES.items():
            if name == "baseline":
                correction = np.zeros(len(data))
            else:
                correction = shrinkage * controllers[name].predict(data[features])

            prediction = data["baseline"].to_numpy() + correction

            error = data[TARGET].to_numpy() - prediction

            rows.append(
                {
                    "run": run,
                    "split": split,
                    "model": name,
                    "rmse": float(np.sqrt(np.mean(error**2))),
                    "mae": float(np.mean(np.abs(error))),
                    "bias": float(np.mean(error)),
                }
            )

    # Continuous feedback experiment:
    # Tune gate strength on validation, evaluate on test.
    feedback_model = controllers["feedback_050"]

    scale = max(float(feedback["error"].std()), 1e-6)

    candidate_kappas = [0.0, 0.5, 1.0, 2.0, 4.0]

    raw_val_correction = feedback_model.predict(validation[HISTORY])

    y_val = validation[TARGET].to_numpy()
    base_val = validation["baseline"].to_numpy()

    validation_scores = {}

    for kappa in candidate_kappas:
        correction, _ = continuous_gate(
            raw_val_correction,
            scale,
            alpha0=0.50,
            kappa=kappa,
        )

        prediction = base_val + correction

        validation_scores[kappa] = np.sqrt(np.mean((y_val - prediction) ** 2))

    chosen_kappa = min(
        validation_scores,
        key=validation_scores.get,
    )

    # Two continuous strategies:
    # 1. Fixed predeclared gate
    # 2. Validation-selected gate
    gate_policies = {
        "continuous_k1": 1.0,
        "continuous_selected": chosen_kappa,
    }

    for split, data in [
        ("validation", validation),
        ("test", test),
    ]:
        raw_correction = feedback_model.predict(data[HISTORY])

        y_true = data[TARGET].to_numpy()
        base_prediction = data["baseline"].to_numpy()

        for policy, kappa in gate_policies.items():
            correction, alpha_t = continuous_gate(
                raw_correction,
                scale,
                alpha0=0.50,
                kappa=kappa,
            )

            prediction = base_prediction + correction
            error = y_true - prediction

            rows.append(
                {
                    "run": run,
                    "split": split,
                    "model": policy,
                    "rmse": float(np.sqrt(np.mean(error**2))),
                    "mae": float(np.mean(np.abs(error))),
                    "bias": float(np.mean(error)),
                    "mean_alpha": float(np.mean(alpha_t)),
                    "alpha_std": float(np.std(alpha_t)),
                    "kappa": float(kappa),
                }
            )

    # Predeclared bounded controller candidates.
    # No use of test labels for parameter selection.
    bounded_candidates = {
        "bounded_025": {
            "alpha0": 0.25,
            "kappa": 1.0,
            "alpha_min": 0.05,
            "alpha_max": 0.25,
        },
        "bounded_035": {
            "alpha0": 0.35,
            "kappa": 1.0,
            "alpha_min": 0.05,
            "alpha_max": 0.35,
        },
    }

    feedback_model = controllers["feedback_050"]
    scale = max(float(feedback["error"].std()), 1e-6)

    # Evaluate both candidates on validation.
    validation_scores = {}

    raw_val = feedback_model.predict(validation[HISTORY])

    for name, params in bounded_candidates.items():
        correction, _ = bounded_continuous_gate(raw_val, scale, **params)

        residual = (
            validation[TARGET].to_numpy()
            - validation["baseline"].to_numpy()
            - correction
        )

        validation_scores[name] = float(np.sqrt(np.mean(residual**2)))

    selected_bounded = min(validation_scores, key=validation_scores.get)

    # Evaluate on both validation and untouched test recordings.
    for split, data in [
        ("validation", validation),
        ("test", test),
    ]:
        raw = feedback_model.predict(data[HISTORY])

        for name, params in bounded_candidates.items():
            correction, alpha_t = bounded_continuous_gate(raw, scale, **params)

            residual = (
                data[TARGET].to_numpy() - data["baseline"].to_numpy() - correction
            )

            rows.append(
                {
                    "run": run,
                    "split": split,
                    "model": name,
                    "rmse": float(np.sqrt(np.mean(residual**2))),
                    "mae": float(np.mean(np.abs(residual))),
                    "bias": float(np.mean(residual)),
                    "mean_alpha": float(np.mean(alpha_t)),
                    "alpha_std": float(np.std(alpha_t)),
                }
            )

            if name == selected_bounded:
                rows.append(
                    {
                        "run": run,
                        "split": split,
                        "model": "bounded_selected",
                        "rmse": float(np.sqrt(np.mean(residual**2))),
                        "mae": float(np.mean(np.abs(residual))),
                        "bias": float(np.mean(residual)),
                        "mean_alpha": float(np.mean(alpha_t)),
                        "alpha_std": float(np.std(alpha_t)),
                    }
                )


results = pd.DataFrame(rows)

output = Path("results/bidmc_feedback_robustness.csv")
output.parent.mkdir(parents=True, exist_ok=True)
results.to_csv(output, index=False)


print("\nPERFORMANCE ACROSS SPLITS")

summary = results.groupby(["split", "model"])["rmse"].agg(["mean", "std"]).round(4)
print(summary.to_string())

validation_rmse = results[results["split"] == "validation"].pivot(
    index="run", columns="model", values="rmse"
)

test_rmse = results[results["split"] == "test"].pivot(
    index="run", columns="model", values="rmse"
)

# Select correction strength on validation only.
candidates = [name for name in POLICIES if name.startswith("feedback_")]

chosen_policy = validation_rmse[candidates].idxmin(axis=1)

selected_test_rmse = pd.Series(
    {run: test_rmse.loc[run, policy] for run, policy in chosen_policy.items()}
)

baseline_test = test_rmse["baseline"]
fixed_test = test_rmse["feedback_025"]

gain = baseline_test - selected_test_rmse
fixed_gain = baseline_test - fixed_test

print("\nVALIDATION-SELECTED CONTROLLER")
print("Selected policies:")
print(chosen_policy.value_counts().to_string())

print("\nHELD-OUT TEST COMPARISON")
print("Baseline RMSE:", round(baseline_test.mean(), 4))
print("Fixed 0.25 RMSE:", round(fixed_test.mean(), 4))
print("Validation-selected RMSE:", round(selected_test_rmse.mean(), 4))

print("\nPAIRED GAINS VS BASELINE")
print("Fixed mean gain:", round(fixed_gain.mean(), 4))
print("Selected mean gain:", round(gain.mean(), 4))
print("Selected win rate:", round((gain > 0).mean(), 3))

print("\nPAIRED SELECTED VS FIXED")
difference = fixed_test - selected_test_rmse
print("Mean gain:", round(difference.mean(), 4))
print("Win rate:", round((difference > 0).mean(), 3))

print(f"\nSaved results to {output}")


print("\nCONTINUOUS CONTROLLER COMPARISON")

comparison = results[
    (results["split"] == "test")
    & results["model"].isin(
        [
            "baseline",
            "feedback_025",
            "feedback_050",
            "continuous_k1",
            "continuous_selected",
        ]
    )
]

print(comparison.groupby("model")["rmse"].agg(["mean", "std"]).round(4).to_string())

paired_continuous = comparison.pivot(
    index="run",
    columns="model",
    values="rmse",
)

print("\nPAIRED IMPROVEMENTS VS FIXED 0.25")

for policy in ["continuous_k1", "continuous_selected"]:
    gains = paired_continuous["feedback_025"] - paired_continuous[policy]

    print(f"\n{policy}")
    print("Mean gain:", round(gains.mean(), 4))
    print("Win fraction:", round((gains > 0).mean(), 3))

print("\nCONTINUOUS WEIGHT DIAGNOSTICS")
print(
    comparison.groupby("model")[["mean_alpha", "alpha_std", "kappa"]]
    .mean()
    .round(4)
    .to_string()
)

print("\nSelected gate strengths:")
print(
    results[(results["split"] == "test") & (results["model"] == "continuous_selected")][
        "kappa"
    ]
    .value_counts()
    .sort_index()
    .to_string()
)


print("\nBOUNDED CONTROLLER EXPERIMENT")

models = [
    "baseline",
    "feedback_025",
    "continuous_selected",
    "bounded_025",
    "bounded_035",
    "bounded_selected",
]

test_results = results[(results["split"] == "test") & (results["model"].isin(models))]

print(test_results.groupby("model")["rmse"].agg(["mean", "std"]).round(4).to_string())

paired = test_results.pivot(index="run", columns="model", values="rmse")

print("\nPAIRED GAINS AGAINST FIXED 0.25")

for policy in [
    "bounded_025",
    "bounded_035",
    "bounded_selected",
]:
    gain = paired["feedback_025"] - paired[policy]

    print(f"\n{policy}")
    print("Mean gain:", round(gain.mean(), 4))
    print("Win fraction:", round((gain > 0).mean(), 3))
    print("Median gain:", round(gain.median(), 4))

print("\nBOUNDED CONTROLLER WEIGHTS")
print(
    test_results.groupby("model")[["mean_alpha", "alpha_std"]]
    .mean()
    .round(4)
    .to_string()
)
