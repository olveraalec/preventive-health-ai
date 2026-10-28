"""Causal residual-feedback experiment on BIDMC recordings."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 12345
HORIZON = 10
OUT = Path("results")
OUT.mkdir(exist_ok=True)

df = pd.read_parquet("data/processed/bidmc_batch.parquet")
df = df.sort_values(["recording_id", "time_seconds"]).reset_index(drop=True)

ids = list(np.random.default_rng(SEED).permutation(sorted(df.recording_id.unique())))

base_ids = ids[:4]
feedback_ids = ids[4:6]
val_ids = ids[6:8]
test_ids = ids[8:10]

target = "target_hr_10s"

BASE = [
    "heart_rate",
    "pulse_rate",
    "respiratory_rate",
    "spo2",
]
BASE += [f"{col}_missing" for col in BASE if f"{col}_missing" in df.columns]


def ridge_model(alpha=10):
    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
        Ridge(alpha=alpha),
    )


base_train = df[df.recording_id.isin(base_ids)].dropna(subset=[target])

base = ridge_model()
base.fit(base_train[BASE], base_train[target])


def prepare(ids_to_use):
    part = df[df.recording_id.isin(ids_to_use)].copy()
    part["baseline"] = base.predict(part[BASE])

    # Residual for the forecast issued at time t.
    part["error"] = part[target] - part["baseline"]

    group = part.groupby("recording_id", sort=False)

    # Error at t-10 becomes known at time t.
    for lag in [10, 12, 20, 30]:
        part[f"error_lag_{lag}"] = group["error"].shift(lag)

    # Rolling statistics contain only observable errors.
    part["energy_12"] = group["error"].transform(
        lambda s: s.shift(HORIZON).pow(2).rolling(12, min_periods=6).mean()
    )

    part["mean_error_12"] = group["error"].transform(
        lambda s: s.shift(HORIZON).rolling(12, min_periods=6).mean()
    )

    part["state"] = part["heart_rate"]

    # State is observable at decision time.
    # Use training-fitted centering, not future statistics.
    return part


feedback = prepare(feedback_ids)
validation = prepare(val_ids)
test = prepare(test_ids)

HISTORY = [
    "error_lag_10",
    "error_lag_12",
    "error_lag_20",
    "error_lag_30",
    "energy_12",
    "mean_error_12",
]

# Center relative to the training distribution.
state_center = feedback["state"].mean()

for frame in [feedback, validation, test]:
    frame["state_centered"] = frame["state"] - state_center

    for col in HISTORY:
        frame[f"state_x_{col}"] = frame["state_centered"] * frame[col]

INTERACTIONS = [f"state_x_{col}" for col in HISTORY]

CANDIDATES = {
    "baseline": [],
    "feedback": HISTORY,
    "feedback_interactions": HISTORY + INTERACTIONS,
}

controllers = {}

fit_frame = feedback.dropna(subset=[target] + HISTORY + INTERACTIONS)

for name, features in CANDIDATES.items():
    if name == "baseline":
        continue

    controller = ridge_model(alpha=20)

    controller.fit(
        fit_frame[features],
        fit_frame["error"],
    )

    controllers[name] = controller


def evaluate(frame, split):
    results = []

    # Same observations for every policy.
    valid = frame.dropna(subset=[target] + HISTORY + INTERACTIONS).copy()

    for name, features in CANDIDATES.items():
        if name == "baseline":
            correction = np.zeros(len(valid))
        else:
            correction = controllers[name].predict(valid[features])

        predictions = valid["baseline"].to_numpy() + correction

        errors = valid[target].to_numpy() - predictions

        for recording, indices in valid.groupby("recording_id").indices.items():
            e = errors[indices]
            results.append(
                {
                    "split": split,
                    "recording": recording,
                    "model": name,
                    "n": len(e),
                    "rmse": np.sqrt(np.mean(e**2)),
                    "mae": np.mean(np.abs(e)),
                    "bias": np.mean(e),
                }
            )

    return results


rows = evaluate(validation, "validation") + evaluate(test, "test")

results = pd.DataFrame(rows)
results.to_csv(
    OUT / "bidmc_feedback_controller.csv",
    index=False,
)

print("\nMEAN PER-RECORDING PERFORMANCE")
print(
    results.groupby(["split", "model"])[["rmse", "mae", "bias"]]
    .mean()
    .round(4)
    .to_string()
)

print("\nPER-RECORDING RMSE")
print(
    results.pivot_table(
        index=["split", "recording"],
        columns="model",
        values="rmse",
    )
    .round(4)
    .to_string()
)

print("\nSaved:", OUT / "bidmc_feedback_controller.csv")
