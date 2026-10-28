"""Recording-held-out BIDMC forecasting benchmark."""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DATA = Path("data/processed/bidmc_batch.parquet")
OUTPUT = Path("results/bidmc_baselines.csv")

TARGET = "target_hr_10s"
ID = "recording_id"
SEED = 12345

df = pd.read_parquet(DATA)
df = df.sort_values([ID, "time_seconds"]).reset_index(drop=True)

# Confirm feature timestamps are correctly ordered.
assert not df.duplicated([ID, "time_seconds"]).any()
assert df.groupby(ID)["time_seconds"].apply(lambda x: x.is_monotonic_increasing).all()

# The future target must belong to the same recording.
expected = df.groupby(ID)["heart_rate"].shift(-10)
np.testing.assert_allclose(
    df[TARGET].to_numpy(dtype=float),
    expected.to_numpy(dtype=float),
    equal_nan=True,
)

# Fixed recording-level split.
ids = sorted(df[ID].unique())
rng = np.random.default_rng(SEED)
ids = list(rng.permutation(ids))

train_ids = ids[:6]
val_ids = ids[6:8]
test_ids = ids[8:10]

assert len(ids) == 10
assert not (set(train_ids) & set(val_ids))
assert not (set(train_ids) & set(test_ids))
assert not (set(val_ids) & set(test_ids))

data = df.dropna(subset=[TARGET]).copy()

train = data[data[ID].isin(train_ids)]
val = data[data[ID].isin(val_ids)]
test = data[data[ID].isin(test_ids)]

current = [
    "heart_rate",
    "pulse_rate",
    "respiratory_rate",
    "spo2",
]

missing_flags = [
    f"{col}_missing" for col in current if f"{col}_missing" in data.columns
]

current = current + missing_flags

# Historical rolling features and lagged telemetry.
history = [
    c
    for c in data.columns
    if c not in current
    and ("_mean_" in c or "_std_" in c or "_lag_" in c or "_missing_fraction_" in c)
]

# Exclude all explicit identifiers, targets and future fields.
excluded = {
    ID,
    TARGET,
    "time_seconds",
    "sampling_interval_seconds",
    "missing_episode_id",
}
excluded.update(c for c in data.columns if "target" in c or "future" in c)

history = [
    c for c in history if c not in excluded and pd.api.types.is_numeric_dtype(data[c])
]

full = [
    c
    for c in data.select_dtypes(include="number").columns
    if c not in excluded and not c.startswith("missing_episode")
]

feature_sets = {
    "current": current,
    "history": list(dict.fromkeys(current + history)),
    "full": full,
}


def evaluate(y, predictions):
    residual = np.asarray(y) - np.asarray(predictions)
    return {
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "bias": float(np.mean(residual)),
    }


rows = []
fitted = {}

for name, cols in feature_sets.items():
    model = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
        Ridge(alpha=10.0),
    )

    model.fit(train[cols], train[TARGET])
    fitted[name] = model

    for split_name, subset in [
        ("validation", val),
        ("test", test),
    ]:
        predictions = model.predict(subset[cols])

        for recording, indices in subset.groupby(ID).indices.items():
            metrics = evaluate(
                subset[TARGET].to_numpy()[indices],
                predictions[indices],
            )
            rows.append(
                {
                    "model": name,
                    "split": split_name,
                    "recording_id": recording,
                    **metrics,
                }
            )

# Persistence baseline: future HR equals current HR.
for split_name, subset in [("validation", val), ("test", test)]:
    for recording, group in subset.groupby(ID):
        metrics = evaluate(
            group[TARGET],
            group["heart_rate"],
        )
        rows.append(
            {
                "model": "persistence",
                "split": split_name,
                "recording_id": recording,
                **metrics,
            }
        )

results = pd.DataFrame(rows)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
results.to_csv(OUTPUT, index=False)

print("\nRECORDING SPLIT")
print("Train:", train_ids)
print("Validation:", val_ids)
print("Test:", test_ids)

print("\nFEATURE COUNTS")
for name, cols in feature_sets.items():
    print(name, len(cols))

print("\nMEAN PER-RECORDING METRICS")
summary = results.groupby(["split", "model"])[["rmse", "mae", "bias"]].mean()
print(summary.round(4).to_string())

print("\nSaved:", OUTPUT)
