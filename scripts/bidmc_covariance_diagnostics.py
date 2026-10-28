"""Causal residual-covariance diagnostics for BIDMC telemetry."""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error

DATA = "data/processed/bidmc_batch.parquet"
HORIZON = 10
WINDOW = 30

df = pd.read_parquet(DATA)
df = df.sort_values(["recording_id", "time_seconds"]).reset_index(drop=True)

# Persistence benchmark: forecast HR(t+10) using HR(t).
df["reference_prediction"] = df["heart_rate"]
df["reference_error"] = df["target_hr_10s"] - df["reference_prediction"]

groups = df.groupby("recording_id", sort=False)

# A forecast error originating at time s is not available
# until s + HORIZON. Shift by HORIZON accordingly.
df["known_error"] = groups["reference_error"].shift(HORIZON)

groups = df.groupby("recording_id", sort=False)

for lag in [1, 2, 5, 10]:
    df[f"known_error_lag_{lag}"] = groups["known_error"].shift(lag)

# Historical error energy.
df["error_energy_30"] = groups["known_error"].transform(
    lambda x: x.pow(2).rolling(WINDOW, min_periods=WINDOW).mean()
)

# Rolling covariance between already observable errors.
for lag in [1, 2, 5, 10]:
    df[f"error_cov_{lag}"] = groups["known_error"].transform(
        lambda x, k=lag: x.rolling(WINDOW, min_periods=WINDOW).cov(x.shift(k))
    )

# Nonlinear interactions between historical errors.
df["error_interaction_1_5"] = df["known_error_lag_1"] * df["known_error_lag_5"]
df["error_interaction_2_10"] = df["known_error_lag_2"] * df["known_error_lag_10"]

# Compare nested information sets.
feature_sets = {
    "energy_only": ["error_energy_30"],
    "energy_covariance": [
        "error_energy_30",
        "error_cov_1",
        "error_cov_2",
        "error_cov_5",
        "error_cov_10",
    ],
    "energy_interactions": [
        "error_energy_30",
        "error_interaction_1_5",
        "error_interaction_2_10",
    ],
}

feature_sets["combined"] = sorted(
    set(feature_sets["energy_covariance"] + feature_sets["energy_interactions"])
)

# Held-out recording comparison.
ids = sorted(df["recording_id"].unique())
rng = np.random.default_rng(12345)
ids = list(rng.permutation(ids))

train_ids = ids[:6]
test_ids = ids[8:10]

target = "reference_error"
all_features = sorted(set(sum(feature_sets.values(), [])))

# Compare models on exactly the same eligible rows.
valid = df.dropna(subset=[target] + all_features).copy()

train = valid[valid.recording_id.isin(train_ids)]
test = valid[valid.recording_id.isin(test_ids)]

rows = []
for name, features in feature_sets.items():
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        Ridge(alpha=20.0),
    )
    model.fit(train[features], train[target])
    pred = model.predict(test[features])

    rows.append(
        {
            "model": name,
            "test_rmse": np.sqrt(mean_squared_error(test[target], pred)),
            "features": len(features),
        }
    )

results = pd.DataFrame(rows).sort_values("test_rmse")
print("\nCOVARIANCE AND INTERACTION DIAGNOSTICS")
print(results.to_string(index=False))

print("\nReference zero-correction RMSE:")
print(np.sqrt(np.mean(test[target].to_numpy() ** 2)))

print("\nHeld-out recordings:", test_ids)

results.to_csv(
    "results/bidmc_covariance_diagnostics.csv",
    index=False,
)
