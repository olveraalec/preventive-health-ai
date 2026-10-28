"""BIDMC residual-dependence diagnostics.

Exploratory analysis: not a validated memory controller.
Uses recording-held-out predictions and never joins
residual histories across recording boundaries.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mutual_info_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 12345
HORIZON = 10
LAGS = [1, 2, 5, 10, 12, 20, 30]
OUT = Path("results")
OUT.mkdir(exist_ok=True)

df = pd.read_parquet("data/processed/bidmc_batch.parquet")
df = df.sort_values(["recording_id", "time_seconds"]).reset_index(drop=True)

ids = list(np.random.default_rng(SEED).permutation(sorted(df["recording_id"].unique())))
assert len(ids) == 10

train_ids = ids[:6]
val_ids = ids[6:8]
test_ids = ids[8:10]

current = [
    "heart_rate",
    "pulse_rate",
    "respiratory_rate",
    "spo2",
]

current += [f"{c}_missing" for c in current if f"{c}_missing" in df.columns]

data = df.dropna(subset=["target_hr_10s"]).copy()

train = data[data["recording_id"].isin(train_ids)]

model = make_pipeline(
    SimpleImputer(strategy="median", add_indicator=True),
    StandardScaler(),
    Ridge(alpha=10),
)
model.fit(train[current], train["target_hr_10s"])


def create_residuals(recording_ids):
    subset = data[data["recording_id"].isin(recording_ids)].copy()

    subset["prediction"] = model.predict(subset[current])
    subset["error"] = subset["target_hr_10s"] - subset["prediction"]
    subset["error_sq"] = subset["error"] ** 2

    group = subset.groupby("recording_id", sort=False)

    for lag in LAGS:
        subset[f"error_lag_{lag}"] = group["error"].shift(lag)
        subset[f"sq_lag_{lag}"] = group["error_sq"].shift(lag)

    # Rolling information known at the decision time.
    # A prediction at t-10 only becomes observable at t.
    subset["past_error_energy"] = group["error_sq"].transform(
        lambda x: x.shift(HORIZON).rolling(12, min_periods=6).mean()
    )

    return subset


def covariance_table(subset, name):
    rows = []

    for lag in LAGS:
        valid = subset[
            ["error", "error_sq", f"error_lag_{lag}", f"sq_lag_{lag}"]
        ].dropna()

        if len(valid) < 3:
            continue

        rows.append(
            {
                "split": name,
                "lag": lag,
                "signed_cov": valid["error"].cov(valid[f"error_lag_{lag}"]),
                "signed_corr": valid["error"].corr(valid[f"error_lag_{lag}"]),
                "squared_cov": valid["error_sq"].cov(valid[f"sq_lag_{lag}"]),
                "squared_corr": valid["error_sq"].corr(valid[f"sq_lag_{lag}"]),
            }
        )

    return pd.DataFrame(rows)


def discretize(series, bins=3):
    """Quantile bins, retaining distinct values safely."""
    values = pd.to_numeric(series, errors="coerce")
    ranks = values.rank(method="average", pct=True)
    return np.minimum((ranks * bins).fillna(0).astype(int), bins - 1)


def conditional_mi(x, y, z):
    """Discrete plug-in estimate of I(X;Y | Z), in nats."""
    table = pd.DataFrame(
        {
            "x": np.asarray(x),
            "y": np.asarray(y),
            "z": np.asarray(z),
        }
    )

    total = len(table)
    score = 0.0

    for _, part in table.groupby("z"):
        score += len(part) / total * mutual_info_score(part["x"], part["y"])
    return float(score)


def information_analysis(subset, name):
    # The lag must be at least the forecasting horizon:
    # otherwise previous errors aren't observable yet.
    required = [
        "error_sq",
        "heart_rate",
        "past_error_energy",
        "error_lag_12",
        "spo2_missing",
    ]
    valid = subset[required].dropna().copy()

    if len(valid) < 30:
        return {"split": name, "observations": len(valid)}

    target = discretize(valid["error_sq"])
    state = discretize(valid["heart_rate"])
    energy = discretize(valid["past_error_energy"])
    past = discretize(valid["error_lag_12"])
    missing = valid["spo2_missing"].astype(int)

    # Jointly encode the past error and missingness.
    # This does not prove a synergistic interaction;
    # it tests joint conditional dependence.
    joint = past * 2 + missing

    result = {
        "split": name,
        "observations": len(valid),
        "mi_past_energy": mutual_info_score(target, energy),
        "cmi_energy_given_state": conditional_mi(target, energy, state),
        "cmi_past_error_given_state": conditional_mi(target, past, state),
        "cmi_joint_given_state": conditional_mi(target, joint, state),
    }

    result["joint_minus_past_cmi"] = (
        result["cmi_joint_given_state"] - result["cmi_past_error_given_state"]
    )

    return result


cov_results = []
info_results = []

for name, recording_ids in [
    ("validation", val_ids),
    ("test", test_ids),
]:
    subset = create_residuals(recording_ids)

    cov_results.append(covariance_table(subset, name))
    info_results.append(information_analysis(subset, name))

    # Per-recording diagnostic avoids pooled recording effects.
    print(f"\n{name.upper()} PER-RECORDING RESIDUALS")
    print(
        subset.groupby("recording_id")["error"]
        .agg(["count", "mean", "std"])
        .round(4)
        .to_string()
    )

cov_results = pd.concat(cov_results, ignore_index=True)
info_results = pd.DataFrame(info_results)

cov_results.to_csv(OUT / "bidmc_residual_covariance.csv", index=False)
info_results.to_csv(OUT / "bidmc_residual_information.csv", index=False)

print("\nLAGGED ERROR COVARIANCE")
print(cov_results.round(5).to_string(index=False))

print("\nINFORMATION DIAGNOSTICS (NATS)")
print(info_results.round(5).to_string(index=False))

print("\nFiles saved in results/")