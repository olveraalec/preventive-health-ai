"""Phase A.1: Information value versus information volume.

Patient-level training, validation, and held-out testing.
Greedy feature selection uses training recordings only.
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

DATA = Path("data/processed/bidmc_batch.parquet")
OUTPUT = Path("results/bidmc_information_value.csv")

TARGET = "target_hr_10s"
SEED = 12345

BASE = [
    "heart_rate",
    "pulse_rate",
    "respiratory_rate",
    "spo2",
]

BUDGETS = [0, 1, 2, 4, 8, 16]


def model():
    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
        Ridge(alpha=10.0),
    )


def score(y, prediction):
    error = np.asarray(y) - np.asarray(prediction)

    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
        "p90_absolute_error": float(np.quantile(np.abs(error), 0.9)),
    }


def train_cv_rmse(data, features):
    """Leave one training recording out at a time."""
    scores = []

    for patient in data.recording_id.unique():
        train = data[data.recording_id != patient]
        held = data[data.recording_id == patient]

        estimator = model()
        estimator.fit(train[features], train[TARGET])

        prediction = estimator.predict(held[features])

        scores.append(mean_squared_error(held[TARGET], prediction) ** 0.5)

    return float(np.mean(scores))


def greedy_select(data, base, candidates, max_features):
    """Select features by incremental patient-level CV gain."""
    selected = []
    remaining = candidates.copy()
    history = []

    previous = train_cv_rmse(data, base)

    for step in range(min(max_features, len(remaining))):
        trials = []

        for feature in remaining:
            value = train_cv_rmse(data, base + selected + [feature])
            trials.append((value, feature))

        best_score, best_feature = min(trials)

        selected.append(best_feature)
        remaining.remove(best_feature)

        history.append(
            {
                "step": step + 1,
                "feature": best_feature,
                "cv_rmse": best_score,
                "marginal_gain": previous - best_score,
            }
        )

        previous = best_score

    return selected, pd.DataFrame(history)


def main():
    df = pd.read_parquet(DATA)

    df = df.sort_values(["recording_id", "time_seconds"]).dropna(subset=[TARGET]).copy()

    missing_flags = [f"{col}_missing" for col in BASE if f"{col}_missing" in df.columns]

    base = BASE + missing_flags

    # Only use historical telemetry summaries.
    # Exclude outcome-derived feedback until its timing is
    # separately audited for this experiment.
    pattern = re.compile(
        r"(_(?:mean|std|min|max|median|slope|"
        r"lag|diff|delta|trend|ema|missing_fraction)_"
        r"|_lag\d+$)"
    )

    history_cols = [
        col
        for col in df.select_dtypes(include="number").columns
        if col not in base
        and col != TARGET
        and not col.startswith(("target_", "true_", "future_", "error_"))
        and pattern.search(col)
    ]

    if not history_cols:
        raise ValueError(
            "No historical features identified. Inspect the processed column names."
        )

    ids = np.array(sorted(df.recording_id.unique()))

    if len(ids) < 10:
        raise ValueError("Expected at least 10 independent recordings.")

    rng = np.random.default_rng(SEED)
    ids = rng.permutation(ids)

    train_ids = ids[:6]
    validation_ids = ids[6:8]
    test_ids = ids[8:10]

    train = df[df.recording_id.isin(train_ids)]
    validation = df[df.recording_id.isin(validation_ids)]
    test = df[df.recording_id.isin(test_ids)]

    print("\nPATIENT SPLITS")
    print("Train:", train_ids.tolist())
    print("Validation:", validation_ids.tolist())
    print("Test:", test_ids.tolist())

    print("\nHISTORICAL CANDIDATES")
    print(history_cols)

    selected, selection_history = greedy_select(
        train,
        base,
        history_cols,
        max_features=min(16, len(history_cols)),
    )

    print("\nGREEDY SELECTION")
    print(selection_history.round(5).to_string(index=False))

    # Compare nested targeted-memory budgets.
    policies = {
        f"targeted_{budget}": (base + selected[:budget])
        for budget in BUDGETS
        if budget <= len(selected)
    }

    policies["full_history"] = base + history_cols

    # Recency baseline using shortest historical windows.
    recent = sorted(
        history_cols,
        key=lambda col: (
            int(re.findall(r"\d+", col)[-1]) if re.findall(r"\d+", col) else 9999,
            col,
        ),
    )[:4]

    policies["recent_4"] = base + recent

    rows = []

    for name, columns in policies.items():
        estimator = model()
        estimator.fit(train[columns], train[TARGET])

        for split_name, part in [
            ("validation", validation),
            ("test", test),
        ]:
            predictions = estimator.predict(part[columns])

            for recording_id, patient in part.groupby("recording_id"):
                positions = part.index.get_indexer(patient.index)

                metrics = score(
                    patient[TARGET],
                    predictions[positions],
                )

                rows.append(
                    {
                        "policy": name,
                        "split": split_name,
                        "recording_id": recording_id,
                        "memory_budget": len(columns) - len(base),
                        **metrics,
                    }
                )

    results = pd.DataFrame(rows)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT, index=False)

    summary = (
        results.groupby(["split", "policy", "memory_budget"])[
            ["rmse", "mae", "bias", "p90_absolute_error"]
        ]
        .mean()
        .round(4)
    )

    print("\nPATIENT-AVERAGED RESULTS")
    print(summary.to_string())

    print("\nSELECTED MEMORY FEATURES")
    print(selected)

    print("\nRECENT BASELINE FEATURES")
    print(recent)

    print("\nSaved:", OUTPUT)


if __name__ == "__main__":
    main()
