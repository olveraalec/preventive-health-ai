"""Phase A.14: Nested patient-grouped controller evaluation.

Outer: leave-one-recording-out.
Inner: three group-disjoint folds, used for policy selection.
No outer-patient outcomes influence selection or training.
"""

from pathlib import Path
import numpy as np
import pandas as pd

from bidmc_psd_covariance import (
    TARGET,
    DATA,
    BASE,
    MEMORY,
    STATE,
    PSDGate,
    crossfit,
    model,
)
from bidmc_hybrid_controller import sequential_weights
from healthcare_ai_lab.evaluation.registry import evaluate

OUT = Path("results/bidmc_nested_validation.csv")
SELECTION_OUT = Path("results/bidmc_nested_selection.csv")

BANDWIDTH = 0.5
RHO = 0.25

CANDIDATES = (
    "equal_blend",
    "covariance",
    "stable",
)


def features_for(df):
    flags = [f"{c}_missing" for c in BASE if f"{c}_missing" in df.columns]
    current = BASE + flags
    return {
        "one": current + MEMORY[:1],
        "four": current + MEMORY,
    }


def fit_system(train, features):
    oof = crossfit(train, features)
    gate = PSDGate().fit(oof)

    experts = {}
    for name, columns in features.items():
        expert = model()
        expert.fit(train[columns], train[TARGET])
        experts[name] = expert

    errors = oof[TARGET].to_numpy() - oof["four"].to_numpy()
    sigma = max(float(np.sqrt(np.mean(errors**2))), 1e-6)

    return gate, experts, sigma


def predict_all(system, patient, features):
    gate, experts, sigma = system
    patient = patient.sort_values("time_seconds")

    p1 = experts["one"].predict(patient[features["one"]])
    p4 = experts["four"].predict(patient[features["four"]])

    local = gate.local_moments(patient, BANDWIDTH)
    moments = (1 - RHO) * local + RHO * gate.global_moments[None, :]

    weights = {
        "equal_blend": np.full(len(patient), 0.5),
        "covariance": sequential_weights(moments, 0.0, 0.0, 0.0),
        "stable": sequential_weights(moments, 0.5, 0.5, 0.0),
    }

    return {
        name: (
            p1 + w * (p4 - p1),
            w,
            sigma,
        )
        for name, w in weights.items()
    }


def score(patient, predictions):
    y = patient[TARGET].to_numpy()
    rows = {}

    for name, (pred, w, sigma) in predictions.items():
        metrics = evaluate(y, pred, sigma=np.full(len(y), sigma))

        metrics.update(
            {
                "mean_weight": float(w.mean()),
                "weight_std": float(w.std()),
                "weight_variation": (
                    float(np.abs(np.diff(w)).mean()) if len(w) > 1 else 0.0
                ),
            }
        )
        rows[name] = metrics

    return rows


def main():
    df = pd.read_parquet(DATA)
    df = df.sort_values(["recording_id", "time_seconds"]).dropna(subset=[TARGET]).copy()

    features = features_for(df)
    required = set(
        [TARGET, "recording_id", "time_seconds"]
        + STATE
        + [c for cols in features.values() for c in cols]
    )
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    ids = sorted(df.recording_id.unique())
    if len(ids) != 10:
        raise ValueError("Expected 10 patient recordings")

    test_rows = []
    selection_rows = []

    for outer_id in ids:
        remaining = [pid for pid in ids if pid != outer_id]

        # Deterministic three-fold patient grouping.
        inner_folds = np.array_split(np.array(remaining), 3)

        inner_scores = {name: [] for name in CANDIDATES}

        for fold, validation_ids in enumerate(inner_folds):
            validation_ids = list(validation_ids)
            training_ids = [pid for pid in remaining if pid not in validation_ids]

            inner_train = df[df.recording_id.isin(training_ids)]
            inner_val = df[df.recording_id.isin(validation_ids)]

            system = fit_system(inner_train, features)

            for pid, patient in inner_val.groupby("recording_id"):
                predictions = predict_all(system, patient, features)
                scores = score(patient, predictions)

                for name in CANDIDATES:
                    inner_scores[name].append(scores[name]["rmse"])

        # Each inner validation patient has equal weight.
        means = {name: float(np.mean(values)) for name, values in inner_scores.items()}
        selected = min(CANDIDATES, key=lambda name: means[name])

        selection_rows.append(
            {
                "outer_recording_id": str(outer_id),
                "selected_policy": selected,
                **{f"inner_rmse_{name}": value for name, value in means.items()},
            }
        )

        outer_train = df[df.recording_id.isin(remaining)]
        outer_patient = df[df.recording_id == outer_id]

        system = fit_system(outer_train, features)
        predictions = predict_all(system, outer_patient, features)
        metrics = score(outer_patient, predictions)

        for name in CANDIDATES:
            test_rows.append(
                {
                    "recording_id": str(outer_id),
                    "policy": name,
                    "selected_policy": selected,
                    "selected": name == selected,
                    **metrics[name],
                }
            )

        print(
            f"{outer_id}: selected={selected}, "
            f"inner_RMSE={means[selected]:.4f}, "
            f"outer_RMSE={metrics[selected]['rmse']:.4f}"
        )

    result = pd.DataFrame(test_rows)
    selection = pd.DataFrame(selection_rows)

    # This is the actual nested-selected evaluation.
    chosen = result[result["selected"]].copy()
    chosen["policy"] = "nested_selected"

    comparison = pd.concat([result, chosen], ignore_index=True)

    metric_columns = [
        "rmse",
        "mae",
        "bias",
        "p90_absolute_error",
        "nll",
        "crps",
        "coverage_90",
        "width_90",
        "mean_weight",
        "weight_std",
        "weight_variation",
    ]

    print("\nOUTER-PATIENT FULL METRIC REGISTRY")
    print(
        comparison.groupby("policy")[metric_columns]
        .agg(["mean", "std"])
        .round(5)
        .to_string()
    )

    print("\nNESTED SELECTION COUNTS")
    print(selection["selected_policy"].value_counts().to_string())

    print("\nOUTER-PATIENT PAIRED RMSE")
    paired = comparison.pivot(
        index="recording_id",
        columns="policy",
        values="rmse",
    )

    for name in ["covariance", "stable", "nested_selected"]:
        gain = paired["equal_blend"] - paired[name]
        print(
            f"{name}: mean={gain.mean():.5f}, "
            f"median={gain.median():.5f}, "
            f"win={(gain > 0).mean():.3f}, "
            f"worst={gain.min():.5f}"
        )

    print("\nOUTER-PATIENT RMSE TABLE")
    print(paired.round(4).to_string())

    OUT.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(OUT, index=False)
    selection.to_csv(SELECTION_OUT, index=False)

    print("\nSaved:", OUT)
    print("Saved:", SELECTION_OUT)


if __name__ == "__main__":
    main()
