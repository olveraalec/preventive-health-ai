"""Phase A.5: Continuous state-dependent memory gate.

Exploratory repeated patient-grouped evaluation.
Gate trained on out-of-fold predictions from training patients.
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
OUTPUT = Path("results/bidmc_dynamic_gate.csv")
TARGET = "target_hr_10s"
STATE = "heart_rate_std_15"
RUNS = 10

BASE = ["heart_rate", "pulse_rate", "respiratory_rate", "spo2"]
MEMORY = [
    "pulse_rate_mean_30",
    "heart_rate_std_15",
    "pulse_rate_std_30",
    "pulse_rate_missing_fraction_15",
]


def model():
    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
        Ridge(alpha=10.0),
    )


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def fit_experts(train, columns):
    m = model()
    m.fit(train[columns], train[TARGET])
    return m


def oof_predictions(train, policies):
    """Leave-one-patient-out expert predictions."""
    result = train[["recording_id", TARGET, STATE]].copy()

    for name in policies:
        result[name] = np.nan

    for pid in train.recording_id.unique():
        inner = train[train.recording_id != pid]
        held = train[train.recording_id == pid]

        for name, cols in policies.items():
            expert = fit_experts(inner, cols)
            result.loc[held.index, name] = expert.predict(held[cols])

    return result


def state_values(series, center, scale):
    values = series.fillna(center).to_numpy(dtype=float)
    return (values - center) / scale


def choose_gate(oof, center, scale):
    """Training-only gate selection, including memory penalty."""
    z = state_values(oof[STATE], center, scale)
    y = oof[TARGET].to_numpy()
    p1 = oof["one"].to_numpy()
    p4 = oof["four"].to_numpy()

    best = None

    # Predeclared, small gate-complexity search.
    for a in [-2, -1, 0, 1, 2]:
        for b in [-2, -1, -0.5, 0, 0.5, 1, 2]:
            weights = sigmoid(a + b * z)
            prediction = p1 + weights * (p4 - p1)

            # Relative extra-memory cost in addition to prediction MSE.
            objective = np.mean((y - prediction) ** 2) + 0.01 * np.mean(weights)

            # Mild complexity penalty for state sensitivity.
            objective += 0.002 * b**2

            if best is None or objective < best[0]:
                best = (objective, a, b)

    return best[1], best[2]


def main():
    df = pd.read_parquet(DATA)
    df = df.sort_values(["recording_id", "time_seconds"]).dropna(subset=[TARGET]).copy()

    flags = [f"{c}_missing" for c in BASE if f"{c}_missing" in df.columns]
    current = BASE + flags

    policies = {
        "one": current + MEMORY[:1],
        "four": current + MEMORY,
    }

    required = set(current + MEMORY + [TARGET, STATE])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    ids = np.array(sorted(df.recording_id.unique()))
    if len(ids) != 10:
        raise ValueError("Expected 10 recordings.")

    rng = np.random.default_rng(12345)
    rows = []
    gate_rows = []

    for run in range(RUNS):
        shuffled = rng.permutation(ids)
        train = df[df.recording_id.isin(shuffled[:6])]

        # The other two recordings remain outside this experiment.
        validation = df[df.recording_id.isin(shuffled[6:8])].copy()

        center = float(train[STATE].median())
        scale = max(float(train[STATE].std()), 1e-6)

        oof = oof_predictions(train, policies)
        a, b = choose_gate(oof, center, scale)

        experts = {name: fit_experts(train, cols) for name, cols in policies.items()}

        z_train = state_values(oof[STATE], center, scale)
        w_train = sigmoid(a + b * z_train)

        y_train = oof[TARGET].to_numpy()
        p1_train = oof["one"].to_numpy()
        p4_train = oof["four"].to_numpy()

        train_predictions = {
            "fixed_one": p1_train,
            "fixed_four": p4_train,
            "equal_blend": 0.5 * (p1_train + p4_train),
            "dynamic_gate": (p1_train + w_train * (p4_train - p1_train)),
        }

        sigmas = {
            name: max(
                float(np.sqrt(np.mean((y_train - pred) ** 2))),
                1e-6,
            )
            for name, pred in train_predictions.items()
        }

        gate_rows.append(
            {
                "run": run,
                "gate_intercept": a,
                "gate_state_coefficient": b,
                "mean_train_weight": w_train.mean(),
            }
        )

        for pid, patient in validation.groupby("recording_id"):
            y = patient[TARGET].to_numpy()
            p1 = experts["one"].predict(patient[policies["one"]])
            p4 = experts["four"].predict(patient[policies["four"]])

            z = state_values(patient[STATE], center, scale)
            w = sigmoid(a + b * z)

            predictions = {
                "fixed_one": p1,
                "fixed_four": p4,
                "equal_blend": 0.5 * (p1 + p4),
                "dynamic_gate": p1 + w * (p4 - p1),
            }

            for name, pred in predictions.items():
                scores = evaluate(
                    y,
                    pred,
                    sigma=np.full(len(y), sigmas[name]),
                )

                rows.append(
                    {
                        "run": run,
                        "recording_id": str(pid),
                        "policy": name,
                        "memory_weight": (
                            float(w.mean())
                            if name == "dynamic_gate"
                            else {
                                "fixed_one": 0.0,
                                "fixed_four": 1.0,
                                "equal_blend": 0.5,
                            }[name]
                        ),
                        "weight_variation": (
                            float(np.mean(np.abs(np.diff(w))))
                            if name == "dynamic_gate" and len(w) > 1
                            else 0.0
                        ),
                        **scores,
                    }
                )

    results = pd.DataFrame(rows)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT, index=False)

    metrics = [
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
    metrics = [m for m in metrics if m in results]

    print("\nFULL CONTROLLER METRIC SUITE")
    print(results.groupby("policy")[metrics].agg(["mean", "std"]).round(4).to_string())

    paired = results.pivot(
        index=["run", "recording_id"],
        columns="policy",
        values="rmse",
    )

    print("\nPAIRED RMSE IMPROVEMENT VS FIXED FOUR")
    for name in ["fixed_one", "equal_blend", "dynamic_gate"]:
        gains = paired["fixed_four"] - paired[name]
        print(
            f"{name:16s} "
            f"mean={gains.mean():+.4f} "
            f"median={gains.median():+.4f} "
            f"win_fraction={(gains > 0).mean():.3f}"
        )

    print("\nSELECTED GATE PARAMETERS")
    print(pd.DataFrame(gate_rows).to_string(index=False))

    print("\nSaved:", OUTPUT)


if __name__ == "__main__":
    main()
