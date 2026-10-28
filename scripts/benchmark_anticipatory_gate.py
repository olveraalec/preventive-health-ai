"""Out-of-sample anticipatory versus reactive covariance gating.

Controlled synthetic experiment, NOT clinical validation.

The pre-decision cue is artificially correlated with the regime.
The true regime and current regret are NEVER supplied to the gate.
Counterfactual outcomes are available to the reactive gate only AFTER
each decision, an assumption requiring separate justification in reality.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

SEED = 12345
N_SEQUENCES = 100
SEQUENCE_LENGTH = 120
TRAIN_CUES_PER_EPISODE = 3

BETA = 0.10
TEMPERATURE = 0.10
THRESHOLD = 0.02
INITIAL_ADVANTAGE = -0.20
KAPPA = 1.0
CUE_NOISE = 0.9

SCENARIOS = ("aligned", "uninformative", "misleading")
CUE_CENTERS = {
    "aligned": 1.0,
    "uninformative": 0.0,
    "misleading": -1.0,
}

SOURCE = Path("results/information_alignment.csv")
OUT = Path("results/anticipatory_gate.csv")
SUMMARY_OUT = Path("results/anticipatory_gate_sequences.csv")


def sigmoid(value):
    return float(1.0 / (1.0 + np.exp(-np.clip(value, -30, 30))))


def load_data():
    raw = pd.read_csv(SOURCE)

    paired = raw.pivot(
        index=["scenario", "recording_id"],
        columns="policy",
        values="regret",
    ).reset_index()

    if paired[["static_equal", "continuous"]].isna().any().any():
        raise ValueError("Incomplete paired regret observations")

    # Preserve episode identity across all three scenarios.
    ids = sorted(paired["recording_id"].unique())
    rng = np.random.default_rng(SEED)
    ids = rng.permutation(ids)

    cutoff = len(ids) // 2
    training = paired[paired.recording_id.isin(ids[:cutoff])]
    testing = paired[paired.recording_id.isin(ids[cutoff:])]

    assert set(training.recording_id).isdisjoint(set(testing.recording_id))
    return training.copy(), testing.copy()


def train_gate(training):
    """Fit an anticipatory gain estimator from synthetic context.

    Bootstrap at the independent episode-ID level, not the row level.
    """
    rng = np.random.default_rng(SEED)
    rows = []

    for row in training.itertuples(index=False):
        advantage = float(row.static_equal - row.continuous)

        for _ in range(TRAIN_CUES_PER_EPISODE):
            cue = CUE_CENTERS[row.scenario] + rng.normal(0, CUE_NOISE)
            rows.append(
                {
                    "recording_id": row.recording_id,
                    "cue": cue,
                    "advantage": advantage,
                }
            )

    data = pd.DataFrame(rows)

    # Fit an ensemble to approximate parameter uncertainty.
    # This is not a calibrated predictive interval.
    ids = data.recording_id.unique()
    models = []

    for _ in range(40):
        sampled = rng.choice(ids, size=len(ids), replace=True)
        counts = pd.Series(sampled).value_counts()
        weights = data.recording_id.map(counts).fillna(0).to_numpy()

        model = Ridge(alpha=10.0)
        model.fit(
            data[["cue"]].to_numpy(),
            data["advantage"].to_numpy(),
            sample_weight=weights,
        )
        models.append(model)

    return models


def predict_gate(models, cue):
    x = np.array([[cue]])
    estimates = np.array([m.predict(x)[0] for m in models])

    mean = float(estimates.mean())
    uncertainty = float(estimates.std(ddof=1))

    lower_bound = mean - KAPPA * uncertainty
    q = sigmoid((lower_bound - THRESHOLD) / TEMPERATURE)

    return q, mean, uncertainty


def generate_sequence(pool, rng):
    """Random transitions; no repeating deterministic regime order."""
    lookup = {
        scenario: pool[pool.scenario == scenario].reset_index(drop=True)
        for scenario in SCENARIOS
    }

    rows = []
    current = rng.choice(SCENARIOS)
    remaining = 0

    for t in range(SEQUENCE_LENGTH):
        if remaining <= 0:
            alternatives = [s for s in SCENARIOS if s != current]
            current = rng.choice(alternatives)
            remaining = int(rng.integers(10, 36))

        part = lookup[current]
        observation = part.iloc[int(rng.integers(0, len(part)))]

        # Cue is available before the decision and is noisy.
        cue = CUE_CENTERS[current] + rng.normal(0, CUE_NOISE)

        rows.append(
            {
                "time": t,
                "scenario": current,  # Evaluation only; hidden from gate.
                "cue": float(cue),
                "static": float(observation.static_equal),
                "continuous": float(observation.continuous),
                "transition": remaining in range(30, 35),
            }
        )
        remaining -= 1

    return rows


def evaluate_sequence(sequence_id, sequence, models, rng):
    history_estimate = INITIAL_ADVANTAGE
    rows = []

    for observation in sequence:
        static = observation["static"]
        continuous = observation["continuous"]

        # Decisions occur BEFORE observing the current advantage.
        reactive_q = sigmoid((history_estimate - THRESHOLD) / TEMPERATURE)

        anticipatory_q, predicted_gain, uncertainty = predict_gate(
            models, observation["cue"]
        )

        reactive_choice = rng.random() < reactive_q
        anticipatory_choice = rng.random() < anticipatory_q

        reactive_regret = continuous if reactive_choice else static
        anticipatory_regret = continuous if anticipatory_choice else static

        rows.append(
            {
                "sequence_id": sequence_id,
                "time": observation["time"],
                "scenario": observation["scenario"],
                "cue": observation["cue"],
                "static_regret": static,
                "continuous_regret": continuous,
                "reactive_regret": reactive_regret,
                "anticipatory_regret": anticipatory_regret,
                "reactive_q": reactive_q,
                "anticipatory_q": anticipatory_q,
                "predicted_gain": predicted_gain,
                "parameter_uncertainty": uncertainty,
                "actual_gain": static - continuous,
            }
        )

        # Delayed update: current outcomes influence future steps only.
        actual_advantage = static - continuous
        history_estimate = (1 - BETA) * history_estimate + BETA * actual_advantage

    return rows


def main():
    training, testing = load_data()
    models = train_gate(training)

    rng = np.random.default_rng(SEED + 1000)
    rows = []

    for sequence_id in range(N_SEQUENCES):
        sequence = generate_sequence(testing, rng)
        rows.extend(evaluate_sequence(sequence_id, sequence, models, rng))

    results = pd.DataFrame(rows)

    policies = [
        "static_regret",
        "continuous_regret",
        "reactive_regret",
        "anticipatory_regret",
    ]

    print("\nOVERALL REGRET (LOWER IS BETTER)")
    print(results[policies].agg(["mean", "std"]).round(4).to_string())

    print("\nPERFORMANCE BY HIDDEN REGIME (ANALYSIS ONLY)")
    print(
        results.groupby("scenario")[policies + ["reactive_q", "anticipatory_q"]]
        .mean()
        .round(4)
        .to_string()
    )

    print("\nPREDICTED VS REALIZED GAIN")
    print("Correlation:", round(results.predicted_gain.corr(results.actual_gain), 4))
    print(
        "Prediction MAE:",
        round(np.mean(np.abs(results.predicted_gain - results.actual_gain)), 4),
    )

    # Independent sequences are the resampling/evaluation units.
    sequence_scores = results.groupby("sequence_id")[policies].mean()

    print("\nPAIRED SEQUENCE-LEVEL IMPROVEMENTS VS STATIC")
    for policy in policies[1:]:
        gains = sequence_scores["static_regret"] - sequence_scores[policy]

        # Descriptive percentile bootstrap interval over sequences.
        boot_rng = np.random.default_rng(SEED)
        boot = np.array(
            [
                boot_rng.choice(gains.to_numpy(), size=len(gains), replace=True).mean()
                for _ in range(2000)
            ]
        )

        print(
            f"{policy}: mean={gains.mean():.4f}, "
            f"win={np.mean(gains > 0):.3f}, "
            f"CI=[{np.quantile(boot, 0.025):.4f}, "
            f"{np.quantile(boot, 0.975):.4f}]"
        )

    print("\nDOWNSIDE RISK")
    print(results[policies].quantile([0.9, 0.99]).round(4))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT, index=False)
    sequence_scores.to_csv(SUMMARY_OUT)

    print("\nSaved:", OUT)
    print("Saved:", SUMMARY_OUT)


if __name__ == "__main__":
    main()
