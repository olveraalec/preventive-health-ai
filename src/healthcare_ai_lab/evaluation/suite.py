"""Patient-aware evaluation and paired model comparisons."""

import numpy as np
import pandas as pd

from healthcare_ai_lab.evaluation.registry import evaluate


def evaluate_recordings(
    data: pd.DataFrame,
    target: str = "y_true",
    prediction: str = "y_pred",
    group: str = "recording_id",
    policy: str = "policy",
    sigma: str | None = None,
) -> pd.DataFrame:
    """Evaluate each recording and policy independently."""

    required = {target, prediction, group, policy}
    if sigma is not None:
        required.add(sigma)

    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    rows = []

    for (policy_id, recording_id), part in data.groupby([policy, group], sort=True):
        y = part[target].to_numpy(dtype=float)
        pred = part[prediction].to_numpy(dtype=float)

        # Use the same valid observations for each metric.
        valid = np.isfinite(y) & np.isfinite(pred)

        if sigma is not None:
            uncertainty = part[sigma].to_numpy(dtype=float)
            valid &= np.isfinite(uncertainty) & (uncertainty > 0)
            uncertainty = uncertainty[valid]
        else:
            uncertainty = None

        if not valid.any():
            raise ValueError(f"No valid observations for {policy_id}, {recording_id}")

        scores = evaluate(y[valid], pred[valid], sigma=uncertainty)

        rows.append(
            {
                "policy": policy_id,
                "recording_id": recording_id,
                "n_observations": int(valid.sum()),
                **scores,
            }
        )

    return pd.DataFrame(rows)


def compare_paired(
    recording_metrics: pd.DataFrame,
    reference: str,
    candidate: str,
    metric: str = "rmse",
    n_bootstrap: int = 2000,
    seed: int = 12345,
) -> dict:
    """Paired, recording-level bootstrap comparison.

    Positive gain means the candidate improves a lower-is-better
    metric relative to the reference.
    """

    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")

    if metric not in recording_metrics.columns:
        raise ValueError(f"Unknown or unavailable metric: {metric}")

    subset = recording_metrics[recording_metrics["policy"].isin([reference, candidate])]

    if subset.duplicated(["recording_id", "policy"]).any():
        raise ValueError(
            "Duplicate recording-policy pairs; aggregate repeated runs first."
        )

    paired = subset.pivot(
        index="recording_id",
        columns="policy",
        values=metric,
    )

    if reference not in paired or candidate not in paired:
        raise ValueError("Reference or candidate not found")

    # Require identical patient coverage.
    if paired[[reference, candidate]].isna().any().any():
        raise ValueError("Unpaired recording comparison")

    gains = (paired[reference] - paired[candidate]).to_numpy(dtype=float)

    if not np.isfinite(gains).all():
        raise ValueError("Nonfinite paired gains")

    rng = np.random.default_rng(seed)

    samples = rng.choice(
        gains,
        size=(n_bootstrap, len(gains)),
        replace=True,
    )

    bootstrap_means = samples.mean(axis=1)

    return {
        "reference": reference,
        "candidate": candidate,
        "metric": metric,
        "n_recordings": len(gains),
        "mean_gain": float(gains.mean()),
        "median_gain": float(np.median(gains)),
        "win_fraction": float(np.mean(gains > 0)),
        "ci_lower": float(np.quantile(bootstrap_means, 0.025)),
        "ci_upper": float(np.quantile(bootstrap_means, 0.975)),
        "worst_gain": float(gains.min()),
    }
