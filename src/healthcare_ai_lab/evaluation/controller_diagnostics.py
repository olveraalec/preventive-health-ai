"""Label-free, within-episode allocation diagnostics. No controller is executed."""

import numpy as np

from healthcare_ai_lab.evaluation.contracts import EvaluationConfig, metric_value


def controller_diagnostics(
    weights,
    *,
    static_probability=None,
    temporal_probability=None,
    emergency_fallback=None,
    config=None,
):
    config = EvaluationConfig.model_validate(config if config is not None else {})
    q = np.asarray(weights, dtype=float)
    if (
        q.ndim != 1
        or not len(q)
        or not np.isfinite(q).all()
        or np.any((q < 0) | (q > 1))
    ):
        raise ValueError("Expected nonempty finite allocation vector in [0,1]")
    delta = np.diff(q)
    n, steps = len(q), len(delta)
    values = {
        "q_mean": float(q.mean()),
        "q_variance": float(q.var(ddof=0)),
        "q_std": float(q.std(ddof=0)),
        "q_min": float(q.min()),
        "q_max": float(q.max()),
        "q_near_static_fraction": float(np.mean(q <= config.near_static_tolerance)),
        "static_mode_fraction": float(np.mean(q == 0)),
        "mean_absolute_change": float(np.abs(delta).mean()) if steps else None,
        "mean_squared_change": float(np.square(delta).mean()) if steps else None,
        "allocation_change_frequency": float(
            np.mean(np.abs(delta) > config.allocation_change_tolerance)
        )
        if steps
        else None,
        "expert_switching_frequency": float(
            np.mean(np.diff(q >= config.switching_threshold) != 0)
        )
        if steps
        else None,
        "q_total_variation": float(np.abs(delta).sum()),
        "allocation_step_count": steps,
    }
    results = {}
    step_metrics = {
        "mean_absolute_change",
        "mean_squared_change",
        "allocation_change_frequency",
        "expert_switching_frequency",
    }
    for name, value in values.items():
        unit = (
            "squared_allocation"
            if name in ("q_variance", "mean_squared_change")
            else "allocation"
        )
        if name.endswith("frequency") or name.endswith("fraction"):
            unit = "fraction"
        results[name] = metric_value(
            name,
            value,
            n=steps if name in step_metrics else n,
            patients=1,
            aggregation="within_patient",
            reason="no_within_patient_transitions" if value is None else None,
            controller=True,
            unit="count" if name == "allocation_step_count" else unit,
        )
    if emergency_fallback is None:
        fallback_count, fallback_frequency = None, None
    else:
        flags = np.asarray(emergency_fallback)
        if flags.shape != q.shape or flags.dtype != np.dtype(bool):
            raise ValueError("Emergency fallback flags must be aligned booleans")
        if np.any(q[flags] != 0):
            raise ValueError(
                "Emergency fallback must have immediate zero temporal weight"
            )
        fallback_count, fallback_frequency = int(flags.sum()), float(flags.mean())
    for name, value in (
        ("emergency_fallback_count", fallback_count),
        ("fallback_frequency", fallback_frequency),
    ):
        results[name] = metric_value(
            name,
            value,
            n=n,
            patients=1,
            controller=True,
            aggregation="within_patient",
            reason="fallback_flags_not_supplied",
            unit="count" if name.endswith("count") else "fraction",
        )
    pair_count = 0
    square, absolute = None, None
    if static_probability is not None and temporal_probability is not None:
        ps, pt = (
            np.asarray(static_probability, dtype=float),
            np.asarray(temporal_probability, dtype=float),
        )
        if ps.shape != q.shape or pt.shape != q.shape or not np.isfinite(ps).all():
            raise ValueError("Invalid expert probability shapes/static values")
        if (
            np.isinf(pt).any()
            or np.any((ps < 0) | (ps > 1))
            or np.any((pt < 0) | (pt > 1))
        ):
            raise ValueError("Invalid expert probabilities")
        valid = np.isfinite(pt)
        pair_count = int(valid.sum())
        if pair_count:
            disagreement = ps[valid] - pt[valid]
            square, absolute = (
                float(np.square(disagreement).mean()),
                float(np.abs(disagreement).mean()),
            )
    for name, value in (
        ("expert_disagreement_mean_squared", square),
        ("expert_disagreement_mean_absolute", absolute),
    ):
        results[name] = metric_value(
            name,
            value,
            n=pair_count,
            patients=1,
            controller=True,
            aggregation="within_patient_available_expert_pairs",
            reason="no_available_expert_pairs",
            unit="squared_probability_difference"
            if name.endswith("squared")
            else "probability_difference",
        )
    results["available_expert_pair_fraction"] = metric_value(
        "available_expert_pair_fraction",
        pair_count / n,
        n=n,
        patients=1,
        controller=True,
        aggregation="within_patient",
        unit="fraction",
    )
    return results
