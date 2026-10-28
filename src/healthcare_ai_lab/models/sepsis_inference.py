"""Research-only inference on causally engineered Sepsis Challenge rows.

Inputs MUST be produced by the same feature pipeline as training. This module
never treats the shifted challenge label as clinical onset probability.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit

from healthcare_ai_lab.controllers.adaptive import (
    ControllerConfig, adaptive_allocation, blend_probabilities,
)


def load_bundle(path: str | Path) -> dict[str, Any]:
    """Load a *trusted local* joblib bundle; never accept a user-supplied path."""
    import joblib
    bundle = joblib.load(Path(path))
    for k in ("static", "temporal", "controller", "meta_features"):
        if k not in bundle:
            raise ValueError(f"Model bundle missing {k}")
    return bundle


def _prob(model_bundle: dict, row: pd.DataFrame) -> float:
    features = model_bundle["features"]
    missing = sorted(set(features) - set(row.columns))
    if missing:
        raise ValueError(f"Missing engineered features: {missing}")
    p = float(model_bundle["model"].predict_proba(row[features])[0, 1])
    if not np.isfinite(p) or not 0 <= p <= 1:
        raise ValueError("Invalid model probability")
    return p


def predict_engineered_row(
    bundle: dict[str, Any], row: dict[str, Any],
    *, previous_weight: float | None = None,
    reliability: float = 0.25,
    config: ControllerConfig | None = None,
) -> dict[str, Any]:
    """One-hour prediction with no outcome-label access or oracle feedback."""
    if "SepsisLabel" in row or "split" in row:
        # Explicitly remove evaluation-only columns from inference input.
        row = {k: v for k, v in row.items() if k not in ("SepsisLabel", "split")}
    frame = pd.DataFrame([row])
    ps = _prob(bundle["static"], frame)
    pt = _prob(bundle["temporal"], frame)
    p_clip = np.clip(ps, 1e-12, 1 - 1e-12)
    h = float(-p_clip * np.log(p_clip) - (1-p_clip) * np.log1p(-p_clip))
    context = dict(row)
    context["entropy_static"] = h
    meta_features = bundle["meta_features"]
    missing = sorted(set(meta_features) - set(context))
    if missing:
        raise ValueError(f"Missing controller features: {missing}")
    raw_gain = float(bundle["controller"].predict(pd.DataFrame([context])[meta_features])[0])
    temperature = float(bundle.get("temperature", 0.01))
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Invalid controller temperature")
    qa = float(expit(raw_gain / temperature))
    # No verified delayed outcomes are available in this endpoint.
    cfg = config or ControllerConfig(fallback_weight=float(bundle.get("fallback_q", 0.0)),
                                     max_deviation=0.25, max_step=0.10)
    q = adaptive_allocation(
        q_anticipatory=qa, q_reactive=cfg.fallback_weight,
        reliability=reliability, anticipatory_share=1.0,
        feedback_available=False, previous_q=previous_weight, config=cfg,
    )
    p = blend_probabilities(ps, pt, q)
    return {
        "icu_hour": float(row.get("ICULOS", 0)),
        "prediction": {"static_probability": ps, "temporal_probability": pt,
                       "final_probability": p,
                       "probability_type": "shifted_challenge_label"},
        "controller": {"temporal_weight": q, "static_weight": 1-q,
                       "anticipatory_proposal": qa, "estimated_gain": raw_gain,
                       "feedback_available": False,
                       "mode": "bounded_anticipatory_research"},
        "reliability": {"predictive_entropy": h, "exposure": reliability},
        "metadata": {"clinical_use": False, "model_status": "research_prototype"},
    }


def predict_engineered_trajectory(bundle: dict, frame: pd.DataFrame,
                                  reliability: float = 0.25) -> list[dict]:
    """Sequentially score one already-engineered patient history."""
    if frame.empty or "ICULOS" not in frame:
        raise ValueError("Trajectory must have ICULOS rows")
    frame = frame.sort_values("ICULOS")
    if frame.ICULOS.duplicated().any():
        raise ValueError("Duplicate patient ICU hour")
    results, previous = [], None
    for _, r in frame.iterrows():
        result = predict_engineered_row(bundle, r.to_dict(),
                                        previous_weight=previous,
                                        reliability=reliability)
        previous = result["controller"]["temporal_weight"]
        results.append(result)
    return results
