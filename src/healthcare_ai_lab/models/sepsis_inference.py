"""Research-only inference on causally engineered Sepsis Challenge rows.

Inputs MUST be produced by the same feature pipeline as training. This module
never treats the shifted challenge label as clinical onset probability.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit, xlogy

from healthcare_ai_lab.controllers.adaptive import (
    ControllerConfig,
    adaptive_allocation,
    blend_probabilities,
)


def load_bundle(path: str | Path) -> dict[str, Any]:
    """Load a *trusted local* joblib bundle; never accept a user-supplied path."""
    import warnings

    import joblib
    from sklearn.exceptions import InconsistentVersionWarning

    from healthcare_ai_lab.models.artifact import (
        ArtifactManifest,
        ArtifactValidationError,
        attach_manifest,
        manifest_path,
        sha256_file,
    )

    path = Path(path)
    sidecar = manifest_path(path)
    manifest = (
        ArtifactManifest.model_validate_json(sidecar.read_text())
        if sidecar.is_file()
        else None
    )
    if manifest is not None and sha256_file(path) != manifest.artifact_sha256:
        raise ArtifactValidationError("Artifact checksum mismatch")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", InconsistentVersionWarning)
            bundle = joblib.load(path)
    except InconsistentVersionWarning:
        raise ArtifactValidationError(
            "Serialized sklearn version is incompatible"
        ) from None
    try:
        for k in ("static", "temporal", "controller", "meta_features"):
            if k not in bundle:
                raise ArtifactValidationError(f"Model bundle missing {k}")
        return attach_manifest(bundle, manifest) if manifest is not None else bundle
    except (KeyError, TypeError, AttributeError):
        raise ArtifactValidationError("Invalid model artifact structure") from None


def static_entropy(probability):
    """Notebook 09 binary_entropy: clip log arguments, retain original probabilities."""
    p = np.asarray(probability)
    clipped = np.clip(p, 1e-10, 1 - 1e-10)
    return -xlogy(p, clipped) - xlogy(1 - p, 1 - clipped)


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
    bundle: dict[str, Any],
    row: dict[str, Any],
    *,
    previous_weight: float | None = None,
    reliability: float = 0.25,
    config: ControllerConfig | None = None,
    static_on_secondary_failure: bool = False,
) -> dict[str, Any]:
    """One-hour prediction with no outcome-label access or oracle feedback."""
    if "SepsisLabel" in row or "split" in row:
        # Explicitly remove evaluation-only columns from inference input.
        row = {k: v for k, v in row.items() if k not in ("SepsisLabel", "split")}
    frame = pd.DataFrame([row])
    ps = _prob(bundle["static"], frame)
    h = float(static_entropy(ps))
    try:
        pt = _prob(bundle["temporal"], frame)
        context = dict(row)
        context["entropy_static"] = h
        meta_features = bundle["meta_features"]
        missing = sorted(set(meta_features) - set(context))
        if missing:
            raise ValueError(f"Missing controller features: {missing}")
        raw_gain = float(
            bundle["controller"].predict(pd.DataFrame([context])[meta_features])[0]
        )
        if not np.isfinite(raw_gain):
            raise ValueError("Invalid controller gain")
        temperature = float(bundle.get("temperature", 0.01))
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("Invalid controller temperature")
        qa = float(expit(raw_gain / temperature))
        # No verified delayed outcomes are available in this endpoint.
        cfg = config or ControllerConfig(
            fallback_weight=float(bundle.get("fallback_q", 0.0)),
            max_deviation=0.25,
            max_step=0.10,
        )
        q = adaptive_allocation(
            q_anticipatory=qa,
            q_reactive=cfg.fallback_weight,
            reliability=reliability,
            anticipatory_share=1.0,
            feedback_available=False,
            previous_q=previous_weight,
            config=cfg,
        )
    except Exception:
        if not static_on_secondary_failure:
            raise
        # Static inference succeeded; a secondary failure overrides switching limits.
        return {
            "icu_hour": float(row.get("ICULOS", 0)),
            "prediction": {
                "static_probability": ps,
                "temporal_probability": None,
                "final_probability": ps,
                "probability_type": "shifted_challenge_label",
            },
            "controller": {
                "temporal_weight": 0.0,
                "static_weight": 1.0,
                "status": "static_only_fallback",
                "reason": "secondary_inference_failed",
            },
        }
    p = blend_probabilities(ps, pt, q)
    return {
        "icu_hour": float(row.get("ICULOS", 0)),
        "prediction": {
            "static_probability": ps,
            "temporal_probability": pt,
            "final_probability": p,
            "probability_type": "shifted_challenge_label",
        },
        "controller": {
            "temporal_weight": q,
            "static_weight": 1 - q,
            "anticipatory_proposal": qa,
            "estimated_gain": raw_gain,
            "feedback_available": False,
            "mode": "bounded_anticipatory_research",
        },
        "reliability": {"predictive_entropy": h, "exposure": reliability},
        "metadata": {"clinical_use": False, "model_status": "research_prototype"},
    }


def predict_engineered_trajectory(
    bundle: dict,
    frame: pd.DataFrame,
    reliability: float = 0.25,
    *,
    config: ControllerConfig | None = None,
    static_on_secondary_failure: bool = False,
) -> list[dict]:
    """Sequentially score one already-engineered patient history."""
    if frame.empty or "ICULOS" not in frame:
        raise ValueError("Trajectory must have ICULOS rows")
    frame = frame.sort_values("ICULOS")
    if frame.ICULOS.duplicated().any():
        raise ValueError("Duplicate patient ICU hour")
    results, previous = [], None
    for _, r in frame.iterrows():
        result = predict_engineered_row(
            bundle,
            r.to_dict(),
            previous_weight=previous,
            reliability=reliability,
            config=config,
            static_on_secondary_failure=static_on_secondary_failure,
        )
        previous = result["controller"]["temporal_weight"]
        results.append(result)
    return results
