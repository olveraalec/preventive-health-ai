"""Server-owned Notebook 02 feature generation before research inference."""

from typing import Literal

import pandas as pd
from pydantic import Field, model_validator

from healthcare_ai_lab.preprocessing.sepsis import (
    FEATURE_SCHEMA,
    NOTEBOOK_TRANSFORM_SHA256,
    PIPELINE_ID,
    PREPROCESSING_VERSION,
    RAW_REQUIRED,
    UNUSED_CHALLENGE_COLUMNS,
    build_sepsis_features,
    feature_sources,
)
from healthcare_ai_lab.trajectories.contracts import (
    Contract,
    FeatureProvenance,
    Finite,
    Identifier,
    ModelMetadata,
    TrajectoryRequest,
    TrajectoryResponse,
)
from healthcare_ai_lab.trajectories.service import (
    ArtifactCompatibilityError,
    predict_trajectory,
)


def runtime_provenance() -> FeatureProvenance:
    return FeatureProvenance(
        feature_schema=FEATURE_SCHEMA,
        preprocessing_version=PREPROCESSING_VERSION,
        pipeline_id=PIPELINE_ID,
        declaration="current_and_past_only",
    )


class RawHourObservation(Contract):
    episode_id: Identifier
    icu_hour: int = Field(gt=0, strict=True)
    measurements: dict[str, Finite | None]

    @model_validator(mode="after")
    def source_schema(self):
        if set(RAW_REQUIRED) - self.measurements.keys():
            raise ValueError(
                "Missing raw source columns; supply null for unmeasured values"
            )
        if self.measurements.keys() - {*RAW_REQUIRED, *UNUSED_CHALLENGE_COLUMNS}:
            raise ValueError("Unknown, engineered or evaluation source column")
        return self


class RawTrajectoryRequest(Contract):
    dataset: Literal["sepsis2019"]
    episode_id: Identifier
    model: ModelMetadata
    observations: list[RawHourObservation] = Field(min_length=1)

    @model_validator(mode="after")
    def ordered_episode(self):
        previous = 0
        for row in self.observations:
            if row.episode_id != self.episode_id:
                raise ValueError("Mixed episode identifiers")
            if row.icu_hour <= previous:
                raise ValueError("ICU hours must be strictly increasing and unique")
            previous = row.icu_hour
        return self


class FeatureEngineeringMetadata(Contract):
    notebook_transform_sha256: str = NOTEBOOK_TRANSFORM_SHA256
    source_columns: dict[str, list[str]]
    gap_policy: Literal["observed_records_no_resampling"] = (
        "observed_records_no_resampling"
    )
    units: Literal["source_units_no_conversion"] = "source_units_no_conversion"


class RawTrajectoryResponse(TrajectoryResponse):
    provenance_status: Literal["runtime_generated_source_availability_unverified"] = (
        "runtime_generated_source_availability_unverified"
    )
    feature_engineering: FeatureEngineeringMetadata


def predict_raw_trajectory(
    bundle: dict, request: RawTrajectoryRequest
) -> RawTrajectoryResponse:
    """No client-engineered features; no silent substitution for unknown model inputs."""
    allowed = set(feature_sources())
    static = set(bundle["static"]["features"])
    temporal = set(bundle["temporal"]["features"])
    meta = set(bundle["meta_features"])
    if (static | temporal) - allowed or meta - (allowed | {"entropy_static"}):
        raise ArtifactCompatibilityError(
            "Model requests features outside Notebook 02 schema"
        )
    raw = pd.DataFrame(
        [
            {
                **row.measurements,
                "ICULOS": row.icu_hour,
                "patient_id": request.episode_id,
            }
            for row in request.observations
        ]
    )
    engineered = build_sepsis_features(raw).drop(columns="patient_id")
    # JSON contracts represent missing measurements as null. Restore NaN in the
    # shared inference adapter before model-owned preprocessing.
    records = (
        engineered.astype(object).where(engineered.notna(), None).to_dict("records")
    )
    history = TrajectoryRequest(
        dataset=request.dataset,
        episode_id=request.episode_id,
        provenance=runtime_provenance(),
        model=request.model,
        observations=[
            {
                "episode_id": request.episode_id,
                "icu_hour": record.pop("ICULOS"),
                "features": record,
            }
            for record in records
        ],
    )
    result = predict_trajectory(bundle, history)
    return RawTrajectoryResponse(
        **result.model_dump(exclude={"provenance_status"}),
        feature_engineering=FeatureEngineeringMetadata(
            source_columns=feature_sources()
        ),
    )
