"""Transport contracts; provenance declarations are not preprocessing verification."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Finite = Annotated[float, Field(allow_inf_nan=False)]
Probability = Annotated[Finite, Field(ge=0, le=1)]
Identifier = Annotated[str, Field(min_length=1)]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FeatureProvenance(Contract):
    feature_schema: Identifier
    preprocessing_version: Identifier
    pipeline_id: Identifier
    declaration: Literal["current_and_past_only"]


class ModelMetadata(Contract):
    artifact_id: Identifier
    model_version: Identifier


class HourObservation(Contract):
    episode_id: Identifier
    icu_hour: Annotated[Finite, Field(gt=0)]
    features: dict[str, Finite | None]


class TrajectoryRequest(Contract):
    dataset: Literal["sepsis2019"]
    episode_id: Identifier
    provenance: FeatureProvenance
    model: ModelMetadata
    observations: list[HourObservation] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_history(self):
        previous = None
        reserved = {"ICULOS", "episode_id", "patient_id", "SepsisLabel", "split"}
        for row in self.observations:
            if row.episode_id != self.episode_id:
                raise ValueError("Mixed episode identifiers")
            if previous is not None and row.icu_hour <= previous:
                raise ValueError("ICU hours must be strictly increasing and unique")
            if reserved.intersection(row.features):
                raise ValueError("Reserved identity, time or evaluation feature")
            previous = row.icu_hour
        return self


class HourPrediction(Contract):
    icu_hour: Finite
    static_probability: Probability
    temporal_probability: Probability | None
    final_probability: Probability
    temporal_weight: Probability
    static_weight: Probability
    status: Literal["bounded_anticipatory", "static_only_fallback"]
    reason: Literal["no_verified_feedback", "secondary_inference_failed"]


class TrajectoryResponse(Contract):
    dataset: Literal["sepsis2019"] = "sepsis2019"
    episode_id: Identifier
    provenance: FeatureProvenance
    provenance_status: Literal["declared_compatible_not_independently_verified"] = (
        "declared_compatible_not_independently_verified"
    )
    model: ModelMetadata
    predictions: list[HourPrediction]
    probability_type: Literal["shifted_challenge_label"] = "shifted_challenge_label"
    research_only: Literal[True] = True
    clinical_use: Literal[False] = False
    configured_parameters: dict[str, float]
    parameters_empirically_calibrated: Literal[False] = False
