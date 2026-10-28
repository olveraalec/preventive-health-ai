// Mirrors healthcare_ai_lab/trajectories/{contracts,raw}.py (RawTrajectoryRequest / RawTrajectoryResponse).
export type Status = "bounded_anticipatory" | "static_only_fallback";
export type Reason = "no_verified_feedback" | "secondary_inference_failed";

export interface HourPrediction {
  icu_hour: number;
  static_probability: number;
  temporal_probability: number | null;
  final_probability: number;
  temporal_weight: number;
  static_weight: number;
  status: Status;
  reason: Reason;
}
export interface ModelMetadata { artifact_id: string; model_version: string }
export interface Provenance {
  feature_schema: string;
  preprocessing_version: string;
  pipeline_id: string;
  declaration: "current_and_past_only";
}
export interface TrajectoryResponse {
  dataset: "sepsis2019";
  episode_id: string;
  provenance: Provenance;
  provenance_status: string;
  model: ModelMetadata;
  predictions: HourPrediction[];
  probability_type: "shifted_challenge_label";
  research_only: true;
  clinical_use: false;
  configured_parameters: Record<string, number>;
  parameters_empirically_calibrated: false;
}
export interface RawHourObservation {
  episode_id: string;
  icu_hour: number;
  measurements: Record<string, number | null>; // null = not measured
}
export interface RawTrajectoryRequest {
  dataset: "sepsis2019";
  episode_id: string;
  model: ModelMetadata;
  observations: RawHourObservation[];
}
export interface Episode { request: RawTrajectoryRequest; response: TrajectoryResponse }
export const VITALS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp"] as const;
