/** Dashboard view of the additive v0.6 contracts. No labels enter inference. */
export interface Unavailable { status: 'unavailable'; value: null; reason: string }
export interface DecisionState {
  icu_hour: number
  static_probability: number
  temporal_probability: number | null
  final_probability: number
  static_weight: number
  temporal_weight: number
  status: 'bounded_anticipatory' | 'static_only_fallback'
  reason: 'no_verified_feedback' | 'secondary_inference_failed'
  static_contribution: number
  temporal_contribution: number | null
}
export interface DecisionSummary {
  schema_version: 'decision-summary.v0.6'
  episode_id: string
  current: DecisionState
  previous: DecisionState | null
  elapsed_hours: number | null
  change: {
    status: 'available' | 'unavailable'; reason: string | null
    interval: 'since_previous_observed_hour'
    probability_delta: number | null; static_delta: number | null
    disagreement_change_at_previous_allocation: number | null
    reallocation_at_current_disagreement: number | null
    reconstruction_residual: number | null
    interpretation: 'arithmetic_attribution_not_medical_causality'
  }
  diagnostics: {
    signed_model_disagreement: number | null
    absolute_model_disagreement: number | null
    final_predictive_entropy_nats: number
    entropy_interpretation: 'predictive_distribution_not_reliability'
  }
  data_quality: {
    status: 'available' | 'unavailable'; reason: string | null
    source_availability_verified: false
    measurements: Record<string, { observed: boolean; age_hours: number | null }>
    missing_count: number | null; missing_fraction: number | null
    engineered_feature_count: number | null; engineered_missing_count: number | null
  }
  controller: {
    policy_id: string; allocation_basis: string; empirically_validated: false
    exposure: number; exposure_empirically_calibrated: false
    anticipatory_proposal: number | null; estimated_gain: number | null
    fallback_active: boolean; fallback_reason: string | null
    emergency_overrides_step_limit: boolean
    configured_constraints: Record<string, number>; constraint_trace: string
    active_constraints?: string[]
    reactive: Unavailable; covariance: Unavailable
  }
  provenance: {
    dataset: 'sepsis2019'; target: 'shifted_challenge_label'
    model: { artifact_id: string; model_version: string }
    artifact_sha256: string | null; artifact_status: string
    features: { feature_schema: string; preprocessing_version: string; pipeline_id: string; declaration: string }
    feature_status: string
  }
  evaluation: Unavailable & { recorded_summary_path: string }
  next_computational_step: 'await_next_valid_observation_then_update_causal_features_and_reevaluate'
  next_hour_forecast: Unavailable; learned_hybrid: Unavailable
  workflow_recommendation: Unavailable
  research_only: true; clinical_use: false
}
export interface TrajectoryResponse {
  wire_schema_version?: string
  episode_id: string
  decision_summaries: DecisionSummary[]
  research_only: true
  clinical_use: false
}
export interface RawTrajectoryRequest {
  dataset: 'sepsis2019'; episode_id: string
  model: { artifact_id: string; model_version: string }
  observations: { episode_id: string; icu_hour: number; measurements: Record<string, number | null> }[]
}
export interface RecordedMetric {
  metric_id: string; value: number; direction: 'minimize' | 'maximize'
  aggregation: string; unit: string; metric_version: string
  category: 'retrospective_label_dependent'
}
export interface EvaluationSummary {
  schema_version: 'evaluation-summary.v0.6'
  category: 'retrospective_label_dependent'
  status: 'available_recorded_synthetic_evidence_only'
  installed_sepsis_model_evaluation: 'unavailable'
  live_labels_used: false
  experiments: {
    experiment_id: string; dataset: 'synthetic_binary_sequences'; target: 'simulated_binary_event'
    split: string; n_test_episodes: number
    provenance: {
      source_path: string; source_sha256: string; evidence_status: string
      precision: string; model_artifact_sha256: null
      score_clipping_epsilon: null; missing_provenance: string[]
    }
    policies: { policy_id: string; metrics: RecordedMetric[] }[]
    comparison: {
      reference_policy_id: string; candidate_policy_id: string; metric: RecordedMetric
      ci95: [number, number]; interval_method: string; bootstrap_replicates: number
    }
    limitations: string[]
  }[]
  unavailable_metrics: string[]; unavailable_reason: string
  research_only: true; clinical_use: false
}

function record(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error('Invalid API object')
  return value as Record<string, unknown>
}
function check(condition: unknown): asserts condition {
  if (!condition) throw new Error('Unsupported or malformed v0.6 research response')
}
const text = (v: unknown) => typeof v === 'string' && v.length > 0
const finite = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v)
const probability = (v: unknown) => finite(v) && v >= 0 && v <= 1
const optional = (v: unknown, validator: (v: unknown) => boolean) => v === null || validator(v)
const textList = (v: unknown) => Array.isArray(v) && v.every(text)
function unavailable(value: unknown) {
  const v = record(value)
  check(v.status === 'unavailable' && v.value === null && text(v.reason))
}
function state(value: unknown) {
  const v = record(value)
  check(finite(v.icu_hour) && v.icu_hour > 0 && ['no_verified_feedback', 'secondary_inference_failed'].includes(String(v.reason)))
  for (const key of ['static_probability', 'final_probability', 'static_weight', 'temporal_weight', 'static_contribution']) check(probability(v[key]))
  for (const key of ['temporal_probability', 'temporal_contribution']) check(optional(v[key], probability))
  check(v.status === 'bounded_anticipatory' || v.status === 'static_only_fallback')
}
export function assertDecisionSummary(value: unknown): asserts value is DecisionSummary {
  const v = record(value)
  check(v.schema_version === 'decision-summary.v0.6' && v.research_only === true && v.clinical_use === false && text(v.episode_id))
  state(v.current)
  if (v.previous !== null) state(v.previous)
  check(optional(v.elapsed_hours, n => finite(n) && n > 0))
  const currentState = record(v.current)
  if (v.previous === null) check(v.elapsed_hours === null)
  else {
    const previousState = record(v.previous)
    check(Number(previousState.icu_hour) < Number(currentState.icu_hour))
    check(v.elapsed_hours === Number(currentState.icu_hour) - Number(previousState.icu_hour))
  }
  const change = record(v.change)
  check(['available', 'unavailable'].includes(String(change.status)) && optional(change.reason, text))
  check(change.interval === 'since_previous_observed_hour' && change.interpretation === 'arithmetic_attribution_not_medical_causality')
  for (const key of ['probability_delta', 'static_delta', 'disagreement_change_at_previous_allocation', 'reallocation_at_current_disagreement', 'reconstruction_residual']) check(optional(change[key], finite))
  const diagnostics = record(v.diagnostics)
  check(optional(diagnostics.signed_model_disagreement, finite) && optional(diagnostics.absolute_model_disagreement, probability))
  check(finite(diagnostics.final_predictive_entropy_nats) && diagnostics.final_predictive_entropy_nats >= 0 && diagnostics.entropy_interpretation === 'predictive_distribution_not_reliability')
  const quality = record(v.data_quality)
  check(['available', 'unavailable'].includes(String(quality.status)) && optional(quality.reason, text) && quality.source_availability_verified === false)
  for (const item of Object.values(record(quality.measurements))) {
    const measurement = record(item)
    check(typeof measurement.observed === 'boolean' && optional(measurement.age_hours, n => finite(n) && n >= 0))
  }
  check(optional(quality.missing_fraction, probability) && optional(quality.missing_count, n => Number.isInteger(n) && Number(n) >= 0))
  check(Number.isInteger(quality.engineered_feature_count) && Number.isInteger(quality.engineered_missing_count))
  const controller = record(v.controller)
  check(text(controller.policy_id) && text(controller.allocation_basis) && controller.empirically_validated === false && controller.exposure_empirically_calibrated === false)
  check(probability(controller.exposure) && optional(controller.anticipatory_proposal, probability) && optional(controller.estimated_gain, finite))
  check(typeof controller.fallback_active === 'boolean' && typeof controller.emergency_overrides_step_limit === 'boolean' && optional(controller.fallback_reason, text) && text(controller.constraint_trace))
  check(Object.values(record(controller.configured_constraints)).every(finite))
  unavailable(controller.reactive); unavailable(controller.covariance)
  const provenance = record(v.provenance)
  check(provenance.dataset === 'sepsis2019' && provenance.target === 'shifted_challenge_label')
  check(optional(provenance.artifact_sha256, s => typeof s === 'string' && /^[0-9a-f]{64}$/.test(s)) && text(provenance.artifact_status) && text(provenance.feature_status))
  const model = record(provenance.model)
  check(text(model.artifact_id) && text(model.model_version))
  const features = record(provenance.features)
  for (const key of ['feature_schema', 'preprocessing_version', 'pipeline_id', 'declaration']) check(text(features[key]))
  for (const key of ['evaluation', 'next_hour_forecast', 'learned_hybrid', 'workflow_recommendation']) unavailable(v[key])
  check(text(record(v.evaluation).recorded_summary_path))
  check(v.next_computational_step === 'await_next_valid_observation_then_update_causal_features_and_reevaluate')
}
export function parseTrajectory(value: unknown): TrajectoryResponse {
  const v = record(value)
  check(text(v.episode_id) && v.research_only === true && v.clinical_use === false)
  check(Array.isArray(v.decision_summaries) && v.decision_summaries.length > 0)
  let previousHour = 0
  for (const summary of v.decision_summaries) {
    assertDecisionSummary(summary)
    check(summary.episode_id === v.episode_id && summary.current.icu_hour > previousHour)
    check(previousHour === 0 ? summary.previous === null : summary.previous?.icu_hour === previousHour)
    previousHour = summary.current.icu_hour
  }
  return value as TrajectoryResponse
}
function metric(value: unknown) {
  const v = record(value)
  check(text(v.metric_id) && finite(v.value) && ['minimize', 'maximize'].includes(String(v.direction)))
  check(text(v.aggregation) && text(v.unit) && text(v.metric_version) && v.category === 'retrospective_label_dependent')
}
export function parseEvaluation(value: unknown): EvaluationSummary {
  const v = record(value)
  check(v.schema_version === 'evaluation-summary.v0.6' && v.research_only === true && v.clinical_use === false && v.live_labels_used === false)
  check(v.status === 'available_recorded_synthetic_evidence_only' && v.category === 'retrospective_label_dependent' && v.installed_sepsis_model_evaluation === 'unavailable')
  check(Array.isArray(v.experiments) && v.experiments.length > 0 && textList(v.unavailable_metrics) && text(v.unavailable_reason))
  for (const item of v.experiments) {
    const experiment = record(item)
    check(text(experiment.experiment_id) && experiment.dataset === 'synthetic_binary_sequences' && experiment.target === 'simulated_binary_event')
    check(text(experiment.split) && Number.isInteger(experiment.n_test_episodes) && Number(experiment.n_test_episodes) > 0 && textList(experiment.limitations))
    const provenance = record(experiment.provenance)
    check(text(provenance.source_path) && typeof provenance.source_sha256 === 'string' && /^[0-9a-f]{64}$/.test(provenance.source_sha256))
    check(text(provenance.evidence_status) && text(provenance.precision) && textList(provenance.missing_provenance))
    check(provenance.model_artifact_sha256 === null && provenance.score_clipping_epsilon === null)
    check(Array.isArray(experiment.policies) && experiment.policies.length > 0)
    for (const item of experiment.policies) {
      const policy = record(item)
      check(text(policy.policy_id) && Array.isArray(policy.metrics) && policy.metrics.length > 0)
      policy.metrics.forEach(metric)
    }
    const comparison = record(experiment.comparison)
    metric(comparison.metric)
    check(text(comparison.reference_policy_id) && text(comparison.candidate_policy_id) && text(comparison.interval_method) && Number.isInteger(comparison.bootstrap_replicates))
    check(Array.isArray(comparison.ci95) && comparison.ci95.length === 2 && comparison.ci95.every(finite) && comparison.ci95[0] <= comparison.ci95[1])
  }
  return value as EvaluationSummary
}
export function parseRawRequest(textValue: string): RawTrajectoryRequest {
  let value: unknown
  try { value = JSON.parse(textValue) } catch { throw new Error('Enter a valid JSON trajectory request.') }
  const v = record(value)
  check(v.dataset === 'sepsis2019' && text(v.episode_id))
  const model = record(v.model)
  check(text(model.artifact_id) && text(model.model_version))
  check(Array.isArray(v.observations) && v.observations.length > 0)
  let hour = 0
  const required = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'Age', 'Gender', 'HospAdmTime', 'Unit1', 'Unit2']
  for (const item of v.observations) {
    const row = record(item)
    check(row.episode_id === v.episode_id && Number.isInteger(row.icu_hour) && Number(row.icu_hour) > hour)
    hour = Number(row.icu_hour)
    const measurements = record(row.measurements)
    check(required.every(key => key in measurements) && Object.values(measurements).every(n => optional(n, finite)))
    check(!['SepsisLabel', 'y_true', 'patient_id', 'split'].some(key => key in measurements))
  }
  return value as RawTrajectoryRequest
}
