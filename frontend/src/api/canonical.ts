/** Wire contract supplied by the authoritative FastAPI backend on port 8003. */
import schema from '../data/sepsis_decision_summary.schema.json' with { type: 'json' }
import type { DecisionState, RawTrajectoryRequest, TrajectoryResponse } from './contracts.ts'

type WireState = Omit<DecisionState, 'static_contribution' | 'temporal_contribution'>
export interface CanonicalDecisionResponse {
  schema_version: 'sepsis-decision-summary.v1'
  dataset: 'sepsis2019'; episode_id: string; target_semantics: 'shifted_challenge_label'
  model: RawTrajectoryRequest['model']; artifact_sha256: string | null
  artifact_integrity: 'checksum_bound_manifest' | 'declared_only'
  provenance: { feature_schema: string; preprocessing_version: string; pipeline_id: string; declaration: string }
  feature_transform_sha256: string; model_feature_hashes: Record<string, string>
  configured_parameters: Record<string, number>
  records: {
    current: WireState; previous: WireState | null
    change: {
      status: 'first_observation' | 'exact' | 'partial_temporal_unavailable'
      since: 'previous_observed_hour'; elapsed_hours: number | null
      total: number | null; static_movement: number | null
      disagreement_movement: number | null; controller_reallocation: number | null
      identity_residual: number | null; unsupported_terms: string[]
    }
    diagnostics: { predictive_entropy_nats: number; expert_gap: number | null; absolute_expert_gap: number | null; interpretation: string }
    information_quality: { observed_vitals: number; total_vitals: 7; vital_completeness: number; missing_vitals: string[]; availability_status: string }
    measurements: Record<string, {
      current_value: number | null; missing: boolean; last_value: number | null
      age_hours: number | null; last_observed_hour: number | null
      freshness_status: 'observed_hour_proxy' | 'never_observed' | 'not_tracked'
      used_by_feature_builder: boolean
    }> | null
    controller: {
      policy: string; mode: string; reason: string; emergency_fallback: boolean
      step_limit_overridden: boolean; active_constraints: string[]; evidence: string; feedback: string
    }
    next_step: { status: string; then: string[] }
    log_odds_contributions: Record<string, {
      status: string; reason: string | null; intercept: number | null
      terms: Record<string, number> | null; total_log_odds: number | null
      reconstructed_probability: number | null; interpretation: string
    }> | null
  }[]
  capabilities: {
    next_hour_forecast: { status: 'unavailable'; reason: string; probability: null; model_version: null; horizon_hours: 1; target: string }
    learned_hybrid: { status: 'unavailable'; reason: string }
    clinician_review: { status: 'unavailable'; reason: string }
  }
  payload_kind: 'api' | 'synthetic_example' | 'sanitized_retrospective_demo'
  research_only: true; clinical_use: false; limitations: string[]
}

type Schema = Record<string, unknown>
/** Small validator for the JSON-Schema features in the supplied, pinned contract.
 * Errors name fields only; no measurement values or episode identifiers are echoed.
 */
function validate(value: unknown, node: Schema, path = '$'): void {
  const fail = () => { throw new Error(`Unsupported sepsis-decision-summary.v1 response at ${path}.`) }
  if (typeof node.$ref === 'string') {
    const name = node.$ref.replace('#/$defs/', '')
    const definition = (schema.$defs as unknown as Record<string, Schema>)[name]
    if (!definition) fail()
    return validate(value, definition, path)
  }
  if (Array.isArray(node.anyOf)) {
    if (!node.anyOf.some(candidate => { try { validate(value, candidate as Schema, path); return true } catch { return false } })) fail()
    return
  }
  if ('const' in node && value !== node.const) fail()
  if (Array.isArray(node.enum) && !node.enum.includes(value)) fail()
  switch (node.type) {
    case 'null': if (value !== null) fail(); break
    case 'boolean': if (typeof value !== 'boolean') fail(); break
    case 'number': case 'integer': {
      if (typeof value !== 'number' || !Number.isFinite(value) || node.type === 'integer' && !Number.isInteger(value)) fail()
      const n = value as number
      if (typeof node.minimum === 'number' && n < node.minimum || typeof node.maximum === 'number' && n > node.maximum) fail()
      break
    }
    case 'string':
      if (typeof value !== 'string' || typeof node.minLength === 'number' && value.length < node.minLength || typeof node.pattern === 'string' && !new RegExp(node.pattern).test(String(value))) fail()
      break
    case 'array': {
      if (!Array.isArray(value)) fail()
      const items = value as unknown[]
      if (typeof node.minItems === 'number' && items.length < node.minItems || typeof node.maxItems === 'number' && items.length > node.maxItems) fail()
      items.forEach((item, i) => {
        const itemSchema = Array.isArray(node.prefixItems) ? node.prefixItems[i] : node.items
        if (itemSchema && typeof itemSchema === 'object') validate(item, itemSchema as Schema, `${path}[${i}]`)
      })
      break
    }
    case 'object': {
      if (!value || typeof value !== 'object' || Array.isArray(value)) fail()
      const object = value as Record<string, unknown>
      if (Array.isArray(node.required) && node.required.some(key => !(String(key) in object))) fail()
      const properties = (node.properties ?? {}) as Record<string, Schema>
      for (const [key, item] of Object.entries(object)) {
        if (node.propertyNames) validate(key, node.propertyNames as Schema, path)
        if (key in properties) validate(item, properties[key], `${path}.${key}`)
        else if (node.additionalProperties === false) fail()
        else if (node.additionalProperties && typeof node.additionalProperties === 'object') validate(item, node.additionalProperties as Schema, `${path}.*`)
      }
      break
    }
  }
}

export function parseCanonical(value: unknown): CanonicalDecisionResponse {
  validate(value, schema as unknown as Schema)
  const result = value as CanonicalDecisionResponse
  if (!result.capabilities || result.schema_version !== 'sepsis-decision-summary.v1' || result.research_only !== true || result.clinical_use !== false) throw new Error('Canonical research metadata is unavailable.')
  if (!result.capabilities.next_hour_forecast || !result.capabilities.learned_hybrid || !result.capabilities.clinician_review ||
      !Number.isFinite(result.configured_parameters.reliability_exposure)) throw new Error('Canonical capability or controller metadata is unavailable.')
  let previous: WireState | null = null
  for (const record of result.records) {
    if (record.current.icu_hour <= (previous?.icu_hour ?? 0) || record.previous?.icu_hour !== previous?.icu_hour) throw new Error('Canonical observation history is inconsistent.')
    if (previous && record.change.elapsed_hours !== record.current.icu_hour - previous.icu_hour) throw new Error('Canonical elapsed time is inconsistent.')
    previous = record.current
  }
  return result
}

/** Normalize field names for the existing React panels; never re-run inference or
 * reconstruct a change term. Only weighted expert contributions are display arithmetic.
 * Keep the wire payload alongside this view for exact provenance and diagnostics.
 */
export function canonicalView(wire: CanonicalDecisionResponse): TrajectoryResponse {
  const state = (value: WireState): DecisionState => ({
    ...value, static_contribution: value.static_weight * value.static_probability,
    temporal_contribution: value.temporal_probability === null ? null : value.temporal_weight * value.temporal_probability,
  })
  const unavailable = (reason: string) => ({ status: 'unavailable' as const, value: null, reason })
  return {
    episode_id: wire.episode_id, research_only: true, clinical_use: false,
    wire_schema_version: wire.schema_version,
    decision_summaries: wire.records.map(record => ({
      schema_version: 'decision-summary.v0.6', episode_id: wire.episode_id,
      current: state(record.current), previous: record.previous ? state(record.previous) : null,
      elapsed_hours: record.change.elapsed_hours,
      change: {
        status: record.change.status === 'exact' ? 'available' : 'unavailable',
        reason: record.change.status === 'exact' ? null : record.change.status === 'first_observation' ? 'first_observation' : 'secondary_probability_unavailable',
        interval: 'since_previous_observed_hour', probability_delta: record.change.total,
        static_delta: record.change.static_movement,
        disagreement_change_at_previous_allocation: record.change.disagreement_movement,
        reallocation_at_current_disagreement: record.change.controller_reallocation,
        reconstruction_residual: record.change.identity_residual,
        interpretation: 'arithmetic_attribution_not_medical_causality',
      },
      diagnostics: {
        signed_model_disagreement: record.diagnostics.expert_gap,
        absolute_model_disagreement: record.diagnostics.absolute_expert_gap,
        final_predictive_entropy_nats: record.diagnostics.predictive_entropy_nats,
        entropy_interpretation: 'predictive_distribution_not_reliability',
      },
      data_quality: {
        status: record.measurements === null ? 'unavailable' : 'available',
        reason: record.measurements === null ? 'measurements_redacted_or_not_supplied' : null,
        source_availability_verified: false,
        measurements: Object.fromEntries(Object.entries(record.measurements ?? {}).filter(([, m]) => m.freshness_status !== 'not_tracked').map(([name, m]) => [name, { observed: !m.missing, age_hours: m.age_hours }])),
        missing_count: record.information_quality.missing_vitals.length,
        missing_vitals: record.information_quality.missing_vitals,
        missing_fraction: 1 - record.information_quality.vital_completeness,
        engineered_feature_count: null, engineered_missing_count: null,
      },
      controller: {
        policy_id: record.controller.policy, allocation_basis: record.controller.evidence,
        empirically_validated: false, exposure: wire.configured_parameters.reliability_exposure,
        exposure_empirically_calibrated: false, anticipatory_proposal: null, estimated_gain: null,
        fallback_active: record.controller.emergency_fallback,
        fallback_reason: record.controller.emergency_fallback ? record.controller.reason : null,
        emergency_overrides_step_limit: record.controller.step_limit_overridden,
        configured_constraints: wire.configured_parameters,
        constraint_trace: record.controller.active_constraints.length ? record.controller.active_constraints.join(', ') : 'none_binding',
        active_constraints: record.controller.active_constraints,
        reactive: unavailable('no_verified_outcome_feedback'), covariance: unavailable('not_integrated_in_this_policy'),
      },
      provenance: {
        dataset: wire.dataset, target: wire.target_semantics, model: wire.model,
        artifact_sha256: wire.artifact_sha256, artifact_status: wire.artifact_integrity,
        features: wire.provenance, feature_status: 'runtime_generated_source_availability_unverified',
      },
      evaluation: { ...unavailable('no_recorded_evaluation_bound_to_this_model'), recorded_summary_path: '/api/v1/research/evaluation-summary' },
      next_computational_step: 'await_next_valid_observation_then_update_causal_features_and_reevaluate',
      next_hour_forecast: unavailable(wire.capabilities.next_hour_forecast.reason),
      learned_hybrid: unavailable(wire.capabilities.learned_hybrid.reason),
      workflow_recommendation: unavailable(wire.capabilities.clinician_review.reason),
      research_only: true, clinical_use: false,
    })),
  }
}
