import assert from 'node:assert/strict'
import test from 'node:test'
import { canonicalView, parseCanonical } from '../src/api/canonical.ts'
import { syntheticDecisionEpisode } from '../src/data/fixtures.ts'
import { observationEvidence } from '../src/lib/observations.ts'

test('canonical adapter preserves every served prediction and exact signed change term', () => {
  const { response, trajectory } = syntheticDecisionEpisode()
  assert.equal(trajectory.wire_schema_version, 'sepsis-decision-summary.v1')
  response.records.forEach((record, index) => {
    const summary = trajectory.decision_summaries[index]
    assert.equal(summary.current.final_probability, record.current.final_probability)
    assert.equal(summary.current.temporal_weight, record.current.temporal_weight)
    assert.equal(summary.change.probability_delta, record.change.total)
    assert.equal(summary.change.static_delta, record.change.static_movement)
    assert.equal(summary.change.disagreement_change_at_previous_allocation, record.change.disagreement_movement)
    assert.equal(summary.change.reallocation_at_current_disagreement, record.change.controller_reallocation)
    assert.equal(summary.change.reconstruction_residual, record.change.identity_residual)
    assert.equal(summary.diagnostics.final_predictive_entropy_nats, record.diagnostics.predictive_entropy_nats)
  })
  const last = trajectory.decision_summaries[2]
  assert.equal(last.elapsed_hours, 2)
  assert.equal(last.data_quality.measurements.HR.age_hours, 2)
  assert.equal(last.data_quality.measurements.HR.observed, false)
  assert.equal(last.provenance.artifact_sha256, null)
  assert.equal(last.next_hour_forecast.value, null)
  assert.equal(last.data_quality.engineered_feature_count, null)
})
test('supplied JSON Schema rejects malformed predictions, provenance and unsupported capabilities', () => {
  for (const value of [NaN, Infinity, -0.01, 1.01]) {
    const { response } = syntheticDecisionEpisode()
    response.records[0].current.final_probability = value
    assert.throws(() => parseCanonical(response), /response at/)
  }
  const good = syntheticDecisionEpisode().response
  assert.throws(() => parseCanonical({ ...good, schema_version: 'unsupported' }))
  assert.throws(() => parseCanonical({ ...good, artifact_sha256: 'invalid-hash' }))
  assert.throws(() => parseCanonical({ ...good, clinical_use: true }))
  assert.throws(() => parseCanonical({ ...good, records: [] }))
  assert.throws(() => parseCanonical({ ...good, capabilities: { ...good.capabilities, next_hour_forecast: { status: 'available', probability: .5 } } }))
})
test('canonical fallback preserves unavailable expert attribution instead of synthesizing terms', () => {
  const { response } = syntheticDecisionEpisode()
  const last = response.records[2]
  last.current = { ...last.current, static_weight: 1, temporal_weight: 0, temporal_probability: null, final_probability: last.current.static_probability, status: 'static_only_fallback', reason: 'secondary_inference_failed' }
  last.change = { ...last.change, status: 'partial_temporal_unavailable', total: last.current.final_probability - last.previous!.final_probability, disagreement_movement: null, controller_reallocation: null, identity_residual: null, unsupported_terms: ['disagreement_movement', 'controller_reallocation'] }
  last.controller = { ...last.controller, emergency_fallback: true, step_limit_overridden: true, mode: 'static_only_fallback', reason: 'secondary_inference_failed' }
  const summary = canonicalView(parseCanonical(response)).decision_summaries[2]
  assert.equal(summary.current.final_probability, last.current.static_probability)
  assert.equal(summary.change.disagreement_change_at_previous_allocation, null)
  assert.equal(summary.change.reallocation_at_current_disagreement, null)
  assert.equal(summary.controller.fallback_active, true)
  assert.equal(summary.controller.emergency_overrides_step_limit, true)
})
test('sanitized response stays redacted and never reconstructs raw measurements', () => {
  const { response } = syntheticDecisionEpisode()
  response.payload_kind = 'sanitized_retrospective_demo'
  response.records.forEach(record => { record.measurements = null })
  const view = canonicalView(parseCanonical(response))
  assert.equal(view.decision_summaries[2].data_quality.status, 'unavailable')
  assert.deepEqual(view.decision_summaries[2].data_quality.measurements, {})
})
test('observation evidence is prefix-causal, unit-preserving and distinguishes current from last-known', () => {
  const { request } = syntheticDecisionEpisode()
  const before = observationEvidence(request, 2)
  const hr = observationEvidence(request, 4).find(item => item.name === 'HR')!
  assert.equal(hr.value, null)
  assert.equal(hr.lastValue, 90)
  assert.equal(hr.age, 2)
  assert.equal(hr.change, null)
  assert.equal(hr.status, 'not_current')
  request.observations[2].measurements.HR = 999
  assert.deepEqual(observationEvidence(request, 2), before)
  assert.equal(before.find(item => item.name === 'HR')!.change, 10)
})
