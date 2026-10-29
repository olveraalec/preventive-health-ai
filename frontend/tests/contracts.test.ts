import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import { assertDecisionSummary, parseEvaluation, parseRawRequest, parseTrajectory } from '../src/api/contracts.ts'
import { syntheticEvaluation, syntheticTrajectory } from '../src/data/fixtures.ts'
import { percent, points } from '../src/lib/format.ts'

test('bundled fixtures match the reviewed backend examples verbatim', () => {
  for (const [local, source] of [
    ['decision-summary.synthetic.json', 'decision-summary.v0.6.synthetic.json'],
    ['evaluation-summary.synthetic.json', 'evaluation-summary.v0.6.synthetic.json'],
  ]) {
    assert.deepEqual(JSON.parse(readFileSync(new URL(`../src/data/${local}`, import.meta.url), 'utf8')),
      JSON.parse(readFileSync(new URL(`../../docs/examples/${source}`, import.meta.url), 'utf8')))
  }
})
test('synthetic timeline uses exactly the two documented observed states and gap', () => {
  const result = syntheticTrajectory()
  const [first, current] = result.decision_summaries
  assert.deepEqual(first.current, current.previous)
  assert.equal(first.previous, null)
  assert.equal(first.change.status, 'unavailable')
  assert.equal(current.elapsed_hours, 3)
  assert.deepEqual(result.decision_summaries.map(d => d.current.icu_hour), [1, 4])
  assert.equal(current.change.reconstruction_residual, 0)
  assert.equal(current.next_hour_forecast.value, null)
  first.current.final_probability = 0
  assert.notEqual(syntheticTrajectory().decision_summaries[0].current.final_probability, 0)
})
test('rejects schema drift, invalid probabilities and missing capabilities', () => {
  for (const change of [
    (v: Record<string, unknown>) => { v.schema_version = 'decision-summary.v0.7' },
    (v: Record<string, unknown>) => { v.next_hour_forecast = { status: 'available', value: 0.5 } },
    (v: Record<string, unknown>) => { v.clinical_use = true },
    (v: Record<string, unknown>) => { delete v.current },
  ]) {
    const summary = structuredClone(syntheticTrajectory().decision_summaries[1]) as unknown as Record<string, unknown>
    change(summary)
    assert.throws(() => assertDecisionSummary(summary))
  }
  for (const value of [NaN, Infinity, -0.1, 1.1]) {
    const summary = syntheticTrajectory().decision_summaries[1]
    summary.current.final_probability = value
    assert.throws(() => assertDecisionSummary(summary))
  }
})
test('rejects future previous hours, mixed episodes, missing summaries and unordered histories', () => {
  const future = syntheticTrajectory()
  future.decision_summaries[1].previous!.icu_hour = 8
  assert.throws(() => parseTrajectory(future))
  const mixed = syntheticTrajectory()
  mixed.decision_summaries[1].episode_id = 'another-synthetic-episode'
  assert.throws(() => parseTrajectory(mixed))
  const reversed = syntheticTrajectory()
  reversed.decision_summaries.reverse()
  assert.throws(() => parseTrajectory(reversed))
  assert.throws(() => parseTrajectory({ episode_id: 'synthetic', decision_summaries: [], research_only: true, clinical_use: false }))
})
test('recorded evaluation remains explicitly synthetic and rejects model-performance relabeling', () => {
  const evaluation = syntheticEvaluation()
  assert.equal(evaluation.installed_sepsis_model_evaluation, 'unavailable')
  assert.equal(evaluation.experiments[0].policies.length, 5)
  assert.equal(evaluation.experiments[0].comparison.metric.value, 0.006495)
  const bad = { ...evaluation, experiments: [{ ...evaluation.experiments[0], dataset: 'sepsis2019' }] }
  assert.throws(() => parseEvaluation(bad))
})
test('validates raw requests without labels or future observations', () => {
  const measurements = Object.fromEntries(['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'Age', 'Gender', 'HospAdmTime', 'Unit1', 'Unit2'].map(key => [key, null]))
  const request = { dataset: 'sepsis2019', episode_id: 'synthetic', model: { artifact_id: 'synthetic', model_version: '1' }, observations: [{ episode_id: 'synthetic', icu_hour: 1, measurements }] }
  assert.equal(parseRawRequest(JSON.stringify(request)).observations.length, 1)
  assert.throws(() => parseRawRequest('{'))
  for (const field of ['SepsisLabel', 'y_true', 'patient_id', 'split']) {
    const bad = structuredClone(request)
    bad.observations[0].measurements[field] = 1 as never
    assert.throws(() => parseRawRequest(JSON.stringify(bad)))
  }
})
test('formatting preserves unavailable values and percentage-point units', () => {
  assert.equal(percent(null), 'Unavailable')
  assert.equal(points(null), 'Unavailable')
  assert.equal(percent(0), '0.00%')
  assert.equal(points(0.13672353553424987), '+13.67 pp')
})
