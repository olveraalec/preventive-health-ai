/** Real HTTP check. No mock transport, model copy, patient data or Python changes. */
import assert from 'node:assert/strict'
import { mkdir, writeFile } from 'node:fs/promises'
import { createResearchClient } from '../src/api/client.ts'
import { SYNTHETIC_REQUEST } from '../src/fixtures/synthetic.ts'
import { observationEvidence } from '../src/lib/observations.ts'

const base = process.env.LIVE_API_URL || 'http://127.0.0.1:8003'
assert.ok(/^https?:\/\/(127\.0\.0\.1|localhost)(:\d+)?$/.test(base), 'This local smoke test only targets loopback services.')
const health = await fetch(`${base}/health`)
assert.equal(health.status, 200, 'FastAPI liveness must pass')
assert.ok(process.env.DEMO_MODEL_ARTIFACT_ID && process.env.DEMO_MODEL_VERSION, 'Supply DEMO_MODEL_ARTIFACT_ID and DEMO_MODEL_VERSION from the installed manifest. Do not read/copy model files here.')
const statuses = []
const fetcher = async (url, init) => {
  const response = await fetch(url, init)
  statuses.push({ method: init?.method ?? 'GET', path: new URL(String(url)).pathname, status: response.status })
  return response
}
const request = structuredClone(SYNTHETIC_REQUEST)
request.model = { artifact_id: process.env.DEMO_MODEL_ARTIFACT_ID, model_version: process.env.DEMO_MODEL_VERSION }
const client = createResearchClient(base, fetcher)
const response = await client.decisionSummary(request)
const prefixRequest = { ...request, observations: request.observations.slice(0, 5) }
const prefix = await client.decisionSummary(prefixRequest)
assert.deepEqual(prefix.records, response.records.slice(0, 5), 'Real API must preserve prefix causality')
assert.equal(response.records.length, 24)
assert.equal(response.schema_version, 'sepsis-decision-summary.v1')
for (const record of response.records) {
  for (const signal of observationEvidence(request, record.current.icu_hour)) {
    assert.equal(signal.value, record.measurements[signal.name].current_value, 'measurement value parity')
    assert.equal(signal.age, record.measurements[signal.name].age_hours, 'measurement age parity')
    assert.equal(signal.lastValue, record.measurements[signal.name].last_value, 'last-observed value parity')
  }
}
const residuals = response.records.flatMap(record => record.change.identity_residual === null ? [] : [Math.abs(record.change.identity_residual)])
assert.ok(Math.max(...residuals, 0) <= 1e-12)
let evaluation = 'unavailable'
try { await client.evaluation(); evaluation = 'available_recorded_synthetic_only' } catch { /* optional endpoint can be absent; do not invent model metrics */ }
const last = response.records.at(-1)
const evidence = {
  transport: 'real_http_no_mocks', inputs: '24_hour_synthetic_observations',
  schema_version: response.schema_version, records: response.records.length,
  artifact_integrity: response.artifact_integrity, prefix_causality: 'passed', measurement_parity: 'passed',
  max_identity_residual: Math.max(...residuals, 0), statuses,
  latest: { icu_hour: last.current.icu_hour, static_probability: last.current.static_probability,
    temporal_probability: last.current.temporal_probability, final_probability: last.current.final_probability,
    temporal_weight: last.current.temporal_weight, entropy_nats: last.diagnostics.predictive_entropy_nats,
    fallback: last.controller.emergency_fallback },
  evaluation, research_only: response.research_only, clinical_use: response.clinical_use,
}
await mkdir(new URL('../test-results/', import.meta.url), { recursive: true })
await writeFile(new URL('../test-results/real-api-summary.json', import.meta.url), JSON.stringify(evidence, null, 2) + '\n')
console.log(JSON.stringify(evidence, null, 2))
