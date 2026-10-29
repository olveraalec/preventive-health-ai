import assert from 'node:assert/strict'
import test from 'node:test'
import { canonicalView } from '../src/api/canonical.ts'
import { MAX_RETROSPECTIVE_BYTES, parseRetrospective, readRetrospectiveFile } from '../src/api/retrospective.ts'
import { syntheticDecisionEpisode } from '../src/data/fixtures.ts'
import { retainedSource, sourceBadge, sourceDescription } from '../src/lib/source.ts'

function redactedSyntheticExport() {
  const response = syntheticDecisionEpisode().response
  response.payload_kind = 'sanitized_retrospective_demo'
  response.artifact_integrity = 'checksum_bound_manifest'
  response.artifact_sha256 = 'a'.repeat(64) // Synthetic provenance, never a real artifact.
  response.records.forEach(record => { record.measurements = null; record.log_odds_contributions = null })
  return response
}

test('sanitized import preserves predictions, allocation and exact attribution without raw reconstruction', () => {
  const response = parseRetrospective(redactedSyntheticExport())
  const view = canonicalView(response)
  for (const [index, record] of response.records.entries()) {
    const summary = view.decision_summaries[index]
    assert.equal(summary.current.final_probability, record.current.final_probability)
    assert.equal(summary.current.temporal_weight, record.current.temporal_weight)
    assert.equal(summary.change.reallocation_at_current_disagreement, record.change.controller_reallocation)
    assert.equal(summary.change.static_delta, record.change.static_movement)
    assert.equal(summary.change.reconstruction_residual, record.change.identity_residual)
    assert.equal(summary.data_quality.status, 'unavailable')
    assert.deepEqual(summary.data_quality.measurements, {})
    assert.deepEqual(summary.data_quality.missing_vitals, record.information_quality.missing_vitals)
  }
})

test('synthetic examples, ordinary API payloads and declared-only provenance cannot become retrospective imports', () => {
  const response = syntheticDecisionEpisode().response
  assert.throws(() => parseRetrospective(response), /sanitized retrospective/)
  response.payload_kind = 'api'
  assert.throws(() => parseRetrospective(response), /sanitized retrospective/)
  const declared = redactedSyntheticExport()
  declared.artifact_integrity = 'declared_only'
  assert.throws(() => parseRetrospective(declared), /checksum-bound/)
  const absentHash = redactedSyntheticExport()
  absentHash.artifact_sha256 = null
  assert.throws(() => parseRetrospective(absentHash), /checksum-bound/)
})

test('rejects unredacted values and explanation vectors even when the file declares sanitized provenance', () => {
  const observations = redactedSyntheticExport()
  observations.records[0].measurements = syntheticDecisionEpisode().response.records[0].measurements
  assert.throws(() => parseRetrospective(observations), /redacted measurements/)
  const contributions = redactedSyntheticExport()
  contributions.records[0].log_odds_contributions = {}
  assert.throws(() => parseRetrospective(contributions), /redacted measurements/)
})

test('file reader rejects empty, oversized, unreadable and invalid JSON without echoing source text', async () => {
  const text = async () => JSON.stringify(redactedSyntheticExport())
  await assert.rejects(readRetrospectiveFile({ size: 0, text }), /nonempty/)
  await assert.rejects(readRetrospectiveFile({ size: MAX_RETROSPECTIVE_BYTES + 1, text }), /smaller than/)
  await assert.rejects(readRetrospectiveFile({ size: 100, text: async () => { throw new Error('private-source-text') } }), /could not be read/)
  await assert.rejects(readRetrospectiveFile({ size: 100, text: async () => 'private-source-text' }), error => error instanceof Error && error.message.includes('not valid JSON') && !error.message.includes('private-source-text'))
  assert.equal((await readRetrospectiveFile({ size: 100, text })).records.length, 3)
})

test('source labels and failure messages distinguish retrospective imports from fixtures and API responses', () => {
  assert.match(sourceBadge['retrospective-export'], /Retrospective model outputs/)
  assert.match(sourceDescription['retrospective-export'], /does not independently authenticate/)
  assert.match(retainedSource['retrospective-export'], /last successfully imported/)
  assert.notEqual(sourceBadge['retrospective-export'], sourceBadge.synthetic)
  assert.notEqual(sourceBadge['retrospective-export'], sourceBadge['api-synthetic-inputs'])
})
