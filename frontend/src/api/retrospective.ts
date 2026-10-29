import { parseCanonical } from './canonical.ts'
import type { CanonicalDecisionResponse } from './canonical.ts'

export const MAX_RETROSPECTIVE_BYTES = 5 * 1024 * 1024

/** Accept only the backend's explicitly sanitized export. This validates its
 * declared structure, not file authenticity or historical model training.
 */
export function parseRetrospective(value: unknown): CanonicalDecisionResponse {
  const response = parseCanonical(value)
  if (response.payload_kind !== 'sanitized_retrospective_demo') {
    throw new Error('Choose a sanitized retrospective export, not an API response or synthetic example.')
  }
  if (response.artifact_integrity !== 'checksum_bound_manifest' || response.artifact_sha256 === null) {
    throw new Error('The retrospective export must declare a checksum-bound model manifest.')
  }
  if (response.records.some(record => record.measurements !== null || record.log_odds_contributions !== null)) {
    throw new Error('This loader requires redacted measurements and log-odds vectors. Choose the sanitized export.')
  }
  return response
}

export async function readRetrospectiveFile(file: Pick<File, 'size' | 'text'>) {
  if (file.size === 0 || file.size > MAX_RETROSPECTIVE_BYTES) {
    throw new Error('Choose a nonempty retrospective JSON export smaller than 5 MiB.')
  }
  let text: string
  try { text = await file.text() } catch { throw new Error('The local retrospective file could not be read.') }
  let value: unknown
  try { value = JSON.parse(text) } catch { throw new Error('The local retrospective file is not valid JSON.') }
  return parseRetrospective(value)
}
