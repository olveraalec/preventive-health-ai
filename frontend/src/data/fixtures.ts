import decision from './decision-summary.synthetic.json' with { type: 'json' }
import evaluation from './evaluation-summary.synthetic.json' with { type: 'json' }
import canonical from './sepsis_decision_summary.synthetic.json' with { type: 'json' }
import { assertDecisionSummary, parseEvaluation, parseTrajectory } from '../api/contracts.ts'
import { parseRawRequest } from '../api/contracts.ts'
import { canonicalView, parseCanonical } from '../api/canonical.ts'

/** Copied verbatim from docs/examples at backend commit 8652c21. */
export function syntheticTrajectory() {
  const current: unknown = structuredClone(decision)
  assertDecisionSummary(current)
  const first = structuredClone(current)
  first.current = structuredClone(current.previous!)
  first.previous = null
  first.elapsed_hours = null
  first.controller.anticipatory_proposal = null
  first.controller.estimated_gain = null
  first.change = {
    status: 'unavailable', reason: 'first_observation', interval: 'since_previous_observed_hour',
    probability_delta: null, static_delta: null, disagreement_change_at_previous_allocation: null,
    reallocation_at_current_disagreement: null, reconstruction_residual: null,
    interpretation: 'arithmetic_attribution_not_medical_causality',
  }
  return parseTrajectory({
    episode_id: current.episode_id, decision_summaries: [first, current], research_only: true, clinical_use: false,
  })
}
export const syntheticEvaluation = () => parseEvaluation(structuredClone(evaluation))

/** Authoritative three-observation fixture, including real missingness fields.
 * Fixed demonstration predictors, not installed-model or patient results.
 */
export function syntheticDecisionEpisode() {
  const example = structuredClone(canonical)
  const request = parseRawRequest(JSON.stringify(example.request))
  const response = parseCanonical(example.response)
  return { request, response, trajectory: canonicalView(response) }
}
