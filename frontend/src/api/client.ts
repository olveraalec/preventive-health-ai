import { parseEvaluation } from './contracts.ts'
import { canonicalView, parseCanonical } from './canonical.ts'
import type { RawTrajectoryRequest } from './contracts.ts'

/** No automatic synthetic substitution, patient logging or browser persistence. */
export function createResearchClient(baseUrl = '', fetcher: typeof fetch = fetch) {
  if (baseUrl && !baseUrl.startsWith('/') && !/^https?:\/\//i.test(baseUrl)) throw new Error('API base URL must be HTTP(S) or an absolute local path.')
  if (baseUrl.startsWith('//')) throw new Error('Protocol-relative API URLs are not supported.')
  if (/^https?:\/\//i.test(baseUrl)) {
    const url = new URL(baseUrl)
    if (url.username || url.password || url.search || url.hash) throw new Error('API base URL must not contain credentials, queries or fragments.')
  }
  const base = baseUrl.replace(/\/$/, '')
  async function request(path: string, init: RequestInit = {}) {
    const timeout = AbortSignal.timeout(20_000)
    const signal = init.signal ? AbortSignal.any([init.signal, timeout]) : timeout
    let response: Response
    try {
      response = await fetcher(`${base}${path}`, { ...init, signal, credentials: 'omit', cache: 'no-store' })
    } catch (error) {
      if (signal.aborted) throw new Error('Request stopped or timed out. Retry when the research API is available.')
      throw new Error('Cannot reach the research API. Check its address and local connection.', { cause: error })
    }
    // Do not echo response bodies: plugin errors can contain sensitive details.
    if (!response.ok) {
      const message = response.status === 503 ? 'Research model or API unavailable.'
        : response.status === 422 ? 'Trajectory or model metadata rejected. Check the request against the installed artifact.'
        : `Research API returned HTTP ${response.status}.`
      throw new Error(message)
    }
    try { return await response.json() as unknown } catch { throw new Error('Research API returned invalid JSON.') }
  }
  async function decisionSummary(payload: RawTrajectoryRequest, signal?: AbortSignal, includeContributions = false) {
      const result = parseCanonical(await request(`/api/v1/research/sepsis/trajectory/decision-summary${includeContributions ? '?include_contributions=true' : ''}`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload), signal,
      }))
      if (result.episode_id !== payload.episode_id || result.records.length !== payload.observations.length ||
        result.model.artifact_id !== payload.model.artifact_id || result.model.model_version !== payload.model.model_version ||
        result.records.some((record, index) => record.current.icu_hour !== payload.observations[index].icu_hour)) {
        throw new Error('Research response does not match the requested episode, observation history or model.')
      }
      return result
  }
  return {
    decisionSummary,
    async trajectory(payload: RawTrajectoryRequest, signal?: AbortSignal) {
      return canonicalView(await decisionSummary(payload, signal))
    },
    async evaluation(signal?: AbortSignal) {
      return parseEvaluation(await request('/api/v1/research/evaluation-summary', { signal }))
    },
  }
}

export function explorerUrl(value: string | undefined): string | null {
  if (!value) return null
  try {
    const url = new URL(value)
    return ['http:', 'https:'].includes(url.protocol) && !url.username && !url.password ? url.href : null
  } catch { return null }
}
