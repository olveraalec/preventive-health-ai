import { useEffect, useRef, useState } from 'react'
import { createResearchClient } from '../api/client.ts'
import { parseRawRequest } from '../api/contracts.ts'
import type { RawTrajectoryRequest } from '../api/contracts.ts'
import { syntheticDecisionEpisode, syntheticEvaluation } from '../data/fixtures.ts'

export function useResearchWorkspace() {
  const [trajectory, setTrajectory] = useState(() => syntheticDecisionEpisode().trajectory)
  const [evaluation, setEvaluation] = useState(syntheticEvaluation)
  const [source, setSource] = useState<'synthetic' | 'api' | 'api-synthetic-inputs'>('synthetic')
  const [observationRequest, setObservationRequest] = useState<RawTrajectoryRequest | null>(() => syntheticDecisionEpisode().request)
  const [selectedIndex, setSelectedIndex] = useState(2)
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle')
  const [error, setError] = useState<string | null>(null)
  const [evaluationNotice, setEvaluationNotice] = useState<string | null>(null)
  const active = useRef<AbortController | null>(null)
  useEffect(() => () => active.current?.abort(), [])
  function useSynthetic() {
    active.current?.abort()
    const example = syntheticDecisionEpisode()
    setTrajectory(example.trajectory); setEvaluation(syntheticEvaluation())
    setSource('synthetic'); setObservationRequest(example.request); setSelectedIndex(2); setStatus('idle')
    setError(null); setEvaluationNotice(null)
  }
  async function load(text: string, syntheticInputs = false) {
    active.current?.abort()
    const controller = new AbortController()
    active.current = controller
    setStatus('loading'); setError(null)
    try {
      const request = parseRawRequest(text)
      const client = createResearchClient(import.meta.env.VITE_API_BASE_URL ?? '')
      const response = await client.trajectory(request, controller.signal)
      if (controller.signal.aborted) return false
      setTrajectory(response); setObservationRequest(request)
      setSelectedIndex(response.decision_summaries.length - 1)
      setSource(syntheticInputs ? 'api-synthetic-inputs' : 'api')
      try {
        const result = await client.evaluation(controller.signal)
        if (controller.signal.aborted) return false
        setEvaluation(result); setEvaluationNotice(null)
      } catch {
        if (controller.signal.aborted) return false
        setEvaluation(syntheticEvaluation())
        setEvaluationNotice('The evaluation API is unavailable. Displaying the bundled, document-reported synthetic study; it is not evaluation of this episode or model.')
      }
      setStatus('success')
      return true
    } catch (failure) {
      if (controller.signal.aborted) return false
      setError(failure instanceof Error ? failure.message : 'The research request failed.')
      setStatus('error')
      return false
    }
  }
  return { trajectory, evaluation, source, observationRequest, selectedIndex, setSelectedIndex, status, error, evaluationNotice, useSynthetic, load }
}
