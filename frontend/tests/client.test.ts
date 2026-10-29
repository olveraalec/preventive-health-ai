import assert from 'node:assert/strict'
import test from 'node:test'
import { createResearchClient, explorerUrl } from '../src/api/client.ts'
import { syntheticDecisionEpisode, syntheticEvaluation } from '../src/data/fixtures.ts'

test('evaluation fetch uses configured URL without credentials and validates output', async () => {
  let calls = 0
  const fetcher: typeof fetch = async (url, init) => {
    calls++
    assert.equal(url, 'http://127.0.0.1:8000/api/v1/research/evaluation-summary')
    assert.equal(init?.credentials, 'omit')
    assert.equal(init?.cache, 'no-store')
    assert.ok(init?.signal)
    return Response.json(syntheticEvaluation())
  }
  const result = await createResearchClient('http://127.0.0.1:8000/', fetcher).evaluation()
  assert.equal(result.live_labels_used, false)
  assert.equal(calls, 1)
})
test('raw trajectory request uses the canonical decision-summary route and validates wire response', async () => {
  const { request, response } = syntheticDecisionEpisode()
  const fetcher: typeof fetch = async (url, init) => {
    assert.equal(url, '/api/v1/research/sepsis/trajectory/decision-summary')
    assert.equal(init?.method, 'POST')
    assert.deepEqual(JSON.parse(String(init?.body)), request)
    return Response.json(response)
  }
  assert.equal((await createResearchClient('', fetcher).trajectory(request)).decision_summaries.length, 3)
  await assert.rejects(createResearchClient('', async () => Response.json(response)).trajectory({ ...request, episode_id: 'other-synthetic' }), /does not match/)
  await assert.rejects(createResearchClient('', async () => Response.json(response)).trajectory({ ...request, observations: [] }), /does not match/)
  const contributions: typeof fetch = async url => {
    assert.equal(url, '/api/v1/research/sepsis/trajectory/decision-summary?include_contributions=true')
    return Response.json(response)
  }
  assert.equal((await createResearchClient('', contributions).decisionSummary(request, undefined, true)).schema_version, 'sepsis-decision-summary.v1')
})
test('HTTP failures never leak response text or silently substitute synthetic success', async () => {
  for (const status of [422, 503, 500]) {
    const client = createResearchClient('', async () => new Response('sensitive-server-details', { status }))
    await assert.rejects(client.evaluation(), error => error instanceof Error && !error.message.includes('sensitive-server-details'))
  }
})
test('network, invalid JSON and malformed schema failures are explicit', async () => {
  await assert.rejects(createResearchClient('', async () => { throw new TypeError('offline') }).evaluation(), /Cannot reach/)
  await assert.rejects(createResearchClient('', async () => new Response('not-json')).evaluation(), /invalid JSON/)
  await assert.rejects(createResearchClient('', async () => Response.json({})).evaluation(), /malformed/)
})
test('aborted requests report cancellation and never return fixture data', async () => {
  const controller = new AbortController()
  controller.abort()
  const client = createResearchClient('', async (_url, init) => {
    assert.ok(init?.signal?.aborted)
    throw new DOMException('stopped', 'AbortError')
  })
  await assert.rejects(client.evaluation(controller.signal), /stopped or timed out/)
})
test('only explicit HTTP(S) explorer URLs are linked and invalid API URLs fail', () => {
  assert.equal(explorerUrl('http://127.0.0.1:8501'), 'http://127.0.0.1:8501/')
  for (const value of [undefined, '', 'javascript:alert(1)', 'data:text/html,no', '//example.com', 'https://user:secret@example.com']) assert.equal(explorerUrl(value), null)
  assert.throws(() => createResearchClient('javascript:alert(1)'))
  assert.throws(() => createResearchClient('//example.com'))
  assert.throws(() => createResearchClient('https://user:secret@example.com'))
})
