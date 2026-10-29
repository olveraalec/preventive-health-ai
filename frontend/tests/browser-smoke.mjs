// Optional real-browser checks. Uses local Chrome + CDP; no downloaded test runtime.
import assert from 'node:assert/strict'
import { spawn } from 'node:child_process'
import { existsSync } from 'node:fs'
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { syntheticDecisionEpisode, syntheticEvaluation } from '../src/data/fixtures.ts'

const root = fileURLToPath(new URL('..', import.meta.url))
const chrome = process.env.CHROME_BIN || (process.platform === 'darwin'
  ? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
  : '/usr/bin/google-chrome')
assert.ok(existsSync(chrome), 'Set CHROME_BIN to a locally installed Chrome/Chromium executable.')
const profile = await mkdtemp(join(tmpdir(), 'preventive-browser-'))
const outputs = join(root, 'test-results')
const realApi = process.env.REAL_API === '1'
if (realApi) assert.ok(process.env.DEMO_MODEL_ARTIFACT_ID && process.env.DEMO_MODEL_VERSION, 'Real API mode requires the installed model identifiers; no model files are read.')
await mkdir(outputs, { recursive: true })
const vite = spawn(process.execPath, [join(root, 'node_modules/vite/bin/vite.js'), '--host', '127.0.0.1', '--port', '4175', '--strictPort'], { cwd: root, stdio: ['ignore', 'pipe', 'pipe'] })
let viteReady = false
vite.stdout.on('data', chunk => { if (String(chunk).includes('Local:')) viteReady = true })
let browser
let socket
const pause = ms => new Promise(resolve => setTimeout(resolve, ms))
async function until(operation, description) {
  for (let attempt = 0; attempt < 100; attempt++) {
    try { const result = await operation(); if (result) return result } catch { /* startup may still be in progress */ }
    await pause(100)
  }
  throw new Error(`Timed out: ${description}`)
}
try {
  await until(() => viteReady, 'our own local Vite startup (port 4175 must be free)')
  assert.ok((await fetch('http://127.0.0.1:4175')).ok)
  browser = spawn(chrome, [
    '--headless', '--disable-gpu', '--no-first-run', '--no-default-browser-check',
    '--disable-background-networking', '--disable-sync', '--disable-extensions',
    '--disable-component-update', '--disable-domain-reliability', '--metrics-recording-only',
    '--disable-features=OptimizationHints,MediaRouter,Translate',
    '--remote-debugging-address=127.0.0.1', '--remote-debugging-port=0',
    `--user-data-dir=${profile}`, 'about:blank',
  ], { stdio: 'ignore' })
  const port = await until(async () => (await readFile(join(profile, 'DevToolsActivePort'), 'utf8')).split('\n')[0], 'isolated Chrome startup')
  const pages = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json()
  socket = new WebSocket(pages.find(page => page.type === 'page').webSocketDebuggerUrl)
  await new Promise((resolve, reject) => { socket.addEventListener('open', resolve, { once: true }); socket.addEventListener('error', reject, { once: true }) })
  let sequence = 0
  const pending = new Map()
  let apiFailure = true
  const send = (method, params = {}) => new Promise((resolve, reject) => {
    const id = ++sequence
    pending.set(id, { resolve, reject })
    socket.send(JSON.stringify({ id, method, params }))
  })
  socket.addEventListener('message', async event => {
    const message = JSON.parse(String(event.data))
    if (message.id) {
      const callback = pending.get(message.id)
      pending.delete(message.id)
      if (message.error) callback?.reject(new Error(message.error.message))
      else callback?.resolve(message.result)
    }
    if (message.method === 'Fetch.requestPaused') {
      const { requestId, request } = message.params
      const evaluation = request.url.includes('evaluation-summary')
      const body = apiFailure ? { detail: 'synthetic test outage' } : evaluation ? syntheticEvaluation() : syntheticDecisionEpisode().response
      await send('Fetch.fulfillRequest', {
        requestId, responseCode: apiFailure ? 503 : 200,
        responseHeaders: [{ name: 'Content-Type', value: 'application/json' }],
        body: Buffer.from(JSON.stringify(body)).toString('base64'),
      })
    }
  })
  const evaluate = async expression => {
    const result = await send('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true })
    if (result.exceptionDetails) throw new Error(result.exceptionDetails.text)
    return result.result.value
  }
  await send('Page.enable')
  await send('Network.enable')
  const apiStatuses = []
  socket.addEventListener('message', event => {
    const message = JSON.parse(String(event.data))
    if (message.method === 'Network.responseReceived' && message.params.response.url.includes('/api/v1/research/')) {
      apiStatuses.push({ path: new URL(message.params.response.url).pathname, status: message.params.response.status })
    }
  })
  if (!realApi) await send('Fetch.enable', { patterns: [{ urlPattern: '*/api/v1/research/*' }] })
  await send('Emulation.setDeviceMetricsOverride', { width: 1440, height: 1100, deviceScaleFactor: 1, mobile: false })
  await send('Page.navigate', { url: 'http://127.0.0.1:4175' })
  await until(() => evaluate('document.querySelector(".snapshot-card") !== null'), 'React render')
  assert.ok(await evaluate('document.body.textContent.includes("Synthetic example")'))
  assert.ok(await evaluate('document.body.textContent.includes("Not results from the PhysioNet cohort")'))
  assert.equal(await evaluate('document.querySelectorAll("iframe").length'), 0)
  await evaluate('document.querySelectorAll(".hour")[0].click()')
  await until(() => evaluate('document.querySelector(".prediction-comparison").textContent.includes("No previous")'), 'first observation selection')
  await evaluate('document.querySelectorAll(".hour")[2].click()')
  await until(() => evaluate('document.querySelector(".change-total").textContent.includes("+2.25 pp")'), 'current observation selection')
  assert.ok(await evaluate('document.documentElement.scrollWidth <= innerWidth'))
  await writeFile(join(outputs, 'desktop.png'), Buffer.from((await send('Page.captureScreenshot', { captureBeyondViewport: true })).data, 'base64'))
  await writeFile(join(outputs, 'desktop-viewport.png'), Buffer.from((await send('Page.captureScreenshot', { captureBeyondViewport: false })).data, 'base64'))
  await evaluate('document.querySelectorAll("nav a")[2].click()')
  await until(() => evaluate('document.querySelector(".research-audit").getBoundingClientRect().height > 0'), 'research perspective')
  await evaluate('document.querySelector("select").value="positive_patient_mean_bernoulli_log_loss"; document.querySelector("select").dispatchEvent(new Event("change", { bubbles:true }))')
  await until(() => evaluate('document.querySelector(".evaluation-table-panel table").textContent.includes("0.394507")'), 'retrospective metric selection')
  await send('Emulation.setDeviceMetricsOverride', { width: 390, height: 844, deviceScaleFactor: 1, mobile: true })
  await evaluate('window.scrollTo(0,0)')
  await pause(200)
  assert.ok(await evaluate('document.documentElement.scrollWidth <= innerWidth'), 'mobile layout must not overflow horizontally')
  await writeFile(join(outputs, 'mobile.png'), Buffer.from((await send('Page.captureScreenshot', { captureBeyondViewport: true })).data, 'base64'))
  await writeFile(join(outputs, 'mobile-viewport.png'), Buffer.from((await send('Page.captureScreenshot', { captureBeyondViewport: false })).data, 'base64'))
  await evaluate('document.querySelector(".toolbar-actions .text-link").click()')
  await until(() => evaluate('document.querySelector(".api-connection details").open'), 'API connection disclosure')
  if (realApi) {
    await evaluate('document.querySelectorAll(".model-identifiers input")[0].focus(); document.querySelectorAll(".model-identifiers input")[0].select()')
    await send('Input.insertText', { text: process.env.DEMO_MODEL_ARTIFACT_ID })
    await evaluate('document.querySelectorAll(".model-identifiers input")[1].focus(); document.querySelectorAll(".model-identifiers input")[1].select()')
    await send('Input.insertText', { text: process.env.DEMO_MODEL_VERSION })
    await evaluate('document.querySelector(".model-identifiers button").click()')
    await until(() => evaluate('Array.from(document.querySelectorAll(".source-badge")).some(node => node.textContent.includes("Synthetic inputs · real API outputs"))'), 'real FastAPI inference with synthetic observations')
    await until(() => evaluate('document.querySelector(".api-form-content").textContent.includes("Research response loaded")'), 'real API completion')
    assert.ok(apiStatuses.some(response => response.path.endsWith('/trajectory/decision-summary') && response.status === 200), 'must receive a real HTTP 200 canonical response')
    await send('Emulation.setDeviceMetricsOverride', { width: 1440, height: 1100, deviceScaleFactor: 1, mobile: false })
    await evaluate('document.querySelector("#patient-monitoring").scrollIntoView()')
    await pause(300)
    await writeFile(join(outputs, 'real-api-journey.png'), Buffer.from((await send('Page.captureScreenshot', { captureBeyondViewport: true })).data, 'base64'))
    console.log('Real React → Vite proxy → FastAPI connection passed. Synthetic inputs, unchanged installed model, canonical response validated by the browser client.')
    console.log(JSON.stringify({ apiStatuses, selectedOutput: await evaluate('document.querySelector(".trajectory-stats").textContent') }))
  } else {
  const request = JSON.stringify(syntheticDecisionEpisode().request)
  await evaluate('document.querySelector("textarea").focus()')
  await send('Input.insertText', { text: request })
  await evaluate('document.querySelector("button[type=submit]").click()')
  await until(() => evaluate('document.querySelector(".error-box")?.textContent.includes("Still showing the explicitly labeled synthetic example")'), 'API error without silent data substitution')
  apiFailure = false
  await evaluate('document.querySelector("button[type=submit]").click()')
  await until(() => evaluate('document.querySelector(".source-badge.api") !== null'), 'successful mocked API connection')
  await until(() => evaluate('document.querySelector("textarea").value === ""'), 'request cleared after success')
  await evaluate('document.querySelector(".toolbar-actions button").click()')
  await until(() => evaluate('document.querySelector(".source-badge.synthetic") !== null'), 'explicit synthetic reset')
  assert.equal(await evaluate('localStorage.length + sessionStorage.length'), 0)
  console.log('Browser checks passed: desktop/mobile rendering, hour and metric selection, API error/retry/reset, no iframe or browser persistence. All API calls were intercepted with synthetic fixtures.')
  console.log('Screenshots: frontend/test-results/desktop.png and mobile.png')
  }
} finally {
  socket?.close()
  browser?.kill()
  vite.kill()
  await pause(300)
  await rm(profile, { recursive: true, force: true })
}
