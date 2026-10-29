import assert from 'node:assert/strict'

/** Verify the actual configured external link and Streamlit pages; no mock service. */
export async function verifyExplorer({ send, evaluate, port, until }) {
  const expected = new URL(process.env.EXPLORER_EXPECTED_URL || 'http://127.0.0.1:8501').href
  assert.ok(/^http:\/\/127\.0\.0\.1:8501\/$/.test(expected), 'This demo check targets the isolated local Explorer only.')
  assert.equal((await fetch(`${expected}_stcore/health`)).status, 200)
  const link = await evaluate('(() => { const a=document.querySelector("a.explorer-link"); return a ? {href:a.href,target:a.target,rel:a.rel} : null })()')
  assert.ok(link, 'React must render the configured Explorer anchor')
  assert.equal(link.href, expected)
  assert.equal(link.target, '_blank')
  assert.ok(link.rel.includes('noopener') && link.rel.includes('noreferrer'))
  const point = await evaluate('(() => { const r=document.querySelector("a.explorer-link").getBoundingClientRect(); return {x:r.x+r.width/2,y:r.y+r.height/2} })()')
  await send('Input.dispatchMouseEvent', { type: 'mousePressed', button: 'left', clickCount: 1, ...point })
  await send('Input.dispatchMouseEvent', { type: 'mouseReleased', button: 'left', clickCount: 1, ...point })
  const target = await until(async () => {
    const targets = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json()
    return targets.find(page => page.type === 'page' && page.url.startsWith(expected))
  }, 'Research Explorer opens in a separate browser tab')
  const socket = new WebSocket(target.webSocketDebuggerUrl)
  await new Promise((resolve, reject) => { socket.addEventListener('open', resolve, { once: true }); socket.addEventListener('error', reject, { once: true }) })
  let id = 0
  const pending = new Map()
  socket.addEventListener('message', event => {
    const message = JSON.parse(String(event.data))
    const callback = pending.get(message.id)
    if (!callback) return
    pending.delete(message.id)
    if (message.error) callback.reject(new Error(message.error.message))
    else callback.resolve(message.result)
  })
  const call = (method, params = {}) => new Promise((resolve, reject) => {
    const token = ++id
    pending.set(token, { resolve, reject })
    socket.send(JSON.stringify({ id: token, method, params }))
  })
  const read = async expression => {
    const result = await call('Runtime.evaluate', { expression, returnByValue: true })
    assert.ok(!result.exceptionDetails, 'Explorer browser expression must execute')
    return result.result.value
  }
  try {
    await until(() => read('document.body.textContent.includes("ICU Retrospective Research Dashboard") && document.body.textContent.includes("SIMULATED DATA")'), 'main Streamlit page and synthetic notice')
    assert.equal(await read('document.querySelectorAll("[data-testid=stException]").length'), 0)
    await read('Array.from(document.querySelectorAll("[data-testid=stSidebarNav] a")).find(a=>a.textContent.includes("Adaptive"))?.click()')
    await until(() => read('document.body.textContent.includes("Adaptive AI decisions") && document.body.textContent.includes("SIMULATED DATA")'), 'Adaptive Decisions and synthetic notice')
    assert.equal(await read('document.querySelectorAll("[data-testid=stException]").length'), 0)
    await read('Array.from(document.querySelectorAll("[data-testid=stSidebarNav] a")).find(a=>a.textContent.includes("Model Evaluation"))?.click()')
    await until(() => read('document.body.textContent.includes("Model evaluation and reliability") && document.body.textContent.includes("not experimental results") && document.body.textContent.includes("Calibration (equal-count bins)")'), 'Model Evaluation and explicit non-results notice')
    assert.equal(await read('document.querySelectorAll("[data-testid=stException]").length'), 0)
    console.log('Explorer link passed: React opened the real Streamlit app in a separate tab; all three pages rendered with synthetic notices, no runtime errors, and no iframe.')
  } finally {
    socket.close()
    await send('Page.bringToFront')
  }
}
