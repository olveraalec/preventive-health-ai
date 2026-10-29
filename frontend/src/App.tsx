import { useEffect, useState } from 'react'
import { explorerUrl } from './api/client.ts'
import { AdaptiveIntelligence } from './components/DecisionInterpretation.tsx'
import { Icon } from './components/Icon.tsx'
import { ModelEvaluation } from './components/ModelEvaluation.tsx'
import { PatientMonitoring } from './components/PatientMonitoring.tsx'
import { ObservationEvidence } from './components/ObservationEvidence.tsx'
import { ResearchAudit } from './components/ResearchAudit.tsx'
import { SYNTHETIC_REQUEST } from './fixtures/synthetic.ts'
import { useResearchWorkspace } from './hooks/useResearchWorkspace.ts'
import { percent, points } from './lib/format.ts'
import { retainedSource, sourceDescription } from './lib/source.ts'
import './App.css'

const navigation = [
  ['patient-monitoring', 'Patient Monitoring'], ['adaptive-intelligence', 'Adaptive Intelligence'],
  ['model-evaluation', 'Model Evaluation'], ['research-methodology', 'Research Methodology'],
] as const

function App() {
  const workspace = useResearchWorkspace()
  const summary = workspace.trajectory.decision_summaries[workspace.selectedIndex]
  const [request, setRequest] = useState('')
  const [connectionOpen, setConnectionOpen] = useState(false)
  const [viewMode, setViewMode] = useState<'monitoring' | 'research'>(() => typeof window !== 'undefined' && window.location.hash === '#model-evaluation' ? 'research' : 'monitoring')
  const [artifactId, setArtifactId] = useState(import.meta.env.VITE_DEMO_MODEL_ARTIFACT_ID ?? '')
  const [modelVersion, setModelVersion] = useState(import.meta.env.VITE_DEMO_MODEL_VERSION ?? '')
  const [hash, setHash] = useState(() => typeof window === 'undefined' ? '' : window.location.hash)
  const explorer = explorerUrl(import.meta.env.VITE_STREAMLIT_URL)
  useEffect(() => {
    const update = () => setHash(window.location.hash)
    window.addEventListener('hashchange', update)
    return () => window.removeEventListener('hashchange', update)
  }, [])
  useEffect(() => {
    if (hash === '#model-evaluation' && viewMode === 'research') document.getElementById('model-evaluation')?.scrollIntoView()
  }, [hash, viewMode])
  return <>
    <a className="skip-link" href="#main">Skip to content</a>
    <header className="site-header"><div className="header-inner">
      <a className="brand" href="#" aria-label="Preventive home"><span className="brand-icon"><Icon name="signal" size={24} /></span><span>preventive<span className="brand-subtitle">HEALTH INTELLIGENCE</span></span></a>
      <nav aria-label="Main navigation">{navigation.map(([id, name]) => <a href={`#${id}`} key={id} onClick={() => { if (id === 'model-evaluation') setViewMode('research'); if (id === 'patient-monitoring') setViewMode('monitoring') }} aria-current={hash === `#${id}` ? 'location' : undefined}>{name}</a>)}</nav>
      {explorer ? <a className="explorer-link" href={explorer} target="_blank" rel="noopener noreferrer">Research Explorer <Icon name="external" size={15} /></a> : <span className="explorer-link unconfigured" title="The separate Streamlit explorer is not linked in this workspace.">Research Explorer <span className="connection-dot" /></span>}
    </div></header>
    <main id="main">
      <section className="landing" aria-labelledby="landing-title">
        <div className="hero-copy"><span className="eyebrow hero-eyebrow"><span className="tiny-cross">+</span> Research with a clearer perspective</span>
          <h1 id="landing-title">See the signal.<br /><em>Understand the decision.</em></h1>
          <p className="vision">An interpretable, time-aware preventive health intelligence system that determines which information is useful, estimates evolving patient risk, and adapts its inference strategy only when the evidence supports adaptation.</p>
          <div className="hero-actions"><a href="#patient-monitoring" className="button primary">Explore the workspace <Icon name="arrow" size={18} /></a><a href="#research-methodology" className="text-link">Our research approach <span>↗</span></a></div>
          <p className="hero-disclaimer"><span className="status-dot" /> Research prototype. Not clinically validated.</p>
        </div>
        <div className="hero-instrument"><div className="instrument-caption"><span>OBSERVE / INTERPRET / REEVALUATE</span><span>01—04</span></div>
          <div className="snapshot-card"><div className="snapshot-heading"><span><Icon name="signal" size={16} /> Decision snapshot</span><span className="snapshot-label">{workspace.source === 'synthetic' ? 'SYNTHETIC FIXTURE' : workspace.source === 'api-synthetic-inputs' ? 'SYNTHETIC INPUTS' : workspace.source === 'retrospective-export' ? 'RETROSPECTIVE EXPORT' : 'API OUTPUT'}</span></div>
            <div className="snapshot-number"><span>Blended probability · ICU hour {summary.current.icu_hour}</span><strong>{percent(summary.current.final_probability)}<span aria-hidden="true">{summary.change.probability_delta === null ? '—' : summary.change.probability_delta > 0 ? '↗' : summary.change.probability_delta < 0 ? '↘' : '→'}</span></strong></div>
            <div className="snapshot-delta">{points(summary.change.probability_delta)} <span>since previous observed hour</span></div>
            <div className="snapshot-rule" />
            <div className="snapshot-row"><span><i className="dot-static" />Static prediction</span><strong>{percent(summary.current.static_probability)}</strong></div>
            <div className="snapshot-row"><span><i className="dot-temporal" />Temporal prediction</span><strong>{percent(summary.current.temporal_probability)}</strong></div>
            <div className="snapshot-foot"><Icon name="layers" size={16} /><span>{summary.controller.fallback_active ? 'Static fallback active' : 'Bounded controller allocation'}</span><span>{percent(summary.current.temporal_weight, 1)} temporal</span></div>
          </div><div className="instrument-foot"><span>SHIFTED CHALLENGE-LABEL PROBABILITY</span><span>Research only</span></div>
        </div>
      </section>
      <div className="principles-strip"><span><Icon name="signal" size={17} /> Time-aware observations</span><span><Icon name="layers" size={17} /> Visible model contributions</span><span><Icon name="chart" size={17} /> Evidence before adaptation</span><span><Icon name="book" size={17} /> Traceable research</span></div>
      <div className="workspace-container">
        <div className="workspace-toolbar"><div><span className="eyebrow">The research workspace</span><p>{sourceDescription[workspace.source]}</p></div><div className="toolbar-actions"><button type="button" className="button secondary" onClick={workspace.useSynthetic} disabled={workspace.source === 'synthetic' && workspace.status === 'idle'}>Use synthetic example</button><a className="text-link" href="#api-connection" onClick={() => setConnectionOpen(true)}>Connect API / export <Icon name="arrow" size={16} /></a></div></div>
        <div className="view-mode" role="group" aria-label="Workspace perspective"><button type="button" aria-pressed={viewMode === 'monitoring'} onClick={() => setViewMode('monitoring')}>Monitoring perspective</button><button type="button" aria-pressed={viewMode === 'research'} onClick={() => setViewMode('research')}>Research perspective</button><span>One episode, shared decisions. Research-only use.</span></div>
        <PatientMonitoring trajectory={workspace.trajectory} selectedIndex={workspace.selectedIndex} onSelect={workspace.setSelectedIndex} source={workspace.source} />
        <div hidden={viewMode !== 'monitoring'}><ObservationEvidence request={workspace.observationRequest} hour={summary.current.icu_hour} source={workspace.source} /></div>
        <AdaptiveIntelligence summary={summary} />
        <div hidden={viewMode !== 'research'}><ResearchAudit summaries={workspace.trajectory.decision_summaries} selectedIndex={workspace.selectedIndex} /><ModelEvaluation evaluation={workspace.evaluation} notice={workspace.evaluationNotice} /></div>
        <section id="research-methodology" className="workspace-section methodology">
          <div className="section-heading"><div><span className="eyebrow">04 / Research methodology</span><h2>Every number has a boundary.</h2></div><Icon name="book" size={28} /></div>
          <div className="method-grid">{[
            ['01', 'Respect observation time', 'Use current and past observations only. Preserve elapsed ICU hours and never fill a gap with invented evidence.'],
            ['02', 'Keep the models visible', 'Show frozen static and temporal predictions, their mixture allocation, and immediate static fallback when the secondary path fails.'],
            ['03', 'Explain the arithmetic', 'Separate static change, disagreement change, and reallocation. Mathematical attribution is not physiological causality.'],
            ['04', 'Keep evaluation separate', 'Compare recorded, patient-disjoint evidence retrospectively. Future outcomes never become inputs to live inference.'],
          ].map(([n, title, description]) => <article key={n}><span className="method-number">{n}</span><h3>{title}</h3><p>{description}</p></article>)}</div>
          <div className="explorer-callout"><div className="explorer-icon"><Icon name="chart" size={30} /></div><div><span className="eyebrow">A separate research explorer</span><h3>Go deeper with Streamlit.</h3><p>The teammate’s research dashboard remains independent: exploratory trajectories, thresholds, vital-sign trends and retrospective metrics.</p></div>{explorer ? <a className="button secondary" href={explorer} target="_blank" rel="noopener noreferrer">Open Research Explorer <Icon name="external" size={16} /></a> : <span className="small-chip">Explorer link not configured</span>}</div>
        </section>
        <section id="api-connection" className="workspace-section api-connection">
          <details open={connectionOpen} onToggle={event => setConnectionOpen(event.currentTarget.open)}><summary><span><Icon name="layers" /> Connect an API or open a local export</span><span className="small-chip">Optional</span></summary><div className="api-form-content"><p>Submit a raw-hourly Sepsis trajectory with model identifiers from the installed manifest, or select a sanitized retrospective export. Nothing is saved in browser storage. Do not paste credentials or labels.</p>
            <div className="sample-request"><h3>Open private retrospective outputs</h3><p>Select the backend's sanitized retrospective JSON export. It is read locally into browser memory, never uploaded or bundled into this website. Export provenance remains as declared by the file.</p><label htmlFor="retrospective-export">Sanitized retrospective export</label><input id="retrospective-export" type="file" accept="application/json,.json" disabled={workspace.status === 'loading'} onChange={async event => {
              const input = event.currentTarget
              const file = input.files?.[0]
              if (file && await workspace.loadRetrospective(file)) { setRequest(''); setViewMode('monitoring'); document.querySelector('#patient-monitoring')?.scrollIntoView() }
              input.value = ''
            }} /></div>
            <div className="sample-request"><h3>A complete synthetic-input journey</h3><p>Use the teammate's 24-hour synthetic observations with the installed, unchanged research model. These are not patient records. Supply identifiers from the approved model manifest.</p><div className="model-identifiers"><label>Artifact identifier<input value={artifactId} onChange={event => setArtifactId(event.target.value)} placeholder="From installed manifest" /></label><label>Model version<input value={modelVersion} onChange={event => setModelVersion(event.target.value)} placeholder="From installed manifest" /></label><button type="button" className="button secondary" disabled={!artifactId || !modelVersion || workspace.status === 'loading'} onClick={async () => {
              const sample = JSON.stringify({ ...SYNTHETIC_REQUEST, model: { artifact_id: artifactId, model_version: modelVersion } })
              if (await workspace.load(sample, true)) { setRequest(''); setViewMode('monitoring'); document.querySelector('#patient-monitoring')?.scrollIntoView() }
            }}>Run model on synthetic observations <Icon name="arrow" size={16} /></button></div></div>
            <form onSubmit={async event => { event.preventDefault(); if (await workspace.load(request, false)) setRequest('') }}>
            <label htmlFor="trajectory-request">Or submit an authorized raw-hourly request JSON</label><textarea id="trajectory-request" value={request} onChange={event => setRequest(event.target.value)} spellCheck={false} disabled={workspace.status === 'loading'} placeholder={'{"dataset":"sepsis2019","episode_id":"…","model":{"artifact_id":"…","model_version":"…"},"observations":[…]}'} required rows={7} />
            <div className="form-actions"><button className="button primary" type="submit" disabled={workspace.status === 'loading'}>{workspace.status === 'loading' ? 'Running inference…' : 'Run research inference'}<Icon name="arrow" size={17} /></button><span className="fine-print">Requires an available FastAPI model artifact.</span></div>
            {workspace.status === 'loading' && <p role="status" className="inline-note">Loading research data. Existing displayed data has not been replaced.</p>}
            {workspace.error && <div className="error-box" role="alert"><strong>{workspace.error}</strong><p>{retainedSource[workspace.source]} Check the request, file or connection, then retry.</p></div>}
            {workspace.status === 'success' && <p role="status" className="inline-note">{workspace.source === 'retrospective-export' ? 'Retrospective export loaded locally. Timeline, controller and interpretation display the imported decisions. No file was uploaded.' : 'Research response loaded. Monitoring and interpretation now display API outputs.'}</p>}
          </form></div></details>
        </section>
      </div>
    </main>
    <footer className="site-footer"><div><a className="footer-brand" href="#">preventive<span> / research</span></a><p>Interpretable by design. Experimental by definition.</p></div><p>Research-only prototype · Not clinically validated.<br />No autonomous treatment decisions or certified risk estimates.</p><a href="#research-methodology">Research boundaries <Icon name="arrow" size={16} /></a></footer>
  </>
}
export default App
