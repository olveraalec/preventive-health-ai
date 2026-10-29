import type { DecisionSummary } from '../api/contracts.ts'
import { decimal, label, percent, points } from '../lib/format.ts'
import { Icon } from './Icon.tsx'

export function DecisionInterpretation({ summary }: { summary: DecisionSummary }) {
  const change = summary.change
  const terms = [
    ['Static prediction', 'ΔS', change.static_delta],
    ['Expert disagreement', 'qprev × Δ(T − S)', change.disagreement_change_at_previous_allocation],
    ['Controller reallocation', 'Δq × (T − S)current', change.reallocation_at_current_disagreement],
  ] as const
  return <article className="panel interpretation" aria-labelledby="interpretation-title">
    <div className="panel-heading"><span className="eyebrow">Inside the decision</span><span className="small-chip">v0.6</span></div>
    <h3 id="interpretation-title">What changed?</h3>
    <div className="prediction-comparison">
      <div><span>Previous observed · {summary.previous ? `${summary.previous.icu_hour}h` : '—'}</span><strong>{summary.previous ? percent(summary.previous.final_probability) : 'No previous'}</strong></div>
      <Icon name="arrow" />
      <div><span>Selected observation · {summary.current.icu_hour}h</span><strong>{percent(summary.current.final_probability)}</strong></div>
    </div>
    <div className="change-total"><span>{summary.elapsed_hours === null ? 'First observation' : `Over ${summary.elapsed_hours} elapsed hours`}</span><strong>{points(change.probability_delta)}</strong></div>
    <div className="decomposition">{terms.map(([name, formula, value], i) => <div className="decomposition-row" key={name}>
      <span className={`term-number term-${i}`}>{String(i + 1).padStart(2, '0')}</span><div><strong>{name}</strong><span className="formula">{formula}</span></div><span className="term-value">{points(value)}</span>
    </div>)}</div>
    {change.status === 'unavailable' && <p className="inline-note"><Icon name="info" size={16} />{change.reason === 'first_observation' ? 'A second observation is needed to explain a change.' : 'An expert probability is missing. Its attribution remains unavailable.'}</p>}
    <p className="interpretation-caveat">Exact arithmetic attribution. It does not explain physiological causes. Values displayed in percentage points (pp), rounded to two decimals.</p>
    <details className="detail"><summary>Inspect full-precision attribution</summary><pre>{JSON.stringify(change, null, 2)}</pre></details>
  </article>
}

export function AdaptiveIntelligence({ summary }: { summary: DecisionSummary }) {
  const { current, controller, data_quality: quality } = summary
  return <section id="adaptive-intelligence" className="workspace-section">
    <div className="section-heading"><div><span className="eyebrow">02 / Adaptive intelligence</span><h2>Two perspectives. One visible decision.</h2></div><span className="section-context">Selected ICU hour {current.icu_hour}</span></div>
    <div className="adaptive-grid">
      <article className="panel allocation-panel">
        <div className="panel-heading"><h3>Controller allocation</h3><Icon name="layers" /></div>
        <p className="muted">Learned gain proposal with heuristic exposure and configured bounds.</p>
        <div className="allocation-labels"><div><span className="legend-static">Static</span><strong>{percent(current.static_weight, 1)}</strong></div><div><span className="legend-temporal">Temporal</span><strong>{percent(current.temporal_weight, 1)}</strong></div></div>
        <div className="allocation-track" role="img" aria-label={`Static allocation ${percent(current.static_weight)}, temporal allocation ${percent(current.temporal_weight)}`}><span style={{ width: `${current.static_weight * 100}%` }} /><span style={{ width: `${current.temporal_weight * 100}%` }} /></div>
        <div className="expert-values"><div><span>Static prediction</span><strong>{percent(current.static_probability)}</strong><small>Contribution {percent(current.static_contribution)}</small></div><div><span>Temporal prediction</span><strong>{percent(current.temporal_probability)}</strong><small>Contribution {percent(current.temporal_contribution)}</small></div></div>
        <div className={controller.fallback_active ? 'status-box warning' : 'status-box'}><Icon name={controller.fallback_active ? 'info' : 'check'} size={18} /><div><strong>{controller.fallback_active ? 'Static emergency fallback active' : 'Bounded anticipatory mode'}</strong><p>{controller.fallback_reason ? label(controller.fallback_reason) : 'No verified outcome feedback. Reactive adaptation is unavailable.'}</p></div></div>
        <p className="fine-print">Allocation is not empirically validated. Exposure {decimal(controller.exposure, 2)} is not calibrated reliability.</p>
      </article>
      <article className="panel evidence-panel">
        <div className="panel-heading"><h3>Evidence & uncertainty</h3><Icon name="signal" /></div>
        <div className="diagnostic-pair"><div><span>Predictive entropy</span><strong>{decimal(summary.diagnostics.final_predictive_entropy_nats)} <small>nats</small></strong></div><div><span>Expert disagreement</span><strong>{points(summary.diagnostics.absolute_model_disagreement).replace(/^\+/, '')}</strong></div></div>
        <p className="fine-print">Entropy describes the predictive distribution, not confidence that the model is correct.</p>
        <div className="evidence-heading"><strong>Measurement freshness</strong><span className="small-chip">{quality.status === 'available' ? `${quality.missing_count ?? '—'} / 7 missing` : 'Unavailable'}</span></div>
        {quality.status === 'available' ? <ul className="freshness-list">{Object.entries(quality.measurements).map(([name, item]) => <li key={name}><strong>{name}</strong><span>{item.observed ? 'Observed now' : 'Missing now'}</span><span>{item.age_hours === null ? 'Never observed' : `${item.age_hours}h since observed`}</span></li>)}</ul> : <div className="empty-evidence"><Icon name="info" /><p>Raw measurements were not supplied.<br /><span>Freshness cannot be inferred from engineered features.</span></p></div>}
        <p className="fine-print">{quality.engineered_feature_count === null ? 'Engineered input counts are not reported.' : `${quality.engineered_missing_count} / ${quality.engineered_feature_count} engineered inputs missing.`} Source availability has not been independently verified.</p>
      </article>
    </div>
    <div className="next-step"><div className="next-icon"><Icon name="arrow" /></div><div><span className="eyebrow">Next computational step</span><h3>Await the next valid observation.</h3><p>Update causal temporal features, then reevaluate inference and allocation.</p></div><span className="small-chip">Observation-driven</span></div>
    <div className="capability-grid">{[
      ['Next-hour forecast', summary.next_hour_forecast, 'No validated next-hour forecaster.'],
      ['Learned hybrid', summary.learned_hybrid, 'Not integrated or validated.'],
      ['Workflow recommendation', summary.workflow_recommendation, 'No supported recommendation layer.'],
    ].map(([name, capability, description]) => <div className="capability" key={String(name)}><span className="eyebrow">Unavailable</span><h4>{String(name)}</h4><p>{String(description)}</p><span className="sr-only">{typeof capability === 'object' ? capability.reason : ''}</span></div>)}</div>
    <details className="detail provenance-detail"><summary>Model, feature and constraint provenance</summary><dl><dt>Artifact</dt><dd>{summary.provenance.model.artifact_id}</dd><dt>SHA-256</dt><dd>{summary.provenance.artifact_sha256 ?? 'Unavailable — no validated manifest in this response'}</dd><dt>Feature schema</dt><dd>{summary.provenance.features.feature_schema}</dd><dt>Feature builder</dt><dd>{summary.provenance.features.pipeline_id} · version {summary.provenance.features.preprocessing_version}</dd><dt>Feature status</dt><dd>{label(summary.provenance.feature_status)}</dd><dt>Constraints</dt><dd>{Object.entries(controller.configured_constraints).map(([key, value]) => `${label(key)}: ${value}`).join(' · ')}</dd><dt>Binding constraints</dt><dd>{label(controller.constraint_trace)}</dd><dt>Model evaluation</dt><dd>Unavailable — no recorded evaluation bound to this model.</dd></dl></details>
  </section>
}
