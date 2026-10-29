import { useState } from 'react'
import type { EvaluationSummary } from '../api/contracts.ts'
import { label } from '../lib/format.ts'
import { Icon } from './Icon.tsx'

const policyNames: Record<string, string> = {
  static: 'Static', training_selected_constant: 'Training-selected constant',
  global_brier_moment: 'Global Brier / moment', conditional_two_regime_moment: 'Conditional two-regime',
  shuffled_context_control: 'Shuffled-context control',
}
const metricNames: Record<string, string> = {
  patient_mean_bernoulli_log_loss: 'Patient-mean log loss',
  patient_cvar90_bernoulli_log_loss: 'Patient CVaR90 log loss',
  positive_patient_mean_bernoulli_log_loss: 'Positive-patient mean log loss',
}
export function ModelEvaluation({ evaluation, notice }: { evaluation: EvaluationSummary; notice: string | null }) {
  const experiment = evaluation.experiments[0]
  const [metricId, setMetricId] = useState('patient_mean_bernoulli_log_loss')
  const selectedMetrics = experiment.policies.map(policy => policy.metrics.find(m => m.metric_id === metricId))
  const max = Math.max(...selectedMetrics.map(m => m?.value ?? 0), 0.000001)
  return <section id="model-evaluation" className="workspace-section">
    <div className="section-heading"><div><span className="eyebrow">03 / Model evaluation</span><h2>Evidence before adaptation.</h2></div><span className="source-badge synthetic"><span />Recorded synthetic study</span></div>
    <div className="evaluation-notice"><Icon name="info" /><p><strong>Simulated binary sequences. Not results from the PhysioNet cohort.</strong><span>Document-reported held-out results across {experiment.n_test_episodes} synthetic episodes. Not reproduced here; no evaluation is bound to the installed Sepsis model.</span></p></div>
    {notice && <p className="inline-note" role="status">{notice}</p>}
    <div className="evaluation-grid"><article className="panel evaluation-table-panel">
      <div className="panel-heading"><h3>Policy comparison</h3><label className="metric-select"><span className="sr-only">Evaluation metric</span><select value={metricId} onChange={event => setMetricId(event.target.value)}>{Object.entries(metricNames).map(([id, name]) => <option value={id} key={id}>{name}</option>)}</select></label></div>
      <p className="fine-print">Lower is better · nats · {label(selectedMetrics.find(Boolean)?.aggregation ?? 'unavailable')}</p>
      <div className="table-wrap"><table><caption className="sr-only">Recorded synthetic policy evaluation, not real-patient performance</caption><thead><tr><th scope="col">Policy</th><th scope="col">Recorded loss</th><th scope="col" className="bar-column">Relative magnitude</th></tr></thead><tbody>{experiment.policies.map((policy, i) => <tr key={policy.policy_id}><th scope="row">{policyNames[policy.policy_id] ?? label(policy.policy_id)}</th><td>{selectedMetrics[i]?.value.toFixed(6) ?? 'Unavailable'}</td><td className="bar-column"><span className="metric-track"><span style={{ width: `${Math.max(0, (selectedMetrics[i]?.value ?? 0) / max * 100)}%` }} /></span></td></tr>)}</tbody></table></div>
    </article><aside className="panel evidence-readout"><span className="eyebrow">Read the whole result</span><h3>Mean gain is not universal benefit.</h3><div className="paired-gain"><span>Paired patient log-loss gain vs constant</span><strong>+{experiment.comparison.metric.value.toFixed(6)}</strong><small>95% interval [{experiment.comparison.ci95.map(value => value.toFixed(6)).join(', ')}]</small></div><p>The conditional policy improved average loss but worsened positive-patient mean loss relative to the constant blend.</p><p className="fine-print">Exploratory paired patient bootstrap · {experiment.comparison.bootstrap_replicates.toLocaleString('en-US')} replicates. This synthetic result cannot establish a benefit for the Sepsis model.</p></aside></div>
    <details className="detail provenance-detail"><summary>Study provenance and unavailable metrics</summary><dl><dt>Dataset / split</dt><dd>{experiment.dataset} / {experiment.split}</dd><dt>Source</dt><dd>{experiment.provenance.source_path}</dd><dt>Source SHA-256</dt><dd>{experiment.provenance.source_sha256}</dd><dt>Evidence</dt><dd>{label(experiment.provenance.evidence_status)}</dd><dt>Missing provenance</dt><dd>{experiment.provenance.missing_provenance.map(label).join('; ')}</dd><dt>Unavailable metrics</dt><dd>{evaluation.unavailable_metrics.map(label).join(', ')}</dd></dl></details>
  </section>
}
