import type { TrajectoryResponse } from '../api/contracts.ts'
import { percent, points } from '../lib/format.ts'
import { DecisionInterpretation } from './DecisionInterpretation.tsx'
import { RiskTimeline } from './RiskTimeline.tsx'
import { sourceBadge } from '../lib/source.ts'
import type { WorkspaceSource } from '../lib/source.ts'

export function PatientMonitoring({ trajectory, selectedIndex, onSelect, source }: {
  trajectory: TrajectoryResponse; selectedIndex: number; onSelect: (index: number) => void
  source: WorkspaceSource
}) {
  const summary = trajectory.decision_summaries[selectedIndex]
  return <section id="patient-monitoring" className="workspace-section">
    <div className="section-heading"><div><span className="eyebrow">01 / Patient monitoring</span><h2>Follow the signal, observation by observation.</h2></div><span className={`source-badge ${source === 'api' || source === 'retrospective-export' ? 'api' : 'synthetic'}`}><span />{sourceBadge[source]}</span></div>
    <div className="monitoring-grid">
      <article className="panel trajectory-panel">
        <div className="panel-heading"><div><h3>Patient risk trajectory</h3><p className="muted">{source === 'synthetic' ? 'Illustrative episode · not a real patient' : source === 'api-synthetic-inputs' ? 'Synthetic observations · installed model inference' : source === 'retrospective-export' ? 'Retrospective export · source values redacted · provenance as declared' : 'Authorized episode · request-local history'}</p></div><span className="small-chip">{trajectory.decision_summaries.length} observations</span></div>
        <div className="trajectory-stats"><div><span>Blended probability</span><strong>{percent(summary.current.final_probability)}</strong></div><div><span>Since previous observed hour</span><strong className="delta-value">{points(summary.change.probability_delta)}</strong></div><div><span>Selected ICU hour</span><strong>{String(summary.current.icu_hour).padStart(2, '0')}<small>h</small></strong></div></div>
        <RiskTimeline summaries={trajectory.decision_summaries} selectedIndex={selectedIndex} onSelect={onSelect} />
        <div className="target-note">Target: shifted Sepsis Challenge label. This is not a clinically validated onset probability.</div>
      </article>
      <DecisionInterpretation summary={summary} />
    </div>
  </section>
}
