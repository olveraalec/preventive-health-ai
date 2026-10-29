import type { DecisionSummary } from '../api/contracts.ts'
import { Timeline } from './Panels.tsx'

/** Keep the teammate's chart; add accessible observation selection and scope notes. */
export function RiskTimeline({ summaries, selectedIndex, onSelect }: {
  summaries: DecisionSummary[]; selectedIndex: number; onSelect: (index: number) => void
}) {
  const states = summaries.map(item => item.current)
  return <div className="timeline">
    <div className="chart-legend"><span className="legend-final">Blended</span><span className="legend-static">Static</span><span className="legend-temporal">Temporal</span></div>
    <div className="risk-chart reused-timeline"><Timeline preds={states} sel={selectedIndex} /></div>
    <div className="timeline-controls"><span>Observed ICU hour</span><div role="group" aria-label="Select observed ICU hour">{states.map((p, index) => <button key={p.icu_hour} type="button" className={selectedIndex === index ? 'hour selected' : 'hour'} aria-pressed={selectedIndex === index} onClick={() => onSelect(index)}>{p.icu_hour}h</button>)}</div></div>
    <p className="chart-note">Lines connect observed hours. Gaps are retained. No future forecast.</p>
  </div>
}
