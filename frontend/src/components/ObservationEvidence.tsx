import type { RawTrajectoryRequest } from '../api/contracts.ts'
import { observationEvidence } from '../lib/observations.ts'
import { InfoQuality } from './Panels.tsx'
import { UNITS } from '../types/trajectory.ts'
import { Icon } from './Icon.tsx'

function MeasurementTrend({ points, name }: { points: { hour: number; value: number | null }[]; name: string }) {
  const observed = points.filter((point): point is { hour: number; value: number } => point.value !== null)
  if (!observed.length) return <p className="no-measurement">No observations through this hour</p>
  const firstHour = points[0].hour, lastHour = points.at(-1)!.hour
  const min = Math.min(...observed.map(p => p.value)), max = Math.max(...observed.map(p => p.value))
  const padding = Math.max((max - min) * .25, 0.5)
  const x = (hour: number) => 38 + (hour - firstHour) / Math.max(lastHour - firstHour, 1) * 240
  const y = (value: number) => 80 - (value - min + padding) / (max - min + 2 * padding) * 58
  const line = points.map((point, index) => point.value === null ? '' : `${index === 0 || points[index - 1].value === null ? 'M' : 'L'}${x(point.hour)},${y(point.value)}`).join(' ')
  return <svg className="measurement-chart" viewBox="0 0 300 108" role="img" aria-label={`${name}: observed measurements through ICU hour ${lastHour}. Missing values are gaps; no imputation.`}>
    <line x1="38" x2="278" y1="22" y2="22" className="grid-line" /><line x1="38" x2="278" y1="80" y2="80" className="grid-line" />
    <text x="32" y="26" textAnchor="end">{(max + padding).toFixed(1)}</text><text x="32" y="84" textAnchor="end">{(min - padding).toFixed(1)}</text>
    <path d={line} fill="none" stroke="currentColor" strokeWidth="1.8" />
    {observed.map(point => <circle key={point.hour} cx={x(point.hour)} cy={y(point.value)} r="2.4" fill="currentColor" />)}
    <text x="38" y="102">{firstHour}h</text><text x="278" y="102" textAnchor="end">{lastHour}h</text>
  </svg>
}

export function ObservationEvidence({ request, hour, source }: {
  request: RawTrajectoryRequest | null; hour: number; source: 'synthetic' | 'api' | 'api-synthetic-inputs'
}) {
  if (!request) return <section className="panel observation-unavailable" aria-label="Observation evidence"><Icon name="info" /><div><h3>Observation values are unavailable in this documented example.</h3><p>The v0.6 engineered fixture contains probabilities, not raw vital signs. Connect a raw-hourly request to see its measurements, units, changes and age. No signals are invented to fill this view.</p></div></section>
  const evidence = observationEvidence(request, hour)
  return <section className="observation-evidence" aria-labelledby="observation-title">
    <div className="panel-heading"><div><span className="eyebrow">Available observations</span><h3 id="observation-title">What changed in the evidence?</h3></div><span className={source !== 'api' ? 'source-badge synthetic' : 'source-badge api'}>{source === 'synthetic' ? 'Documented synthetic observations' : source === 'api-synthetic-inputs' ? 'Synthetic inputs · model outputs' : 'Authorized request inputs'}</span></div>
    <p className="fine-print">Source units, not independently verified. Selected ICU hour {hour}; no future observations shown. Older measurements are not carried forward as current values.</p>
    <div className="measurement-grid">{evidence.map(item => <article className="panel measurement-card" key={item.name}>
      <div className="panel-heading"><h4>{item.name} <span>{UNITS[item.name]}</span></h4><span className={`measurement-status ${item.status}`}>{item.status === 'observed_now' ? 'Observed' : item.status === 'not_current' ? `${item.age}h old · not current` : 'Never observed'}</span></div>
      <div className="measurement-value">{item.value === null ? 'Missing' : item.value.toFixed(1)}<span>{item.value === null ? item.lastValue === null ? 'No earlier value' : `Last observed: ${item.lastValue} ${UNITS[item.name]}` : item.change === null ? 'No comparable previous measurement' : `${item.change > 0 ? '+' : ''}${item.change.toFixed(1)} ${UNITS[item.name]} since previous observed hour`}</span></div>
      <MeasurementTrend points={item.points} name={item.name} />
    </article>)}</div>
    <details className="panel review-evidence"><summary>Review-worthy evidence · descriptive facts, not clinical triggers</summary><ul>{evidence.filter(item => item.value === null || item.change !== null && item.change !== 0).map(item => <li key={item.name}><strong>{item.name}:</strong> {item.value === null ? item.age === null ? 'not observed in this history.' : `missing now; last observed ${item.age} ICU hours ago.` : `changed from ${item.previousValue} to ${item.value} ${UNITS[item.name]} since the previous observed hour.`}</li>)}</ul><p className="fine-print">Age reports observation recency, not a validated clinical expiry threshold. No diagnosis, escalation or treatment trigger is inferred.</p></details>
    <details className="panel measurement-table"><summary>Per-signal availability and last-observed values</summary><div className="table-wrap"><InfoQuality obs={request.observations} hour={hour} /></div></details>
  </section>
}
