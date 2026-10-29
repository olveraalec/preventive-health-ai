import type { DecisionSummary } from '../api/contracts.ts'
import { ControllerPanel, Reliability } from './Panels.tsx'

export function ResearchAudit({ summaries, selectedIndex }: { summaries: DecisionSummary[]; selectedIndex: number }) {
  return <section className="research-audit" aria-label="Research diagnostics">
    <div className="adaptive-grid"><details className="panel legacy-panels"><summary>Controller arithmetic audit</summary><ControllerPanel p={summaries[selectedIndex].current} /></details><article className="panel legacy-panels"><h3>Observed controller behavior</h3><Reliability preds={summaries.map(summary => summary.current)} /><p className="fine-print">Counts summarize observed outputs within this request. Allocation movement and fallback counts do not establish predictive improvement.</p></article></div>
    <div className="unavailable-research"><h3>Calibration & patient-level uncertainty</h3><p>Calibration curves, Brier, AUROC/AUPRC, real-cohort paired intervals and retrospective alert burden are not recorded in the available response. They remain unavailable; no live labels or synthetic substitute are used to calculate them.</p></div>
  </section>
}
