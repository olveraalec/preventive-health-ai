import { useState, type ChangeEvent } from "react";
import { fetchTrajectory } from "./api/client";
import { ControllerPanel, InfoQuality, Reliability, Timeline } from "./components/Panels";
import { SYNTHETIC_EPISODE } from "./fixtures/synthetic";
import type { Episode } from "./types/trajectory";
import "./App.css";

type Origin = "fixture" | "backend" | "file";
const BANNER: Record<Origin, string> = {
  fixture: "SYNTHETIC FIXTURE. Not patient data and not model output.",
  backend: "Synthetic inputs sent to the backend. Probabilities come from the loaded research artifact; the inputs are not patient data.",
  file: "Loaded from a file. Provenance is as declared in the file and is not verified here.",
};

function isEpisode(j: unknown): j is Episode {
  const e = j as Partial<Episode> | null;
  return !!e && Array.isArray(e.request?.observations) && (e.response?.predictions?.length ?? 0) > 0;
}

export default function App() {
  const [ep, setEp] = useState<Episode>(SYNTHETIC_EPISODE);
  const [origin, setOrigin] = useState<Origin>("fixture");
  const [sel, setSel] = useState(0);
  const [msg, setMsg] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const load = (e: Episode, o: Origin) => { setEp(e); setOrigin(o); setSel(0); setMsg(null); };
  const runBackend = async () => {
    setBusy(true);
    try {
      const response = await fetchTrajectory(SYNTHETIC_EPISODE.request);
      load({ request: SYNTHETIC_EPISODE.request, response }, "backend");
    } catch (e) {
      setMsg(e instanceof Error ? e.message : "Backend unreachable.");
    } finally { setBusy(false); }
  };
  const onFile = async (ev: ChangeEvent<HTMLInputElement>) => {
    const file = ev.target.files?.[0];
    if (!file) return;
    try {
      const j: unknown = JSON.parse(await file.text());
      if (!isEpisode(j)) throw new Error('Expected {"request": ..., "response": ...} with at least one prediction.');
      load(j, "file");
    } catch (e) { setMsg(e instanceof Error ? e.message : "Could not read file."); }
  };

  const { response: r, request } = ep;
  const preds = r.predictions;
  const p = preds[Math.min(sel, preds.length - 1)];
  if (!p) return <main><p>No predictions available.</p></main>;

  return (
    <main>
      <h1>Time-aware preventive health intelligence</h1>
      <p className="sub">Retrospective ICU research demonstration. Not for clinical use.</p>
      <p className={`banner ${origin}`}>{BANNER[origin]}</p>

      <div className="bar">
        <button onClick={() => load(SYNTHETIC_EPISODE, "fixture")}>Synthetic fixture</button>
        <button onClick={runBackend} disabled={busy}>{busy ? "Calling backend…" : "Run backend on synthetic inputs"}</button>
        <label>Load saved JSON <input type="file" accept="application/json" onChange={onFile} /></label>
      </div>
      {msg && <p className="warn">{msg}</p>}

      <section className="card">
        <h2>1. Patient timeline</h2>
        <p className="big">{p.final_probability.toFixed(4)} <small>final estimated probability of the shifted PhysioNet Challenge label, ICU hour {p.icu_hour}</small></p>
        <Timeline preds={preds} sel={Math.min(sel, preds.length - 1)} />
        <p className="legend"><b className="static">Static</b> <b className="temporal">Temporal</b> <b className="final">Final</b> (gaps: no temporal estimate)</p>
        <input type="range" min={0} max={preds.length - 1} value={sel} onChange={(e) => setSel(Number(e.target.value))} aria-label="ICU hour" />
        <small>Label overlays are not shown: this response has no retrospective labels, and they are never available to the online policy.</small>
      </section>

      <div className="two">
        <section className="card"><h2>2. Information quality</h2><InfoQuality obs={request.observations} hour={p.icu_hour} /></section>
        <section className="card"><h2>3. Controller allocation (hour {p.icu_hour})</h2><ControllerPanel p={p} /></section>
      </div>

      <div className="two">
        <section className="card"><h2>4. Reliability</h2><Reliability preds={preds} /></section>
        <section className="card">
          <h2>Evaluation and limitations</h2>
          <p><strong>Evaluation metrics:</strong> Not available. The trajectory API returns none. Contract request: patient-level metrics (log loss, Brier, calibration, alert burden).</p>
          <ul>
            <li>Output type: {r.probability_type}. Not a validated probability of sepsis within six hours.</li>
            <li>Research only: {String(r.research_only)}. Clinical use: {String(r.clinical_use)}.</li>
            <li>Provenance: {r.provenance_status}.</li>
            <li>Controller parameters empirically calibrated: {String(r.parameters_empirically_calibrated)}.</li>
            <li>Earlier patient-disjoint research did not establish a robust patient-mean log-loss gain for learned adaptive allocation over static.</li>
            <li>Feature contributions, where shown, describe the model's output and are not causal claims.</li>
          </ul>
        </section>
      </div>
    </main>
  );
}
