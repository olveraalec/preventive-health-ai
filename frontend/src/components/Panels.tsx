import type { HourPrediction, RawHourObservation } from "../types/trajectory";
import { VITALS } from "../types/trajectory";

const NA = "Not available";
const f4 = (v: number) => v.toFixed(4);

export function Timeline({ preds, sel }: { preds: HourPrediction[]; sel: number }) {
  const W = 640, H = 220, L = 44, B = 28, T = 10, R = 10;
  const hrs = preds.map((p) => p.icu_hour);
  const x0 = Math.min(...hrs), x1 = Math.max(...hrs);
  const vals = preds.flatMap((p) => [p.static_probability, p.final_probability, p.temporal_probability ?? 0]);
  const ymax = Math.min(1, Math.ceil(Math.max(...vals, 0.1) * 10) / 10);
  const x = (h: number) => L + ((h - x0) / Math.max(x1 - x0, 1)) * (W - L - R);
  const y = (v: number) => H - B - (v / ymax) * (H - B - T);
  const line = (get: (p: HourPrediction) => number | null) => {
    let d = "", pen = false;
    for (const p of preds) {
      const v = get(p);
      if (v === null) { pen = false; continue; } // gap where the temporal expert returned nothing
      d += `${pen ? "L" : "M"}${x(p.icu_hour).toFixed(1)},${y(v).toFixed(1)}`;
      pen = true;
    }
    return d;
  };
  const cur = preds[sel]?.icu_hour ?? x0;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Static, temporal and final estimated probability by ICU hour">
      {[0, 0.5, 1].map((k) => (
        <g key={k}>
          <line x1={L} x2={W - R} y1={y(ymax * k)} y2={y(ymax * k)} className="grid" />
          <text x={L - 6} y={y(ymax * k) + 4} textAnchor="end" className="tick">{(ymax * k).toFixed(2)}</text>
        </g>
      ))}
      <text x={L} y={H - 8} className="tick">ICU hour {x0}</text>
      <text x={W - R} y={H - 8} textAnchor="end" className="tick">{x1}</text>
      <line x1={x(cur)} x2={x(cur)} y1={T} y2={H - B} className="cursor" />
      <path d={line((p) => p.static_probability)} className="ln static" />
      <path d={line((p) => p.temporal_probability)} className="ln temporal" />
      <path d={line((p) => p.final_probability)} className="ln final" />
    </svg>
  );
}

const STATUS: Record<string, string> = {
  bounded_anticipatory: "Bounded temporal blend",
  static_only_fallback: "Static-only fallback",
};
const REASON: Record<string, string> = {
  no_verified_feedback: "No verified delayed feedback is available (research prototype).",
  secondary_inference_failed: "The temporal expert returned no prediction.",
};

export function ControllerPanel({ p }: { p: HourPrediction }) {
  const s = p.static_probability, t = p.temporal_probability, q = p.temporal_weight;
  const d = t === null ? null : t - s;
  const mix = d === null ? null : s + q * d;
  const diff = mix === null ? null : Math.abs(mix - p.final_probability);
  return (
    <>
      <table>
        <tbody>
          <tr><th>Static S</th><td>{f4(s)}</td></tr>
          <tr><th>Temporal T</th><td>{t === null ? NA : f4(t)}</td></tr>
          <tr><th>Difference d = T − S</th><td>{d === null ? NA : f4(d)}</td></tr>
          <tr><th>Temporal weight q</th><td>{f4(q)}</td></tr>
          <tr><th>Static weight</th><td>{f4(p.static_weight)}</td></tr>
          <tr><th>Final (reported)</th><td>{f4(p.final_probability)}</td></tr>
        </tbody>
      </table>
      {mix !== null && d !== null ? (
        <p className="mono">
          S + q·d = {f4(s)} + {f4(q)} × {f4(d)} = {f4(mix)}
        </p>
      ) : (
        <p>No temporal estimate this hour, so the mixture is not computed.</p>
      )}
      {diff !== null && (diff < 1e-6 ? (
        <p>Reported final matches S + q·(T − S).</p>
      ) : (
        <p className="warn">Reported final differs from S + q·(T − S) by {diff.toExponential(2)}. Contract request: confirm the backend mixture rule.</p>
      ))}
      <p><strong>{STATUS[p.status]}.</strong> {REASON[p.reason]}</p>
    </>
  );
}

export function Reliability({ preds }: { preds: HourPrediction[] }) {
  const fb = preds.filter((p) => p.status === "static_only_fallback");
  const byReason = (r: string) => fb.filter((p) => p.reason === r).length;
  return (
    <>
      <p>
        {fb.length} of {preds.length} hours used the static-only fallback
        {fb.length > 0 && <> (temporal expert returned nothing: {byReason("secondary_inference_failed")}; no verified feedback: {byReason("no_verified_feedback")})</>}.
      </p>
      <p>
        The API exposes only two reasons. Abstention causes such as missing information or
        insufficient support are not available. Contract request: typed reliability reasons.
      </p>
    </>
  );
}

const UNITS: Record<string, string> = { HR: "bpm", O2Sat: "%", Temp: "°C", SBP: "mmHg", MAP: "mmHg", DBP: "mmHg", Resp: "/min" };

export function InfoQuality({ obs, hour }: { obs: RawHourObservation[]; hour: number }) {
  const upto = obs.filter((o) => o.icu_hour <= hour);
  return (
    <>
      <table>
        <thead><tr><th>Signal</th><th>Observed per hour</th><th>Latest observed</th><th>Age</th></tr></thead>
        <tbody>
          {VITALS.map((v) => {
            const latest = [...upto].reverse().find((o) => o.measurements[v] != null);
            const n = upto.filter((o) => o.measurements[v] != null).length;
            return (
              <tr key={v}>
                <th>{v} ({UNITS[v]})</th>
                <td>
                  <span className="strip">
                    {upto.map((o) => (
                      <i key={o.icu_hour} className={o.measurements[v] != null ? "on" : "off"}
                         title={`Hour ${o.icu_hour}: ${o.measurements[v] != null ? "observed" : "not observed"}`} />
                    ))}
                  </span>
                  <small> {n}/{upto.length}</small>
                </td>
                <td>{latest ? String(latest.measurements[v]) : "Not observed yet"}</td>
                <td>{latest ? (hour === latest.icu_hour ? "this hour" : `${hour - latest.icu_hour} h ago`) : NA}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <small>Unobserved values are never filled in. Units follow the Challenge documentation and are not verified by the pipeline.</small>
    </>
  );
}
