import type { Episode, HourPrediction, RawHourObservation } from "../types/trajectory";

// SYNTHETIC FIXTURE: every number is produced by the formulas below.
// It is not patient data and not model output.
const r = (v: number, d: number) => Number(v.toFixed(d));
const clamp = (v: number) => Math.min(1, Math.max(0, v));
const SERIES: Record<string, [(h: number) => boolean, (h: number) => number]> = {
  HR: [() => true, (h) => 85 + 0.8 * h],
  O2Sat: [() => true, (h) => 97 - 0.1 * h],
  Temp: [(h) => h % 4 === 1, (h) => 36.8 + 0.03 * h],
  SBP: [(h) => h % 2 === 0, (h) => 120 - 0.6 * h],
  MAP: [(h) => h < 9 || h > 12, (h) => 80 - 0.5 * h],
  DBP: [(h) => h % 3 === 0, (h) => 62 - 0.2 * h],
  Resp: [() => true, (h) => 17 + 0.2 * h],
};
const hours = Array.from({ length: 24 }, (_, i) => i + 1);
const observations: RawHourObservation[] = hours.map((h) => ({
  episode_id: "SYNTHETIC-DEMO",
  icu_hour: h,
  measurements: {
    ...Object.fromEntries(
      Object.entries(SERIES).map(([k, [seen, val]]) => [k, seen(h) ? r(val(h), 1) : null]),
    ),
    Age: 55, Gender: 1, HospAdmTime: -12, Unit1: 1, Unit2: null,
  },
}));
const predictions: HourPrediction[] = hours.map((h): HourPrediction => {
  const s = r(clamp(0.03 + 0.35 / (1 + Math.exp(-(h - 16) / 3))), 4);
  const failed = h >= 12 && h <= 14; // synthetic temporal-expert outage
  const t = failed ? null : r(clamp(s + 0.03 + 0.04 * Math.sin(h / 3)), 4);
  const q = failed ? 0 : r(Math.min(0.6, 0.1 + 0.02 * h), 4);
  return {
    icu_hour: h,
    static_probability: s,
    temporal_probability: t,
    final_probability: r(t === null ? s : s + q * (t - s), 6),
    temporal_weight: q,
    static_weight: r(1 - q, 4),
    status: failed ? "static_only_fallback" : "bounded_anticipatory",
    reason: failed ? "secondary_inference_failed" : "no_verified_feedback",
  };
});
export const SYNTHETIC_EPISODE: Episode = {
  request: {
    dataset: "sepsis2019",
    episode_id: "SYNTHETIC-DEMO",
    model: { artifact_id: "synthetic", model_version: "1" },
    observations,
  },
  response: {
    dataset: "sepsis2019",
    episode_id: "SYNTHETIC-DEMO",
    provenance: {
      feature_schema: "synthetic", preprocessing_version: "synthetic",
      pipeline_id: "synthetic", declaration: "current_and_past_only",
    },
    provenance_status: "synthetic_fixture",
    model: { artifact_id: "synthetic", model_version: "1" },
    predictions,
    probability_type: "shifted_challenge_label",
    research_only: true,
    clinical_use: false,
    configured_parameters: {},
    parameters_empirically_calibrated: false,
  },
};
