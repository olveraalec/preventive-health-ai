import type { RawTrajectoryRequest, RawHourObservation } from "../types/trajectory.ts";

// SYNTHETIC FIXTURE: every number is produced by the formulas below.
// It is not patient data and not model output.
const r = (v: number, d: number) => Number(v.toFixed(d));
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

// Reused raw observations only; legacy formula-generated predictions are not displayed.
export const SYNTHETIC_REQUEST: RawTrajectoryRequest = {
  dataset: "sepsis2019", episode_id: "SYNTHETIC-DEMO",
  model: { artifact_id: "synthetic", model_version: "1" }, observations,
};
