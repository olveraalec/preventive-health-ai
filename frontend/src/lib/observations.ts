import type { RawTrajectoryRequest } from '../api/contracts.ts'
import { VITALS } from '../types/trajectory.ts'

/** Descriptive prefix-only evidence. No thresholds, diagnoses, labels or imputation. */
export function observationEvidence(request: RawTrajectoryRequest, hour: number) {
  const prefix = request.observations.filter(row => row.icu_hour <= hour)
  const current = prefix.find(row => row.icu_hour === hour)
  const previous = prefix.filter(row => row.icu_hour < hour).at(-1)
  return VITALS.map(name => {
    const last = prefix.findLast(row => row.measurements[name] != null)
    const value = current?.measurements[name] ?? null
    const previousValue = previous?.measurements[name] ?? null
    return {
      name, value, previousValue,
      change: value !== null && previousValue !== null ? value - previousValue : null,
      age: last ? hour - last.icu_hour : null,
      lastValue: last?.measurements[name] ?? null,
      status: value !== null ? 'observed_now' : last ? 'not_current' : 'never_observed',
      points: prefix.map(row => ({ hour: row.icu_hour, value: row.measurements[name] ?? null })),
    }
  })
}
