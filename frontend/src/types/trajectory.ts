/** Compatibility bridge for components reused from origin/frontend/trajectory-ui
 * at e08593d. The v0.6 client remains the source of transport types.
 */
import type { DecisionState, RawTrajectoryRequest } from '../api/contracts.ts'

export type HourPrediction = Pick<DecisionState,
  'icu_hour' | 'static_probability' | 'temporal_probability' | 'final_probability' |
  'temporal_weight' | 'static_weight' | 'status' | 'reason'>
export type RawHourObservation = RawTrajectoryRequest['observations'][number]
export type { RawTrajectoryRequest }
export const VITALS = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp'] as const

export const UNITS: Record<string, string> = { HR: "bpm", O2Sat: "%", Temp: "°C", SBP: "mmHg", MAP: "mmHg", DBP: "mmHg", Resp: "/min" }
