import type { RawTrajectoryRequest, TrajectoryResponse } from "../types/trajectory";

export const RAW_HOURLY = "/api/v1/research/sepsis/trajectory/raw-hourly";

export async function fetchTrajectory(req: RawTrajectoryRequest): Promise<TrajectoryResponse> {
  const res = await fetch(RAW_HOURLY, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (res.status === 503) throw new Error("Backend reachable, but the research model artifact is unavailable (HTTP 503).");
  if (res.status === 422) throw new Error("Backend rejected the request or artifact metadata as incompatible (HTTP 422).");
  if (!res.ok) throw new Error(`Backend error (HTTP ${res.status}).`);
  return (await res.json()) as TrajectoryResponse;
}
