export type WorkspaceSource = 'synthetic' | 'api' | 'api-synthetic-inputs' | 'retrospective-export'

export const sourceBadge: Record<WorkspaceSource, string> = {
  synthetic: 'Synthetic example', api: 'API response · research only',
  'api-synthetic-inputs': 'Synthetic inputs · real API outputs',
  'retrospective-export': 'Retrospective model outputs · local export',
}
export const sourceDescription: Record<WorkspaceSource, string> = {
  synthetic: 'A documented synthetic example. Explore without a model or patient data.',
  api: 'Displaying an API response. This is research inference, not a live clinical monitor.',
  'api-synthetic-inputs': 'Synthetic inputs sent to the unchanged trained model. API predictions, not patient records or cohort evaluation.',
  'retrospective-export': 'Imported retrospective model outputs, not synthetic examples or a live stream. Export provenance is declared by the file; React does not independently authenticate it. Derived trajectories remain sensitive and stay in browser memory.',
}
export const retainedSource: Record<WorkspaceSource, string> = {
  synthetic: 'Still showing the explicitly labeled synthetic example.',
  api: 'Still showing the last successful API response.',
  'api-synthetic-inputs': 'Still showing the last successful API response on synthetic inputs.',
  'retrospective-export': 'Still showing the last successfully imported retrospective export.',
}
