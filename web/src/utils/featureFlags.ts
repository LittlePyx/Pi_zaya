type ViteEnv = Record<string, unknown>

const VITE_ENV = ((import.meta as ImportMeta & { env?: ViteEnv }).env || {}) as ViteEnv

function envTruthy(value: unknown): boolean {
  return ['1', 'true', 'yes', 'on'].includes(String(value ?? '').trim().toLowerCase())
}

/**
 * The project evidence-matrix workspace is temporarily withheld from ordinary
 * builds while its synthesis and usability quality contract is being revised.
 * Matrix-dependent brief, gap, and project-status entry points use the same
 * switch so an ordinary user cannot enter a workflow that has no usable first
 * step. Keep the implementation available to explicit internal/test builds so
 * the existing API, saved data, and regression coverage remain intact.
 */
export function evidenceMatrixWorkspaceBuildEnabled(env: ViteEnv = VITE_ENV): boolean {
  return envTruthy(env.VITE_ENABLE_EVIDENCE_MATRIX_WORKSPACE)
}

export const EVIDENCE_MATRIX_WORKSPACE_ENABLED = evidenceMatrixWorkspaceBuildEnabled()
