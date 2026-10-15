export const QUALITY_DIAGNOSTICS_SESSION_KEY = 'kb.internal.showQualityDiagnostics'

const VITE_ENV = (typeof import.meta.env === 'object' ? import.meta.env : {}) as Record<string, string | boolean | undefined>
const USER_QUALITY_DIAGNOSTICS_ENABLED = VITE_ENV.VITE_SHOW_USER_QUALITY_DIAGNOSTICS === '1'
const INTERNAL_QUALITY_DIAGNOSTICS_BUILD =
  VITE_ENV.VITE_ENABLE_INTERNAL_DEBUG === '1'
  || VITE_ENV.VITE_ENABLE_INTERNAL_ROUTES === '1'

export function qualityDiagnosticsBuildEnabled(): boolean {
  return USER_QUALITY_DIAGNOSTICS_ENABLED || INTERNAL_QUALITY_DIAGNOSTICS_BUILD
}

export function qualityStatusVisible(): boolean {
  return true
}

export function qualityDiagnosticsVisible(): boolean {
  if (USER_QUALITY_DIAGNOSTICS_ENABLED) return true
  if (!INTERNAL_QUALITY_DIAGNOSTICS_BUILD) return false
  try {
    return typeof window !== 'undefined' && window.sessionStorage.getItem(QUALITY_DIAGNOSTICS_SESSION_KEY) === '1'
  } catch {
    return false
  }
}
