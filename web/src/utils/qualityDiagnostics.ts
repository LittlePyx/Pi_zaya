export const QUALITY_DIAGNOSTICS_SESSION_KEY = 'kb.internal.showQualityDiagnostics'

const USER_QUALITY_DIAGNOSTICS_ENABLED = import.meta.env.VITE_SHOW_USER_QUALITY_DIAGNOSTICS === '1'
const INTERNAL_QUALITY_DIAGNOSTICS_BUILD =
  import.meta.env.VITE_ENABLE_INTERNAL_DEBUG === '1'
  || import.meta.env.VITE_ENABLE_INTERNAL_ROUTES === '1'

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
