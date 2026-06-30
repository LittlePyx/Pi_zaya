const TRUE_VALUES = new Set(['1', 'true', 'yes', 'on'])
const INTERNAL_DEBUG_BUILD_ENABLED = import.meta.env.VITE_ENABLE_INTERNAL_DEBUG === '1'

function isTrueFlag(value: unknown) {
  return TRUE_VALUES.has(String(value || '').trim().toLowerCase())
}

export function internalDebugEnvEnabled() {
  return INTERNAL_DEBUG_BUILD_ENABLED
}

export function internalDebugBrowserEnabled() {
  if (!INTERNAL_DEBUG_BUILD_ENABLED) return false
  if (typeof window === 'undefined') return false
  try {
    const params = new URLSearchParams(window.location.search)
    return (
      isTrueFlag(params.get('debug'))
      || isTrueFlag(params.get('perf'))
      || isTrueFlag(params.get('kb_debug'))
      || window.localStorage.getItem('kb:chat-perf-panel') === '1'
      || window.sessionStorage.getItem('kb.internal.debug') === '1'
      || window.sessionStorage.getItem('kb.internal.showSettingsDiagnostics') === '1'
      || window.sessionStorage.getItem('kb.internal.showQualityDiagnostics') === '1'
    )
  } catch {
    return false
  }
}

export function internalSettingsToolsVisible() {
  if (import.meta.env.VITE_SHOW_INTERNAL_SETTINGS === '1') return true
  if (!INTERNAL_DEBUG_BUILD_ENABLED) return false
  if (typeof window === 'undefined') return false
  try {
    return (
      window.sessionStorage.getItem('kb.internal.showSettingsDiagnostics') === '1'
      || window.sessionStorage.getItem('kb.internal.showQualityDiagnostics') === '1'
    )
  } catch {
    return false
  }
}

export function internalDebugEnabled() {
  return internalDebugEnvEnabled()
}
