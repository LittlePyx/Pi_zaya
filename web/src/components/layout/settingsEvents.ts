export type ApiSettingsTarget = 'text' | 'vision'

export interface OpenSettingsDetail {
  target?: ApiSettingsTarget | ''
}

export const OPEN_SETTINGS_EVENT = 'kb:open-settings'

export function apiSettingsTargetFromUnknown(value: unknown): ApiSettingsTarget | '' {
  const raw = String(value || '').trim()
  return raw === 'text' || raw === 'vision' ? raw : ''
}

export function dispatchOpenSettings(target?: ApiSettingsTarget | '') {
  if (typeof window === 'undefined') return
  window.dispatchEvent(new CustomEvent<OpenSettingsDetail>(OPEN_SETTINGS_EVENT, {
    detail: { target: apiSettingsTargetFromUnknown(target) },
  }))
}
