export type ApiSettingsTarget = 'text' | 'vision'
export type SettingsFocusTarget = ApiSettingsTarget | 'updates'

export interface OpenSettingsDetail {
  target?: SettingsFocusTarget | ''
}

export const OPEN_SETTINGS_EVENT = 'kb:open-settings'

export function apiSettingsTargetFromUnknown(value: unknown): ApiSettingsTarget | '' {
  const raw = String(value || '').trim()
  return raw === 'text' || raw === 'vision' ? raw : ''
}

export function settingsFocusTargetFromUnknown(value: unknown): SettingsFocusTarget | '' {
  const raw = String(value || '').trim()
  if (raw === 'updates') return 'updates'
  return apiSettingsTargetFromUnknown(raw)
}

export function dispatchOpenSettings(target?: SettingsFocusTarget | '') {
  if (typeof window === 'undefined') return
  window.dispatchEvent(new CustomEvent<OpenSettingsDetail>(OPEN_SETTINGS_EVENT, {
    detail: { target: settingsFocusTargetFromUnknown(target) },
  }))
}
