import { useSettingsStore } from '../../stores/settingsStore'

function bibliometricsLocalePatch(): Record<string, string> {
  const settings = useSettingsStore.getState()
  const refsLocale = settings.refsCardLocale === 'zh' || settings.refsCardLocale === 'en'
    ? settings.refsCardLocale
    : 'auto'
  const uiLocale = settings.uiLocale === 'en' ? 'en' : 'zh'
  const targetLocale = refsLocale === 'auto' ? uiLocale : refsLocale
  return {
    refsCardLocale: refsLocale,
    refs_card_locale: refsLocale,
    uiLocale,
    ui_locale: uiLocale,
    targetLocale,
    target_locale: targetLocale,
    renderLocale: targetLocale,
    render_locale: targetLocale,
  }
}

export function withBibliometricsLocale(meta: Record<string, unknown>): Record<string, unknown> {
  return {
    ...meta,
    ...bibliometricsLocalePatch(),
  }
}
