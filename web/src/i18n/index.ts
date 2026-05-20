import { useSettingsStore } from '../stores/settingsStore'
import { S as zh } from './zh'
import { S as en } from './en'

export type StringMap = { [K in keyof typeof zh]: string } & Record<string, string>

export function useT(): StringMap {
  const locale = useSettingsStore((s) => s.uiLocale)
  return locale === 'zh' ? zh as StringMap : en as StringMap
}
