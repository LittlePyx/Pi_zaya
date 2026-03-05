import { create } from 'zustand'
import { settingsApi, type SettingsPatch } from '../api/settings'

interface SettingsState {
  topK: number
  temperature: number
  maxTokens: number
  deepRead: boolean
  showContext: boolean
  pdfDir: string
  mdDir: string
  theme: 'light' | 'dark'
  model: string
  hasApiKey: boolean
  loaded: boolean
  load: () => Promise<void>
  update: (patch: SettingsPatch) => Promise<void>
  toggleTheme: () => void
}

export const useSettingsStore = create<SettingsState>((set, get) => ({
  topK: 6,
  temperature: 0.2,
  maxTokens: 1216,
  deepRead: false,
  showContext: false,
  pdfDir: '',
  mdDir: '',
  theme: 'dark',
  model: '',
  hasApiKey: false,
  loaded: false,

  load: async () => {
    try {
      const data = await settingsApi.get()
      const p = data.prefs || {}
      set({
        model: data.model,
        hasApiKey: data.has_api_key,
        topK: (p.top_k as number) || 6,
        temperature: (p.temperature as number) ?? 0.2,
        maxTokens: (p.max_tokens as number) || 1216,
        deepRead: !!p.deep_read,
        showContext: !!p.show_context,
        pdfDir: String(p.pdf_dir || ''),
        mdDir: String(p.md_dir || ''),
        theme: (p.theme as 'light' | 'dark') || 'dark',
        loaded: true,
      })
    } catch { /* ignore */ }
  },

  update: async (patch: SettingsPatch) => {
    const localPatch: Partial<SettingsState> = {}
    if (patch.topK !== undefined) localPatch.topK = patch.topK
    if (patch.temperature !== undefined) localPatch.temperature = patch.temperature
    if (patch.maxTokens !== undefined) localPatch.maxTokens = patch.maxTokens
    if (patch.deepRead !== undefined) localPatch.deepRead = patch.deepRead
    if (patch.showContext !== undefined) localPatch.showContext = patch.showContext
    if (patch.theme !== undefined) localPatch.theme = patch.theme
    if (patch.pdfDir !== undefined) localPatch.pdfDir = patch.pdfDir
    if (patch.mdDir !== undefined) localPatch.mdDir = patch.mdDir
    set(localPatch)
    await settingsApi.update(patch).catch(() => {})
  },

  toggleTheme: () => {
    const next = get().theme === 'dark' ? 'light' : 'dark'
    set({ theme: next })
    settingsApi.update({ theme: next }).catch(() => {})
  },
}))
