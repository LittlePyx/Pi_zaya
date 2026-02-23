import { create } from 'zustand'
import { settingsApi } from '../api/settings'

interface SettingsState {
  topK: number
  temperature: number
  maxTokens: number
  deepRead: boolean
  showContext: boolean
  theme: 'light' | 'dark'
  model: string
  hasApiKey: boolean
  loaded: boolean
  load: () => Promise<void>
  update: (patch: Partial<SettingsState>) => Promise<void>
  toggleTheme: () => void
}

export const useSettingsStore = create<SettingsState>((set, get) => ({
  topK: 6,
  temperature: 0.2,
  maxTokens: 1216,
  deepRead: false,
  showContext: false,
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
        theme: (p.theme as 'light' | 'dark') || 'dark',
        loaded: true,
      })
    } catch { /* ignore */ }
  },

  update: async (patch) => {
    set(patch)
    await settingsApi.update(patch).catch(() => {})
  },

  toggleTheme: () => {
    const next = get().theme === 'dark' ? 'light' : 'dark'
    set({ theme: next })
    settingsApi.update({ theme: next }).catch(() => {})
  },
}))
