import { create } from 'zustand'
import { settingsApi, type SettingsPatch } from '../api/settings'

const MAX_TOKENS_MIN = 512
const MAX_TOKENS_MAX = 3072

function clampMaxTokens(value: unknown): number {
  const n = Number(value)
  if (!Number.isFinite(n)) return 1216
  return Math.max(MAX_TOKENS_MIN, Math.min(MAX_TOKENS_MAX, Math.round(n)))
}

interface SettingsState {
  topK: number
  temperature: number
  maxTokens: number
  deepRead: boolean
  showContext: boolean
  answerContractV1: boolean
  answerDepthAuto: boolean
  answerModeHint: string
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
  answerContractV1: false,
  answerDepthAuto: true,
  answerModeHint: '',
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
        maxTokens: clampMaxTokens(p.max_tokens),
        deepRead: !!p.deep_read,
        showContext: !!p.show_context,
        answerContractV1: !!p.answer_contract_v1,
        answerDepthAuto: p.answer_depth_auto !== false,
        answerModeHint: String(p.answer_mode_hint || ''),
        pdfDir: String(p.pdf_dir || ''),
        mdDir: String(p.md_dir || ''),
        theme: (p.theme as 'light' | 'dark') || 'dark',
        loaded: true,
      })
    } catch { /* ignore */ }
  },

  update: async (patch: SettingsPatch) => {
    const patchToSend: SettingsPatch = { ...patch }
    const localPatch: Partial<SettingsState> = {}
    if (patch.topK !== undefined) localPatch.topK = patch.topK
    if (patch.temperature !== undefined) localPatch.temperature = patch.temperature
    if (patch.maxTokens !== undefined) {
      const clamped = clampMaxTokens(patch.maxTokens)
      localPatch.maxTokens = clamped
      patchToSend.maxTokens = clamped
    }
    if (patch.deepRead !== undefined) localPatch.deepRead = patch.deepRead
    if (patch.showContext !== undefined) localPatch.showContext = patch.showContext
    if (patch.answerContractV1 !== undefined) localPatch.answerContractV1 = patch.answerContractV1
    if (patch.answerDepthAuto !== undefined) localPatch.answerDepthAuto = patch.answerDepthAuto
    if (patch.answerModeHint !== undefined) localPatch.answerModeHint = patch.answerModeHint
    if (patch.theme !== undefined) localPatch.theme = patch.theme
    if (patch.pdfDir !== undefined) localPatch.pdfDir = patch.pdfDir
    if (patch.mdDir !== undefined) localPatch.mdDir = patch.mdDir
    set(localPatch)
    await settingsApi.update(patchToSend).catch(() => {})
  },

  toggleTheme: () => {
    const next = get().theme === 'dark' ? 'light' : 'dark'
    set({ theme: next })
    settingsApi.update({ theme: next }).catch(() => {})
  },
}))
