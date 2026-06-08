import { create } from 'zustand'
import { settingsApi, type AppReadinessPayload, type LlmReadinessPayload, type SettingsPatch } from '../api/settings'

const MAX_TOKENS_MIN = 512
const MAX_TOKENS_MAX = 3072
const THEME_STORAGE_KEY = 'kb_theme_mode'

function readInitialTheme(): 'light' | 'dark' {
  try {
    const raw = window.localStorage.getItem(THEME_STORAGE_KEY)
    if (raw === 'light' || raw === 'dark') return raw
  } catch { /* ignore */ }
  return 'dark'
}

function persistTheme(theme: 'light' | 'dark') {
  try {
    window.localStorage.setItem(THEME_STORAGE_KEY, theme)
  } catch { /* ignore */ }
}

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
  answerOutputMode: string
  refsCardLocale: 'auto' | 'zh' | 'en'
  pdfDir: string
  mdDir: string
  uiLocale: 'zh' | 'en'
  theme: 'light' | 'dark'
  sidebarCollapsed: boolean
  model: string
  hasApiKey: boolean
  textModel: string
  textBaseUrl: string
  hasTextApiKey: boolean
  visionModel: string
  visionBaseUrl: string
  hasVisionApiKey: boolean
  visionUsesTextFallback: boolean
  autoRoute: boolean
  llmReadiness: LlmReadinessPayload | null
  appReadiness: AppReadinessPayload | null
  loaded: boolean
  load: () => Promise<void>
  refreshReadiness: () => Promise<void>
  refreshAppReadiness: () => Promise<void>
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
  answerOutputMode: '',
  refsCardLocale: 'auto',
  pdfDir: '',
  mdDir: '',
  uiLocale: 'zh',
  theme: readInitialTheme(),
  sidebarCollapsed: false,
  model: '',
  hasApiKey: false,
  textModel: '',
  textBaseUrl: '',
  hasTextApiKey: false,
  visionModel: '',
  visionBaseUrl: '',
  hasVisionApiKey: false,
  visionUsesTextFallback: false,
  autoRoute: false,
  llmReadiness: null,
  appReadiness: null,
  loaded: false,

  load: async () => {
    try {
      const data = await settingsApi.get()
      const p = data.prefs || {}
      const nextTheme = (p.theme as 'light' | 'dark') || 'dark'
      const nextUiLocale = ((p.ui_locale as 'zh' | 'en') || 'zh')
      const rawRefsCardLocale = String(p.refs_card_locale || 'auto')
      const nextRefsCardLocale = rawRefsCardLocale === 'zh' || rawRefsCardLocale === 'en' ? rawRefsCardLocale : 'auto'
      const textStatus = data.connection?.text
      const visionStatus = data.connection?.vision
      const nextTextModel = String(textStatus?.model || data.model || '')
      const nextTextBaseUrl = String(textStatus?.base_url || data.base_url || '')
      const nextHasTextApiKey = Boolean(textStatus?.has_api_key ?? data.has_api_key)
      const nextVisionModel = String(visionStatus?.model || nextTextModel || '')
      const nextVisionBaseUrl = String(visionStatus?.base_url || nextTextBaseUrl || '')
      const nextHasVisionApiKey = Boolean(visionStatus?.has_api_key ?? nextHasTextApiKey)
      persistTheme(nextTheme)
      set({
        model: nextTextModel,
        hasApiKey: nextHasTextApiKey,
        textModel: nextTextModel,
        textBaseUrl: nextTextBaseUrl,
        hasTextApiKey: nextHasTextApiKey,
        visionModel: nextVisionModel,
        visionBaseUrl: nextVisionBaseUrl,
        hasVisionApiKey: nextHasVisionApiKey,
        visionUsesTextFallback: Boolean(visionStatus?.uses_text_fallback),
        autoRoute: Boolean(data.connection?.auto_route),
        llmReadiness: data.readiness || null,
        appReadiness: data.app_readiness || null,
        topK: (p.top_k as number) || 6,
        temperature: (p.temperature as number) ?? 0.2,
        maxTokens: clampMaxTokens(p.max_tokens),
        deepRead: !!p.deep_read,
        showContext: !!p.show_context,
        answerContractV1: !!p.answer_contract_v1,
        answerDepthAuto: p.answer_depth_auto !== false,
        answerModeHint: String(p.answer_mode_hint || ''),
        answerOutputMode: String(p.answer_output_mode || ''),
        refsCardLocale: nextRefsCardLocale,
        pdfDir: String(p.pdf_dir || ''),
        mdDir: String(p.md_dir || ''),
        uiLocale: nextUiLocale,
        theme: nextTheme,
        sidebarCollapsed: Boolean(p.sidebar_collapsed),
        loaded: true,
      })
      if (!String((p as Record<string, unknown>).ui_locale || '').trim()) {
        settingsApi.update({ uiLocale: nextUiLocale }).catch(() => {})
      }
    } catch { /* ignore */ }
  },

  refreshReadiness: async () => {
    try {
      const readiness = await settingsApi.readiness()
      const text = readiness.providers.text
      const vision = readiness.providers.vision
      set({
        llmReadiness: readiness,
        hasTextApiKey: Boolean(text?.has_api_key),
        hasApiKey: Boolean(text?.has_api_key),
        model: String(text?.model || ''),
        textModel: String(text?.model || ''),
        textBaseUrl: String(text?.base_url || ''),
        hasVisionApiKey: Boolean(vision?.has_api_key),
        visionModel: String(vision?.model || ''),
        visionBaseUrl: String(vision?.base_url || ''),
        visionUsesTextFallback: Boolean(vision?.uses_text_fallback),
      })
    } catch { /* ignore */ }
  },

  refreshAppReadiness: async () => {
    try {
      const appReadiness = await settingsApi.appReadiness()
      set({ appReadiness })
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
    if (patch.answerOutputMode !== undefined) localPatch.answerOutputMode = patch.answerOutputMode
    if (patch.refsCardLocale !== undefined) localPatch.refsCardLocale = patch.refsCardLocale
    if (patch.theme !== undefined) {
      localPatch.theme = patch.theme
      persistTheme(patch.theme)
    }
    if (patch.pdfDir !== undefined) localPatch.pdfDir = patch.pdfDir
    if (patch.mdDir !== undefined) localPatch.mdDir = patch.mdDir
    if (patch.uiLocale !== undefined) localPatch.uiLocale = patch.uiLocale
    if (patch.sidebarCollapsed !== undefined) localPatch.sidebarCollapsed = patch.sidebarCollapsed
    if (patch.textBaseUrl !== undefined) localPatch.textBaseUrl = patch.textBaseUrl
    if (patch.textModel !== undefined) {
      localPatch.textModel = patch.textModel
      localPatch.model = patch.textModel
    }
    if (patch.textApiKey !== undefined && patch.textApiKey.trim()) {
      localPatch.hasTextApiKey = true
      localPatch.hasApiKey = true
    }
    if (patch.visionBaseUrl !== undefined) localPatch.visionBaseUrl = patch.visionBaseUrl
    if (patch.visionModel !== undefined) localPatch.visionModel = patch.visionModel
    if (patch.visionApiKey !== undefined && patch.visionApiKey.trim()) {
      localPatch.hasVisionApiKey = true
      localPatch.visionUsesTextFallback = false
    }
    set(localPatch)
    await settingsApi.update(patchToSend).catch(() => {})
  },

  toggleTheme: () => {
    const next = get().theme === 'dark' ? 'light' : 'dark'
    persistTheme(next)
    set({ theme: next })
    settingsApi.update({ theme: next }).catch(() => {})
  },
}))
