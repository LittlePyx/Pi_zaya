import { create } from 'zustand'
import { appApi, type AppUpdateCheckOptions, type AppUpdateCheckPayload } from '../api/app'
import { settingsApi, type AppReadinessPayload, type LlmReadinessPayload, type SettingsPatch } from '../api/settings'

const MAX_TOKENS_MIN = 512
const MAX_TOKENS_MAX = 3072
const THEME_STORAGE_KEY = 'kb_theme_mode'
const APP_UPDATE_AUTO_STORAGE_KEY = 'kb_app_update_last_auto_check'
const APP_UPDATE_PAYLOAD_STORAGE_KEY = 'kb_app_update_payload'
const APP_UPDATE_AUTO_COOLDOWN_MS = 6 * 60 * 60 * 1000
const APP_UPDATE_PAYLOAD_TTL_MS = 6 * 60 * 60 * 1000
const APP_UPDATE_ERROR_PAYLOAD_TTL_MS = 5 * 60 * 1000

type AppUpdateRefreshInput = boolean | (AppUpdateCheckOptions & { auto?: boolean })

let appUpdateInFlight: Promise<AppUpdateCheckPayload> | null = null
let appUpdateInFlightKey = ''

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

function readStoredNumber(key: string): number {
  try {
    const raw = window.localStorage.getItem(key)
    const n = Number(raw)
    return Number.isFinite(n) ? n : 0
  } catch {
    return 0
  }
}

function writeStoredNumber(key: string, value: number) {
  try {
    window.localStorage.setItem(key, String(Math.round(value)))
  } catch { /* ignore */ }
}

function parseRetryAfterFromError(error: string): number {
  const match = String(error || '').match(/Try again after\s+(\d{4})-(\d{2})-(\d{2})\s+(\d{2}):(\d{2}):(\d{2})/i)
  if (!match) return 0
  const [, year, month, day, hour, minute, second] = match
  const ms = new Date(
    Number(year),
    Number(month) - 1,
    Number(day),
    Number(hour),
    Number(minute),
    Number(second),
  ).getTime()
  return Number.isFinite(ms) ? ms : 0
}

function appUpdateRetryAfterMs(payload: AppUpdateCheckPayload): number {
  const retryAfter = Number(payload.retry_after || 0)
  if (Number.isFinite(retryAfter) && retryAfter > 0) return retryAfter * 1000
  return parseRetryAfterFromError(String(payload.error || ''))
}

function appUpdateCheckedAtMs(payload: AppUpdateCheckPayload): number {
  const checkedAt = Number(payload.checked_at || 0)
  return Number.isFinite(checkedAt) && checkedAt > 0 ? checkedAt * 1000 : 0
}

function shouldDiscardCachedAppUpdate(payload: AppUpdateCheckPayload): boolean {
  const now = Date.now()
  const status = String(payload.status || '').trim().toLowerCase()
  const checkedAtMs = appUpdateCheckedAtMs(payload)
  const ageMs = checkedAtMs > 0 ? now - checkedAtMs : Number.POSITIVE_INFINITY
  if (status === 'unknown' && String(payload.error || '').toLowerCase().includes('cached update')) return true
  if (status === 'unavailable') {
    const retryAfterMs = appUpdateRetryAfterMs(payload)
    if (retryAfterMs > 0) return now >= retryAfterMs
    return ageMs > APP_UPDATE_ERROR_PAYLOAD_TTL_MS
  }
  return ageMs > APP_UPDATE_PAYLOAD_TTL_MS
}

function readCachedAppUpdate(): AppUpdateCheckPayload | null {
  try {
    const raw = window.localStorage.getItem(APP_UPDATE_PAYLOAD_STORAGE_KEY)
    if (!raw) return null
    const parsed = JSON.parse(raw) as AppUpdateCheckPayload
    if (!parsed || typeof parsed !== 'object') return null
    if (shouldDiscardCachedAppUpdate(parsed)) {
      window.localStorage.removeItem(APP_UPDATE_PAYLOAD_STORAGE_KEY)
      return null
    }
    return parsed
  } catch {
    return null
  }
}

function persistAppUpdate(payload: AppUpdateCheckPayload) {
  try {
    window.localStorage.setItem(APP_UPDATE_PAYLOAD_STORAGE_KEY, JSON.stringify(payload))
  } catch { /* ignore */ }
}

function normalizeUpdateInput(input: AppUpdateRefreshInput = {}): AppUpdateCheckOptions & { auto: boolean } {
  if (typeof input === 'boolean') return { refresh: input, auto: false }
  return { ...input, auto: Boolean(input.auto) }
}

function shouldSkipAutoUpdateCheck(): boolean {
  const lastChecked = readStoredNumber(APP_UPDATE_AUTO_STORAGE_KEY)
  return Date.now() - lastChecked < APP_UPDATE_AUTO_COOLDOWN_MS
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
  appUpdate: AppUpdateCheckPayload | null
  loaded: boolean
  load: () => Promise<void>
  refreshReadiness: () => Promise<void>
  refreshAppReadiness: () => Promise<void>
  refreshAppUpdate: (options?: AppUpdateRefreshInput) => Promise<void>
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
  appUpdate: readCachedAppUpdate(),
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

  refreshAppUpdate: async (input = {}) => {
    const options = normalizeUpdateInput(input)
    if (options.auto && shouldSkipAutoUpdateCheck()) {
      return
    }
    if (options.auto) {
      writeStoredNumber(APP_UPDATE_AUTO_STORAGE_KEY, Date.now())
    }
    const requestOptions = {
      refresh: Boolean(options.refresh),
      cacheOnly: Boolean(options.cacheOnly),
    }
    const requestKey = `${requestOptions.refresh ? 'refresh' : 'cached'}:${requestOptions.cacheOnly ? 'cache-only' : 'network'}`
    const request = options.refresh
      ? appApi.updateCheck(requestOptions)
      : (appUpdateInFlight && appUpdateInFlightKey === requestKey)
        ? appUpdateInFlight
        : (appUpdateInFlightKey = requestKey, appUpdateInFlight = appApi.updateCheck(requestOptions).finally(() => {
          appUpdateInFlightKey = ''
          appUpdateInFlight = null
        }))
    const appUpdate = await request
    persistAppUpdate(appUpdate)
    set({ appUpdate })
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
