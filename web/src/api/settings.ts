import { api } from './client'

export interface ConnectionProviderStatus {
  has_api_key: boolean
  base_url: string
  model: string
  uses_text_fallback?: boolean
}

export interface SettingsConnectionStatus {
  text: ConnectionProviderStatus
  vision: ConnectionProviderStatus
  auto_route?: boolean
}

export interface SettingsPayload {
  model: string
  base_url: string
  has_api_key: boolean
  connection?: SettingsConnectionStatus
  db_dir: string
  prefs: Record<string, unknown>
}

export interface SettingsPatch {
  topK?: number
  temperature?: number
  maxTokens?: number
  deepRead?: boolean
  showContext?: boolean
  theme?: 'light' | 'dark'
  pdfDir?: string
  mdDir?: string
  answerContractV1?: boolean
  answerDepthAuto?: boolean
  answerModeHint?: string
  answerOutputMode?: string
  refsCardLocale?: 'auto' | 'zh' | 'en'
  uiLocale?: 'zh' | 'en'
  sidebarCollapsed?: boolean
  textApiKey?: string
  textBaseUrl?: string
  textModel?: string
  visionApiKey?: string
  visionBaseUrl?: string
  visionModel?: string
}

export interface PickDirResponse {
  ok: boolean
  path: string | null
}

export interface LlmTestOverrides {
  apiKey?: string
  baseUrl?: string
  model?: string
}

function toServerPatch(patch: SettingsPatch) {
  const out: Record<string, unknown> = {}
  if (patch.topK !== undefined) out.top_k = patch.topK
  if (patch.temperature !== undefined) out.temperature = patch.temperature
  if (patch.maxTokens !== undefined) out.max_tokens = patch.maxTokens
  if (patch.deepRead !== undefined) out.deep_read = patch.deepRead
  if (patch.showContext !== undefined) out.show_context = patch.showContext
  if (patch.theme !== undefined) out.theme = patch.theme
  if (patch.pdfDir !== undefined) out.pdf_dir = patch.pdfDir
  if (patch.mdDir !== undefined) out.md_dir = patch.mdDir
  if (patch.answerContractV1 !== undefined) out.answer_contract_v1 = patch.answerContractV1
  if (patch.answerDepthAuto !== undefined) out.answer_depth_auto = patch.answerDepthAuto
  if (patch.answerModeHint !== undefined) out.answer_mode_hint = patch.answerModeHint
  if (patch.answerOutputMode !== undefined) out.answer_output_mode = patch.answerOutputMode
  if (patch.refsCardLocale !== undefined) out.refs_card_locale = patch.refsCardLocale
  if (patch.uiLocale !== undefined) out.ui_locale = patch.uiLocale
  if (patch.sidebarCollapsed !== undefined) out.sidebar_collapsed = patch.sidebarCollapsed
  if (patch.textApiKey !== undefined) out.text_api_key = patch.textApiKey
  if (patch.textBaseUrl !== undefined) out.text_base_url = patch.textBaseUrl
  if (patch.textModel !== undefined) out.text_model = patch.textModel
  if (patch.visionApiKey !== undefined) out.vision_api_key = patch.visionApiKey
  if (patch.visionBaseUrl !== undefined) out.vision_base_url = patch.visionBaseUrl
  if (patch.visionModel !== undefined) out.vision_model = patch.visionModel
  return out
}

export const settingsApi = {
  get: () => api.get<SettingsPayload>('/api/settings'),
  update: (patch: SettingsPatch) =>
    api.patch('/api/settings', toServerPatch(patch)),
  pickDir: (target: 'pdf' | 'md', initialDir?: string) =>
    api.post<PickDirResponse>('/api/settings/pick-dir', {
      target,
      initial_dir: initialDir || '',
    }),
  testLlm: (target: 'text' | 'vision' = 'text', overrides: LlmTestOverrides = {}) =>
    api.post<{ ok: boolean; reply?: string; error?: string }>('/api/settings/test-llm', {
      target,
      api_key: overrides.apiKey || undefined,
      base_url: overrides.baseUrl || undefined,
      model: overrides.model || undefined,
    }),
  health: () => api.get<{ status: string }>('/api/health'),
}
