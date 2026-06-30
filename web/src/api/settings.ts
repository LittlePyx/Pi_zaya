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

export type LlmReadinessTarget = 'text' | 'vision'
export type LlmReadinessSeverity = 'ok' | 'warning' | 'error'
export type LlmReadinessStatus = 'missing' | 'fallback' | 'configured' | 'ok' | 'failed'

export interface LlmLastTest {
  ok: boolean
  checked_at: number
  error?: string
  error_type?: string
  reply?: string
}

export interface LlmProviderReadiness {
  target: LlmReadinessTarget
  has_api_key: boolean
  base_url: string
  model: string
  uses_text_fallback?: boolean
  status: LlmReadinessStatus
  severity: LlmReadinessSeverity
  reason: string
  last_test?: LlmLastTest | null
}

export interface LlmReadinessPayload {
  providers: Record<LlmReadinessTarget, LlmProviderReadiness>
  overall: {
    status: LlmReadinessSeverity
    reason: string
    target?: LlmReadinessTarget | ''
  }
}

export type AppReadinessSeverity = 'ok' | 'warning' | 'error'

export interface AppReadinessItem {
  key: string
  status: AppReadinessSeverity
  severity: AppReadinessSeverity
  label: string
  detail?: string
  action?: string
}

export interface AppReadinessPayload {
  status: AppReadinessSeverity
  env: string
  production: boolean
  auth_required: boolean
  items: AppReadinessItem[]
  llm?: LlmReadinessPayload
  restore?: {
    acknowledged?: boolean
    latest?: {
      event?: string
      status?: string
      backup?: string
      created_at?: number
      ok?: boolean
      restart_required?: boolean
      components?: Record<string, boolean>
      errors?: string[]
      warnings?: string[]
    } | null
    acknowledgement?: {
      event?: string
      status?: string
      backup?: string
      created_at?: number
      ok?: boolean
      restart_required?: boolean
      components?: Record<string, boolean>
      errors?: string[]
      warnings?: string[]
    } | null
  }
}

export interface AuthStatusPayload {
  required: boolean
  configured: boolean
  authenticated: boolean
  env: string
  production: boolean
}

export interface SettingsPayload {
  model: string
  base_url: string
  has_api_key: boolean
  connection?: SettingsConnectionStatus
  readiness?: LlmReadinessPayload
  app_readiness?: AppReadinessPayload
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
  autoBackupEnabled?: boolean
  qualityDataSharingEnabled?: boolean
}

export interface SettingsUpdateResponse {
  ok: boolean
  quality_data_cleanup?: {
    ok: boolean
    removed: number
    error?: string
  }
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
  if (patch.autoBackupEnabled !== undefined) out.auto_backup_enabled = patch.autoBackupEnabled
  if (patch.qualityDataSharingEnabled !== undefined) out.quality_data_sharing_enabled = patch.qualityDataSharingEnabled
  return out
}

export const settingsApi = {
  get: () => api.get<SettingsPayload>('/api/settings'),
  update: (patch: SettingsPatch) =>
    api.patch<SettingsUpdateResponse>('/api/settings', toServerPatch(patch)),
  pickDir: (target: 'pdf' | 'md', initialDir?: string) =>
    api.post<PickDirResponse>('/api/settings/pick-dir', {
      target,
      initial_dir: initialDir || '',
    }),
  testLlm: (target: 'text' | 'vision' = 'text', overrides: LlmTestOverrides = {}) =>
    api.post<{ ok: boolean; reply?: string; error?: string; error_type?: string; checked_at?: number }>('/api/settings/test-llm', {
      target,
      api_key: overrides.apiKey || undefined,
      base_url: overrides.baseUrl || undefined,
      model: overrides.model || undefined,
    }),
  readiness: () => api.get<LlmReadinessPayload>('/api/settings/readiness'),
  appReadiness: () => api.get<AppReadinessPayload>('/api/readiness'),
  health: () => api.get<{ status: string; env?: string; production?: boolean; auth?: { required: boolean; configured: boolean } }>('/api/health'),
  authStatus: () => api.get<AuthStatusPayload>('/api/auth/status'),
  authLogin: (token: string) => api.post<{ ok: boolean } & AuthStatusPayload>('/api/auth/login', { token }),
  authLogout: () => api.post<{ ok: boolean }>('/api/auth/logout'),
}
