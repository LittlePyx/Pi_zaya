import { api } from './client'

export interface SettingsPayload {
  model: string
  base_url: string
  has_api_key: boolean
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
}

export interface PickDirResponse {
  ok: boolean
  path: string | null
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
  testLlm: () => api.post<{ ok: boolean; reply?: string; error?: string }>('/api/settings/test-llm'),
  health: () => api.get<{ status: string }>('/api/health'),
}
