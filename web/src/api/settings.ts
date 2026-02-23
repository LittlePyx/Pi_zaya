import { api } from './client'

export const settingsApi = {
  get: () => api.get<{
    model: string
    base_url: string
    has_api_key: boolean
    db_dir: string
    prefs: Record<string, unknown>
  }>('/api/settings'),
  update: (patch: Record<string, unknown>) =>
    api.patch('/api/settings', patch),
  testLlm: () => api.post<{ ok: boolean; reply?: string; error?: string }>('/api/settings/test-llm'),
  health: () => api.get<{ status: string }>('/api/health'),
}
