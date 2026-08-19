import { api } from './client'

export interface AppVersionPayload {
  name: string
  version: string
  version_source?: string
  commit?: string
  build_time?: string
  repository?: string
}

export interface AppLatestRelease {
  tag_name: string
  name: string
  html_url: string
  published_at?: string
  body?: string
  prerelease?: boolean
}

export interface AppUpdateCheckPayload {
  enabled: boolean
  status: 'ok' | 'unknown' | 'unavailable' | 'disabled'
  checked_at: number
  current: AppVersionPayload
  latest: AppLatestRelease | null
  update_available: boolean
  instructions: string[]
  error?: string
  retry_after?: number
}

export interface AppUpdateCheckOptions {
  refresh?: boolean
  cacheOnly?: boolean
}

export type OnboardingStep = 'connect_model' | 'prepare_document' | 'ask_question' | 'completed'

export interface AppOnboardingStatus {
  text_model_ready: boolean
  imported_document_count: number
  ready_document_count: number
  grounded_answer_count: number
  current_step: OnboardingStep
  completed: boolean
}

export const appApi = {
  version: () => api.get<AppVersionPayload>('/api/app/version'),
  onboardingStatus: () => api.get<AppOnboardingStatus>('/api/app/onboarding-status'),
  updateCheck: (options: AppUpdateCheckOptions = {}) => {
    const params = new URLSearchParams()
    if (options.refresh) params.set('refresh', 'true')
    if (options.cacheOnly) params.set('cache_only', 'true')
    const query = params.toString()
    return api.get<AppUpdateCheckPayload>(`/api/app/update-check${query ? `?${query}` : ''}`)
  },
}
