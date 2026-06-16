import { api } from './client'

export interface UserIssuePayload {
  source?: string
  domain?: string
  severity?: 'info' | 'warning' | 'error' | string
  summary: string
  detail?: string
  route?: string
  context?: Record<string, unknown>
  payload?: Record<string, unknown>
  fingerprint?: string
}

export const userIssuesApi = {
  record: (body: UserIssuePayload) => api.post<{ ok: boolean; issue?: Record<string, unknown> }>('/api/user-issues', body),
}
