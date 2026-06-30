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

export interface UserIssueOutboxSummary {
  total: number
  pending: number
  retryable: number
  sent: number
  latest_error?: string
  latest_attempts?: number
  next_attempt_at?: number
}

export interface UserIssueRemoteStatus {
  ok: boolean
  enabled: boolean
  remote_enabled: boolean
  remote_url_configured: boolean
  remote_url_host?: string
  remote_url_scheme?: string
  remote_url_has_valid_scheme?: boolean
  remote_url_has_valid_port?: boolean
  remote_url_has_credentials?: boolean
  remote_url_is_local?: boolean
  remote_url_local_allowed?: boolean
  remote_url_secure?: boolean
  remote_url_allowed?: boolean
  remote_block_reason?: string
  remote_token_configured: boolean
  remote_token_required?: boolean
  remote_unauthenticated_allowed?: boolean
  quality_data_sharing_enabled: boolean
  outbox: UserIssueOutboxSummary
}

export interface UserIssueRemoteTestResponse {
  ok: boolean
  enabled?: boolean
  status_code?: number
  error?: string
  remote?: Omit<UserIssueRemoteStatus, 'ok' | 'outbox'>
  outbox?: UserIssueOutboxSummary
}

export interface UserIssueOutboxFlushResponse {
  ok: boolean
  enabled: boolean
  sent: number
  failed: number
  summary: UserIssueOutboxSummary
}

export const userIssuesApi = {
  record: (body: UserIssuePayload) => api.post<{ ok: boolean; issue?: Record<string, unknown> }>('/api/user-issues', body),
  remoteStatus: () => api.get<UserIssueRemoteStatus>('/api/user-issues/remote/status'),
  testRemote: () => api.post<UserIssueRemoteTestResponse>('/api/user-issues/remote/test'),
  flushOutbox: (limit = 20) => api.post<UserIssueOutboxFlushResponse>(`/api/user-issues/outbox/flush?limit=${encodeURIComponent(String(limit))}`),
}
