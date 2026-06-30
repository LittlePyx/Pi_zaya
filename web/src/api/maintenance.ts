import { api, authFetch, responseError } from './client'

export interface MaintenanceBackupItem {
  name: string
  created_at: number
  label?: string
  size_bytes: number
  path?: string
}

export interface MaintenanceBackupList {
  items: MaintenanceBackupItem[]
}

export interface MaintenanceStatus {
  data_protection: {
    enabled: boolean
    status: string
    can_toggle?: boolean
    manual_backup_available?: boolean
    backup_count: number
    latest_backup?: MaintenanceBackupItem | null
  }
  auto_backup: {
    enabled: boolean
    strict: boolean
    min_interval_s: number
    source?: string
    locked?: boolean
  }
  backups: {
    count: number
    latest?: MaintenanceBackupItem | null
    keep: number
    directory?: string
  }
}

export interface MaintenanceBackupVerification {
  ok: boolean
  name: string
  created_at?: number
  label?: string
  size_bytes?: number
  errors?: string[]
  warnings?: string[]
  checks?: Record<string, unknown>
  verified_at?: number
}

export interface MaintenanceBackupCleanup {
  ok: boolean
  keep: number
  before: number
  deleted: number
  failed: number
  dry_run: boolean
  items?: Array<{ name: string; size_bytes?: number; created_at?: number }>
  errors?: Array<{ name: string; error?: string }>
}

export interface MaintenanceBackupRestoreDryRun {
  ok: boolean
  can_restore: boolean
  name: string
  created_at?: number
  label?: string
  size_bytes?: number
  extracted_file_count?: number
  destinations?: Array<{
    kind?: string
    label?: string
    archive?: string
    target?: string
    source_exists?: boolean
    target_exists?: boolean
    source_file_count?: number
    source_size_bytes?: number
    action?: string
  }>
  errors?: string[]
  warnings?: string[]
  restore_steps?: string[]
  checked_at?: number
}

export interface MaintenanceBackupRestoreResult {
  ok: boolean
  name: string
  status?: string
  expected_confirmation?: string
  components?: Record<string, boolean>
  pre_restore_backup?: MaintenanceBackupItem | null
  restored?: Array<{ kind?: string; target?: string; size_bytes?: number; file_count?: number; warnings?: string[] }>
  errors?: string[]
  warnings?: string[]
  restart_required?: boolean
  audit_path?: string
}

export interface MaintenanceRestoreReviewAck {
  ok: boolean
  status: string
  backup?: string
  restore_created_at?: number
  acknowledged_at?: number
  audit_path?: string
  errors?: string[]
}

export interface MaintenanceRestoreAuditEvent {
  event: string
  status: string
  ok: boolean
  backup?: string
  created_at?: number
  restore_created_at?: number
  restart_required?: boolean
  components?: Record<string, boolean>
  checks?: Record<string, boolean>
  errors?: string[]
  warnings?: string[]
  restored_count?: number
  pre_restore_backup?: string
}

export interface MaintenanceRestoreAuditList {
  items: MaintenanceRestoreAuditEvent[]
}

function filenameFromDisposition(header: string | null, fallback: string) {
  const raw = String(header || '')
  const match = raw.match(/filename="?([^";]+)"?/i)
  return match?.[1] || fallback
}

async function downloadUrl(url: string, fallbackName: string) {
  const res = await authFetch(url, { method: 'GET' })
  if (!res.ok) {
    throw await responseError(res)
  }
  const blob = await res.blob()
  const filename = filenameFromDisposition(res.headers.get('content-disposition'), fallbackName)
  const href = URL.createObjectURL(blob)
  try {
    const a = document.createElement('a')
    a.href = href
    a.download = filename
    document.body.appendChild(a)
    a.click()
    a.remove()
  } finally {
    window.setTimeout(() => URL.revokeObjectURL(href), 2000)
  }
}

export const maintenanceApi = {
  status: () => api.get<MaintenanceStatus>('/api/maintenance/status'),
  listBackups: () => api.get<MaintenanceBackupList>('/api/maintenance/backups'),
  createBackup: (label = '') => api.post<MaintenanceBackupItem>('/api/maintenance/backups', { label }),
  verifyBackup: (name: string) => api.get<MaintenanceBackupVerification>(`/api/maintenance/backups/${encodeURIComponent(name)}/verify`),
  restoreDryRunBackup: (name: string) => api.get<MaintenanceBackupRestoreDryRun>(`/api/maintenance/backups/${encodeURIComponent(name)}/restore-dry-run`),
  restoreBackup: (name: string, confirm: string) => api.post<MaintenanceBackupRestoreResult>(`/api/maintenance/backups/${encodeURIComponent(name)}/restore`, {
    confirm,
    create_pre_restore_backup: true,
  }),
  acknowledgeRestoreReview: () => api.post<MaintenanceRestoreReviewAck>('/api/maintenance/restore-review/acknowledge', {
    checks: {
      api_restarted: true,
      api_keys_checked: true,
      knowledge_base_checked: true,
      chat_history_checked: true,
      library_data_checked: true,
    },
  }),
  restoreAudit: (limit = 12) => api.get<MaintenanceRestoreAuditList>(`/api/maintenance/restore-audit?limit=${encodeURIComponent(String(limit))}`),
  cleanupBackups: (keep = 30) => api.post<MaintenanceBackupCleanup>('/api/maintenance/backups/cleanup', { keep }),
  downloadBackup: (name: string) => downloadUrl(`/api/maintenance/backups/${encodeURIComponent(name)}`, name),
  downloadDiagnostics: () => downloadUrl('/api/maintenance/diagnostics/export', 'pi-zaya-diagnostics.zip'),
}
