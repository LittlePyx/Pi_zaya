import { type ReactNode, useCallback, useEffect, useRef, useState } from 'react'
import { ApiOutlined, LockOutlined, ReloadOutlined } from '@ant-design/icons'
import { Alert, Button, Drawer, Input, Popconfirm, Select, Segmented, Slider, Space, Switch, Typography, message } from 'antd'
import { useSettingsStore } from '../../stores/settingsStore'
import { settingsApi, type AuthStatusPayload, type SettingsPatch } from '../../api/settings'
import { userIssuesApi, type UserIssueRemoteStatus } from '../../api/userIssues'
import { authGateBuildEnabled } from '../../api/authGate'
import { AUTH_REQUIRED_EVENT, setAccessToken } from '../../api/client'
import { maintenanceApi, type MaintenanceStatus } from '../../api/maintenance'
import { useT } from '../../i18n'
import type { SettingsFocusTarget } from './settingsEvents'
import { SettingsUpdatePanel } from './SettingsUpdatePanel'
import { internalSettingsToolsVisible } from '../../utils/internalDebug'

const { Text } = Typography
type LocalTestResult = { ok: boolean; checked_at: number; error_type?: string; transient: boolean }

function authGateEnabled() {
  return authGateBuildEnabled()
}

function appReadinessActionLabel(S: Record<string, string>, action: string | undefined) {
  const clean = String(action || '').trim()
  if (!clean) return ''
  return S[`settings_release_action_${clean}`] || clean.replace(/_/g, ' ')
}

function SettingsSection({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="kb-settings-section">
      <div className="kb-settings-section-title">{title}</div>
      <div className="kb-settings-section-body">{children}</div>
    </section>
  )
}

function SettingsRow({
  title,
  description,
  children,
}: {
  title: string
  description?: string
  children: ReactNode
}) {
  return (
    <div className="kb-settings-row">
      <div className="kb-settings-row-copy">
        <Text className="kb-settings-row-title">{title}</Text>
        {description ? <Text className="kb-settings-row-desc">{description}</Text> : null}
      </div>
      <div className="kb-settings-row-control">{children}</div>
    </div>
  )
}

function SettingsValue({ value, muted = false }: { value: ReactNode; muted?: boolean }) {
  return <span className={`kb-settings-value ${muted ? 'is-muted' : ''}`}>{value}</span>
}

function SettingsStatusCard({
  tone,
  title,
  status,
  description,
  children,
}: {
  tone: 'ok' | 'warn' | 'error'
  title: string
  status: string
  description: ReactNode
  children?: ReactNode
}) {
  return (
    <div className={`kb-settings-status-card is-${tone}`}>
      <div className="kb-settings-status-main">
        <span className="kb-settings-status-dot" aria-hidden="true" />
        <div className="kb-settings-status-copy">
          <Text className="kb-settings-status-title">{title}</Text>
          <Text className="kb-settings-status-desc">{description}</Text>
        </div>
      </div>
      <div className="kb-settings-status-side">
        <span className={`kb-settings-status-badge is-${tone}`}>{status}</span>
        {children}
      </div>
    </div>
  )
}

function formatCheckedAt(ts: number | undefined) {
  if (!ts) return ''
  try {
    return new Date(ts * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  } catch {
    return ''
  }
}

function isNoCachedUpdate(payload: { status?: string; error?: string } | null | undefined) {
  return Boolean(
    payload
    && String(payload.status || '').trim().toLowerCase() === 'unknown'
    && String(payload.error || '').toLowerCase().includes('cached update'),
  )
}

export function SettingsDrawer({
  open,
  focusTarget = '',
  onClose,
}: {
  open: boolean
  focusTarget?: SettingsFocusTarget | ''
  onClose: () => void
}) {
  const S = useT()
  const s = useSettingsStore()
  const refreshReadinessStore = useSettingsStore((state) => state.refreshReadiness)
  const refreshAppReadinessStore = useSettingsStore((state) => state.refreshAppReadiness)
  const refreshAppUpdateStore = useSettingsStore((state) => state.refreshAppUpdate)
  const textReadiness = s.llmReadiness?.providers.text
  const visionReadiness = s.llmReadiness?.providers.vision
  const updateSectionRef = useRef<HTMLDivElement | null>(null)
  const textCredentialRef = useRef<HTMLElement | null>(null)
  const visionCredentialRef = useRef<HTMLElement | null>(null)
  const [textApiKey, setTextApiKey] = useState('')
  const [textBaseUrl, setTextBaseUrl] = useState('')
  const [textModel, setTextModel] = useState('')
  const [visionApiKey, setVisionApiKey] = useState('')
  const [visionBaseUrl, setVisionBaseUrl] = useState('')
  const [visionModel, setVisionModel] = useState('')
  const [savingConnection, setSavingConnection] = useState(false)
  const [testingTarget, setTestingTarget] = useState<'text' | 'vision' | null>(null)
  const [localTestResults, setLocalTestResults] = useState<Record<'text' | 'vision', LocalTestResult | null>>({ text: null, vision: null })
  const [appReadinessLoading, setAppReadinessLoading] = useState(false)
  const [appReadinessError, setAppReadinessError] = useState('')
  const [appUpdateLoading, setAppUpdateLoading] = useState(false)
  const [appUpdateError, setAppUpdateError] = useState('')
  const [maintenanceStatus, setMaintenanceStatus] = useState<MaintenanceStatus | null>(null)
  const [maintenanceStatusLoading, setMaintenanceStatusLoading] = useState(false)
  const [maintenanceStatusError, setMaintenanceStatusError] = useState('')
  const [autoBackupSaving, setAutoBackupSaving] = useState(false)
  const [restoreReviewAcking, setRestoreReviewAcking] = useState(false)
  const [diagnosticsExporting, setDiagnosticsExporting] = useState(false)
  const [authStatus, setAuthStatus] = useState<AuthStatusPayload | null>(null)
  const [authLocking, setAuthLocking] = useState(false)
  const [managementToken, setManagementToken] = useState('')
  const [managementUnlocking, setManagementUnlocking] = useState(false)
  const [qualityCollectorStatus, setQualityCollectorStatus] = useState<UserIssueRemoteStatus | null>(null)
  const [qualityCollectorLoading, setQualityCollectorLoading] = useState(false)
  const [qualityCollectorError, setQualityCollectorError] = useState('')
  const [qualityCollectorTesting, setQualityCollectorTesting] = useState(false)
  const [qualityCollectorFlushing, setQualityCollectorFlushing] = useState(false)
  const [qualitySharingSaving, setQualitySharingSaving] = useState(false)
  const lastPreferenceErrorAtRef = useRef(0)
  const appReadiness = s.appReadiness
  const appUpdate = s.appUpdate
  const readinessError = s.readinessError
  const updateSettingsPreference = s.update
  const showInternalSettingsTools = internalSettingsToolsVisible()
  const showAuthGateTools = authGateEnabled()

  const savePreference = useCallback((patch: SettingsPatch) => {
    void updateSettingsPreference(patch).catch((err) => {
      const now = Date.now()
      if (now - lastPreferenceErrorAtRef.current < 1500) return
      lastPreferenceErrorAtRef.current = now
      message.error(err instanceof Error ? err.message : S.settings_save_preferences_failed)
    })
  }, [S.settings_save_preferences_failed, updateSettingsPreference])

  const refreshMaintenanceStatus = useCallback(async () => {
    setMaintenanceStatusLoading(true)
    setMaintenanceStatusError('')
    try {
      const payload = await maintenanceApi.status()
      setMaintenanceStatus(payload)
    } catch (err) {
      const detail = err instanceof Error ? err.message : S.settings_maintenance_status_failed
      setMaintenanceStatusError(detail)
    } finally {
      setMaintenanceStatusLoading(false)
    }
  }, [S.settings_maintenance_status_failed])

  const refreshAppReadiness = useCallback(async () => {
    setAppReadinessLoading(true)
    setAppReadinessError('')
    try {
      await refreshAppReadinessStore()
    } catch (err) {
      setAppReadinessError(err instanceof Error ? err.message : S.settings_release_check_failed)
    } finally {
      setAppReadinessLoading(false)
    }
  }, [S.settings_release_check_failed, refreshAppReadinessStore])

  const refreshAppUpdate = useCallback(async (options?: Parameters<typeof refreshAppUpdateStore>[0]) => {
    setAppUpdateLoading(true)
    setAppUpdateError('')
    try {
      await refreshAppUpdateStore(options)
    } catch (err) {
      setAppUpdateError(err instanceof Error ? err.message : S.settings_update_check_failed)
    } finally {
      setAppUpdateLoading(false)
    }
  }, [S.settings_update_check_failed, refreshAppUpdateStore])

  const refreshQualityCollectorStatus = useCallback(async () => {
    setQualityCollectorLoading(true)
    setQualityCollectorError('')
    try {
      const payload = await userIssuesApi.remoteStatus()
      setQualityCollectorStatus(payload)
    } catch (err) {
      setQualityCollectorStatus(null)
      setQualityCollectorError(err instanceof Error ? err.message : S.settings_quality_collector_status_failed.replace('{error}', S.settings_test_unknown_error))
    } finally {
      setQualityCollectorLoading(false)
    }
  }, [S.settings_quality_collector_status_failed, S.settings_test_unknown_error])

  useEffect(() => {
    if (!open) return
    setTextApiKey('')
    setVisionApiKey('')
    setTextBaseUrl(s.textBaseUrl || '')
    setTextModel(s.textModel || s.model || '')
    setVisionBaseUrl(s.visionBaseUrl || '')
    setVisionModel(s.visionModel || '')
    setLocalTestResults({ text: null, vision: null })
  }, [open, s.model, s.textBaseUrl, s.textModel, s.visionBaseUrl, s.visionModel])

  useEffect(() => {
    if (!open) return
    void refreshAppReadiness()
    void refreshReadinessStore().catch(() => {})
    if (showInternalSettingsTools) {
      void refreshMaintenanceStatus()
      void refreshQualityCollectorStatus()
    }
    settingsApi.authStatus().then(setAuthStatus).catch(() => setAuthStatus(null))
  }, [
    open,
    refreshAppReadiness,
    refreshMaintenanceStatus,
    refreshQualityCollectorStatus,
    refreshReadinessStore,
    showInternalSettingsTools,
  ])

  useEffect(() => {
    if (!open) return
    if (!appUpdate) {
      void refreshAppUpdate({ cacheOnly: true })
      return
    }
    if (isNoCachedUpdate(appUpdate)) {
      void refreshAppUpdate({ auto: true })
    }
  }, [appUpdate, open, refreshAppUpdate])

  useEffect(() => {
    if (!open || !focusTarget || typeof window === 'undefined') return
    const timer = window.setTimeout(() => {
      if (focusTarget === 'updates') {
        updateSectionRef.current?.scrollIntoView({ block: 'center', behavior: 'smooth' })
        return
      }
      const card = focusTarget === 'vision' ? visionCredentialRef.current : textCredentialRef.current
      if (!card) return
      card.scrollIntoView({ block: 'center', behavior: 'smooth' })
      const input = card.querySelector<HTMLInputElement>('input[type="password"]')
      input?.focus({ preventScroll: true })
    }, 120)
    return () => window.clearTimeout(timer)
  }, [focusTarget, open])

  const saveConnection = async () => {
    setSavingConnection(true)
    try {
      await settingsApi.update({
        ...(textApiKey.trim() ? { textApiKey: textApiKey.trim() } : {}),
        textBaseUrl: textBaseUrl.trim(),
        textModel: textModel.trim(),
        ...(visionApiKey.trim() ? { visionApiKey: visionApiKey.trim() } : {}),
        visionBaseUrl: visionBaseUrl.trim(),
        visionModel: visionModel.trim(),
      })
      setTextApiKey('')
      setVisionApiKey('')
      setLocalTestResults({ text: null, vision: null })
      await s.load()
      message.success(S.settings_api_settings_saved)
    } catch (err) {
      message.error(err instanceof Error ? err.message : S.settings_save_api_settings_failed)
    } finally {
      setSavingConnection(false)
    }
  }

  const testLlm = async (target: 'text' | 'vision') => {
    setTestingTarget(target)
    try {
      const res = await settingsApi.testLlm(target, target === 'text'
        ? {
            apiKey: textApiKey.trim(),
            baseUrl: textBaseUrl.trim(),
            model: textModel.trim(),
          }
        : {
            apiKey: visionApiKey.trim(),
            baseUrl: visionBaseUrl.trim(),
            model: visionModel.trim(),
          })
      message[res.ok ? 'success' : 'error'](
        res.ok
          ? S.settings_test_ok.replace('{reply}', String(res.reply || S.settings_test_default_reply))
          : S.settings_test_failed.replace('{error}', String(res.error || S.settings_test_unknown_error)),
      )
      setLocalTestResults((current) => ({
        ...current,
        [target]: {
          ok: Boolean(res.ok),
          checked_at: Number(res.checked_at || Date.now() / 1000),
          error_type: String(res.error_type || ''),
          transient: target === 'text'
            ? Boolean(textApiKey.trim() || textBaseUrl.trim() !== s.textBaseUrl || textModel.trim() !== s.textModel)
            : Boolean(visionApiKey.trim() || visionBaseUrl.trim() !== s.visionBaseUrl || visionModel.trim() !== s.visionModel),
        },
      }))
      await Promise.all([
        s.refreshReadiness().catch(() => {}),
        s.refreshAppReadiness().catch(() => {}),
      ])
    } finally {
      setTestingTarget(null)
    }
  }

  const exportDiagnostics = async () => {
    setDiagnosticsExporting(true)
    try {
      await maintenanceApi.downloadDiagnostics()
      message.success(S.settings_maintenance_diag_exported)
    } catch (err) {
      message.error(err instanceof Error ? err.message : S.settings_maintenance_diag_failed)
    } finally {
      setDiagnosticsExporting(false)
    }
  }

  const updateAutoBackup = async (enabled: boolean) => {
    if (maintenanceStatus?.auto_backup?.locked) return
    setAutoBackupSaving(true)
    try {
      await settingsApi.update({ autoBackupEnabled: enabled })
      await s.load()
      await refreshMaintenanceStatus()
      message.success(enabled ? S.settings_auto_backup_enabled : S.settings_auto_backup_disabled)
    } catch (err) {
      message.error(err instanceof Error ? err.message : S.settings_auto_backup_update_failed)
    } finally {
      setAutoBackupSaving(false)
    }
  }

  const updateQualityDataSharing = async (enabled: boolean) => {
    setQualitySharingSaving(true)
    try {
      const result = await updateSettingsPreference({ qualityDataSharingEnabled: enabled })
      await s.load().catch(() => {})
      if (showInternalSettingsTools) await refreshQualityCollectorStatus()
      const cleanup = result?.quality_data_cleanup
      if (!enabled && cleanup && cleanup.ok === false) {
        message.warning(
          S.settings_quality_data_cleanup_failed_msg
            .replace('{error}', cleanup.error || S.settings_test_unknown_error),
        )
      } else {
        message.success(enabled ? S.settings_quality_data_sharing_enabled_msg : S.settings_quality_data_sharing_disabled_msg)
      }
    } catch (err) {
      message.error(err instanceof Error ? err.message : S.settings_quality_data_sharing_update_failed)
    } finally {
      setQualitySharingSaving(false)
    }
  }

  const testQualityCollector = async () => {
    setQualityCollectorTesting(true)
    try {
      const result = await userIssuesApi.testRemote()
      await refreshQualityCollectorStatus()
      if (result.ok) {
        message.success(S.settings_quality_collector_test_ok)
      } else {
        message.error(S.settings_quality_collector_test_failed.replace('{error}', String(result.error || result.status_code || S.settings_test_unknown_error)))
      }
    } catch (err) {
      message.error(err instanceof Error ? err.message : S.settings_quality_collector_test_failed.replace('{error}', S.settings_test_unknown_error))
    } finally {
      setQualityCollectorTesting(false)
    }
  }

  const flushQualityCollectorOutbox = async () => {
    setQualityCollectorFlushing(true)
    try {
      const result = await userIssuesApi.flushOutbox(20)
      await refreshQualityCollectorStatus()
      if (result.enabled === false) {
        message.warning(S.settings_quality_collector_disabled)
      } else if (result.failed > 0) {
        message.warning(S.settings_quality_collector_flush_partial
          .replace('{sent}', String(result.sent || 0))
          .replace('{failed}', String(result.failed || 0)))
      } else {
        message.success(S.settings_quality_collector_flush_ok.replace('{sent}', String(result.sent || 0)))
      }
    } catch (err) {
      message.error(err instanceof Error ? err.message : S.settings_quality_collector_flush_failed)
    } finally {
      setQualityCollectorFlushing(false)
    }
  }

  const lockAccessSession = async () => {
    setAuthLocking(true)
    try {
      await settingsApi.authLogout()
      setAccessToken('')
      const next = await settingsApi.authStatus().catch(() => null)
      setAuthStatus(next)
      message.success(S.settings_auth_locked)
      window.dispatchEvent(new CustomEvent(AUTH_REQUIRED_EVENT))
    } catch (err) {
      message.error(err instanceof Error ? err.message : S.settings_auth_lock_failed)
    } finally {
      setAuthLocking(false)
    }
  }

  const unlockManagementAccess = async () => {
    const token = managementToken.trim()
    if (!token) return
    setManagementUnlocking(true)
    try {
      setAccessToken(token)
      const next = await settingsApi.authLogin(token)
      setAuthStatus(next)
      if (!next.management_authenticated) {
        throw new Error(S.settings_management_access_failed)
      }
      setManagementToken('')
      await Promise.all([
        s.load(),
        refreshAppReadiness(),
        refreshReadinessStore().catch(() => {}),
      ])
      message.success(S.settings_management_access_unlocked)
    } catch (err) {
      setAccessToken('')
      message.error(err instanceof Error ? err.message : S.settings_management_access_failed)
    } finally {
      setManagementUnlocking(false)
    }
  }

  const acknowledgeRestoreReview = async () => {
    setRestoreReviewAcking(true)
    try {
      const result = await maintenanceApi.acknowledgeRestoreReview()
      if (result.ok) {
        message.success(S.settings_restore_review_ack_done)
        await refreshAppReadiness()
      } else {
        message.error(S.settings_restore_review_ack_failed)
      }
    } catch (err) {
      message.error(err instanceof Error ? err.message : S.settings_restore_review_ack_failed)
    } finally {
      setRestoreReviewAcking(false)
    }
  }

  const refreshStatusOverview = async () => {
    await Promise.all([
      refreshAppReadiness(),
      ...(showInternalSettingsTools ? [refreshMaintenanceStatus(), refreshQualityCollectorStatus()] : []),
      refreshReadinessStore().catch(() => {}),
    ])
  }

  const renderProviderState = (target: 'text' | 'vision') => {
    const item = target === 'text' ? textReadiness : visionReadiness
    if (!item) return null
    const local = localTestResults[target]
    const last = local || item.last_test || null
    const effectiveSeverity = local ? (local.ok ? 'ok' : 'error') : item.severity
    const effectiveStatus = local ? (local.ok ? 'ok' : 'failed') : item.status
    const label = effectiveStatus === 'ok'
      ? S.settings_ready_status_ok
      : effectiveStatus === 'failed'
        ? S.settings_ready_status_failed
        : effectiveStatus === 'fallback'
          ? S.settings_ready_status_fallback
          : effectiveStatus === 'missing'
            ? S.settings_ready_status_missing
            : S.settings_ready_status_configured
    const time = formatCheckedAt(last?.checked_at)
    const detail = local
      ? local.ok
        ? (local.transient ? S.settings_last_test_unsaved_ok : S.settings_last_test_ok)
          .replace('{time}', time || S.settings_last_test_unknown_time)
        : (local.transient ? S.settings_last_test_unsaved_failed : S.settings_last_test_failed)
          .replace('{time}', time || S.settings_last_test_unknown_time)
          .replace('{type}', local.error_type || S.settings_test_unknown_error)
      : last
        ? last.ok
          ? S.settings_last_test_ok.replace('{time}', time || S.settings_last_test_unknown_time)
          : S.settings_last_test_failed
            .replace('{time}', time || S.settings_last_test_unknown_time)
            .replace('{type}', last.error_type || S.settings_test_unknown_error)
        : S.settings_last_test_none
    return (
      <div className={`kb-settings-provider-state is-${effectiveSeverity}`}>
        <span className="kb-settings-provider-dot" aria-hidden="true" />
        <span>{label}</span>
        <Text>{detail}</Text>
      </div>
    )
  }

  const renderStatusOverview = () => {
    const llmStatus = s.llmReadiness?.overall?.status || (s.hasTextApiKey ? 'warning' : 'error')
    const apiTone = llmStatus === 'ok' ? 'ok' : llmStatus === 'warning' ? 'warn' : 'error'
    const apiStatus = llmStatus === 'ok'
      ? S.settings_status_ok
      : llmStatus === 'warning'
        ? S.settings_status_review
        : S.settings_status_blocked
    const apiDescription = apiTone === 'ok'
      ? S.settings_api_status_ok_desc
      : apiTone === 'warn'
        ? S.settings_api_status_warn_desc
        : S.settings_api_status_error_desc

    const restoreItem = (appReadiness?.items || []).find((item) => item.key === 'recent_restore')
    const restorePending = Boolean(restoreItem && appReadiness?.restore?.acknowledged !== true)
    const restoreAction = appReadinessActionLabel(S, restoreItem?.action)
    const restoreNoticeAction = showInternalSettingsTools && restorePending && restoreItem?.action === 'restart_and_check' ? (
      <Popconfirm
        title={S.settings_restore_review_ack_confirm}
        okText={S.confirm_ok}
        cancelText={S.confirm_cancel}
        onConfirm={() => { void acknowledgeRestoreReview() }}
      >
        <Button size="small" loading={restoreReviewAcking}>
          {restoreAction || S.settings_release_action_restart_and_check}
        </Button>
      </Popconfirm>
    ) : showInternalSettingsTools && restorePending ? (
      <Button size="small" loading={diagnosticsExporting} onClick={() => { void exportDiagnostics() }}>
        {S.settings_maintenance_diag_export}
      </Button>
    ) : null

    return (
      <div className="kb-settings-status" data-testid="settings-release-readiness">
        <div className="kb-settings-status-head">
          <div className="kb-settings-maintenance-copy">
            <Text className="kb-settings-maintenance-title">{S.settings_status_title}</Text>
            <Text className="kb-settings-maintenance-desc">{S.settings_status_desc}</Text>
          </div>
          <Button
            size="small"
            icon={<ReloadOutlined />}
            loading={appReadinessLoading}
            onClick={() => { void refreshStatusOverview() }}
          >
            {S.settings_release_refresh}
          </Button>
        </div>
        <div className="kb-settings-status-grid">
          <SettingsStatusCard
            tone={apiTone}
            title={S.settings_api_status_title}
            status={apiStatus}
            description={apiDescription}
          />
        </div>
        {showInternalSettingsTools && restorePending ? (
          <Alert
            className="kb-settings-restore-notice"
            type="warning"
            showIcon
            message={S.settings_restore_review_notice_title}
            description={S.settings_restore_review_pending_desc}
            action={restoreNoticeAction}
          />
        ) : null}
        {appReadinessError ? (
          <Alert
            type="warning"
            showIcon
            message={S.settings_status_refresh_failed}
            description={appReadinessError}
          />
        ) : null}
      </div>
    )
  }

  const autoBackupEnabled = Boolean(maintenanceStatus?.auto_backup?.enabled)
  const autoBackupLocked = Boolean(
    maintenanceStatus?.auto_backup?.locked || maintenanceStatus?.data_protection?.can_toggle === false,
  )
  const autoBackupDescription = maintenanceStatusError
    ? S.settings_maintenance_status_failed
    : autoBackupLocked
      ? S.settings_auto_backup_locked_desc
      : S.settings_auto_backup_desc
  const autoBackupDisabled = autoBackupLocked || maintenanceStatusLoading || autoBackupSaving || Boolean(maintenanceStatusError)
  const qualityCollectorPending = Number(qualityCollectorStatus?.outbox?.pending || 0)
  const qualityCollectorHost = String(qualityCollectorStatus?.remote_url_host || S.settings_quality_collector_host_missing)
  const qualityCollectorOff = !s.qualityDataSharingEnabled || qualityCollectorStatus?.quality_data_sharing_enabled === false
  const qualityCollectorLoadFailed = Boolean(qualityCollectorError)
  const qualityCollectorLocalBlocked = Boolean(
    qualityCollectorStatus?.remote_url_is_local
    && !qualityCollectorStatus?.remote_url_local_allowed,
  )
  const qualityCollectorCredentials = Boolean(
    qualityCollectorStatus?.remote_url_configured
    && qualityCollectorStatus.remote_url_has_credentials,
  )
  const qualityCollectorInvalidUrl = Boolean(
    qualityCollectorStatus?.remote_url_configured
    && (
      qualityCollectorStatus.remote_url_has_valid_scheme === false
      || qualityCollectorStatus.remote_url_has_valid_port === false
    ),
  )
  const qualityCollectorInsecure = Boolean(
    qualityCollectorStatus?.remote_url_configured
    && !qualityCollectorLocalBlocked
    && qualityCollectorStatus.remote_url_secure === false,
  )
  const qualityCollectorMissingToken = Boolean(
    qualityCollectorStatus?.remote_url_configured
    && qualityCollectorStatus.remote_token_required !== false
    && !qualityCollectorStatus.remote_token_configured,
  )
  const qualityCollectorSafetyBlocked = Boolean(
    qualityCollectorStatus?.remote_url_configured
    && qualityCollectorStatus.remote_url_allowed === false
    && !qualityCollectorCredentials
    && !qualityCollectorInvalidUrl
    && !qualityCollectorInsecure
    && !qualityCollectorLocalBlocked
    && !qualityCollectorMissingToken,
  )
  const qualityCollectorNeedsSetup = !qualityCollectorOff && (
    !qualityCollectorLoadFailed
    && (!qualityCollectorStatus
    || !qualityCollectorStatus.remote_enabled
    || !qualityCollectorStatus.remote_url_configured
    || qualityCollectorCredentials
    || qualityCollectorLocalBlocked
    || qualityCollectorInvalidUrl
    || qualityCollectorInsecure
    || qualityCollectorMissingToken
    || qualityCollectorSafetyBlocked)
  )
  const qualityCollectorTone = qualityCollectorOff
    ? 'warn'
    : qualityCollectorLoadFailed || qualityCollectorNeedsSetup
      ? 'error'
      : qualityCollectorStatus?.enabled
        ? 'ok'
        : 'warn'
  const qualityCollectorBadge = qualityCollectorLoading
    ? S.settings_status_checking
    : qualityCollectorOff
      ? S.settings_quality_collector_off
      : qualityCollectorLoadFailed
        ? S.settings_update_status_unavailable
      : qualityCollectorNeedsSetup
        ? S.settings_quality_collector_needs_setup
        : qualityCollectorStatus?.enabled
          ? S.settings_quality_collector_ready
          : S.settings_status_review
  const qualityCollectorDescription = qualityCollectorOff
    ? S.settings_quality_collector_off_desc
    : qualityCollectorLoadFailed
      ? S.settings_quality_collector_status_failed.replace('{error}', qualityCollectorError)
    : !qualityCollectorStatus?.remote_enabled
      ? S.settings_quality_collector_env_off_desc
    : !qualityCollectorStatus?.remote_url_configured
      ? S.settings_quality_collector_missing_url_desc
      : qualityCollectorCredentials
        ? S.settings_quality_collector_credentials_desc.replace('{host}', qualityCollectorHost)
      : qualityCollectorInvalidUrl
        ? S.settings_quality_collector_invalid_url_desc.replace('{host}', qualityCollectorHost)
      : qualityCollectorInsecure
        ? S.settings_quality_collector_insecure_url_desc.replace('{host}', qualityCollectorHost)
      : qualityCollectorMissingToken
        ? S.settings_quality_collector_missing_token_desc.replace('{host}', qualityCollectorHost)
      : qualityCollectorLocalBlocked
        ? S.settings_quality_collector_local_url_desc.replace('{host}', qualityCollectorHost)
        : qualityCollectorSafetyBlocked
          ? S.settings_quality_collector_blocked_desc
            .replace('{host}', qualityCollectorHost)
            .replace('{reason}', String(qualityCollectorStatus?.remote_block_reason || S.settings_test_unknown_error))
        : S.settings_quality_collector_ready_desc
          .replace('{host}', qualityCollectorHost)
          .replace('{pending}', String(qualityCollectorPending))

  return (
    <>
    <Drawer
      title={(
        <div className="kb-settings-title">
          <span>{S.settings}</span>
          <Text className="kb-settings-subtitle">{S.settings_subtitle}</Text>
        </div>
      )}
      open={open}
      onClose={onClose}
      size="default"
      rootClassName="kb-settings-drawer-root"
      className="kb-settings-drawer"
    >
      <div className="kb-settings-shell">
        <SettingsSection title={S.settings_section_interface}>
          <SettingsRow title={S.ui_lang} description={S.settings_language_desc}>
            <Select
              className="kb-settings-select"
              value={s.uiLocale}
              onChange={(v) => { savePreference({ uiLocale: v as 'zh' | 'en' }) }}
              options={[
                { label: S.lang_zh, value: 'zh' },
                { label: S.lang_en, value: 'en' },
              ]}
            />
          </SettingsRow>
          <SettingsRow title={S.settings_theme} description={S.settings_theme_desc}>
            <Segmented
              className="kb-settings-segmented"
              value={s.theme}
              onChange={(v) => { savePreference({ theme: v as 'light' | 'dark' }) }}
              options={[
                { label: S.theme_light, value: 'light' },
                { label: S.theme_dark, value: 'dark' },
              ]}
            />
          </SettingsRow>
        </SettingsSection>

        <SettingsSection title={S.settings_section_privacy}>
          {authStatus?.management_required && !authStatus.management_authenticated ? (
            <SettingsRow title={S.settings_management_access_title} description={S.settings_management_access_desc}>
              <Space.Compact>
                <Input.Password
                  value={managementToken}
                  onChange={(event) => setManagementToken(event.target.value)}
                  onPressEnter={() => { void unlockManagementAccess() }}
                  placeholder={S.settings_management_access_placeholder}
                  autoComplete="current-password"
                  data-testid="settings-management-access-token"
                />
                <Button
                  icon={<LockOutlined />}
                  type="primary"
                  loading={managementUnlocking}
                  onClick={() => { void unlockManagementAccess() }}
                  data-testid="settings-management-access-unlock"
                >
                  {S.settings_management_access_unlock}
                </Button>
              </Space.Compact>
            </SettingsRow>
          ) : null}
          {showAuthGateTools && authStatus?.required ? (
            <SettingsRow title={S.settings_auth_lock_title} description={S.settings_auth_lock_desc}>
              <Popconfirm
                title={S.settings_auth_lock_confirm}
                okText={S.confirm_ok}
                cancelText={S.confirm_cancel}
                onConfirm={() => { void lockAccessSession() }}
              >
                <Button
                  data-testid="settings-auth-lock-button"
                  loading={authLocking}
                >
                  {S.settings_auth_lock_action}
                </Button>
              </Popconfirm>
            </SettingsRow>
          ) : null}
          <SettingsRow title={S.settings_quality_data_sharing_title} description={S.settings_quality_data_sharing_desc}>
            <Switch
              data-testid="settings-quality-data-sharing-switch"
              checked={s.qualityDataSharingEnabled}
              loading={qualitySharingSaving}
              onChange={(v) => { void updateQualityDataSharing(v) }}
            />
          </SettingsRow>
          {showInternalSettingsTools ? (
          <div data-testid="settings-quality-collector-status">
            <SettingsStatusCard
              tone={qualityCollectorTone}
              title={S.settings_quality_collector_title}
              status={qualityCollectorBadge}
              description={qualityCollectorDescription}
            >
              <Space size={6} wrap>
                <Button
                  size="small"
                  icon={<ReloadOutlined />}
                  loading={qualityCollectorLoading}
                  onClick={() => { void refreshQualityCollectorStatus() }}
                >
                  {S.settings_release_refresh}
                </Button>
                <Button
                  size="small"
                  loading={qualityCollectorTesting}
                  disabled={!qualityCollectorStatus?.enabled}
                  onClick={() => { void testQualityCollector() }}
                >
                  {S.settings_quality_collector_test}
                </Button>
                <Button
                  size="small"
                  loading={qualityCollectorFlushing}
                  disabled={!qualityCollectorStatus?.enabled || qualityCollectorPending <= 0}
                  onClick={() => { void flushQualityCollectorOutbox() }}
                >
                  {S.settings_quality_collector_flush}
                </Button>
              </Space>
            </SettingsStatusCard>
          </div>
          ) : null}
        </SettingsSection>

        <SettingsUpdatePanel
          appUpdate={appUpdate}
          error={appUpdateError}
          loading={appUpdateLoading}
          sectionRef={updateSectionRef}
          onRefresh={() => { void refreshAppUpdate({ refresh: true }) }}
        />

        <SettingsSection title={S.settings_section_answer}>
          <SettingsRow title={S.answer_mode_hint} description={S.settings_answer_mode_desc}>
            <Select
              className="kb-settings-select"
              value={s.answerModeHint || ''}
              onChange={(v) => { savePreference({ answerModeHint: String(v || '') }) }}
              options={[
                { label: S.mode_auto, value: '' },
                { label: S.mode_reading, value: 'reading' },
                { label: S.mode_compare, value: 'compare' },
                { label: S.mode_idea, value: 'idea' },
                { label: S.mode_experiment, value: 'experiment' },
                { label: S.mode_troubleshoot, value: 'troubleshoot' },
                { label: S.mode_writing, value: 'writing' },
              ]}
            />
          </SettingsRow>
          <SettingsRow title={S.settings_answer_shape} description={S.settings_answer_shape_desc}>
            <Select
              className="kb-settings-select"
              value={s.answerOutputMode || ''}
              onChange={(v) => { savePreference({ answerOutputMode: String(v || '') }) }}
              options={[
                { label: S.settings_shape_auto, value: '' },
                { label: S.settings_shape_reading_guide, value: 'reading_guide' },
                { label: S.settings_shape_fact_answer, value: 'fact_answer' },
                { label: S.settings_shape_critical_review, value: 'critical_review' },
              ]}
            />
          </SettingsRow>
          <SettingsRow title={S.settings_auto_depth} description={S.settings_auto_depth_desc}>
            <Switch
              data-testid="settings-answer-depth-auto-switch"
              checked={s.answerDepthAuto}
              onChange={(v) => { savePreference({ answerDepthAuto: v }) }}
            />
          </SettingsRow>
          <SettingsRow title={S.settings_citation_language} description={S.settings_citation_language_desc}>
            <Segmented
              className="kb-settings-segmented"
              value={s.refsCardLocale}
              onChange={(v) => { savePreference({ refsCardLocale: v as 'auto' | 'zh' | 'en' }) }}
              options={[
                { label: S.settings_citation_auto, value: 'auto' },
                { label: S.settings_citation_zh, value: 'zh' },
                { label: S.settings_citation_en, value: 'en' },
              ]}
            />
          </SettingsRow>
        </SettingsSection>

        <SettingsSection title={S.settings_section_connection}>
          {renderStatusOverview()}
          <div className="kb-settings-connection-alerts">
            {readinessError ? (
              <Alert
                type="warning"
                showIcon
                message={S.settings_status_refresh_failed}
                description={readinessError}
              />
            ) : null}
            {!s.hasTextApiKey ? (
              <Alert
                type="warning"
                showIcon
                message={S.settings_missing_text_api_title}
                description={S.settings_missing_text_api_desc}
              />
            ) : null}
            {s.hasTextApiKey && s.visionUsesTextFallback ? (
              <Alert
                type="info"
                showIcon
                message={S.settings_missing_vision_api_title}
                description={S.settings_missing_vision_api_desc}
              />
            ) : null}
          </div>
          <div className="kb-settings-credential-grid">
            <section
              ref={textCredentialRef}
              className={`kb-settings-credential ${focusTarget === 'text' ? 'is-targeted' : ''}`}
              data-api-target="text"
            >
              <div className="kb-settings-credential-head">
                <div>
                  <Text className="kb-settings-credential-title">{S.settings_text_api_title}</Text>
                  <Text className="kb-settings-credential-desc">{S.settings_text_api_desc}</Text>
                </div>
                <SettingsValue value={s.hasTextApiKey ? S.settings_api_key_ready : S.settings_api_key_missing} muted={!s.hasTextApiKey} />
              </div>
              {renderProviderState('text')}
              <div className="kb-settings-credential-fields">
                <Input.Password
                  value={textApiKey}
                  onChange={(event) => setTextApiKey(event.target.value)}
                  placeholder={S.settings_api_key_placeholder}
                  autoComplete="off"
                  data-testid="settings-text-api-key"
                />
                <Input
                  value={textBaseUrl}
                  onChange={(event) => setTextBaseUrl(event.target.value)}
                  placeholder={S.settings_base_url}
                />
                <Input
                  value={textModel}
                  onChange={(event) => setTextModel(event.target.value)}
                  placeholder={S.settings_model_id}
                />
              </div>
              <div className="kb-settings-credential-actions">
                <Button
                  icon={<ApiOutlined />}
                  loading={testingTarget === 'text'}
                  onClick={() => { void testLlm('text') }}
                >
                  {S.settings_test_text_connection}
                </Button>
              </div>
            </section>

            <section
              ref={visionCredentialRef}
              className={`kb-settings-credential ${focusTarget === 'vision' ? 'is-targeted' : ''}`}
              data-api-target="vision"
            >
              <div className="kb-settings-credential-head">
                <div>
                  <Text className="kb-settings-credential-title">{S.settings_vision_api_title}</Text>
                  <Text className="kb-settings-credential-desc">{S.settings_vision_api_desc}</Text>
                </div>
                <SettingsValue
                  value={s.visionUsesTextFallback ? S.settings_api_key_fallback : (s.hasVisionApiKey ? S.settings_api_key_ready : S.settings_api_key_missing)}
                  muted={!s.hasVisionApiKey || s.visionUsesTextFallback}
                />
              </div>
              {renderProviderState('vision')}
              <div className="kb-settings-credential-fields">
                <Input.Password
                  value={visionApiKey}
                  onChange={(event) => setVisionApiKey(event.target.value)}
                  placeholder={S.settings_api_key_placeholder}
                  autoComplete="off"
                  data-testid="settings-vision-api-key"
                />
                <Input
                  value={visionBaseUrl}
                  onChange={(event) => setVisionBaseUrl(event.target.value)}
                  placeholder={S.settings_base_url}
                />
                <Input
                  value={visionModel}
                  onChange={(event) => setVisionModel(event.target.value)}
                  placeholder={S.settings_model_id}
                />
              </div>
              <div className="kb-settings-credential-actions">
                <Button
                  icon={<ApiOutlined />}
                  loading={testingTarget === 'vision'}
                  onClick={() => { void testLlm('vision') }}
                >
                  {S.settings_test_vision_connection}
                </Button>
              </div>
            </section>
          </div>
          <div className="kb-settings-action-row">
            <Button type="primary" loading={savingConnection} onClick={() => { void saveConnection() }}>
              {S.settings_save_api_settings}
            </Button>
          </div>
        </SettingsSection>

        <details className="kb-settings-advanced">
          <summary>
            <span>{S.settings_section_advanced}</span>
            <Text>{S.settings_advanced_desc}</Text>
          </summary>
          <div className="kb-settings-advanced-body">
            {showInternalSettingsTools ? (
            <SettingsRow title={S.settings_auto_backup_title} description={autoBackupDescription}>
              <Switch
                data-testid="settings-auto-backup-switch"
                checked={autoBackupEnabled}
                disabled={autoBackupDisabled}
                loading={maintenanceStatusLoading || autoBackupSaving}
                onChange={(v) => { void updateAutoBackup(v) }}
              />
            </SettingsRow>
            ) : null}
            <SettingsRow title={`${S.top_k}: ${s.topK}`} description={S.settings_top_k_desc}>
              <Slider min={2} max={20} value={s.topK} onChange={(v) => { savePreference({ topK: v }) }} />
            </SettingsRow>
            <SettingsRow title={`${S.temp}: ${s.temperature}`} description={S.settings_temperature_desc}>
              <Slider min={0} max={1} step={0.05} value={s.temperature} onChange={(v) => { savePreference({ temperature: v }) }} />
            </SettingsRow>
            <SettingsRow title={`${S.max_tokens}: ${s.maxTokens}`} description={S.settings_max_tokens_desc}>
              <Slider min={512} max={3072} step={128} value={s.maxTokens} onChange={(v) => { savePreference({ maxTokens: v }) }} />
            </SettingsRow>
            <SettingsRow title={S.answer_contract} description={S.settings_structured_answer_desc}>
              <Switch checked={s.answerContractV1} onChange={(v) => { savePreference({ answerContractV1: v }) }} />
            </SettingsRow>
          </div>
        </details>
      </div>
    </Drawer>
    </>
  )
}
