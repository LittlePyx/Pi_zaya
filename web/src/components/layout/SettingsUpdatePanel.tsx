import { Button, Typography } from 'antd'
import { type RefObject } from 'react'
import type { AppUpdateCheckPayload } from '../../api/app'
import { useT } from '../../i18n'

const { Text } = Typography

function formatDateTime(ts: number | undefined) {
  if (!ts) return ''
  try {
    return new Date(ts * 1000).toLocaleString([], {
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
    })
  } catch {
    return ''
  }
}

function retryAfterFromError(error: string): number | undefined {
  const match = String(error || '').match(/Try again after\s+(\d{4})-(\d{2})-(\d{2})\s+(\d{2}):(\d{2}):(\d{2})/i)
  if (!match) return undefined
  const [, year, month, day, hour, minute, second] = match
  const ms = new Date(
    Number(year),
    Number(month) - 1,
    Number(day),
    Number(hour),
    Number(minute),
    Number(second),
  ).getTime()
  return Number.isFinite(ms) && ms > 0 ? ms / 1000 : undefined
}

function UpdateStatusCard({
  tone,
  title,
  status,
  description,
  refreshLabel,
  loading,
  onRefresh,
}: {
  tone: 'ok' | 'warn' | 'error'
  title: string
  status: string
  description: string
  refreshLabel: string
  loading: boolean
  onRefresh: () => void
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
        <Button size="small" loading={loading} onClick={onRefresh}>
          {refreshLabel}
        </Button>
      </div>
    </div>
  )
}

export function SettingsUpdatePanel({
  appUpdate,
  error,
  loading,
  sectionRef,
  onRefresh,
}: {
  appUpdate: AppUpdateCheckPayload | null
  error: string
  loading: boolean
  sectionRef: RefObject<HTMLDivElement | null>
  onRefresh: () => void
}) {
  const S = useT()
  const latestTag = appUpdate?.latest?.tag_name || ''
  const currentVersion = appUpdate?.current?.version || S.settings_update_unknown_version
  const latestVersion = latestTag || S.settings_update_latest_unknown
  const checkedAt = formatDateTime(appUpdate?.checked_at)
  const rawError = String(error || appUpdate?.error || '')
  const isRateLimited = rawError.toLowerCase().includes('rate limit')
  const noRelease = rawError.toLowerCase().includes('release')
  const retryAfter = formatDateTime(appUpdate?.retry_after || retryAfterFromError(rawError))
  const retryAfterText = retryAfter
    ? S.settings_update_retry_after.replace('{time}', retryAfter)
    : S.settings_update_later
  const noCachedCheck = rawError.toLowerCase().includes('cached update')
  const notChecked = !appUpdate || Boolean(appUpdate && appUpdate.status === 'unknown' && noCachedCheck)
  const unavailable = Boolean(appUpdate && !notChecked && (appUpdate.status === 'unavailable' || appUpdate.status === 'unknown'))
  const versionUnknown = Boolean(appUpdate && !notChecked && appUpdate.status === 'unknown')
  const disabled = Boolean(appUpdate && !appUpdate.enabled)
  const hasUpdate = Boolean(appUpdate?.update_available)
  const tone = hasUpdate || unavailable ? 'warn' : 'ok'
  const status = loading
    ? S.settings_update_status_checking
    : notChecked
      ? S.settings_update_status_idle
      : disabled
      ? S.settings_update_status_disabled
      : hasUpdate
        ? S.settings_update_status_available
        : noRelease
          ? S.settings_update_status_no_release
        : versionUnknown
          ? S.settings_update_status_unknown
          : unavailable
            ? S.settings_update_status_unavailable
            : S.settings_update_status_current
  const description = disabled
    ? S.settings_update_disabled_desc
    : notChecked
      ? S.settings_update_idle_desc
    : hasUpdate
      ? S.settings_update_available_desc
        .replace('{current}', currentVersion)
        .replace('{latest}', latestVersion)
      : versionUnknown
        ? S.settings_update_unknown_desc
        : unavailable
          ? noRelease
            ? S.settings_update_no_release_hint
            : isRateLimited
            ? S.settings_update_rate_limit_desc.replace('{time}', retryAfterText)
            : rawError
            ? S.settings_update_unavailable_with_error_desc.replace('{error}', rawError)
            : S.settings_update_unavailable_desc
          : S.settings_update_current_desc.replace('{current}', currentVersion)
  const errorText = rawError && !noCachedCheck && !isRateLimited && !noRelease
    ? rawError.toLowerCase().includes('comparable')
        ? S.settings_update_version_hint
        : S.settings_update_error_hint
    : ''

  return (
    <div ref={sectionRef}>
      <section className="kb-settings-section">
        <div className="kb-settings-section-title">{S.settings_section_updates}</div>
        <div className="kb-settings-section-body">
          <div className="kb-settings-update-panel" data-testid="settings-update-panel">
            <UpdateStatusCard
              tone={tone}
              title={S.settings_update_title}
              status={status}
              description={description}
              refreshLabel={S.settings_update_refresh}
              loading={loading}
              onRefresh={onRefresh}
            />
            <div className="kb-settings-update-meta">
              <span>{S.settings_update_current_version}: <strong>{currentVersion}</strong></span>
              <span>{S.settings_update_latest_version}: <strong>{latestVersion}</strong></span>
              {checkedAt ? <span>{S.settings_update_last_checked}: {checkedAt}</span> : null}
            </div>
            {errorText ? (
              <Text className="kb-settings-update-note">
                {errorText}
              </Text>
            ) : null}
            {hasUpdate ? (
              <details
                className="kb-settings-update-details"
                key={`${currentVersion}:${latestVersion}`}
                open
              >
                <summary>{S.settings_update_steps_title}</summary>
                <div className="kb-settings-update-details-body">
                  <Text className="kb-settings-update-note">{S.settings_update_steps_desc}</Text>
                  {appUpdate?.latest?.body ? (
                    <pre className="kb-settings-update-notes">{appUpdate.latest.body}</pre>
                  ) : null}
                  {appUpdate?.instructions?.length ? (
                    <pre className="kb-settings-update-commands">{appUpdate.instructions.join('\n')}</pre>
                  ) : null}
                  {appUpdate?.latest?.html_url ? (
                    <Button size="small" href={appUpdate.latest.html_url} target="_blank">
                      {S.settings_update_view_release}
                    </Button>
                  ) : null}
                </div>
              </details>
            ) : null}
          </div>
        </div>
      </section>
    </div>
  )
}
