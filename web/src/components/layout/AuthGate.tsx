import { LockOutlined } from '@ant-design/icons'
import { Alert, Button, Input, Typography } from 'antd'
import { useCallback, useEffect, useMemo, useState } from 'react'
import { AUTH_REQUIRED_EVENT, setAccessToken } from '../../api/client'
import { settingsApi, type AuthStatusPayload } from '../../api/settings'
import { useSettingsStore } from '../../stores/settingsStore'

const { Text } = Typography

const COPY = {
  zh: {
    title: '需要访问令牌',
    subtitle: '这个 Pi-zaya 实例已启用 API 访问保护。',
    placeholder: '输入访问令牌',
    submit: '进入',
    checking: '检查中',
    invalid: '访问令牌不正确，请重试。',
    missing: '服务端要求访问令牌，但还没有配置 KB_ACCESS_TOKEN。',
    offline: '无法连接后端，请确认服务已启动。',
  },
  en: {
    title: 'Access Token Required',
    subtitle: 'API access protection is enabled for this Pi-zaya instance.',
    placeholder: 'Enter access token',
    submit: 'Unlock',
    checking: 'Checking',
    invalid: 'The access token is invalid. Please try again.',
    missing: 'The server requires an access token, but KB_ACCESS_TOKEN is not configured.',
    offline: 'Cannot connect to the backend. Please confirm the service is running.',
  },
} as const

function isRegressionRoute() {
  return typeof window !== 'undefined' && window.location.pathname.startsWith('/__')
}

export function AuthGate() {
  const locale = useSettingsStore(s => s.uiLocale)
  const loadSettings = useSettingsStore(s => s.load)
  const T = COPY[locale === 'en' ? 'en' : 'zh']
  const [status, setStatus] = useState<AuthStatusPayload | null>(null)
  const [checked, setChecked] = useState(false)
  const [token, setToken] = useState('')
  const [error, setError] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const disabled = useMemo(() => isRegressionRoute(), [])

  const refreshStatus = useCallback(async () => {
    if (disabled) return
    try {
      const next = await settingsApi.authStatus()
      setStatus(next)
      setError(next.required && !next.configured ? T.missing : '')
    } catch (err) {
      if (err instanceof Error && /^404\b/.test(err.message.trim())) {
        setStatus({
          required: false,
          configured: false,
          authenticated: false,
          env: 'development',
          production: false,
        })
        setError('')
        return
      }
      setError(T.offline)
    } finally {
      setChecked(true)
    }
  }, [disabled, T.missing, T.offline])

  useEffect(() => {
    void refreshStatus()
    const onAuthRequired = () => {
      setStatus(current => current ? { ...current, required: true, authenticated: false } : current)
      setError('')
      void refreshStatus()
    }
    window.addEventListener(AUTH_REQUIRED_EVENT, onAuthRequired)
    return () => window.removeEventListener(AUTH_REQUIRED_EVENT, onAuthRequired)
  }, [refreshStatus])

  const configured = Boolean(status?.configured)
  const showGate = checked && Boolean(status?.required) && !status?.authenticated
  if (disabled || !showGate) return null

  const submit = async () => {
    const clean = token.trim()
    if (!clean) return
    setSubmitting(true)
    setError('')
    try {
      setAccessToken(clean)
      const next = await settingsApi.authLogin(clean)
      setStatus(next)
      if (!next.authenticated) {
        setError(T.invalid)
        setAccessToken('')
        return
      }
      await loadSettings()
    } catch {
      setAccessToken('')
      setError(T.invalid)
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="kb-auth-gate" role="dialog" aria-modal="true">
      <div className="kb-auth-card">
        <div className="kb-auth-icon" aria-hidden="true"><LockOutlined /></div>
        <div className="kb-auth-copy">
          <div className="kb-auth-title">{T.title}</div>
          <Text className="kb-auth-subtitle">{T.subtitle}</Text>
        </div>
        {error ? <Alert type={!status || configured ? 'error' : 'warning'} showIcon message={error} /> : null}
        <form
          className="kb-auth-form"
          onSubmit={(event) => {
            event.preventDefault()
            void submit()
          }}
        >
          <Input.Password
            value={token}
            onChange={(event) => setToken(event.target.value)}
            placeholder={configured ? T.placeholder : T.checking}
            disabled={!configured || submitting}
            autoFocus
          />
          <Button
            type="primary"
            htmlType="submit"
            loading={submitting}
            disabled={!configured || !token.trim()}
          >
            {T.submit}
          </Button>
        </form>
      </div>
    </div>
  )
}
