import { Alert, Button } from 'antd'
import type { ApiSettingsTarget } from '../layout/settingsEvents'
import type { ResearchRuntimeContext } from './researchContext'

export function ChatConnectionAlert({
  labels,
  researchContext,
  onOpenSettings,
}: {
  labels: Record<string, string>
  researchContext: ResearchRuntimeContext
  onOpenSettings: (target: ApiSettingsTarget) => void
}) {
  const alertTarget = researchContext.api.connectionAlertTarget
  if (!alertTarget) return null

  const provider = alertTarget === 'vision'
    ? researchContext.api.vision
    : researchContext.api.text
  const description = alertTarget === 'vision'
    ? labels.settings_missing_vision_api_desc
    : provider.status === 'failed' && (provider.lastError || provider.reason)
      ? labels.chat_api_failed_desc.replace('{error}', provider.lastError || provider.reason)
      : labels.chat_api_missing_desc

  return (
    <div className="kb-chat-connection-alert">
      <Alert
        type="warning"
        showIcon
        message={alertTarget === 'vision' ? labels.settings_missing_vision_api_title : labels.chat_api_missing_title}
        description={description}
        action={(
          <Button size="small" onClick={() => onOpenSettings(alertTarget)}>
            {labels.chat_open_api_settings}
          </Button>
        )}
      />
    </div>
  )
}
