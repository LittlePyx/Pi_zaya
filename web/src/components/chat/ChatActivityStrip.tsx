import { BugOutlined, LoadingOutlined } from '@ant-design/icons'
import type { ChatPerfSnapshot } from './useChatPerfSnapshot'

export interface ChatActivityItem {
  key: string
  label: string
  tone: 'active' | 'ready' | 'warning'
}

export function ChatActivityStrip({
  items,
  debugEnabled,
  debugSnapshot,
  labels,
}: {
  items: ChatActivityItem[]
  debugEnabled: boolean
  debugSnapshot: ChatPerfSnapshot
  labels: Record<string, string>
}) {
  if (!debugEnabled && items.length <= 0) return null
  return (
    <div className="kb-chat-activity-shell">
      <div className="kb-chat-activity-strip" data-testid="chat-activity-strip" aria-live="polite">
        {items.map((item) => (
          <span
            key={item.key}
            className={`kb-chat-activity-pill is-${item.tone}`}
            data-testid={`chat-activity-${item.key}`}
          >
            {item.tone === 'active' ? <LoadingOutlined spin /> : <span className="kb-chat-activity-dot" aria-hidden="true" />}
            <span>{item.label}</span>
          </span>
        ))}
        {debugEnabled ? (
          <span className="kb-chat-debug-strip" data-testid="chat-perf-panel">
            <BugOutlined />
            <span>{labels.chat_debug_switch.replace('{n}', String(debugSnapshot.switchTotal)).replace('{ms}', String(debugSnapshot.switchAvgMs))}</span>
            <span>{labels.chat_debug_refs.replace('{n}', String(debugSnapshot.refsTotal)).replace('{ms}', String(debugSnapshot.refsAvgMs))}</span>
            <span>{labels.chat_debug_open.replace('{n}', String(debugSnapshot.openPhases))}</span>
            <span>{labels.chat_debug_prep.replace('{n}', String(debugSnapshot.messagePrep))}</span>
          </span>
        ) : null}
      </div>
    </div>
  )
}
