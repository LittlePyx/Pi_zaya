import type { ReactNode } from 'react'
import type { StringMap } from '../../i18n'
import { tx } from './agentTracePanelUtils'

export function AgentTraceFrame({
  labels,
  summaryStatus,
  summaryContext,
  open,
  onOpen,
  children,
}: {
  labels: Partial<StringMap>
  summaryStatus: ReactNode
  summaryContext: ReactNode
  open?: boolean
  onOpen?: () => void | Promise<void>
  children: ReactNode
}) {
  return (
    <details className="kb-agent-trace" open={open} onToggle={(event) => {
      if ((event.currentTarget as HTMLDetailsElement).open) void onOpen?.()
    }}>
      <summary>
        <span>{tx(labels, 'agent_trace_title', 'Sources & evidence')}</span>
        <span>{summaryStatus}</span>
        <span>{summaryContext}</span>
      </summary>
      {children}
    </details>
  )
}
