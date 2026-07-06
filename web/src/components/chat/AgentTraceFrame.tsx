import type { ReactNode } from 'react'
import type { AgentTraceLabels } from './agentTraceTypes'
import { tx } from './agentTracePanelUtils'

export function AgentTraceFrame({
  labels,
  summaryStatus,
  summaryContext,
  open,
  onOpen,
  children,
}: AgentTraceLabels & {
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
