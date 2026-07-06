import type { StringMap } from '../../i18n'
import { AgentTraceToolCall } from './AgentTraceToolCall'
import type { AgentTraceReferenceHandlers } from './agentTraceReferenceTypes'
import { tx } from './agentTracePanelUtils'

export function AgentTraceCheckActivity({
  labels,
  steps,
  onOpenReference,
  onAddReferenceToShelf,
}: AgentTraceReferenceHandlers & {
  labels: Partial<StringMap>
  steps: Record<string, unknown>[]
}) {
  if (steps.length <= 0) return null

  return (
    <div className="kb-agent-trace-section">
      <div className="kb-agent-trace-heading">{tx(labels, 'agent_trace_check_activity', 'Check activity')}</div>
      {steps.map((step, idx) => (
        <AgentTraceToolCall
          key={`${String(step.tool || 'tool')}-${idx}`}
          labels={labels}
          step={step}
          onOpenReference={onOpenReference}
          onAddReferenceToShelf={onAddReferenceToShelf}
        />
      ))}
    </div>
  )
}
