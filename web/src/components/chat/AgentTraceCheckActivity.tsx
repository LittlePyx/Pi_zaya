import { AgentTraceToolCall } from './AgentTraceToolCall'
import type { AgentTraceReferenceHandlers } from './agentTraceReferenceTypes'
import type { AgentTraceLabels, AgentTraceRecord } from './agentTraceTypes'
import { tx } from './agentTracePanelUtils'

export function AgentTraceCheckActivity({
  labels,
  steps,
  onOpenReference,
  onAddReferenceToShelf,
}: AgentTraceReferenceHandlers & AgentTraceLabels & {
  steps: AgentTraceRecord[]
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
