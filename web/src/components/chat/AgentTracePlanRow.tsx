import {
  shortText,
  statusClass,
} from './agentTracePanelUtils'

export function AgentTracePlanRow({
  step,
}: {
  step: Record<string, unknown>
}) {
  return (
    <div className="kb-agent-trace-row">
      <span className={`kb-agent-trace-status ${statusClass(step.status)}`}>{String(step.status || 'pending')}</span>
      <span className="kb-agent-trace-tool">{String(step.tool || '')}</span>
      <span className="kb-agent-trace-text">{shortText(step.goal)}</span>
    </div>
  )
}
