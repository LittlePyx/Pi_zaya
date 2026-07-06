import type { StringMap } from '../../i18n'
import {
  shortText,
  statusClass,
  tx,
} from './agentTracePanelUtils'

export function AgentTracePlanSection({
  labels,
  plan,
}: {
  labels: Partial<StringMap>
  plan: Record<string, unknown>[]
}) {
  if (plan.length <= 0) return null

  return (
    <div className="kb-agent-trace-section">
      <div className="kb-agent-trace-heading">{tx(labels, 'agent_trace_plan', 'Plan')}</div>
      {plan.map((step, idx) => (
        <div className="kb-agent-trace-row" key={`${String(step.tool || 'plan')}-${idx}`}>
          <span className={`kb-agent-trace-status ${statusClass(step.status)}`}>{String(step.status || 'pending')}</span>
          <span className="kb-agent-trace-tool">{String(step.tool || '')}</span>
          <span className="kb-agent-trace-text">{shortText(step.goal)}</span>
        </div>
      ))}
    </div>
  )
}
