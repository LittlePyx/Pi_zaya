import type { StringMap } from '../../i18n'
import { AgentTracePlanRow } from './AgentTracePlanRow'
import { tx } from './agentTracePanelUtils'

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
        <AgentTracePlanRow key={`${String(step.tool || 'plan')}-${idx}`} step={step} />
      ))}
    </div>
  )
}
