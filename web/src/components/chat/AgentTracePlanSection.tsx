import { AgentTracePlanRow } from './AgentTracePlanRow'
import type { AgentTraceLabels, AgentTraceRecord } from './agentTraceTypes'
import { tx } from './agentTracePanelUtils'

export function AgentTracePlanSection({
  labels,
  plan,
}: AgentTraceLabels & {
  plan: AgentTraceRecord[]
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
