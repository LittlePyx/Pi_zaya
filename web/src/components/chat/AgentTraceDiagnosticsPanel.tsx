import { AgentTraceCheckActivity } from './AgentTraceCheckActivity'
import { AgentTracePlanSection } from './AgentTracePlanSection'
import type { AgentTraceReferenceHandlers } from './agentTraceReferenceTypes'
import type { AgentTraceLabels } from './agentTraceTypes'
import type { AgentTraceDiagnosticsViewModel } from './useAgentTraceViewModel'
import {
  tx,
  txFmt,
} from './agentTracePanelUtils'

export function AgentTraceDiagnosticsPanel({
  labels,
  viewModel,
  onOpenReference,
  onAddReferenceToShelf,
}: AgentTraceReferenceHandlers & AgentTraceLabels & {
  viewModel: AgentTraceDiagnosticsViewModel
}) {
  const { plan, steps, planStepCount, toolCallCount } = viewModel

  if (plan.length <= 0 && steps.length <= 0) return null
  return (
    <details className="kb-agent-trace-details" data-testid="agent-trace-execution-details">
      <summary>
        <span>{tx(labels, 'agent_trace_diagnostics', 'Diagnostics')}</span>
        <span>{txFmt(labels, 'agent_trace_plan_count', '{n} plan', { n: planStepCount })}</span>
        <span>{txFmt(labels, 'agent_trace_check_count', '{n} checks', { n: toolCallCount })}</span>
      </summary>
      <AgentTracePlanSection labels={labels} plan={plan} />
      <AgentTraceCheckActivity
        labels={labels}
        steps={steps}
        onOpenReference={onOpenReference}
        onAddReferenceToShelf={onAddReferenceToShelf}
      />
    </details>
  )
}
