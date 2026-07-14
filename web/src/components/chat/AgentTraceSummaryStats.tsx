import { AgentTraceSummaryChip } from './AgentTraceSummaryChip'
import type { AgentTraceLabels } from './agentTraceTypes'
import { buildAgentTraceSummaryChips } from './agentTraceSummaryChips'
import type { AgentSourceSummaryViewModel } from './agentTraceViewModel'

export function AgentTraceSummaryStats({
  labels,
  viewModel,
  showInternalDetails,
}: AgentTraceLabels & {
  viewModel: AgentSourceSummaryViewModel
  showInternalDetails: boolean
}) {
  const summaryChips = buildAgentTraceSummaryChips(labels, viewModel, { includeInternal: showInternalDetails })

  return (
    <div className="kb-agent-trace-summary">
      {summaryChips
        .filter((chip) => chip.visible !== false)
        .map((chip) => (
          <AgentTraceSummaryChip
            key={chip.id}
            className={chip.className}
            label={chip.label}
            value={chip.value}
            title={chip.title}
            testId={chip.testId}
          />
        ))}
    </div>
  )
}
