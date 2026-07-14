import { AgentSourceSummaryPanel } from './AgentSourceSummaryPanel'
import { AgentTraceDiagnosticsPanel } from './AgentTraceDiagnosticsPanel'
import { AgentTraceFrame } from './AgentTraceFrame'
import type { AgentTraceReferenceHandlers } from './agentTraceReferenceTypes'
import type { AgentTraceLabels } from './agentTraceTypes'
import type { AgentTraceViewModel } from './agentTraceViewModel'
import type { ArchivedAgentTraceLoadStatus } from './agentTraceArchiveState'

export function AgentTraceResolvedPanel({
  labels,
  viewModel,
  loadStatus,
  showInternalDetails,
  onOpen,
  onOpenReference,
  onAddReferenceToShelf,
}: AgentTraceReferenceHandlers & AgentTraceLabels & {
  viewModel: AgentTraceViewModel
  loadStatus: ArchivedAgentTraceLoadStatus
  showInternalDetails: boolean
  onOpen: () => void | Promise<void>
}) {
  return (
    <AgentTraceFrame
      labels={labels}
      summaryStatus={showInternalDetails
        ? viewModel.headerEvidence
        : (viewModel.sourceSummary.evidenceLabel || labels.agent_trace_available || 'Sources available')}
      summaryContext={showInternalDetails
        ? viewModel.headerContext
        : ''}
      open={loadStatus === 'loaded' ? true : undefined}
      onOpen={onOpen}
    >
      <AgentSourceSummaryPanel
        labels={labels}
        viewModel={viewModel.sourceSummary}
        showInternalDetails={showInternalDetails}
        onOpenReference={onOpenReference}
        onAddReferenceToShelf={onAddReferenceToShelf}
      />
      {showInternalDetails ? (
        <AgentTraceDiagnosticsPanel
          labels={labels}
          viewModel={viewModel.diagnostics}
          onOpenReference={onOpenReference}
          onAddReferenceToShelf={onAddReferenceToShelf}
        />
      ) : null}
    </AgentTraceFrame>
  )
}
