import { AgentSourceSummaryPanel } from './AgentSourceSummaryPanel'
import { AgentTraceDiagnosticsPanel } from './AgentTraceDiagnosticsPanel'
import { AgentTraceFrame } from './AgentTraceFrame'
import type { AgentTraceReferenceHandlers } from './agentTraceReferenceTypes'
import type { AgentTraceLabels } from './agentTraceTypes'
import type { AgentTraceViewModel } from './agentTraceViewModel'
import type { ArchivedAgentTraceLoadStatus } from './useArchivedAgentTrace'

export function AgentTraceResolvedPanel({
  labels,
  viewModel,
  loadStatus,
  showDiagnostics,
  onOpen,
  onOpenReference,
  onAddReferenceToShelf,
}: AgentTraceReferenceHandlers & AgentTraceLabels & {
  viewModel: AgentTraceViewModel
  loadStatus: ArchivedAgentTraceLoadStatus
  showDiagnostics: boolean
  onOpen: () => void | Promise<void>
}) {
  return (
    <AgentTraceFrame
      labels={labels}
      summaryStatus={viewModel.headerEvidence}
      summaryContext={viewModel.headerContext}
      open={loadStatus === 'loaded' ? true : undefined}
      onOpen={onOpen}
    >
      <AgentSourceSummaryPanel
        labels={labels}
        viewModel={viewModel.sourceSummary}
        onOpenReference={onOpenReference}
        onAddReferenceToShelf={onAddReferenceToShelf}
      />
      {showDiagnostics ? (
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
