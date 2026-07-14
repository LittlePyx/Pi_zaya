import { AgentEvidenceMatrix } from './AgentEvidenceMatrix'
import { AgentTraceReferenceSection } from './AgentTraceReferenceSection'
import { AgentTraceSummaryStats } from './AgentTraceSummaryStats'
import { AgentUnsupportedClaims } from './AgentUnsupportedClaims'
import type { AgentTraceReferenceHandlers } from './agentTraceReferenceTypes'
import type { AgentTraceLabels } from './agentTraceTypes'
import type { AgentSourceSummaryViewModel } from './agentTraceViewModel'

export function AgentSourceSummaryPanel({
  labels,
  viewModel,
  showInternalDetails,
  onOpenReference,
  onAddReferenceToShelf,
}: AgentTraceReferenceHandlers & AgentTraceLabels & {
  viewModel: AgentSourceSummaryViewModel
  showInternalDetails: boolean
}) {
  const {
    evidenceMatrix,
    subtaskCount,
    unsupportedClaimRows,
    references,
  } = viewModel

  return (
    <>
      <AgentTraceSummaryStats labels={labels} viewModel={viewModel} showInternalDetails={showInternalDetails} />
      <AgentEvidenceMatrix
        labels={labels}
        rows={evidenceMatrix}
        subtaskCount={showInternalDetails ? subtaskCount : 0}
      />
      <AgentUnsupportedClaims labels={labels} claims={unsupportedClaimRows} />
      <AgentTraceReferenceSection
        labels={labels}
        references={references}
        onOpenReference={onOpenReference}
        onAddReferenceToShelf={onAddReferenceToShelf}
      />
    </>
  )
}
