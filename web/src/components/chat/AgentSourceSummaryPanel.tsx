import type { StringMap } from '../../i18n'
import { AgentEvidenceMatrix } from './AgentEvidenceMatrix'
import { AgentTraceReferenceSection } from './AgentTraceReferenceSection'
import { AgentTraceSummaryStats } from './AgentTraceSummaryStats'
import { AgentUnsupportedClaims } from './AgentUnsupportedClaims'
import type { AgentTraceReferenceHandlers } from './agentTraceReferenceTypes'
import type { AgentSourceSummaryViewModel } from './useAgentTraceViewModel'

export function AgentSourceSummaryPanel({
  labels,
  viewModel,
  onOpenReference,
  onAddReferenceToShelf,
}: AgentTraceReferenceHandlers & {
  labels: Partial<StringMap>
  viewModel: AgentSourceSummaryViewModel
}) {
  const {
    evidenceMatrix,
    subtaskCount,
    unsupportedClaimRows,
    references,
  } = viewModel

  return (
    <>
      <AgentTraceSummaryStats labels={labels} viewModel={viewModel} />
      <AgentEvidenceMatrix labels={labels} rows={evidenceMatrix} subtaskCount={subtaskCount} />
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
