import type { StringMap } from '../../i18n'
import type { CiteDetail } from './citationState'
import { AgentEvidenceMatrix } from './AgentEvidenceMatrix'
import { AgentTraceReferenceList } from './AgentTraceReferenceList'
import { AgentTraceSummaryStats } from './AgentTraceSummaryStats'
import { AgentUnsupportedClaims } from './AgentUnsupportedClaims'
import type { AgentSourceSummaryViewModel } from './useAgentTraceViewModel'
import { tx } from './agentTracePanelUtils'

export function AgentSourceSummaryPanel({
  labels,
  viewModel,
  onOpenReference,
  onAddReferenceToShelf,
}: {
  labels: Partial<StringMap>
  viewModel: AgentSourceSummaryViewModel
  onOpenReference?: (detail: CiteDetail, ref: Record<string, unknown>) => void
  onAddReferenceToShelf?: (detail: CiteDetail, ref: Record<string, unknown>) => void
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
      {references.length > 0 ? (
        <div className="kb-agent-trace-section kb-agent-trace-public-refs" data-testid="agent-trace-public-references">
          <div className="kb-agent-trace-heading">{tx(labels, 'agent_trace_references', 'References')}</div>
          <AgentTraceReferenceList
            references={references}
            labels={labels}
            onOpenReference={onOpenReference}
            onAddReferenceToShelf={onAddReferenceToShelf}
          />
        </div>
      ) : null}
    </>
  )
}
