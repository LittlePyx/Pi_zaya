import { useT } from '../../i18n'
import { internalDebugEnvEnabled } from '../../utils/internalDebug'
import type { CiteDetail } from './citationState'
import { AgentTraceFrame } from './AgentTraceFrame'
import { AgentSourceSummaryPanel } from './AgentSourceSummaryPanel'
import { AgentTraceDiagnosticsPanel } from './AgentTraceDiagnosticsPanel'
import { AgentTraceStoredPrompt } from './AgentTraceStoredPrompt'
import { useArchivedAgentTrace, type LoadArchivedAgentTrace } from './useArchivedAgentTrace'
import { useAgentTraceViewModel } from './useAgentTraceViewModel'

export function AgentTracePanel({
  trace,
  messageId,
  canLoadTrace,
  onLoadTrace,
  onOpenReference,
  onAddReferenceToShelf,
}: {
  trace?: Record<string, unknown> | null
  messageId?: number
  canLoadTrace?: boolean
  onLoadTrace?: LoadArchivedAgentTrace
  onOpenReference?: (detail: CiteDetail, ref: Record<string, unknown>) => void
  onAddReferenceToShelf?: (detail: CiteDetail, ref: Record<string, unknown>) => void
}) {
  const S = useT()
  const {
    traceRecord,
    hasTrace,
    canLazyLoad,
    loadStatus,
    loadArchivedTrace,
  } = useArchivedAgentTrace({
    trace,
    messageId,
    canLoadTrace,
    onLoadTrace,
  })
  const viewModel = useAgentTraceViewModel(traceRecord, S)
  if (!hasTrace && !canLazyLoad) return null
  const mode = String(traceRecord.mode || '').trim()
  if (hasTrace && mode && mode !== 'research_agent') return null

  if (!hasTrace) {
    return (
      <AgentTraceStoredPrompt
        labels={S}
        loadStatus={loadStatus}
        onLoad={loadArchivedTrace}
      />
    )
  }

  const showDiagnostics = internalDebugEnvEnabled()

  return (
    <AgentTraceFrame
      labels={S}
      summaryStatus={viewModel.headerEvidence}
      summaryContext={viewModel.headerContext}
      open={loadStatus === 'loaded' ? true : undefined}
      onOpen={loadArchivedTrace}
    >
      <AgentSourceSummaryPanel
        labels={S}
        viewModel={viewModel.sourceSummary}
        onOpenReference={onOpenReference}
        onAddReferenceToShelf={onAddReferenceToShelf}
      />
      {showDiagnostics ? (
        <AgentTraceDiagnosticsPanel
          labels={S}
          viewModel={viewModel.diagnostics}
          onOpenReference={onOpenReference}
          onAddReferenceToShelf={onAddReferenceToShelf}
        />
      ) : null}
    </AgentTraceFrame>
  )
}
