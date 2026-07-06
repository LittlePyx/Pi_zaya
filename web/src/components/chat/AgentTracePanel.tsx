import { useT } from '../../i18n'
import { internalDebugEnvEnabled } from '../../utils/internalDebug'
import type { CiteDetail } from './citationState'
import { AgentSourceSummaryPanel } from './AgentSourceSummaryPanel'
import { AgentTraceDiagnosticsPanel } from './AgentTraceDiagnosticsPanel'
import { tx } from './agentTracePanelUtils'
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
    const note = loadStatus === 'loading'
      ? tx(S, 'agent_trace_loading_stored', 'Loading saved source check...')
      : loadStatus === 'error'
        ? tx(S, 'agent_trace_load_failed', 'Saved source check could not be loaded.')
        : loadStatus === 'empty'
          ? tx(S, 'agent_trace_no_stored', 'No saved source check is available.')
          : tx(S, 'agent_trace_open_to_load', 'Open to load saved source check.')
    return (
      <details className="kb-agent-trace" onToggle={(event) => {
        if ((event.currentTarget as HTMLDetailsElement).open) void loadArchivedTrace()
      }}>
        <summary>
          <span>{tx(S, 'agent_trace_title', 'Sources & evidence')}</span>
          <span>{tx(S, 'agent_trace_stored', 'Saved check')}</span>
          <span>{loadStatus === 'loading' ? tx(S, 'agent_trace_loading', 'loading') : tx(S, 'agent_trace_open_load', 'open to load')}</span>
        </summary>
        <div className="kb-agent-trace-empty">{note}</div>
      </details>
    )
  }

  const showDiagnostics = internalDebugEnvEnabled()

  return (
    <details className="kb-agent-trace" onToggle={(event) => {
      if ((event.currentTarget as HTMLDetailsElement).open) void loadArchivedTrace()
    }}>
      <summary>
        <span>{tx(S, 'agent_trace_title', 'Sources & evidence')}</span>
        <span>{viewModel.headerEvidence}</span>
        <span>{viewModel.headerContext}</span>
      </summary>
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
    </details>
  )
}
