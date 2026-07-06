import { useState } from 'react'
import type { AgentTraceAuditResponse } from '../../api/chat'
import { useT } from '../../i18n'
import { internalDebugEnvEnabled } from '../../utils/internalDebug'
import type { CiteDetail } from './citationState'
import { AgentSourceSummaryPanel } from './AgentSourceSummaryPanel'
import { AgentTraceDiagnosticsPanel } from './AgentTraceDiagnosticsPanel'
import { tx } from './agentTracePanelUtils'
import { asTraceRecord } from './messageTraceUtils'
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
  onLoadTrace?: (messageId: number) => Promise<AgentTraceAuditResponse>
  onOpenReference?: (detail: CiteDetail, ref: Record<string, unknown>) => void
  onAddReferenceToShelf?: (detail: CiteDetail, ref: Record<string, unknown>) => void
}) {
  const S = useT()
  const initialTrace = asTraceRecord(trace)
  const [loadedState, setLoadedState] = useState<{
    messageId: number
    trace: Record<string, unknown> | null
    status: 'idle' | 'loading' | 'loaded' | 'empty' | 'error'
  }>({ messageId: 0, trace: null, status: 'idle' })

  const hasInitialTrace = Object.keys(initialTrace).length > 0
  const currentMessageId = Number(messageId || 0)
  const loadedTraceRecord = loadedState.messageId === currentMessageId ? asTraceRecord(loadedState.trace) : {}
  const loadStatus = loadedState.messageId === currentMessageId ? loadedState.status : 'idle'
  const tr = hasInitialTrace ? initialTrace : loadedTraceRecord
  const hasTrace = Object.keys(tr).length > 0
  const canLazyLoad = Boolean(!hasInitialTrace && canLoadTrace && onLoadTrace && Number(messageId || 0) > 0)
  const viewModel = useAgentTraceViewModel(tr, S)
  if (!hasTrace && !canLazyLoad) return null
  const mode = String(tr.mode || '').trim()
  if (hasTrace && mode && mode !== 'research_agent') return null

  const loadArchivedTrace = async () => {
    if (!canLazyLoad || loadStatus === 'loading' || loadStatus === 'loaded') return
    const mid = Number(messageId || 0)
    if (!mid || !onLoadTrace) return
    setLoadedState({ messageId: mid, trace: null, status: 'loading' })
    try {
      const res = await onLoadTrace(mid)
      const loadedTrace = asTraceRecord(res.agent_trace)
      const auditSummary = asTraceRecord(res.summary)
      const nextTrace = Object.keys(loadedTrace).length > 0 && Object.keys(auditSummary).length > 0 && Object.keys(asTraceRecord(loadedTrace.summary)).length <= 0
        ? { ...loadedTrace, summary: auditSummary }
        : loadedTrace
      if (res.available !== false && Object.keys(nextTrace).length > 0) {
        setLoadedState({ messageId: mid, trace: nextTrace, status: 'loaded' })
      } else {
        setLoadedState({ messageId: mid, trace: null, status: 'empty' })
      }
    } catch {
      setLoadedState({ messageId: mid, trace: null, status: 'error' })
    }
  }

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
