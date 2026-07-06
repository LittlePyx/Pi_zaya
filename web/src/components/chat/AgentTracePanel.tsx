import { useState } from 'react'
import type { AgentTraceAuditResponse } from '../../api/chat'
import { useT } from '../../i18n'
import { internalDebugEnvEnabled } from '../../utils/internalDebug'
import type { CiteDetail } from './citationState'
import { AgentSourceSummaryPanel } from './AgentSourceSummaryPanel'
import { AgentTraceDiagnosticsPanel } from './AgentTraceDiagnosticsPanel'
import {
  compactStringList,
  evidenceStatusLabel,
  evidenceStatusValue,
  questionTypeLabel,
  records,
  shortText,
  traceBool,
  traceStepReferences,
  tx,
  verificationHeaderText,
} from './agentTracePanelUtils'
import { asTraceRecord, traceNum } from './messageTraceUtils'

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

  const plan = records(tr.plan)
  const steps = records(tr.steps)
  const context = asTraceRecord(tr.context)
  const verification = asTraceRecord(tr.verification)
  const researchRun = asTraceRecord(tr.research_run)
  const summary = asTraceRecord(tr.summary)
  const errors = Array.isArray(tr.errors) ? tr.errors : []
  const evidenceMatrix = records(researchRun.evidence_matrix)
  const researchSubtasks = records(researchRun.subtasks)
  const claimRows = records(verification.claims)
  const unsupportedClaimRows = claimRows
    .filter((claim) => claim.supported === false || String(claim.unsupported_reason || '').trim())
    .slice(0, 3)
  const totalClaims = 'total_claims' in summary ? traceNum(summary.total_claims) : traceNum(verification.total_claims)
  const supportedClaims = 'supported_claims' in summary ? traceNum(summary.supported_claims) : traceNum(verification.supported_claims)
  const unsupportedClaims = 'unsupported_claims' in summary ? traceNum(summary.unsupported_claims) : traceNum(verification.unsupported_claims)
  const planStepCount = 'plan_step_count' in summary ? traceNum(summary.plan_step_count) : plan.length
  const toolCallCount = 'tool_call_count' in summary ? traceNum(summary.tool_call_count) : steps.length
  const hasErrors = 'has_errors' in summary ? traceBool(summary.has_errors) : errors.length > 0
  const researchRunStatus = String(summary.research_run_status || researchRun.status || '').trim()
  const sourcePolicy = String(summary.source_policy || researchRun.source_policy || '').trim()
  const evidenceMatrixRows = 'evidence_matrix_rows' in summary ? traceNum(summary.evidence_matrix_rows) : evidenceMatrix.length
  const subtaskCount = 'subtask_count' in summary ? traceNum(summary.subtask_count) : researchSubtasks.length
  const questionType = String(summary.question_type || tr.question_type || 'unknown').trim()
  const queryScope = String(summary.query_scope || context.query_scope || context.queryScope || '').trim()
  const requestedScope = String(summary.requested_query_scope || context.requested_query_scope || context.requestedQueryScope || '').trim()
  const evidenceStatus = evidenceStatusValue(summary.evidence_status || verification.evidence_status)
  const evidenceLabel = evidenceStatusLabel(evidenceStatus, S)
  const qualityGateStatus = String(summary.quality_gate_status || '').trim().toLowerCase()
  const qualityGateTitle = [
    ...compactStringList(summary.quality_gate_reasons),
    ...compactStringList(summary.quality_gate_warnings),
  ].join(' / ')
  const taskLabel = evidenceStatus === 'not_applicable' ? tx(S, 'agent_trace_type_general', 'General') : questionTypeLabel(questionType, S)
  const selectedCount = traceNum(context.selected_research_context_count || context.selectedResearchContextCount)
  const currentSource = shortText(context.current_source_name || context.currentSourceName || context.current_source_path || context.currentSourcePath, 90)
  const scopeBits = [
    queryScope,
    requestedScope && requestedScope !== queryScope ? `requested ${requestedScope}` : '',
    selectedCount > 0 ? `${selectedCount} selected` : '',
    queryScope === 'current_paper' && currentSource ? currentSource : '',
  ].filter(Boolean)
  const scopeSummary = scopeBits.join(' / ')
  const claimSummary = verificationHeaderText(totalClaims, supportedClaims, unsupportedClaims, hasErrors, S)
  const headerEvidence = evidenceLabel || claimSummary
  const headerContext = totalClaims > 0 && evidenceLabel ? claimSummary : (scopeSummary ? shortText(scopeSummary, 42) : taskLabel)
  const publicReferences = traceStepReferences(steps)
  const showDiagnostics = internalDebugEnvEnabled()

  return (
    <details className="kb-agent-trace" onToggle={(event) => {
      if ((event.currentTarget as HTMLDetailsElement).open) void loadArchivedTrace()
    }}>
      <summary>
        <span>{tx(S, 'agent_trace_title', 'Sources & evidence')}</span>
        <span>{headerEvidence}</span>
        <span>{headerContext}</span>
      </summary>
      <AgentSourceSummaryPanel
        labels={S}
        evidenceLabel={evidenceLabel}
        evidenceStatus={evidenceStatus}
        totalClaims={totalClaims}
        supportedClaims={supportedClaims}
        unsupportedClaims={unsupportedClaims}
        qualityGateStatus={qualityGateStatus}
        qualityGateTitle={qualityGateTitle}
        taskLabel={taskLabel}
        scopeSummary={scopeSummary}
        hasErrors={hasErrors}
        researchRunStatus={researchRunStatus}
        evidenceMatrixRows={evidenceMatrixRows}
        sourcePolicy={sourcePolicy}
        evidenceMatrix={evidenceMatrix}
        subtaskCount={subtaskCount}
        unsupportedClaimRows={unsupportedClaimRows}
        references={publicReferences}
        onOpenReference={onOpenReference}
        onAddReferenceToShelf={onAddReferenceToShelf}
      />
      {showDiagnostics ? (
        <AgentTraceDiagnosticsPanel
          labels={S}
          plan={plan}
          steps={steps}
          planStepCount={planStepCount}
          toolCallCount={toolCallCount}
          onOpenReference={onOpenReference}
          onAddReferenceToShelf={onAddReferenceToShelf}
        />
      ) : null}
    </details>
  )
}
