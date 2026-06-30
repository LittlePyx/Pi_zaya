import { useState } from 'react'
import type { AgentTraceAuditResponse } from '../../api/chat'
import { normalizeCiteDetail, type CiteDetail } from './citationState'
import { asTraceRecord, traceNum } from './messageTraceUtils'

function records(value: unknown): Record<string, unknown>[] {
  return Array.isArray(value)
    ? value.map((item) => asTraceRecord(item)).filter((item) => Object.keys(item).length > 0)
    : []
}

function shortText(value: unknown, limit = 180): string {
  const text = String(value || '').replace(/\s+/g, ' ').trim()
  return text.length > limit ? `${text.slice(0, limit - 3).trim()}...` : text
}

function statusClass(status: unknown) {
  const s = String(status || '').trim().toLowerCase()
  if (s === 'error' || s === 'canceled') return 'is-warning'
  if (s === 'done') return 'is-done'
  return ''
}

function refText(ref: Record<string, unknown>, ...keys: string[]): string {
  for (const key of keys) {
    const text = String(ref[key] || '').trim()
    if (text) return text
  }
  return ''
}

function referenceTitle(ref: Record<string, unknown>): string {
  const refNum = traceNum(ref.ref_num || ref.num)
  const prefix = refNum > 0 ? `[${refNum}] ` : ''
  return `${prefix}${shortText(ref.title || ref.raw || ref.source_name || 'Reference', 132)}`
}

function referenceMeta(ref: Record<string, unknown>): string {
  return [ref.authors, ref.year, ref.venue, ref.doi]
    .map((item) => shortText(item, 70))
    .filter(Boolean)
    .join(' / ')
}

function unsupportedReasonText(value: unknown): string {
  const reason = String(value || '').trim()
  if (reason === 'missing_citation') return 'Missing citation'
  if (reason === 'no_evidence_hits') return 'No evidence hits'
  if (reason === 'missing_evidence_overlap') return 'Citation does not match retrieved evidence'
  return reason || 'Unsupported'
}

function questionTypeLabel(value: unknown): string {
  const text = String(value || '').trim().toLowerCase()
  if (text === 'single_paper_qa') return 'Single paper'
  if (text === 'multi_paper_comparison') return 'Comparison'
  if (text === 'reading_guide') return 'Reading guide'
  if (text === 'reference_followup') return 'Reference follow-up'
  return 'General'
}

function verificationHeaderText(totalClaims: number, supportedClaims: number, unsupportedClaims: number, hasErrors: boolean): string {
  if (hasErrors) return 'Needs check'
  if (totalClaims > 0 && unsupportedClaims > 0) return `Review ${unsupportedClaims}/${totalClaims}`
  if (totalClaims > 0) return `Checked ${supportedClaims}/${totalClaims}`
  return 'Trace available'
}

function evidenceStatusValue(value: unknown): 'grounded' | 'needs_review' | 'insufficient' | 'not_applicable' | '' {
  const text = String(value || '').trim().toLowerCase()
  if (text === 'grounded' || text === 'needs_review' || text === 'insufficient' || text === 'not_applicable') return text
  return ''
}

function evidenceStatusLabel(value: unknown): string {
  const status = evidenceStatusValue(value)
  if (status === 'grounded') return 'Evidence grounded'
  if (status === 'needs_review') return 'Needs review'
  if (status === 'insufficient') return 'Insufficient evidence'
  if (status === 'not_applicable') return 'General answer'
  return ''
}

function evidenceStatusClass(value: unknown): string {
  const status = evidenceStatusValue(value)
  if (status === 'grounded') return 'is-grounded'
  if (status === 'needs_review') return 'is-warning'
  if (status === 'insufficient') return 'is-danger'
  return ''
}

function traceBool(value: unknown, fallback = false): boolean {
  if (typeof value === 'boolean') return value
  const text = String(value ?? '').trim().toLowerCase()
  if (!text) return fallback
  if (['1', 'true', 'yes', 'on'].includes(text)) return true
  if (['0', 'false', 'no', 'off'].includes(text)) return false
  return fallback
}

function referenceDetail(ref: Record<string, unknown>): CiteDetail | null {
  const refNum = traceNum(ref.ref_num || ref.num)
  const sourcePath = refText(ref, 'source_path', 'sourcePath')
  const title = refText(ref, 'title') || refText(ref, 'raw') || refText(ref, 'source_name', 'sourceName') || 'Reference'
  const anchor = refText(ref, 'anchor') || `agent-ref:${sourcePath}:${refNum}:${title}`.slice(0, 180)
  return normalizeCiteDetail({
    ...ref,
    anchor,
    num: refNum,
    display_num: refNum,
    source_name: refText(ref, 'source_name', 'sourceName', 'source_paper') || sourcePath,
    source_path: sourcePath,
    raw: refText(ref, 'raw', 'cite_fmt', 'citeFmt') || title,
    cite_fmt: refText(ref, 'cite_fmt', 'citeFmt', 'raw') || title,
    is_inpaper: ref.is_inpaper ?? ref.isInpaper ?? true,
    title,
    authors: refText(ref, 'authors'),
    venue: refText(ref, 'venue'),
    year: refText(ref, 'year'),
    doi: refText(ref, 'doi'),
    doi_url: refText(ref, 'doi_url', 'doiUrl'),
    heading_path: refText(ref, 'heading_path', 'headingPath'),
    evidence_quote: refText(ref, 'evidence_quote', 'evidenceQuote', 'evidence_preview', 'evidencePreview'),
    evidence_source: refText(ref, 'evidence_source', 'evidenceSource') || 'agent_trace_reference',
    citation_context: refText(ref, 'citation_context', 'citationContext', 'evidence_preview', 'evidencePreview'),
    citation_context_source: refText(ref, 'citation_context_source', 'citationContextSource') || 'agent_trace',
    upstream_work_role: refText(ref, 'upstream_work_role', 'upstreamWorkRole', 'why_relevant', 'whyRelevant'),
    user_question_relation: refText(ref, 'user_question_relation', 'userQuestionRelation', 'why_relevant', 'whyRelevant'),
    why_line: refText(ref, 'why_line', 'whyLine', 'why_relevant', 'whyRelevant'),
    support_relation: refText(ref, 'support_relation', 'supportRelation', 'why_relevant', 'whyRelevant'),
    location_label: refText(ref, 'location_label', 'locationLabel', 'heading_path', 'headingPath'),
    shelf_item_kind: refText(ref, 'shelf_item_kind', 'shelfItemKind') || 'reference',
    shelf_origin: refText(ref, 'shelf_origin', 'shelfOrigin') || 'agent_trace',
    shelf_excerpt: refText(ref, 'shelf_excerpt', 'shelfExcerpt', 'raw') || title,
    shelf_excerpt_label: refText(ref, 'shelf_excerpt_label', 'shelfExcerptLabel') || 'Reference entry',
    card_kind: refText(ref, 'card_kind', 'cardKind') || 'reference',
    card_title: refText(ref, 'card_title', 'cardTitle') || title,
    card_subtitle: refText(ref, 'card_subtitle', 'cardSubtitle') || referenceMeta(ref),
    card_reference_entry: refText(ref, 'card_reference_entry', 'cardReferenceEntry', 'raw'),
    card_context_summary: refText(ref, 'card_context_summary', 'cardContextSummary', 'why_relevant', 'whyRelevant'),
    card_evidence: refText(ref, 'card_evidence', 'cardEvidence', 'evidence_preview', 'evidencePreview'),
    card_locator: refText(ref, 'card_locator', 'cardLocator', 'heading_path', 'headingPath'),
    card_support_explanation: refText(ref, 'card_support_explanation', 'cardSupportExplanation', 'why_relevant', 'whyRelevant'),
    page_start: traceNum(ref.page_start || ref.pageStart),
    page_end: traceNum(ref.page_end || ref.pageEnd),
  })
}

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
      ? 'Loading stored trace...'
      : loadStatus === 'error'
        ? 'Stored trace could not be loaded.'
        : loadStatus === 'empty'
          ? 'No stored trace is available.'
          : 'Open to load stored trace.'
    return (
      <details className="kb-agent-trace" onToggle={(event) => {
        if ((event.currentTarget as HTMLDetailsElement).open) void loadArchivedTrace()
      }}>
        <summary>
          <span>Research Agent Trace</span>
          <span>Stored trace</span>
          <span>{loadStatus === 'loading' ? 'loading' : 'open to load'}</span>
        </summary>
        <div className="kb-agent-trace-empty">{note}</div>
      </details>
    )
  }

  const plan = records(tr.plan)
  const steps = records(tr.steps)
  const context = asTraceRecord(tr.context)
  const verification = asTraceRecord(tr.verification)
  const summary = asTraceRecord(tr.summary)
  const errors = Array.isArray(tr.errors) ? tr.errors : []
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
  const questionType = String(summary.question_type || tr.question_type || 'unknown').trim()
  const queryScope = String(summary.query_scope || context.query_scope || context.queryScope || '').trim()
  const requestedScope = String(summary.requested_query_scope || context.requested_query_scope || context.requestedQueryScope || '').trim()
  const evidenceStatus = evidenceStatusValue(summary.evidence_status || verification.evidence_status)
  const evidenceLabel = evidenceStatusLabel(evidenceStatus)
  const taskLabel = evidenceStatus === 'not_applicable' ? 'General' : questionTypeLabel(questionType)
  const selectedCount = traceNum(context.selected_research_context_count || context.selectedResearchContextCount)
  const currentSource = shortText(context.current_source_name || context.currentSourceName || context.current_source_path || context.currentSourcePath, 90)
  const scopeBits = [
    queryScope,
    requestedScope && requestedScope !== queryScope ? `requested ${requestedScope}` : '',
    selectedCount > 0 ? `${selectedCount} selected` : '',
    queryScope === 'current_paper' && currentSource ? currentSource : '',
  ].filter(Boolean)
  const scopeSummary = scopeBits.join(' / ')
  const claimSummary = verificationHeaderText(totalClaims, supportedClaims, unsupportedClaims, hasErrors)
  const headerEvidence = evidenceLabel || claimSummary
  const headerContext = totalClaims > 0 && evidenceLabel ? claimSummary : (scopeSummary ? shortText(scopeSummary, 42) : taskLabel)

  return (
    <details className="kb-agent-trace" onToggle={(event) => {
      if ((event.currentTarget as HTMLDetailsElement).open) void loadArchivedTrace()
    }}>
      <summary>
        <span>Research Agent Trace</span>
        <span>{headerEvidence}</span>
        <span>{headerContext}</span>
      </summary>
      <div className="kb-agent-trace-summary">
        {evidenceLabel ? (
          <div className={`kb-agent-trace-evidence-status ${evidenceStatusClass(evidenceStatus)}`} data-testid="agent-trace-evidence-status">
            <span>Evidence</span>
            <strong>{evidenceLabel}</strong>
          </div>
        ) : null}
        {totalClaims > 0 ? (
          <div>
            <span>Claims</span>
            <strong>{supportedClaims}/{totalClaims}</strong>
          </div>
        ) : null}
        {unsupportedClaims > 0 ? (
          <div className="is-warning">
            <span>Needs Review</span>
            <strong>{unsupportedClaims}</strong>
          </div>
        ) : null}
        <div>
          <span>Task</span>
          <strong>{taskLabel}</strong>
        </div>
        {scopeSummary ? (
          <div>
            <span>Scope</span>
            <strong title={scopeSummary}>{shortText(scopeSummary, 72)}</strong>
          </div>
        ) : null}
        {hasErrors ? (
          <div className="is-warning">
            <span>Run</span>
            <strong>errors</strong>
          </div>
        ) : null}
      </div>
      {unsupportedClaimRows.length > 0 ? (
        <div className="kb-agent-trace-section kb-agent-trace-unsupported">
          <div className="kb-agent-trace-heading">Needs Review</div>
          {unsupportedClaimRows.map((claim, idx) => (
            <div className="kb-agent-trace-claim" key={`${String(claim.index || 'claim')}-${idx}`} data-testid="agent-trace-unsupported-claim">
              <strong>{shortText(claim.claim_text || claim.text, 240)}</strong>
              <span>
                Needs Review: {unsupportedReasonText(claim.unsupported_reason)}
                {traceNum(claim.matched_evidence_count) > 0 ? ` / ${traceNum(claim.matched_evidence_count)} evidence match(es)` : ''}
              </span>
            </div>
          ))}
        </div>
      ) : null}
      {plan.length > 0 || steps.length > 0 ? (
        <details className="kb-agent-trace-details" data-testid="agent-trace-execution-details">
          <summary>
            <span>Execution Details</span>
            <span>{planStepCount} plan</span>
            <span>{toolCallCount} tools</span>
          </summary>
          {plan.length > 0 ? (
            <div className="kb-agent-trace-section">
              <div className="kb-agent-trace-heading">Plan</div>
              {plan.map((step, idx) => (
                <div className="kb-agent-trace-row" key={`${String(step.tool || 'plan')}-${idx}`}>
                  <span className={`kb-agent-trace-status ${statusClass(step.status)}`}>{String(step.status || 'pending')}</span>
                  <span className="kb-agent-trace-tool">{String(step.tool || '')}</span>
                  <span className="kb-agent-trace-text">{shortText(step.goal)}</span>
                </div>
              ))}
            </div>
          ) : null}
          {steps.length > 0 ? (
            <div className="kb-agent-trace-section">
              <div className="kb-agent-trace-heading">Tool Calls</div>
              {steps.map((step, idx) => {
                const output = asTraceRecord(step.output)
                const refs = records(output.references).slice(0, 3)
                return (
                  <div className="kb-agent-trace-call" key={`${String(step.tool || 'tool')}-${idx}`}>
                    <div className="kb-agent-trace-call-head">
                      <span className={`kb-agent-trace-status ${statusClass(step.status)}`}>{String(step.status || '')}</span>
                      <strong>{String(step.tool || '')}</strong>
                      {traceNum(step.elapsed_ms) > 0 ? <span>{traceNum(step.elapsed_ms)}ms</span> : null}
                    </div>
                    {step.observation ? <div className="kb-agent-trace-observation">{shortText(step.observation, 260)}</div> : null}
                    {refs.length > 0 ? (
                      <div className="kb-agent-trace-refs">
                        {refs.map((ref, refIdx) => {
                          const detail = referenceDetail(ref)
                          const canOpen = Boolean(detail?.sourcePath && onOpenReference)
                          const canAdd = Boolean(detail && onAddReferenceToShelf)
                          return (
                            <div className="kb-agent-trace-ref" key={`${String(ref.ref_num || ref.title || 'ref')}-${refIdx}`} data-testid="agent-trace-reference">
                              <strong data-testid="agent-trace-ref-title">{referenceTitle(ref)}</strong>
                              {referenceMeta(ref) ? <span>{referenceMeta(ref)}</span> : null}
                              {ref.why_relevant ? <em>{shortText(ref.why_relevant, 180)}</em> : null}
                              {canOpen || canAdd ? (
                                <div className="kb-agent-trace-ref-actions">
                                  {canOpen && detail ? (
                                    <button type="button" onClick={() => onOpenReference?.(detail, ref)} data-testid="agent-trace-ref-open">
                                      Open
                                    </button>
                                  ) : null}
                                  {canAdd && detail ? (
                                    <button type="button" onClick={() => onAddReferenceToShelf?.(detail, ref)} data-testid="agent-trace-ref-add">
                                      Add
                                    </button>
                                  ) : null}
                                </div>
                              ) : null}
                            </div>
                          )
                        })}
                      </div>
                    ) : null}
                    {step.error ? <div className="kb-agent-trace-error">{shortText(step.error, 260)}</div> : null}
                  </div>
                )
              })}
            </div>
          ) : null}
        </details>
      ) : null}
    </details>
  )
}
