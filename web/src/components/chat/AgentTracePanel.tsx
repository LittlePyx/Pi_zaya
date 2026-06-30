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
  onOpenReference,
  onAddReferenceToShelf,
}: {
  trace?: Record<string, unknown> | null
  onOpenReference?: (detail: CiteDetail, ref: Record<string, unknown>) => void
  onAddReferenceToShelf?: (detail: CiteDetail, ref: Record<string, unknown>) => void
}) {
  const tr = asTraceRecord(trace)
  if (Object.keys(tr).length <= 0) return null
  const mode = String(tr.mode || '').trim()
  if (mode && mode !== 'research_agent') return null

  const plan = records(tr.plan)
  const steps = records(tr.steps)
  const verification = asTraceRecord(tr.verification)
  const totalClaims = traceNum(verification.total_claims)
  const supportedClaims = traceNum(verification.supported_claims)
  const unsupportedClaims = traceNum(verification.unsupported_claims)
  const questionType = String(tr.question_type || 'unknown').trim()
  const status = String(tr.status || '').trim() || 'done'

  return (
    <details className="kb-agent-trace">
      <summary>
        <span>Research Agent Trace</span>
        <span>{questionType}</span>
        <span>{status}</span>
      </summary>
      <div className="kb-agent-trace-summary">
        <div>
          <span>Claims</span>
          <strong>{supportedClaims}/{totalClaims}</strong>
        </div>
        <div className={unsupportedClaims > 0 ? 'is-warning' : ''}>
          <span>Unsupported</span>
          <strong>{unsupportedClaims}</strong>
        </div>
      </div>
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
  )
}
