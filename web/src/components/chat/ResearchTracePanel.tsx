import {
  asTraceRecord,
  formatTraceMs,
  shouldShowResearchTracePanel,
  traceNum,
  traceSourceLabels,
} from './messageTraceUtils'

export function ResearchTracePanel({ trace }: { trace?: Record<string, unknown> | null }) {
  if (!shouldShowResearchTracePanel()) return null
  const tr = asTraceRecord(trace)
  const traceId = String(tr.trace_id || '').trim()
  const timings = asTraceRecord(tr.timings_ms)
  const retrieval = asTraceRecord(tr.retrieval)
  const answer = asTraceRecord(tr.answer)
  const refsTrace = asTraceRecord(tr.refs)
  const citeSystems = asTraceRecord(tr.citation_systems)
  if (!traceId && Object.keys(timings).length <= 0 && Object.keys(retrieval).length <= 0) {
    return null
  }
  const topSources = traceSourceLabels(retrieval.top_hits)
  const answerSources = traceSourceLabels(answer.answer_sources)
  const refSources = traceSourceLabels(refsTrace.final_display_sources || refsTrace.seed_sources)
  const evidenceMismatch = Boolean(refsTrace.primary_evidence_mismatch)
  const evidenceHeading = String(refsTrace.primary_evidence_heading || '').trim()
  const evidenceTerms = Array.isArray(refsTrace.primary_evidence_terms)
    ? refsTrace.primary_evidence_terms.map((item) => String(item || '').trim()).filter(Boolean).slice(0, 4)
    : []
  const evidenceLabel = evidenceHeading ? (evidenceMismatch ? 'weak' : 'ok') : 'n/a'
  const total = formatTraceMs(timings.total)
  const status = String(tr.status || '').trim() || 'running'
  return (
    <details className="kb-research-trace">
      <summary>
        <span>Trace</span>
        <span>{status}</span>
        <span>{total}</span>
      </summary>
      <div className="kb-research-trace-grid">
        <div><span>retrieve</span><strong>{formatTraceMs(timings.retrieve)}</strong></div>
        <div><span>answer</span><strong>{formatTraceMs(timings.llm_answer)}</strong></div>
        <div><span>refs</span><strong>{formatTraceMs(timings.refs_precompute)}</strong></div>
        <div><span>hits</span><strong>{traceNum(retrieval.raw_hit_count)}</strong></div>
        <div><span>answer docs</span><strong>{traceNum(answer.answer_hit_count)}</strong></div>
        <div><span>cards</span><strong>{traceNum(refsTrace.final_display_count || refsTrace.seed_count)}</strong></div>
        <div><span>System B</span><strong>{traceNum(citeSystems.system_b_validated_count || citeSystems.system_b_opportunity_count)}</strong></div>
        <div className={evidenceMismatch ? 'is-warning' : ''}>
          <span>evidence</span>
          <strong title={evidenceTerms.join(', ') || evidenceHeading}>{evidenceLabel}</strong>
        </div>
        <div><span>refs async</span><strong>{refsTrace.async_will_run ? 'yes' : 'no'}</strong></div>
      </div>
      {topSources.length > 0 || answerSources.length > 0 || refSources.length > 0 ? (
        <div className="kb-research-trace-sources">
          {topSources.length > 0 ? <div><span>retrieved</span>{topSources.join(' / ')}</div> : null}
          {answerSources.length > 0 ? <div><span>answer</span>{answerSources.join(' / ')}</div> : null}
          {refSources.length > 0 ? <div><span>refs</span>{refSources.join(' / ')}</div> : null}
        </div>
      ) : null}
      {traceId ? <div className="kb-research-trace-id">{traceId}</div> : null}
    </details>
  )
}
