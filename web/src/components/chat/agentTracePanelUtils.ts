import type { StringMap } from '../../i18n'
import { normalizeCiteDetail, type CiteDetail } from './citationState'
import { asTraceRecord, traceNum } from './messageTraceUtils'

export function tx(labels: Partial<StringMap> | undefined, key: string, fallback: string): string {
  return String(labels?.[key] || fallback)
}

export function txFmt(labels: Partial<StringMap> | undefined, key: string, fallback: string, values: Record<string, string | number>): string {
  let text = tx(labels, key, fallback)
  for (const [name, value] of Object.entries(values)) {
    text = text.replaceAll(`{${name}}`, String(value))
  }
  return text
}

export function records(value: unknown): Record<string, unknown>[] {
  return Array.isArray(value)
    ? value.map((item) => asTraceRecord(item)).filter((item) => Object.keys(item).length > 0)
    : []
}

export function shortText(value: unknown, limit = 180): string {
  const text = String(value || '').replace(/\s+/g, ' ').trim()
  return text.length > limit ? `${text.slice(0, limit - 3).trim()}...` : text
}

export function statusClass(status: unknown) {
  const s = String(status || '').trim().toLowerCase()
  if (s === 'error' || s === 'canceled') return 'is-warning'
  if (s === 'done') return 'is-done'
  return ''
}

export function refText(ref: Record<string, unknown>, ...keys: string[]): string {
  for (const key of keys) {
    const text = String(ref[key] || '').trim()
    if (text) return text
  }
  return ''
}

export function referenceTitle(ref: Record<string, unknown>): string {
  const refNum = traceNum(ref.ref_num || ref.num)
  const prefix = refNum > 0 ? `[${refNum}] ` : ''
  return `${prefix}${shortText(ref.title || ref.raw || ref.source_name || 'Reference', 132)}`
}

export function referenceMeta(ref: Record<string, unknown>): string {
  return [ref.authors, ref.year, ref.venue, ref.doi]
    .map((item) => shortText(item, 70))
    .filter(Boolean)
    .join(' / ')
}

export function unsupportedReasonText(value: unknown, labels?: Partial<StringMap>): string {
  const reason = String(value || '').trim()
  if (reason === 'missing_citation') return tx(labels, 'agent_trace_missing_citation', 'Missing citation')
  if (reason === 'no_evidence_hits') return tx(labels, 'agent_trace_no_evidence_hits', 'No evidence hits')
  if (reason === 'missing_evidence_overlap') return tx(labels, 'agent_trace_missing_evidence_overlap', 'Citation does not match retrieved evidence')
  return reason || tx(labels, 'agent_trace_unsupported', 'Unsupported')
}

export function compactStringList(value: unknown, limit = 4): string[] {
  return Array.isArray(value)
    ? value.map((item) => shortText(item, 90)).filter(Boolean).slice(0, limit)
    : []
}

export function questionTypeLabel(value: unknown, labels?: Partial<StringMap>): string {
  const text = String(value || '').trim().toLowerCase()
  if (text === 'single_paper_qa') return tx(labels, 'agent_trace_type_single', 'Single paper')
  if (text === 'multi_paper_comparison') return tx(labels, 'agent_trace_type_comparison', 'Comparison')
  if (text === 'reading_guide') return tx(labels, 'agent_trace_type_reading_guide', 'Reading guide')
  if (text === 'reference_followup') return tx(labels, 'agent_trace_type_reference_followup', 'Reference follow-up')
  return tx(labels, 'agent_trace_type_general', 'General')
}

export function verificationHeaderText(totalClaims: number, supportedClaims: number, unsupportedClaims: number, hasErrors: boolean, labels?: Partial<StringMap>): string {
  if (hasErrors) return tx(labels, 'agent_trace_needs_check', 'Needs check')
  if (totalClaims > 0 && unsupportedClaims > 0) {
    return txFmt(labels, 'agent_trace_review_fraction', 'Review {unsupported}/{total}', { unsupported: unsupportedClaims, total: totalClaims })
  }
  if (totalClaims > 0) {
    return txFmt(labels, 'agent_trace_checked_fraction', 'Checked {supported}/{total}', { supported: supportedClaims, total: totalClaims })
  }
  return tx(labels, 'agent_trace_available', 'Source check available')
}

export function evidenceStatusValue(value: unknown): 'grounded' | 'needs_review' | 'insufficient' | 'not_applicable' | '' {
  const text = String(value || '').trim().toLowerCase()
  if (text === 'grounded' || text === 'needs_review' || text === 'insufficient' || text === 'not_applicable') return text
  return ''
}

export function evidenceStatusLabel(value: unknown, labels?: Partial<StringMap>): string {
  const status = evidenceStatusValue(value)
  if (status === 'grounded') return tx(labels, 'agent_trace_evidence_grounded', 'Evidence grounded')
  if (status === 'needs_review') return tx(labels, 'agent_trace_evidence_needs_review', 'Needs review')
  if (status === 'insufficient') return tx(labels, 'agent_trace_evidence_insufficient', 'Insufficient evidence')
  if (status === 'not_applicable') return tx(labels, 'agent_trace_evidence_not_from_kb', 'Not from KB')
  return ''
}

export function evidenceStatusClass(value: unknown): string {
  const status = evidenceStatusValue(value)
  if (status === 'grounded') return 'is-grounded'
  if (status === 'needs_review') return 'is-warning'
  if (status === 'insufficient') return 'is-danger'
  return ''
}

export function qualityGateLabel(value: unknown, labels?: Partial<StringMap>): string {
  const text = String(value || '').trim().toLowerCase()
  if (text === 'passed') return tx(labels, 'agent_trace_quality_passed', 'Passed')
  if (text === 'repaired') return tx(labels, 'agent_trace_quality_repaired', 'Repaired')
  if (text === 'fallback') return tx(labels, 'agent_trace_quality_fallback', 'Fallback')
  return ''
}

export function qualityGateClass(value: unknown): string {
  const text = String(value || '').trim().toLowerCase()
  if (text === 'passed') return 'is-grounded'
  if (text === 'repaired') return 'is-warning'
  if (text === 'fallback') return 'is-danger'
  return ''
}

export function sourcePolicyLabel(value: unknown, labels?: Partial<StringMap>): string {
  const text = String(value || '').trim()
  if (text === 'local_only') return tx(labels, 'agent_trace_source_local_only', 'Local KB')
  if (text === 'local_plus_external_background') return tx(labels, 'agent_trace_source_local_external', 'Local + external')
  if (text === 'external_allowed_with_notice') return tx(labels, 'agent_trace_source_external_allowed', 'External allowed')
  if (text === 'trusted_sites_only') return tx(labels, 'agent_trace_source_trusted_sites', 'Trusted sites')
  return text
}

export function traceBool(value: unknown, fallback = false): boolean {
  if (typeof value === 'boolean') return value
  const text = String(value ?? '').trim().toLowerCase()
  if (!text) return fallback
  if (['1', 'true', 'yes', 'on'].includes(text)) return true
  if (['0', 'false', 'no', 'off'].includes(text)) return false
  return fallback
}

export function referenceDetail(ref: Record<string, unknown>): CiteDetail | null {
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

export function traceStepReferences(steps: Record<string, unknown>[], limit = 4): Record<string, unknown>[] {
  const out: Record<string, unknown>[] = []
  const seen = new Set<string>()
  for (const step of steps) {
    const output = asTraceRecord(step.output)
    for (const ref of records(output.references)) {
      const key = [
        refText(ref, 'source_path', 'sourcePath'),
        traceNum(ref.ref_num || ref.num),
        refText(ref, 'title', 'raw', 'source_name', 'sourceName'),
      ].join('|')
      if (seen.has(key)) continue
      seen.add(key)
      out.push(ref)
      if (out.length >= limit) return out
    }
  }
  return out
}
