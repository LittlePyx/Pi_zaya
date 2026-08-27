import type { ResearchNoteSourceLink, ResearchNoteSourceState } from '../api/chat'

export interface ResearchNoteEvidenceReviewIssue {
  sourceIndex: number
  status: 'skipped' | 'failed'
  reason: string
  checkedAt?: number
  link: ResearchNoteSourceLink
}

export interface ResearchNoteEvidenceReviewSummary {
  needsReview: number
  failed: number
  total: number
  firstSourceIndex: number | null
}

const USER_DECISION_REASONS = new Set(['candidate', 'source_changed', 'location_changed'])

export function researchNoteEvidenceReviewIssues(
  sourceState: ResearchNoteSourceState | null | undefined,
): ResearchNoteEvidenceReviewIssue[] {
  const links = Array.isArray(sourceState?.links) ? sourceState.links : []
  return links.flatMap((link, sourceIndex) => {
    if (!link || typeof link !== 'object' || link.kind === 'answer') return []
    if (!String(link.source_path || '').trim()) return []
    if (link.evidence_context_snapshot != null || link.evidence_verification?.status === 'invalid') return []
    const check = link.evidence_baseline_check
    const status = String(check?.status || '')
    const reason = String(check?.reason || '').trim()
    if (status !== 'skipped' || !USER_DECISION_REASONS.has(reason)) return []
    return [{
      sourceIndex,
      status,
      reason,
      checkedAt: check?.checked_at,
      link,
    }]
  })
}

export function researchNoteEvidenceSystemRetryRequired(
  sourceState: ResearchNoteSourceState | null | undefined,
): boolean {
  const links = Array.isArray(sourceState?.links) ? sourceState.links : []
  return links.some(link => {
    if (
      !link
      || typeof link !== 'object'
      || link.kind === 'answer'
      || !String(link.source_path || '').trim()
      || link.evidence_context_snapshot != null
      || link.evidence_verification?.status === 'invalid'
    ) return false
    const check = link.evidence_baseline_check
    return check?.status === 'failed' || check?.reason === 'unavailable'
  })
}

export function researchNoteEvidenceMaintenanceRequired(
  sourceState: ResearchNoteSourceState | null | undefined,
): boolean {
  const links = Array.isArray(sourceState?.links) ? sourceState.links : []
  return links.some(link => (
    Boolean(link && typeof link === 'object')
    && link.kind !== 'answer'
    && Boolean(String(link.source_path || '').trim())
    && link.evidence_context_snapshot == null
    && link.evidence_verification?.status !== 'invalid'
  ))
}

export function researchNoteEvidenceReviewSummary(
  sourceState: ResearchNoteSourceState | null | undefined,
): ResearchNoteEvidenceReviewSummary {
  const issues = researchNoteEvidenceReviewIssues(sourceState)
  const failed = issues.filter(issue => issue.status === 'failed').length
  return {
    needsReview: issues.length - failed,
    failed,
    total: issues.length,
    firstSourceIndex: issues[0]?.sourceIndex ?? null,
  }
}
