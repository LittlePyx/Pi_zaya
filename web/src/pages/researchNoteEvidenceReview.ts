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

export function researchNoteEvidenceReviewIssues(
  sourceState: ResearchNoteSourceState | null | undefined,
): ResearchNoteEvidenceReviewIssue[] {
  const links = Array.isArray(sourceState?.links) ? sourceState.links : []
  return links.flatMap((link, sourceIndex) => {
    if (!link || typeof link !== 'object' || link.kind === 'answer') return []
    if (link.evidence_context_snapshot != null || link.evidence_verification?.status === 'invalid') return []
    const check = link.evidence_baseline_check
    const status = String(check?.status || '')
    const reason = String(check?.reason || '').trim()
    if (!reason || (status !== 'skipped' && status !== 'failed')) return []
    return [{
      sourceIndex,
      status,
      reason,
      checkedAt: check?.checked_at,
      link,
    }]
  })
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
