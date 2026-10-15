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

export const RESEARCH_NOTE_EVIDENCE_USER_DECISION_REASONS = new Set(['candidate', 'source_changed', 'location_changed'])
export const RESEARCH_NOTE_EVIDENCE_SYSTEM_RETRY_REASONS = new Set([
  'unavailable',
  'source_missing',
  'conversion_pending',
  'verification_failed',
  'source_changed_during_capture',
])
export const RESEARCH_NOTE_EVIDENCE_SYSTEM_TERMINAL_REASONS = new Set([
  'source_lost',
  'source_unreadable',
  'source_unstable',
])
const EVIDENCE_MAINTENANCE_VERSION = 3

export type ResearchNoteEvidenceResolutionMode = 'system_auto' | 'user_decision' | 'source_boundary'

export function researchNoteEvidenceResolutionMode(
  link: Pick<ResearchNoteSourceLink, 'evidence_baseline_check'>,
): ResearchNoteEvidenceResolutionMode {
  const check = link.evidence_baseline_check
  const storedMode = String(check?.resolution_mode || '')
  if (storedMode === 'user_decision' || storedMode === 'source_boundary' || storedMode === 'system_auto') {
    return storedMode
  }
  const reason = String(check?.reason || '')
  if (check?.terminal || RESEARCH_NOTE_EVIDENCE_SYSTEM_TERMINAL_REASONS.has(reason)) return 'source_boundary'
  if (RESEARCH_NOTE_EVIDENCE_USER_DECISION_REASONS.has(reason)) return 'user_decision'
  return 'system_auto'
}

function localMaintenanceLinks(sourceState: ResearchNoteSourceState | null | undefined) {
  const links = Array.isArray(sourceState?.links) ? sourceState.links : []
  return links.filter(link => (
    Boolean(link && typeof link === 'object')
    && link.kind !== 'answer'
    && Boolean(String(link.source_path || '').trim())
    && !RESEARCH_NOTE_EVIDENCE_SYSTEM_TERMINAL_REASONS.has(
      String(link.evidence_baseline_check?.reason || ''),
    )
    && !link.evidence_baseline_check?.terminal
    && (
      link.evidence_context_snapshot == null
      || (
        RESEARCH_NOTE_EVIDENCE_USER_DECISION_REASONS.has(String(link.evidence_baseline_check?.reason || ''))
        && Number(link.evidence_baseline_check?.maintenance_version || 0) < EVIDENCE_MAINTENANCE_VERSION
      )
      || link.evidence_baseline_check?.status === 'failed'
      || RESEARCH_NOTE_EVIDENCE_SYSTEM_RETRY_REASONS.has(String(link.evidence_baseline_check?.reason || ''))
    )
    && link.evidence_verification?.status !== 'invalid'
  ))
}

export function researchNoteEvidenceReviewIssues(
  sourceState: ResearchNoteSourceState | null | undefined,
): ResearchNoteEvidenceReviewIssue[] {
  const links = Array.isArray(sourceState?.links) ? sourceState.links : []
  return links.flatMap((link, sourceIndex) => {
    if (!link || typeof link !== 'object' || link.kind === 'answer') return []
    if (!String(link.source_path || '').trim()) return []
    if (link.evidence_verification?.status === 'invalid') return []
    const check = link.evidence_baseline_check
    const status = String(check?.status || '')
    const reason = String(check?.reason || '').trim()
    if (
      status !== 'skipped'
      || researchNoteEvidenceResolutionMode(link) !== 'user_decision'
      || !RESEARCH_NOTE_EVIDENCE_USER_DECISION_REASONS.has(reason)
    ) return []
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
  return localMaintenanceLinks(sourceState).some(link => {
    const check = link.evidence_baseline_check
    return check?.status === 'failed' || RESEARCH_NOTE_EVIDENCE_SYSTEM_RETRY_REASONS.has(String(check?.reason || ''))
  })
}

export function researchNoteEvidenceMaintenanceDue(
  sourceState: ResearchNoteSourceState | null | undefined,
  now = Date.now() / 1000,
): boolean {
  return localMaintenanceLinks(sourceState).some(link => {
    const check = link.evidence_baseline_check
    if (!check) return true
    const reason = String(check.reason || '')
    const maintenanceVersion = Number(check.maintenance_version || 0)
    if (RESEARCH_NOTE_EVIDENCE_USER_DECISION_REASONS.has(reason)) {
      return maintenanceVersion < EVIDENCE_MAINTENANCE_VERSION
    }
    if (check.status === 'failed' || RESEARCH_NOTE_EVIDENCE_SYSTEM_RETRY_REASONS.has(reason)) {
      const retryAfter = Number(check.retry_after || 0)
      return !Number.isFinite(retryAfter) || retryAfter <= now
    }
    return maintenanceVersion < EVIDENCE_MAINTENANCE_VERSION
  })
}

export function researchNoteEvidenceMaintenanceTracked(
  sourceState: ResearchNoteSourceState | null | undefined,
): boolean {
  return localMaintenanceLinks(sourceState).some(link => Boolean(link.evidence_baseline_check))
}

export function researchNoteEvidenceNextRetryAt(
  sourceState: ResearchNoteSourceState | null | undefined,
): number | null {
  const retryTimes = localMaintenanceLinks(sourceState).flatMap(link => {
    const check = link.evidence_baseline_check
    const reason = String(check?.reason || '')
    if (!check || (check.status !== 'failed' && !RESEARCH_NOTE_EVIDENCE_SYSTEM_RETRY_REASONS.has(reason))) return []
    const retryAfter = Number(check.retry_after || 0)
    return Number.isFinite(retryAfter) && retryAfter > 0 ? [retryAfter] : []
  })
  return retryTimes.length > 0 ? Math.min(...retryTimes) : null
}

export function researchNoteEvidenceMaintenanceRequired(
  sourceState: ResearchNoteSourceState | null | undefined,
): boolean {
  return localMaintenanceLinks(sourceState).length > 0
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
