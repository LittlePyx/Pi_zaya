import {
  candidateDisplayLabel,
  candidateIdentityKey,
  candidateVisibilityKey,
} from './reader/readerDomUtils'
import type { ReaderLocateCandidate } from './reader/readerTypes'

export type ReaderLocateCandidateRoleTone = 'neutral' | 'accent' | 'success' | 'warning' | 'danger'

export interface ReaderLocateCandidateOption {
  displayIndex: number
  targetIndex: number
  label: string
  distinctKey: string
  roleLabel?: string
  roleTone?: ReaderLocateCandidateRoleTone
}

export interface ReaderLocateCandidateViewModelInput {
  activeAltIndex: number
  altChangeSource: string
  alternatives: ReaderLocateCandidate[]
  candidatePickerExpanded: boolean
  evidenceAlternatives?: ReaderLocateCandidate[]
  initialAltIndex?: number
  locateHint: string
  requestedCandidate: ReaderLocateCandidate
  strictLocate: boolean
  title: string
  visibleAlternatives?: ReaderLocateCandidate[]
  S: Record<string, string>
}

export interface ReaderLocateCandidateViewModel {
  activeCandidateDistinctKey: string
  candidateOptions: ReaderLocateCandidateOption[]
  candidateToggleLabel: string
  evidenceAlternatives: ReaderLocateCandidate[]
  hasDistinctAlternatives: boolean
  requestedAltIndex: number
  requestedCandidateIdentity: string
  shouldAutoExpandCandidatePicker: boolean
}

interface VisibleCandidateOption {
  targetIndex: number
  label: string
  distinctKey: string
}

function normalizeEvidenceAlternatives(rawList: ReaderLocateCandidate[] | undefined): ReaderLocateCandidate[] {
  if (!Array.isArray(rawList) || rawList.length <= 0) return []
  const out: ReaderLocateCandidate[] = []
  const seen = new Set<string>()
  for (const item of rawList) {
    if (!item || typeof item !== 'object') continue
    const key = candidateIdentityKey(item)
    if (!key || seen.has(key)) continue
    seen.add(key)
    out.push({
      headingPath: String(item.headingPath || '').trim() || undefined,
      snippet: String(item.snippet || '').trim() || undefined,
      highlightSnippet: String(item.highlightSnippet || '').trim() || undefined,
      blockId: String(item.blockId || '').trim() || undefined,
      anchorId: String(item.anchorId || '').trim() || undefined,
      anchorKind: String(item.anchorKind || '').trim() || undefined,
      anchorNumber: Number.isFinite(Number(item.anchorNumber || 0))
        ? Math.floor(Number(item.anchorNumber || 0))
        : undefined,
    })
  }
  return out
}

function buildVisibleCandidateOptions({
  alternatives,
  title,
  visibleAlternatives,
}: Pick<ReaderLocateCandidateViewModelInput, 'alternatives' | 'title' | 'visibleAlternatives'>): VisibleCandidateOption[] {
  const rawList = Array.isArray(visibleAlternatives) && visibleAlternatives.length > 0
    ? visibleAlternatives
    : alternatives
  if (!Array.isArray(rawList) || rawList.length <= 0) return []
  const internalIndexByKey = new Map<string, number>()
  alternatives.forEach((item, idx) => {
    internalIndexByKey.set(candidateIdentityKey(item), idx)
  })
  const out: VisibleCandidateOption[] = []
  const seenDistinct = new Set<string>()
  for (const raw of rawList) {
    if (!raw || typeof raw !== 'object') continue
    const key = candidateIdentityKey(raw)
    const targetIndex = internalIndexByKey.get(key)
    if (!Number.isFinite(targetIndex)) continue
    const safeIndex = Number(targetIndex)
    const item = alternatives[safeIndex]
    if (!item) continue
    const distinctKey = candidateVisibilityKey(item, title) || `alt:${safeIndex + 1}`
    if (seenDistinct.has(distinctKey)) continue
    seenDistinct.add(distinctKey)
    out.push({
      targetIndex: safeIndex,
      label: candidateDisplayLabel(item, title) || `Candidate ${safeIndex + 1}`,
      distinctKey,
    })
  }
  return out
}

function describeCandidateRole({
  activeAlt,
  activeAltIndex,
  altChangeSource,
  candidate,
  evidenceCandidateIdentitySet,
  requestedAltIndex,
  requestedCandidateIdentity,
  strictLocate,
  S,
}: {
  activeAlt: ReaderLocateCandidate | null
  activeAltIndex: number
  altChangeSource: string
  candidate: ReaderLocateCandidate | null | undefined
  evidenceCandidateIdentitySet: Set<string>
  requestedAltIndex: number
  requestedCandidateIdentity: string
  strictLocate: boolean
  S: Record<string, string>
}): { roleLabel?: string; roleTone?: ReaderLocateCandidateRoleTone } {
  const identity = candidateIdentityKey(candidate)
  if (!identity) return {}
  const isActive = identity === candidateIdentityKey(activeAlt)
  if (requestedCandidateIdentity && identity === requestedCandidateIdentity) {
    return {
      roleLabel: strictLocate
        ? (S.reader_candidate_requested || 'Requested')
        : (S.reader_candidate_primary || 'Primary'),
      roleTone: 'accent',
    }
  }
  if (isActive && strictLocate && altChangeSource === 'system' && activeAltIndex !== requestedAltIndex) {
    return {
      roleLabel: S.reader_candidate_resolved || 'Resolved',
      roleTone: 'success',
    }
  }
  if (isActive && strictLocate && altChangeSource === 'manual' && activeAltIndex !== requestedAltIndex) {
    return {
      roleLabel: S.reader_candidate_manual || 'Manual',
      roleTone: 'accent',
    }
  }
  if (evidenceCandidateIdentitySet.has(identity)) {
    return {
      roleLabel: S.reader_candidate_evidence || 'Evidence',
      roleTone: 'success',
    }
  }
  return {
    roleLabel: strictLocate
      ? (S.reader_candidate_backup || 'Backup')
      : (S.reader_candidate_alt || 'Alt'),
    roleTone: 'neutral',
  }
}

export function buildReaderLocateCandidateViewModel({
  activeAltIndex,
  altChangeSource,
  alternatives,
  candidatePickerExpanded,
  evidenceAlternatives: rawEvidenceAlternatives,
  initialAltIndex,
  locateHint,
  requestedCandidate,
  strictLocate,
  title,
  visibleAlternatives,
  S,
}: ReaderLocateCandidateViewModelInput): ReaderLocateCandidateViewModel {
  const requestedCandidateIdentity = candidateIdentityKey(requestedCandidate)
  const evidenceAlternatives = normalizeEvidenceAlternatives(rawEvidenceAlternatives)
  const evidenceCandidateIdentitySet = new Set(
    evidenceAlternatives.map((item) => candidateIdentityKey(item)).filter(Boolean),
  )
  const activeAlt = alternatives[activeAltIndex] || null
  const activeCandidateDistinctKey = activeAlt
    ? candidateVisibilityKey(activeAlt, title) || candidateIdentityKey(activeAlt)
    : ''
  const hintIndex = Number(initialAltIndex || 0)
  const requestedAltIndex = Number.isFinite(hintIndex)
    ? Math.max(0, Math.min(alternatives.length - 1, Math.floor(hintIndex)))
    : 0
  const visibleCandidateOptions = buildVisibleCandidateOptions({
    alternatives,
    title,
    visibleAlternatives,
  })
  const roleInput = {
    activeAlt,
    activeAltIndex,
    altChangeSource,
    evidenceCandidateIdentitySet,
    requestedAltIndex,
    requestedCandidateIdentity,
    strictLocate,
    S,
  }
  const out = visibleCandidateOptions.map((item, displayIndex) => {
    const candidate = alternatives[item.targetIndex] || null
    const role = describeCandidateRole({
      ...roleInput,
      candidate,
    })
    return {
      displayIndex,
      targetIndex: item.targetIndex,
      label: item.label,
      distinctKey: item.distinctKey,
      roleLabel: role.roleLabel,
      roleTone: role.roleTone,
    }
  })

  const activeOptionExists = out.some((item) => item.distinctKey === activeCandidateDistinctKey)
  const candidateOptions = activeOptionExists || !activeAlt || !activeCandidateDistinctKey
    ? out
    : [
        ...out,
        {
          displayIndex: out.length,
          targetIndex: activeAltIndex,
          label: candidateDisplayLabel(activeAlt, title) || `Candidate ${activeAltIndex + 1}`,
          distinctKey: activeCandidateDistinctKey,
          ...describeCandidateRole({
            ...roleInput,
            candidate: activeAlt,
          }),
        },
      ]
  const distinct = new Set(candidateOptions.map((item) => item.distinctKey).filter(Boolean))
  const hasDistinctAlternatives = candidateOptions.length > 1 && distinct.size > 1
  const shouldAutoExpandCandidatePicker = hasDistinctAlternatives && (
    (altChangeSource === 'system' && activeAltIndex > requestedAltIndex)
    || /\b(not found|fallback|strict locate|neighbor evidence|was not found)\b/i.test(String(locateHint || ''))
  )
  const activeDisplayIndex = Math.max(
    1,
    candidateOptions.findIndex((item) => item.distinctKey === activeCandidateDistinctKey) + 1,
  )
  const candidateToggleLabel = hasDistinctAlternatives
    ? (candidatePickerExpanded
      ? (S.reader_hide_list || 'Hide list')
      : activeAltIndex > 0
        ? (S.reader_alt_index || 'Alt {i}/{n}')
          .replace('{i}', String(activeDisplayIndex))
          .replace('{n}', String(candidateOptions.length))
        : (S.reader_candidates_count || '{n} candidates').replace('{n}', String(candidateOptions.length)))
    : ''

  return {
    activeCandidateDistinctKey,
    candidateOptions,
    candidateToggleLabel,
    evidenceAlternatives,
    hasDistinctAlternatives,
    requestedAltIndex,
    requestedCandidateIdentity,
    shouldAutoExpandCandidatePicker,
  }
}
