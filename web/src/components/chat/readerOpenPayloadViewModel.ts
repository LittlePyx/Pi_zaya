import {
  candidateIdentityKey,
} from './reader/readerDomUtils'
import type {
  ReaderLocateCandidate,
  ReaderOpenPayload,
} from './reader/readerTypes'
import type { ReaderDocLocatorResolution } from '../../api/references'

export type ReaderNormalizedLocateCandidate = Required<Pick<
  ReaderLocateCandidate,
  'headingPath' | 'snippet' | 'highlightSnippet' | 'anchorId' | 'blockId' | 'anchorKind' | 'anchorNumber'
>>

export interface ReaderOpenPayloadViewModel {
  activeHitLevel: string
  alternatives: ReaderNormalizedLocateCandidate[]
  evidenceAlternatives?: ReaderLocateCandidate[]
  hasStructuredLocateTarget: boolean
  initialAltIndex?: number
  locateFeedbackKey: string
  locateRequestId: number
  primaryCandidate: ReaderNormalizedLocateCandidate
  relatedBlockIds: string[]
  sourceName: string
  sourcePath: string
  strictLocate: boolean
  visibleAlternatives?: ReaderLocateCandidate[]
}

export interface ReaderActiveLocateCandidateViewModel {
  activeAlt: ReaderNormalizedLocateCandidate | null
  activeAnchorId: string
  activeAnchorKind: string
  activeAnchorNumber: number
  activeBlockId: string
  activeFocusSnippet: string
  activeHeadingPath: string
  activeHighlightSnippet: string
  expectsEquationBinding: boolean
}

export interface ReaderValidatedLocateCandidates {
  alternatives: ReaderNormalizedLocateCandidate[]
  primaryCandidate: ReaderNormalizedLocateCandidate
  recovered: boolean
  validated: boolean
}

function toTrimmedString(value: unknown): string {
  return String(value || '').trim()
}

function toAnchorNumber(value: unknown): number {
  return Number.isFinite(Number(value)) ? Math.floor(Number(value)) : 0
}

function normalizeCandidate(input: ReaderLocateCandidate): ReaderNormalizedLocateCandidate {
  return {
    anchorId: toTrimmedString(input.anchorId),
    anchorKind: toTrimmedString(input.anchorKind).toLowerCase(),
    anchorNumber: toAnchorNumber(input.anchorNumber),
    blockId: toTrimmedString(input.blockId),
    headingPath: toTrimmedString(input.headingPath),
    highlightSnippet: toTrimmedString(input.highlightSnippet),
    snippet: toTrimmedString(input.snippet),
  }
}

function hasCandidateSignal(candidate: ReaderNormalizedLocateCandidate): boolean {
  return Boolean(
    candidate.headingPath
    || candidate.snippet
    || candidate.highlightSnippet
    || candidate.anchorId
    || candidate.blockId
    || candidate.anchorKind
    || candidate.anchorNumber > 0,
  )
}

function normalizeRelatedBlockIds(value: unknown): string[] {
  return Array.isArray(value)
    ? value.map((item) => toTrimmedString(item)).filter(Boolean)
    : []
}

function rawCandidateList(value: unknown): ReaderLocateCandidate[] | undefined {
  return Array.isArray(value) ? value : undefined
}

export function buildReaderOpenPayloadViewModel(
  payload: ReaderOpenPayload | null,
): ReaderOpenPayloadViewModel {
  const sourcePath = toTrimmedString(payload?.sourcePath)
  const sourceName = toTrimmedString(payload?.sourceName)
  const headingPath = toTrimmedString(payload?.headingPath)
  const focusSnippet = toTrimmedString(payload?.snippet)
  const highlightSnippet = toTrimmedString(payload?.highlightSnippet)
  const locateTarget = (payload?.locateTarget && typeof payload.locateTarget === 'object')
    ? payload.locateTarget
    : null
  const hasStructuredLocateTarget = Boolean(locateTarget)
  const primaryCandidate = normalizeCandidate({
    anchorId: locateTarget?.anchorId || payload?.anchorId,
    anchorKind: locateTarget?.anchorKind || payload?.anchorKind,
    anchorNumber: locateTarget?.anchorNumber || payload?.anchorNumber,
    blockId: locateTarget?.blockId || payload?.blockId,
    headingPath: locateTarget?.headingPath || headingPath,
    highlightSnippet: locateTarget?.highlightSnippet || highlightSnippet || locateTarget?.snippet || focusSnippet,
    snippet: locateTarget?.snippet || focusSnippet,
  })
  const rawAlternatives = [
    ...(rawCandidateList(payload?.visibleAlternatives) || []),
    ...(rawCandidateList(payload?.evidenceAlternatives) || []),
    ...(rawCandidateList(payload?.alternatives) || []),
  ]
  const alternatives: ReaderNormalizedLocateCandidate[] = []
  const seen = new Set<string>()
  const pushCandidate = (candidate: ReaderLocateCandidate) => {
    const normalized = normalizeCandidate(candidate)
    if (!hasCandidateSignal(normalized)) return
    const key = candidateIdentityKey(normalized)
    if (seen.has(key)) return
    seen.add(key)
    alternatives.push(normalized)
  }

  pushCandidate(primaryCandidate)
  for (const item of rawAlternatives) {
    if (!item || typeof item !== 'object') continue
    pushCandidate(item)
    if (alternatives.length >= 6) break
  }

  const relatedBlockIds = Array.isArray(locateTarget?.relatedBlockIds)
    ? normalizeRelatedBlockIds(locateTarget.relatedBlockIds)
    : normalizeRelatedBlockIds(payload?.relatedBlockIds)
  return {
    activeHitLevel: toTrimmedString(locateTarget?.hitLevel).toLowerCase(),
    alternatives,
    evidenceAlternatives: rawCandidateList(payload?.evidenceAlternatives),
    hasStructuredLocateTarget,
    initialAltIndex: payload?.initialAltIndex,
    locateFeedbackKey: toTrimmedString(payload?.locateFeedbackKey),
    locateRequestId: Number.isFinite(Number(payload?.locateRequestId || 0))
      ? Math.max(0, Math.floor(Number(payload?.locateRequestId || 0)))
      : 0,
    primaryCandidate,
    relatedBlockIds,
    sourceName,
    sourcePath,
    strictLocate: Boolean(payload?.strictLocate || hasStructuredLocateTarget),
    visibleAlternatives: rawCandidateList(payload?.visibleAlternatives),
  }
}

export function buildReaderActiveLocateCandidate({
  activeAltIndex,
  alternatives,
  primaryCandidate,
}: {
  activeAltIndex: number
  alternatives: ReaderNormalizedLocateCandidate[]
  primaryCandidate: ReaderNormalizedLocateCandidate
}): ReaderActiveLocateCandidateViewModel {
  const activeAlt = alternatives[activeAltIndex] || null
  const activeHeadingPath = toTrimmedString(activeAlt?.headingPath || primaryCandidate.headingPath)
  const activeFocusSnippet = toTrimmedString(activeAlt?.snippet || primaryCandidate.snippet)
  const activeHighlightSnippet = toTrimmedString(
    activeAlt?.highlightSnippet || primaryCandidate.highlightSnippet || activeFocusSnippet,
  )
  const activeAnchorId = toTrimmedString(activeAlt?.anchorId || primaryCandidate.anchorId)
  const activeBlockId = toTrimmedString(activeAlt?.blockId || primaryCandidate.blockId)
  const activeAnchorKind = toTrimmedString(activeAlt?.anchorKind || primaryCandidate.anchorKind).toLowerCase()
  const activeAnchorNumber = Number.isFinite(Number(activeAlt?.anchorNumber || primaryCandidate.anchorNumber || 0))
    ? Math.floor(Number(activeAlt?.anchorNumber || primaryCandidate.anchorNumber || 0))
    : 0
  const expectsEquationBinding = activeAnchorKind === 'equation'
    || alternatives.some((item) => toTrimmedString(item.anchorKind).toLowerCase() === 'equation')

  return {
    activeAlt,
    activeAnchorId,
    activeAnchorKind,
    activeAnchorNumber,
    activeBlockId,
    activeFocusSnippet,
    activeHeadingPath,
    activeHighlightSnippet,
    expectsEquationBinding,
  }
}

export function applyReaderDocLocatorResolution({
  alternatives,
  primaryCandidate,
  resolution,
  strictLocate,
}: {
  alternatives: ReaderNormalizedLocateCandidate[]
  primaryCandidate: ReaderNormalizedLocateCandidate
  resolution: ReaderDocLocatorResolution | null
  strictLocate: boolean
}): ReaderValidatedLocateCandidates {
  if (!resolution) {
    return { alternatives, primaryCandidate, recovered: false, validated: false }
  }

  const locator = resolution.ok && resolution.locator && typeof resolution.locator === 'object'
    ? resolution.locator
    : null
  const resolvedPrimary = locator
    ? normalizeCandidate({
      ...primaryCandidate,
      blockId: locator.block_id || primaryCandidate.blockId,
      anchorId: locator.anchor_id || primaryCandidate.anchorId,
      anchorKind: locator.anchor_kind || primaryCandidate.anchorKind,
      headingPath: locator.heading_path || primaryCandidate.headingPath,
    })
    : strictLocate
      ? { ...primaryCandidate, blockId: '', anchorId: '' }
      : primaryCandidate

  const resolvedAlternatives: ReaderNormalizedLocateCandidate[] = []
  const seen = new Set<string>()
  for (const candidate of [resolvedPrimary, ...alternatives.slice(1)]) {
    if (!hasCandidateSignal(candidate)) continue
    const key = candidateIdentityKey(candidate)
    if (seen.has(key)) continue
    seen.add(key)
    resolvedAlternatives.push(candidate)
  }
  if (resolvedAlternatives.length <= 0 && hasCandidateSignal(resolvedPrimary)) {
    resolvedAlternatives.push(resolvedPrimary)
  }
  return {
    alternatives: resolvedAlternatives,
    primaryCandidate: resolvedPrimary,
    recovered: Boolean(resolution.ok && resolution.changed && locator),
    validated: true,
  }
}
