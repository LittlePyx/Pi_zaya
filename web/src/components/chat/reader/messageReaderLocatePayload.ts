import type { ReaderLocateCandidate, ReaderLocateClaimGroup, ReaderLocateTarget, ReaderOpenPayload } from './readerTypes'
import { coerceStringArray, stripMarkdownInline, stripProvenanceNoise, type LocateCandidate } from './messageLocateCandidates'

export interface ReaderOpenStructuredEntry {
  segmentId: string
  label: string
  segmentText: string
  evidenceQuote: string
  locateTarget?: ReaderLocateTarget
  readerOpen?: ReaderOpenPayload
  hitLevel?: string
  claimType?: string
  mustLocate?: boolean
  locatePolicy?: string
  locateSurfacePolicy?: string
  claimGroupId?: string
  claimGroupKind?: string
  formulaOrigin?: string
  anchorKind?: string
  anchorText?: string
  equationNumber?: number
  supportFigureNumber?: number
  supportPanelLetters?: string[]
  snippetKey: string
  snippetAliases: string[]
  primary: LocateCandidate
  alternatives: LocateCandidate[]
  relatedBlockIds?: string[]
  sourceSegmentId?: string
  groupLeadText?: string
  groupDistance?: number
}

export function coerceReaderLocateTarget(input: unknown): ReaderLocateTarget | null {
  if (!input || typeof input !== 'object') return null
  const raw = input as Record<string, unknown>
  const anchorNumberRaw = Number(raw.anchorNumber || 0)
  const target: ReaderLocateTarget = {
    segmentId: String(raw.segmentId || '').trim() || undefined,
    sourceSegmentId: String(raw.sourceSegmentId || '').trim() || undefined,
    headingPath: String(raw.headingPath || '').trim() || undefined,
    snippet: String(raw.snippet || '').trim() || undefined,
    highlightSnippet: String(raw.highlightSnippet || '').trim() || undefined,
    evidenceQuote: String(raw.evidenceQuote || '').trim() || undefined,
    anchorText: String(raw.anchorText || '').trim() || undefined,
    hitLevel: String(raw.hitLevel || '').trim().toLowerCase() || undefined,
    blockId: String(raw.blockId || '').trim() || undefined,
    anchorId: String(raw.anchorId || '').trim() || undefined,
    anchorKind: String(raw.anchorKind || '').trim().toLowerCase() || undefined,
    anchorNumber: Number.isFinite(anchorNumberRaw) && anchorNumberRaw > 0
      ? Math.floor(anchorNumberRaw)
      : undefined,
    claimType: String(raw.claimType || '').trim() || undefined,
    locatePolicy: String(raw.locatePolicy || '').trim() || undefined,
    locateSurfacePolicy: String(raw.locateSurfacePolicy || '').trim() || undefined,
    snippetAliases: coerceStringArray(raw.snippetAliases, 8, 360),
    relatedBlockIds: coerceStringArray(raw.relatedBlockIds, 8, 180),
  }
  if (!Object.values(target).some((value) => Array.isArray(value) ? value.length > 0 : Boolean(value))) {
    return null
  }
  return target
}

function coerceReaderLocateClaimGroup(input: unknown): ReaderLocateClaimGroup | null {
  if (!input || typeof input !== 'object') return null
  const raw = input as Record<string, unknown>
  const distanceRaw = Number(raw.distance || 0)
  const claimGroup: ReaderLocateClaimGroup = {
    id: String(raw.id || '').trim() || undefined,
    kind: String(raw.kind || '').trim() || undefined,
    leadText: String(raw.leadText || '').trim() || undefined,
    distance: Number.isFinite(distanceRaw) && distanceRaw > 0
      ? Math.floor(distanceRaw)
      : undefined,
  }
  if (!Object.values(claimGroup).some(Boolean)) return null
  return claimGroup
}

function coerceReaderLocateCandidateArray(input: unknown, maxItems = 6): ReaderLocateCandidate[] {
  if (!Array.isArray(input) || input.length <= 0) return []
  const out: ReaderLocateCandidate[] = []
  const seen = new Set<string>()
  for (const item of input) {
    if (!item || typeof item !== 'object') continue
    const raw = item as Record<string, unknown>
    const anchorNumberRaw = Number(raw.anchorNumber || 0)
    const candidate: ReaderLocateCandidate = {
      headingPath: String(raw.headingPath || '').trim() || undefined,
      snippet: String(raw.snippet || '').trim() || undefined,
      highlightSnippet: String(raw.highlightSnippet || '').trim() || undefined,
      anchorId: String(raw.anchorId || '').trim() || undefined,
      blockId: String(raw.blockId || '').trim() || undefined,
      anchorKind: String(raw.anchorKind || '').trim().toLowerCase() || undefined,
      anchorNumber: Number.isFinite(anchorNumberRaw) && anchorNumberRaw > 0
        ? Math.floor(anchorNumberRaw)
        : undefined,
    }
    const key = [
      String(candidate.blockId || '').trim().toLowerCase(),
      String(candidate.anchorId || '').trim().toLowerCase(),
      String(candidate.anchorKind || '').trim().toLowerCase(),
      Number.isFinite(Number(candidate.anchorNumber || 0)) ? Math.floor(Number(candidate.anchorNumber || 0)) : 0,
      String(candidate.headingPath || '').trim().toLowerCase(),
      String(candidate.highlightSnippet || '').trim().toLowerCase().slice(0, 180),
      String(candidate.snippet || '').trim().toLowerCase().slice(0, 180),
    ].join('::')
    if (seen.has(key)) continue
    seen.add(key)
    out.push(candidate)
    if (out.length >= maxItems) break
  }
  return out
}

function readerLocateCandidateIdentityKey(item: Partial<ReaderLocateCandidate> | null | undefined): string {
  return [
    String(item?.blockId || '').trim().toLowerCase(),
    String(item?.anchorId || '').trim().toLowerCase(),
    String(item?.anchorKind || '').trim().toLowerCase(),
    Number.isFinite(Number(item?.anchorNumber || 0)) ? Math.floor(Number(item?.anchorNumber || 0)) : 0,
    String(item?.headingPath || '').trim().toLowerCase(),
    String(item?.highlightSnippet || '').trim().toLowerCase().slice(0, 180),
    String(item?.snippet || '').trim().toLowerCase().slice(0, 180),
  ].join('::')
}

export function toPositiveIntOrUndefined(value: unknown): number | undefined {
  const n = Number(value || 0)
  return Number.isFinite(n) && n > 0 ? Math.floor(n) : undefined
}

function dedupeReaderLocateCandidates(
  candidates: Array<ReaderLocateCandidate | null | undefined>,
  maxItems = 6,
): ReaderLocateCandidate[] {
  const out: ReaderLocateCandidate[] = []
  const seen = new Set<string>()
  for (const item of candidates) {
    if (!item || typeof item !== 'object') continue
    const key = readerLocateCandidateIdentityKey(item)
    if (!key || seen.has(key)) continue
    seen.add(key)
    out.push(item)
    if (out.length >= maxItems) break
  }
  return out
}

function buildReaderLocateCandidateFromLocateCandidate(
  cand: LocateCandidate | null | undefined,
  opts: {
    snippet: string
    highlightSnippet: string
    anchorKind?: string
    anchorNumber?: number
  },
): ReaderLocateCandidate | null {
  if (!cand) return null
  const snippet = String(opts.snippet || cand.focusSnippet || '').trim()
  const highlightSnippet = String(opts.highlightSnippet || snippet || cand.focusSnippet || '').trim()
  const anchorNumber = toPositiveIntOrUndefined(opts.anchorNumber || cand.anchorNumber || 0)
  const candidate: ReaderLocateCandidate = {
    headingPath: String(cand.headingPath || '').trim() || undefined,
    snippet: snippet || undefined,
    highlightSnippet: highlightSnippet || undefined,
    blockId: String(cand.blockId || '').trim() || undefined,
    anchorId: String(cand.anchorId || '').trim() || undefined,
    anchorKind: String(opts.anchorKind || cand.anchorKind || '').trim().toLowerCase() || undefined,
    anchorNumber,
  }
  return readerLocateCandidateIdentityKey(candidate) ? candidate : null
}

function buildReaderCandidateCollections(
  primaryCandidate: ReaderLocateCandidate | null,
  secondaryCandidates: Array<ReaderLocateCandidate | null | undefined>,
  opts?: {
    visibleCandidates?: Array<ReaderLocateCandidate | null | undefined>
    evidenceCandidates?: Array<ReaderLocateCandidate | null | undefined>
    maxItems?: number
  },
): Pick<ReaderOpenPayload, 'alternatives' | 'visibleAlternatives' | 'evidenceAlternatives'> {
  const maxItems = Number.isFinite(Number(opts?.maxItems || 6))
    ? Math.max(1, Math.floor(Number(opts?.maxItems || 6)))
    : 6
  const alternatives = dedupeReaderLocateCandidates(secondaryCandidates, maxItems)
  const visibleAlternatives = dedupeReaderLocateCandidates(
    [
      primaryCandidate,
      ...((opts?.visibleCandidates && opts.visibleCandidates.length > 0)
        ? opts.visibleCandidates
        : alternatives),
    ],
    maxItems,
  )
  const evidenceAlternatives = dedupeReaderLocateCandidates(
    [
      primaryCandidate,
      ...((opts?.evidenceCandidates && opts.evidenceCandidates.length > 0)
        ? opts.evidenceCandidates
        : alternatives),
    ],
    maxItems,
  )
  return {
    alternatives: alternatives.length > 0 ? alternatives : undefined,
    visibleAlternatives: visibleAlternatives.length > 1 ? visibleAlternatives : undefined,
    evidenceAlternatives: evidenceAlternatives.length > 1 ? evidenceAlternatives : undefined,
  }
}

export function coerceReaderOpenPayload(input: unknown): ReaderOpenPayload | null {
  if (!input || typeof input !== 'object') return null
  const raw = input as Record<string, unknown>
  const anchorNumberRaw = Number(raw.anchorNumber || 0)
  const initialAltIndexRaw = Number(raw.initialAltIndex || 0)
  const locateTarget = coerceReaderLocateTarget(raw.locateTarget)
  const claimGroup = coerceReaderLocateClaimGroup(raw.claimGroup)
  const payload: ReaderOpenPayload = {
    sourcePath: String(raw.sourcePath || '').trim(),
    sourceName: String(raw.sourceName || '').trim() || undefined,
    headingPath: String(raw.headingPath || '').trim() || undefined,
    snippet: String(raw.snippet || '').trim() || undefined,
    highlightSnippet: String(raw.highlightSnippet || '').trim() || undefined,
    anchorId: String(raw.anchorId || '').trim() || undefined,
    blockId: String(raw.blockId || '').trim() || undefined,
    relatedBlockIds: coerceStringArray(raw.relatedBlockIds, 8, 180),
    anchorKind: String(raw.anchorKind || '').trim().toLowerCase() || undefined,
    anchorNumber: Number.isFinite(anchorNumberRaw) && anchorNumberRaw > 0
      ? Math.floor(anchorNumberRaw)
      : undefined,
    strictLocate: raw.strictLocate === undefined ? undefined : Boolean(raw.strictLocate),
    locateTarget: locateTarget || undefined,
    claimGroup: claimGroup || undefined,
    alternatives: coerceReaderLocateCandidateArray(raw.alternatives, 6),
    visibleAlternatives: coerceReaderLocateCandidateArray(raw.visibleAlternatives, 6),
    evidenceAlternatives: coerceReaderLocateCandidateArray(raw.evidenceAlternatives, 6),
    initialAltIndex: Number.isFinite(initialAltIndexRaw)
      ? Math.max(0, Math.floor(initialAltIndexRaw))
      : undefined,
  }
  if (!Object.values(payload).some((value) => Array.isArray(value) ? value.length > 0 : Boolean(value))) {
    return null
  }
  return payload
}

export function buildLocateCandidateFromReaderLocateCandidate(
  raw: Partial<ReaderLocateCandidate> | null | undefined,
  opts: {
    sourcePath: string
    sourceName: string
    sourceType?: 'guide' | 'refs'
  },
): LocateCandidate | null {
  if (!raw) return null
  const sourcePath = String(opts.sourcePath || '').trim()
  if (!sourcePath) return null
  const snippet = String(raw.snippet || raw.highlightSnippet || '').trim()
  const highlightSnippet = String(raw.highlightSnippet || snippet).trim()
  const headingPath = String(raw.headingPath || '').trim()
  const blockId = String(raw.blockId || '').trim()
  const anchorId = String(raw.anchorId || '').trim()
  const anchorKind = String(raw.anchorKind || '').trim().toLowerCase()
  const anchorNumber = toPositiveIntOrUndefined(raw.anchorNumber || 0)
  const focusSnippet = String(highlightSnippet || snippet || headingPath).trim()
  if (!(focusSnippet || blockId || anchorId || headingPath)) return null
  return {
    sourcePath,
    sourceName: String(opts.sourceName || '').trim(),
    headingPath,
    focusSnippet,
    matchText: [headingPath, snippet || highlightSnippet].filter(Boolean).join('\n') || focusSnippet,
    sourceType: opts.sourceType || 'guide',
    blockId: blockId || undefined,
    anchorId: anchorId || undefined,
    anchorKind: anchorKind || undefined,
    anchorNumber,
  }
}

export function buildStructuredEntryReaderOpenPayload(
  entry: ReaderOpenStructuredEntry,
  fallbackSnippet: string,
): ReaderOpenPayload | null {
  const primary = entry.primary
  if (!primary) return null
  const baseReaderOpen = entry.readerOpen || null
  const baseLocateTarget = entry.locateTarget || baseReaderOpen?.locateTarget || null
  const sourcePath = String(baseReaderOpen?.sourcePath || primary.sourcePath || '').trim()
  if (!sourcePath) return null
  const queryRaw = stripProvenanceNoise(
    stripMarkdownInline(String(entry.evidenceQuote || fallbackSnippet || entry.segmentText || entry.label || '')),
  ).trim()
  const structuredSnippet = String(
    queryRaw
    || baseReaderOpen?.snippet
    || baseLocateTarget?.snippet
    || primary.focusSnippet
    || fallbackSnippet,
  ).trim()
  const structuredHighlight = String(
    entry.evidenceQuote
    || baseReaderOpen?.highlightSnippet
    || baseLocateTarget?.highlightSnippet
    || queryRaw
    || primary.focusSnippet
    || fallbackSnippet,
  ).trim()
  const structuredAnchorKind = String(
    entry.anchorKind
    || baseReaderOpen?.anchorKind
    || primary.anchorKind
    || '',
  ).trim()
  const structuredAnchorNumber = toPositiveIntOrUndefined(
    entry.equationNumber
    || entry.supportFigureNumber
    || baseReaderOpen?.anchorNumber
    || primary.anchorNumber
    || 0,
  )
  const groupDistance = toPositiveIntOrUndefined(entry.groupDistance || 0)
  const baseClaimGroup = baseReaderOpen?.claimGroup || null
  const locateTarget: ReaderLocateTarget = {
    ...baseLocateTarget,
    segmentId: String(entry.segmentId || baseLocateTarget?.segmentId || '').trim() || undefined,
    sourceSegmentId: String(entry.sourceSegmentId || baseLocateTarget?.sourceSegmentId || '').trim() || undefined,
    headingPath: String(primary.headingPath || baseReaderOpen?.headingPath || baseLocateTarget?.headingPath || '').trim() || undefined,
    snippet: structuredSnippet || undefined,
    highlightSnippet: structuredHighlight || undefined,
    evidenceQuote: String(entry.evidenceQuote || '').trim() || undefined,
    anchorText: String(entry.anchorText || '').trim() || undefined,
    hitLevel: String(entry.hitLevel || '').trim() || undefined,
    blockId: String(primary.blockId || '').trim() || undefined,
    anchorId: String(primary.anchorId || '').trim() || undefined,
    anchorKind: structuredAnchorKind || undefined,
    anchorNumber: structuredAnchorNumber,
    claimType: String(entry.claimType || '').trim() || undefined,
    locatePolicy: String(entry.locatePolicy || '').trim() || undefined,
    locateSurfacePolicy: String(entry.locateSurfacePolicy || '').trim() || undefined,
    snippetAliases: Array.isArray(entry.snippetAliases)
      ? entry.snippetAliases.map((item) => String(item || '').trim()).filter(Boolean)
      : undefined,
    relatedBlockIds: Array.isArray(entry.relatedBlockIds)
      ? entry.relatedBlockIds.map((item) => String(item || '').trim()).filter(Boolean)
      : undefined,
  }
  const claimGroup: ReaderLocateClaimGroup | undefined = (
    entry.claimGroupId
    || entry.claimGroupKind
    || baseClaimGroup?.id
    || baseClaimGroup?.kind
    || entry.groupLeadText
    || baseClaimGroup?.leadText
    || groupDistance
    || toPositiveIntOrUndefined(baseClaimGroup?.distance || 0)
  )
    ? {
      id: String(entry.claimGroupId || baseClaimGroup?.id || '').trim() || undefined,
      kind: String(entry.claimGroupKind || baseClaimGroup?.kind || '').trim() || undefined,
      leadText: String(entry.groupLeadText || baseClaimGroup?.leadText || '').trim() || undefined,
      distance: groupDistance || toPositiveIntOrUndefined(baseClaimGroup?.distance || 0),
    }
    : undefined
  const primaryReaderCandidate = buildReaderLocateCandidateFromLocateCandidate(primary, {
    snippet: structuredSnippet,
    highlightSnippet: structuredHighlight,
    anchorKind: structuredAnchorKind,
    anchorNumber: structuredAnchorNumber,
  })
  const primaryCandidateIdentity = readerLocateCandidateIdentityKey(primaryReaderCandidate)
  const filterWithoutPrimary = (items: ReaderLocateCandidate[] | undefined): ReaderLocateCandidate[] => {
    return dedupeReaderLocateCandidates(
      (items || []).filter((item) => readerLocateCandidateIdentityKey(item) !== primaryCandidateIdentity),
      6,
    )
  }
  const backendAlternatives = filterWithoutPrimary(baseReaderOpen?.alternatives)
  const backendVisibleAlternatives = filterWithoutPrimary(baseReaderOpen?.visibleAlternatives)
  const backendEvidenceAlternatives = filterWithoutPrimary(baseReaderOpen?.evidenceAlternatives)
  const fallbackAlternatives = dedupeReaderLocateCandidates(
    (entry.alternatives || [])
      .filter((item) => Boolean(item) && item !== primary)
      .map((item) => buildReaderLocateCandidateFromLocateCandidate(item, {
        snippet: String(item.focusSnippet || structuredSnippet).trim(),
        highlightSnippet: structuredHighlight || String(item.focusSnippet || structuredSnippet).trim(),
        anchorKind: String(item.anchorKind || structuredAnchorKind).trim(),
        anchorNumber: toPositiveIntOrUndefined(item.anchorNumber || structuredAnchorNumber || 0),
      })),
    6,
  )
  const openAlternatives = backendAlternatives.length > 0 ? backendAlternatives : fallbackAlternatives
  const candidateCollections = buildReaderCandidateCollections(
    primaryReaderCandidate,
    openAlternatives,
    {
      visibleCandidates: backendVisibleAlternatives.length > 0 ? backendVisibleAlternatives : undefined,
      evidenceCandidates: backendEvidenceAlternatives.length > 0 ? backendEvidenceAlternatives : undefined,
    },
  )
  return {
    sourcePath,
    sourceName: String(baseReaderOpen?.sourceName || primary.sourceName || '').trim() || undefined,
    headingPath: String(primary.headingPath || baseReaderOpen?.headingPath || '').trim() || undefined,
    snippet: structuredSnippet || undefined,
    highlightSnippet: structuredHighlight || undefined,
    blockId: String(primary.blockId || '').trim() || undefined,
    anchorId: String(primary.anchorId || '').trim() || undefined,
    relatedBlockIds: locateTarget.relatedBlockIds || baseReaderOpen?.relatedBlockIds,
    anchorKind: structuredAnchorKind || undefined,
    anchorNumber: structuredAnchorNumber,
    strictLocate: baseReaderOpen?.strictLocate ?? true,
    locateTarget,
    claimGroup,
    ...candidateCollections,
    initialAltIndex: Number.isFinite(Number(baseReaderOpen?.initialAltIndex))
      ? Math.max(0, Math.floor(Number(baseReaderOpen?.initialAltIndex)))
      : 0,
  }
}

export function buildHeuristicReaderOpenPayload(
  pickedList: LocateCandidate[],
  snippet: string,
  opts?: { strictLocate?: boolean; highlightSnippet?: string; relatedBlockIds?: string[] },
): ReaderOpenPayload | null {
  const picked = pickedList[0] || null
  if (!picked) return null
  const sourcePath = String(picked.sourcePath || '').trim()
  if (!sourcePath) return null
  const highlightSnippet = String(opts?.highlightSnippet || snippet).trim()
  const primarySnippet = String(picked.focusSnippet || snippet).trim()
  const primaryHighlight = String(highlightSnippet || picked.focusSnippet || snippet).trim()
  const primaryCandidate = buildReaderLocateCandidateFromLocateCandidate(picked, {
    snippet: primarySnippet,
    highlightSnippet: primaryHighlight,
    anchorKind: picked.anchorKind,
    anchorNumber: picked.anchorNumber,
  })
  const secondaryCandidates = pickedList.slice(1).map((item) => buildReaderLocateCandidateFromLocateCandidate(item, {
    snippet: String(item.focusSnippet || snippet).trim(),
    highlightSnippet: String(highlightSnippet || item.focusSnippet || snippet).trim(),
    anchorKind: item.anchorKind,
    anchorNumber: item.anchorNumber,
  }))
  const candidateCollections = buildReaderCandidateCollections(primaryCandidate, secondaryCandidates)
  return {
    sourcePath,
    sourceName: String(picked.sourceName || '').trim() || undefined,
    headingPath: String(picked.headingPath || '').trim() || undefined,
    snippet: primarySnippet || undefined,
    highlightSnippet: primaryHighlight || undefined,
    blockId: String(picked.blockId || '').trim() || undefined,
    anchorId: String(picked.anchorId || '').trim() || undefined,
    anchorKind: String(picked.anchorKind || '').trim() || undefined,
    anchorNumber: toPositiveIntOrUndefined(picked.anchorNumber || 0),
    strictLocate: Boolean(opts?.strictLocate),
    locateMode: 'heuristic',
    relatedBlockIds: Array.isArray(opts?.relatedBlockIds)
      ? opts.relatedBlockIds.map((item) => String(item || '').trim()).filter(Boolean)
      : undefined,
    ...candidateCollections,
    initialAltIndex: 0,
  }
}
