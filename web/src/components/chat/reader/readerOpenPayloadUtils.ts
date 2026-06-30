import type { ReaderLocateCandidate, ReaderLocateTarget, ReaderOpenPayload } from './readerTypes'
import { basenameFromSourcePath } from '../../../utils/sourcePath'

const DEFAULT_READER_CANDIDATE_LIMIT = 6

export function inferReaderSourceNameFromPath(sourcePath: string, fallback = 'paper'): string {
  const leaf = basenameFromSourcePath(sourcePath)
  return String(leaf || fallback).trim() || fallback
}

function positiveIntegerOrUndefined(input: unknown): number | undefined {
  const value = Number(input || 0)
  return Number.isFinite(value) && value > 0 ? Math.floor(value) : undefined
}

function trimCompactText(input: unknown, maxChars = 2200): string {
  const text = String(input || '').replace(/\s+/g, ' ').trim()
  if (!text) return ''
  return text.length > maxChars ? `${text.slice(0, maxChars).trimEnd()}...` : text
}

function sanitizeStringArray(input: unknown, maxItems = 8, maxChars = 360): string[] {
  const values = Array.isArray(input) ? input : (input === undefined || input === null ? [] : [input])
  const out: string[] = []
  const seen = new Set<string>()
  for (const value of values) {
    if (out.length >= maxItems) break
    const text = trimCompactText(value, maxChars)
    if (!text) continue
    const key = text.toLowerCase()
    if (seen.has(key)) continue
    seen.add(key)
    out.push(text)
  }
  return out
}

export function readerLocateCandidateIdentityKey(
  candidate: Partial<ReaderLocateCandidate> | null | undefined,
): string {
  if (!candidate) return ''
  const hasLocationSignal = Boolean(
    String(candidate.headingPath || '').trim()
    || String(candidate.snippet || '').trim()
    || String(candidate.highlightSnippet || '').trim()
    || String(candidate.anchorId || '').trim()
    || String(candidate.blockId || '').trim()
    || positiveIntegerOrUndefined(candidate.anchorNumber),
  )
  if (!hasLocationSignal) return ''
  return [
    String(candidate.blockId || '').trim().toLowerCase(),
    String(candidate.anchorId || '').trim().toLowerCase(),
    String(candidate.anchorKind || '').trim().toLowerCase(),
    positiveIntegerOrUndefined(candidate.anchorNumber) || 0,
    trimCompactText(candidate.headingPath, 360).toLowerCase(),
    trimCompactText(candidate.highlightSnippet, 720).toLowerCase().slice(0, 180),
    trimCompactText(candidate.snippet, 720).toLowerCase().slice(0, 180),
  ].join('::')
}

function normalizeReaderLocateCandidate(input: unknown): ReaderLocateCandidate | null {
  if (!input || typeof input !== 'object') return null
  const raw = input as Record<string, unknown>
  const candidate: ReaderLocateCandidate = {
    headingPath: trimCompactText(raw.headingPath, 360) || undefined,
    snippet: trimCompactText(raw.snippet, 2200) || undefined,
    highlightSnippet: trimCompactText(raw.highlightSnippet, 2200) || undefined,
    anchorId: trimCompactText(raw.anchorId, 180) || undefined,
    blockId: trimCompactText(raw.blockId, 180) || undefined,
    anchorKind: trimCompactText(raw.anchorKind, 80).toLowerCase() || undefined,
    anchorNumber: positiveIntegerOrUndefined(raw.anchorNumber),
  }
  return readerLocateCandidateIdentityKey(candidate) ? candidate : null
}

export function sanitizeReaderLocateCandidates(
  input: unknown,
  opts?: {
    maxItems?: number
    exclude?: Array<Partial<ReaderLocateCandidate> | null | undefined> | Partial<ReaderLocateCandidate> | null
  },
): ReaderLocateCandidate[] {
  const maxItemsRaw = Number(opts?.maxItems || DEFAULT_READER_CANDIDATE_LIMIT)
  const maxItems = Number.isFinite(maxItemsRaw) && maxItemsRaw > 0
    ? Math.floor(maxItemsRaw)
    : DEFAULT_READER_CANDIDATE_LIMIT
  const excluded = Array.isArray(opts?.exclude)
    ? opts.exclude
    : (opts?.exclude ? [opts.exclude] : [])
  const excludedKeys = new Set(
    excluded
      .map((item) => readerLocateCandidateIdentityKey(item))
      .filter(Boolean),
  )
  const values = Array.isArray(input) ? input : (input === undefined || input === null ? [] : [input])
  const out: ReaderLocateCandidate[] = []
  const seen = new Set<string>()
  for (const value of values) {
    if (out.length >= maxItems) break
    const candidate = normalizeReaderLocateCandidate(value)
    const key = readerLocateCandidateIdentityKey(candidate)
    if (!candidate || !key || excludedKeys.has(key) || seen.has(key)) continue
    seen.add(key)
    out.push(candidate)
  }
  return out
}

export function sanitizeReaderLocateTarget(input: unknown): ReaderLocateTarget | undefined {
  if (!input || typeof input !== 'object') return undefined
  const raw = input as Record<string, unknown>
  const target: ReaderLocateTarget = {
    segmentId: trimCompactText(raw.segmentId, 180) || undefined,
    sourceSegmentId: trimCompactText(raw.sourceSegmentId, 180) || undefined,
    headingPath: trimCompactText(raw.headingPath, 360) || undefined,
    snippet: trimCompactText(raw.snippet, 2200) || undefined,
    highlightSnippet: trimCompactText(raw.highlightSnippet, 2200) || undefined,
    evidenceQuote: trimCompactText(raw.evidenceQuote, 2200) || undefined,
    anchorText: trimCompactText(raw.anchorText, 2200) || undefined,
    hitLevel: trimCompactText(raw.hitLevel, 80).toLowerCase() || undefined,
    blockId: trimCompactText(raw.blockId, 180) || undefined,
    anchorId: trimCompactText(raw.anchorId, 180) || undefined,
    anchorKind: trimCompactText(raw.anchorKind, 80).toLowerCase() || undefined,
    anchorNumber: positiveIntegerOrUndefined(raw.anchorNumber),
    claimType: trimCompactText(raw.claimType, 120) || undefined,
    locatePolicy: trimCompactText(raw.locatePolicy, 80) || undefined,
    locateSurfacePolicy: trimCompactText(raw.locateSurfacePolicy, 80) || undefined,
    snippetAliases: sanitizeStringArray(raw.snippetAliases, 8, 360),
    relatedBlockIds: sanitizeStringArray(raw.relatedBlockIds, 8, 180),
  }
  if (target.snippetAliases?.length === 0) delete target.snippetAliases
  if (target.relatedBlockIds?.length === 0) delete target.relatedBlockIds
  if (!Object.values(target).some((value) => Array.isArray(value) ? value.length > 0 : Boolean(value))) {
    return undefined
  }
  return target
}

export function buildBasicReaderOpenPayload(input: {
  sourcePath?: string
  sourceName?: string
  headingPath?: string
  snippet?: string
  highlightSnippet?: string
  anchorId?: string
  blockId?: string
  relatedBlockIds?: string[]
  anchorKind?: string
  anchorNumber?: number
  strictLocate?: boolean
  locateTarget?: ReaderLocateTarget | null
  alternatives?: ReaderLocateCandidate[]
  visibleAlternatives?: ReaderLocateCandidate[]
  evidenceAlternatives?: ReaderLocateCandidate[]
  initialAltIndex?: number
  fallbackSourceName?: string
  locateFeedbackKey?: string
}): ReaderOpenPayload | null {
  const sourcePath = String(input?.sourcePath || '').trim()
  if (!sourcePath) return null
  const fallbackSourceName = String(input?.fallbackSourceName || 'paper').trim() || 'paper'
  const snippet = String(input?.snippet || '').trim()
  const highlightSnippet = String(input?.highlightSnippet || snippet).trim()
  const anchorNumber = positiveIntegerOrUndefined(input?.anchorNumber)
  const primaryCandidate = normalizeReaderLocateCandidate({
    headingPath: input?.headingPath,
    snippet,
    highlightSnippet,
    anchorId: input?.anchorId,
    blockId: input?.blockId,
    anchorKind: input?.anchorKind,
    anchorNumber,
  })
  const alternatives = sanitizeReaderLocateCandidates(input?.alternatives, { exclude: primaryCandidate })
  const visibleCandidates = sanitizeReaderLocateCandidates(input?.visibleAlternatives)
  const evidenceCandidates = sanitizeReaderLocateCandidates(input?.evidenceAlternatives)
  const visibleAlternatives = visibleCandidates.length > 0
    ? sanitizeReaderLocateCandidates([primaryCandidate, ...visibleCandidates])
    : []
  const evidenceAlternatives = evidenceCandidates.length > 0
    ? sanitizeReaderLocateCandidates([primaryCandidate, ...evidenceCandidates])
    : []
  const initialAltIndexRaw = Number(input?.initialAltIndex)
  const initialAltCandidateCount = evidenceAlternatives.length || visibleAlternatives.length || alternatives.length
  const initialAltIndex = Number.isFinite(initialAltIndexRaw)
    ? Math.min(
      Math.max(0, Math.floor(initialAltIndexRaw)),
      Math.max(0, initialAltCandidateCount - 1),
    )
    : undefined
  const relatedBlockIds = sanitizeStringArray(input?.relatedBlockIds, 8, 180)
  return {
    sourcePath,
    sourceName: String(input?.sourceName || '').trim() || inferReaderSourceNameFromPath(sourcePath, fallbackSourceName),
    headingPath: String(input?.headingPath || '').trim() || undefined,
    snippet: snippet || undefined,
    highlightSnippet: highlightSnippet || undefined,
    anchorId: String(input?.anchorId || '').trim() || undefined,
    blockId: String(input?.blockId || '').trim() || undefined,
    relatedBlockIds: relatedBlockIds.length > 0 ? relatedBlockIds : undefined,
    anchorKind: String(input?.anchorKind || '').trim().toLowerCase() || undefined,
    anchorNumber,
    strictLocate: Boolean(input?.strictLocate),
    locateTarget: sanitizeReaderLocateTarget(input?.locateTarget),
    alternatives: alternatives.length > 0 ? alternatives : undefined,
    visibleAlternatives: visibleAlternatives.length > 1 ? visibleAlternatives : undefined,
    evidenceAlternatives: evidenceAlternatives.length > 1 ? evidenceAlternatives : undefined,
    initialAltIndex,
    locateFeedbackKey: String(input?.locateFeedbackKey || '').trim() || undefined,
  }
}
