import type { ReaderLocateCandidate, ReaderLocateTarget, ReaderOpenPayload } from './readerTypes'

export function inferReaderSourceNameFromPath(sourcePath: string, fallback = 'paper'): string {
  const raw = String(sourcePath || '').trim()
  if (!raw) return fallback
  const leaf = raw.split(/[\\/]/).pop() || ''
  return String(leaf || fallback).trim() || fallback
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
}): ReaderOpenPayload | null {
  const sourcePath = String(input?.sourcePath || '').trim()
  if (!sourcePath) return null
  const fallbackSourceName = String(input?.fallbackSourceName || 'paper').trim() || 'paper'
  const snippet = String(input?.snippet || '').trim()
  const highlightSnippet = String(input?.highlightSnippet || snippet).trim()
  const relatedBlockIds = Array.isArray(input?.relatedBlockIds)
    ? input.relatedBlockIds.map((item) => String(item || '').trim()).filter(Boolean)
    : []
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
    anchorNumber: Number.isFinite(Number(input?.anchorNumber || 0)) && Number(input?.anchorNumber || 0) > 0
      ? Math.floor(Number(input?.anchorNumber || 0))
      : undefined,
    strictLocate: Boolean(input?.strictLocate),
    locateTarget: input?.locateTarget && typeof input.locateTarget === 'object'
      ? input.locateTarget
      : undefined,
    alternatives: Array.isArray(input?.alternatives) && input.alternatives.length > 0 ? input.alternatives : undefined,
    visibleAlternatives: Array.isArray(input?.visibleAlternatives) && input.visibleAlternatives.length > 0
      ? input.visibleAlternatives
      : undefined,
    evidenceAlternatives: Array.isArray(input?.evidenceAlternatives) && input.evidenceAlternatives.length > 0
      ? input.evidenceAlternatives
      : undefined,
    initialAltIndex: Number.isFinite(Number(input?.initialAltIndex))
      ? Math.max(0, Math.floor(Number(input?.initialAltIndex)))
      : undefined,
  }
}
