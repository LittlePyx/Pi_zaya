import type {
  ReaderLocateCandidate,
  ReaderLocateResult,
  ReaderOpenPayload,
} from './readerTypes'

function cleanText(value: unknown): string {
  return String(value || '').trim()
}

function withoutDirectIdentity(candidate: ReaderLocateCandidate): ReaderLocateCandidate {
  const next: ReaderLocateCandidate = {
    ...candidate,
    blockId: undefined,
    anchorId: undefined,
  }
  return next
}

function recoveryCandidates(value: ReaderLocateCandidate[] | undefined): ReaderLocateCandidate[] | undefined {
  if (!Array.isArray(value) || value.length <= 0) return undefined
  return value.map(withoutDirectIdentity)
}

export function shouldRecoverReaderLocateHeuristically(
  result: ReaderLocateResult,
  currentPayload: ReaderOpenPayload | null | undefined,
): boolean {
  if (!currentPayload || currentPayload.locateMode === 'heuristic') return false
  if (!currentPayload.strictLocate && !currentPayload.locateTarget) return false
  const status = cleanText(result.status).toLowerCase()
  return !result.ok || !['exact', 'block'].includes(status)
}

export function buildHeuristicReaderLocateRecoveryPayload(
  currentPayload: ReaderOpenPayload,
  locateRequestId: number,
): ReaderOpenPayload {
  const target = currentPayload.locateTarget
  const snippet = cleanText(
    target?.snippet
    || target?.evidenceQuote
    || target?.anchorText
    || currentPayload.snippet,
  )
  const highlightSnippet = cleanText(
    target?.highlightSnippet
    || target?.evidenceQuote
    || target?.anchorText
    || currentPayload.highlightSnippet
    || snippet,
  )
  return {
    ...currentPayload,
    // Prefer the evidence text during recovery. A stale-but-still-valid heading can
    // otherwise stop the locate at section level before the exact phrase is tried.
    headingPath: snippet ? undefined : cleanText(target?.headingPath || currentPayload.headingPath) || undefined,
    snippet: snippet || undefined,
    highlightSnippet: highlightSnippet || undefined,
    blockId: undefined,
    anchorId: undefined,
    relatedBlockIds: undefined,
    strictLocate: false,
    locateMode: 'heuristic',
    locateTarget: undefined,
    locateRequestId,
    alternatives: recoveryCandidates(currentPayload.alternatives),
    visibleAlternatives: recoveryCandidates(currentPayload.visibleAlternatives),
    evidenceAlternatives: recoveryCandidates(currentPayload.evidenceAlternatives),
    initialAltIndex: 0,
  }
}
