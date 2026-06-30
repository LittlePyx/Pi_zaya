import type { ReaderSessionHighlight } from './readerTypes'
import { normalizeReaderSourcePathForMatch } from './readerLocateGuard'

function cleanText(value: unknown): string {
  return String(value || '').trim().replace(/\s+/g, ' ')
}

function stableNumber(value: unknown): number | null {
  const num = Number(value)
  return Number.isFinite(num) ? num : null
}

function stableHash(value: string): string {
  let hash = 2166136261
  for (let idx = 0; idx < value.length; idx += 1) {
    hash ^= value.charCodeAt(idx)
    hash = Math.imul(hash, 16777619)
  }
  return (hash >>> 0).toString(36)
}

export function stableReaderHighlightId(item: Partial<ReaderSessionHighlight>): string {
  const parts = [
    cleanText(item.text),
    normalizeReaderSourcePathForMatch(item.sourcePath),
    cleanText(item.sourceName),
    cleanText(item.headingPath),
    cleanText(item.blockId),
    cleanText(item.anchorId),
    stableNumber(item.startOffset),
    stableNumber(item.endOffset),
    stableNumber(item.occurrence),
    stableNumber(item.readableIndex),
    stableNumber(item.documentOccurrence),
    stableNumber(item.startReadableIndex),
    stableNumber(item.endReadableIndex),
  ].map((part) => (part === null ? '' : String(part)))
  return `imported-${stableHash(parts.join('\u001f'))}`
}

export function readerHighlightsSignature(items: ReaderSessionHighlight[]): string {
  return JSON.stringify((items || []).map((item) => ({
    id: cleanText(item.id),
    text: cleanText(item.text),
    noteKind: cleanText(item.noteKind),
    sourcePath: normalizeReaderSourcePathForMatch(item.sourcePath),
    sourceName: cleanText(item.sourceName),
    conversationId: cleanText(item.conversationId),
    messageId: stableNumber(item.messageId),
    locateRequestId: stableNumber(item.locateRequestId),
    locateFeedbackKey: cleanText(item.locateFeedbackKey),
    feedback: cleanText(item.feedback),
    feedbackAt: stableNumber(item.feedbackAt),
    headingPath: cleanText(item.headingPath),
    startOffset: stableNumber(item.startOffset),
    endOffset: stableNumber(item.endOffset),
    blockId: cleanText(item.blockId),
    anchorId: cleanText(item.anchorId),
    occurrence: stableNumber(item.occurrence),
    readableIndex: stableNumber(item.readableIndex),
    documentOccurrence: stableNumber(item.documentOccurrence),
    startReadableIndex: stableNumber(item.startReadableIndex),
    endReadableIndex: stableNumber(item.endReadableIndex),
    createdAt: stableNumber(item.createdAt),
    updatedAt: stableNumber(item.updatedAt),
  })))
}
