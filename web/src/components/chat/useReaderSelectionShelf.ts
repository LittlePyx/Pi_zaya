import { useCallback } from 'react'

import type { ReaderSelectionState } from './reader/readerDomUtils'
import type {
  ReaderSelectionShelfPayload,
  ReaderSessionHighlight,
} from './reader/readerTypes'

export type ReaderSelectionShelfAddHandler = (payload: ReaderSelectionShelfPayload) => void

export interface ReaderSelectionShelfBaseOptions {
  activeAnchorId: string
  activeAnchorKind: string
  activeBlockId: string
  activeHeadingPath: string
  now?: () => number
  sourceName: string
  sourcePath: string
}

export interface BuildReaderSelectionShelfPayloadOptions extends ReaderSelectionShelfBaseOptions {
  selection: string
  selectionBubble: ReaderSelectionState | null
}

export interface BuildReaderHighlightShelfPayloadOptions extends ReaderSelectionShelfBaseOptions {
  activeHighlight: ReaderSessionHighlight | null
}

export interface UseReaderSelectionShelfOptions extends ReaderSelectionShelfBaseOptions {
  activeHighlight: ReaderSessionHighlight | null
  onAddSelectionToShelf?: ReaderSelectionShelfAddHandler
  onClearSelection: (clearNative?: boolean) => void
  onCloseHighlight: () => void
  selection: string
  selectionBubble: ReaderSelectionState | null
}

export interface ReaderSelectionShelfController {
  addActiveHighlightToShelf?: () => void
  addSelectionToShelf: () => void
  canAddSelectionToShelf: boolean
}

function compactOptional(value: unknown): string | undefined {
  const text = String(value || '').trim()
  return text || undefined
}

function finiteOptionalNumber(value: unknown): number | undefined {
  const number = Number(value)
  return Number.isFinite(number) ? number : undefined
}

function finiteNonNegativeOptionalNumber(value: unknown): number | undefined {
  if (value == null) return undefined
  const number = Number(value)
  return Number.isFinite(number) && number >= 0 ? number : undefined
}

export function buildReaderSelectionShelfPayload({
  activeAnchorId,
  activeAnchorKind,
  activeBlockId,
  activeHeadingPath,
  now = Date.now,
  selection,
  selectionBubble,
  sourceName,
  sourcePath,
}: BuildReaderSelectionShelfPayloadOptions): ReaderSelectionShelfPayload | null {
  const selected = selectionBubble
  const text = String(selected?.text || selection || '').trim()
  if (!selected || !text) return null

  return {
    text,
    sourcePath,
    sourceName,
    headingPath: compactOptional(activeHeadingPath),
    blockId: compactOptional(selected.blockId || activeBlockId),
    anchorId: compactOptional(selected.anchorId || activeAnchorId),
    anchorKind: compactOptional(activeAnchorKind),
    startOffset: selected.startOffset >= 0 ? selected.startOffset : undefined,
    endOffset: selected.endOffset > selected.startOffset ? selected.endOffset : undefined,
    occurrence: finiteOptionalNumber(selected.occurrence),
    readableIndex: selected.readableIndex >= 0 ? selected.readableIndex : undefined,
    documentOccurrence: selected.documentOccurrence >= 0 ? selected.documentOccurrence : undefined,
    startReadableIndex: selected.startReadableIndex >= 0 ? selected.startReadableIndex : undefined,
    endReadableIndex: selected.endReadableIndex >= 0 ? selected.endReadableIndex : undefined,
    createdAt: now(),
  }
}

export function buildReaderHighlightShelfPayload({
  activeAnchorId,
  activeAnchorKind,
  activeBlockId,
  activeHeadingPath,
  activeHighlight,
  now = Date.now,
  sourceName,
  sourcePath,
}: BuildReaderHighlightShelfPayloadOptions): ReaderSelectionShelfPayload | null {
  const item = activeHighlight
  const text = String(item?.text || '').trim()
  if (!item || !text) return null

  return {
    text,
    sourcePath,
    sourceName,
    headingPath: compactOptional(item.headingPath || activeHeadingPath),
    blockId: compactOptional(item.blockId || activeBlockId),
    anchorId: compactOptional(item.anchorId || activeAnchorId),
    anchorKind: compactOptional(activeAnchorKind),
    startOffset: finiteNonNegativeOptionalNumber(item.startOffset),
    endOffset: finiteNonNegativeOptionalNumber(item.endOffset),
    occurrence: finiteOptionalNumber(item.occurrence),
    readableIndex: finiteNonNegativeOptionalNumber(item.readableIndex),
    documentOccurrence: finiteNonNegativeOptionalNumber(item.documentOccurrence),
    startReadableIndex: finiteNonNegativeOptionalNumber(item.startReadableIndex),
    endReadableIndex: finiteNonNegativeOptionalNumber(item.endReadableIndex),
    createdAt: now(),
  }
}

export function useReaderSelectionShelf({
  activeAnchorId,
  activeAnchorKind,
  activeBlockId,
  activeHeadingPath,
  activeHighlight,
  now,
  onAddSelectionToShelf,
  onClearSelection,
  onCloseHighlight,
  selection,
  selectionBubble,
  sourceName,
  sourcePath,
}: UseReaderSelectionShelfOptions): ReaderSelectionShelfController {
  const canAddSelectionToShelf = Boolean(onAddSelectionToShelf)

  const addSelectionToShelf = useCallback(() => {
    if (!onAddSelectionToShelf) return
    const payload = buildReaderSelectionShelfPayload({
      activeAnchorId,
      activeAnchorKind,
      activeBlockId,
      activeHeadingPath,
      now,
      selection,
      selectionBubble,
      sourceName,
      sourcePath,
    })
    if (!payload) return
    onAddSelectionToShelf(payload)
    onClearSelection(true)
  }, [
    activeAnchorId,
    activeAnchorKind,
    activeBlockId,
    activeHeadingPath,
    now,
    onAddSelectionToShelf,
    onClearSelection,
    selection,
    selectionBubble,
    sourceName,
    sourcePath,
  ])

  const addActiveHighlightToShelf = useCallback(() => {
    if (!onAddSelectionToShelf) return
    const payload = buildReaderHighlightShelfPayload({
      activeAnchorId,
      activeAnchorKind,
      activeBlockId,
      activeHeadingPath,
      activeHighlight,
      now,
      sourceName,
      sourcePath,
    })
    if (!payload) return
    onAddSelectionToShelf(payload)
    onCloseHighlight()
  }, [
    activeAnchorId,
    activeAnchorKind,
    activeBlockId,
    activeHeadingPath,
    activeHighlight,
    now,
    onAddSelectionToShelf,
    onCloseHighlight,
    sourceName,
    sourcePath,
  ])

  return {
    addActiveHighlightToShelf: activeHighlight && onAddSelectionToShelf
      ? addActiveHighlightToShelf
      : undefined,
    addSelectionToShelf,
    canAddSelectionToShelf,
  }
}
