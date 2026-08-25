import { useCallback } from 'react'

import type { ReaderBlockShelfPayload } from './MarkdownRenderer'
import type { ReaderSelectionShelfPayload } from './reader/readerTypes'

export type ReaderBlockShelfAddHandler = (payload: ReaderSelectionShelfPayload) => void

export interface BuildReaderBlockShelfPayloadOptions {
  block: ReaderBlockShelfPayload | null | undefined
  now?: () => number
  sourceName: string
  sourcePath: string
}

export interface UseReaderBlockShelfOptions {
  now?: () => number
  onAddSelectionToShelf?: ReaderBlockShelfAddHandler
  sourceName: string
  sourcePath: string
}

export interface ReaderBlockShelfController {
  addBlockToShelf: (block: ReaderBlockShelfPayload) => void
  canAddBlockToShelf: boolean
}

function compactOptional(value: unknown): string | undefined {
  const text = String(value || '').trim()
  return text || undefined
}

export function buildReaderBlockShelfPayload({
  block,
  now = Date.now,
  sourceName,
  sourcePath,
}: BuildReaderBlockShelfPayloadOptions): ReaderSelectionShelfPayload | null {
  const text = String(block?.text || '').trim()
  const cleanSourcePath = String(sourcePath || '').trim()
  const assetSrc = compactOptional(block?.assetSrc)
  if (!text || !cleanSourcePath) return null

  return {
    text,
    sourcePath: cleanSourcePath,
    sourceName,
    headingPath: compactOptional(block?.headingPath),
    blockId: compactOptional(block?.blockId),
    anchorId: compactOptional(block?.anchorId),
    anchorKind: compactOptional(block?.anchorKind),
    createdAt: now(),
    ...(assetSrc ? { assetSrc } : {}),
  }
}

export function useReaderBlockShelf({
  now,
  onAddSelectionToShelf,
  sourceName,
  sourcePath,
}: UseReaderBlockShelfOptions): ReaderBlockShelfController {
  const canAddBlockToShelf = Boolean(onAddSelectionToShelf && String(sourcePath || '').trim())

  const addBlockToShelf = useCallback((block: ReaderBlockShelfPayload) => {
    if (!onAddSelectionToShelf) return
    const payload = buildReaderBlockShelfPayload({
      block,
      now,
      sourceName,
      sourcePath,
    })
    if (!payload) return
    onAddSelectionToShelf(payload)
  }, [now, onAddSelectionToShelf, sourceName, sourcePath])

  return {
    addBlockToShelf,
    canAddBlockToShelf,
  }
}
