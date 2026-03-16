import { useEffect, useRef, useState, type RefObject } from 'react'
import type { ReaderSessionHighlight } from './readerTypes'
import {
  createSessionHighlightId,
  sameHighlightTarget,
  selectionStateInside,
  type ReaderSelectionState,
} from './readerDomUtils'

interface UseReaderSelectionInteractionsArgs {
  open: boolean
  sourcePath: string
  markdown: string
  locateRequestId: number
  contentRef: RefObject<HTMLDivElement | null>
  sessionHighlights: ReaderSessionHighlight[]
  onAddSessionHighlight?: (highlight: ReaderSessionHighlight) => void
  onRemoveSessionHighlight?: (highlightId: string) => void
  onAppendSelection: (text: string) => void
  sourceLabel: string
}

export function useReaderSelectionInteractions({
  open,
  sourcePath,
  markdown,
  locateRequestId,
  contentRef,
  sessionHighlights,
  onAddSessionHighlight,
  onRemoveSessionHighlight,
  onAppendSelection,
  sourceLabel,
}: UseReaderSelectionInteractionsArgs) {
  const [selection, setSelection] = useState('')
  const [selectionBubble, setSelectionBubble] = useState<ReaderSelectionState | null>(null)
  const selectionSyncRafRef = useRef<number | null>(null)

  const clearSelectionState = (clearNative = false) => {
    setSelection('')
    setSelectionBubble(null)
    if (!clearNative) return
    try {
      const sel = window.getSelection()
      sel?.removeAllRanges()
    } catch {
      // ignore
    }
  }

  const syncSelectionState = () => {
    const nextRaw = selectionStateInside(contentRef.current)
    if (!nextRaw) {
      setSelection('')
      setSelectionBubble(null)
      return
    }
    const matchedHighlight = sessionHighlights.find((item) => sameHighlightTarget(item, nextRaw)) || null
    const next = {
      ...nextRaw,
      highlightId: String(matchedHighlight?.id || '').trim(),
    }
    setSelection(next.text || '')
    setSelectionBubble(next)
  }

  const queueSelectionStateSync = () => {
    if (selectionSyncRafRef.current != null) {
      window.cancelAnimationFrame(selectionSyncRafRef.current)
    }
    selectionSyncRafRef.current = window.requestAnimationFrame(() => {
      selectionSyncRafRef.current = null
      syncSelectionState()
    })
  }

  const appendSelection = () => {
    const text = String(selection || '').trim()
    if (!text) return
    const quoted = text.split('\n').map((line) => `> ${line}`).join('\n')
    const sourceLine = sourceLabel ? `> Source: ${sourceLabel}\n` : ''
    onAppendSelection(`${sourceLine}${quoted}\n\n`)
    setSelection('')
    setSelectionBubble(null)
    try {
      const sel = window.getSelection()
      sel?.removeAllRanges()
    } catch {
      // ignore
    }
  }

  const toggleSelectionHighlight = () => {
    const selected = selectionBubble
    if (!selected || !selected.canHighlight) return
    if (selected.highlightId) {
      onRemoveSessionHighlight?.(selected.highlightId)
      setSelectionBubble((current) => (current ? { ...current, highlightId: '' } : current))
      return
    }
    const nextId = createSessionHighlightId()
    onAddSessionHighlight?.({
      id: nextId,
      text: selected.text,
      startOffset: selected.startOffset >= 0 ? selected.startOffset : undefined,
      endOffset: selected.endOffset > selected.startOffset ? selected.endOffset : undefined,
      blockId: selected.blockId || undefined,
      anchorId: selected.anchorId || undefined,
      occurrence: selected.occurrence,
      readableIndex: selected.readableIndex >= 0 ? selected.readableIndex : undefined,
      documentOccurrence: selected.documentOccurrence >= 0 ? selected.documentOccurrence : undefined,
      startReadableIndex: selected.startReadableIndex >= 0 ? selected.startReadableIndex : undefined,
      endReadableIndex: selected.endReadableIndex >= 0 ? selected.endReadableIndex : undefined,
    })
    setSelectionBubble((current) => (current ? { ...current, highlightId: nextId } : current))
  }

  useEffect(() => {
    if (!open) {
      clearSelectionState()
      return
    }
    clearSelectionState()
  }, [locateRequestId, open, sourcePath])

  useEffect(() => {
    return () => {
      if (selectionSyncRafRef.current != null) {
        window.cancelAnimationFrame(selectionSyncRafRef.current)
        selectionSyncRafRef.current = null
      }
    }
  }, [])

  useEffect(() => {
    if (!open) {
      clearSelectionState()
      return
    }
    const handleViewportChange = () => {
      syncSelectionState()
    }
    window.addEventListener('resize', handleViewportChange)
    const content = contentRef.current
    content?.addEventListener('scroll', handleViewportChange, { passive: true })
    return () => {
      if (selectionSyncRafRef.current != null) {
        window.cancelAnimationFrame(selectionSyncRafRef.current)
        selectionSyncRafRef.current = null
      }
      window.removeEventListener('resize', handleViewportChange)
      content?.removeEventListener('scroll', handleViewportChange)
    }
  }, [contentRef, markdown, locateRequestId, open, sessionHighlights])

  return {
    selection,
    selectionBubble,
    clearSelectionState,
    queueSelectionStateSync,
    appendSelection,
    toggleSelectionHighlight,
  }
}
